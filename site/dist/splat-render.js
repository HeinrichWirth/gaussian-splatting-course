// A readable point-sampled renderer. Pixels and direction conventions match Python.
import {worldToCamera, projectPoint, projectedCovariance, covariance3d,
        multiply, transpose} from './geometry-math.js';

export const FORMAT='gs-course-3d-v1';
export const C0=1/Math.sqrt(4*Math.PI), C1=Math.sqrt(3/(4*Math.PI));
const identity=[[1,0,0],[0,1,0],[0,0,1]];
const clamp=(x,a=0,b=1)=>Math.min(b,Math.max(a,x));
const copy=value=>structuredClone(value);

export const splat=(center,scales,color,opacity,rotation=[1,0,0,0])=>
  ({center,scales,color,opacity,rotation});
export const makeScene=(...splats)=>({format:FORMAT,splats});
export function orbitCamera(angle=0,focal=48,target=[0,0,4],radius=4){
  const a=angle*Math.PI/180, co=Math.cos(a), si=Math.sin(a);
  const rotation=[[co,0,si],[0,1,0],[-si,0,co]];
  return {center:target.map((x,i)=>x-radius*rotation[i][2]),rotation,focal};
}
export function shBasis(direction){
  const norm=Math.hypot(...direction);
  if(!Number.isFinite(norm)||!norm)throw Error('Undefined direction');
  const [x,y,z]=direction.map(v=>v/norm);
  return [C0,-C1*y,C1*z,-C1*x];
}
export function shColor(direction,coeff){
  if(![1,4].includes(coeff.length))throw Error('Expected 1 or 4 RGB coefficients');
  const basis=shBasis(direction);
  return [0,1,2].map(c=>clamp(.5+coeff.reduce((sum,row,i)=>sum+basis[i]*row[c],0)));
}
export function projectSplats(scene,camera,width,height){
  if(scene.format!==FORMAT)throw Error('Unknown scene format');
  camera=camera||{};
  const R=camera.rotation||identity, origin=camera.center||[0,0,0];
  const focal=camera.focal??Math.min(width,height), principal=camera.principal??[(width-1)/2,(height-1)/2];
  const spots=[];
  for(const [index,item] of scene.splats.entries()){
    const p=worldToCamera(item.center,origin,R);
    if(p[2]<=(camera.near??.05))continue;
    const cov=covariance3d(item.scales,item.rotation);
    const covariance=projectedCovariance(p,multiply(multiply(transpose(R),cov),R),focal);
    const [[a,b],[,d]]=covariance, det=a*d-b*b;
    const color=item.sh?shColor(item.center.map((x,i)=>x-origin[i]),item.sh):item.color;
    if(!color.every(v=>v>=0&&v<=1)||!(item.opacity>=0&&item.opacity<=1))throw Error('Invalid RGB or opacity');
    spots.push({index,depth:p[2],color,mean:projectPoint(p,focal,principal,camera.near??.05),
      opacity:item.opacity,covariance,precision:[[d/det,-b/det],[-b/det,a/det]]});
  }
  return spots.sort((a,b)=>a.depth-b.depth);
}
export function footprintAlpha([u,v],item){
  const x=u-item.mean[0], y=v-item.mean[1], P=item.precision;
  const q=P[0][0]*x*x+2*P[0][1]*x*y+P[1][1]*y*y;
  return item.opacity*Math.exp(-.5*q);
}
export function renderScene(scene,camera,width=65,height=65,background=[1,1,1],reverse=false){
  const spots=projectSplats(scene,camera,width,height);
  if(reverse)spots.reverse();
  const image=new Float64Array(3*width*height);
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    const i=3*(y*width+x);
    let remaining=1;
    for(const item of spots){
      const alpha=footprintAlpha([x,y],item);
      for(let c=0;c<3;c++)image[i+c]+=remaining*alpha*item.color[c];
      remaining*=1-alpha;
    }
    for(let c=0;c<3;c++)image[i+c]+=remaining*background[c];
  }
  return image;
}
export const overlapScene=(z=3.5)=>makeScene(splat([0,0,z],[.55,.35,.25],[1,0,0],.6),
  splat([0,0,4.5],[.38,.65,.22],[0,0,1],.5));
export const translatedCamera=(x,focal=28)=>({center:[x,0,0],rotation:identity,focal});
export const rayScene=z=>makeScene(splat([.10*z,-.05*z,z],[.12*z,.08*z,.06*z],[.85,.2,.08],.9));
export function mse(a,b){
  let error=0;
  for(let i=0;i<a.length;i++)error+=(a[i]-b[i])**2;
  return error/a.length;
}
export function rayLoss(z,baselines=[0,.8],size=33){
  return baselines.reduce((sum,x)=>{
    const cam=translatedCamera(x);
    return sum+mse(renderScene(rayScene(z),cam,size,size),renderScene(rayScene(4),cam,size,size));
  },0)/baselines.length;
}
export function depthStep(z,baselines,rate=80,h=.002){
  const gradient=(rayLoss(z+h,baselines)-rayLoss(z-h,baselines))/(2*h);
  return [clamp(z-rate*gradient,1.25,7),gradient];
}
export function densityScene(target=false){
  const color=[.12,.37,.83];
  if(target)return makeScene(...[-.54,.42].map(x=>splat([x,0,4],[.23,.34,.16],color,.9)));
  return makeScene(splat([-.16,0,4],[.63,.38,.18],color,.77),
                   splat([.78,-.55,4.2],[.20,.18,.13],color,.015));
}
export function densityLoss(scene,size=33){
  const cam=translatedCamera(0,42);
  return mse(renderScene(scene,cam,size,size),renderScene(densityScene(true),cam,size,size));
}
export function densityGradients(scene,h=.0001){
  return scene.splats.map((item,i)=>['x','opacity'].map(field=>{
    const value=field==='x'?item.center[0]:item.opacity;
    const lo=field==='x'?value-h:clamp(value-h), hi=field==='x'?value+h:clamp(value+h);
    function loss(v){
      const candidate=copy(scene), spot=candidate.splats[i];
      if(field==='x')spot.center[0]=v;else spot.opacity=v;
      return densityLoss(candidate);
    }
    return (loss(hi)-loss(lo))/(hi-lo);
  }));
}
export function densityStep(scene,rate=3){
  const updated=copy(scene);
  densityGradients(scene).forEach(([dx,da],i)=>{
    const s=updated.splats[i];
    s.center[0]=clamp(s.center[0]-rate*dx,-2,2);
    s.opacity=clamp(s.opacity-rate*da);
  });
  return updated;
}
export function densityEdit(scene,operation,selected=0){
  const updated=copy(scene), items=updated.splats;
  if(operation==='prune')updated.splats=items.filter(p=>p.opacity>=.03);
  else if(operation==='reset')for(const p of items)p.opacity=Math.min(p.opacity,.01);
  else if(operation==='clone'||operation==='split'){
    if(!items[selected]||items.length>=16)throw Error('Invalid selection or 16-spot limit');
    const item=items[selected];
    if(operation==='clone')items.push(copy(item));
    else{
      const children=[-1,1].map(sign=>{
        const child=copy(item);
        child.center[0]+=sign*.65*item.scales[0];
        child.scales=item.scales.map(v=>v/1.6);
        return child;
      });
      items.splice(selected,1,...children);
    }
  }else throw Error('Unknown density operation');
  return updated;
}
export function densityPass(scene,threshold=.001){
  const gradients=densityGradients(scene);
  let updated=copy(scene);
  for(let i=gradients.length-1;i>=0;i--){
    const spot=scene.splats[i];
    if(spot.opacity<.03)updated.splats.splice(i,1);
    else if(Math.abs(gradients[i][0])>threshold && updated.splats.length<16)
      updated=densityEdit(updated,spot.scales[0]>.32?'split':'clone',i);
  }
  return updated;
}
