import {splat,makeScene} from './splat-render.js';
import {validateScene} from './reconstruction-math.js';

export const MOTION_FORMAT='gs-course-motion-v1';
export const TRAIN_TIMES=[.25,.75];
const focal=64, baseline=.8, principal=[32,24];

export function ribbon(){
  const colors=[[.13,.28,.68],[.09,.51,.65],[.12,.65,.49],[.88,.65,.14],[.93,.34,.14]];
  return makeScene(...colors.map((color,i)=>splat([-.8+.4*i,0,4],[.27,.15,.12],color,.93)));
}
export function fieldValue(x,t,corner=1){
  if(!(x>=0&&x<=1&&t>=0&&t<=1&&Number.isFinite(corner)))throw Error('Query outside the grid');
  const early=(1-x)*0+x*0, late=(1-x)*0+x*corner;
  return (1-t)*early+t*late;
}
export function fieldScene(t,corner=1){
  const scene=ribbon();
  for(const p of scene.splats){
    const f=fieldValue((p.center[0]+1)/2,t,corner),angle=-35*f*Math.PI/180;
    p.center[1]-=.8*f;
    p.rotation=[Math.cos(angle/2),0,0,Math.sin(angle/2)];
    p.scales[0]*=1+.3*f;
  }
  return scene;
}
export const pulse=t=>4*t*(1-t);
export function validateModel(model){
  if(model?.format!==MOTION_FORMAT||!Number.isFinite(model.duration)||model.duration<=0)
    throw Error('Invalid motion header');
  const canonical=validateScene(model.canonical);
  if(!Array.isArray(model.amplitudes)||model.amplitudes.length!==canonical.splats.length||
     !model.amplitudes.every(a=>Array.isArray(a)&&a.length===3&&a.every(Number.isFinite)))
    throw Error('Invalid motion amplitudes');
  return model;
}
export function sceneAt(model,t){
  validateModel(model);
  if(!(t>=0&&t<=model.duration))throw Error('Time outside the clip');
  const scene=structuredClone(model.canonical),weight=pulse(t/model.duration);
  scene.splats.forEach((s,i)=>{s.center=s.center.map((v,j)=>v+weight*model.amplitudes[i][j]);});
  return scene;
}
export function referenceScene(t){
  if(!(t>=0&&t<=1))throw Error('Time outside the reference clip');
  const scene=ribbon();
  scene.splats.forEach((p,i)=>{
    const s=i/4,h=pulse(t),w=Math.sin(2*Math.PI*t);
    const d=[.12*s*h,-.8*s*s*h+.12*s*w,.45*s*h+.08*s*w];
    p.center=p.center.map((x,j)=>x+d[j]);
  });
  return scene;
}
export function stereoTracks(noise=0){
  return TRAIN_TIMES.map((t,k)=>{
    const left=[],right=[];
    referenceScene(t).splats.forEach((p,i)=>{
      const [x,y,z]=p.center,u=principal[0]+focal*x/z,v=principal[1]+focal*y/z;
      left.push([u,v]);
      right.push([u-focal*baseline/z+noise*(-1)**(k+i),v]);
    });
    return {t,left,right};
  });
}
export function fitMotion(canonical,observations){
  const amplitudes=canonical.splats.map(()=>[0,0,0]);
  let denominator=0;
  for(const {t,left,right} of observations){
    if(!(t>=0&&t<=1)||left.length!==amplitudes.length||right.length!==amplitudes.length)
      throw Error('Invalid stereo observation');
    const h=pulse(t);
    denominator+=h*h;
    canonical.splats.forEach((p,i)=>{
      const disparity=left[i][0]-right[i][0];
      if(!(Number.isFinite(disparity)&&disparity>0))throw Error('Positive disparity required');
      const z=focal*baseline/disparity;
      const xyz=[(left[i][0]-principal[0])*z/focal,(left[i][1]-principal[1])*z/focal,z];
      for(let j=0;j<3;j++)amplitudes[i][j]+=h*(xyz[j]-p.center[j]);
    });
  }
  if(!(denominator>0))throw Error('An interior time is required');
  return validateModel({format:MOTION_FORMAT,duration:1,canonical:structuredClone(canonical),
    amplitudes:amplitudes.map(a=>a.map(v=>v/denominator))});
}
export const exampleModel=(noise=0)=>fitMotion(ribbon(),stereoTracks(noise));
