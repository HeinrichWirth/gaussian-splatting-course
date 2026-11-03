import {projectSplats, renderScene} from './splat-render.js';

export function smallScene() {
  const rows = [
    [[-.62,.13,3.45],[.31,.45,.22],[.15,.45,.83]],
    [[.48,.18,3.7],[.38,.41,.22],[.92,.31,.12]],
    [[.02,-.52,3.28],[.31,.28,.20],[.95,.72,.12]],
    [[.02,.54,3.04],[.25,.12,.19],[.18,.67,.36]],
    [[.38,-.38,4.1],[.40,.12,.23],[.59,.23,.66]],
  ];
  return {format:'gs-course-3d-v1',splats:rows.map(([center,scales,color])=>({
    center,scales,color,rotation:[1,0,0,0],opacity:.88
  }))};
}

export function orbit(angle,width=41,height=33) {
  const a=angle*Math.PI/180,c=Math.cos(a),s=Math.sin(a);
  return {center:[3.5*s,0,3.5*(1-c)],rotation:[[c,0,-s],[0,1,0],[s,0,c]],
    focal:.97*width,principal:[(width-1)/2,(height-1)/2],near:.05};
}

function footprint(p) {
  const [[a,b],[,d]]=p.covariance;
  const det=a*d-b*b;
  return {...p, inverse:[d/det,-b/det,a/det]};
}

function distance(x,y,p) {
  const dx=x-p.mean[0],dy=y-p.mean[1],[a,b,d]=p.inverse;
  return a*dx*dx+2*b*dx*dy+d*dy*dy;
}

export function tiledFrame(scene,camera,width,height,tile=8,cutoff=3,tiled=true) {
  const splats=projectSplats(scene,camera,width,height).map(footprint);
  const nx=Math.ceil(width/tile),ny=Math.ceil(height/tile);
  const bins=Array.from({length:nx*ny},()=>[]);
  for (const p of splats) {
    const rx=cutoff*Math.sqrt(p.covariance[0][0]),ry=cutoff*Math.sqrt(p.covariance[1][1]);
    const x0=Math.max(0,Math.floor(Math.ceil(p.mean[0]-rx)/tile));
    const y0=Math.max(0,Math.floor(Math.ceil(p.mean[1]-ry)/tile));
    const x1=Math.min(nx-1,Math.floor(Math.floor(p.mean[0]+rx)/tile));
    const y1=Math.min(ny-1,Math.floor(Math.floor(p.mean[1]+ry)/tile));
    for(let y=y0;y<=y1;y++) for(let x=x0;x<=x1;x++) bins[y*nx+x].push(p);
  }
  const rgb=new Float64Array(width*height*3);
  let pairs=0;
  for(let y=0;y<height;y++) for(let x=0;x<width;x++) {
    const bucket=bins[Math.floor(y/tile)*nx+Math.floor(x/tile)];
    const index=3*(y*width+x);
    let T=1;
    for(const p of tiled ? bucket : splats) {
      pairs++;
      const q=distance(x,y,p);
      if(q>cutoff*cutoff) continue;
      const alpha=p.opacity*Math.exp(-q/2);
      for(let c=0;c<3;c++) rgb[index+c]+=T*alpha*p.color[c];
      T*=1-alpha;
    }
    for(let c=0;c<3;c++) rgb[index+c]+=T;
  }
  return {rgb,pairs,nx,ny,counts:bins.map(b=>b.length),visible:splats.length};
}

export function triangulate(f,B,left,right) {
  return B>0 && f>0 && left>right ? f*B/(left-right) : null;
}

export function syntheticCapture() {
  const target=smallScene();
  const frames=[[-32,'train'],[-15,'test'],[0,'train'],[21,'test'],[35,'train']].map(([a,split])=>{
    const width=41,height=33,camera=orbit(a,width,height);
    const rgb=Float64Array.from(renderScene(target,camera,width,height),v=>Math.round(255*v)/255);
    return {name:(a<0?'':'+')+a+'°',width,height,camera,split,rgb};
  });
  const initial=smallScene();
  initial.splats.forEach(p=>{p.color=[.5,.5,.5];});
  return {initial,frames};
}

const vec=(v,n)=>Array.isArray(v)&&v.length===n&&v.every(Number.isFinite);
function validCameraRotation(R) {
  if(!Array.isArray(R)||R.length!==3||!R.every(row=>vec(row,3))) return false;
  const dot=(a,b)=>a.reduce((sum,x,i)=>sum+x*b[i],0);
  if(!R.every((a,i)=>R.every((b,j)=>Math.abs(dot(a,b)-Number(i===j))<1e-5))) return false;
  const [a,b,c]=R;
  return a[0]*(b[1]*c[2]-b[2]*c[1])+a[1]*(b[2]*c[0]-b[0]*c[2])+a[2]*(b[0]*c[1]-b[1]*c[0])>0;
}
export function validateScene(data) {
  if(data?.format!=='gs-course-3d-v1' || !Array.isArray(data.splats) ||
    data.splats.length<1 || data.splats.length>128) throw Error('scene');
  for(const p of data.splats) {
    if(!vec(p.center,3)||!vec(p.color,3)||!p.color.every(x=>x>=0&&x<=1)||
       !vec(p.scales,3)||!p.scales.every(x=>x>0)||!vec(p.rotation,4)||
       Math.hypot(...p.rotation)<1e-8||!Number.isFinite(p.opacity)||p.opacity<0||p.opacity>1||
       p.sh!=null&&(!Array.isArray(p.sh)||![1,4].includes(p.sh.length)||!p.sh.every(v=>vec(v,3))))
      throw Error('scene');
  }
  return data;
}

export function decodeCapture(data) {
  if(data?.format!=='gs-course-capture-v1'||!Array.isArray(data.frames)||
     data.frames.length<2||data.frames.length>32) throw Error('capture');
  const initial=validateScene(data.initial);
  const frames=data.frames.map(f=>{
    const {width:w,height:h,camera:cam}=f;
    if(!Number.isInteger(w)||!Number.isInteger(h)||Math.min(w,h)<1||Math.max(w,h)>128||
       !['train','test'].includes(f.split)||typeof f.rgb8!=='string'||f.rgb8.length!==4*Math.ceil(w*h*3/3)||
       !vec(cam?.center,3)||!validCameraRotation(cam.rotation)||
       cam.near!=null&&(!Number.isFinite(cam.near)||cam.near<0)||!(Number.isFinite(cam.focal)&&cam.focal>0||
       vec(cam.focal,2)&&cam.focal.every(x=>x>0))||cam.principal!=null&&!vec(cam.principal,2)) throw Error('capture');
    const bytes=atob(f.rgb8);
    if(bytes.length!==w*h*3) throw Error('capture');
    return {...f,name:String(f.name),rgb:Float64Array.from(bytes,c=>c.charCodeAt(0)/255)};
  });
  if(!frames.some(f=>f.split==='train')||!frames.some(f=>f.split==='test')) throw Error('capture');
  if(frames.reduce((n,f)=>n+f.width*f.height,0)*initial.splats.length**2>24000000)
    throw Error('size');
  return {initial,frames};
}

// Store the normal equations once. Fitting never reads the held-out frames.
export function trainingSystem(scene,frames) {
  const n=scene.splats.length,gram=new Float64Array(n*n),rhs=new Float64Array(3*n);
  for(const frame of frames.filter(f=>f.split==='train')) {
    const {width:w,height:h,camera,rgb}=frame;
    const splats=projectSplats(scene,camera,w,h).map(footprint);
    const weights=new Float64Array(n);
    for(let y=0;y<h;y++) for(let x=0;x<w;x++) {
      weights.fill(0);
      let T=1;
      for(const p of splats) {
        const alpha=p.opacity*Math.exp(-.5*distance(x,y,p));
        weights[p.index]=T*alpha;
        T*=1-alpha;
      }
      const pixel=3*(y*w+x);
      for(let i=0;i<n;i++) {
        for(let j=0;j<n;j++) gram[n*i+j]+=weights[i]*weights[j];
        for(let c=0;c<3;c++) rhs[i*3+c]+=weights[i]*(rgb[pixel+c]-T);
      }
    }
  }
  let bound=1e-15;
  for(let i=0;i<n;i++) {
    let sum=0;
    for(let j=0;j<n;j++) sum+=gram[i*n+j];
    bound=Math.max(bound,sum);
  }
  return {gram,rhs,bound};
}

export function fitColors(scene,system,steps=1) {
  const result=structuredClone(scene),n=result.splats.length;
  for(let step=0;step<steps;step++) {
    const old=result.splats.map(p=>p.color.slice());
    for(let i=0;i<n;i++) for(let c=0;c<3;c++) {
      let gradient=-system.rhs[i*3+c];
      for(let j=0;j<n;j++) gradient+=system.gram[i*n+j]*old[j][c];
      result.splats[i].color[c]=Math.min(1,Math.max(0,old[i][c]-gradient/system.bound));
    }
  }
  return result;
}

export function imageScores(rgb,target) {
  let abs=0,sq=0;
  for(let i=0;i<rgb.length;i++) {
    const d=rgb[i]-target[i]; abs+=Math.abs(d); sq+=d*d;
  }
  const mse=sq/rgb.length;
  return {mae:abs/rgb.length,mse,psnr:mse===0?Infinity:-10*Math.log10(mse)};
}

export function damagedScene(original,kind,strength) {
  const scene=structuredClone(original);
  if(kind==='blur') scene.splats.forEach(p=>{p.scales=p.scales.map(s=>s*(1+strength));});
  if(kind==='tint') scene.splats.forEach(p=>{p.color[1]*=1-strength;});
  if(kind==='floater' && strength>0) {
    const p=structuredClone(scene.splats[2]||scene.splats[0]);
    p.center=[.14,-.1,2.55];p.scales=[.14,.26,.12];p.color=[.12,.18,.18];p.opacity=strength;
    scene.splats.push(p);
  }
  return scene;
}
