import {matVec, transpose} from './geometry-math.js';
import {footprintAlpha} from './splat-render.js';

const dot = (a,b) => a.reduce((sum,v,i) => sum+v*b[i],0);
const subtract = (a,b) => a.map((v,i) => v-b[i]);

export function observedPixel(track, frame) {
  const [u,v]=track.uv[frame.name];
  return [u*frame.width/track.width-.5,v*frame.height/track.height-.5];
}

function ray(frame, pixel) {
  const {rotation,focal,principal}=frame.camera;
  const xy=pixel.map((v,i)=>(v-principal[i])/focal[i]);
  const dir=matVec(rotation,[...xy,1]);
  const length=Math.hypot(...dir);
  return dir.map(v=>v/length);
}

export function intersectRays(a, uvA, b, uvB) {
  const r=ray(a,uvA), s=ray(b,uvB);
  const baseline=subtract(b.camera.center,a.camera.center);
  const k=dot(r,s), determinant=1-k*k;
  if(determinant < 1e-12)return null;
  const alongA=(dot(r,baseline)-k*dot(s,baseline))/determinant;
  const alongB=k*alongA-dot(s,baseline);
  if(Math.min(alongA,alongB)<=0)return null;
  const p=r.map((v,i)=>a.camera.center[i]+alongA*v);
  const q=s.map((v,i)=>b.camera.center[i]+alongB*v);
  return {point:p.map((v,i)=>(v+q[i])/2),gap:Math.hypot(...subtract(p,q))};
}

export function projectFeature(point, frame) {
  const {center,rotation,focal,principal}=frame.camera;
  const p=matVec(transpose(rotation),subtract(point,center));
  return principal.map((c,i)=>c+focal[i]*p[i]/p[2]);
}

export function pixelLedger(projected, u, v) {
  const rows=[], rgb=[0,0,0];
  let trans=1;
  for(const spot of projected) {
    const alpha=footprintAlpha([u,v],spot), weight=trans*alpha;
    rows.push({id:spot.index,alpha,trans,weight});
    for(let c=0;c<3;c++)rgb[c]+=weight*spot.color[c];
    trans*=1-alpha;
  }
  return {rows:rows.sort((a,b)=>b.weight-a.weight),background:trans,rgb:rgb.map(c=>c+trans)};
}

export function regionScores(frame,prediction,left,top,width,height) {
  let all=0, part=0;
  for(let v=0;v<frame.height;v++)for(let u=0;u<frame.width;u++) {
    const i=3*(v*frame.width+u);
    let sq=0;
    for(let c=0;c<3;c++)sq+=(prediction[i+c]-frame.rgb[i+c])**2;
    all+=sq;
    if(u>=left&&u<left+width&&v>=top&&v<top+height)part+=sq;
  }
  const n=width*height, total=frame.width*frame.height;
  return {mse:part/(3*n),outside:(all-part)/(3*(total-n)),global:all/(3*total),share:all?part/all:0};
}
