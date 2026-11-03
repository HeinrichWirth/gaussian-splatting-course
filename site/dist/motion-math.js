import {makeScene, renderScene, splat, translatedCamera} from './splat-render.js';
import {axisAngleQuaternion} from './geometry-math.js';

export const F=80, WIDTH=97, HEIGHT=61, U0=48, DEPTH=4, BASELINE=.8;
const orange=[.93,.31,.09], blue=[.16,.39,.74];
export const horizontalPixel=(x,cameraX,z=DEPTH)=>U0+F*(x-cameraX)/z;

export function observingPair(time=.8,objectSpeed=1,cameraSpeed=0,lag=0,gate=false) {
  const views=[[time,0],[time+lag,BASELINE]].map(([exposure,baseline])=>{
    const cameraX=cameraSpeed*exposure+baseline, x=objectSpeed*exposure;
    const u=horizontalPixel(x,cameraX), rayAtGate=(cameraX+x)/2;
    const visible=u>=0 && u<WIDTH && !(gate && rayAtGate>=.28 && rayAtGate<=.52);
    return {time:exposure,cameraX,x,u,anchor:horizontalPixel(-1.2,cameraX),visible};
  });
  const disparity=views[0].u-views[1].u;
  const estimate=views.every(v=>v.visible) && disparity>1e-9 ? F*BASELINE/disparity : null;
  return {views,disparity,estimate};
}

export function observationImage(view,gate=false) {
  const scene=makeScene(splat([view.x,0,DEPTH],[.18,.14,.09],orange,.96),
    splat([-1.2,.54,DEPTH],[.14,.14,.10],blue,.94));
  const image=renderScene(scene,translatedCamera(view.cameraX,F),WIDTH,HEIGHT);
  if(gate) {
    const left=horizontalPixel(.28,view.cameraX,2), right=horizontalPixel(.52,view.cameraX,2);
    for(let y=0;y<HEIGHT;y++)for(let x=Math.max(0,Math.ceil(left));x<=Math.min(WIDTH-1,right);x++){
      image.set([.23,.29,.30],3*(y*WIDTH+x));
    }
  }
  return image;
}

export const trueX=(time,bend=.2)=>-.5+.6*time+bend*time*(time-2);
export function fitPath(time,middle=false,noise=0,bend=.2) {
  const start=trueX(0,bend), end=trueX(2,bend), line=start+(end-start)*time/2;
  if(!middle)return line;
  const offset=trueX(1,bend)+noise-(start+end)/2;
  return line+offset*time*(2-time);
}
export function trajectorySplat(time,centerX) {
  return splat([centerX,0,DEPTH],[.24+.08*time,.11,.08],orange,.96,
    axisAngleQuaternion([0,0,1],30*time));
}
export function persistentPair(angle=90,noise=.4,strength=4,bend=0) {
  if(strength<0)throw Error('The squared penalty needs a nonnegative weight');
  const edge=degrees=>[.8*Math.cos(degrees*Math.PI/180),.8*Math.sin(degrees*Math.PI/180)];
  const rigid=edge(angle),actual=edge(angle+bend);
  const observed=[actual[0]+noise,actual[1]-noise/2];
  const fitted=observed.map((x,i)=>(x+strength*rigid[i])/(1+strength));
  const distance=(p,q)=>Math.hypot(p[0]-q[0],p[1]-q[1]);
  return {rigid,actual,observed,fitted,rigidityError:distance(fitted,rigid),trackingError:distance(fitted,actual)};
}
