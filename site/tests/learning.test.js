import test from 'node:test';
import assert from 'node:assert/strict';
import {imageLoss,chain,curveLoss,numericalDerivative,targetImage,initializeSpots,renderAndGradient,sgdStep,exportScene} from '../dist/learning-math.js';
import {compositeAt} from '../dist/math.js';
const close=(a,b,tol=1e-8)=>assert.ok(Math.abs(a-b)<=tol,`${a} != ${b}`);

test('opposite pixel residuals cancel in signed error, never in MAE/MSE',()=>{
  const loss=imageLoss([.2,.8,.1,.9],[0,1,.5,.5]);close(loss.signed,0);close(loss.mae,.3);close(loss.mse,.1);
  close(imageLoss([51,204,25.5,229.5],[0,255,127.5,127.5]).mse,.1*255**2);
  assert.throws(()=>imageLoss([],[]));assert.throws(()=>imageLoss([NaN],[0]));
});
test('chain rule and curve derivative agree with central differences',()=>{
  close(chain().weight,Math.exp(-.5));close(chain().loss,(Math.exp(-.5)-.8)**2);
  for(const mu of [-1,0,.4,1,2])close(chain(mu).gradient,numericalDerivative(m=>chain(m).loss,mu,1e-5),1e-9);
  for(const mu of [-2,-1,.4,1.5,2.5])close(curveLoss(mu,1.5,1,41).gradient,numericalDerivative(m=>curveLoss(m,1.5,1,41).loss,mu,1e-5),1e-9);
  let mu=-1;for(let k=0;k<100;k++)mu-=2*curveLoss(mu,1.5,1,41).gradient;close(mu,1.5,1e-5);
  assert.throws(()=>chain(0,1,0));assert.throws(()=>numericalDerivative(x=>x*x,1,0));
});
test('all eight image parameters have correct analytic gradients across overlapping layers',()=>{
  const size=6,target=targetImage('flower',size),p=new Float64Array([-.4,.3,1.1,.8,.7,.8,.2,.3,.4,-.2,.7,1.3,.6,.1,.4,.8]);
  const actual=renderAndGradient(p,target,size);
  for(let j=0;j<p.length;j++) {const q=new Float64Array(p),h=1e-5;q[j]+=h;const plus=renderAndGradient(q,target,size,false).loss;q[j]-=2*h;const minus=renderAndGradient(q,target,size,false).loss;close(actual.gradient[j],(plus-minus)/(2*h),1e-8);}
  const scene=exportScene(p),x=-4,y=4;
  const reference=compositeAt(x,y,scene.layers,scene.background);
  const color=reference.color??reference;
  for(let k=0;k<3;k++)close(actual.image[k],color[k]);
});
test('real SGD reduces image error, is deterministic, and stays within scene bounds',()=>{
  const size=16,target=targetImage('flower',size);let p=initializeSpots(target,size,32,42,'grid',.7);
  assert.deepEqual(p,initializeSpots(target,size,32,42,'grid',.7));
  const start=renderAndGradient(p,target,size).loss;
  for(let i=0;i<100;i++){const out=renderAndGradient(p,target,size);p=sgdStep(p,out.gradient,4);}
  const end=renderAndGradient(p,target,size,false).loss;assert.ok(end<start*.75,`loss ${start} -> ${end}`);
  for(const layer of exportScene(p).layers){assert.ok(layer.center.every(x=>x>=-4&&x<=4));assert.ok(layer.scales.every(x=>x>=.2&&x<=3));assert.ok(layer.color.every(x=>x>=0&&x<=1));assert.ok(layer.opacity>=0&&layer.opacity<=1);}
  assert.throws(()=>renderAndGradient(new Float64Array([0,0,0,1,.5,1,0,0]),target,size));
  assert.throws(()=>sgdStep(p,new Float64Array(1),4));
});

test('zero opacity and an opaque foreground keep boundary derivatives finite',()=>{
  const target=new Float64Array(27).fill(.4),p=new Float64Array([0,0,1,1,1,1,0,0,0,0,1,1,.5,0,0,1]);
  const opaque=renderAndGradient(p,target,3);assert.ok(Array.from(opaque.gradient).every(Number.isFinite));
  close(opaque.image[12],1);close(opaque.image[13],0);close(opaque.image[14],0);
  const q=new Float64Array(p);q[4]=0;const zero=renderAndGradient(q,target,3);close(zero.gradient[0],0);close(zero.gradient[1],0);close(zero.gradient[5],0);
  // The derivative of opacity at zero remains meaningful: a currently invisible layer can emerge.
  const h=1e-6;q[4]=h;const right=(renderAndGradient(q,target,3,false).loss-zero.loss)/h;close(zero.gradient[4],right,1e-6);
  assert.throws(()=>initializeSpots(target,3,129));
});
