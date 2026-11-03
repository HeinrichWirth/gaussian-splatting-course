import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {decodeCapture} from '../dist/reconstruction-math.js';
import {projectSplats,renderScene} from '../dist/splat-render.js';
import {observedPixel,intersectRays,projectFeature,pixelLedger,regionScores} from '../dist/photo-diagnostics.js';

function asset(name) {
  return JSON.parse(readFileSync(new URL('../dist/assets/reconstruction/'+name,import.meta.url),'utf8'));
}
const {frames}=decodeCapture(asset('photographs-80.json'));
const scene=asset('photo-reference.json');
const frame=frames.find(f=>f.split==='test');
const close=(actual,expected,tolerance=1e-7)=>assert.ok(Math.abs(actual-expected)<tolerance,`${actual} ≠ ${expected}`);

test('two measured rays predict the third view; perturbing one mark changes the prediction',()=>{
  const track=asset('track-148.json');
  const [a,b,c]=['view-03.jpg','view-08.jpg','view-07.jpg'].map(name=>frames.find(f=>f.name===name));
  const [ua,ub,uc]=[a,b,c].map(f=>observedPixel(track,f));
  const error=dx=>{
    const fit=intersectRays(a,ua,b,[ub[0]+dx,ub[1]]);
    return Math.hypot(...projectFeature(fit.point,c).map((v,i)=>v-uc[i]));
  };
  close(error(0),.4,.01);
  close(error(3),2.01,.01);
  assert.equal(intersectRays(a,ua,a,ua),null);
});

test('individual contributions reconstruct the actual reference render',()=>{
  const projected=projectSplats(scene,frame.camera,frame.width,frame.height);
  const image=renderScene(scene,frame.camera,frame.width,frame.height);
  for(const [u,v,first] of [[40,30,25],[0,0,13]]){
    const ledger=pixelLedger(projected,u,v);
    close(ledger.background+ledger.rows.reduce((sum,r)=>sum+r.weight,0),1);
    assert.equal(ledger.rows[0].id,first);
    ledger.rgb.forEach((value,c)=>close(value,image[3*(v*frame.width+u)+c]));
  }
});

test('moving a region redistributes a fixed total error',()=>{
  const predicted=renderScene(scene,frame.camera,frame.width,frame.height);
  const center=regionScores(frame,predicted,20,15,40,30);
  const corner=regionScores(frame,predicted,0,0,40,30);
  close(center.global,.00981748); close(center.mse,.01193155);
  close(center.global,center.mse/4+center.outside*3/4);
  close(center.share,center.mse/4/center.global);
  close(center.global,corner.global);
  assert.notEqual(center.share,corner.share);
  assert.equal(regionScores(frame,frame.rgb,0,0,40,30).share,0);
});
