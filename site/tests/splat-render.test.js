import test from 'node:test';
import assert from 'node:assert/strict';
import {overlapScene,orbitCamera,renderScene,shColor,rayLoss,depthStep,
  densityScene,densityLoss,densityEdit,densityPass} from '../dist/splat-render.js';

const close=(a,b,eps=1e-8)=>assert.ok(Math.abs(a-b)<eps,a+' ≠ '+b);
test('the public renderer mixes front-to-back and reversing the splats changes RGB',()=>{
  const cam=orbitCamera(),scene=overlapScene(),center=3*(65*32+32);
  for(const [reverse,expected] of [[false,[.8,.2,.4]],[true,[.5,.2,.7]]]){
    const img=renderScene(scene,cam,65,65,[1,1,1],reverse);
    expected.forEach((v,i)=>close(img[center+i],v));
  }
});
test('SH sign convention, camera normalization and one-view ambiguity match Python',()=>{
  close(shColor([3,0,4],[[0,0,0],[0,0,0],[.4,0,0],[.6,0,0]])[0],.480455899524);
  close(rayLoss(2.4,[0]),0);
  close(depthStep(2.4,[0,.8])[0],2.7333520655);
  assert.ok(rayLoss(depthStep(2.4,[0,.8])[0],[-.6])<rayLoss(2.4,[-.6]));
});
test('split, prune and opacity reset affect scene topology and measured pixels',()=>{
  const scene=densityScene();
  close(densityLoss(densityEdit(scene,'split')), .003472106339);
  close(densityLoss(densityEdit(scene,'clone')), .012146590248);
  const adaptive=densityPass(scene);
  assert.equal(adaptive.splats.length,2);
  close(adaptive.splats[0].center[0],-.5695);
  assert.equal(densityEdit(densityEdit(scene,'reset'),'prune').splats.length,0);
  assert.deepEqual(scene,densityScene());
});
