import assert from 'node:assert/strict';
import test from 'node:test';
import {observingPair, observationImage, trueX, fitPath, trajectorySplat, persistentPair} from '../dist/motion-math.js';
import {makeScene, translatedCamera, projectSplats} from '../dist/splat-render.js';

const close=(actual,expected)=>assert.ok(Math.abs(actual-expected)<1e-9,actual+' ≠ '+expected);
test('a delayed shutter biases stereo; the occluder removes only the left mark',()=>{
  close(observingPair().estimate,4);
  close(observingPair(.8,1,0,.2).estimate,16/3);
  assert.equal(observingPair(.8,1,0,.8).estimate,null);
  const {views,estimate}=observingPair(.8,1,0,0,true);
  assert.deepEqual(views.map(v=>v.visible),[false,true]);assert.equal(estimate,null);
  views.forEach(v=>{
    const i=3*(30*97+Math.round(v.u));
    const pixel=image=>Array.from(image.slice(i,i+3));
    assert.deepEqual(pixel(observationImage(v,true)),v.visible?pixel(observationImage(v)):[.23,.29,.30]);
  });
});
test('three timed centers recover the quadratic and amplify middle noise beyond the samples',()=>{
  close(fitPath(1),.1);close(trueX(1),-.1);
  close(fitPath(.5,true,.1)-trueX(.5),.075);close(fitPath(3,true,.1)-trueX(3),-.3);
  const [projected]=projectSplats(makeScene(trajectorySplat(1,fitPath(1,true))),translatedCamera(0,80),97,61);
  close(projected.mean[0],46);assert.ok(projected.covariance[0][1]>0);
});
test('the neighbour prior can repair a noisy rigid track or bias a true bend',()=>{
  const result=persistentPair();
  close(result.fitted[0],.08);close(result.fitted[1],.76);
  close(result.trackingError,.4472135955/5);
  close(persistentPair(90,0,0,40).trackingError,0);
  assert.ok(persistentPair(90,0,4,40).trackingError>.4);
});
