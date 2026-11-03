import test from 'node:test';
import assert from 'node:assert/strict';
import {tiledFrame,smallScene,orbit,syntheticCapture,trainingSystem,fitColors,
        imageScores,triangulate,decodeCapture,damagedScene} from '../dist/reconstruction-math.js';
import {renderScene} from '../dist/splat-render.js';

test('tile lists preserve the finite-support reference and reduce candidate tests',()=>{
  const view=orbit(0,73,57),scene=smallScene();
  const fast=tiledFrame(scene,view,73,57,8,3),slow=tiledFrame(scene,view,73,57,8,3,false);
  assert.equal(fast.pairs,8688);assert.equal(slow.pairs,20805);assert.deepEqual(fast.rgb,slow.rgb);
  const changed=tiledFrame(scene,view,73,57,16,1.5);
  assert.ok(imageScores(changed.rgb,fast.rgb).mse>0);
});

test('only train RGB affects fitted scene; held-out view matches the NumPy fit',()=>{
  const {initial,frames}=syntheticCapture();
  const trained=fitColors(initial,trainingSystem(initial,frames),12);
  const testFrame=frames.find(f=>f.split==='test');
  const mse=imageScores(renderScene(trained,testFrame.camera,41,33),testFrame.rgb).mse;
  assert.ok(Math.abs(mse-9.065363618718987e-6)<1e-11);
  testFrame.rgb.fill(0);
  assert.deepEqual(trained,fitColors(initial,trainingSystem(initial,frames),12));
  assert.deepEqual(damagedScene(trained,'floater',0),trained);
});

test('identical frames, degenerate baselines, invalid portable captures',()=>{
  assert.equal(imageScores([.1,.2],[.1,.2]).psnr,Infinity);
  assert.equal(triangulate(80,0,40,40),null);
  assert.equal(triangulate(80,.4,40,32),4);
  assert.throws(()=>decodeCapture({format:'gs-course-capture-v1',frames:[]}));
});
