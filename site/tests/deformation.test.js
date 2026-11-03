import test from 'node:test';
import assert from 'node:assert/strict';
import {ribbon,fieldValue,fieldScene,exampleModel,sceneAt,referenceScene,
        validateModel,fitMotion,stereoTracks} from '../dist/deformation-math.js';
import {renderScene,orbitCamera,mse} from '../dist/splat-render.js';

test('one bilinear sample controls lift, spin and stretch',()=>{
  assert.equal(fieldValue(.9,.5),.45);
  const tip=fieldScene(.5,-1).splats.at(-1);
  assert.ok(Math.abs(tip.center[1]-.36)<1e-12);
  assert.ok(Math.abs(tip.scales[0]-.23355)<1e-12);
  assert.ok(Math.abs(Math.hypot(...tip.rotation)-1)<1e-12);
});
test('the stereo fit gives the same unseen view and time as Python',()=>{
  const model=exampleModel();
  const fitted=sceneAt(model,.375),reference=referenceScene(.375),cam=orbitCamera(32,64);
  assert.ok(Math.abs(mse(renderScene(fitted,cam,65,49),renderScene(reference,cam,65,49))-.00036425478483558)<1e-12);
  const longer=JSON.parse(JSON.stringify(model));longer.duration=2;
  assert.deepEqual(sceneAt(longer,.75),fitted);
  assert.throws(()=>sceneAt(longer,-1));assert.throws(()=>validateModel({...model,duration:0}));
  assert.throws(()=>validateModel({...model,amplitudes:[]}));
  const row=stereoTracks()[0];assert.throws(()=>fitMotion(ribbon(),[{...row,t:1}]));
});
