import test from 'node:test';
import assert from 'node:assert/strict';
import {smallScene} from '../dist/reconstruction-math.js';
import {memoryBudget,packStatic,unpackStatic,packMotion,unpackMotion,sampleTrack,chooseCandidate,
        defaultStudy,studySettings} from '../dist/compression-math.js';

test('byte layout, quantization, and invalid static data',()=>{
  assert.equal(memoryBudget().weights,6552);
  const packet=packStatic(smallScene(),'u8',50,'gray');
  assert.equal(packet.length,56+3*12);
  const decoded=unpackStatic(packet);
  assert.equal(decoded.splats.length,3);
  assert.ok(decoded.splats.every(s=>new Set(s.color).size===1));
  assert.throws(()=>unpackStatic(packet.slice(0,-1)));
  const corrupt=packStatic(smallScene(),'f32');
  new DataView(corrupt.buffer).setFloat32(8+12,-1,true);
  assert.throws(()=>unpackStatic(corrupt));
});
test('signed residuals, shortened last interval, exact anchors',()=>{
  const track=Array.from({length:6},(_,i)=>[[i/5*(1-i/5),0,0]]);
  const packet=packMotion(track,'residual',3,.002),out=unpackMotion(packet);
  assert.equal(packet.length,16+3*12+3*3);
  for(let i=0;i<6;i++)assert.ok(Math.abs(out[i][0][0]-track[i][0][0])<.001001);
  assert.equal(sampleTrack(out,.5)[0][0],(out[2][0][0]+out[3][0][0])/2);
  const bad=packet.slice(),v=new DataView(bad.buffer);v.setUint16(6,65535,true);v.setUint16(8,65535,true);
  assert.throws(()=>unpackMotion(bad));
});
test('empty budgets and selection order',()=>{
  assert.throws(()=>studySettings({...defaultStudy,repeats:0}));
  const rows=[{name:'x',bytes:100,validationPsnr:20,renderMs:4},{name:'y',bytes:200,validationPsnr:21,renderMs:5}];
  assert.equal(chooseCandidate(rows,defaultStudy).name,'y');
  assert.equal(chooseCandidate(rows,{...defaultStudy,priority:'latency'}).name,'x');
  assert.equal(chooseCandidate(rows,{...defaultStudy,maxBytes:1}),null);
});
