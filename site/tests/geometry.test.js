import test from 'node:test';
import assert from 'node:assert/strict';
import {axisAngleQuaternion, quaternionMatrix, covariance3d, projectPoint,
  worldToCamera, projectionJacobian, projectedCovariance} from '../dist/geometry-math.js';

const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-8, `${a} differs from ${b}`);

test('a right turn reads world x as camera depth',()=>{
  const rot=quaternionMatrix(axisAngleQuaternion([0,1,0],90));
  const p=worldToCamera([5,-1,1],[1,-2,3],rot);
  p.forEach((n,i)=>close(n,[2,1,4][i]));
});
test('rotated scales and an unnormalized negative quaternion describe the same shell',()=>{
  const q=axisAngleQuaternion([0,0,1],45);
  const cov=covariance3d([2,.75,1],q.map(n=>-4*n));
  close(cov[0][0],2.28125);close(cov[0][1],1.71875);close(cov[2][2],1);
});
test('local change agrees with perspective using unequal focal lengths',()=>{
  const p=[1.1,-.3,2.7], f=[80,115], eps=1e-5, j=projectionJacobian(p,f);
  for(let i=0;i<3;i++){
    const a=p.slice(),b=p.slice();a[i]+=eps;b[i]-=eps;
    const plus=projectPoint(a,f),minus=projectPoint(b,f);
    j.forEach((row,k)=>close(row[i],(plus[k]-minus[k])/(2*eps)));
  }
  assert.throws(()=>projectPoint([1,0,.2],80,[100,60],.2));
  assert.deepEqual(projectedCovariance([1,0,4],[[.25,0,0],[0,.09,0],[0,0,1]],80),[[125,0],[0,36]]);
});
