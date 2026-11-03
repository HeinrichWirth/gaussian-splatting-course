// Camera geometry. Points are columns; matrices are ordinary arrays of rows.
export const dot = (a, b) => a.reduce((sum, value, i) => sum + value*b[i], 0);
export const transpose = a => a[0].map((_, j) => a.map(row => row[j]));
export const matVec = (a, v) => a.map(row => dot(row, v));
export const multiply = (a, b) => a.map(row => matVec(transpose(b), row));

function vector(value, size) {
  if (value?.length !== size || !Array.from(value).every(Number.isFinite)) {
    throw new RangeError('Expected a finite vector of length '+size);
  }
  return Array.from(value);
}
function unit(value, size) {
  const v = vector(value,size);
  const scale = Math.max(...v.map(Math.abs));
  if (!scale) throw new RangeError('Zero vector has no direction');
  const scaled = v.map(x => x/scale);
  const length = Math.hypot(...scaled);
  return scaled.map(x => x/length);
}
export function axisAngleQuaternion(axis, angleDeg) {
  const direction = unit(axis,3);
  if (!Number.isFinite(angleDeg)) throw new RangeError('Angle must be finite');
  const half = angleDeg*Math.PI/360;
  return [Math.cos(half), ...direction.map(x => x*Math.sin(half))];
}
export function quaternionMatrix(quaternion) {
  const [w,x,y,z] = unit(quaternion,4);
  return [
    [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
    [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
    [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
  ];
}
export function worldToCamera(point, cameraCenter, cameraToWorld) {
  return matVec(transpose(cameraToWorld), point.map((x,i) => x-cameraCenter[i]));
}
function focalPair(focal) {
  const pair = typeof focal==='number' ? [focal,focal] : vector(focal,2);
  if (!pair.every(x => Number.isFinite(x) && x>0)) throw new RangeError('Invalid focal length');
  return pair;
}
export function projectPoint(cameraXYZ, focal, principal=[0,0], near=.05) {
  const [x,y,z] = vector(cameraXYZ,3);
  if (!Number.isFinite(near) || near<0 || z<=near) throw new RangeError('Point is behind near plane');
  const [fx,fy] = focalPair(focal);
  return [principal[0]+fx*x/z,principal[1]+fy*y/z];
}
export function projectionJacobian(cameraXYZ,focal) {
  const [x,y,z] = vector(cameraXYZ,3), [fx,fy] = focalPair(focal);
  if (z<=0) throw new RangeError('Depth must be positive');
  return [[fx/z,0,-fx*x/(z*z)],[0,fy/z,-fy*y/(z*z)]];
}
export function covariance3d(scales,quaternion) {
  if (!vector(scales,3).every(x=>x>0)) throw new RangeError('Scales must be positive');
  const basis = quaternionMatrix(quaternion).map(row=>row.map((x,j)=>x*scales[j]));
  return multiply(basis,transpose(basis));
}
export function projectedCovariance(cameraXYZ,cameraCovariance,focal) {
  const j = projectionJacobian(cameraXYZ,focal);
  return multiply(multiply(j,cameraCovariance),transpose(j));
}
