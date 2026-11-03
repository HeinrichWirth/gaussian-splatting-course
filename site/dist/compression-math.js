// Little-endian RGB splat packets and fixed-identity motion tracks.
import {renderScene} from './splat-render.js';
import {imageScores} from './reconstruction-math.js';

export function memoryBudget(count=117,degree=0,width=80,height=60) {
  const fields={center:12*count,scales:12*count,rotation:16*count,opacity:4*count,
                color:12*count*(degree+1)**2};
  const weights=Object.values(fields).reduce((a,b)=>a+b,0),rgb=width*height*12;
  return {fields,weights,rgb,viewer:weights+rgb,adam:4*weights};
}
const text=new TextEncoder(),decoder=new TextDecoder();
function header(bytes,signature) {bytes.set(text.encode(signature));return new DataView(bytes.buffer);}
function checkedView(bytes,magic,size) {
  if(bytes.length<size||decoder.decode(bytes.subarray(0,4))!==magic)throw Error('Invalid packet header');
  return new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);
}
export function packStatic(scene,precision='u8',keep=100,color='rgb') {
  const splats=scene.splats;
  if(scene.format!=='gs-course-3d-v1'||splats.length<1||splats.length>65535||splats.some(p=>p.sh)||
     !['u8','f32'].includes(precision)||!['rgb','gray'].includes(color)||!(keep>0&&keep<=100))throw Error('Unsupported static settings');
  const count=Math.ceil(splats.length*keep/100),colors=color==='rgb'?3:1,size=precision==='f32'?4:1;
  const rank=splats.map((p,i)=>{const widths=[...p.scales].sort((a,b)=>b-a);
    return {i,score:p.opacity*widths[0]*widths[1]};}).sort((a,b)=>b.score-a.score);
  const rows=rank.slice(0,count).sort((a,b)=>a.i-b.i).map(({i})=>{
    const p=splats[i],norm=Math.hypot(...p.rotation),rgb=colors===3?p.color:[p.color.reduce((a,b)=>a+b)/3];
    const data=[...p.center,...p.scales,...p.rotation,p.opacity,...rgb];
    if(!data.every(Number.isFinite)||p.scales.some(v=>v<=0)||!norm||
       data.slice(10).some(v=>v<0||v>1))throw Error('Invalid Gaussian');
    if(size===1)data.splice(6,4,...p.rotation.map(v=>v/norm));
    return data;
  });
  const bytes=new Uint8Array(8+(size===1?48:0)+count*(11+colors)*size),v=header(bytes,'GSC1');
  v.setUint8(4,size);v.setUint8(5,colors);v.setUint16(6,count,true);
  let pos=8;
  const lo=[],hi=[];
  if(size===1){
    for(let i=0;i<6;i++){let a=Infinity,b=-Infinity;
      for(const row of rows){a=Math.min(a,row[i]);b=Math.max(b,row[i]);}
      lo.push(Math.fround(a));hi.push(Math.fround(b));}
    for(const b of [...lo,...hi]){v.setFloat32(pos,b,true);pos+=4;}
    lo.push(-1,-1,-1,-1,...Array(1+colors).fill(0));hi.push(...Array(5+colors).fill(1));
  }
  for(const row of rows)for(const [i,x] of row.entries()){
    if(size===4){v.setFloat32(pos,x,true);pos+=4;}
    else {const u=hi[i]===lo[i]?0:(x-lo[i])/(hi[i]-lo[i]);
      bytes[pos++]=Math.floor(255*Math.max(0,Math.min(1,u))+.5);}
  }
  return bytes;
}
export function unpackStatic(bytes) {
  const v=checkedView(bytes,'GSC1',8),size=v.getUint8(4),colors=v.getUint8(5),n=v.getUint16(6,true);
  if(![1,4].includes(size)||![1,3].includes(colors)||!n||
     bytes.length!==8+(size===1?48:0)+n*(11+colors)*size)throw Error('Invalid GSC1 length');
  let pos=8;const splats=[],lo=[],hi=[];
  if(size===1){for(const bound of [lo,hi])for(let j=0;j<6;j++){bound.push(v.getFloat32(pos,true));pos+=4;}
    if(lo.some((x,i)=>!Number.isFinite(x)||!Number.isFinite(hi[i])||hi[i]<x))throw Error('Invalid geometry bounds');
    lo.push(-1,-1,-1,-1,...Array(1+colors).fill(0));hi.push(...Array(5+colors).fill(1));}
  for(let i=0;i<n;i++){const row=[];
    for(let j=0;j<11+colors;j++){row.push(size===4?v.getFloat32(pos,true):lo[j]+v.getUint8(pos)/255*(hi[j]-lo[j]));pos+=size;}
    if(!row.every(Number.isFinite)||row.slice(3,6).some(x=>x<=0)||!Math.hypot(...row.slice(6,10))||
       row.slice(10).some(x=>x<0||x>1))throw Error('Invalid decoded Gaussian');
    splats.push({center:row.slice(0,3),scales:row.slice(3,6),rotation:row.slice(6,10),
                 opacity:row[10],color:colors===3?row.slice(11):Array(3).fill(row[11])});
  }
  return {format:'gs-course-3d-v1',splats};
}
export function anchorFrames(frames,stride) {
  const keys=[];for(let i=0;i<frames-1;i+=stride)keys.push(i);keys.push(frames-1);return keys;
}
function mixPoints(a,b,u) {return a.map((p,i)=>p.map((v,d)=>v*(1-u)+b[i][d]*u));}
function keyPrediction(keys,values,frame) {
  let right=1;while(right<keys.length-1&&keys[right]<frame)right++;
  return mixPoints(values[right-1],values[right],(frame-keys[right-1])/(keys[right]-keys[right-1]));
}
export function packMotion(xyz,mode='residual',stride=4,quantum=.002) {
  const frames=xyz.length,count=xyz[0]?.length,code=['dense','keys','residual'].indexOf(mode);
  if(!(frames>=2&&frames<=65535&&count>=1&&count<=65535&&frames*count<=250000&&Number.isInteger(stride)&&stride>=1&&stride<=65535&&
      code>=0&&Number.isFinite(quantum)&&quantum>0)||xyz.some(row=>row.length!==count||
      row.some(p=>p.length!==3||!p.every(Number.isFinite))))throw Error('Invalid motion settings');
  quantum=Math.fround(quantum);if(code===0)stride=1;
  const keys=anchorFrames(frames,stride),isKey=new Set(keys),values=keys.map(k=>xyz[k].map(p=>p.map(Math.fround)));
  const bytes=new Uint8Array(16+keys.length*count*12+(code===2?(frames-keys.length)*count*3:0));
  const v=header(bytes,'GSM1');v.setUint8(4,code);v.setUint16(6,frames,true);v.setUint16(8,count,true);
  v.setUint16(10,stride,true);v.setFloat32(12,quantum,true);let pos=16;
  for(const row of values)for(const p of row)for(const x of p){v.setFloat32(pos,x,true);pos+=4;}
  if(code===2)for(let f=0;f<frames;f++)if(!isKey.has(f)){
    const pred=keyPrediction(keys,values,f);
    xyz[f].forEach((p,i)=>p.forEach((x,d)=>v.setInt8(pos++,Math.max(-127,Math.min(127,Math.floor((x-pred[i][d])/quantum+.5))))));
  }
  return bytes;
}
export function unpackMotion(bytes) {
  const v=checkedView(bytes,'GSM1',16),code=v.getUint8(4),frames=v.getUint16(6,true),count=v.getUint16(8,true);
  const stride=v.getUint16(10,true),quantum=v.getFloat32(12,true);
  if(code>2||v.getUint8(5)||frames<2||!count||!stride||frames*count>250000||(code===0&&stride!==1)||
     !Number.isFinite(quantum)||quantum<=0)throw Error('Invalid GSM1 header');
  const keys=anchorFrames(frames,stride),isKey=new Set(keys);
  if(bytes.length!==16+keys.length*count*12+(code===2?(frames-keys.length)*count*3:0))throw Error('Invalid GSM1 length');
  let pos=16;
  const values=keys.map(()=>Array.from({length:count},()=>Array.from({length:3},()=>{
    const x=v.getFloat32(pos,true);pos+=4;if(!Number.isFinite(x))throw Error('Nonfinite motion key');return x;})));
  if(code===0)return values;
  return Array.from({length:frames},(_,f)=>{
    const prediction=keyPrediction(keys,values,f);
    if(code===2&&!isKey.has(f))for(const p of prediction)for(let d=0;d<3;d++)p[d]+=v.getInt8(pos++)*quantum;
    return prediction;
  });
}
export function sampleTrack(track,t) {
  if(!(t>=0&&t<=1))throw Error('Time outside clip');
  const f=t*(track.length-1),i=Math.min(Math.floor(f),track.length-2);
  return mixPoints(track[i],track[i+1],f-i);
}
export function poseWithCenters(canonical,centers) {
  return {...canonical,splats:canonical.splats.map((p,i)=>({...p,center:centers[i]}))};
}
export function medianTime(fn,repeats=3) {
  fn();const ms=[];let batch=1;
  function timed(){const start=performance.now();for(let j=0;j<batch;j++)fn();return performance.now()-start;}
  while(timed()<2&&batch<8192)batch*=2;
  for(let i=0;i<repeats;i++)ms.push(timed()/batch);
  ms.sort((a,b)=>a-b);const m=Math.floor(ms.length/2);
  return ms.length%2?ms[m]:(ms[m-1]+ms[m])/2;
}
export const defaultStudy={format:'gs-course-codec-study-v1',width:80,height:60,repeats:3,
                           maxBytes:2200,minPsnr:16,priority:'quality'};
export const candidates=[['f32-rgb','f32',100,'rgb'],['u8-rgb','u8',100,'rgb'],
                         ['u8-half','u8',50,'rgb'],['u8-gray-half','u8',50,'gray']];
export function studySettings(cfg) {
  if(!cfg||Object.keys(cfg).sort().join()!==Object.keys(defaultStudy).sort().join()||
     cfg.format!==defaultStudy.format||cfg.width!==80||cfg.height!==60||!Number.isInteger(cfg.repeats)||
     cfg.repeats<1||cfg.repeats>15||!Number.isInteger(cfg.maxBytes)||cfg.maxBytes<1||cfg.maxBytes>100000||
     !Number.isFinite(cfg.minPsnr)||cfg.minPsnr<0||cfg.minPsnr>60||!['quality','size','latency'].includes(cfg.priority))
    throw Error('Unsupported study settings');
  return {...cfg};
}
export function chooseCandidate(rows,cfg) {
  const eligible=rows.filter(r=>r.bytes<=cfg.maxBytes&&r.validationPsnr>=cfg.minPsnr);
  const rank=r=>cfg.priority==='quality'?-r.validationPsnr:cfg.priority==='size'?r.bytes:r.renderMs;
  return eligible.sort((a,b)=>rank(a)-rank(b))[0]||null;
}
export function photoStudy(scene,frames,settings) {
  const cfg=studySettings(settings),view=frames.find(f=>f.name==='view-00.jpg'),packets=new Map();
  const rows=candidates.map(([name,precision,keep,color])=>{
    const packet=packStatic(scene,precision,keep,color);packets.set(name,packet);
    const decoded=unpackStatic(packet),render=()=>renderScene(decoded,view.camera,80,60);
    return {name,bytes:packet.length,splats:decoded.splats.length,validationPsnr:imageScores(render(),view.rgb).psnr,
      decodeMs:medianTime(()=>unpackStatic(packet),cfg.repeats),renderMs:medianTime(render,cfg.repeats),bufferBytes:80*60*3*8};
  });
  const winner=chooseCandidate(rows,cfg);let test=null;
  if(winner){const held=frames.find(f=>f.split==='test'),model=unpackStatic(packets.get(winner.name));
    const {mse,psnr}=imageScores(renderScene(model,held.camera,80,60),held.rgb);
    test={view:held.name,mse,psnr};}
  return {settings:cfg,candidates:rows,selected:winner?.name??null,validationView:view.name,test};
}
