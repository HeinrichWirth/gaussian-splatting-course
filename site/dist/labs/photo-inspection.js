import {addSelect,el} from './shared.js';
import {lab,number,panel,pixels,stats,done} from './reconstruction.js';
import {decodeCapture} from '../reconstruction-math.js';
import {projectSplats,renderScene} from '../splat-render.js';
import {observedPixel,intersectRays,projectFeature,pixelLedger,regionScores} from '../photo-diagnostics.js';

async function asset(name) {
  const response=await fetch(new URL('../assets/reconstruction/'+name,import.meta.url));
  if(!response.ok)throw Error(response.statusText);
  return response.json();
}
const fmt=(x,n=3)=>Number(x.toFixed(n)).toString();
const vec=(a,n=3)=>a.map(v=>fmt(v,n)).join(' / ');

function start(ui,state,extra,render) {
  let live=true;
  const credit=el('a','',ui.t('Фотографии NVIDIA · CC BY 4.0','NVIDIA photographs · CC BY 4.0'));
  credit.href='https://github.com/HeinrichWirth/gaussian-splatting-course/tree/main/data/photographic-example';
  ui.footer.append(credit);
  const status=el('p','',ui.t('Загрузка снимков…','Loading photographs…'));ui.footer.append(status);
  Promise.all([asset('photographs-80.json'),asset(extra)]).then(([data,item])=>{
    if(live){render(decodeCapture(data).frames,item);status.remove();}
  }).catch(()=>{if(live)status.textContent=ui.t('Не удалось загрузить фотографии.','The photographs could not be loaded.');});
  return done(ui,state,()=>{live=false;});
}

// A cropped photo is enlarged only for display. Coordinates stay in source pixels.
function zoom(canvas,frame,center,marks) {
  const original=el('canvas');pixels(original,frame.rgb,frame.width,frame.height);
  const size=16, scale=12;
  canvas.width=canvas.height=size*scale;
  const ctx=canvas.getContext('2d');ctx.imageSmoothingEnabled=false;
  const left=center[0]+.5-size/2, top=center[1]+.5-size/2;
  ctx.drawImage(original,left,top,size,size,0,0,canvas.width,canvas.height);
  for(const [i,[u,v]] of marks.entries()){
    const x=(u+.5-left)*scale,y=(v+.5-top)*scale;
    ctx.strokeStyle=i?'#3dbbff':'#ffad3f';ctx.lineWidth=2;ctx.beginPath();
    if(i)ctx.arc(x,y,9,0,Math.PI*2);
    else {ctx.moveTo(x-7,y);ctx.lineTo(x+7,y);ctx.moveTo(x,y-7);ctx.lineTo(x,y+7);}
    ctx.stroke();
  }
}

export function mountTrack(host,lang,saved={}) {
  const ui=lab(host,lang,'Одна метка на трёх фотографиях','One feature in three photographs',
    'Первые два кадра задают точку. Третий проверяет её. Крест — отметка, синий круг — проекция.',
    'Two images place the point. The third checks it. Crosses mark observations, blue rings mark projections.');
  const t=ui.t,s={pair:8,offset:0,...saved};
  const canvases=['view-03',t('Второй снимок','Second image'),'view-07'].map(v=>panel(ui,v));
  return start(ui,s,'track-148.json',(frames,track)=>{
    const get=n=>frames.find(f=>f.name==='view-0'+n+'.jpg');
    addSelect(ui.controls,{label:t('Второй снимок','Second image'),options:[6,8].map(n=>({value:n,label:'view-0'+n})),
      value:s.pair,onChange:v=>{s.pair=Number(v);draw();}});
    number(ui,t('Ошибка метки по горизонтали, px','Horizontal mark error, px'),s.offset,-3,3,.1,v=>{s.offset=v;draw();});
    function draw(){
      const views=[get(3),get(s.pair),get(7)],marks=views.map(f=>observedPixel(track,f));
      marks[1][0]+=s.offset;
      const fit=intersectRays(views[0],marks[0],views[1],marks[1]);
      if(!fit)return;
      const projected=views.map(f=>projectFeature(fit.point,f));
      views.forEach((f,i)=>zoom(canvases[i],f,observedPixel(track,f),[marks[i],projected[i]]));
      const error=Math.hypot(...projected[2].map((v,i)=>v-marks[2][i]));
      stats(ui,[['X / Y / Z',vec(fit.point,4)],[t('Зазор лучей, ед. сцены','Ray gap, scene units'),fmt(fit.gap,4)],
        [t('Метка на третьем снимке, px','Third observation, px'),vec(marks[2])],
        [t('Проекция на третьем снимке, px','Third-view projection, px'),vec(projected[2])],
        [t('Ошибка на третьем снимке, px','Third-view error, px'),fmt(error,4)]]);
    }
    draw();
  });
}

export function mountLedger(host,lang,saved={}) {
  const ui=lab(host,lang,'Какие пятна нарисовали пиксель','Which splats made this pixel?',
    'Щелчок по фотографии переносит метку на другой пиксель. Под кадром — три самых больших вклада из обученной сцены.',
    'A click moves the marker to another pixel. The table shows the three largest contributions in the trained scene.');
  const t=ui.t,s={u:40,v:30,...saved},photo=panel(ui,t('Снимок view-04','Photo view-04'));
  return start(ui,s,'photo-reference.json',(frames,model)=>{
    const f=frames.find(x=>x.split==='test'), spots=projectSplats(model,f.camera,f.width,f.height);
    const setU=number(ui,'u, px',s.u,0,f.width-1,1,v=>{s.u=Math.round(v);draw();});
    const setV=number(ui,'v, px',s.v,0,f.height-1,1,v=>{s.v=Math.round(v);draw();});
    photo.style.cursor='crosshair';
    photo.addEventListener('click',event=>{
      const bounds=photo.getBoundingClientRect();
      const x=(event.clientX-bounds.left-photo.clientLeft)/photo.clientWidth;
      const y=(event.clientY-bounds.top-photo.clientTop)/photo.clientHeight;
      s.u=Math.max(0,Math.min(f.width-1,Math.floor(x*f.width)));
      s.v=Math.max(0,Math.min(f.height-1,Math.floor(y*f.height)));
      setU(s.u);setV(s.v);draw();
    });
    function draw(){
      pixels(photo,f.rgb,f.width,f.height);
      const ctx=photo.getContext('2d');ctx.strokeStyle='orange';ctx.lineWidth=.6;
      ctx.strokeRect(s.u-1,s.v-1,3,3);
      const ledger=pixelLedger(spots,s.u,s.v),i=3*(s.v*f.width+s.u);
      stats(ui,[...ledger.rows.slice(0,3).map(r=>['#'+r.id+' · T × α',fmt(r.trans,4)+' × '+fmt(r.alpha,4)+' = '+fmt(r.weight,4)]),
        [t('Вес белого фона','White background weight'),fmt(ledger.background,5)],['RGB · GS',vec(ledger.rgb)],
        ['RGB · '+t('фото','photo'),vec(f.rgb.slice(i,i+3))]]);
    }
    draw();
  });
}

export function mountRegion(host,lang,saved={}) {
  const ui=lab(host,lang,'Где спряталась ошибка','Where the error sits',
    'Рамка делит один и тот же снимок на две части. Для каждой можно посчитать свою ошибку.',
    'The box divides a single photograph into two parts. Each part has its own error.');
  const t=ui.t,s={left:20,top:15,...saved},photo=panel(ui,'RGB'),map=panel(ui,'RMS Δ');
  return start(ui,s,'photo-reference.json',(frames,model)=>{
    const f=frames.find(x=>x.split==='test'),pred=renderScene(model,f.camera,f.width,f.height);
    const difference=new Float64Array(pred.length);
    for(let i=0;i<pred.length;i+=3)difference.fill(Math.hypot(...pred.slice(i,i+3).map((v,j)=>v-f.rgb[i+j]))/Math.sqrt(3),i,i+3);
    number(ui,'u₀, px',s.left,0,40,1,v=>{s.left=Math.round(v);draw();});
    number(ui,'v₀, px',s.top,0,30,1,v=>{s.top=Math.round(v);draw();});
    function draw(){
      pixels(photo,f.rgb,f.width,f.height);pixels(map,difference,f.width,f.height);
      for(const c of [photo,map]){const ctx=c.getContext('2d');ctx.strokeStyle='#25d9df';ctx.lineWidth=.5;ctx.strokeRect(s.left,s.top,40,30);}
      const r=regionScores(f,pred,s.left,s.top,40,30);
      stats(ui,[['MSE · '+t('внутри','inside'),fmt(r.mse,6)],['MSE · '+t('снаружи','outside'),fmt(r.outside,6)],
        ['MSE · '+t('весь кадр','full frame'),fmt(r.global,6)],
        [t('Доля всей ошибки в рамке','Share of total squared error inside'),fmt(r.share*100,2)+' %']]);
    }
    draw();
  });
}
