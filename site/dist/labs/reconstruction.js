import {makeLab,el,addSelect,addButton} from './shared.js';
import {renderScene} from '../splat-render.js';
import {smallScene,orbit,tiledFrame,triangulate,syntheticCapture,trainingSystem,fitColors,
        validateScene,decodeCapture,imageScores,damagedScene} from '../reconstruction-math.js';

let nextId=0;
function lab(host,lang,ru,en,subRu,subEn) {
  const t=(a,b)=>lang==='en'?b:a;
  const ui=makeLab(host,{lang,title:t(ru,en),subtitle:t(subRu,subEn)});
  if(!document.querySelector('[data-reconstruction-css]')) {
    const link=el('link');link.rel='stylesheet';link.href=new URL('./reconstruction.css',import.meta.url);
    link.dataset.reconstructionCss='';document.head.append(link);
  }
  ui.root.classList.add('reconstruction-lab');
  ui.t=t;
  ui.images=el('div','recon-images');ui.view.append(ui.images);
  return ui;
}
function number(ui,label,value,min,max,step,change) {
  const field=el('div','control'),header=el('div','control-top'),caption=el('label','',label);
  const input=el('input'),range=el('input');
  input.type='number';range.type='range';input.id='recon-number-'+(++nextId);caption.htmlFor=input.id;
  for(const n of [input,range]) {Object.assign(n,{value,min,max,step});n.setAttribute('aria-label',label);}
  input.addEventListener('input',()=>{
    const v=input.valueAsNumber;
    if(Number.isFinite(v)&&v>=min&&v<=max){range.value=String(v);change(v);}
  });
  input.addEventListener('blur',()=>{input.value=range.value;});
  range.oninput=()=>{input.value=range.value;change(Number(range.value));};
  header.append(caption,input);field.append(header,range);ui.controls.append(field);
  return v=>{input.value=range.value=v;};
}
function choose(ui,label,opts,value,change) {
  return addSelect(ui.controls,{label,value:String(value),options:opts.map(([v,s])=>({value:String(v),label:s})),
    onChange:change});
}
function panel(ui,title) {
  const f=el('figure'),caption=el('figcaption','',title),c=el('canvas');
  c.setAttribute('role','img');c.setAttribute('aria-label',title);f.append(caption,c);ui.images.append(f);
  return c;
}
function pixels(canvas,rgb,w,h) {
  canvas.width=w;canvas.height=h;
  const ctx=canvas.getContext('2d'),img=ctx.createImageData(w,h);
  for(let i=0;i<w*h;i++) {
    for(let c=0;c<3;c++) img.data[4*i+c]=Math.round(255*Math.max(0,Math.min(1,rgb[3*i+c])));
    img.data[4*i+3]=255;
  }
  ctx.putImageData(img,0,0);
}
function stats(ui,rows) {
  ui.readout.replaceChildren();
  const table=el('table');const body=el('tbody');
  for(const [name,value] of rows) {
    const tr=el('tr'),th=el('th','',name);th.scope='row';tr.append(th,el('td','',String(value)));body.append(tr);
  }
  table.append(body);ui.readout.append(table);
}
const fmt=(v,p=5)=>Number(v.toFixed(p)).toString();
function done(ui,state,shutdown=()=>{}) {
  return {getState:()=>state,destroy:()=>{shutdown();ui.root.remove();}};
}
export function mountTiles(host,lang='ru',saved={}) {
  const ui=lab(host,lang,'Сколько пятен видит тайл','How many splats reach a tile',
    'Цвет слева получается из отсортированных пятен. Чем длиннее список тайла, тем насыщеннее синий справа.',
    'The image on the left blends sorted splats. Deeper blue on the right marks a longer tile list.');
  const t=ui.t,s={tile:8,cutoff:3,angle:0,...saved},w=73,h=57;
  const frame=panel(ui,t('Кадр','Frame')),map=panel(ui,t('Списки тайлов','Tile lists'));
  choose(ui,t('Сторона тайла, пикс.','Tile side, px'),[4,8,16,32].map(n=>[n,String(n)]),s.tile,
    v=>{s.tile=+v;draw();});
  number(ui,t('Граница, σ','Cutoff, σ'),s.cutoff,1,4,.1,v=>{s.cutoff=v;draw();});
  number(ui,t('Ракурс, градусы','View angle, degrees'),s.angle,-45,45,1,v=>{s.angle=v;draw();});
  function draw() {
    const sc=smallScene(),cam=orbit(s.angle,w,h),fast=tiledFrame(sc,cam,w,h,s.tile,s.cutoff);
    const full=tiledFrame(sc,cam,w,h,s.tile,s.cutoff,false),exact=renderScene(sc,cam,w,h);
    pixels(frame,fast.rgb,w,h);const colors=new Float64Array(w*h*3);
    for(let y=0;y<h;y++) for(let x=0;x<w;x++) {
      const n=fast.counts[Math.floor(y/s.tile)*fast.nx+Math.floor(x/s.tile)]/sc.splats.length;
      const grid=x%s.tile===0||y%s.tile===0;
      colors.set(grid?[.68,.72,.73]:[.96-.70*n,.98-.52*n,.99-.10*n],(y*w+x)*3);
    }
    pixels(map,colors,w,h);
    stats(ui,[[t('Проверок по всему кадру','Full-frame tests'),full.pairs],[t('Проверок по спискам','Binned tests'),fast.pairs],
      [t('Ссылок «тайл → пятно»','Tile–splat references'),fast.counts.reduce((a,b)=>a+b,0)],
      [t('Макс. разность при одной границе','Max difference at equal cutoff'),fmt(maxDiff(fast.rgb,full.rgb),9)],
      [t('Макс. изменение от отсечения хвоста','Max change from cutting tails'),fmt(maxDiff(fast.rgb,exact))]]);
  }
  draw();return done(ui,s);
}
function maxDiff(a,b){let max=0;for(let i=0;i<a.length;i++) max=Math.max(max,Math.abs(a[i]-b[i]));return max;}

export function mountCapture(host,lang='ru',saved={}) {
  const ui=lab(host,lang,'Две камеры и одна метка','Two cameras and one marked point',
    'Вид сверху. Положение камер и метка известны. Смещение наблюдения поворачивает правый луч.',
    'Top view. The cameras and target are known. An offset in the right observation tilts its ray.');
  const t=ui.t,s={baseline:.4,noise:0,z:4,...saved},canvas=panel(ui,t('Камеры и лучи','Cameras and rays'));
  number(ui,t('База B, м','Baseline B, m'),s.baseline,0,1,.01,v=>{s.baseline=v;draw();});
  number(ui,t('Глубина Z, м','True depth Z, m'),s.z,1,8,.1,v=>{s.z=v;draw();});
  number(ui,t('Ошибка правой метки, пикс.','Right observation error, px'),s.noise,-4,4,.1,v=>{s.noise=v;draw();});
  function draw() {
    const f=80,left=40,right=40-f*s.baseline/s.z+s.noise,estimate=triangulate(f,s.baseline,left,right);
    canvas.width=600;canvas.height=330;
    const ctx=canvas.getContext('2d');ctx.fillStyle='#fbfbf6';ctx.fillRect(0,0,600,330);
    const X=x=>235+250*x,Y=z=>292-28*z;
    function line(x,z,x1,z1,color){ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=3;ctx.moveTo(X(x),Y(z));ctx.lineTo(X(x1),Y(z1));ctx.stroke();}
    function point(x,z,color,label){ctx.fillStyle=color;ctx.beginPath();ctx.arc(X(x),Y(z),6,0,2*Math.PI);ctx.fill();
      ctx.font='17px system-ui';ctx.fillText(label,X(x)+9,Y(z)-8);}
    line(0,0,0,10,'#486dc0');line(s.baseline,0,s.baseline+(right-40)*10/f,10,'#cc7750');
    point(0,0,'#486dc0','L');point(s.baseline,0,'#ac5425','R');
    point(0,s.z,'#40804e','Z');if(estimate!==null&&estimate<10) point(0,estimate,'#7e4596','Z′');
    stats(ui,[['uL, uR',fmt(left)+' / '+fmt(right)+' px'],[t('Диспаратность uL − uR','Disparity uL − uR'),fmt(left-right)+' px'],
      [t('Глубина по лучам','Depth from rays'),estimate===null?t('Нет конечного решения перед камерой','No finite solution in front'):fmt(estimate)+' m'],
      [t('Ошибка глубины','Depth error'),estimate===null?'—':fmt(estimate-s.z)+' m']]);
  }
  draw();return done(ui,s);
}

function inputFile(ui,label,onData,onError) {
  const wrap=el('label','control',label),file=el('input');file.type='file';file.accept='.json,application/json';
  file.setAttribute('aria-label',label);
  file.onchange=async()=>{const f=file.files[0];if(!f)return;try{
    if(f.size>12e6)throw Error('size');onData(JSON.parse(await f.text()));
  }catch{onError();}finally{file.value='';}};
  wrap.append(file);ui.controls.append(wrap);
}
function saveJson(scene) {
  const url=URL.createObjectURL(new Blob([JSON.stringify(scene)],{type:'application/json'}));
  const a=el('a');a.href=url;a.download='static-gaussians.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}

export function mountBuildScene(host,lang='ru',saved={}) {
  const ui=lab(host,lang,'Сцена из нескольких кадров','A scene from several views',
    'Серые пятна уже стоят в пространстве. Цвет меняется по обучающим кадрам. Перенос камеры даёт новый вид той же модели.',
    'Gray splats already occupy 3D space. Training images change their colors. Moving the camera reveals another view of the same model.');
  const t=ui.t,s={capture:saved.capture||syntheticCapture(),step:saved.step||0,view:saved.view??1,
    offset:saved.offset||0,kind:saved.kind||'small',scene:saved.scene};
  s.scene=s.scene||structuredClone(s.capture.initial);
  let system,active=true,serial=0;
  const target=panel(ui,t('Снятый кадр','Recorded frame')),model=panel(ui,t('Кадр модели','Model view'));
  const msg=el('p');msg.setAttribute('role','status');ui.controls.append(msg);
  const source=choose(ui,t('Набор','Dataset'),[['small',t('Цветные пятна','Colored splats')],
    ['photo',t('Фотографии скульптуры · 24 пятна','Sculpture photographs · 24 splats')],
    ['reference',t('Готовая скульптура · 117 пятен','Trained sculpture · 117 splats')],
    ['custom',t('Загруженный файл','Imported file')]],s.kind,async v=>{
    const request=++serial;
    if(v==='custom'){source.value=s.kind;msg.textContent=t('Поле JSON ниже принимает набор с камерами или сцену.','The JSON field below accepts a capture with cameras or a scene.');return;}
    if(v==='small'){replace(syntheticCapture(),'small');return;}
    msg.textContent=t('Загрузка снимков…','Loading photographs…');
    const captureFile=v==='reference'?'photographs-80.json':'photographs.json';
    try {const response=await fetch(new URL('../assets/reconstruction/'+captureFile,import.meta.url));
      if(!response.ok)throw Error();const data=decodeCapture(await response.json());
      if(v==='reference'){
        const ref=await fetch(new URL('../assets/reconstruction/photo-reference.json',import.meta.url));
        if(!ref.ok)throw Error();
        data.initial=validateScene(await ref.json());
      }
      if(active&&request===serial)replace(data,v);
    }catch{if(active&&request===serial){source.value=s.kind;msg.textContent=t('Не удалось прочитать набор фотографий.','Could not load the photographs.');}}
  });
  const view=choose(ui,t('Кадр и выборка','Frame and split'),[],s.view,v=>{s.view=+v;draw();});
  const setOffset=number(ui,t('Сдвиг камеры, доля расстояния','Camera shift, fraction of distance'),s.offset,-.2,.2,.005,v=>{s.offset=v;draw();});
  const one=addButton(ui.controls,t('Один шаг цвета','One color step'),()=>step(1));
  const many=addButton(ui.controls,t('12 шагов','12 steps'),()=>step(12),'primary');
  addButton(ui.controls,t('Исходная сцена','Initial scene'),()=>{s.scene=structuredClone(s.capture.initial);s.step=0;rebuild();draw();});
  addButton(ui.controls,t('Скачать сцену JSON','Download scene JSON'),()=>saveJson(s.scene));
  inputFile(ui,t('Локальный набор или сцена JSON','Local capture or scene JSON'),data=>{
    if(!active)return;
    if(data.format==='gs-course-capture-v1'){++serial;replace(decodeCapture(data),'custom');}
    else {
      const scene=validateScene(data);
      ++serial;s.scene=scene;s.step=0;rebuild();draw();
      msg.textContent=t('Сцена открыта с камерами выбранного набора.','Scene opened with the selected capture’s cameras.');
    }
  },()=>{msg.textContent=t('Файл не подходит по формату или размеру.','The file format or dimensions are unsupported.');});
  const credit=el('p');ui.footer.append(credit);
  function rebuild(){
    const work=s.capture.frames.reduce((n,f)=>n+f.width*f.height,0)*s.scene.splats.length**2;
    const viewOnly=s.kind==='reference'||work>24000000||s.scene.splats.some(p=>p.sh);
    system=viewOnly?null:trainingSystem(s.scene,s.capture.frames);
    one.disabled=many.disabled=viewOnly;
    credit.replaceChildren();
    if(s.kind==='photo'||s.kind==='reference'){
      const a=el('a','',t('Фотографии и изменения','Photographs and adaptations'));
      a.href='https://github.com/HeinrichWirth/gaussian-splatting-course/tree/main/data/photographic-example';
      a.target='_blank';a.rel='noopener';
      credit.append('NVIDIA Corporation · PhysicalAI-NuRec-PPISP / struktur28_auto · CC BY 4.0 · ',a);
    }
    if(viewOnly) credit.append(' · '+t('Просмотр готовой сцены','Trained-scene viewer'));
  }
  function views(){
    view.replaceChildren();
    s.capture.frames.forEach((f,i)=>{const o=el('option','',f.name+' · '+f.split);o.value=i;view.append(o);});
    view.value=s.view;
  }
  function replace(capture,kind){s.capture=capture;s.scene=structuredClone(capture.initial);
    s.kind=source.value=kind;s.step=0;s.view=capture.frames.findIndex(f=>f.split==='test');s.offset=0;
    setOffset(0);views();rebuild();msg.textContent='';draw();}
  function step(count){s.scene=fitColors(s.scene,system,count);s.step+=count;draw();}
  function draw(){
    const f=s.capture.frames[s.view],cam=structuredClone(f.camera),points=s.scene.splats;
    const mid=[0,1,2].map(c=>points.reduce((a,p)=>a+p.center[c],0)/points.length);
    const d=Math.hypot(...mid.map((x,i)=>x-cam.center[i]));
    for(let i=0;i<3;i++)cam.center[i]+=s.offset*d*cam.rotation[i][0];
    const img=renderScene(s.scene,cam,f.width,f.height);
    pixels(target,f.rgb,f.width,f.height);pixels(model,img,f.width,f.height);
    const progress=system?[t('Шаг / пятен','Step / splats'),s.step+' / '+points.length]
      :[t('Пятен в сцене','Splats in scene'),points.length];
    stats(ui,[progress,
      [t('Обучающих кадров','Training frames'),s.capture.frames.filter(f=>f.split==='train').length],
      ['MSE · '+f.split,s.offset===0?fmt(imageScores(img,f.rgb).mse,8):t('Новый ракурс без эталона','Novel pose without a reference')]]);
  }
  views();rebuild();draw();
  if(!saved.capture){source.value='photo';source.dispatchEvent(new Event('change'));}
  return done(ui,s,()=>{active=false;++serial;});
}

export function mountEvaluation(host,lang='ru',saved={}) {
  const ui=lab(host,lang,'Что скрывают средние числа','What an average hides',
    'Цвета подобраны по трём камерам. Две другие камеры сохранили только для проверки. Яркость карты равна средней абсолютной ошибке RGB.',
    'Three cameras supplied the training colors. Two others were reserved for evaluation. Map brightness is the mean absolute RGB error.');
  const t=ui.t,s={index:1,kind:'blur',strength:.45,...saved};
  const cap=syntheticCapture(),base=fitColors(cap.initial,trainingSystem(cap.initial,cap.frames),12);
  const a=panel(ui,t('Эталон','Reference')),b=panel(ui,t('Модель','Model')),e=panel(ui,t('Карта |ошибки|','Absolute error'));
  choose(ui,t('Камера','Camera'),cap.frames.map((f,i)=>[i,f.name+' · '+f.split]),s.index,v=>{s.index=+v;draw();});
  choose(ui,t('Изменение сцены','Scene edit'),[['blur',t('Раздуть пятна','Widen splats')],['floater',t('Лишнее пятно','Floater')],
    ['tint',t('Убрать зелёный','Reduce green')]],s.kind,v=>{s.kind=v;draw();});
  number(ui,t('Сила изменения','Edit strength'),s.strength,0,1,.01,v=>{s.strength=v;draw();});
  const speed=el('p');speed.setAttribute('role','status');ui.controls.append(speed);
  addButton(ui.controls,t('Замер времени кадра','Time this frame'),()=>{
    const scene=damagedScene(base,s.kind,s.strength),f=cap.frames[s.index];
    const render=()=>renderScene(scene,f.camera,f.width,f.height);render();
    const samples=[];
    for(let i=0;i<7;i++){const start=performance.now();for(let j=0;j<4;j++)render();samples.push((performance.now()-start)/4);}
    const ms=samples.sort((u,v)=>u-v)[3];
    speed.textContent=t('Медиана браузерного рендера ','Median browser render ')+fmt(ms,3)+' ms';
  });
  function draw(){
    const sc=damagedScene(base,s.kind,s.strength),f=cap.frames[s.index],rgb=renderScene(sc,f.camera,f.width,f.height);
    pixels(a,f.rgb,f.width,f.height);pixels(b,rgb,f.width,f.height);
    const err=new Float64Array(rgb.length);
    for(let i=0;i<rgb.length;i+=3){const d=([0,1,2].reduce((z,k)=>z+Math.abs(rgb[i+k]-f.rgb[i+k]),0))/3;err.fill(d,i,i+3);}
    pixels(e,err,f.width,f.height);const m=imageScores(rgb,f.rgb);
    stats(ui,[['MAE',fmt(m.mae,7)],['MSE',fmt(m.mse,8)],['PSNR',fmt(m.psnr,2)+' dB'],
      [t('Параметры float32, байт','Packed float32 parameter bytes'),sc.splats.length*14*4]]);
    speed.textContent='';
  }
  draw();return done(ui,s);
}
