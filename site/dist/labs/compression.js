import {makeLab,el,addSelect,addSlider,addButton} from './shared.js';
import {renderScene,orbitCamera} from '../splat-render.js';
import {decodeCapture,imageScores} from '../reconstruction-math.js';
import {exampleModel,sceneAt,validateModel} from '../deformation-math.js';
import {memoryBudget,packStatic,unpackStatic,packMotion,unpackMotion,sampleTrack,poseWithCenters,
        medianTime,chooseCandidate,defaultStudy,studySettings,photoStudy,candidates} from '../compression-math.js';

let photos;
function photoAssets() {
  if(!photos)photos=Promise.all(['photo-reference.json','photographs-80.json'].map(async name=>{
    const response=await fetch(new URL('../assets/reconstruction/'+name,import.meta.url));
    if(!response.ok)throw Error('Photographs unavailable');return response.json();
  })).then(([scene,capture])=>[scene,decodeCapture(capture).frames]).catch(error=>{photos=null;throw error;});
  return photos;
}
function lab(host,lang,title,description) {
  const t=(ru,en)=>lang==='en'?en:ru,ui=makeLab(host,{title:t(...title),subtitle:t(...description),lang});
  if(!document.querySelector('link[data-codec-css]')){const css=el('link');css.rel='stylesheet';
    css.href=new URL('./reconstruction.css',import.meta.url);css.dataset.codecCss='';document.head.append(css);}
  ui.root.classList.add('reconstruction-lab');ui.t=t;ui.images=el('div','recon-images');ui.view.append(ui.images);
  return ui;
}
function panel(ui,title){const fig=el('figure'),c=el('canvas');c.setAttribute('role','img');
  c.setAttribute('aria-label',title);fig.append(el('figcaption','',title),c);ui.images.append(fig);return c;}
function paint(canvas,rgb,w,h) {
  canvas.width=w;canvas.height=h;const ctx=canvas.getContext('2d'),img=ctx.createImageData(w,h);
  for(let i=0;i<w*h;i++){for(let d=0;d<3;d++)img.data[4*i+d]=Math.round(255*rgb[3*i+d]);img.data[4*i+3]=255;}
  ctx.putImageData(img,0,0);
}
function bars(canvas,rows) {
  const w=550,h=rows.length*56+16;canvas.width=w;canvas.height=h;
  const ctx=canvas.getContext('2d');ctx.fillStyle='#f6f6ed';ctx.fillRect(0,0,w,h);
  const max=Math.max(1,...rows.map(r=>r[1]));
  rows.forEach(([name,value],i)=>{const y=i*56+19;ctx.fillStyle='#293426';ctx.font='16px system-ui';
    ctx.fillText(name+' · '+value+' B',10,y);ctx.fillStyle=['#566cce','#b46945','#597e50'][i%3];
    ctx.fillRect(10,y+8,Math.max(1,(w-20)*value/max),14);});
}
function table(ui,rows,head=null) {
  ui.readout.replaceChildren();const tbl=el('table');
  if(head){const tr=el('tr');head.forEach(s=>tr.append(el('th','',s)));tbl.append(tr);}
  for(const row of rows){const tr=el('tr');row.forEach((s,i)=>tr.append(el(i?'td':'th','',String(s))));tbl.append(tr);}
  ui.readout.append(tbl);
}
function number(ui,label,s,key,min,max,step,onInput) {
  const range=addSlider(ui.controls,{label,min,max,step,value:s[key],
    onChange:v=>{s[key]=v;input.value=v;onInput();}});
  const input=el('input');input.type='number';Object.assign(input,{min,max,step,value:s[key]});
  input.setAttribute('aria-label',label);range.parentElement.append(input);
  input.oninput=()=>{const v=input.valueAsNumber;if(Number.isFinite(v)&&v>=min&&v<=max&&
    (step<1||Number.isInteger(v))){s[key]=v;range.setValue(v);onInput();}};
  input.onblur=()=>{input.value=s[key];};
  return ()=>{input.value=s[key];range.setValue(s[key]);};
}
function select(ui,label,s,key,options,draw) {
  const input=addSelect(ui.controls,{label,value:s[key],options:options.map(([value,name])=>({value,label:name})),
    onChange:v=>{s[key]=v;draw();}});return ()=>{input.value=s[key];};
}
const fmt=(v,n=3)=>Number(v).toFixed(n);
function mounted(ui,state,ready=null) {
  let active=true;
  if(ready){const message=el('p','',ui.t('Загружаются снимки…','Loading photographs…'));ui.footer.append(message);
    photoAssets().then(data=>{if(active){message.remove();ready(...data);}}).catch(()=>{
      if(active)message.textContent=ui.t('Не удалось открыть фотографии.','The photographs could not be loaded.');});}
  return {getState:()=>({...state}),destroy:()=>{active=false;ui.root.remove();}};
}
export function mountStorageBudget(host,lang='ru',saved={}) {
  const ui=lab(host,lang,['Байты на весах','Where the bytes go'],
    ['Число пятен и порядок цвета меняют длину столбцов. Буфер кадра зависит от его разрешения.',
     'The count and color degree change the bars. The frame buffer depends on its resolution.']);
  const t=ui.t,s={count:117,degree:0,width:80,...saved},graph=panel(ui,t('Модель и буферы','Model and buffers'));
  number(ui,t('Число пятен','Splat count'),s,'count',1,1000,1,draw);
  number(ui,t('Порядок цвета','Color degree'),s,'degree',0,3,1,draw);
  number(ui,t('Ширина кадра, пикс.','Frame width, px'),s,'width',4,320,1,draw);
  function draw(){const h=Math.max(1,Math.round(s.width*3/4)),b=memoryBudget(s.count,s.degree,s.width,h);
    bars(graph,[[t('Параметры','Weights'),b.weights],[t('Кадр float32','Float32 image'),b.rgb],
                [t('Параметры, градиент, два момента','Weights, gradient, two moments'),b.adam]]);
    table(ui,[[t('Центры / масштабы / повороты, Б','Centers / scales / rotations, B'),[b.fields.center,b.fields.scales,b.fields.rotation].join(' / ')],
      [t('Цвет / непрозрачность, Б','Color / opacity, B'),b.fields.color+' / '+b.fields.opacity],
      [t('Параметры + кадр, Б','Weights + frame, B'),b.viewer],[t('Кадр','Frame'),s.width+' × '+h]]);}
  draw();return mounted(ui,s);
}
export function mountStaticCodec(host,lang='ru',saved={}) {
  const ui=lab(host,lang,['Что осталось после упаковки','What survives the packet'],
    ['Слева та же модель до упаковки. Справа кадр появляется только после чтения настоящих байтов.',
     'The left image is the model before packing. The right image comes from decoding the actual bytes.']);
  const t=ui.t,s={keep:100,precision:'u8',color:'rgb',view:'4',...saved};
  const left=panel(ui,t('До','Before')),right=panel(ui,t('После','After'));let scene,frames;
  number(ui,t('Сохранено, %','Retained, %'),s,'keep',1,100,1,draw);
  select(ui,t('Числа','Numbers'),s,'precision',[['f32','float32'],['u8','uint8']],draw);
  select(ui,t('Цвет','Color'),s,'color',[['rgb','RGB'],['gray',t('Среднее RGB','RGB mean')]],draw);
  select(ui,t('Камера','Camera'),s,'view',[['0','view-00'],['4','view-04'],['8','view-08']],draw);
  function draw(){if(!scene)return;const f=frames[+s.view],w=f.width,h=f.height;
    const bytes=packStatic(scene,s.precision,s.keep,s.color),model=unpackStatic(bytes);
    const src=renderScene(scene,f.camera,w,h),render=()=>renderScene(model,f.camera,w,h),dst=render();
    paint(left,src,w,h);paint(right,dst,w,h);table(ui,[[t('Пятна','Splats'),scene.splats.length+' → '+model.splats.length],
      [t('Байты пакета','Packet bytes'),bytes.length],[t('MSE до упаковки → после (между моделями)','MSE between model images'),fmt(imageScores(src,dst).mse,8)],
      [t('Медиана кадра до / после, мс','Median frame before / after, ms'),
       fmt(medianTime(()=>renderScene(scene,f.camera,w,h)))+' / '+fmt(medianTime(render))]]);}
  return mounted(ui,s,(a,b)=>{scene=a;frames=b;draw();});
}
export function mountMotionCodec(host,lang='ru',saved={}) {
  const ui=lab(host,lang,['Движение между опорами','Motion between anchors'],
    ['Две копии сцены движутся по одному времени. В правой копии центры прошли через двоичный пакет.',
     'Both copies share a clock. The right copy takes its centers from a binary packet.']);
  const t=ui.t,s={mode:'keys',stride:8,quantum:.002,time:.37,angle:32,...saved};
  const left=panel(ui,t('Исходное движение','Original motion')),right=panel(ui,t('Декодированное','Decoded motion'));
  const tracks=m=>Array.from({length:17},(_,i)=>sceneAt(m,i*m.duration/16).splats.map(p=>p.center));
  let model=s.imported||exampleModel(),xyz=tracks(model);
  number(ui,t('Доля длительности','Fraction of clip'),s,'time',0,1,.01,draw);
  number(ui,t('Ракурс, °','View, °'),s,'angle',-45,45,1,draw);
  select(ui,t('Способ','Method'),s,'mode',[['dense',t('Все кадры','Every frame')],['keys',t('Опоры','Anchors')],['residual',t('Опоры + поправки','Anchors + residuals')]],draw);
  number(ui,t('Шаг опор, кадров','Anchor stride, frames'),s,'stride',1,16,1,draw);
  number(ui,t('Цена кода, м','Residual unit, m'),s,'quantum',.001,.03,.001,draw);
  const file=el('input');file.type='file';file.accept='.json,application/json';
  const field=el('label','control',t('Движение из урока 26 · JSON','Motion from lesson 26 · JSON'));
  field.append(file);ui.controls.append(field);
  const status=el('p');status.setAttribute('role','status');ui.footer.append(status);
  const reset=addButton(ui.controls,t('Исходная лента','Original ribbon'),()=>useModel(null));
  reset.hidden=!s.imported;
  function useModel(imported){
    const next=imported||exampleModel();
    const samples=tracks(next);
    model=next;xyz=samples;s.imported=imported;reset.hidden=!imported;
    status.textContent=imported?t('Сцена из файла','Scene from file'):'';
    draw();
  }
  file.onchange=async()=>{
    const source=file.files[0];if(!source)return;
    try{
      if(source.size>1000000)throw Error();
      useModel(validateModel(JSON.parse(await source.text())));
    }catch{status.textContent=t('Файл не содержит подходящей динамической сцены.','The file does not contain a supported motion clip.');}
    finally{file.value='';}
  };
  function draw(){const bytes=packMotion(xyz,s.mode,s.stride,s.quantum),track=unpackMotion(bytes),centers=sampleTrack(track,s.time);
    const seconds=s.time*model.duration;
    const cam=orbitCamera(s.angle,55),ref=sceneAt(model,seconds),dst=poseWithCenters(model.canonical,centers),w=65,h=49;
    const a=renderScene(ref,cam,w,h),b=renderScene(dst,cam,w,h);paint(left,a,w,h);paint(right,b,w,h);
    const max=Math.max(...centers.map((p,i)=>Math.hypot(...p.map((x,d)=>x-ref.splats[i].center[d]))));
    table(ui,[[t('Время, с','Time, s'),fmt(seconds,3)],[t('Байты движения','Motion bytes'),bytes.length],[t('Наибольшая ошибка центра, м','Largest center error, m'),fmt(max,5)],
      ['MSE RGB',fmt(imageScores(a,b).mse,8)],[t('Декодирование клипа, мс','Clip decode, ms'),fmt(medianTime(()=>unpackMotion(bytes)),4)]]);}
  draw();return mounted(ui,s);
}
function download(data,name) {
  const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));
  const a=el('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
export function mountCodecStudy(host,lang='ru',saved={}) {
  const ui=lab(host,lang,['Какая сцена помещается','Which scene fits'],
    ['Четыре пакета, одна камера для выбора и отдельная фотография для финальной проверки.',
     'Four packets, one selection view and a separate photograph for the final check.']);
  const t=ui.t,s={...defaultStudy,...saved},left=panel(ui,t('Отложенное фото','Held-out photo')),right=panel(ui,t('Выбранная модель','Selected model'));
  let scene,frames,report;const sync=[];
  sync.push(number(ui,t('Лимит, байт','Byte limit'),s,'maxBytes',1,100000,1,draw));
  sync.push(number(ui,t('Порог PSNR, дБ','PSNR floor, dB'),s,'minPsnr',0,60,.1,draw));
  sync.push(select(ui,t('Приоритет','Priority'),s,'priority',[['quality',t('Качество','Quality')],['size',t('Размер','Size')],['latency',t('Время кадра','Frame time')]],draw));
  sync.push(number(ui,t('Повторы замера','Timing repetitions'),s,'repeats',1,15,1,measure));
  addButton(ui.controls,t('Измерить ещё раз','Measure again'),measure);
  addButton(ui.controls,t('Скачать настройки','Download settings'),()=>download(studySettings(s),'codec-settings.json'));
  const wrap=el('label','control',t('Настройки из JSON','Settings from JSON')),file=el('input');file.type='file';file.accept='.json,application/json';wrap.append(file);ui.controls.append(wrap);
  const status=el('p');status.setAttribute('role','status');ui.footer.append(status);
  file.onchange=async()=>{try{if(!file.files[0]||file.files[0].size>8192)throw Error();
    Object.assign(s,studySettings(JSON.parse(await file.files[0].text())));sync.forEach(f=>f());measure();
  }catch{status.textContent=t('Формат настроек не совпадает.','Unsupported settings.');}finally{file.value='';}};
  function measure(){if(scene){report=photoStudy(scene,frames,s);draw();}}
  function draw(){if(!report)return;const rows=report.candidates,winner=chooseCandidate(rows,s),f=frames.find(x=>x.split==='test');
    paint(left,f.rgb,80,60);const figure=right.parentElement;figure.hidden=!winner;
    if(winner){const [,precision,keep,color]=candidates.find(c=>c[0]===winner.name);
      const model=unpackStatic(packStatic(scene,precision,keep,color)),rgb=renderScene(model,f.camera,80,60);
      paint(right,rgb,80,60);status.textContent=t('Выбор','Selected')+' · '+winner.name+' · PSNR (view-04) '+fmt(imageScores(rgb,f.rgb).psnr,2)+' dB';}
    else status.textContent=t('Ни один пакет не проходит оба порога.','No packet clears both thresholds.');
    table(ui,rows.map(r=>[r.name,r.bytes,fmt(r.validationPsnr,2),fmt(r.decodeMs),fmt(r.renderMs)]),
      [t('Пакет','Packet'),'B','PSNR (00)',t('Чтение, мс','Decode, ms'),t('Кадр, мс','Frame, ms')]);}
  return mounted(ui,s,(a,b)=>{scene=a;frames=b;measure();});
}
