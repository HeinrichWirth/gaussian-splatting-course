import {el,addButton,addSelect} from './shared.js';
import {lab,number,panel,pixels,stats,done} from './reconstruction.js';
import {orbitCamera,renderScene,mse} from '../splat-render.js';
import {ribbon,fieldValue,fieldScene,pulse,referenceScene,exampleModel,sceneAt,validateModel} from '../deformation-math.js';

const width=65,height=49,focal=64;
const rounded=(v,places=4)=>Number(v.toFixed(places));
const rgb=(scene,angle)=>renderScene(scene,orbitCamera(angle,focal),width,height);

function gridPicture(canvas,x,time,corner){
  canvas.width=200;canvas.height=152;
  const ctx=canvas.getContext('2d');
  const l=28,top=12,w=157,h=112;
  for(let i=0;i<32;i++)for(let j=0;j<24;j++){
    const f=fieldValue((i+.5)/32,1-(j+.5)/24,corner);
    ctx.fillStyle=f>=0?'rgb('+[245-132*f,247-86*f,238-51*f].map(Math.round)+')'
      :'rgb('+[245+19*f,247+99*f,238+113*f].map(Math.round)+')';
    ctx.fillRect(l+i*w/32,top+j*h/24,Math.ceil(w/32),Math.ceil(h/24));
  }
  ctx.strokeStyle='#41584d';ctx.strokeRect(l,top,w,h);
  ctx.strokeStyle='#8f3440';ctx.lineWidth=1.5;ctx.beginPath();
  const px=l+x*w,py=top+(1-time)*h;
  ctx.moveTo(l,py);ctx.lineTo(px,py);ctx.lineTo(px,top+h);ctx.stroke();
  ctx.fillStyle='#8f3440';ctx.beginPath();ctx.arc(px,py,4,0,2*Math.PI);ctx.fill();
  ctx.fillStyle='#33423a';ctx.font='12px system-ui';
  ctx.fillText('0',l-7,top+h+14);ctx.fillText('1',l+w-5,top+h+14);
  ctx.fillText('1',l-13,top+5);ctx.fillText('t',7,top+h/2);ctx.fillText('u',l+w/2,top+h+17);
}
export function mountDeformField(host,lang='ru',saved={}){
  const ui=lab(host,lang,'Как сгибается лента','How the ribbon bends',
    'Точка на зелёной карте выбирает смещение одного пятна. Обе картинки показывают ленту из одной камеры.',
    'The dot on the field samples one splat’s displacement. Both ribbon views share a camera.');
  const t=ui.t,s={time:.5,angle:0,corner:1,index:4,...saved};
  const base=panel(ui,t('Исходная лента','Rest ribbon')),moved=panel(ui,t('В этот момент','At this time'));
  const grid=panel(ui,t('Поле по положению и времени','Position–time field'));
  number(ui,t('Время, с','Time, s'),s.time,0,1,.025,v=>{s.time=v;draw();});
  number(ui,t('Угол камеры, °','Camera angle, °'),s.angle,-65,65,1,v=>{s.angle=v;draw();});
  number(ui,t('Правый верхний узел','Upper-right node'),s.corner,-1,1,.05,v=>{s.corner=v;draw();});
  addSelect(ui.controls,{label:t('Пятно на карте','Splat on the field'),value:String(s.index),
    options:ribbon().splats.map((p,i)=>({value:String(i),label:(i+1)+' · x = '+rounded(p.center[0],1)+' m'})),
    onChange:v=>{s.index=+v;draw();}});
  function draw(){
    const scene=ribbon(),warped=fieldScene(s.time,s.corner);
    pixels(base,rgb(scene,s.angle),width,height);pixels(moved,rgb(warped,s.angle),width,height);
    const x=(scene.splats[s.index].center[0]+1)/2,f=fieldValue(x,s.time,s.corner);
    gridPicture(grid,x,s.time,s.corner);
    stats(ui,[['u / t',rounded(x)+' / '+rounded(s.time)],['F',rounded(f)],
      ['Δy',rounded(-.8*f)+' m'],[t('Поворот','Rotation'),rounded(-35*f)+'°'],
      ['σx',rounded(warped.splats[s.index].scales[0])+' m']]);
  }
  draw();return done(ui,s);
}

export function mountDynamicFit(host,lang='ru',saved={}){
  const ui=lab(host,lang,'Лента между двумя кадрами','The ribbon between two frames',
    'Время и камера независимы. Слева — синтетический кадр, справа — модель по двум парам меток.',
    'Time and camera move independently. A synthetic reference faces a model fitted to two sets of stereo landmarks.');
  const t=ui.t,s={time:.375,angle:32,noise:0,kind:'fitted',imported:null,...saved};
  const first=panel(ui,t('Эталон','Reference')),second=panel(ui,t('Модель','Model'));
  const caption=first.parentElement.querySelector('figcaption');
  const setTime=number(ui,t('Доля длительности','Fraction of clip'),s.time,0,1,.005,v=>{s.time=v;draw();});
  const setAngle=number(ui,t('Угол камеры, °','Camera angle, °'),s.angle,-65,65,1,v=>{s.angle=v;draw();});
  number(ui,t('Ошибка правой метки, px','Right landmark error, px'),s.noise,0,1,.05,v=>{s.noise=v;draw();});
  const modes=[['fitted',t('Движение по меткам','Fitted motion')],['static',t('Неподвижная лента','Static ribbon')]];
  if(s.imported)modes.push(['imported',t('Сцена из файла','Imported clip')]);
  const mode=addSelect(ui.controls,{label:t('Модель','Model'),value:s.kind,
    options:modes.map(([value,label])=>({value,label})),onChange:v=>{s.kind=v;draw();}});
  addButton(ui.controls,t('Отложенное время','Held-out time'),()=>preset(.375,0));
  addButton(ui.controls,t('Отложенная камера','Held-out camera'),()=>preset(.25,32));
  addButton(ui.controls,t('Обе проверки','Both held out'),()=>preset(.375,32));
  function preset(time,angle){
    s.time=time;s.angle=angle;setTime(time);setAngle(angle);draw();
  }
  function model(){
    if(s.kind==='imported')return s.imported;
    const fitted=exampleModel(s.noise);
    if(s.kind==='static')fitted.amplitudes=fitted.amplitudes.map(()=>[0,0,0]);
    return fitted;
  }
  addButton(ui.controls,t('Скачать движение JSON','Download motion JSON'),()=>{
    const url=URL.createObjectURL(new Blob([JSON.stringify(model())],{type:'application/json'}));
    const link=el('a');link.href=url;link.download='dynamic-ribbon.json';link.click();
    setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
  const file=el('input');file.type='file';file.accept='.json,application/json';
  const wrap=el('label','control',t('Движение из JSON','Motion from JSON'));wrap.append(file);
  const msg=el('span');msg.setAttribute('role','status');ui.controls.append(wrap,msg);
  file.onchange=async()=>{
    const source=file.files[0];if(!source)return;
    try{
      if(source.size>1000000)throw Error('Too large');
      const candidate=validateModel(JSON.parse(await source.text()));
      if(!mode.querySelector('option[value="imported"]')){
        const opt=el('option','',t('Сцена из файла','Imported clip'));opt.value='imported';mode.append(opt);
      }
      s.imported=candidate;s.kind=mode.value='imported';msg.textContent='';draw();
    }catch{msg.textContent=t('Формат или параметры движения не подходят.','Unsupported motion format or parameters.');}
    file.value='';
  };
  function draw(){
    const m=model(),seconds=s.time*m.duration,scene=sceneAt(m,seconds),imported=s.kind==='imported';
    const target=rgb(imported?m.canonical:referenceScene(s.time),s.angle),output=rgb(scene,s.angle);
    caption.textContent=imported?t('Исходная сцена файла','Canonical scene from file'):t('Эталон','Reference');
    first.setAttribute('aria-label',caption.textContent);
    pixels(first,target,width,height);pixels(second,output,width,height);
    const end=m.amplitudes.at(-1),delta=end.map(v=>rounded(v*pulse(s.time)));
    const rows=[['t',rounded(seconds)+' s'],['h',rounded(pulse(s.time))],
      [t('Сдвиг последнего центра, м','Last center offset, m'),delta.join(' / ')]];
    if(!imported)rows.push([t('MSE кадра','Frame MSE'),mse(target,output).toFixed(8)]);
    stats(ui,rows);
  }
  draw();return done(ui,s);
}
