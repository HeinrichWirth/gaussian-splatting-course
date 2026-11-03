import {makeLab, el, addToggle, addButton} from './shared.js';
import {F, WIDTH, HEIGHT, observingPair, observationImage, trueX, fitPath, trajectorySplat,
  horizontalPixel, persistentPair} from '../motion-math.js';
import {makeScene, renderScene, translatedCamera} from '../splat-render.js';

const green='#427c5d', orange='#be6135', blue='#365ab2', ink='#35413c';
const fmt=(v,n=3)=>Number(v.toFixed(n)).toString();
const point=p=>'('+p.map(x=>fmt(x)).join(', ')+')';

function panel(parent,title,w=320,h=210) {
  const figure=el('figure');figure.style.cssText='margin:0;min-width:0';
  const label=el('figcaption','',title);label.style.cssText='font-size:.83rem;color:#53624d;margin-bottom:.4rem';
  const canvas=el('canvas');canvas.width=w;canvas.height=h;canvas.setAttribute('aria-label',title);
  canvas.style.cssText='display:block;width:100%;height:auto;border:1px solid #d8dfd3;background:white';
  figure.append(label,canvas);parent.append(figure);return canvas.getContext('2d');
}
function photo(ctx,rgb) {
  const image=ctx.createImageData(WIDTH,HEIGHT);
  for(let i=0;i<WIDTH*HEIGHT;i++){
    for(let c=0;c<3;c++)image.data[4*i+c]=Math.round(255*rgb[3*i+c]);
    image.data[4*i+3]=255;
  }
  ctx.putImageData(image,0,0);
}
function line(ctx,points,color,width=2,dash=[]) {
  ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);
  points.forEach(([x,y],i)=>{if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y);});ctx.stroke();ctx.setLineDash([]);
}
function mark(ctx,[x,y],color,r=4) {
  ctx.beginPath();ctx.fillStyle=color;ctx.arc(x,y,r,0,2*Math.PI);ctx.fill();
}
function text(ctx,x,y,label,color=ink) {
  ctx.fillStyle=color;ctx.font='12px system-ui';ctx.fillText(label,x,y);
}
function controls(ui,state,entries,redraw,t) {
  const setters={};
  for(const [key,ru,en,min,max,step,start] of entries){
    const label=t(ru,en);
    if(typeof start==='boolean'){
      state[key]=typeof state[key]==='boolean'?state[key]:start;
      const check=addToggle(ui.controls,label,state[key],v=>{state[key]=v;redraw();});
      setters[key]=value=>{state[key]=value;check.checked=value;};
      continue;
    }
    const row=el('div','control'),header=el('div','control-top'),labelEl=el('span','',label);
    const box=el('input'),slider=el('input');box.type='number';slider.type='range';
    box.style.cssText='width:4.8rem;max-width:42%';
    for(const input of [box,slider]){Object.assign(input,{min,max,step});input.setAttribute('aria-label',label);}
    const set=v=>{state[key]=v;box.value=v;slider.value=v;};setters[key]=set;
    set(Number.isFinite(state[key])&&state[key]>=min&&state[key]<=max?state[key]:start);
    slider.oninput=()=>{set(slider.valueAsNumber);redraw();};
    box.oninput=()=>{const v=box.valueAsNumber;if(Number.isFinite(v)&&v>=min&&v<=max){state[key]=v;slider.value=v;redraw();}};
    box.onchange=box.onblur=()=>{const v=box.valueAsNumber;set(Number.isFinite(v)?Math.max(min,Math.min(max,v)):state[key]);redraw();};
    header.append(labelEl,box);row.append(header,slider);ui.controls.append(row);
  }
  return setters;
}
function lab(host,lang,saved,title,note,entries,draw) {
  const t=(ru,en)=>lang==='en'?en:ru;
  const ui=makeLab(host,{lang,title:t(...title),subtitle:t(...note)}),state={...saved};
  ui.view.style.cssText='display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,185px),1fr));gap:12px;align-items:center';
  const update=()=>{
    ui.view.replaceChildren();const rows=draw(ui.view,state,t);
    const table=el('table');table.style.cssText='width:100%;font-size:.88rem;text-align:left';
    for(const [label,value] of rows){const row=el('tr'),th=el('th','',label),td=el('td','',String(value));
      th.scope='row';th.style.fontWeight='500';td.style.overflowWrap='anywhere';row.append(th,td);table.append(row);}
    ui.readout.replaceChildren(table);
  };
  const setters=controls(ui,state,entries,update,t);
  let animation=0;
  if(setters.time){
    const play=t('Движение','Play motion'),pause=t('Пауза','Pause');
    const duration=entries.find(e=>e[0]==='time')[4];
    const button=addButton(ui.controls,play,()=>{
      if(animation){cancelAnimationFrame(animation);animation=0;button.textContent=play;return;}
      button.textContent=pause;
      let last=performance.now();
      function tick(now){
        const elapsed=(now-last)/1000;
        if(elapsed>=.05){last=now;setters.time(Math.round(((state.time+elapsed)%duration)*100)/100);update();}
        animation=requestAnimationFrame(tick);
      }
      animation=requestAnimationFrame(tick);
    });
  }
  update();return {getState:()=>({...state}),destroy:()=>{cancelAnimationFrame(animation);ui.root.remove();}};
}

export function mountMotionViews(host,lang='ru',saved={}) {
  return lab(host,lang,saved,['Два кадра движущейся точки','Two exposures of a moving point'],
    ['Оранжевое пятно движется, синяя метка закреплена в мире. Между камерами 0.8 м.','Orange moves through the scene. The blue mark stays fixed in the world. The cameras are 0.8 m apart.'],[
      ['time','Время A, с','A time, s',0,2,.05,.8],['objectSpeed','Скорость пятна, м/с','Splat speed, m/s',-1,1,.1,1],
      ['cameraSpeed','Скорость камер, м/с','Rig speed, m/s',-1,1,.1,0],['lag','Задержка B, с','B delay, s',0,.8,.05,0],
      ['gate','Ширма','Occluder',0,1,1,false]],(view,s,t)=>{
    const {views,disparity,estimate}=observingPair(s.time,s.objectSpeed,s.cameraSpeed,s.lag,s.gate);
    views.forEach((v,i)=>{const ctx=panel(view,(i?'B':'A')+' · '+fmt(v.time)+' s',WIDTH,HEIGHT);
      photo(ctx,observationImage(v,s.gate));});
    const marks=views.map(v=>v.visible?fmt(v.u):'—');
    return [[t('Пятно в A / B, пикс.','Splat in A / B, px'),marks.join(' / ')],
      [t('Синяя метка в A / B, пикс.','Blue mark in A / B, px'),views.map(v=>fmt(v.anchor)).join(' / ')],
      [t('Разность отметок, пикс.','Mark disparity, px'),views.every(v=>v.visible)?fmt(disparity):'—'],
      [t('Глубина по формуле синхронной пары','Depth under the synchronized-pair formula'),
        estimate===null?t('Нет пары для конечной глубины','No pair giving finite depth'):fmt(estimate)+' m']];
  });
}

export function mountTrajectory(host,lang='ru',saved={}) {
  return lab(host,lang,saved,['Траектория между замерами','A trajectory between observations'],
    ['Зелёная кривая — исходное движение, синяя — восстановленное. Справа штрих отмечает истинный центр.','Green traces the reference motion, blue the fit. The vertical mark in the image follows the reference center.'],[
      ['time','Время, с','Time, s',0,3,.05,1],['bend','Коэффициент изгиба, м/с²','Quadratic term, m/s²',-.25,.25,.01,.2],
      ['middle','Третий замер','Third observation',0,1,1,false],['noise','Сдвиг замера, м','Middle error, m',-.2,.2,.01,0]],(view,s,t)=>{
    const chart=panel(view,t('Положение, м · время, с','Position, m · time, s')),ctx=panel(view,t('Пятно модели','Fitted splat'),WIDTH,HEIGHT);
    const samples=Array.from({length:61},(_,i)=>i/20),trueAt=k=>trueX(k,s.bend),fitAt=k=>fitPath(k,s.middle,s.noise,s.bend);
    const values=samples.flatMap(k=>[trueAt(k),fitAt(k)]),low=Math.min(-.6,...values)-.15,high=Math.max(1,...values)+.15;
    const pos=(k,x)=>[37+88*k,175-151*(x-low)/(high-low)];
    chart.fillStyle='#f3eddf';chart.fillRect(213,15,88,164);
    line(chart,[[37,20],[37,179],[302,179]],'#aab3a6',1);
    for(const k of [0,1,2,3])text(chart,34+k*88,198,k+' s');
    text(chart,2,24,fmt(high,1));text(chart,2,178,fmt(low,1));
    line(chart,samples.map(k=>pos(k,trueAt(k))),green);
    line(chart,samples.map(k=>pos(k,fitAt(k))),blue,2,[5,3]);
    for(const k of s.middle?[0,1,2]:[0,2])mark(chart,pos(k,trueAt(k)+(k===1?s.noise:0)),ink);
    mark(chart,pos(s.time,fitAt(s.time)),orange);
    const x=fitAt(s.time);
    photo(ctx,renderScene(makeScene(trajectorySplat(s.time,x)),translatedCamera(0,F),WIDTH,HEIGHT));
    const u=horizontalPixel(trueAt(s.time),0);line(ctx,[[u,0],[u,HEIGHT]],green,1,[2,2]);
    return [[t('Центр модели / образца, м','Fitted / reference center, m'),fmt(x)+' / '+fmt(trueAt(s.time))],
      [t('Расхождение по горизонтали, пикс.','Horizontal separation, px'),fmt(Math.abs(x-trueAt(s.time))*F/4)],
      [t('Длинная полуось / поворот','Long scale / rotation'),fmt(.24+.08*s.time)+' m / '+fmt(30*s.time)+'°']];
  });
}
export function mountPersistent(host,lang='ru',saved={}) {
  return lab(host,lang,saved,['Один поворот для двух соседей','One rotation, two neighbours'],
    ['Синим пунктиром показан поворот исходного отрезка, зелёным — известное положение B, оранжевым — оценка.','A dashed blue line rotates the original edge. Green marks the known position of B and orange the estimate.'],[
      ['angle','Поворот A, °','Rotation of A, °',0,150,5,90],['strength','Вес соседства','Neighbour weight',0,10,.25,4],
      ['noise','Ошибка отметки по x, м','Mark error in x, m',-.5,.5,.025,.4],['bend','Сгиб, °','Bend, °',-60,60,5,0]],(view,s,t)=>{
    const p=persistentPair(s.angle,s.noise,s.strength,s.bend);
    for(const [key,title] of [['observed',t('Одна отметка','Measurement alone')],['fitted',t('Отметка и соседство','Measurement and neighbour')]]){
      const ctx=panel(view,title,270,216),origin=[132,106],pos=q=>[origin[0]+85*q[0],origin[1]+85*q[1]];
      line(ctx,[[9,106],[261,106]],'#c9d1c4',1);line(ctx,[[132,8],[132,206]],'#c9d1c4',1);
      for(const [loc,color] of [['rigid',blue],['actual',green],[key,orange]]){
        const end=pos(p[loc]);line(ctx,[origin,end],color,loc==='actual'?4:2,loc==='rigid'?[4,4]:[]);
        mark(ctx,end,color,loc==='actual'?7:3);
      }
      mark(ctx,origin,ink);text(ctx,origin[0]-14,origin[1]-7,'A');text(ctx,244,101,'x');text(ctx,136,207,'y');
    }
    return [[t('Отрезок A → B после поворота, м','Rotated edge A → B, m'),point(p.rigid)],
      [t('Отметка B → оценка B, м','Measured B → fitted B, m'),point(p.observed)+' → '+point(p.fitted)],
      [t('До жёсткого прогноза, м','Distance to rigid prediction, m'),fmt(p.rigidityError)],
      [t('До истины, м','Distance to ground truth, m'),fmt(p.trackingError)]];
  });
}
