import {el, makeLab, addButton, addSelect} from './shared.js';
import {orbitCamera, overlapScene, renderScene, projectSplats, footprintAlpha,
  shColor, shBasis, splat, makeScene, translatedCamera, rayScene, rayLoss,
  depthStep, densityScene, densityLoss, densityEdit, densityStep, densityPass, densityGradients} from '../splat-render.js';

const text=(lang)=>(ru,en)=>lang==='ru'?ru:en;
const show=(v,d=4)=>Number(v.toFixed(d)).toString();
const triple=v=>'('+v.map(k=>show(k,3)).join(', ')+')';
function setup(host,lang,title,subtitle,labels){
  if(!document.querySelector('link[data-learning-style]')){
    const link=document.createElement('link');link.rel='stylesheet';
    link.href=new URL('./learning.css',import.meta.url);link.dataset.learningStyle='';document.head.append(link);
  }
  const ui=makeLab(host,{title,subtitle,lang});ui.root.classList.add('learning-lab');
  const panels=el('div','learn-images');panels.style.gridTemplateColumns='repeat('+labels.length+',minmax(0,1fr))';
  ui.view.append(panels);ui.images=labels.map(label=>{
    const fig=el('figure'),canvas=el('canvas','learn-pixels');
    canvas.setAttribute('role','img');canvas.setAttribute('aria-label',label);
    fig.append(el('figcaption','',label),canvas);panels.append(fig);return canvas;
  });
  return ui;
}
function paint(canvas,rgb,size){
  canvas.width=canvas.height=size;
  const ctx=canvas.getContext('2d'),buffer=ctx.createImageData(size,size);
  for(let p=0;p<size*size;p++){
    for(let c=0;c<3;c++)buffer.data[4*p+c]=Math.round(255*rgb[3*p+c]);
    buffer.data[4*p+3]=255;
  }
  ctx.putImageData(buffer,0,0);
}
function numberInput(parent,label,min,max,step,value,change){
  const wrap=el('div','learn-control'),line=el('label','learn-number-row',label);
  const n=el('input'),s=el('input');n.type='number';s.type='range';
  for(const node of [n,s]){
    Object.assign(node,{min,max,step,value});node.setAttribute('aria-label',label);
  }
  const set=v=>{n.value=v;s.value=v;};
  s.oninput=()=>{n.value=s.value;change(+s.value);};
  n.oninput=()=>{const v=n.valueAsNumber;if(Number.isFinite(v)&&v>=min&&v<=max){s.value=v;change(v);}};
  n.onchange=n.onblur=()=>{const v=n.valueAsNumber;set(Number.isFinite(v)?Math.min(max,Math.max(min,v)):+s.value);change(+n.value);};
  line.append(n);wrap.append(line,s);parent.append(wrap);
  return set;
}
function readout(ui,rows){
  const box=el('div','learn-table-wrap'),table=el('table','learn-table'),body=el('tbody');
  for(const [label,value] of rows){
    const tr=el('tr');tr.append(el('th','',label),el('td','',String(value)));body.append(tr);
  }
  table.append(body);box.append(table);ui.readout.replaceChildren(box);
}
const instance=(ui,state)=>({getState:()=>structuredClone(state),destroy:()=>ui.root.remove()});
const stateFrom=(defaults,saved)=>({...defaults,...structuredClone(saved)});

export function mountVisibility(host,lang='ru',saved={}){
  const t=text(lang),ui=setup(host,lang,t('Кто перед кем','Which splat comes first'),
    t('Два кадра одной камеры. Изменён только порядок смешивания.','One camera, two images. Only the compositing order changes.'),
    [t('По глубине','Depth order'),t('Обратный порядок','Reversed order')]);
  const state=stateFrom({angle:0,depth:3.5,pixel:32,isolate:'both',background:'white'},saved);
  function draw(){
    const camera=orbitCamera(state.angle),scene=overlapScene(state.depth),size=65;
    if(state.isolate!=='both')scene.splats=scene.splats.filter((_,i)=>i===Number(state.isolate));
    const bg=state.background==='white'?[1,1,1]:[0,0,0];
    const image=renderScene(scene,camera,size,size,bg), wrong=renderScene(scene,camera,size,size,bg,true);
    paint(ui.images[0],image,size);paint(ui.images[1],wrong,size);
    const spots=projectSplats(scene,camera,size,size),u=Math.round(state.pixel),offset=3*(32*size+u);
    for(const canvas of ui.images){
      const context=canvas.getContext('2d');
      context.strokeStyle='#202b25';context.lineWidth=.6;
      context.strokeRect(u-1.5,30.5,4,4);
    }
    const alpha=spots.map(p=>footprintAlpha([u,32],p));
    let remaining=1;
    const shares=alpha.map(a=>{const share=remaining*a;remaining*=1-a;return share;});
    const order=spots.map(p=>p.color[0]===1?t('красное','red'):t('синее','blue')).join(' → ');
    readout(ui,[[t('Пиксель (u, v)','Pixel (u, v)'),[u,32].join(', ')],[t('Ближнее → дальнее','Near → far'),order],
      [t('Глубины в этом порядке','Depths in this order'),spots.map(s=>show(s.depth)).join(' / ')],
      ['α',alpha.map(x=>show(x)).join(', ')],
      [t('Доли слоёв в этом порядке','Layer shares in this order'),shares.map(x=>show(x)).join(' / ')],
      [t('Доля фона','Background share'),show(remaining)],['RGB',triple(Array.from(image.slice(offset,offset+3)))],
      [t('RGB при перестановке','RGB after swapping'),triple(Array.from(wrong.slice(offset,offset+3)))]]);
  }
  numberInput(ui.controls,t('Угол камеры, °','Camera angle, °'),-140,140,1,state.angle,v=>{state.angle=v;draw();});
  numberInput(ui.controls,t('Глубина красного в мире','World Z of red splat'),2,6,.05,state.depth,v=>{state.depth=v;draw();});
  numberInput(ui.controls,t('Пиксель u при v = 32','Pixel u at v = 32'),0,64,1,state.pixel,v=>{state.pixel=Math.round(v);draw();});
  addSelect(ui.controls,{label:t('Видимые пятна','Visible splats'),options:[{value:'both',label:t('Оба','Both')},
    {value:'0',label:t('Красное','Red')},{value:'1',label:t('Синее','Blue')}],value:state.isolate,onChange:v=>{state.isolate=v;draw();}});
  addSelect(ui.controls,{label:t('Фон','Background'),options:[{value:'white',label:t('Белый','White')},
    {value:'black',label:t('Чёрный','Black')}],value:state.background,onChange:v=>{state.background=v;draw();}});
  draw();return instance(ui,state);
}

export function mountHarmonics(host,lang='ru',saved={}){
  const t=text(lang),ui=setup(host,lang,t('Цвет с двух сторон','Color from two sides'),
    t('Одно сферическое пятно остаётся в центре кадра при повороте камеры.','One spherical splat stays at the center as the camera orbits.'),
    [t('Только постоянный цвет','Constant color only'),t('Цвет с направлением','Directional color')]);
  const state=stateFrom({angle:-36.87,cx:.6,cz:.4,cy:0,elevation:0},saved);
  function draw(){
    const a=state.angle*Math.PI/180,b=state.elevation*Math.PI/180;
    const direction=[Math.sin(a)*Math.cos(b),Math.sin(b),Math.cos(a)*Math.cos(b)];
    const right=[Math.cos(a),0,-Math.sin(a)],down=[-Math.sin(a)*Math.sin(b),Math.cos(b),-Math.cos(a)*Math.sin(b)];
    const camera={center:direction.map((v,i)=>[0,0,4][i]-4*v),rotation:right.map((v,i)=>[v,down[i],direction[i]]),focal:65};
    const dot=splat([0,0,4],[.65,.65,.65],[.5,.5,.5],1);
    const coeff=[[0,0,0],[state.cy,0,0],[state.cz,0,0],[state.cx,0,0]];
    paint(ui.images[0],renderScene(makeScene(dot),camera,65,65),65);
    paint(ui.images[1],renderScene(makeScene({...dot,sh:coeff}),camera,65,65),65);
    const basis=shBasis(direction),contributions=basis.map((v,i)=>v*coeff[i][0]);
    readout(ui,[['d = (x, y, z)',triple(direction)],['Y₀, Yᵧ, Y𝓏, Yₓ',basis.map(x=>show(x)).join(' / ')],
      [t('Четыре вклада в красный','Four red-channel contributions'),contributions.map(x=>show(x)).join(' / ')],
      [t('Красный до ограничения','Red before clipping'),show(.5+contributions.reduce((s,x)=>s+x,0))],
      ['RGB',triple(shColor(direction,coeff))]]);
  }
  for(const [key,label,min,max,step] of [
    ['angle',t('Азимут, °','Azimuth, °'),-180,180,.1],['elevation',t('Наклон, °','Elevation, °'),-80,80,.1],
    ['cx',t('Коэффициент при −x','Coefficient for −x'),-2,2,.05],
    ['cy',t('Коэффициент при −y','Coefficient for −y'),-2,2,.05],
    ['cz',t('Коэффициент при +z','Coefficient for +z'),-2,2,.05]]){
    numberInput(ui.controls,label,min,max,step,state[key],v=>{state[key]=v;draw();});
  }
  draw();return instance(ui,state);
}

export function mountMultiview(host,lang='ru',saved={}){
  const t=text(lang),state=stateFrom({depth:2.4,views:'two',viewX:.8,steps:0},saved);
  const ui=setup(host,lang,t('У пятна появилась глубина','A depth for the splat'),
    t('Левый кадр принадлежит образцу. Справа та же камера смотрит на обучаемую сцену.','The target is on the left. The same camera sees the fitted scene on the right.'),
    [t('Образец','Target'),t('Модель','Model')]);
  const views=()=>state.views==='two'?[0,.8]:[0];
  function draw(){
    const cam=translatedCamera(state.viewX);
    paint(ui.images[0],renderScene(rayScene(4),cam,33,33),33);
    paint(ui.images[1],renderScene(rayScene(state.depth),cam,33,33),33);
    readout(ui,[[t('Шаг / глубина','Step / depth'),state.steps+' / '+show(state.depth,5)],
      [t('Обучающая MSE','Training MSE'),show(rayLoss(state.depth,views()),8)],
      [t('Отложенная MSE, камера −0.6','Held-out MSE, camera −0.6'),show(rayLoss(state.depth,[-.6]),8)],
      [t('Текущая MSE показанного кадра','MSE of displayed frame'),show(rayLoss(state.depth,[state.viewX]),8)]]);
  }
  addSelect(ui.controls,{label:t('Камеры обучения','Training cameras'),options:[{value:'one',label:'0'},{value:'two',label:'0, +0.8'}],
    value:state.views,onChange:v=>{state.views=v;draw();}});
  addSelect(ui.controls,{label:t('Камера для просмотра','Preview camera'),options:[0,.8,-.6].map(x=>({value:String(x),label:String(x)})),
    value:String(state.viewX),onChange:v=>{state.viewX=+v;draw();}});
  const setDepth=numberInput(ui.controls,t('Глубина модели','Model depth'),1.25,7,.001,state.depth,v=>{state.depth=v;state.steps=0;draw();});
  function step(n){for(let i=0;i<n;i++)state.depth=depthStep(state.depth,views())[0];state.steps+=n;setDepth(state.depth);draw();}
  addButton(ui.controls,t('Один шаг','One step'),()=>step(1));
  addButton(ui.controls,t('20 шагов','20 steps'),()=>step(20));
  addButton(ui.controls,t('Исходная глубина','Starting depth'),()=>{state.steps=0;state.depth=2.4;setDepth(2.4);draw();});
  draw();return instance(ui,state);
}

export function mountDensity(host,lang='ru',saved={}){
  const t=text(lang),state=stateFrom({scene:densityScene(),selected:0,threshold:.001,steps:0,pixel:16},saved);
  const ui=setup(host,lang,t('Одно широкое или два узких','One wide splat or two narrow ones'),
    t('Обычный шаг двигает существующие пятна. Структурное действие сразу меняет состав или непрозрачность сцены.','A gradient step moves existing splats. A structural edit immediately changes the list or its opacities.'),
    [t('Цель','Target'),t('Сцена','Scene')]);
  let previous=densityLoss(state.scene);
  let measuredScene=null, gradients=[];
  const select=addSelect(ui.controls,{label:t('Номер пятна','Splat index'),options:[],value:'0',onChange:v=>{state.selected=+v;draw();}});
  const buttons=[];
  function change(f){previous=densityLoss(state.scene);state.scene=f(state.scene);state.steps++;draw();}
  for(const [op,ru,en] of [['clone','Клон','Clone'],['split','Разделение','Split'],['prune','Удаление слабых','Prune weak splats'],['reset','Сброс непрозрачности','Opacity reset']])
    buttons.push(addButton(ui.controls,t(ru,en),()=>change(s=>densityEdit(s,op,state.selected))));
  numberInput(ui.controls,t('Порог градиента','Gradient threshold'),0,.025,.0001,state.threshold,v=>{state.threshold=v;draw();});
  const setPixel=numberInput(ui.controls,t('Столбец u (строка 16)','Column u (row 16)'),0,32,1,state.pixel,v=>{
    state.pixel=Math.round(v);setPixel(state.pixel);draw();
  });
  addButton(ui.controls,t('Проверка плотности','Adaptive pass'),()=>change(s=>densityPass(s,state.threshold)));
  addButton(ui.controls,t('Шаг градиента','Gradient step'),()=>change(s=>densityStep(s)));
  addButton(ui.controls,t('Исходная сцена','Starting scene'),()=>{state.scene=densityScene();state.steps=0;state.selected=0;previous=densityLoss(state.scene);draw();});
  function draw(){
    const n=state.scene.splats.length,cam=translatedCamera(0,42);state.selected=Math.max(0,Math.min(state.selected,n-1));
    select.replaceChildren(...state.scene.splats.map((_,i)=>new Option(String(i),String(i))));select.value=String(state.selected);
    buttons[0].disabled=buttons[1].disabled=n===0||n>=16;
    const frames=[renderScene(densityScene(true),cam,33,33),renderScene(state.scene,cam,33,33)];
    frames.forEach((frame,index)=>{
      const canvas=ui.images[index];
      paint(canvas,frame,33);
      const context=canvas.getContext('2d');
      context.strokeStyle='#172a40';context.lineWidth=.25;
      context.strokeRect(state.pixel-.75,15.25,2.5,2.5);
    });
    const [targetRed,modelRed]=frames.map(frame=>frame[3*(16*33+state.pixel)]);
    const s=state.scene.splats[state.selected];
    if(state.scene!==measuredScene){gradients=densityGradients(state.scene);measuredScene=state.scene;}
    const gx=s?gradients[state.selected][0]:0;
    let decision=t('Без изменения','Keep');
    if(s && s.opacity<.03)decision=t('Удалить','Prune');
    else if(s && Math.abs(gx)>state.threshold)
      decision=s.scales[0]>.32?t('Разделить','Split'):t('Клонировать','Clone');
    readout(ui,[[t('Пятен','Splats'),n],[t('Числа сцены / байты float32','Scene scalars / float32 bytes'),(14*n)+' / '+(56*n)],
      [t('MSE до → после','MSE before → after'),show(previous,7)+' → '+show(densityLoss(state.scene),7)],
      [t('Строка 16 · R образца / модели / разность','Row 16 · target R / model R / difference'),triple([targetRed,modelRed,modelRed-targetRed])],
      [t('Выбранное пятно · x, σₓ, o','Selected splat · x, σₓ, o'),s?triple([s.center[0],s.scales[0],s.opacity]):'—'],
      [t('Градиент по x / кандидат до учёта лимита','X gradient / candidate before the count limit'),s?show(gx,7)+' / '+decision:'—']]);
  }
  draw();return instance(ui,state);
}
