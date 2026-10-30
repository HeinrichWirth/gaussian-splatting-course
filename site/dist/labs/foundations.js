// Browser versions of the first two course experiments. All calculations are local.
import { gaussian1D } from '../math.js';
import { makeLab, el as element } from './shared.js';
const TAU = 2 * Math.PI;
const BLUE = '#2463a9';
const ORANGE = '#cf5a28';
const INK = '#26364a';
const MUTED = '#52647a';
const PAPER = '#f4f7fb';
const BACKGROUND = [244, 247, 251];
const tr = (lang, ru, en) => lang === 'en' ? en : ru;
const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);
const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const clamp = (value, low, high, fallback) => Number.isFinite(Number(value)) ? Math.min(high, Math.max(low, Number(value))) : fallback;
const rgb = color => `rgb(${color.map(value => Math.round(value * 255)).join(' ')})`;

function number(value, digits = 6) {
  if (!Number.isFinite(value)) return String(value);
  if (value === 0 || Object.is(value, -0)) return '0';
  if (Math.abs(value) < 1e-5 || Math.abs(value) >= 1e7) return value.toExponential(4);
  return String(Number(value.toFixed(digits)));
}

function installStyles() {
  if (document.querySelector('link[data-foundations-style]')) return;
  const link = document.createElement('link');
  link.rel = 'stylesheet';
  link.href = new URL('./foundations.css', import.meta.url).href;
  link.dataset.foundationsStyle = '';
  document.head.append(link);
}

function frame(container, lang, title, description) {
  if (!(container instanceof HTMLElement)) throw new TypeError('A lab container is required');
  installStyles();
  const {root,view:visual,controls}=makeLab(container,{title,subtitle:description,lang});
  root.classList.add('foundations-lab');
  root.querySelector('h3').classList.add('foundations-title');
  root.querySelector('.lab-head p').classList.add('foundations-intro');
  root.querySelector('.lab-body').classList.add('foundations-layout');
  visual.classList.add('foundations-visual');
  const canvas = element('canvas', 'foundations-canvas');
  canvas.setAttribute('role', 'img');
  canvas.setAttribute('aria-label', title);
  canvas.textContent = description;
  visual.append(canvas);
  controls.classList.add('foundations-controls');
  controls.setAttribute('aria-label', tr(lang, 'Параметры', 'Parameters'));
  const cleanup = [];
  let alive = true;
  let frameId = 0;
  let painter = null;
  function schedule() {
    if (!alive || frameId) return;
    frameId = requestAnimationFrame(() => {
      frameId = 0;
      if (alive) painter?.();
    });
  }
  function on(node, name, callback, options) {
    node.addEventListener(name, callback, options);
    cleanup.push(() => node.removeEventListener(name, callback, options));
  }
  if (typeof ResizeObserver !== 'undefined') {
    const observer = new ResizeObserver(schedule);
    observer.observe(visual);
    cleanup.push(() => observer.disconnect());
  } else on(window, 'resize', schedule);
  return {
    root, visual, canvas, controls, on, schedule,
    paint(callback) { painter = callback; schedule(); },
    changed(state) { root.dispatchEvent(new CustomEvent('labchange', { bubbles: true, detail: { ...state } })); },
    destroy() {
      alive = false;
      cancelAnimationFrame(frameId);
      cleanup.splice(0).forEach(fn => fn());
      root.remove();
    },
  };
}

let controlId = 0;
function rangeControl(ui, key, label, value, min, max, step, change) {
  const wrap = element('div', 'foundations-control');
  const line = element('div', 'foundations-control-line');
  const caption = element('label', '', label);
  const range = element('input');
  range.type = 'range';
  range.name = key;
  range.id = `foundation-${++controlId}`;
  range.min = String(min); range.max = String(max); range.step = String(step);
  range.value = String(value); caption.htmlFor = range.id;
  const numeric = element('input', 'foundations-number');
  numeric.type = 'number'; numeric.name = `${key}-value`;
  numeric.min = String(min); numeric.max = String(max); numeric.step = String(step);
  numeric.value = String(value); numeric.setAttribute('aria-label', label);
  const set = next => { range.value = String(next); numeric.value = String(next); };
  ui.on(range, 'input', () => { numeric.value = range.value; change(Number(range.value)); });
  ui.on(numeric, 'input', () => {
    const next = numeric.valueAsNumber;
    // Incomplete edits stay in the field; a usable value updates the experiment now.
    if (!Number.isFinite(next) || next < min || next > max) return;
    range.value = String(next);
    change(next);
  });
  const normalizeNumeric = () => {
    const next = clamp(numeric.valueAsNumber, min, max, Number(range.value));
    range.value = String(next);
    const snapped = Number(range.value);
    numeric.value = range.value;
    change(snapped);
  };
  ui.on(numeric, 'change', normalizeNumeric);
  ui.on(numeric, 'blur', normalizeNumeric);
  line.append(caption, numeric); wrap.append(line, range); ui.controls.append(wrap);
  return { set, range, numeric };
}

function selectControl(ui, key, label, choices, value, change) {
  const wrap = element('label', 'foundations-select-label');
  wrap.append(element('span', '', label));
  const select = element('select'); select.name = key;
  choices.forEach(([id, name]) => { const option = element('option', '', name); option.value = id; select.append(option); });
  select.value = value;
  ui.on(select, 'change', () => change(select.value));
  wrap.append(select); ui.controls.append(wrap);
  return select;
}

function resetButton(ui, lang, action) {
  const button = element('button', 'foundations-reset', tr(lang, 'Исходные параметры', 'Initial parameters'));
  button.type = 'button'; button.name = 'reset'; ui.on(button, 'click', action); ui.controls.append(button);
}

function setupCanvas(canvas, heightRatio = 1, minimumHeight = 0) {
  const width = Math.max(1, canvas.clientWidth || canvas.parentElement.clientWidth || 560);
  const height = Math.max(minimumHeight, Math.round(width * heightRatio));
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const pixelWidth = Math.round(width*dpr), pixelHeight = Math.round(height*dpr);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth; canvas.height = pixelHeight;
  }
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return { ctx, width, height };
}

// The same deterministic scene as course_demos.py: 128 surface splats, 64 floor splats.
function sceneData() {
  const result = [];
  for (let i = 0; i < 128; i++) {
    const z = 1 - 2*(i+.5)/128;
    const radius = Math.sqrt(1-z*z);
    const phi = i*Math.PI*(3-Math.sqrt(5));
    const normal = [radius*Math.cos(phi), radius*Math.sin(phi), z];
    const tangentRaw = cross(normal, [0,0,1]);
    const length = Math.hypot(...tangentRaw);
    const tangent = tangentRaw.map(value => value/length);
    const bitangent = cross(normal, tangent);
    const axes = [tangent, bitangent, normal];
    const variances = [.16**2, .13**2, .045**2];
    const covariance = Array.from({length:3}, (_, row) => Array.from({length:3}, (_, col) =>
      axes.reduce((sum, axis, k) => sum+axis[row]*variances[k]*axis[col], 0)));
    result.push({ mean: [normal[0]*.8, normal[1]*.8, normal[2]*.8+1.1], covariance,
      color: [.35+.55*(normal[0]+1)/2, .30+.50*(normal[1]+1)/2, .35+.55*(normal[2]+1)/2],
      opacity: .85, scales:[.16,.13,.045] });
  }
  for (let row = 0; row < 8; row++) for (let col = 0; col < 8; col++) {
    result.push({mean:[-1.55+col*3.1/7, -1.55+row*3.1/7, 0],
      covariance:[[.23**2,0,0],[0,.23**2,0],[0,0,.025**2]],
      color:(row+col)%2 ? [.24,.49,.57] : [.72,.82,.78], opacity:.88, scales:[.23,.23,.025]});
  }
  return result;
}

function projectScene(data, azimuth, elevation) {
  const az = azimuth*Math.PI/180, el = elevation*Math.PI/180;
  const toward = [Math.cos(el)*Math.cos(az), Math.cos(el)*Math.sin(az), Math.sin(el)];
  const right = [-Math.sin(az), Math.cos(az), 0];
  const up = cross(toward, right);
  const bilinear = (a, matrix, b) => dot(a, matrix.map(row => dot(row,b)));
  return data.map((spot, index) => {
    const relative = [spot.mean[0], spot.mean[1], spot.mean[2]-.75];
    return {index, x:dot(relative,right), y:dot(relative,up), depth:dot(relative,toward),
      a:bilinear(right,spot.covariance,right), b:bilinear(right,spot.covariance,up), c:bilinear(up,spot.covariance,up)};
  });
}

function sceneRaster(data, projected, state, size) {
  const half = 2.2/state.zoom;
  const step = 2*half/(size-1);
  const pixels = new Float32Array(size*size*3);
  for (let offset=0; offset<pixels.length; offset+=3) pixels.set(BACKGROUND,offset);
  const order = state.isolate ? [projected[state.selected]] : [...projected].sort((a,b) => a.depth-b.depth || a.index-b.index);
  for (const point of order) {
    const spot = data[point.index];
    const opacity = point.index === state.selected ? state.opacity : spot.opacity;
    if (!opacity) continue;
    const rx=4*Math.sqrt(point.a), ry=4*Math.sqrt(point.c);
    const x0=Math.max(0,Math.ceil((point.x-rx+half)/step));
    const x1=Math.min(size-1,Math.floor((point.x+rx+half)/step));
    const y0=Math.max(0,Math.ceil((point.y-ry+half)/step));
    const y1=Math.min(size-1,Math.floor((point.y+ry+half)/step));
    const determinant = point.a*point.c-point.b*point.b;
    const ia=point.c/determinant, ib=-point.b/determinant, ic=point.a/determinant;
    const r=spot.color[0]*255, g=spot.color[1]*255, b=spot.color[2]*255;
    for (let iy=y0; iy<=y1; iy++) {
      const dy=-half+iy*step-point.y;
      for (let ix=x0; ix<=x1; ix++) {
        const dx=-half+ix*step-point.x;
        const q=ia*dx*dx+2*ib*dx*dy+ic*dy*dy;
        if (q>16) continue;
        const alpha=opacity*Math.exp(-.5*q), remaining=1-alpha;
        const offset=((size-1-iy)*size+ix)*3;
        pixels[offset]=alpha*r+remaining*pixels[offset];
        pixels[offset+1]=alpha*g+remaining*pixels[offset+1];
        pixels[offset+2]=alpha*b+remaining*pixels[offset+2];
      }
    }
  }
  const image = new ImageData(size,size);
  for (let i=0; i<size*size; i++) {
    image.data[i*4]=pixels[i*3]; image.data[i*4+1]=pixels[i*3+1];
    image.data[i*4+2]=pixels[i*3+2]; image.data[i*4+3]=255;
  }
  return image;
}

/** Mount the 192-Gaussian scene explorer. State is independent for every instance. */
export function mountScene(container, lang='ru', initial={}) {
  const data=sceneData();
  const defaults={azimuth:-55,elevation:24,zoom:1,selected:0,opacity:.85,isolate:false,mode:'composite'};
  const selected=Math.round(clamp(initial.selected,0,191,0));
  const state={azimuth:clamp(initial.azimuth,-180,180,-55), elevation:clamp(initial.elevation,-10,80,24),
    zoom:clamp(initial.zoom,.65,1.8,1), selected,
    opacity:clamp(initial.opacity,0,1,data[selected].opacity), isolate:initial.isolate===true,
    mode:['centers','contours','composite'].includes(initial.mode)?initial.mode:'composite'};
  const ui=frame(container,lang,tr(lang,'Одна сцена · 192 гауссианы','One scene · 192 Gaussians'),
    tr(lang,'128 пятен образуют шар, ещё 64 — плоскость. Камера меняет вид этой же сцены; параметры заданы вручную, обучения по фотографиям здесь нет.',
      '128 spots form a sphere and another 64 form a plane. The camera changes the view of this same hand-built scene; no training from photographs takes place.'));
  const controls={};
  const notify=()=>{ updateCard(); ui.schedule(); ui.changed(state); };
  const mode=selectControl(ui,'mode',tr(lang,'Представление','View'),[
    ['centers',tr(lang,'Центры','Centers')],['contours',tr(lang,'Контуры веса 0.607','Weight-0.607 contours')],
    ['composite',tr(lang,'Итоговое изображение','Composite image')]],state.mode,value=>{state.mode=value;notify();});
  for (const [key,ru,en,min,max,step] of [
    ['azimuth','Поворот камеры, °','Camera orbit, °',-180,180,5],
    ['elevation','Высота взгляда, °','View elevation, °',-10,80,2],
    ['zoom','Приближение','Zoom',.65,1.8,.05],
    ['selected','Номер гауссианы','Gaussian index',0,191,1],
    ['opacity','Непрозрачность','Opacity',0,1,.01],
  ]) controls[key]=rangeControl(ui,key,tr(lang,ru,en),state[key],min,max,step,value=>{
    state[key]=key==='selected'?Math.round(value):value;
    if(key==='selected'){state.opacity=data[state.selected].opacity;controls.opacity.set(state.opacity);}
    notify();
  });
  const checkLabel=element('label','foundations-check');
  const isolate=element('input'); isolate.type='checkbox'; isolate.name='isolate'; isolate.checked=state.isolate;
  checkLabel.append(isolate,document.createTextNode(tr(lang,'Только выбранная гауссиана','Only the selected Gaussian')));
  ui.controls.append(checkLabel);
  ui.on(isolate,'change',()=>{state.isolate=isolate.checked;notify();});
  resetButton(ui,lang,()=>{Object.assign(state,defaults); Object.entries(controls).forEach(([key,control])=>control.set(state[key]));
    mode.value=state.mode;isolate.checked=state.isolate;notify();});
  const card=element('div','foundations-card');
  card.setAttribute('aria-live','polite'); card.setAttribute('aria-atomic','true'); ui.visual.append(card);
  const explanation=element('p','foundations-note',tr(lang,
    'Проекция ортографическая. В итоговом изображении пятна накладываются от дальнего центра к ближнему. Это учебное приближение сортировки, не перспективный растеризатор 3DGS. Непрозрачность влияет на изображение, а диагностические центры и контуры сохраняются.',
    'The projection is orthographic. The composite draws spots from the farthest center to the nearest. This is a teaching approximation of ordering, not the perspective 3DGS rasterizer. Opacity affects the composite; diagnostic centers and contours stay visible.'));
  const cutoff=element('p','foundations-note',tr(lang,
    'Контур показывает вес exp(−1/2) ≈ 0.607, а не край пятна. Для скорости расчёт хвоста обрывается при q > 16: там вес меньше exp(−8) ≈ 0.000335. Это численное усечение; математическая гауссиана продолжается.',
    'The contour shows weight exp(−1/2) ≈ 0.607, not a spot boundary. For speed, the tail calculation stops at q > 16, where its weight is below exp(−8) ≈ 0.000335. This numerical cutoff does not end the mathematical Gaussian.'));
  ui.root.append(explanation,cutoff);
  function updateCard(){
    const spot=data[state.selected];
    const projected=projectScene(data,state.azimuth,state.elevation)[state.selected];
    card.replaceChildren();
    card.append(element('h4','',tr(lang,`Гауссиана № ${state.selected} · ${state.selected<128?'шар':'плоскость'}`,
      `Gaussian #${state.selected} · ${state.selected<128?'sphere':'plane'}`)));
    const definitions=element('dl','foundations-parameters');
    for(const [label,value] of [
      [tr(lang,'Положение (x, y, z)','Position (x, y, z)'),`(${spot.mean.map(v=>number(v,3)).join(', ')})`],
      [tr(lang,'Масштабы собственных осей','Own-axis scales'),`(${spot.scales.map(v=>number(v,3)).join(', ')})`],
      [tr(lang,'Цвет RGB','RGB color'),`(${spot.color.map(v=>number(v,3)).join(', ')})`],
      [tr(lang,'Непрозрачность в центре','Opacity at the center'),number(state.opacity,3)],
      [tr(lang,'Центр на экране','Projected center'),`(${number(projected.x,3)}, ${number(projected.y,3)})`],
      [tr(lang,'Глубина центра: больше — ближе','Center depth: larger is nearer'),number(projected.depth,3)],
    ]) definitions.append(element('dt','',label),element('dd','',value));
    card.append(definitions);
    ui.canvas.setAttribute('aria-label',tr(lang,
      `Учебная сцена, ${state.mode}, поворот ${state.azimuth}°, высота ${state.elevation}°. Выбрана гауссиана ${state.selected}.`,
      `Teaching scene, ${state.mode}, orbit ${state.azimuth}°, elevation ${state.elevation}°. Selected Gaussian ${state.selected}.`));
  }
  const rasterCanvas=document.createElement('canvas');
  let rasterKey='',cachedRaster=null;
  ui.paint(()=>{
    const {ctx,width,height}=setupCanvas(ui.canvas);
    ctx.fillStyle=PAPER;ctx.fillRect(0,0,width,height);
    const half=2.2/state.zoom, scale=width/(2*half);
    const sx=x=>(x+half)*scale, sy=y=>(half-y)*scale;
    const projected=projectScene(data,state.azimuth,state.elevation);
    const order=state.isolate?[projected[state.selected]]:[...projected].sort((a,b)=>a.depth-b.depth||a.index-b.index);
    ctx.save();ctx.beginPath();ctx.rect(0,0,width,height);ctx.clip();
    if(state.mode==='composite'){
      const size=321;
      const key=[state.azimuth,state.elevation,state.zoom,state.selected,state.opacity,state.isolate].join('|');
      if(key!==rasterKey){cachedRaster=sceneRaster(data,projected,state,size);rasterKey=key;
        rasterCanvas.width=size;rasterCanvas.height=size;rasterCanvas.getContext('2d').putImageData(cachedRaster,0,0);}
      ctx.imageSmoothingEnabled=true;ctx.drawImage(rasterCanvas,0,0,width,height);
    } else for(const point of order){
      ctx.strokeStyle=rgb(data[point.index].color);ctx.fillStyle=ctx.strokeStyle;
      ctx.beginPath();
      if(state.mode==='centers'){ctx.arc(sx(point.x),sy(point.y),2.5,0,TAU);ctx.fill();}
      else{
        const difference=Math.hypot(point.a-point.c,2*point.b);
        const major=Math.sqrt((point.a+point.c+difference)/2)*scale;
        const minor=Math.sqrt(Math.max(0,(point.a+point.c-difference)/2))*scale;
        const angle=.5*Math.atan2(2*point.b,point.a-point.c);
        ctx.ellipse(sx(point.x),sy(point.y),major,minor,-angle,0,TAU);ctx.lineWidth=1.1;ctx.stroke();
      }
    }
    const active=projected[state.selected], px=sx(active.x),py=sy(active.y);
    ctx.lineWidth=3;ctx.strokeStyle='#ffffff';
    for(const color of ['#ffffff',INK]){ctx.strokeStyle=color;ctx.beginPath();ctx.moveTo(px-7,py);ctx.lineTo(px+7,py);ctx.moveTo(px,py-7);ctx.lineTo(px,py+7);ctx.stroke();ctx.lineWidth=1.4;}
    ctx.restore();
  });
  updateCard();
  return {destroy:()=>ui.destroy(),getState:()=>({...state})};
}

/** Mount the peak-one 1D Gaussian with its complete arithmetic. */
export function mountBell(container,lang='ru',initial={}){
  const presets={default:{mu:2,sigma:2,x:4},mirror:{mu:2,sigma:2,x:0},narrow:{mu:2,sigma:1,x:4},
    center:{mu:2,sigma:2,x:2},tail:{mu:2,sigma:2,x:8}};
  const state={mu:clamp(initial.mu,-4,4,2),sigma:clamp(initial.sigma,.2,4,2),x:clamp(initial.x,-12,12,4)};
  const ui=frame(container,lang,tr(lang,'Центр, масштаб и одна точка','Center, scale, and one point'),
    tr(lang,'Кривая и полоса яркости показывают один и тот же вес. Оси остаются неподвижными: изменение масштаба действительно меняет ширину колокола.',
      'The curve and brightness strip show the same weight. The axes stay fixed, so changing the scale visibly changes the width of the bell.'));
  ui.root.classList.add('foundations-bell');
  const controls={};
  function matchingPreset(){return Object.keys(presets).find(key=>Object.keys(state).every(name=>presets[key][name]===state[name]))||'custom';}
  const update=()=>{preset.value=matchingPreset();updateSteps();ui.schedule();ui.changed(state);};
  for(const [key,ru,en,min,max,step] of [['mu','Центр μ','Center μ',-4,4,.1],['sigma','Масштаб σ','Scale σ',.2,4,.05],['x','Точка x','Point x',-12,12,.1]]){
    controls[key]=rangeControl(ui,key,tr(lang,ru,en),state[key],min,max,step,value=>{state[key]=value;update();});
  }
  const preset=selectControl(ui,'preset',tr(lang,'Пример','Example'),[
    ['default',tr(lang,'Исходный пример','Starting example')],['mirror',tr(lang,'Симметричная точка','Symmetric point')],
    ['narrow',tr(lang,'Узкий колокол','Narrower bell')],['center',tr(lang,'Точка в центре','Point at the center')],
    ['tail',tr(lang,'Расстояние 3σ','Distance 3σ')],['custom',tr(lang,'Текущие параметры','Current parameters')],
  ],matchingPreset(),value=>{if(!presets[value])return;Object.assign(state,presets[value]);Object.entries(controls).forEach(([key,control])=>control.set(state[key]));update();});
  resetButton(ui,lang,()=>{Object.assign(state,presets.default);Object.entries(controls).forEach(([key,control])=>control.set(state[key]));update();});
  const calculation=element('div','foundations-calculation');calculation.setAttribute('aria-live','polite');
  calculation.setAttribute('aria-atomic','true');ui.root.append(calculation);
  ui.root.append(element('p','foundations-note',tr(lang,
    'Расчёт использует неокруглённые значения. На расстоянии σ вес ≈ 0.606531; на 2σ ≈ 0.135335; на 3σ ≈ 0.011109. Высота в центре всегда равна 1: это вес, а не нормированная плотность вероятности.',
    'The calculation uses unrounded values. At distance σ the weight is about 0.606531; at 2σ, 0.135335; at 3σ, 0.011109. The peak is always 1: this is a weight, not a normalized probability density.')));
  function steps(){const d=state.x-state.mu,u=d/state.sigma,q=u*u,t=-q/2;return {d,u,q,t,weight:Math.exp(t)};}
  function updateSteps(){
    const {d,u,q,t,weight}=steps();
    const table=element('table','foundations-step-table');
    const caption=element('caption','',tr(lang,'От координаты до веса','From coordinate to weight'));table.append(caption);
    const body=element('tbody');
    for(const [label,formula] of [
      [tr(lang,'1. Смещение d','1. Offset d'),`x − μ = ${number(state.x)} − (${number(state.mu)}) = ${number(d)}`],
      [tr(lang,'2. В единицах масштаба u','2. In scale units u'),`d / σ = ${number(d)} / ${number(state.sigma)} = ${number(u)}`],
      [tr(lang,'3. Квадрат q','3. Square q'),`u² = (${number(u)}) × (${number(u)}) = ${number(q)}`],
      [tr(lang,'4. Показатель t','4. Exponent t'),`−q / 2 = −(${number(q)}) / 2 = ${number(t)}`],
      [tr(lang,'5. Вес f(x)','5. Weight f(x)'),`exp(t) = exp(${number(t)}) = ${number(weight)}`],
    ]){const row=element('tr');const heading=element('th','',label);heading.scope='row';row.append(heading,element('td','',formula));body.append(row);}
    table.append(body);calculation.replaceChildren(table);
    if(weight===0)calculation.append(element('p','foundations-note',tr(lang,
      'Математически вес положителен. Здесь он настолько мал, что числовой формат браузера сохраняет 0.',
      'The mathematical weight is positive. Here it is too small for the browser’s numeric format and is stored as 0.')));
    ui.canvas.setAttribute('aria-label',tr(lang,`Гауссиана: μ=${number(state.mu)}, σ=${number(state.sigma)}, x=${number(state.x)}, вес ${number(weight)}. Оси x от −12 до 12, вес от 0 до 1.05.`,
      `Gaussian: μ=${number(state.mu)}, σ=${number(state.sigma)}, x=${number(state.x)}, weight ${number(weight)}. Fixed x axis from −12 to 12 and weight axis from 0 to 1.05.`));
  }
  ui.paint(()=>{
    const {ctx,width,height}=setupCanvas(ui.canvas,.72,245);
    const left=45,right=15,top=23,bottom=height-78,plotW=width-left-right,plotH=bottom-top;
    const sx=x=>left+(x+12)/24*plotW,sy=y=>bottom-y/1.05*plotH;
    const weight=x=>gaussian1D(x,state.mu,state.sigma);
    ctx.fillStyle='#ffffff';ctx.fillRect(0,0,width,height);
    ctx.font='11px system-ui, sans-serif';ctx.fillStyle=MUTED;ctx.lineWidth=1;
    for(let x=-12;x<=12;x+=width<380?6:3){ctx.strokeStyle='#e6edf4';ctx.beginPath();ctx.moveTo(sx(x),top);ctx.lineTo(sx(x),bottom);ctx.stroke();ctx.textAlign='center';ctx.fillText(String(x),sx(x),bottom+17);}
    for(let y=0;y<=1;y+=.25){ctx.strokeStyle='#e6edf4';ctx.beginPath();ctx.moveTo(left,sy(y));ctx.lineTo(width-right,sy(y));ctx.stroke();ctx.textAlign='right';ctx.fillText(number(y,2),left-7,sy(y)+4);}
    ctx.fillStyle=INK;ctx.textAlign='left';ctx.fillText('f(x)',left,14);ctx.textAlign='right';ctx.fillText('x',width-right,bottom+17);
    ctx.save();ctx.beginPath();ctx.rect(left,top,plotW,plotH);ctx.clip();
    ctx.beginPath();ctx.moveTo(sx(-12),sy(0));
    for(let i=0;i<=800;i++){const x=-12+24*i/800;ctx.lineTo(sx(x),sy(weight(x)));}
    ctx.lineTo(sx(12),sy(0));ctx.closePath();ctx.fillStyle='rgba(36,99,169,.09)';ctx.fill();
    ctx.beginPath();for(let i=0;i<=800;i++){const x=-12+24*i/800;const px=sx(x),py=sy(weight(x));if(i===0)ctx.moveTo(px,py);else ctx.lineTo(px,py);}
    ctx.strokeStyle=BLUE;ctx.lineWidth=2.5;ctx.stroke();
    for(let k=-3;k<=3;k++){
      const x=state.mu+k*state.sigma;if(x< -12||x>12)continue;
      ctx.setLineDash([2,4]);ctx.lineWidth=1;ctx.strokeStyle='#8293a5';ctx.beginPath();ctx.moveTo(sx(x),sy(0));ctx.lineTo(sx(x),sy(weight(x)));ctx.stroke();ctx.setLineDash([]);
      ctx.beginPath();ctx.arc(sx(x),sy(weight(x)),2.5,0,TAU);ctx.fillStyle=BLUE;ctx.fill();
      if(k===0||state.sigma>=1.4){const label=k===0?'μ':`μ${k<0?'−':'+'}${Math.abs(k)===1?'':Math.abs(k)}σ`;ctx.fillStyle=MUTED;ctx.textAlign='center';ctx.fillText(label,sx(x),sy(1)+12);}
    }
    ctx.setLineDash([5,4]);ctx.strokeStyle=ORANGE;ctx.beginPath();ctx.moveTo(sx(state.x),top);ctx.lineTo(sx(state.x),bottom);ctx.stroke();ctx.setLineDash([]);
    ctx.beginPath();ctx.arc(sx(state.x),sy(steps().weight),5,0,TAU);ctx.fillStyle=ORANGE;ctx.fill();ctx.strokeStyle='white';ctx.lineWidth=1.5;ctx.stroke();ctx.restore();
    const stripY=bottom+35,stripH=19;
    for(let i=0;i<Math.ceil(plotW);i++){const x=-12+24*i/plotW;const w=weight(x);const brightness=Math.round(255*w);ctx.fillStyle=`rgb(${brightness} ${brightness} ${brightness})`;ctx.fillRect(left+i,stripY,1.3,stripH);}
    ctx.strokeStyle=ORANGE;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(sx(state.x),stripY-3);ctx.lineTo(sx(state.x),stripY+stripH+3);ctx.stroke();
    ctx.fillStyle=MUTED;ctx.textAlign='left';ctx.fillText(tr(lang,'Яркость: 0 — чёрное, 1 — белое','Brightness: 0 black, 1 white'),left,height-5);
  });
  updateSteps();
  return {destroy:()=>ui.destroy(),getState:()=>({...state})};
}
