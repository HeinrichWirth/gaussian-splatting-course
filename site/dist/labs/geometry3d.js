import {el, makeLab, addSelect, addButton} from './shared.js';
import {
  axisAngleQuaternion, quaternionMatrix, worldToCamera, projectPoint,
  matVec, multiply, transpose, covariance3d, projectionJacobian, projectedCovariance,
} from '../geometry-math.js';

const blue='#385bbd', orange='#bd6137', green='#518269', neutral='#586664';
const ns='http://www.w3.org/2000/svg', near=.2, principal=[100,60];
const axes={z:[0,0,1],y:[0,1,0],xyz:[1,1,1]};
const rotation=(axis,deg)=>quaternionMatrix(axisAngleQuaternion(axis,deg));
const fmt=x=>Number(x.toFixed(3)).toString();
const tuple=v=>'('+v.map(fmt).join(', ')+')';
const matrix=m=>m.map(row=>'['+row.map(fmt).join('  ')+']').join('   ');
const tFor=lang=>(ru,en)=>lang==='en'?en:ru;
let nextId=0;

function svgNode(tag, attributes) {
  const node=document.createElementNS(ns,tag);
  for (const [key,value] of Object.entries(attributes)) node.setAttribute(key,value);
  return node;
}
// Independent world/pixel coordinates; the scale is equal along both axes.
function panel(parent,title,xLabel,yLabel,xRange,yRange) {
  const figure=el('figure'),caption=el('figcaption','',title);
  const svg=svgNode('svg',{viewBox:'0 0 360 245',role:'img','aria-label':title,width:'100%'});
  figure.append(caption,svg); figure.style.margin='0'; parent.append(figure);
  const [xmin,xmax]=xRange, [bottom,top]=yRange;
  const scale=Math.min(280/(xmax-xmin),176/Math.abs(top-bottom));
  const left=182-(xmax-xmin)*scale/2, base=117+Math.abs(top-bottom)*scale/2;
  const px=([x,y])=>[left+(x-xmin)*scale,base-(y-bottom)*scale*Math.sign(top-bottom)];
  const [right,upper]=px([xmax,top]);
  const text=(x,y,label)=>{const n=svgNode('text',{x,y,fill:neutral,'font-size':12});n.textContent=label;svg.append(n);};
  const id='geo-clip-'+(++nextId), defs=svgNode('defs',{}),clip=svgNode('clipPath',{id});
  const bounds={x:left,y:upper,width:right-left,height:base-upper};
  clip.append(svgNode('rect',bounds));defs.append(clip);svg.append(defs);
  svg.append(svgNode('rect',{...bounds,fill:'#f8faf5',stroke:'#acbcae'}));
  text(left-6,base+15,fmt(xmin));text(right-18,base+15,fmt(xmax));text((left+right)/2-8,base+31,xLabel);
  text(Math.max(1,left-33),upper+6,fmt(top));text(Math.max(1,left-33),base,fmt(bottom));text(left,upper-8,yLabel);
  const layer=svgNode('g',{'clip-path':'url(#'+id+')'});svg.append(layer);
  function path(points,color=blue,width=2,opacity=1) {
    let data='',start=true;
    for(const p of points) {
      if(!p){start=true;continue;}
      const [x,y]=px(p);data+=(start?'M':'L')+x.toFixed(2)+' '+y.toFixed(2)+' ';start=false;
    }
    layer.append(svgNode('path',{d:data,fill:'none',stroke:color,'stroke-width':width,opacity}));
  }
  function mark(p,color=orange,r=4) {
    const [cx,cy]=px(p);layer.append(svgNode('circle',{cx,cy,r,fill:color}));
  }
  return {path,mark};
}

function experiment(host, lang, saved, config, draw) {
  const t=tFor(lang), ui=makeLab(host,{lang,title:t(...config.title),subtitle:t(...config.note)});
  const state={}, reset=[];
  ui.view.style.display='grid';
  ui.view.style.gridTemplateColumns='repeat(auto-fit, minmax(min(100%, 220px), 1fr))';
  ui.view.style.alignItems='center';
  const table=el('table','geometry-readout');
  table.style.width='100%';ui.readout.append(table);
  function update() {
    ui.view.replaceChildren();table.replaceChildren();
    const rows=draw(ui.view,state,t);
    for(const [label,value] of rows){
      const tr=el('tr'),th=el('th','',label),td=el('td','',value);
      th.scope='row';th.style.textAlign='left';th.style.padding='0.35rem';td.style.overflowWrap='anywhere';
      tr.append(th,td);table.append(tr);
    }
  }
  for(const [key,ru,en,min,max,step,initial] of config.numbers){
    const label=t(ru,en), row=el('div','control'), header=el('div','control-top');
    const caption=el('label','',label), numeric=el('input'), slider=el('input');
    caption.htmlFor=numeric.id='geo-number-'+(++nextId);
    numeric.type='number';slider.type='range';
    numeric.style.width='5.5rem';numeric.style.maxWidth='40%';
    for(const input of [numeric,slider]){Object.assign(input,{min,max,step});input.setAttribute('aria-label',label);}
    const set=value=>{state[key]=value;numeric.value=value;slider.value=value;};
    set(Number.isFinite(saved[key]) && saved[key]>=min && saved[key]<=max ? saved[key] : initial);
    slider.oninput=()=>{set(slider.valueAsNumber);update();};
    numeric.oninput=()=>{const value=numeric.valueAsNumber;if(Number.isFinite(value)&&value>=min&&value<=max){state[key]=value;slider.value=value;update();}};
    const normalize=()=>{const v=numeric.valueAsNumber;set(Number.isFinite(v)?Math.min(max,Math.max(min,v)):state[key]);update();};
    numeric.onchange=normalize;numeric.onblur=normalize;
    header.append(caption,numeric);row.append(header,slider);ui.controls.append(row);
    reset.push(()=>set(initial));
  }
  if(config.axis){
    state.axis=Object.hasOwn(axes,saved.axis)?saved.axis:'z';
    const select=addSelect(ui.controls,{label:t('Ось','Axis'),options:Object.keys(axes).map(axis=>({value:axis,label:tuple(axes[axis])})),
      value:state.axis,onChange:v=>{state.axis=v;update();}});
    reset.push(()=>{state.axis='z';select.value='z';});
  }
  addButton(ui.controls,t('Исходные параметры','Initial settings'),()=>{reset.forEach(fn=>fn());update();});
  update();
  return {getState:()=>({...state}),destroy:()=>ui.root.remove()};
}

export function mountFrames(host,lang='ru',saved={}) {
  return experiment(host,lang,saved,{
    title:['Один треугольник в двух системах','One triangle in two frames'],
    note:['Вид сверху. Оранжевая точка — подвижная вершина, тёмная точка — камера. Короткие линии показывают её правую и переднюю оси.',
      'Overhead view. Orange marks the movable vertex and the dark dot marks the camera. Short lines show its right and forward axes.'],
    numbers:[['pointX','Точка x','Point x',-2,2,.1,1],['objectX','Объект x','Object x',-2,2,.1,1],['cameraX','Камера x','Camera x',-2,2,.1,1],['yaw','Поворот камеры °','Camera yaw °',-70,70,1,0]],
  },(view,s,t)=>{
    const local=[[0,0,3],[s.pointX,1,4],[-.6,0,4.5],[0,0,3]], c=[s.cameraX,0,0];
    const world=local.map(([x,y,z])=>[x+s.objectX,y,z]), basis=rotation(axes.y,s.yaw);
    const camera=world.map(p=>worldToCamera(p,c,basis));
    const xz=p=>[p[0],p[2]];
    for(const [name,points,origin,r] of [[t('Мир','World'),world,c,basis],[t('Камера','Camera'),camera,[0,0,0],rotation(axes.y,0)]]){
      const fig=panel(view,name,'x','z',[-6,6],[-2,7]);
      fig.path(points.map(xz));fig.mark(xz(points[1]));fig.mark(xz(origin),neutral);
      for(const [i,color] of [[0,orange],[2,green]])fig.path([xz(origin),[origin[0]+r[0][i],origin[2]+r[2][i]]],color,3);
    }
    return [['p',tuple(local[1])],['P',tuple(world[1])],['P − C',tuple(world[1].map((x,i)=>x-c[i]))],['P camera',tuple(camera[1])]];
  });
}

export function mountPerspective(host,lang='ru',saved={}) {
  return experiment(host,lang,saved,{
    title:['Лучи и пиксели','Rays and pixels'],
    note:['Синяя перекладина имеет ширину 1 м. Зелёная черта находится в 0.2 м перед камерой. Центры пикселей идут от 0 до 200 по горизонтали и от 0 до 120 по вертикали.',
      'The blue bar is one meter wide. The green line lies 0.2 m in front of the camera. Pixel centers run from 0 to 200 horizontally and 0 to 120 vertically.'],
    numbers:[['f','Фокус, px','Focal length, px',20,160,1,80],['depth','Объект z','Object z',1,7,.1,4],['cameraZ','Камера z','Camera z',-2,6,.1,0]],
  },(view,s,t)=>{
    const z=s.depth-s.cameraZ,side=panel(view,t('Лучи сверху','Rays from above'),'x','z',[-2.2,2.2],[-2.5,8]);
    side.path([[.5,s.depth],[1.5,s.depth]],blue,4);side.mark([0,s.cameraZ],neutral);
    for(const x of [.5,1,1.5])side.path([[0,s.cameraZ],[x,s.depth]],orange,1.3);
    side.path([[-2.2,s.cameraZ+near],[2.2,s.cameraZ+near]],green,1);
    const screen=panel(view,t('Сенсор','Sensor'),'u [px]','v [px]',[-.5,200.5],[120.5,-.5]);
    screen.mark(principal,neutral,2);
    const rows=[['z',fmt(z)+' m']];
    if(z<=near)return [...rows,[t('Изображение','Image'),t('Отсечено ближней плоскостью','Clipped by the near plane')]];
    const [a,b,mid]=[.5,1.5,1].map(x=>projectPoint([x,.5,z],s.f,principal,near));
    screen.path([a,b],blue,4);screen.mark(mid);
    return [...rows,['(u, v)',tuple(mid)+' px'],['Δu',fmt(b[0]-a[0])+' px']];
  });
}

function sphereLines() {
  const lines=[];
  for(let j=1;j<12;j++){
    const angle=-Math.PI/2+j*Math.PI/12;
    lines.push(Array.from({length:97},(_,i)=>{const t=i*Math.PI/48;return [Math.cos(angle)*Math.cos(t),Math.cos(angle)*Math.sin(t),Math.sin(angle)];}));
  }
  for(let j=0;j<12;j++){
    const angle=j*Math.PI/12;
    lines.push(Array.from({length:97},(_,i)=>{const t=i*Math.PI/48;return [Math.cos(t)*Math.cos(angle),Math.cos(t)*Math.sin(angle),Math.sin(t)];}));
  }
  return lines;
}
const sphere=sphereLines();
function shellLines(r,scales){return sphere.map(line=>line.map(p=>matVec(r,p.map((x,i)=>x*scales[i]))));}

export function mountEllipsoid(host,lang='ru',saved={}) {
  return experiment(host,lang,saved,{
    title:['Один вес на всей оболочке','One weight across the shell'],
    note:['Линии соединяют точки веса exp(−1/2). Три цветных отрезка показывают собственные оси. Обе картинки ортографические, дальние отрезки сохраняют размер.',
      'Mesh lines join points of weight exp(−1/2). The three colored segments are the principal axes. Both drawings are orthographic, so distance does not change the scale.'],
    numbers:[['sx','Масштаб x','Scale x',.2,3,.05,2],['sy','Масштаб y','Scale y',.2,3,.05,.75],['sz','Масштаб z','Scale z',.2,3,.05,1],['angle','Угол °','Angle °',-180,180,1,45]],axis:true,
  },(view,s,t)=>{
    const q=axisAngleQuaternion(axes[s.axis],s.angle),r=quaternionMatrix(q),scales=[s.sx,s.sy,s.sz];
    const oblique=multiply(rotation([1,0,0],-20),rotation(axes.y,30));
    for(const [title,basis,lx,ly] of [[t('Косой вид','Oblique view'),oblique,'a','b'],['XY',rotation(axes.y,0),'x','y']]){
      const fig=panel(view,title,lx,ly,[-3.1,3.1],[3.1,-3.1]);
      const transform=multiply(basis,r);
      for(const line of shellLines(transform,scales))fig.path(line,blue,1,.5);
      for(const [i,color] of [[0,orange],[1,green],[2,'#7952a4']]){
        fig.path([[0,0],[transform[0][i]*scales[i],transform[1][i]*scales[i]]],color,3);
      }
    }
    return [['q (w, x, y, z)',tuple(q)],['Σ [m²]',matrix(covariance3d(scales,q))]];
  });
}

export function mountFootprint(host,lang='ru',saved={}) {
  return experiment(host,lang,saved,{
    title:['Пятно и точная проекция оболочки','The footprint and the projected shell'],
    note:['Синим показан контур одного масштаба по якобиану. Оранжевые линии получены делением на собственную глубину каждой точки. Чёрная метка — проекция центра.',
      'Blue marks one scale of the Jacobian footprint. Each point of the orange mesh is divided by its own depth. The dark dot marks the projected center.'],
    numbers:[['z','Глубина центра','Center depth',.5,7,.1,4],['sz','Масштаб z','Scale z',.1,2,.1,1],['yaw','Ракурс °','View angle °',-90,90,1,0],
      ['tilt','Поворот формы °','Shape rotation °',-90,90,1,0]],
  },(view,s,t)=>{
    const mu=[1,0,s.z],scales=[.5,.3,s.sz],worldQ=axisAngleQuaternion(axes.z,s.tilt);
    const cameraBasis=rotation(axes.y,-s.yaw),r=multiply(cameraBasis,quaternionMatrix(worldQ));
    const cov=multiply(multiply(cameraBasis,covariance3d(scales,worldQ)),transpose(cameraBasis));
    const center=projectPoint(mu,80,principal,near),cov2=projectedCovariance(mu,cov,80);
    const fig=panel(view,t('Кадр 201 × 121','201 × 121 frame'),'u [px]','v [px]',[-.5,200.5],[120.5,-.5]);
    for(const line of shellLines(r,scales)){
      const image=line.map(offset=>{const p=offset.map((v,i)=>v+mu[i]);return p[2]>near?projectPoint(p,80,principal,near):null;});
      fig.path(image,orange,1,.55);
    }
    // L Lᵀ = covariance. L maps the unit circle to the covariance ellipse.
    const l00=Math.sqrt(cov2[0][0]), l10=cov2[1][0]/l00, l11=Math.sqrt(cov2[1][1]-l10*l10);
    fig.path(Array.from({length:129},(_,i)=>{const a=i*Math.PI/64;return [center[0]+l00*Math.cos(a),center[1]+l10*Math.cos(a)+l11*Math.sin(a)];}),blue,2.5);
    fig.mark(center,neutral,3);
    const zmin=s.z-Math.sqrt(cov[2][2]);
    return [['(u, v)',tuple(center)+' px'],['J [px/m]',matrix(projectionJacobian(mu,80))],['Σ₂ [px²]',matrix(cov2)],
      [t('Ближайшая глубина оболочки','Nearest shell depth'),fmt(zmin)+' m'+(zmin<=near?t(' · часть отсечена',' · partly clipped'):'')]];
  });
}
