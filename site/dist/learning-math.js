// Small CPU reference for image fitting. Parameters stay in the units shown in the lessons.
import {compositeAt} from './math.js';
export const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
export const bell = (x, mu, sigma = 1) => Math.exp(-.5 * ((x - mu) / sigma) ** 2);
export const linspace = (a, b, n) => Array.from({length:n}, (_, i) => a + (b-a)*i/(n-1));

export function imageLoss(prediction, target) {
  if (!prediction.length || prediction.length !== target.length) throw Error('Images must have equal, nonempty shapes');
  let signed=0, absolute=0, squared=0;
  for (let i=0;i<target.length;i++) {
    if (!Number.isFinite(prediction[i]) || !Number.isFinite(target[i])) throw Error('Finite image values are required');
    const d=prediction[i]-target[i]; signed+=d; absolute+=Math.abs(d); squared+=d*d;
  }
  return {signed:signed/target.length, mae:absolute/target.length, mse:squared/target.length};
}

export function chain(mu=0, x=1, sigma=1, target=.8) {
  if (![mu,x,sigma,target].every(Number.isFinite) || sigma<=0) throw Error('Finite parameters and positive sigma are required');
  const d=x-mu, u=d/sigma, q=u*u, t=-q/2, weight=Math.exp(t), residual=weight-target, loss=residual*residual;
  const backwards={residual:2*residual,weight:2*residual,t:2*residual*weight,q:-residual*weight,u:-2*residual*weight*u,d:-2*residual*weight*u/sigma,mu:2*residual*weight*u/sigma};
  return {mu,x,sigma,target,d,u,q,t,weight,residual,loss,backwards,gradient:backwards.mu};
}

export function curveLoss(mu, targetMu=1, sigma=1, samples=81) {
  if (![mu,targetMu,sigma].every(Number.isFinite) || sigma<=0 || samples<2) throw Error('Invalid curve parameters');
  const xs=linspace(-4,4,samples); let loss=0, gradient=0;
  for (const x of xs) { const p=bell(x,mu,sigma), r=p-bell(x,targetMu,sigma); loss+=r*r; gradient+=2*r*p*(x-mu)/(sigma*sigma); }
  return {loss:loss/samples,gradient:gradient/samples};
}

export function numericalDerivative(fn, value, h=.01) {
  if (!Number.isFinite(h) || h<=0) throw Error('h must be positive');
  return (fn(value+h)-fn(value-h))/(2*h);
}

export function random(seed=42) {
  let n=seed>>>0;
  return () => { n=(1664525*n+1013904223)>>>0; return n/4294967296; };
}

export function targetImage(kind='shapes', size=32) {
  if (!Number.isInteger(size) || size<2 || size>128) throw Error('Image size must be 2..128');
  if(!['shapes','sunset','checker','flower'].includes(kind))throw Error('Unknown target');
  const result=new Float64Array(size*size*3);
  const layer=(center,scales,angle_deg,color,opacity)=>({center,scales,angle_deg,color,opacity});
  const flower=[layer([0,.65],[.43,.43],0,[1,.65,.06],.98)];
  for(let i=0;i<6;i++){const angle=i*60,a=angle*Math.PI/180;flower.push(layer([.95*Math.cos(a),.65+.95*Math.sin(a)],[.75,.38],(angle+180)%360-180,[.92,.15+.045*i,.38+.035*i],.92));}
  flower.push(layer([-.6,-1.35],[.75,.24],-28,[.15,.55,.22],.92),layer([.58,-1.8],[.7,.24],30,[.12,.48,.2],.92),layer([0,-1.2],[.2,1.45],0,[.12,.45,.2],.95));
  for(let iy=0;iy<size;iy++) for(let ix=0;ix<size;ix++) {
    const x=-4+8*ix/(size-1), y=4-8*iy/(size-1);
    let c=[1,1,1];
    if(kind==='shapes') {
      if((x+1.15)**2+(y-1)**2<1.45**2)c=[.95,.32,.12];
      if(x>.1&&x<2.9&&y>-2.7&&y<-.3)c=[.18,.36,.85];
      if(Math.abs(y-.65*x-1.1)<.27&&x>-.6&&x<3)c=[.12,.58,.27];
    } else if(kind==='sunset') {
      c=y<-.4?[.1+.025*(y+4),.23+.02*(y+4),.42+.02*(y+4)]:[1,.62+.06*y,.38+.05*y];
      if((x-1.2)**2+(y-1.2)**2<.75**2)c=[1,.92,.42];
      if(y<-.4 && Math.abs(x-1.2)<(.55+.14*(-y)))c=[.8,.55,.35];
    } else if(kind==='checker') {
      if(Math.abs(x)<3&&Math.abs(y)<3)c=((Math.floor((x+4)*1.5)+Math.floor((y+4)*1.5))%2)?[.95,.62,.15]:[.2,.27,.65];
    } else {
      c=compositeAt(x,y,flower,[1,1,1]).color;
    }
    const j=(iy*size+ix)*3; for(let k=0;k<3;k++)result[j+k]=clamp(c[k],0,1);
  }
  return result;
}

// Each row contains [center x, center y, scale x, scale y, opacity, red, green, blue].
export function initializeSpots(target,size,count=32,seed=42,mode='grid',scale=.7) {
  if (!Number.isInteger(count) || count<1 || count>128 || target.length!==size*size*3 || !Number.isFinite(scale) || scale<.2 || scale>3) throw Error('Invalid initialization');
  const rng=random(seed), params=new Float64Array(count*8), cols=Math.ceil(Math.sqrt(count));
  const normal=()=>.35*Math.sqrt(-2*Math.log(Math.max(Number.EPSILON,rng())))*Math.cos(2*Math.PI*rng());
  for(let i=0;i<count;i++) {
    const x=mode==='center'?clamp(normal(),-4,4):mode==='random'?-3.7+7.4*rng():-4+8*((i%cols)+.5)/cols;
    const y=mode==='center'?clamp(normal(),-4,4):mode==='random'?-3.7+7.4*rng():4-8*(Math.floor(i/cols)+.5)/cols;
    const ix=clamp(Math.round((x+4)*(size-1)/8),0,size-1), iy=clamp(Math.round((4-y)*(size-1)/8),0,size-1), p=(iy*size+ix)*3;
    params.set([x,y,scale,scale,.65,.9*target[p]+.1,.9*target[p+1]+.1,.9*target[p+2]+.1],i*8);
  }
  return params;
}

export function validateParameters(params) {
  if(!params.length || params.length%8 || params.length>128*8)throw Error('Expected 1..128 rows of eight parameters');
  for(let i=0;i<params.length;i+=8) {
    if(!Array.from(params.slice(i,i+8)).every(Number.isFinite) || params[i+2]<=0 || params[i+3]<=0 || params[i+4]<0 || params[i+4]>1 || Array.from(params.slice(i+5,i+8)).some(c=>c<0||c>1))throw Error('Invalid spot parameters');
  }
}

// Exact back-to-front image derivative for the same front-to-back alpha compositing as lesson 5.
export function renderAndGradient(params,target,size,withGradient=true) {
  validateParameters(params);
  if(!Number.isInteger(size)||size<2||size>128 || target.length!==size*size*3 || !Array.from(target).every(Number.isFinite))throw Error('Invalid target image');
  const n=params.length/8, image=new Float64Array(target.length), grad=new Float64Array(params.length);
  const a=new Float64Array(n),w=new Float64Array(n),trans=new Float64Array(n);
  const dxs=new Float64Array(n),dys=new Float64Array(n); let loss=0;
  for(let iy=0;iy<size;iy++)for(let ix=0;ix<size;ix++) {
    const x=-4+8*ix/(size-1),y=4-8*iy/(size-1),p=(iy*size+ix)*3;
    let T=1,r=0,g=0,b=0;
    for(let i=0;i<n;i++) {
      const j=i*8,dx=x-params[j],dy=y-params[j+1],sx=params[j+2],sy=params[j+3];
      dxs[i]=dx;dys[i]=dy;w[i]=Math.exp(-.5*(dx*dx/(sx*sx)+dy*dy/(sy*sy)));a[i]=params[j+4]*w[i];trans[i]=T;
      r+=T*a[i]*params[j+5];g+=T*a[i]*params[j+6];b+=T*a[i]*params[j+7];T*=1-a[i];
    }
    r+=T;g+=T;b+=T;image[p]=r;image[p+1]=g;image[p+2]=b;
    const dr=r-target[p],dg=g-target[p+1],db=b-target[p+2];loss+=dr*dr+dg*dg+db*db;
    if(!withGradient)continue;
    const gr=2*dr/target.length,gg=2*dg/target.length,gb=2*db/target.length;
    let br=1,bg=1,bb=1;
    for(let i=n-1;i>=0;i--) {
      const j=i*8, sx=params[j+2],sy=params[j+3],cr=params[j+5],cg=params[j+6],cb=params[j+7];
      const da=trans[i]*(gr*(cr-br)+gg*(cg-bg)+gb*(cb-bb));
      grad[j]+=da*a[i]*dxs[i]/(sx*sx);grad[j+1]+=da*a[i]*dys[i]/(sy*sy);
      grad[j+2]+=da*a[i]*dxs[i]*dxs[i]/(sx*sx*sx);grad[j+3]+=da*a[i]*dys[i]*dys[i]/(sy*sy*sy);
      grad[j+4]+=da*w[i];grad[j+5]+=gr*trans[i]*a[i];grad[j+6]+=gg*trans[i]*a[i];grad[j+7]+=gb*trans[i]*a[i];
      br=a[i]*cr+(1-a[i])*br;bg=a[i]*cg+(1-a[i])*bg;bb=a[i]*cb+(1-a[i])*bb;
    }
  }
  return {image,gradient:grad,loss:loss/target.length};
}

export function sgdStep(params,gradient,rate) {
  if(params.length!==gradient.length || !Number.isFinite(rate) || rate<0 || !Array.from(gradient).every(Number.isFinite))throw Error('Invalid update');
  const result=new Float64Array(params.length);
  for(let j=0;j<params.length;j++) {const k=j%8,lo=k<2?-4:k<4?.2:0,hi=k<2?4:k<4?3:1;result[j]=clamp(params[j]-rate*gradient[j],lo,hi);}
  return result;
}

export function exportScene(params) {
  validateParameters(params);
  return {format:'gaussian-splatting-course-2d',version:1,background:[1,1,1],layers:Array.from({length:params.length/8},(_,i)=>{const j=i*8;return {name:`Gaussian ${i+1}`,center:[params[j],params[j+1]],scales:[params[j+2],params[j+3]],angle_deg:0,color:Array.from(params.slice(j+5,j+8)),opacity:params[j+4]};})};
}
