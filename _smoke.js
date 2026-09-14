/* 无头验证：diffusion-forge 引擎 */
const fs=require('fs'),vm=require('vm'),path=require('path');
const html=fs.readFileSync(path.join(__dirname,'index.html'),'utf8');
const m=html.match(/<script id="engine">([\s\S]*?)<\/script>/);
const ctx={console,Math,TextEncoder,TextDecoder,Uint8Array,Int32Array,Int8Array,Float64Array,Object,Array,JSON,isFinite,Infinity,globalThis:{}};
ctx.globalThis=ctx;vm.createContext(ctx);vm.runInContext(m[1],ctx,{filename:'engine.js'});
const DIFF=ctx.DIFF;

let pass=0,fail=0,fails=[];
function ok(name,cond,detail){
  if(cond){pass++;}
  else{fail++;fails.push(name+(detail?' :: '+detail:''));}
}

/* 1. 调度合法性 */
const sched=DIFF.makeBetas(50);
let ok1=sched.betas.every(b=>b>0&&b<1);
let dec=true;for(let t=1;t<=50;t++)if(sched.abar[t]>=sched.abar[t-1]||sched.abar[t]<=0)dec=false;
ok('betas in (0,1)',ok1);
ok('abar decreasing & positive',dec,'abar[50]='+sched.abar[50].toFixed(4));

/* 2. 前向闭式解 vs 逐步迭代：矩匹配（两条独立算法路径） */
const rng=DIFF.mulberry32(11);
const x0=[0.7,-1.3];
const N=6000;
let mxC=0,myC=0,mxI=0,myI=0;
for(let i=0;i<N;i++){
  const a=DIFF.fwdClosed(x0,20,sched,rng),b=DIFF.fwdIter(x0,20,sched,rng);
  mxC+=a[0]/N;myC+=a[1]/N;mxI+=b[0]/N;myI+=b[1]/N;
}
const s20=Math.sqrt(sched.abar[20]);
const sd=Math.sqrt((1-sched.abar[20])/N)*3;
ok('fwd closed vs iter mean',Math.abs(mxC-mxI)<sd&&Math.abs(myC-myI)<sd,
  `closed(${mxC.toFixed(3)},${myC.toFixed(3)}) iter(${mxI.toFixed(3)},${myI.toFixed(3)}) tol±${sd.toFixed(3)}`);
ok('fwd mean ≈ sqrt(abar)x0',Math.abs(mxC-s20*x0[0])<sd&&Math.abs(myC-s20*x0[1])<sd,
  `expect(${(s20*x0[0]).toFixed(3)},${(s20*x0[1]).toFixed(3)})`);

/* 3. 后验代数恒等式：x_t = sqrt(abar_t)x0 代入 posterior mean 应还原 sqrt(abar_{t-1})x0 */
let postOk=true,worst=0;
for(let t=1;t<=50;t++){
  const p=DIFF.posterior(t,sched);
  for(const v of [[1.7,-0.9],[-2.2,2.5],[0,0]]){
    const xt=[Math.sqrt(sched.abar[t])*v[0],Math.sqrt(sched.abar[t])*v[1]];
    const mean=[p.c0*v[0]+p.ct*xt[0],p.c0*v[1]+p.ct*xt[1]];
    const e=Math.max(Math.abs(mean[0]-Math.sqrt(sched.abar[t-1])*v[0]),
                     Math.abs(mean[1]-Math.sqrt(sched.abar[t-1])*v[1]));
    worst=Math.max(worst,e);
    if(!(e<1e-12))postOk=false;
  }
  if(!(p.var>=0&&isFinite(p.var)))postOk=false;
  if(t>=2&&!(p.var>0))postOk=false;   /* t=1 时 ᾱ₀=1 ⇒ var=0，数学上正确 */
}
ok('posterior algebra identity',postOk,'worst='+worst.toExponential(2));

/* 4. 梯度检验 */
const gc=DIFF.gradCheck(3,1e-6);
ok('gradCheck',gc<1e-5,'maxRel='+gc);

/* 5. 训练确定性（短程 bit 级） */
const data=DIFF.data.blobs(DIFF.mulberry32(42),1200);
const t1=DIFF.train(data,{T:50,hidden:16,steps:60,seed:7,batch:32});
const t2=DIFF.train(data,{T:50,hidden:16,steps:60,seed:7,batch:32});
let dmax=0;
for(let l=0;l<t1.net.W.length;l++)for(let i=0;i<t1.net.W[l].length;i++)
  dmax=Math.max(dmax,Math.abs(t1.net.W[l][i]-t2.net.W[l][i]));
ok('train determinism',dmax===0,'maxdiff='+dmax);

/* 6. 训练收敛 + 采样产出四簇（主不变量） */
const model=DIFF.train(data,{T:50,hidden:32,steps:1200,batch:32,lr:0.002,seed:42});
const h=model.lossHist;
const first5=h.slice(0,5).reduce((a,b)=>a+b)/5,last5=h.slice(-5).reduce((a,b)=>a+b)/5;
ok('loss decreases',last5<first5,'first5='+first5.toFixed(4)+' last5='+last5.toFixed(4));

const rngS=DIFF.mulberry32(99);
const pts=DIFF.sampleDDPM(model,800,rngS);
const C=[[-1.8,-1.8],[1.8,-1.8],[-1.8,1.8],[1.8,1.8]];
const counts=[0,0,0,0];
let distSum=0;
pts.forEach(p=>{
  let best=0,bd=1e9;
  C.forEach((c,ci)=>{const d=(p[0]-c[0])**2+(p[1]-c[1])**2;if(d<bd){bd=d;best=ci;}});
  counts[best]++;distSum+=Math.sqrt(bd);
});
const shares=counts.map(c=>c/800);
ok('ddpm covers all 4 blobs',shares.every(s=>s>0.12&&s<0.45),JSON.stringify(shares.map(s=>s.toFixed(3))));
const dataDist=DIFF.data.blobs(DIFF.mulberry32(5),800).map(p=>{
  let bd=1e9;C.forEach(c=>{const d=(p[0]-c[0])**2+(p[1]-c[1])**2;if(d<bd)bd=d;});return Math.sqrt(bd);
}).reduce((a,b)=>a+b,0)/800;
const sampDist=distSum/800;
ok('sample spread ≈ data spread',sampDist<2.5*dataDist&&sampDist>0.3*dataDist,
  'samp='+sampDist.toFixed(3)+' data='+dataDist.toFixed(3));

/* 7. DDIM：eta=0 确定性，eta>0 随机 */
const d1=DIFF.sampleDDIM(model,50,0,DIFF.mulberry32(1));
const d2=DIFF.sampleDDIM(model,50,0,DIFF.mulberry32(1));
let dd=0;
for(let i=0;i<50;i++)for(let j=0;j<2;j++)dd=Math.max(dd,Math.abs(d1[i][j]-d2[i][j]));
ok('ddim eta=0 deterministic',dd===0,'maxdiff='+dd);
const d3=DIFF.sampleDDIM(model,50,1,DIFF.mulberry32(1));
let dd3=0;
for(let i=0;i<50;i++)dd3=Math.max(dd3,Math.abs(d1[i][0]-d3[i][0]));
ok('ddim eta=1 stochastic',dd3>0,'maxdiff='+dd3);
const countsD=[0,0,0,0];
d1.forEach(p=>{
  let best=0,bd=1e9;
  C.forEach((c,ci)=>{const d=(p[0]-c[0])**2+(p[1]-c[1])**2;if(d<bd){bd=d;best=ci;}});
  countsD[best]++;
});
ok('ddim covers all 4 blobs',countsD.every(c=>c>=5),JSON.stringify(countsD));

/* 8. 矩有限性 + 采样点有界 */
ok('samples finite & bounded',pts.every(p=>isFinite(p[0])&&isFinite(p[1])&&Math.abs(p[0])<30&&Math.abs(p[1])<30));

/* 9. 边界：T=1、batch=1、空数据 */
const m1=DIFF.train(data.slice(0,1),{T:5,hidden:8,steps:20,batch:1,seed:1});
ok('batch=1 single-sample trains',!!m1&&m1.lossHist.length===20&&m1.lossHist.every(x=>isFinite(x)));
const sT1=DIFF.sampleDDPM(m1,5,DIFF.mulberry32(2));
ok('T=5 tiny model samples',sT1.length===5&&sT1.every(p=>isFinite(p[0])));

/* 10. moments 辅助 */
const mo=DIFF.moments([[1,0],[-1,0],[1,0],[-1,0]]);
ok('moments helper',Math.abs(mo.mx)<1e-15&&Math.abs(mo.vxx-1)<1e-12&&Math.abs(mo.vxy)<1e-12);

/* 11. unicode/引擎冒烟：makeBetas T=1 */
const s1=DIFF.makeBetas(1);
ok('T=1 schedule',s1.betas.length===1&&s1.abar.length===2&&s1.abar[1]<1);

fs.writeFileSync(path.join(__dirname,'_smoke.log'),
  `PASS ${pass} / ${pass+fail}\n`+(fail?'FAIL '+fails.join(' | '):'ALL GREEN')+'\n');
console.log(fail?`FAIL ${fail}: `+fails.join(' | '):`ALL GREEN ${pass}/${pass+fail}`);
process.exit(fail?1:0);
