/* 语义探针：采样结构与矩 dump 到 _probe.txt 供人工检查 */
const fs=require('fs'),vm=require('vm'),path=require('path');
const html=fs.readFileSync(path.join(__dirname,'index.html'),'utf8');
const m=html.match(/<script id="engine">([\s\S]*?)<\/script>/);
const ctx={console,Math,TextEncoder,TextDecoder,Uint8Array,Int32Array,Int8Array,Float64Array,Object,Array,JSON,isFinite,Infinity,globalThis:{}};
ctx.globalThis=ctx;vm.createContext(ctx);vm.runInContext(m[1],ctx,{filename:'engine.js'});
const DIFF=ctx.DIFF;

const L=[];
const data=DIFF.data.blobs(DIFF.mulberry32(42),1200);
const model=DIFF.train(data,{T:50,hidden:32,steps:1200,batch:32,lr:0.002,seed:42});

L.push('=== loss 轨迹（每 100 步） ===');
model.lossHist.forEach((v,i)=>{if(i%100===0||i===model.lossHist.length-1)L.push('step '+i+'  '+v.toFixed(5));});

L.push('');
L.push('=== 调度 ===');
[1,10,25,40,50].forEach(t=>L.push(`t=${t}  beta=${model.sched.betas[t-1].toFixed(4)}  abar=${model.sched.abar[t].toFixed(4)}`));

L.push('');
L.push('=== 矩对比（DDPM 1500 样本 vs 数据） ===');
const pts=DIFF.sampleDDPM(model,1500,DIFF.mulberry32(99));
const ms=DIFF.moments(pts),md=DIFF.moments(data);
L.push(`        样本        数据`);
L.push(`mean x  ${ms.mx.toFixed(3)}       ${md.mx.toFixed(3)}`);
L.push(`mean y  ${ms.my.toFixed(3)}       ${md.my.toFixed(3)}`);
L.push(`var x   ${ms.vxx.toFixed(3)}       ${md.vxx.toFixed(3)}`);
L.push(`var y   ${ms.vyy.toFixed(3)}       ${md.vyy.toFixed(3)}`);

L.push('');
L.push('=== 簇占比（最近中心分配） ===');
const C=[[-1.8,-1.8],[1.8,-1.8],[-1.8,1.8],[1.8,1.8]];
function clusterShare(sample){
  const counts=[0,0,0,0];
  sample.forEach(p=>{
    let best=0,bd=1e9;
    C.forEach((c,ci)=>{const d=(p[0]-c[0])**2+(p[1]-c[1])**2;if(d<bd){bd=d;best=ci;}});
    counts[best]++;
  });
  return counts.map(c=>(100*c/sample.length).toFixed(1)+'%').join('  ');
}
L.push('数据:  '+clusterShare(data));
L.push('DDPM:  '+clusterShare(pts));
L.push('DDIM:  '+clusterShare(DIFF.sampleDDIM(model,1500,0,DIFF.mulberry32(98))));
L.push('DDIM η=1: '+clusterShare(DIFF.sampleDDIM(model,1500,1,DIFF.mulberry32(98))));

L.push('');
L.push('=== ASCII 密度图（DDPM 1500 样本，60x24 格） ===');
const W=60,H=24,grid=[];
for(let r=0;r<H;r++)grid.push(new Array(W).fill(0));
pts.forEach(p=>{
  const cx=Math.floor((p[0]+3.2)/6.4*W),cy=Math.floor((3.2-p[1])/6.4*H);
  if(cx>=0&&cx<W&&cy>=0&&cy<H)grid[cy][cx]++;
});
const ramp=' .:*#@';
grid.forEach(row=>L.push(row.map(v=>ramp[Math.min(5,Math.floor(v/6))]).join('')));

fs.writeFileSync(path.join(__dirname,'_probe.txt'),L.join('\n'));
console.log('probe written');
