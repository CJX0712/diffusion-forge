/* 无头验证 v2：引擎内 18 项新增不变量
 * Flow Matching / cosine 调度 / W1 生成质量 / SiLU / AdamW / DDIM 跳步 / Euler 积分阶数
 * 与浏览器「运行引擎不变量」面板、以及 index.html 内 DIFF.invariants 是同一套判定逻辑。
 */
const fs=require('fs'),vm=require('vm'),path=require('path');
const html=fs.readFileSync(path.join(__dirname,'index.html'),'utf8');
const m=html.match(/<script id="engine">([\s\S]*?)<\/script>/);
if(!m){console.error('FAIL: engine script not found');process.exit(1);}
const ctx={console,Math,TextEncoder,TextDecoder,Uint8Array,Int32Array,Int8Array,Float64Array,Object,Array,JSON,isFinite,Infinity,globalThis:{}};
ctx.globalThis=ctx;vm.createContext(ctx);vm.runInContext(m[1],ctx,{filename:'engine.js'});
const DIFF=ctx.DIFF;

let pass=0,fail=0,fails=[];
const res=DIFF.runInvariants(function(msg){
  console.log(msg);
  if(msg.indexOf('❌')===0){fail++;fails.push(msg);}else pass++;
});

const summary='SMOKE2 '+res.passed+' / '+res.total+(fail?'  FAIL '+fails.length:'  ALL GREEN');
console.log('=== '+summary+' ===');
fs.writeFileSync(path.join(__dirname,'_smoke2.log'),
  'PASS '+res.passed+' / '+res.total+'\n'+(fail?'FAIL '+fails.join(' | '):'ALL GREEN')+'\n');
process.exit(fail?1:0);
