/* UI 接线检查：最小 DOM stub 下执行 <script id="ui">，点遍所有控件（v2：新增 Flow/采样/自检面板） */
const fs=require('fs'),vm=require('vm'),path=require('path');
const html=fs.readFileSync(path.join(__dirname,'index.html'),'utf8');
const eng=html.match(/<script id="engine">([\s\S]*?)<\/script>/)[1];
const ui=html.match(/<script id="ui">([\s\S]*?)<\/script>/)[1];

function makeCtx2d(){
  const calls=[];
  const grad={addColorStop(){}};
  return {
    calls,
    canvas:{width:600,height:400},
    clearRect(){calls.push('clearRect')},
    fillRect(){calls.push('fillRect')},
    beginPath(){calls.push('beginPath')},
    arc(){calls.push('arc')},
    fill(){calls.push('fill')},
    stroke(){calls.push('stroke')},
    strokeRect(){calls.push('strokeRect')},
    moveTo(){},lineTo(){},
    fillText(){calls.push('fillText')},
    measureText(){return{width:10}},
    save(){},restore(){},translate(){},scale(){},
    createLinearGradient(){return grad},
    createImageData(w,h){return{width:w,height:h,data:new Uint8ClampedArray(w*h*4)}},
    getImageData(w,h){return{width:w,height:h,data:new Uint8ClampedArray(w*h*4)}},
    putImageData(){}
  };
}
function makeEl(id){
  const listeners={};
  const el={
    id,value:'0',textContent:'',innerHTML:'',checked:false,
    style:{},_children:[],
    addEventListener(ev,fn){listeners[ev]=fn},
    _fire(ev){if(listeners[ev])listeners[ev]();},
    appendChild(c){el._children.push(c)},
    _ctx:null,
    getContext(){if(!el._ctx)el._ctx=makeCtx2d();return el._ctx;}   /* 缓存：与浏览器同一 canvas 同一 ctx */
  };
  let _html='';
  Object.defineProperty(el,'innerHTML',{
    get(){return _html},
    set(v){
      _html=String(v);
      if(_html==='')el._children.length=0;
      if(/<option/.test(_html)){
        const opts=[],sel=[];
        for(const mm of _html.matchAll(/<option([^>]*)>/g)){
          const attrs=mm[1];
          const vm2=attrs.match(/value="([^"]+)"/);
          if(!vm2)continue;
          opts.push(vm2[1]);
          if(/\bselected\b/.test(attrs))sel.push(vm2[1]);
        }
        el.value=sel[0]!==undefined?sel[0]:(opts[0]!==undefined?opts[0]:'');
      }
    }
  });
  return el;
}
const document={
  getElementById(id){return els[id]||null},
  createElement(tag){return makeEl('_'+tag);}
};
const els={};
['dataset','T','hidden','steps','batch','lr','seed','btnTrain','btnReset','status',
 'loss','sched','cData','cDdpm','cDdim','cFwd','capDdpm','capDdim','momHint',
 /* v2 新增 */
 'selSched','selAct','selMode','inpEta','inpStride','chkClip','btnSample','btnFlow',
 'cFlow','capFlow','stW1','btnInv','invSummary','invBody'].forEach(id=>els[id]=makeEl(id));
els.dataset.innerHTML='<option value="blobs" selected>四簇高斯 blobs</option><option value="roll">瑞士卷 swiss roll</option><option value="moons">双月牙 moons</option>';
els.dataset.value='blobs';
els.selSched.innerHTML='<option value="linear" selected>linear</option><option value="cosine">cosine</option>';
els.selSched.value='linear';
els.selAct.innerHTML='<option value="relu" selected>ReLU</option><option value="silu">SiLU</option>';
els.selAct.value='relu';
els.selMode.innerHTML='<option value="ddpm">DDPM</option><option value="ddim" selected>DDIM</option><option value="flow">Flow</option>';
els.selMode.value='ddim';
els.T.value='16';els.hidden.value='16';els.steps.value='150';els.batch.value='32';els.lr.value='0.002';els.seed.value='42';
els.inpEta.value='0';els.inpStride.value='5';els.chkClip.checked=false;

const ctx={console,Math,document,Uint8ClampedArray,Uint8Array,Float64Array,Int32Array,
  Object,Array,JSON,isFinite,Infinity,globalThis:{}};
ctx.globalThis=ctx;
vm.createContext(ctx);
vm.runInContext(eng,ctx,{filename:'engine.js'});
vm.runInContext(ui,ctx,{filename:'ui.js'});   /* 自动 train() 一次（150 步小配置） */

els.btnTrain._fire('click');
els.btnReset._fire('click');   /* 会用默认大配置重训 —— 覆盖 reset 路径 */
els.dataset.value='roll';
els.btnTrain._fire('click');   /* 换数据集再训一次 */

/* v2：Flow 训练 + 三种范式采样 + 不变量面板 */
els.btnFlow._fire('click');
els.selMode.value='ddpm';els.btnSample._fire('click');
els.selMode.value='ddim';els.btnSample._fire('click');
els.selMode.value='flow';els.btnSample._fire('click');
els.btnInv._fire('click');

let fail=0;
function ok(name,cond,detail){if(!cond){fail++;console.log('FAIL '+name+' :: '+(detail||''));}else console.log('ok '+name);}
ok('status filled',els.status.textContent.indexOf('就绪')>=0,els.status.textContent);
ok('canvas drew',els.loss.getContext('2d').calls.length>0&&els.sched.getContext('2d').calls.length>0);
ok('all 5 canvases used',
  els.cData.getContext('2d').calls.length>0&&
  els.cDdpm.getContext('2d').calls.length>0&&
  els.cDdim.getContext('2d').calls.length>0&&
  els.cFlow.getContext('2d').calls.length>0&&
  els.cFwd.getContext('2d').calls.length>0);
ok('ddim caption',els.capDdpm.textContent.length>0);
ok('flow caption',els.capFlow.textContent.length>0,els.capFlow.textContent);
ok('momHint empty for roll',els.momHint.textContent==='');
els.dataset.value='blobs';
els.btnTrain._fire('click');
ok('momHint filled for blobs',els.momHint.textContent.indexOf('%')>=0,els.momHint.textContent);
ok('dataset select value',els.dataset.value==='blobs');
ok('w1 stats filled',els.stW1.textContent.indexOf('W1')>=0,els.stW1.textContent);
ok('invariant panel ran',els.invBody._children.length>=18&&els.invSummary.textContent.indexOf('结果')>=0,
  'rows='+els.invBody._children.length+' summary='+els.invSummary.textContent);

if(fail){process.exit(1);}
console.log('UICHECK ALL GREEN');