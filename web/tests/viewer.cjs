const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const root=path.join(__dirname,'..'),elements={};
const context2d=new Proxy({}, {get:(o,k)=>o[k]??((...args)=>{for(const a of args)if(typeof a==='number')assert.ok(Number.isFinite(a),'nonfinite canvas argument');})});
function element(id){return elements[id]??=( {style:{},value:id==='speed'?'1':'0',options:[{value:'1'},{value:'2'}],getContext:()=>context2d,getBoundingClientRect:()=>({width:1000,height:520}),setAttribute(){},addEventListener(){}} );}
const s=vm.createContext({document:{getElementById:element},window:{addEventListener(){}},devicePixelRatio:1,requestAnimationFrame(){},console});
for(const f of ['data.js','app.js','geometry.js'])vm.runInContext(fs.readFileSync(path.join(root,f),'utf8'),s);
const run=code=>vm.runInContext(code,s);
run('start()');assert.equal(elements.homePct.textContent,'47.78%');assert.equal(elements.minute.textContent,'42:55');
let count=0;
for(const period of [1,2]){
 run(`selectPeriod(${period})`);
 const n=run('frames.length');assert.equal(n,period===1?545:595);
 for(let i=0;i<n;i++){
  run(`frameIndex=${i};updateReadout()`);
  const f=run('frames[frameIndex]');
  assert.equal(elements.homePct.textContent,(f.home_shot_probability*100).toFixed(2)+'%');
  assert.equal(elements.awayPct.textContent,(f.away_shot_probability*100).toFixed(2)+'%');count++;
 }
 run('frameIndex=frames.length-2;playing=true;carry=0;last=0;animate(1000);animate(1100)');
 assert.equal(run('playing'),false);assert.equal(run('frameIndex'),n-1);
}
assert.throws(()=>run("validate({...data,frames:[{...data.frames[0],home_shot_probability:2},data.frames[1]]})"));
assert.throws(()=>run("validate({...data,frames:[data.frames[0],data.frames[0]]})"));
run("loadData({...data,frames:[{period:1,period_seconds:0,home_shot_probability:0,away_shot_probability:0},{period:1,period_seconds:5,home_shot_probability:0,away_shot_probability:0}]});draw()");
assert.equal(elements.homePct.textContent,'0.00%');
console.log('PASS: all '+count+' frame readouts match source probabilities; half-end stopping, invalid probabilities, duplicate timestamps and all-zero rendering verified.');
