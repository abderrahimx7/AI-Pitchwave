'use strict';
const $ = id => document.getElementById(id);
const canvas = $('pitch'), ctx = canvas.getContext('2d'), timeline = $('time');
const clamp = (v,a,b) => Math.max(a,Math.min(b,v));
let data, frames=[], frameIndex=0, minute=0, angle=-.55, tilt=.70, zoom=1;
let playing=false,last=0,carry=0,drag=null,W=0,H=0,chartScale=.1;

// No inferred spatial locations: fixed decorative lobes, weighted by the supplied risks.
function fields(x,y){
  const f=frames[frameIndex];
  const g=cx=>Math.exp(-((x-cx)**2/.32+y*y/.40));
  return [f.home_shot_probability*g(-.48),f.away_shot_probability*g(.48)];
}
function height(x,y){const [a,b]=fields(x,y);return 1.5*(a+b);}
function formatTime(seconds){return Math.floor(seconds/60)+':'+String(Math.floor(seconds%60)).padStart(2,'0');}
function validate(input){
  if(!input||input.schema!=='pitchwave-shot-risk-v1')throw new Error('Expected a pitchwave-shot-risk-v1 JSON export.');
  for(const key of ['home','away','source','model'])if(typeof input[key]!=='string'||!input[key].trim()||input[key].length>120)throw new Error('Invalid '+key+'.');
  if(!Number.isSafeInteger(input.match_id)||input.match_id<0)throw new Error('Invalid match ID.');
  if(!Number.isFinite(input.horizon_seconds)||input.horizon_seconds<=0||input.horizon_seconds>120)throw new Error('Invalid prediction horizon.');
  if(!Array.isArray(input.frames)||input.frames.length<2||input.frames.length>100000)throw new Error('Expected 2–100,000 prediction frames.');
  let previous=null;
  for(const f of input.frames){
    if(![1,2].includes(f.period)||!Number.isFinite(f.period_seconds)||f.period_seconds<0||f.period_seconds>7200)throw new Error('Invalid period or time.');
    for(const key of ['home_shot_probability','away_shot_probability'])if(!Number.isFinite(f[key])||f[key]<0||f[key]>1)throw new Error('Probabilities must be numeric values between 0 and 1.');
    if(previous&&(f.period<previous.period||(f.period===previous.period&&f.period_seconds<=previous.period_seconds)))throw new Error('Frames must have unique, increasing times within ordered periods.');
    previous=f;
  }
  return input;
}
function setPlaying(value){playing=value;carry=0;$('play').textContent=value?'Ⅱ':'▶';$('play').setAttribute('aria-label',value?'Pause match':'Play match');}
function selectPeriod(period){
  setPlaying(false);frames=data.frames.filter(f=>f.period===Number(period));frameIndex=0;
  timeline.max=frames.length-1;timeline.value=0;
  chartScale=Math.max(.1,Math.ceil(frames.reduce((max,f)=>Math.max(max,f.home_shot_probability,f.away_shot_probability),0)*10)/10);
  $('startTick').textContent=formatTime(frames[0].period_seconds);
  $('endTick').textContent=formatTime(frames[frames.length-1].period_seconds)+' / HALF CLOCK';
  $('scaleTick').textContent='CHART RANGE: 0–'+Math.round(chartScale*100)+'% PER TEAM';
  draw();
}
function loadData(input){
  validate(input);data=input;
  for(const id of ['homeName','homeLegend','homeRiskName'])$(id).textContent=data.home;
  for(const id of ['awayName','awayLegend','awayRiskName'])$(id).textContent=data.away;
  $('datasetLabel').textContent='Match '+data.match_id+' · '+data.frames.length.toLocaleString()+' samples';
  $('riskHeading').textContent='SHOT IN THE NEXT '+data.horizon_seconds+' SECONDS';
  $('horizonText').textContent=data.horizon_seconds;
  $('coverage').textContent=data.frames.length+' supplied samples across '+new Set(data.frames.map(f=>f.period)).size+' halves. The half clock includes stoppage time. Only supplied times are shown; no extra time or shootout is available. Playback stops at the last sample of each half.';
  for(const option of $('period').options)option.disabled=!data.frames.some(f=>f.period===Number(option.value));
  $('period').value=String(data.frames[0].period);selectPeriod(data.frames[0].period);
}
function updateReadout(){
  const f=frames[frameIndex];
  $('minute').textContent=formatTime(f.period_seconds);
  $('periodLabel').textContent=(f.period===1?'FIRST':'SECOND')+' HALF · PERIOD CLOCK';
  $('timeLabel').textContent=(frameIndex+1)+' / '+frames.length;
  $('homePct').textContent=(f.home_shot_probability*100).toFixed(2)+'%';
  $('awayPct').textContent=(f.away_shot_probability*100).toFixed(2)+'%';
  $('homeMeter').style.width=f.home_shot_probability*100+'%';
  $('awayMeter').style.width=f.away_shot_probability*100+'%';
  timeline.value=frameIndex;
}
function drawMomentum(){
  const c=$('momentum'),r=c.getBoundingClientRect(),d=Math.min(devicePixelRatio||1,2);
  c.width=r.width*d;c.height=60*d;const g=c.getContext('2d');g.scale(d,d);
  const w=r.width,h=60;
  g.strokeStyle='#41434b';g.beginPath();g.moveTo(0,h/2);g.lineTo(w,h/2);g.stroke();
  const first=frames[0].period_seconds,span=Math.max(1,frames[frames.length-1].period_seconds-first);
  for(const [key,color,sign] of [['home_shot_probability','#d95272',-1],['away_shot_probability','#f0bb41',1]]){
    g.beginPath();frames.forEach((f,i)=>{const x=(f.period_seconds-first)/span*w,y=h/2+sign*f[key]/chartScale*27;i?g.lineTo(x,y):g.moveTo(x,y);});
    g.strokeStyle=color;g.lineWidth=1.2;g.stroke();
  }
  g.fillStyle='#fff';g.fillRect((frames[frameIndex].period_seconds-first)/span*(w-1),0,1,60);
}
function resize(){const r=canvas.getBoundingClientRect(),d=Math.min(devicePixelRatio||1,2);W=r.width;H=r.height;canvas.width=W*d;canvas.height=H*d;ctx.setTransform(d,0,0,d,0,0);if(frames.length)draw();}
function peak(side){setPlaying(false);const key=side+'_shot_probability';frameIndex=frames.reduce((best,f,i)=>f[key]>frames[best][key]?i:best,0);draw();}
function exportImage(){
  const c=document.createElement('canvas');c.width=1600;c.height=1200;const g=c.getContext('2d');
  g.fillStyle='#131419';g.fillRect(0,0,1600,1200);g.fillStyle='#eeeef1';g.font='bold 48px Arial';g.fillText('pitchwave / '+data.home+' × '+data.away,70,90);
  const f=frames[frameIndex];g.font='24px Arial';g.fillStyle='#adb4c1';g.fillText('Half '+f.period+' · '+formatTime(f.period_seconds)+' (half clock) · Shot in next '+data.horizon_seconds+' seconds',70,145);
  const scale=Math.min(1500/canvas.width,740/canvas.height),dw=canvas.width*scale,dh=canvas.height*scale;g.drawImage(canvas,(1600-dw)/2,175+(740-dh)/2,dw,dh);
  g.font='30px Arial';g.fillStyle='#e56a85';g.fillText(data.home+': '+(f.home_shot_probability*100).toFixed(2)+'%',70,980);g.fillStyle='#f0bb41';g.fillText(data.away+': '+(f.away_shot_probability*100).toFixed(2)+'%',850,980);
  g.font='20px Arial';g.fillStyle='#b0b7c3';g.fillText('Model estimates · Illustrative surface, NOT a spatial prediction',70,1050);g.fillText('Data: '+data.source+' · Model export: Abderrahim · AI-assisted viewer',70,1090);g.fillText('Independent probabilities; not possession or win probabilities.',70,1130);
  const a=document.createElement('a');a.download='Pitchwave-'+data.match_id+'-H'+f.period+'-'+f.period_seconds+'.png';a.href=c.toDataURL('image/png');a.click();
}
function start(){
  resize();loadData(window.PITCHWAVE_DATA);peak('home');
  $('play').onclick=()=>{if(frameIndex===frames.length-1)frameIndex=0;setPlaying(!playing);draw();};
  timeline.oninput=()=>{setPlaying(false);frameIndex=Number(timeline.value);draw();};
  $('period').onchange=()=>selectPeriod($('period').value);
  $('peakHome').onclick=()=>peak('home');$('peakAway').onclick=()=>peak('away');
  $('reset').onclick=()=>{angle=-.55;tilt=.70;zoom=1;draw();};
  canvas.onpointerdown=e=>{drag={x:e.clientX,y:e.clientY};canvas.setPointerCapture(e.pointerId);};
  canvas.onpointermove=e=>{if(!drag)return;angle+=(e.clientX-drag.x)*.006;tilt=clamp(tilt+(e.clientY-drag.y)*.004,.2,1.25);drag={x:e.clientX,y:e.clientY};draw();};
  canvas.onpointerup=canvas.onpointercancel=()=>drag=null;
  canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=clamp(zoom-e.deltaY*.001,.65,1.5);draw();},{passive:false});
  canvas.onkeydown=e=>{if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','-','='].includes(e.key))return;e.preventDefault();if(e.key==='ArrowLeft')angle-=.1;if(e.key==='ArrowRight')angle+=.1;if(e.key==='ArrowUp')tilt=clamp(tilt-.1,.2,1.25);if(e.key==='ArrowDown')tilt=clamp(tilt+.1,.2,1.25);if(['+','='].includes(e.key))zoom=clamp(zoom+.1,.65,1.5);if(e.key==='-')zoom=clamp(zoom-.1,.65,1.5);draw();};
  $('about').onclick=()=>{$('method').open=true;$('method').scrollIntoView({behavior:'smooth'});};$('export').onclick=exportImage;
  $('import').onchange=async e=>{const file=e.target.files[0];if(!file)return;try{if(file.size>10*1024*1024)throw new Error('File is larger than 10 MB.');const input=JSON.parse(await file.text());loadData(input);$('status').textContent='Loaded '+data.home+' × '+data.away+' locally.';}catch(error){$('status').textContent='Import failed: '+error.message;}e.target.value='';};
  window.addEventListener('resize',resize);requestAnimationFrame(animate);
}
function animate(now){
  if(playing){carry+=Math.min((now-last)/1000,.1)*25*Number($('speed').value);let changed=false;
    while(frameIndex<frames.length-1){const delta=frames[frameIndex+1].period_seconds-frames[frameIndex].period_seconds;if(carry<delta)break;carry-=delta;frameIndex++;changed=true;}
    if(frameIndex===frames.length-1)setPlaying(false);if(changed)draw();
  }last=now;requestAnimationFrame(animate);
}
