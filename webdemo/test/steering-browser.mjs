// Real Chromium worker + page integration. Install playwright or set PLAYWRIGHT_MODULE.
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { createReadStream, existsSync } from 'node:fs';
import { readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { join, extname, resolve } from 'node:path';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = fileURLToPath(new URL('../',import.meta.url));
const requests=[];
const mime={'.html':'text/html','.js':'text/javascript','.mjs':'text/javascript','.wasm':'application/wasm','.json':'application/json','.npy':'application/octet-stream','.onnx':'application/octet-stream','.emb':'application/octet-stream'};
const server=createServer((req,res)=>{
  const path=new URL(req.url,'http://localhost').pathname; requests.push(path);
  res.setHeader('Cross-Origin-Opener-Policy','same-origin'); res.setHeader('Cross-Origin-Embedder-Policy','require-corp');
  if(path==='/test/harness'){res.setHeader('Content-Type','text/html');res.end('<!doctype html><title>WASM integration</title>');return;}
  const file=path==='/test/source.wav' ? resolve(root,'../voices/example.wav') : resolve(root,'.'+(path==='/'?'/index.html':path));
  if((path!=='/test/source.wav'&&!file.startsWith(root))||!existsSync(file)){res.statusCode=404;res.end('Not found');return;}
  res.setHeader('Content-Type',mime[extname(file)]||'application/octet-stream');createReadStream(file).pipe(res);
});
await new Promise(r=>server.listen(0,'127.0.0.1',r));
const url=`http://127.0.0.1:${server.address().port}`;
const browser=await chromium.launch({headless:true});
const results=[];
try {
  for(const config of [{engine:'native',variant:'fixed',soura:false},{engine:'native',variant:'growth',soura:true,prepareCaches:true},{engine:'ts',soura:true}]) {
    const page=await browser.newPage();
    await page.goto(url+'/test/harness');
    await page.evaluate(({engine})=>{
      window.events=[];window.worker=new Worker(engine==='ts'?'/src/tts-worker.js':'/src/tts-worker-native.js',{type:'module'});
      worker.onmessage=({data})=>events.push(data);
      worker.onerror=e=>events.push({type:'error',message:e.message});
      window.call=async(payload,done='ack')=>{
        const start=events.length,id=String(Math.random()); worker.postMessage({...payload,id});
        const until=Date.now()+120000;
        while(Date.now()<until){
          const fresh=events.slice(start); const error=fresh.find(e=>e.type==='error');if(error)throw Error(JSON.stringify(error));
          if(fresh.some(e=>e.type===done&&(done!=='ack'||e.id===id)))return fresh;
          await new Promise(r=>setTimeout(r,10));
        }throw Error('Worker request timed out: '+payload.type);
      };
    },config);
    const first=requests.length;
    await page.evaluate(async cfg=>call({type:'init',...cfg,modelsUrl:location.origin+'/models',pool:2,threads:2,spin:1} ,'ready'),config);
    if(!config.soura) assert(!requests.slice(first).some(p=>p.includes('soura')),'disabled startup fetched optional assets');
    await page.evaluate(()=>call({type:'loadPresets',presets:[{key:'alba',label:'Alba',url:'/presets/alba.emb'}]}));
    await page.evaluate(()=>call({type:'prewarm',voiceKey:'alba'}));
    const speak=async controls=>page.evaluate(async controls=>{
      const events=await call({type:'speak',text:'There you are. It is good to see you again.',voiceKey:'alba',temperature:.6,seed:31,...controls});
      const chunks=events.filter(e=>e.type==='chunk').map(e=>new Float32Array(e.samples));
      return {count:chunks.reduce((n,c)=>n+c.length,0),finite:chunks.every(c=>c.every(Number.isFinite)),peak:Math.max(...chunks.map(c=>c.reduce((m,v)=>Math.max(m,Math.abs(v)),0)))};
    },controls);
    const neutral=await speak({emotion:'neutral'});assert(neutral.count>24000&&neutral.finite&&neutral.peak>.001);
    if(config.soura){
      for(const label of ['angry','happy','sad']){const audio=await speak({emotion:label,intensity:.8});assert(audio.count>24000&&audio.finite);}
      await page.evaluate(async()=>call({type:'loadVectors',buffer:await(await fetch('/models/soura_vectors.derived.npy')).arrayBuffer()}));
      const custom=await speak({emotion:'happy',intensity:.9});assert(custom.count>24000&&custom.finite);
      await page.evaluate(async()=>{
        const start=events.length;
        worker.postMessage({type:'speak',text:'This is a long sentence that we will interrupt. '.repeat(30),voiceKey:'alba',seed:31,emotion:'angry',intensity:.8,id:'interrupted'});
        while(!events.slice(start).some(e=>e.type==='chunk'))await new Promise(r=>setTimeout(r,10));
        worker.postMessage({type:'stop'});
        while(!events.slice(start).some(e=>e.type==='ack'&&e.id==='interrupted'))await new Promise(r=>setTimeout(r,10));
      });
      assert((await speak({emotion:'neutral'})).finite);
      if(config.prepareCaches){
        // Create a cloned-voice record using the actual preset embedding, then
        // transfer its cache back through the restore protocol. No fake KV data.
        await page.evaluate(async()=>{
          const b=await(await fetch('/presets/alba.emb')).arrayBuffer(),d=new DataView(b),n=d.getInt32(4,true);
          const dims=Array.from({length:n},(_,i)=>Number(d.getBigInt64(8+i*8,true)));
          await call({type:'restoreVoice',key:'copy',label:'Copy',embDims:dims,emb:b.slice(8+n*8)});
          const out=await call({type:'prewarmVoice',voiceKey:'copy'});
          const cache=out.find(e=>e.type==='voiceCaches');if(!cache?.kvCache?.byteLength)throw Error('No transferred KV cache');
          await call({type:'dropVoice',key:'copy'});
          await call({type:'restoreVoice',key:'copy',label:'Restored',embDims:dims,emb:b.slice(8+n*8),embCache:cache.embCache,kvCache:cache.kvCache,cacheVersion:cache.cacheVersion});
          await call({type:'speak',voiceKey:'copy',text:'Hello again.',seed:31});
        });
      }
    }
    results.push({...config,passed:true,neutral});console.log(JSON.stringify(results.at(-1)));
    await page.close();
  }
  const page=await browser.newPage({viewport:{width:1200,height:950}});
  await page.goto(url+'/');
  await page.waitForFunction(()=>window.PKTTS?.state.ready && PKTTS.state.activeVoice==='varkos',null,{timeout:120000});
  assert.equal(await page.locator('#steering-enabled').isChecked(),true);
  assert.equal(await page.locator('#voice-options').getAttribute('open'),null);
  assert.deepEqual(await page.locator('#emotion option').allTextContents(),['Neutral','Preset 1','Preset 2','Preset 3','Preset 4','Preset 5']);
  assert.deepEqual(await page.locator('#vector-set option').allTextContents(),['Default']);
  assert.equal(await page.locator('#vector-set').inputValue(),'default');
  assert.equal(await page.evaluate(()=>location.search),'');
  await page.locator('#voice-options summary').click();
  await page.evaluate(async()=>PKTTS.fireEvent('VectorsRequest',await(await fetch('/models/soura_vectors.derived.npy')).arrayBuffer()));
  await page.waitForFunction(()=>document.getElementById('vector-status').textContent.includes('Custom vectors loaded'));
  await page.selectOption('#emotion','happy');await page.fill('#sampling-seed','31');
  await page.locator('#emotion-strength').fill('0.65');
  assert.equal(await page.locator('#emotion-intensity').inputValue(),'0.65');
  assert.equal(await page.evaluate(()=>PKTTS.state.intensity),0.65);
  await page.fill('#text','Hello there. It is good to see you again.');
  await page.click('#speak');
  await page.waitForFunction(()=>window.PKTTS?.state.lastAudio?.length>24000,null,{timeout:60000});
  if(process.env.PTT_SCREENSHOT)await page.screenshot({path:process.env.PTT_SCREENSHOT,fullPage:true});
  // Disabling must survive reload without fetching optional steering assets.
  const offStart=requests.length;
  await page.locator('#steering-enabled').uncheck();
  await page.waitForURL('**/?soura=0');
  await page.waitForFunction(()=>PKTTS.state.ready&&PKTTS.state.activeVoice==='varkos',null,{timeout:120000});
  assert.equal(await page.locator('#steering-enabled').isChecked(),false);
  assert.equal(await page.locator('#emotion').isDisabled(),true);
  assert(!requests.slice(offStart).some(p=>p.includes('soura')));
  await page.fill('#text','Hello again.');await page.click('#speak');
  await page.waitForFunction(()=>PKTTS.state.lastAudio?.length>12000,null,{timeout:60000});
  await page.locator('#voice-options summary').click();
  await page.locator('#steering-enabled').check();
  await page.waitForURL(url+'/');
  await page.waitForFunction(()=>PKTTS.state.ready&&PKTTS.state.activeVoice==='varkos',null,{timeout:120000});
  assert.equal(await page.locator('#steering-enabled').isChecked(),true);
  results.push({pageDefaultsAndToggle:true,passed:true});console.log('Plain URL, Varkos, preset labels, controls, playback and off/on reload: PASS');
  await page.close();
  // Exercise the mobile restart path with real audio and OPFS, in Chromium's
  // mobile emulation. This is not a substitute for a physical Safari test.
  const mobile=await browser.newPage({viewport:{width:390,height:844},isMobile:true,hasTouch:true,
    userAgent:'Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148'});
  await mobile.goto(url+'/?soura=1&prepareCaches=1&variant=growth');
  await mobile.waitForFunction(()=>window.PKTTS?.state.ready,null,{timeout:120000});
  await mobile.evaluate(async()=>{
    window.cloneErrors=[];PKTTS.listenFor('EngineError',e=>cloneErrors.push(e));
    const context=new OfflineAudioContext(1,1,24000);
    const audio=await context.decodeAudioData(await(await fetch('/test/source.wav')).arrayBuffer());
    PKTTS.fireEvent('CloneRequest',{key:'parity-test',label:'Parity test',audio:audio.getChannelData(0).slice(0,72000).buffer});
  });
  await mobile.waitForFunction(()=>window.cloneErrors.length || (PKTTS.state.ready&&PKTTS.state.voices.some(v=>v.key==='parity-test')&&PKTTS.state.voicePayloads['parity-test']?.kvCache),null,{timeout:120000});
  assert.deepEqual(await mobile.evaluate(()=>cloneErrors),[]);
  await mobile.waitForFunction(async()=>{
    const dir=await(await navigator.storage.getDirectory()).getDirectoryHandle('voices');
    try{const file=await(await dir.getFileHandle('parity-test.kv')).getFile();return file.size>0;}catch{return false;}
  },null,{timeout:30000});
  await mobile.reload();
  await mobile.waitForFunction(()=>PKTTS.state.ready&&PKTTS.state.voices.some(v=>v.key==='parity-test'),null,{timeout:120000});
  assert(await mobile.evaluate(()=>PKTTS.state.voicePayloads['parity-test'].kvCache.byteLength>0));
  await mobile.evaluate(()=>PKTTS.fireEvent('SpeakRequest',{text:'Hello again.',voiceKey:'parity-test',emotion:'happy',intensity:.8,seed:31}));
  await mobile.waitForFunction(()=>PKTTS.state.lastAudio?.length>12000,null,{timeout:60000});
  await mobile.locator('#voice-options summary').click();
  assert(await mobile.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  if(process.env.PTT_SCREENSHOT)await mobile.screenshot({path:process.env.PTT_SCREENSHOT.replace('.png','-mobile.png'),fullPage:true});
  results.push({mobileEmulationCloneAndOPFS:true,passed:true});console.log('Mobile-emulation cloning, cache persistence/restoration, playback: PASS');
  if(process.env.PTT_RESULTS)await writeFile(process.env.PTT_RESULTS,JSON.stringify(results,null,2));
} finally {await browser.close();await new Promise(r=>server.close(r));}
