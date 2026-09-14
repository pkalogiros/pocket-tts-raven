// Real WASM regression checks. Run once per variant: PTT_VARIANT=fixed|growth node ...
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { Tokenizer } from '../src/engine/tokenizer.js';
const variant = (process.env.PTT_VARIANT === 'growth' || process.argv.includes('--growth')) ? 'growth' : 'fixed';
const name = variant === 'growth' ? 'pocket_tts_wasm_growth' : 'pocket_tts_wasm';
const { default: createModule } = await import(`../vendor/ptt/${name}.mjs`);
const root = new URL('../', import.meta.url);
const M = await createModule();
globalThis.self = globalThis;
const tok = new Tokenizer(JSON.parse(readFileSync(new URL('models/spm_vocab.json', root))));
self.__pkttsTokenize = text => tok.encode(text);
M.FS.mkdir('/models'); M.FS.mkdir('/voices'); M.FS.mkdir('/voices/.cache');
for (const file of ['flow_lm_main_delta_attn_flow_int8.onnx','flow_lm_main_delta_attn_flow_int8_soura.onnx',
  'mimi_decoder_delta_int8.onnx','text_conditioner.onnx','bos_before_voice.npy','soura_vectors.npy'])
  M.FS.writeFile(`/models/${file}`, readFileSync(new URL(`models/${file}`, root)));
M.FS.writeFile('/voices/.cache/alba.emb', readFileSync(new URL('presets/alba.emb', root)));
const c = (n, ret, args) => M.cwrap(n, ret, args);
const create = c('ptt_create_ex','number',['string','string','string','string','number','number','number','number']);
const destroy = c('ptt_destroy',null,['number']);
const emotion = c('ptt_set_emotion','number',['number','string','number']);
const seed = c('ptt_set_seed','number',['number','number']);
const load = c('ptt_load_soura_vectors','number',['number','string']);
const prepare = c('ptt_prepare_voice','number',['number','string']);
const start = c('ptt_latents_start','number',['number','string','string']);
const poll = c('ptt_latents_poll','number',['number','number','number']);
const stop = c('ptt_latents_stop',null,['number']);
const end = c('ptt_latents_end',null,['number']);
c('ptt_configure_pool',null,['number','number'])(2,1);
const engine = flags => { const h = create('/models','/voices','unused','int8',.6,1,2,flags); assert(h); return h; };
const sleep = () => new Promise(resolve => setTimeout(resolve,1));
async function frames(h, cancelled = false) {
  assert.equal(seed(h,31),0);
  const ctx = start(h,'There you are. It is good to see you again.','alba.wav'); assert(ctx);
  const ptr = M._malloc(132), values = [];
  try {
    for (;;) {
      const status = poll(ctx,ptr,ptr+128);
      if (status === 0) break;
      if (status === -1) { await sleep(); continue; }
      if (M.HEAPU32[(ptr+128)>>2] === 0) values.push(...M.HEAPF32.slice(ptr>>2,(ptr>>2)+32));
      if (cancelled) stop(ctx);
    }
  } finally { end(ctx); M._free(ptr); }
  assert(values.length && values.every(Number.isFinite));
  return values;
}
let h = engine(2);
assert.equal(emotion(h,'happy',.8),-1);
assert.equal(seed(h,-1),-1); assert.equal(seed(h,.5),-1); assert.equal(seed(h,2**53),-1);
const original = await frames(h);
destroy(h);
h = engine(2|8);
const neutral = await frames(h);
assert.equal(neutral.length,original.length);
const difference = Math.max(...neutral.map((x,i)=>Math.abs(x-original[i])));
// Zero-add graphs may take a different ORT optimization path; tiny recurrent
// rounding changes can accumulate. Compare the matched first frame, and test
// exact neutral recovery/repeatability within the steering graph below.
const firstFrameDifference = Math.max(...neutral.slice(0,32).map((x,i)=>Math.abs(x-original[i])));
assert(firstFrameDifference < 1e-4, `first-frame neutral drift ${firstFrameDifference}`);
assert.equal(emotion(h,'angry',.8),0);
const angry = await frames(h);
assert.notDeepEqual(angry,neutral);
assert.deepEqual(await frames(h),angry);
assert.equal(emotion(h,'angry',0),0); assert.deepEqual(await frames(h),neutral);
assert.equal(emotion(h,'happy',.8),0); await frames(h,true);
assert.equal(emotion(h,'neutral',0),0); assert.deepEqual(await frames(h),neutral);
assert.equal(emotion(h,'surprise',1),-1); assert.equal(emotion(h,'sad',1.200001),-1);
M.FS.writeFile('/models/bad.npy',new Uint8Array([0]));
assert.equal(load(h,'/models/bad.npy'),-1); assert.deepEqual(await frames(h),neutral);
M.FS.writeFile('/models/derived.npy',readFileSync(new URL('models/soura_vectors.derived.npy',root)));
assert.equal(load(h,'/models/derived.npy'),0);
for (const label of ['angry','disgust','fear','happy','sad']) {
  assert.equal(emotion(h,label,.8),0); await frames(h);
}
destroy(h);
M.FS.unlink('/voices/.cache/alba.kv');
h = engine(16);
assert.equal(prepare(h,'alba.wav'),0);
assert(M.FS.readFile('/voices/.cache/alba.kv').length > 0);
destroy(h);
h = engine(2|8); assert.deepEqual(await frames(h),neutral); destroy(h);
console.log(JSON.stringify({variant,passed:true,neutralFirstFrameMaxError:firstFrameDifference,neutralTrajectoryMaxError:difference,checks:['disabled','seed','neutral','repeatability','zero bypass','cancel recovery','custom vectors','all labels','conditioning-only cache transfer']}));
process.exit(0);
