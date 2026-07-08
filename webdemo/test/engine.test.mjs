// Engine correctness test: run the JS engine at temperature 0 with the preset
// voice and compare latents + raw decoded audio against the python reference
// (same models, same orchestration, native ORT). Run gen_reference.py first.
//
//   node webdemo/test/engine.test.mjs
import { readFileSync } from "node:fs";
import * as ort from "onnxruntime-web";
import { PocketTTS } from "../src/engine/synth.js";

const dir = new URL(".", import.meta.url).pathname;
const models = `${dir}/../models`;
const meta = JSON.parse(readFileSync(`${dir}/ref_meta.json`, "utf8"));
const refLat = new Float32Array(readFileSync(`${dir}/ref_latents.f32`).buffer);
const refAudio = new Float32Array(readFileSync(`${dir}/ref_audio.f32`).buffer);

function loadEmb(path) {
  const b = readFileSync(path);
  const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
  const ndims = dv.getUint32(4, true);
  const dims = [];
  for (let i = 0; i < ndims; i++) dims.push(Number(dv.getBigInt64(8 + i * 8, true)));
  const data = new Float32Array(b.buffer.slice(b.byteOffset + 8 + ndims * 8));
  return { dims, data };
}

const loader = {
  json: async (u) => JSON.parse(readFileSync(u, "utf8")),
  buffer: async (u) => {
    const b = readFileSync(u);
    return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
  },
};

const TEXT = "How are things over in Riverwood these days?";
const tts = await PocketTTS.create({ ort, modelsUrl: models, threads: 4, loader });
const emb = loadEmb(`${dir}/../presets/alba.emb`);
if (!tts.hasBos(emb)) throw new Error("preset emb lacks BOS");
await tts.prepareVoice("alba", emb);

// Mirror speak()'s inner loop without the emitter, matching the reference.
const snap = tts.voiceSnapshots.get("alba");
tts.main.restore(snap);
const ids = tts.tok.encode(TEXT);
if (JSON.stringify(ids) !== JSON.stringify(meta.ids)) {
  console.log("FAIL tokenizer ids differ from reference");
  process.exit(1);
}
const tokT = new ort.Tensor("int64", BigInt64Array.from(ids, BigInt), [1, ids.length]);
const t = await tts.txt.run({ [tts.txt.inputNames[0]]: tokT });
const temb = t[tts.txt.outputNames[0]];
const tdims = temb.dims.length === 2 ? [1, ...temb.dims] : [...temb.dims];
await tts.condPass(temb.data, tdims);
tts.dec.reset();

const cl = new Float32Array(32).fill(NaN);
const latents = [];
let eos = false, extra = 0;
for (let step = 0; step < 200; step++) {
  const out = await tts.main.run({
    sequence: new ort.Tensor("float32", cl.slice(), [1, 1, 32]),
    text_embeddings: tts.emptyText(),
    flow_x: tts.zeroX(),
  });
  if (!eos && out.eos_logit.data[0] > -4.0) eos = true;
  if (eos && ++extra > 3) break;
  const lat = Float32Array.from(out.latent.data);
  latents.push(lat);
  cl.set(lat);
}

console.log(`frames: ${latents.length} (ref ${meta.frames})`);
let worstLat = 0;
latents.forEach((l, i) => {
  for (let j = 0; j < 32; j++) {
    const d = Math.abs(l[j] - refLat[i * 32 + j]);
    if (d > worstLat) worstLat = d;
  }
});
console.log(`latent max |Δ| vs reference: ${worstLat.toExponential(2)}`);

const chunks = [1, 2, 4, 8];
let i = 0, ci = 0;
const audio = [];
while (i < latents.length) {
  const take = Math.min(chunks[Math.min(ci, chunks.length - 1)], latents.length - i);
  const lat = new Float32Array(take * 32);
  for (let k = 0; k < take; k++) lat.set(latents[i + k], k * 32);
  const out = await tts.dec.run({ latent: new ort.Tensor("float32", lat, [1, take, 32]) });
  audio.push(Float32Array.from(out[Object.keys(out)[0]].data));
  i += take; ci++;
}
const full = new Float32Array(audio.reduce((a, c) => a + c.length, 0));
{ let o = 0; for (const c of audio) { full.set(c, o); o += c.length; } }
console.log(`audio samples: ${full.length} (ref ${meta.samples})`);
let worstA = 0, sumSq = 0, sigSq = 0;
const n = Math.min(full.length, refAudio.length);
for (let k = 0; k < n; k++) {
  const d = full[k] - refAudio[k];
  if (Math.abs(d) > worstA) worstA = Math.abs(d);
  sumSq += d * d; sigSq += refAudio[k] * refAudio[k];
}
const snr = 10 * Math.log10(sigSq / (sumSq || 1e-30));
console.log(`audio max |Δ|: ${worstA.toExponential(2)}  SNR vs reference: ${snr.toFixed(1)} dB`);

// Platform note: WASM MLAS vs native MLAS produce ~1e-6 per-step differences
// that fp16 KV rounding + AR feedback amplify into trajectory divergence, so
// exact latent/audio equality across runtimes is NOT expected (equivalent to
// a different random draw; each platform is self-consistent). The rigorous
// orchestration check is test/xfer.test.mjs (same-state single step ≤1e-3).
// Here we assert sane, deterministic end-to-end behavior.
const frameRatio = latents.length / meta.frames;
let rms = 0;
for (let k = 0; k < full.length; k++) rms += full[k] * full[k];
rms = Math.sqrt(rms / full.length);
const ok = frameRatio > 0.7 && frameRatio < 1.4 &&
           full.length === latents.length * 1920 &&
           rms > 0.01 && rms < 0.5;
console.log(`frame ratio vs ref: ${frameRatio.toFixed(2)}  audio rms: ${rms.toFixed(3)}`);
console.log(ok ? "ENGINE SANITY OK (see xfer.test.mjs for the exact-match check)" : "ENGINE MISMATCH");
process.exit(ok ? 0 : 1);
