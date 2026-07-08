// TTS engine worker: owns the ONNX sessions so the UI thread never blocks.
// Speaks a tiny message protocol that app.js bridges onto the PKTTS bus.
import * as ort from "../vendor/ort/ort.wasm.min.mjs";
import { PocketTTS, SR } from "./engine/synth.js";
import { SilenceSqueeze } from "./engine/silence.js";

ort.env.wasm.wasmPaths = new URL("../vendor/ort/", import.meta.url).href;

let tts = null;

// Serialize engine work: speak/clone/prewarm share session state.
let chain = Promise.resolve();
const serialize = (fn) => { chain = chain.then(fn, fn); return chain; };

function post(type, payload, transfer) {
  self.postMessage({ type, ...payload }, transfer || []);
}

function cloneCapSamples(capSeconds) {
  const n = Number(capSeconds);
  const seconds = Number.isFinite(n) && n > 0 ? Math.min(30, Math.max(1, n)) : 12;
  return Math.round(seconds * SR);
}

// Rate-limit download progress: per-chunk posting floods the main thread.
let lastProgressT = 0;
function postProgress(label, pct) {
  const now = performance.now();
  if (pct < 1 && now - lastProgressT < 100) return;
  lastProgressT = now;
  post("progress", { label, pct });
}

async function fetchWithProgress(url, label, base, weight) {
  const res = await fetch(url);
  const total = Number(res.headers.get("Content-Length")) || 0;
  const reader = res.body.getReader();
  const parts = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    got += value.length;
    if (total) postProgress(label, base + Math.min(1, got / total) * weight);
  }
  const buf = new Uint8Array(got);
  let o = 0;
  for (const p of parts) { buf.set(p, o); o += p.length; }
  return buf;
}

const handlers = {
  async init({ modelsUrl, threads }) {
    // Pre-fetch the big models with download progress, hand bytes to ORT.
    const buffers = {};
    const parts = [
      ["flow_lm_main_delta_flow_int8.onnx", 0.62],
      ["mimi_decoder_delta_int8.onnx", 0.18],
      ["text_conditioner.onnx", 0.12],
    ];
    let base = 0;
    for (const [name, weight] of parts) {
      buffers[name] = await fetchWithProgress(`${modelsUrl}/${name}`, "Downloading models (67 MB)", base, weight);
      base += weight;
    }
    post("progress", { label: "Compiling sessions", pct: 0.95 });
    tts = await PocketTTS.create({ ort, modelsUrl, threads, buffers });
    post("ready", { sr: SR });
  },

  async loadPresets({ presets }) {
    for (const p of presets) {
      const buf = await (await fetch(p.url)).arrayBuffer();
      const dv = new DataView(buf);
      const ndims = dv.getUint32(4, true);
      const dims = [];
      for (let i = 0; i < ndims; i++) dims.push(Number(dv.getBigInt64(8 + i * 8, true)));
      const emb = { dims, data: new Float32Array(buf.slice(8 + ndims * 8)) };
      await tts.prepareVoice(p.key, emb);
      post("voice", { key: p.key, label: p.label, builtin: true });
    }
  },

  async clone({ key, label, audio, capSeconds }) {
    post("progress", { label: "Loading voice encoder", pct: 0 });
    await tts.loadEncoder();
    post("progress", { label: "Encoding voice", pct: 0.4 });
    let samples = audio;
    const CAP = cloneCapSamples(capSeconds);
    if (samples.length > CAP) samples = samples.subarray(0, CAP);
    const emb = await tts.encodeVoice(samples);
    post("progress", { label: "Conditioning voice", pct: 0.75 });
    await tts.prepareVoice(key, emb);
    // Ship the embedding back so the app can persist it (OPFS).
    const raw = emb.data.buffer.slice(0);
    post("voice", { key, label, builtin: false, embDims: emb.dims, emb: raw }, [raw]);
  },

  async dropVoice({ key }) {
    tts.voiceSnapshots.delete(key); // frees the ~50MB KV snapshot
  },

  async restoreVoice({ key, label, embDims, emb, capSeconds }) {
    if (embDims && embDims.length) {
      await tts.prepareVoice(key, { dims: embDims, data: new Float32Array(emb) });
    } else {
      // Native-engine record: a 24kHz float32 WAV of the source audio.
      // Re-encode it (44-byte canonical header written by the native worker).
      let samples = new Float32Array(emb, 44);
      const CAP = cloneCapSamples(capSeconds);
      if (samples.length > CAP) samples = samples.subarray(0, CAP);
      const embedding = await tts.encodeVoice(samples);
      await tts.prepareVoice(key, embedding);
    }
    post("voice", { key, label, builtin: false });
  },

  async stop() {
    if (tts) tts.cancel();
  },

  async shutdown() {
    if (tts) tts.cancel();
    close();
  },

  async prewarmVoice() {
    // TS engine conditions voices at prepareVoice/restore time — the KV
    // snapshot is already cached, so select-time warming is a no-op.
  },

  async prewarm({ voiceKey }) {
    await serialize(async () => {
      const noop = () => {};
      await tts.speak("Hi.", voiceKey, noop, { temperature: 0.5 });
    });
  },

  async speakInner({ text, voiceKey, temperature, seed }) {
    post("speakStarted", {});
    const squeeze = new SilenceSqueeze();
    let total = 0;
    const emit = (samples) => {
      if (!samples.length) return;
      total += samples.length;
      const copy = samples.slice();
      post("chunk", { samples: copy.buffer }, [copy.buffer]);
    };
    await tts.speak(text, voiceKey, (chunk) => emit(squeeze.push(chunk)),
                    { temperature, seed });
    emit(squeeze.flush());
    post("speakDone", { totalSamples: total });
  },

  async speak(payload) {
    await serialize(() => handlers.speakInner(payload));
  },
};

self.onmessage = async (e) => {
  const { type, id, ...payload } = e.data;
  try {
    await handlers[type](payload);
    if (id) post("ack", { id });
  } catch (err) {
    post("error", { message: String(err && err.message || err), during: type });
  }
};
