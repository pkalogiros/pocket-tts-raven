// Emscripten-native TTS engine worker: the full pocket_tts.cpp compiled to
// WASM, driven through its ptt_* C API. Speaks the exact same message
// protocol as tts-worker.js, so the UI cannot tell the engines apart.
import { Tokenizer } from "./engine/tokenizer.js";
import { SilenceSqueeze } from "./engine/silence.js";

// Fixed-memory module is ~10-25% faster (bounds-check elimination) but its
// 768MB SharedArrayBuffer can be refused on mobile Safari — fall back to the
// growable-memory variant. `prefer` ("fixed"|"growth") forces one for A/B.
let variantLoaded = "";
function assetJoin(base, path) {
  base = String(base || "").replace(/\/+$/, "");
  path = String(path || "").replace(/^\/+/, "");
  return base ? `${base}/${path}` : "";
}

function moduleWorker(url) {
  const u = new URL(url, import.meta.url);
  if (u.origin === self.location.origin) return new Worker(u.href, { type: "module" });
  const code = `import ${JSON.stringify(u.href)};`;
  const blob = new Blob([code], { type: "text/javascript" });
  const blobUrl = URL.createObjectURL(blob);
  const worker = new Worker(blobUrl, { type: "module" });
  setTimeout(() => URL.revokeObjectURL(blobUrl), 30000);
  return worker;
}

function cloneCapSamples(capSeconds) {
  const n = Number(capSeconds);
  const seconds = Number.isFinite(n) && n > 0 ? Math.min(30, Math.max(1, n)) : 12;
  return Math.round(seconds * SR);
}

function pttModuleOptions(assetBase, mainScriptUrl) {
  return {
    mainScriptUrlOrBlob: mainScriptUrl,
    locateFile: (path) => assetJoin(assetBase, `vendor/ptt/${path}`) || path,
  };
}

async function importPttModule(assetBase, fileName, fallback) {
  const remote = assetJoin(assetBase, `vendor/ptt/${fileName}`);
  if (!remote) {
    const { default: create } = await import(fallback);
    return { create, options: {} };
  }

  // Emscripten pthread builds spawn the main .mjs as a Worker. Browser
  // module workers must be same-origin, so import the S3 wrapper through a
  // same-origin blob while locateFile keeps the .wasm on the CDN.
  const res = await fetch(remote);
  if (!res.ok) throw new Error(`fetch ${fileName}: ${res.status}`);
  const blobUrl = URL.createObjectURL(new Blob([await res.text()], { type: "text/javascript" }));
  const { default: create } = await import(blobUrl);
  return { create, options: pttModuleOptions(assetBase, blobUrl) };
}

async function instantiateModule(prefer, assetBase) {
  if (prefer !== "growth") {
    try {
      const { create, options } = await importPttModule(assetBase, "pocket_tts_wasm.mjs", "../vendor/ptt/pocket_tts_wasm.mjs");
      const m = await create(options);
      variantLoaded = "fixed";
      return m;
    } catch (e) {
      console.warn("fixed-memory module failed, using growth variant:", e.message);
    }
  }
  const { create, options } = await importPttModule(assetBase, "pocket_tts_wasm_growth.mjs", "../vendor/ptt/pocket_tts_wasm_growth.mjs");
  const m = await create(options);
  variantLoaded = "growth";
  return m;
}

let M = null;       // emscripten module
let handle = 0;
let api = null;

const SR = 24000;
// The 16MB (br) voice encoder is only needed for cloning: fetched lazily on
// first use (defer_encoder create flag), like the TS engine.
const MODELS = ["flow_lm_main_delta_attn_flow_int8.onnx",
                "mimi_decoder_delta_int8.onnx", "text_conditioner.onnx",
                "bos_before_voice.npy"];
const WEIGHTS = { "flow_lm_main_delta_attn_flow_int8.onnx": 0.72,
                  "mimi_decoder_delta_int8.onnx": 0.17, "text_conditioner.onnx": 0.10,
                  "bos_before_voice.npy": 0.01 };

function post(type, payload, transfer) {
  self.postMessage({ type, ...payload }, transfer || []);
}

function readExportFile(path, kind, ext, type) {
  try {
    const bytes = M.FS.readFile(path);
    return {
      file: {
        kind, ext, type,
        buffer: bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
      },
    };
  } catch (e) {
    return null;
  }
}

// Progress arrives per network chunk (hundreds/sec on fast links); posting
// them all floods the page's main thread and makes the overlay flicker.
let lastProgressT = 0;
function postProgress(label, pct) {
  const now = performance.now();
  if (pct < 1 && now - lastProgressT < 100) return;
  lastProgressT = now;
  post("progress", { label, pct });
}

async function fetchInto(path, url, label, base, weight) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url}: ${res.status}`);
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
  M.FS.writeFile(path, buf);
}

function wavF32(samples) {
  const head = new ArrayBuffer(44);
  const dv = new DataView(head);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i)); };
  const bytes = samples.length * 4;
  str(0, "RIFF"); dv.setUint32(4, 36 + bytes, true); str(8, "WAVE");
  str(12, "fmt "); dv.setUint32(16, 16, true); dv.setUint16(20, 3, true); // IEEE float
  dv.setUint16(22, 1, true); dv.setUint32(24, SR, true);
  dv.setUint32(28, SR * 4, true); dv.setUint16(32, 4, true); dv.setUint16(34, 32, true);
  str(36, "data"); dv.setUint32(40, bytes, true);
  const out = new Uint8Array(44 + bytes);
  out.set(new Uint8Array(head));
  out.set(new Uint8Array(samples.buffer, samples.byteOffset, bytes), 44);
  return out;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let activeStream = 0;

// Engine calls share state (sessions, caches): serialize speak/clone/prewarm
// so a Speak arriving mid-prewarm queues instead of interleaving.
let chain = Promise.resolve();
const serialize = (fn) => { chain = chain.then(fn, fn); return chain; };

// Async poll loop: never blocks the Emscripten runtime thread, so proxied
// filesystem calls from the generator pthread keep being serviced.
async function streamOut(text, voiceFile, temperature, emitChunks) {
  api.set_temperature(handle, temperature);
  const ctx = api.stream_start(handle, text, voiceFile);
  if (!ctx) throw new Error("stream_start failed");
  activeStream = ctx;
  const pp = M._malloc(8);
  const squeeze = new SilenceSqueeze();
  let total = 0;
  const emit = (samples) => {
    if (!samples.length) return;
    total += samples.length;
    const copy = samples.slice();
    post("chunk", { samples: copy.buffer }, [copy.buffer]);
  };
  try {
    for (;;) {
      const r = api.stream_poll(ctx, pp, pp + 4);
      if (r === 0) break;
      if (r === -1) { await sleep(0); continue; }
      const ptr = M.HEAPU32[pp >> 2];
      const len = M.HEAPU32[(pp + 4) >> 2];
      if (len > 0 && emitChunks) {
        emit(squeeze.push(M.HEAPF32.subarray(ptr >> 2, (ptr >> 2) + len)));
      } else {
        total += len;
      }
      api.free_audio(ptr);
    }
    if (emitChunks) emit(squeeze.flush());
  } finally {
    activeStream = 0;
    api.stream_end(ctx);
    M._free(pp);
  }
  return total;
}

// Voices whose conditioning caches (.emb + KV snapshot) are already built.
const warmed = new Set();

// Restored WAV-format voices encode lazily on first use. Encoding runs in a
// DISPOSABLE worker (own growth-memory instance, terminated afterwards) so
// the encoder's ~0.5GB activation peak never touches this engine's heap.
const needsEncoder = new Set();
let savedModelsUrl = "";
let savedAssetBase = "";
function encodeInWorker(key, wavBytes) {
  return new Promise((resolve, reject) => {
    post("progress", { label: "Encoding voice (fetches 15 MB once)", pct: 0.4 });
    const w = moduleWorker(new URL("./encode-worker.js", import.meta.url).href);
    const pool = Math.min(4, self.navigator?.hardwareConcurrency || 2);
    w.onmessage = (e) => {
      w.terminate(); // frees the encode instance's memory for real
      if (e.data.ok) {
        M.FS.writeFile(`/voices/.cache/${key}.emb`, new Uint8Array(e.data.emb));
        resolve();
      } else {
        reject(new Error(e.data.message));
      }
    };
    w.onerror = (e) => { w.terminate(); reject(new Error(e.message || "encode worker crashed")); };
    w.postMessage({ assetBase: savedAssetBase, modelsUrl: savedModelsUrl, wav: wavBytes.buffer, key, pool },
                  [wavBytes.buffer]);
  });
}

// Cheap per-voice warm: start a generation and abort as soon as the first
// chunk arrives — by then the voice embedding, conditioning pass and KV
// snapshot are all cached. Costs ~conditioning + a few AR steps.
async function warmVoice(voiceKey) {
  if (warmed.has(voiceKey)) return;
  if (needsEncoder.has(voiceKey)) {
    await encodeInWorker(voiceKey, M.FS.readFile(`/voices/${voiceKey}.wav`));
    needsEncoder.delete(voiceKey);
  }
  warmed.add(voiceKey);
  api.set_temperature(handle, 0.5);
  const ctx = api.stream_start(handle, "Hi.", `${voiceKey}.wav`);
  if (!ctx) { warmed.delete(voiceKey); return; }
  activeStream = ctx;
  const pp = M._malloc(8);
  let aborted = false;
  try {
    for (;;) {
      const r = api.stream_poll(ctx, pp, pp + 4);
      if (r === 0) break;
      if (r === -1) { await sleep(0); continue; }
      M._free(M.HEAPU32[pp >> 2]);
      if (!aborted) { aborted = true; api.stream_stop(ctx); }
    }
  } finally {
    activeStream = 0;
    api.stream_end(ctx);
    M._free(pp);
  }
}

const handlers = {
  async init({ assetBase, modelsUrl, threads, pool: reqPool, spin, variant, bench }) {
    // Pool sizing, from measured sweeps (?pool=N overrides for tuning):
    //  - phones: 2. Only ~2 performance cores; ORT's parallel-for waits for
    //    the slowest partition, so anything scheduled on an E-core gates
    //    every op. 2 threads = P-cores only, measured best on-device.
    //  - desktop: 8; 10 on very wide machines (>=14 cores) where it buys a
    //    few percent. 12+ regresses even on 16 cores.
    const hc = self.navigator?.hardwareConcurrency || 8;
    const mobile = /iPhone|iPad|iPod|Android/i.test(self.navigator?.userAgent || "");
    const defPool = mobile ? 2 : (hc >= 14 ? 10 : Math.min(8, Math.max(4, hc - 2)));
    const pool = (reqPool >= 1 && reqPool <= 12) ? reqPool : defPool;
    savedAssetBase = assetBase || "";
    M = await instantiateModule(variant, savedAssetBase);
    api = {
      create: M.cwrap("ptt_create", "number",
        ["string", "string", "string", "string", "number", "number", "number"]),
      set_temperature: M.cwrap("ptt_set_temperature", null, ["number", "number"]),
      stream_start: M.cwrap("ptt_stream_start", "number", ["number", "string", "string"]),
      stream_read: M.cwrap("ptt_stream_read", "number", ["number", "number", "number"]),
      stream_poll: M.cwrap("ptt_stream_poll", "number", ["number", "number", "number"]),
      stream_stop: M.cwrap("ptt_stream_stop", null, ["number"]),
      stream_end: M.cwrap("ptt_stream_end", null, ["number"]),
      free_audio: (p) => M._free(p),
      destroy: M.cwrap("ptt_destroy", null, ["number"]),
    };
    M.FS.mkdir("/models");
    M.FS.mkdir("/voices");
    M.FS.mkdir("/voices/.cache");
    savedModelsUrl = modelsUrl;

    let base = 0;
    for (const name of MODELS) {
      await fetchInto(`/models/${name}`, `${modelsUrl}/${name}`, "Downloading models (67 MB)", base, WEIGHTS[name]);
      base += WEIGHTS[name];
    }
    // Tokenization is delegated from C++ to the validated JS implementation.
    const vocab = await (await fetch(`${modelsUrl}/spm_vocab.json`)).json();
    const tok = new Tokenizer(vocab);
    self.__pkttsTokenize = (text) => tok.encode(text);

    post("progress", { label: "Starting engine", pct: 0.97 });
    M.cwrap("ptt_configure_pool", null, ["number", "number"])(pool, spin === 0 ? 0 : 1);
    handle = M.cwrap("ptt_create_ex", "number",
      ["string", "string", "string", "string", "number", "number", "number", "number"])(
      "/models", "/voices", "/models/tokenizer.model", "int8", 0.7, 1, pool, 2 /* defer encoder */);
    if (!handle) throw new Error("ptt_create failed (see console)");
    // Web demo speaks prose: let the model see commas and phrase clauses
    // itself (the game build keeps the soften-commas default).
    M.cwrap("ptt_set_soften_commas", null, ["number", "number"])(handle, 0);
    post("ready", { sr: SR, variant: variantLoaded, pool, spin: spin === 0 ? 0 : 1 });
    if (bench) {
      // Isolated on-device numbers: AR step + 15-frame decoder chunk.
      const db = M.cwrap("ptt_debug_bench", "number", ["number", "number", "number", "number"]);
      post("bench", { ar: db(handle, 1, 0, 30), dec15: db(handle, 0, 15, 6),
                      pool, variant: variantLoaded });
    }
  },

  async loadPresets({ presets }) {
    for (const p of presets) {
      const buf = new Uint8Array(await (await fetch(p.url)).arrayBuffer());
      const stem = p.key;
      M.FS.writeFile(`/voices/.cache/${stem}.emb`, buf);
      post("voice", { key: p.key, label: p.label, builtin: true });
    }
  },

  async cloneInner({ key, label, audio, capSeconds }) {
    let samples = new Float32Array(audio);
    const CAP = cloneCapSamples(capSeconds); // embedding prefix scales with length; keep sources tight
    if (samples.length > CAP) samples = samples.subarray(0, CAP);
    const wavBytes = wavF32(samples);
    M.FS.writeFile(`/voices/${key}.wav`, wavBytes);
    await encodeInWorker(key, wavF32(samples)); // fresh copy: buffer is transferred
    post("progress", { label: "Encoding + conditioning voice", pct: 0.5 });
    // Warm: run a short synthesis; encode_voice + KV conditioning results are
    // cached in the engine (memory + MEMFS), exactly like the native CLI.
    await streamOut("Hi.", `${key}.wav`, 0.7, false);
    // Persist the source WAV via the app's OPFS store (engine re-derives the
    // caches from it on restore).
    const wav = M.FS.readFile(`/voices/${key}.wav`);
    post("voice", { key, label, builtin: false, embDims: null, emb: wav.buffer },
         [wav.buffer]);
  },

  async clone(payload) {
    await serialize(() => handlers.cloneInner(payload));
  },

  async restoreVoice({ key, label, embDims, emb, embCache, capSeconds }) {
    if (embDims && embDims.length) {
      // TS-engine record: a raw float voice embedding. Write it as an EMB1
      // cache file — the engine then treats it exactly like a preset.
      const dims = BigInt64Array.from(embDims, BigInt);
      const head = new Uint8Array(8 + dims.length * 8);
      new DataView(head.buffer).setUint32(0, 0x31424d45, true); // "EMB1"
      new DataView(head.buffer).setUint32(4, dims.length, true);
      head.set(new Uint8Array(dims.buffer), 8);
      const body = new Uint8Array(emb);
      const out = new Uint8Array(head.length + body.length);
      out.set(head); out.set(body, head.length);
      M.FS.writeFile(`/voices/.cache/${key}.emb`, out);
    } else {
      const capBytes = 44 + cloneCapSamples(capSeconds) * 4;
      let bytes = new Uint8Array(emb);
      if (bytes.length > capBytes) {
        bytes = bytes.slice(0, capBytes);
        // patch RIFF/data sizes so the truncated wav stays valid
        const dv = new DataView(bytes.buffer);
        dv.setUint32(4, bytes.length - 8, true);
        dv.setUint32(40, bytes.length - 44, true);
      }
      M.FS.writeFile(`/voices/${key}.wav`, bytes);
      if (embCache) {
        M.FS.writeFile(`/voices/.cache/${key}.emb`, new Uint8Array(embCache));
        needsEncoder.delete(key);
      } else {
        needsEncoder.add(key);
      }
    }
    post("voice", { key, label, builtin: false });
  },

  async dropVoice({ key }) {
    for (const p of [`/voices/${key}.wav`, `/voices/.cache/${key}.emb`, `/voices/.cache/${key}.kv`]) {
      try { M.FS.unlink(p); } catch (e) { /* absent is fine */ }
    }
  },

  async exportVoice({ key, label }) {
    await serialize(async () => {
      const wavPath = `/voices/${key}.wav`;
      const embPath = `/voices/.cache/${key}.emb`;
      if (needsEncoder.has(key) || !readExportFile(embPath, "emb", ".emb", "application/octet-stream")) {
        const wavBytes = M.FS.readFile(wavPath);
        await encodeInWorker(key, wavBytes);
        needsEncoder.delete(key);
      }

      const entries = [
        readExportFile(wavPath, "wav", ".wav", "audio/wav"),
        readExportFile(embPath, "emb", ".emb", "application/octet-stream"),
        readExportFile(`/voices/.cache/${key}.kv`, "kv", ".kv", "application/octet-stream"),
      ].filter(Boolean);
      const files = entries.map((e) => e.file);
      if (!files.some((f) => f.kind === "wav") || !files.some((f) => f.kind === "emb")) {
        post("voiceExportError", { key, label });
        return;
      }
      post("voiceExport", { key, label, files }, files.map((f) => f.buffer));
    });
  },

  async stop() {
    if (activeStream) api.stream_stop(activeStream);
  },

  async shutdown() {
    if (activeStream) api.stream_stop(activeStream);
    close();
  },

  async prewarm({ voiceKey }) {
    // One tiny silent generation: warms the voice-conditioning KV cache,
    // ORT arenas, and hot wasm kernels before the user's first Speak.
    await serialize(async () => {
      await streamOut("Hi.", `${voiceKey}.wav`, 0.5, false);
      warmed.add(voiceKey);
    });
  },

  async prewarmVoice({ voiceKey }) {
    await serialize(() => warmVoice(voiceKey));
  },

  async speak({ text, voiceKey, temperature }) {
    await serialize(async () => {
      if (needsEncoder.has(voiceKey)) {
        await encodeInWorker(voiceKey, M.FS.readFile(`/voices/${voiceKey}.wav`));
        needsEncoder.delete(voiceKey);
      }
      warmed.add(voiceKey); // a full speak conditions it as a side effect
      post("speakStarted", {});
      // Presets resolve via /voices/.cache/{key}.emb even when the .wav is absent.
      const total = await streamOut(text, `${voiceKey}.wav`, temperature ?? 0.45, true);
      post("speakDone", { totalSamples: total });
    });
  },
};

self.onmessage = async (e) => {
  const { type, ...payload } = e.data;
  try {
    await handlers[type](payload);
  } catch (err) {
    post("error", { message: String(err && err.message || err), during: type });
  }
};
