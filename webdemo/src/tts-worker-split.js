// Emscripten-native TTS engine worker: the full pocket_tts.cpp compiled to
// WASM, driven through its ptt_* C API. Speaks the exact same message
// protocol as tts-worker.js, so the UI cannot tell the engines apart.
import { Tokenizer } from "./engine/tokenizer.js";
import { SilenceSqueeze } from "./engine/silence.js";
import { Emitter } from "./engine/synth.js";

// Fixed-memory module is ~10-25% faster (bounds-check elimination) but its
// 768MB SharedArrayBuffer can be refused on mobile Safari — fall back to the
// growable-memory variant. `prefer` ("fixed"|"growth") forces one for A/B.
let variantLoaded = "";
function assetJoin(base, path) {
  base = String(base || "").replace(/\/+$/, "");
  path = String(path || "").replace(/^\/+/, "");
  return base ? `${base}/${path}` : "";
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

let M = null;       // emscripten module (instance A: conditioning + AR)
let handle = 0;
let api = null;
let MB = null;      // instance B: decoder-only copy of this module (own pool)
let dec = null;     // { run(staging, T) -> Float32Array, reset() }
let activeLat = 0;

const SR = 24000;
// The C++ constructor opens every session eagerly, so the encoder ships in
// the initial download here (unlike the TS engine, which lazy-loads it).
const MODELS = ["flow_lm_main_delta_attn_flow_int8.onnx", "mimi_encoder.onnx",
                "mimi_decoder_delta_int8.onnx", "text_conditioner.onnx",
                "bos_before_voice.npy"];
const WEIGHTS = { "flow_lm_main_delta_attn_flow_int8.onnx": 0.53, "mimi_encoder.onnx": 0.22,
                  "mimi_decoder_delta_int8.onnx": 0.14, "text_conditioner.onnx": 0.10,
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
  return buf;
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

// Cheap per-voice warm: start a generation and abort as soon as the first
// chunk arrives — by then the voice embedding, conditioning pass and KV
// snapshot are all cached. Costs ~conditioning + a few AR steps.
async function warmVoice(voiceKey) {
  if (warmed.has(voiceKey)) return;
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
  async init({ assetBase, modelsUrl, threads, arPool: reqAr, decPool: reqDec, spin, variant, bench }) {
    // Pool sizing, from measured sweeps (?pool=N overrides for tuning):
    //  - phones: 2. Only ~2 performance cores; ORT's parallel-for waits for
    //    the slowest partition, so anything scheduled on an E-core gates
    //    every op. 2 threads = P-cores only, measured best on-device.
    //  - desktop: 8; 10 on very wide machines (>=14 cores) where it buys a
    //    few percent. 12+ regresses even on 16 cores.
    // Split pools, from the mock sweep: total must respect core count
    // (6/6 best on 16 cores; oversubscription regresses). Mobile: 2/2 —
    // AR holds the P-cores, decoder rides the E-cores.
    const hc = self.navigator?.hardwareConcurrency || 8;
    const mobile = /iPhone|iPad|iPod|Android/i.test(self.navigator?.userAgent || "");
    const defAr = mobile ? 2 : (hc >= 14 ? 6 : Math.min(4, Math.max(2, hc >> 1)));
    const defDec = mobile ? 2 : (hc >= 14 ? 6 : Math.min(4, Math.max(2, hc >> 1)));
    const pool = (reqAr >= 1 && reqAr <= 12) ? reqAr : defAr;
    const decPool = (reqDec >= 1 && reqDec <= 12) ? reqDec : defDec;
    M = await instantiateModule(variant, assetBase);
    api = {
      create: M.cwrap("ptt_create", "number",
        ["string", "string", "string", "string", "number", "number", "number"]),
      set_temperature: M.cwrap("ptt_set_temperature", null, ["number", "number"]),
      stream_start: M.cwrap("ptt_stream_start", "number", ["number", "string", "string"]),
      stream_read: M.cwrap("ptt_stream_read", "number", ["number", "number", "number"]),
      stream_poll: M.cwrap("ptt_stream_poll", "number", ["number", "number", "number"]),
      stream_stop: M.cwrap("ptt_stream_stop", null, ["number"]),
      latents_start: M.cwrap("ptt_latents_start", "number", ["number", "string", "string"]),
      latents_poll: M.cwrap("ptt_latents_poll", "number", ["number", "number", "number"]),
      latents_stop: M.cwrap("ptt_latents_stop", null, ["number"]),
      latents_end: M.cwrap("ptt_latents_end", null, ["number"]),
      stream_end: M.cwrap("ptt_stream_end", null, ["number"]),
      free_audio: (p) => M._free(p),
      destroy: M.cwrap("ptt_destroy", null, ["number"]),
    };
    M.FS.mkdir("/models");
    M.FS.mkdir("/voices");
    M.FS.mkdir("/voices/.cache");
    self.__modelsUrl = modelsUrl;

    let base = 0;
    let decoderBytes = null;
    for (const name of MODELS) {
      const bytes = await fetchInto(`/models/${name}`, `${modelsUrl}/${name}`, "Downloading models (84 MB)", base, WEIGHTS[name]);
      if (name === "mimi_decoder_delta_int8.onnx") decoderBytes = bytes;
      base += WEIGHTS[name];
    }
    // Instance B: a decoder-only copy of our own (trampoline-free) module,
    // with its OWN thread pool. Growth variant: its heap stays tiny and a
    // second fixed SAB would sink mobile Safari.
    {
      const { create, options } = await importPttModule(assetBase, "pocket_tts_wasm_growth.mjs", "../vendor/ptt/pocket_tts_wasm_growth.mjs");
      MB = await create(options);
      MB.FS.mkdir("/models"); MB.FS.mkdir("/voices");
      MB.FS.writeFile("/models/mimi_decoder_delta_int8.onnx", decoderBytes);
      MB.cwrap("ptt_configure_pool", null, ["number", "number"])(decPool, spin === 0 ? 0 : 1);
      const createEx = MB.cwrap("ptt_create_ex", "number",
        ["string", "string", "string", "string", "number", "number", "number", "number"]);
      const hB = createEx("/models", "/voices", "/models/tokenizer.model", "int8", 0.5, 1, decPool, 1);
      if (!hB) throw new Error("decoder-only create failed");
      const decodeC = MB.cwrap("ptt_decode", "number", ["number", "number", "number", "number", "number"]);
      const resetC = MB.cwrap("ptt_decoder_reset", null, ["number"]);
      const latBuf = MB._malloc(15 * 32 * 4);
      const outPP = MB._malloc(8);
      dec = {
        reset: () => resetC(hB),
        run: (staging, T) => {
          MB.HEAPF32.set(staging.subarray(0, T * 32), latBuf >> 2);
          if (decodeC(hB, latBuf, T, outPP, outPP + 4) !== 0) throw new Error("decode failed");
          const ptr = MB.HEAPU32[outPP >> 2], n = MB.HEAPU32[(outPP + 4) >> 2];
          const audio = MB.HEAPF32.slice(ptr >> 2, (ptr >> 2) + n);
          MB._free(ptr);
          return audio;
        },
      };
    }
    // Tokenization is delegated from C++ to the validated JS implementation.
    const vocab = await (await fetch(`${modelsUrl}/spm_vocab.json`)).json();
    const tok = new Tokenizer(vocab);
    self.__pkttsTokenize = (text) => tok.encode(text);

    post("progress", { label: "Starting engine", pct: 0.97 });
    M.cwrap("ptt_configure_pool", null, ["number", "number"])(pool, spin === 0 ? 0 : 1);
    handle = api.create("/models", "/voices", "/models/tokenizer.model", "int8",
                        0.7, 1, pool);
    if (!handle) throw new Error("ptt_create failed (see console)");
    // Web demo speaks prose: let the model see commas and phrase clauses
    // itself (the game build keeps the soften-commas default).
    M.cwrap("ptt_set_soften_commas", null, ["number", "number"])(handle, 0);
    post("ready", { sr: SR, variant: variantLoaded + "-split", pool: pool + "+" + decPool,
                    spin: spin === 0 ? 0 : 1 });
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

  async cloneInner({ key, label, audio }) {
    const samples = new Float32Array(audio);
    M.FS.writeFile(`/voices/${key}.wav`, wavF32(samples));
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

  async restoreVoice({ key, label, embDims, emb }) {
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
      M.FS.writeFile(`/voices/${key}.wav`, new Uint8Array(emb));
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
      if (!readExportFile(embPath, "emb", ".emb", "application/octet-stream")) {
        await streamOut("Hi.", `${key}.wav`, 0.7, false);
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
    if (activeLat) api.latents_stop(activeLat);
  },

  async shutdown() {
    if (activeStream) api.stream_stop(activeStream);
    if (activeLat) api.latents_stop(activeLat);
    close();
  },

  async prewarm({ voiceKey }) {
    // Two tiny silent generations: warms the voice-conditioning KV cache,
    // ORT arenas, and hot wasm kernels before the user's first Speak.
    await serialize(async () => {
      await streamOut("Hi.", `${voiceKey}.wav`, 0.5, false);
      await streamOut("Ready to speak.", `${voiceKey}.wav`, 0.5, false);
      warmed.add(voiceKey);
    });
  },

  async prewarmVoice({ voiceKey }) {
    await serialize(() => warmVoice(voiceKey));
  },

  async speak({ text, voiceKey, temperature }) {
    await serialize(async () => {
      warmed.add(voiceKey);
      post("speakStarted", {});
      const squeeze = new SilenceSqueeze();
      let total = 0;
      const emit = (samples) => {
        const out = squeeze.push(samples);
        if (!out.length) return;
        total += out.length;
        const copy = out.slice();
        post("chunk", { samples: copy.buffer }, [copy.buffer]);
      };
      const emitter = new Emitter(emit);

      api.set_temperature(handle, temperature ?? 0.45);
      const lat = api.latents_start(handle, text, `${voiceKey}.wav`);
      if (!lat) throw new Error("latents_start failed");
      activeLat = lat;
      const fp = M._malloc(32 * 4 + 4);   // frame + flags in one allocation
      const flagsOff = 32 * 4;

      // staging holds up to a max decode batch; ort tensor wraps a subarray
      // (no copy) — safe because we await the run before reuse.
      const MAXB = 15;
      const staging = new Float32Array(MAXB * 32);
      let count = 0, target = 1;

      const decode = async (finalOfSentence) => {
        if (!count) return;
        const T = count;
        count = 0;
        emitter.pushDecoded(dec.run(staging, T), finalOfSentence);
        target = Math.min(target * 2, MAXB);
      };

      try {
        outer: for (;;) {
          // drain everything available before yielding
          for (;;) {
            const r = api.latents_poll(lat, fp, fp + flagsOff);
            if (r === 0) break outer;
            if (r === -1) break;
            const flags = M.HEAPU32[(fp + flagsOff) >> 2];
            if (flags & 1) {
              await decode(true);                       // sentence-final batch
              emitter.finishSentence(!!(flags & 2), !!(flags & 4));
              dec.reset();                              // fresh decoder state
              target = 1;                               // sentence-start latency
            } else {
              staging.set(M.HEAPF32.subarray(fp >> 2, (fp >> 2) + 32), count * 32);
              if (++count >= target) await decode(false);
            }
          }
          if (count >= target) await decode(false);
          else await sleep(2);
        }
      } finally {
        activeLat = 0;
        api.latents_end(lat);
        M._free(fp);
      }
      emit(squeeze.flush());
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
