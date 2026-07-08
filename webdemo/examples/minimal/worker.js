// Minimal PocketTTS WASM worker: load models, one preset voice, speak on
// request. See ../../src/tts-worker-native.js for the full-featured version
// (cloning, prewarm, stop, memory-variant fallback).
import createPocketTTS from "../../vendor/ptt/pocket_tts_wasm.mjs";
import { Tokenizer } from "../../src/engine/tokenizer.js";

const BASE = new URL("../../", import.meta.url).href; // -> webdemo/
const post = (type, payload = {}, transfer = []) =>
  self.postMessage({ type, ...payload }, transfer);

const M = await createPocketTTS();
M.FS.mkdir("/models"); M.FS.mkdir("/voices"); M.FS.mkdir("/voices/.cache");

// models into the module's virtual filesystem
for (const f of ["flow_lm_main_delta_attn_flow_int8.onnx",
                 "mimi_decoder_delta_int8.onnx", "text_conditioner.onnx",
                 "bos_before_voice.npy"]) {
  post("progress", { label: `downloading ${f}…` });
  M.FS.writeFile(`/models/${f}`,
    new Uint8Array(await (await fetch(`${BASE}models/${f}`)).arrayBuffer()));
}

// tokenization is delegated to JS (the engine calls self.__pkttsTokenize)
const vocab = await (await fetch(`${BASE}models/spm_vocab.json`)).json();
const tok = new Tokenizer(vocab);
self.__pkttsTokenize = (t) => tok.encode(t);

// preset voice: a precomputed embedding drops straight into the cache
M.FS.writeFile("/voices/.cache/alba.emb",
  new Uint8Array(await (await fetch(`${BASE}presets/alba.emb`)).arrayBuffer()));

// engine: flags bit1 = defer encoder (only needed for cloning)
M.cwrap("ptt_configure_pool", null, ["number", "number"])(4, 1);
const h = M.cwrap("ptt_create_ex", "number",
  ["string", "string", "string", "string", "number", "number", "number", "number"])(
  "/models", "/voices", "/models/tokenizer.model", "int8", 0.7, 1, 4, 2);
if (!h) throw new Error("engine init failed");

const start = M.cwrap("ptt_stream_start", "number", ["number", "string", "string"]);
const poll = M.cwrap("ptt_stream_poll", "number", ["number", "number", "number"]);
const end = M.cwrap("ptt_stream_end", null, ["number"]);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
post("ready");

self.onmessage = async (e) => {
  try {
    const s = start(h, e.data.text, "alba.wav"); // resolves via the .emb cache
    const pp = M._malloc(8);
    for (;;) {
      const r = poll(s, pp, pp + 4);              // NEVER use blocking read here
      if (r === 0) break;
      if (r === -1) { await sleep(2); continue; }
      const ptr = M.HEAPU32[pp >> 2], n = M.HEAPU32[(pp + 4) >> 2];
      const chunk = M.HEAPF32.slice(ptr >> 2, (ptr >> 2) + n);
      M._free(ptr);
      post("chunk", { samples: chunk.buffer }, [chunk.buffer]);
    }
    end(s); M._free(pp);
    post("done");
  } catch (err) {
    post("error", { message: String(err?.message || err) });
  }
};
