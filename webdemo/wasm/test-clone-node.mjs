// Replicates the browser worker's LAZY-ENCODER clone lifecycle exactly:
// create(defer) -> presets -> prewarm -> load encoder -> clone -> speak.
import { readFileSync } from "node:fs";
import { Tokenizer } from "../src/engine/tokenizer.js";

const dir = new URL(".", import.meta.url).pathname;
const models = `${dir}/../models`;
globalThis.self = globalThis.self || globalThis;
const vocab = JSON.parse(readFileSync(`${models}/spm_vocab.json`, "utf8"));
const tok = new Tokenizer(vocab);
self.__pkttsTokenize = (t) => tok.encode(t);

const VAR = process.env.PTT_VARIANT === "growth" ? "pocket_tts_wasm_growth" : "pocket_tts_wasm";
const { default: createPocketTTS } = await import(`../vendor/ptt/${VAR}.mjs`);
const M = await createPocketTTS();
const heap = () => (M.HEAPU8.length / 1048576).toFixed(0) + "MB heap";
M.FS.mkdir("/models"); M.FS.mkdir("/voices"); M.FS.mkdir("/voices/.cache");
for (const f of ["flow_lm_main_delta_attn_flow_int8.onnx", "mimi_decoder_delta_int8.onnx",
                 "text_conditioner.onnx", "bos_before_voice.npy"]) {
  M.FS.writeFile(`/models/${f}`, readFileSync(`${models}/${f}`));
}
M.FS.writeFile("/voices/.cache/alba.emb", readFileSync(`${dir}/../presets/alba.emb`));
M.cwrap("ptt_configure_pool", null, ["number", "number"])(10, 1);
const h = M.cwrap("ptt_create_ex", "number",
  ["string","string","string","string","number","number","number","number"])(
  "/models", "/voices", "/models/tokenizer.model", "int8", 0.5, 1, 10, 2 /*defer enc*/);
if (!h) throw new Error("create_ex failed");
console.log("created (deferred encoder),", heap());

const api = {
  set_temperature: M.cwrap("ptt_set_temperature", null, ["number","number"]),
  stream_start: M.cwrap("ptt_stream_start", "number", ["number","string","string"]),
  stream_poll: M.cwrap("ptt_stream_poll", "number", ["number","number","number"]),
  stream_end: M.cwrap("ptt_stream_end", null, ["number"]),
};
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function speak(text, voice, label) {
  api.set_temperature(h, 0.5);
  const ctx = api.stream_start(h, text, voice);
  const pp = M._malloc(8);
  let total = 0;
  for (;;) {
    const r = api.stream_poll(ctx, pp, pp + 4);
    if (r === 0) break;
    if (r === -1) { await sleep(2); continue; }
    total += M.HEAPU32[(pp + 4) >> 2];
    M._free(M.HEAPU32[pp >> 2]);
  }
  api.stream_end(ctx); M._free(pp);
  console.log(`${label}: ${(total/24000).toFixed(2)}s audio, ${heap()}`);
  return total;
}

// prewarm exactly like the worker
await speak("Hi.", "alba.wav", "prewarm-1");
await speak("Ready to speak.", "alba.wav", "prewarm-2");

// disposable encoder-only instance (mirrors encode-worker.js)
async function encodeExternal(key, wavBytes) {
  const E = await createPocketTTS(); // growth-or-fixed irrelevant for the test
  E.FS.mkdir("/models"); E.FS.mkdir("/voices"); E.FS.mkdir("/voices/.cache");
  E.FS.writeFile("/models/mimi_encoder.onnx", readFileSync(`${models}/mimi_encoder.onnx`));
  E.FS.writeFile("/models/bos_before_voice.npy", readFileSync(`${models}/bos_before_voice.npy`));
  E.FS.writeFile(`/voices/${key}.wav`, wavBytes);
  E.cwrap("ptt_configure_pool", null, ["number", "number"])(4, 1);
  const eh = E.cwrap("ptt_create_ex", "number",
    ["string","string","string","string","number","number","number","number"])(
    "/models", "/voices", "/models/tokenizer.model", "int8", 0.5, 1, 4, 4);
  if (!eh) throw new Error("encoder-only create failed");
  const rc = E.cwrap("ptt_encode_voice", "number", ["number","string"])(eh, `${key}.wav`);
  if (rc !== 0) throw new Error("encode failed");
  const emb = E.FS.readFile(`/voices/.cache/${key}.emb`);
  console.log(`external encode ok (${(emb.length/1e6).toFixed(1)}MB emb), encode-instance heap ${(E.HEAPU8.length/1048576).toFixed(0)}MB`);
  return emb;
}

// worst-case clone: 30s of speech-like modulated audio
const SR = 24000, n = Number(process.env.PTT_CLONE_SECS || 30) * SR;
const audio = new Float32Array(n);
for (let i = 0; i < n; i++) {
  const t = i / SR;
  audio[i] = 0.22 * Math.sin(2 * Math.PI * (130 + 25 * Math.sin(2 * Math.PI * 0.7 * t)) * t)
           * (0.55 + 0.45 * Math.sin(2 * Math.PI * 3.3 * t));
}
const bytes = n * 4, head = new DataView(new ArrayBuffer(44));
const w = (o, t2) => { for (let k = 0; k < t2.length; k++) head.setUint8(o + k, t2.charCodeAt(k)); };
w(0,"RIFF"); head.setUint32(4, 36 + bytes, true); w(8,"WAVE");
w(12,"fmt "); head.setUint32(16,16,true); head.setUint16(20,3,true); head.setUint16(22,1,true);
head.setUint32(24,SR,true); head.setUint32(28,SR*4,true); head.setUint16(32,4,true); head.setUint16(34,32,true);
w(36,"data"); head.setUint32(40,bytes,true);
const wav = new Uint8Array(44 + bytes);
wav.set(new Uint8Array(head.buffer));
wav.set(new Uint8Array(audio.buffer), 44);
M.FS.writeFile("/voices/clone-test.wav", wav);
const embBytes = await encodeExternal("clone-test", wav);
M.FS.writeFile("/voices/.cache/clone-test.emb", embBytes);

const got = await speak("Hi.", "clone-test.wav", "clone-warm (30s source)");
const got2 = await speak("The crypt accepts your voice.", "clone-test.wav", "clone-speak");
console.log(got > 0 && got2 > 0 ? "CLONE PATH OK" : "CLONE PATH FAILED");
process.exit(got > 0 && got2 > 0 ? 0 : 1);
