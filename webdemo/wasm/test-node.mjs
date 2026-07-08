// Headless smoke + benchmark of the Emscripten engine under Node.
//   node webdemo/wasm/test-node.mjs
import { readFileSync, writeFileSync } from "node:fs";
const VARIANT = process.env.PTT_VARIANT === "growth" ? "pocket_tts_wasm_growth"
  : process.env.PTT_VARIANT === "relaxed" ? "pocket_tts_wasm_relaxed" : "pocket_tts_wasm";
const { default: createPocketTTS } = await import(`../vendor/ptt/${VARIANT}.mjs`);
import { Tokenizer } from "../src/engine/tokenizer.js";

const dir = new URL(".", import.meta.url).pathname;
const models = process.env.PTT_MODELS_DIR || `${dir}/../models`;

globalThis.self = globalThis.self || globalThis;
const vocab = JSON.parse(readFileSync(`${models}/spm_vocab.json`, "utf8"));
const tok = new Tokenizer(vocab);
self.__pkttsTokenize = (t) => tok.encode(t);

const POOL = Number(process.env.PTT_POOL || 4);
const SPIN = process.env.PTT_SPIN || "1";
const t0 = performance.now();
const M = await createPocketTTS();
console.log(`pool=${POOL} spin=${SPIN}`);
M.FS.mkdir("/models"); M.FS.mkdir("/voices"); M.FS.mkdir("/voices/.cache");
for (const f of ["flow_lm_main_delta_attn_flow_int8.onnx", "mimi_encoder.onnx",
                 "mimi_decoder_delta_int8.onnx", "text_conditioner.onnx",
                 "bos_before_voice.npy"]) {
  M.FS.writeFile(`/models/${f}`, readFileSync(`${models}/${f}`));
}
M.FS.writeFile("/voices/.cache/alba.emb", readFileSync(`${dir}/../presets/alba.emb`));

const api = {
  create: M.cwrap("ptt_create", "number", ["string","string","string","string","number","number","number"]),
  set_temperature: M.cwrap("ptt_set_temperature", null, ["number","number"]),
  stream_start: M.cwrap("ptt_stream_start", "number", ["number","string","string"]),
  stream_poll: M.cwrap("ptt_stream_poll", "number", ["number","number","number"]),
  stream_end: M.cwrap("ptt_stream_end", null, ["number"]),
  set_profiling: M.cwrap("ptt_set_profiling", null, ["number"]),
  print_profile: M.cwrap("ptt_print_profile", null, []),
};
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const tCreate0 = performance.now();
M.cwrap("ptt_configure_pool", null, ["number", "number"])(POOL, Number(SPIN));
let h;
if (process.env.PTT_OP_PROFILE) {
  M.FS.mkdir("/prof");
  h = M.cwrap("ptt_create_profiled", "number",
    ["string","string","string","string","number","number","number","string"])(
    "/models", "/voices", "/models/tokenizer.model", "int8", 0.7, 1, POOL, "/prof");
} else {
  h = api.create("/models", "/voices", "/models/tokenizer.model", "int8", 0.7, 1, POOL);
}
if (process.env.PTT_CHUNK) {
  M.cwrap("ptt_set_max_chunk", null, ["number", "number"])(h, Number(process.env.PTT_CHUNK));
}
if (!h) throw new Error("ptt_create failed");
console.log(`module+create: ${(performance.now() - t0).toFixed(0)}ms (create ${(performance.now() - tCreate0).toFixed(0)}ms)`);

async function speak(text, temp, label) {
  api.set_temperature(h, temp);
  const t1 = performance.now();
  const ctx = api.stream_start(h, text, "alba.wav");
  const pp = M._malloc(8);
  let total = 0, first = 0;
  const parts = [];
  for (;;) {
    const r = api.stream_poll(ctx, pp, pp + 4);
    if (r === 0) break;
    if (r === -1) { await sleep(1); continue; }
    const ptr = M.HEAPU32[pp >> 2], len = M.HEAPU32[(pp + 4) >> 2];
    if (!first) first = performance.now();
    parts.push(M.HEAPF32.slice(ptr >> 2, (ptr >> 2) + len));
    total += len;
    M._free(ptr);
  }
  api.stream_end(ctx);
  M._free(pp);
  const wall = (performance.now() - t1) / 1000;
  const dur = total / 24000;
  console.log(`${label}: ${dur.toFixed(2)}s audio in ${wall.toFixed(2)}s ` +
    `(RTFx ${(dur / wall).toFixed(1)}x, first audio ${(first - t1).toFixed(0)}ms)`);
  return parts;
}

if (process.env.PTT_BENCH_ONLY) {
  const db = M.cwrap("ptt_debug_bench", "number", ["number","number","number","number"]);
  db(h, 0, 15, 2); // decoder warmup initializes shared machinery
  db(h, 1, 0, 8);
  console.log(`AR step: ${db(h, 1, 0, 60).toFixed(3)} ms`);
  process.exit(0);
}
await speak("Hi.", 0, "warmup");

// Split-module mock mode: after warmup, run ONLY a timed bench loop of one
// role and emit JSON (driven by mock-split.mjs; no effect otherwise).
if (process.env.PTT_BENCH_LOOP) {
  const role = process.env.PTT_BENCH_LOOP;
  const secs = Number(process.env.PTT_BENCH_SECS || 12);
  const startAt = Number(process.env.PTT_BENCH_START || 0);
  const db = M.cwrap("ptt_debug_bench", "number", ["number","number","number","number"]);
  if (role === "ar") db(h, 1, 0, 8); else db(h, 0, 15, 2); // bench warmup
  console.log("READY");
  while (Date.now() < startAt) await sleep(5);
  const t0 = performance.now();
  let calls = 0, acc = 0;
  while (performance.now() - t0 < secs * 1000) {
    acc += role === "ar" ? db(h, 1, 0, 20) : db(h, 0, 15, 4);
    calls++;
  }
  console.log(JSON.stringify({ role, avg: acc / calls, calls }));
  process.exit(0);
}
const dbench = M.cwrap("ptt_debug_bench", "number", ["number","number","number","number"]);
console.log(`ISOLATED ar_step: ${dbench(h, 1, 0, 60).toFixed(2)} ms/step`);
console.log(`ISOLATED dec 1f : ${dbench(h, 0, 1, 30).toFixed(2)} ms/run`);
console.log(`ISOLATED dec 15f: ${dbench(h, 0, 15, 15).toFixed(2)} ms/run`);
api.set_profiling(1);
const parts = await speak(
  "Beware: the nearest mirror hides a portal to the abyss—look, but don't step!",
  0, "bench t0");
await speak("How are things over in Riverwood these days?", 0.7, "bench t0.7");
if (process.env.PTT_PROFILE) api.set_profiling(1);
await speak("The road ahead is more dangerous than it looks. Bandits have taken the old watchtower, and the bridge is out past the mill. If you mean to reach Whiterun before nightfall, take the eastern path through the pines. And keep your blade loose in its sheath.", 0, "bench long");
const runon = await speak("The caravan left at dawn with twelve guards, four wagons loaded with iron and salt, a cook who never stopped complaining, two scouts who rode ahead through the pines, and an old map that nobody fully trusted, which is why we followed the river instead of the mountain pass that winter.", 0, "bench runon");
{ // interior-gap analysis on the run-on (multi-chunk) render
  const all = new Float32Array(runon.reduce((a, c) => a + c.length, 0));
  let o = 0; for (const c of runon) { all.set(c, o); o += c.length; }
  const win = 240, n = Math.floor(all.length / win);
  let gaps = 0, run = 0;
  for (let i = 0; i < n; i++) {
    let s2 = 0;
    for (let j = i * win; j < (i + 1) * win; j++) s2 += all[j] * all[j];
    if (Math.sqrt(s2 / win) < 0.004) run++;
    else { if (run >= 15 && i * 10 > 300 && i * 10 < n * 10 - 300) gaps++; run = 0; }
  }
  console.log(`runon interior gaps >=150ms: ${gaps}`);
if (process.env.PTT_PROFILE) api.print_profile();
}


// write output for an ear check
const all = new Float32Array(parts.reduce((a, c) => a + c.length, 0));
let o = 0; for (const c of parts) { all.set(c, o); o += c.length; }
const pcm = new Int16Array(all.length);
for (let i = 0; i < all.length; i++) pcm[i] = Math.max(-32768, Math.min(32767, Math.round(all[i] * 32767)));
const hdr = Buffer.alloc(44);
hdr.write("RIFF", 0); hdr.writeUInt32LE(36 + pcm.byteLength, 4); hdr.write("WAVE", 8);
hdr.write("fmt ", 12); hdr.writeUInt32LE(16, 16); hdr.writeUInt16LE(1, 20); hdr.writeUInt16LE(1, 22);
hdr.writeUInt32LE(24000, 24); hdr.writeUInt32LE(48000, 28); hdr.writeUInt16LE(2, 32); hdr.writeUInt16LE(16, 34);
hdr.write("data", 36); hdr.writeUInt32LE(pcm.byteLength, 40);
writeFileSync("/private/tmp/ptt-emscripten-test.wav", Buffer.concat([hdr, Buffer.from(pcm.buffer)]));
console.log("wrote /private/tmp/ptt-emscripten-test.wav");
console.log(`variant=${VARIANT} peak heap: ${(M.HEAPU8.length / 1048576).toFixed(0)}MB`);
if (process.env.PTT_OP_PROFILE) {
  M.cwrap("ptt_destroy", null, ["number"])(h); // flushes ORT profile JSONs
  for (const f of M.FS.readdir("/prof")) {
    if (!f.endsWith(".json") || !f.includes("main")) continue;
    const events = JSON.parse(new TextDecoder().decode(M.FS.readFile(`/prof/${f}`)));
    const agg = new Map();
    for (const e of events) {
      if (e.cat !== "Node" || !e.args?.op_name) continue;
      const k = e.args.op_name;
      const a = agg.get(k) ?? { us: 0, n: 0 };
      a.us += e.dur; a.n++;
      agg.set(k, a);
    }
    const total = [...agg.values()].reduce((x, a) => x + a.us, 0);
    console.log(`\n== ${f} (op time ${(total/1000).toFixed(0)}ms) ==`);
    [...agg.entries()].sort((a, b) => b[1].us - a[1].us).slice(0, 14).forEach(([op, a]) =>
      console.log(`${op.padEnd(26)} ${(a.us/1000).toFixed(1).padStart(8)}ms  ${String(a.n).padStart(7)}x  ${(100*a.us/total).toFixed(1).padStart(5)}%`));
  }
}
process.exit(0);
