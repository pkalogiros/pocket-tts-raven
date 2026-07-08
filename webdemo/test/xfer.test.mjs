// Cross-runtime single-step check: load python's post-conditioning states
// into the JS StateRunner, run one AR step, compare latent + eos.
import { readFileSync } from "node:fs";
import * as ort from "onnxruntime-web";
import { StateRunner } from "../src/engine/states.js";

const dir = new URL(".", import.meta.url).pathname;
const meta = JSON.parse(readFileSync(`${dir}/xfer_meta.json`, "utf8"));
const blob = readFileSync(`${dir}/xfer_states.bin`);

ort.env.wasm.numThreads = 4;
const sess = await ort.InferenceSession.create(`${dir}/../models/flow_lm_main_delta_flow_int8.onnx`,
  { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
const r = new StateRunner(sess, ort);

for (const s of meta.states) {
  const buf = blob.buffer.slice(blob.byteOffset + s.offset, blob.byteOffset + s.offset + s.bytes);
  let t;
  if (s.dtype === "int64") t = new ort.Tensor("int64", new BigInt64Array(buf), s.dims);
  else if (s.dtype === "bool") t = new ort.Tensor("bool", new Uint8Array(buf), s.dims);
  else if (s.dtype === "float16") t = new ort.Tensor("float16", new Uint16Array(buf), s.dims);
  else t = new ort.Tensor("float32", new Float32Array(buf), s.dims);
  r.states.set(s.name, t);
}

const cl = new Float32Array(32).fill(NaN);
const out = await r.run({
  sequence: new ort.Tensor("float32", cl, [1, 1, 32]),
  text_embeddings: new ort.Tensor("float32", new Float32Array(0), [1, 0, 1024]),
  flow_x: new ort.Tensor("float32", new Float32Array(32), [1, 32]),
});
let worst = 0;
for (let i = 0; i < 32; i++) worst = Math.max(worst, Math.abs(out.latent.data[i] - meta.latent[i]));
const eosD = Math.abs(out.eos_logit.data[0] - meta.eos);
console.log(`same-state single step: latent max|Δ|=${worst.toExponential(2)} eosΔ=${eosD.toExponential(2)}`);
console.log(worst < 1e-3 && eosD < 1e-2 ? "ORCHESTRATION OK (divergence is platform fp)" : "ORCHESTRATION BUG");
