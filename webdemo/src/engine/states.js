// Stateful session runner mirroring the C++ StateBufferIO/StatefulRunner
// semantics: state_* inputs zero-initialized (bool states true, dynamic dims
// empty), out_state_* outputs fed back each run, delta-KV outputs scattered
// into a persistent cache with circular wraparound at the step counter.

const CAPACITY = 1000;

function numel(dims) {
  return dims.reduce((a, b) => a * b, 1);
}

export class StateRunner {
  /** @param {import("onnxruntime-web").InferenceSession} session @param {any} ort */
  constructor(session, ort) {
    this.sess = session;
    this.ort = ort;
    this.stateNames = [];
    this.meta = new Map();
    session.inputNames.forEach((name, i) => {
      const md = session.inputMetadata[i];
      if (!name.startsWith("state_")) return;
      const type = md.type.replace("tensor(", "").replace(")", "");
      const dims = md.shape.map((d) => (typeof d === "number" && d > 0 ? d : 0));
      this.stateNames.push(name);
      this.meta.set(name, { type, initDims: dims, dynamic: dims.includes(0) });
    });

    // Delta-KV discovery (mirrors discover_delta_kv_states): f32 5D
    // [2,1,...] with a 1000-capacity axis whose out_state has a different
    // (dynamic) length on that axis; step counter is the next int64 scalar.
    this.delta = new Map(); // state name -> {axis, stepName}
    const outDims = new Map();
    session.outputNames.forEach((name, i) => {
      const md = session.outputMetadata[i];
      if (name.startsWith("out_state_")) outDims.set(name.slice(4), md.shape);
    });
    this.stateNames.forEach((name, idx) => {
      const m = this.meta.get(name);
      if (m.type !== "float32" || m.initDims.length !== 5) return;
      const axis = m.initDims.indexOf(CAPACITY);
      if (axis !== 2 && axis !== 3) return;
      const od = outDims.get(name);
      if (!od || od[axis] === CAPACITY) return; // full output, not delta
      for (const j of [1, 2]) {
        const cand = this.stateNames[idx + j];
        if (cand && this.meta.get(cand).type === "int64" && numel(this.meta.get(cand).initDims) === 1) {
          this.delta.set(name, { axis, stepName: cand });
          break;
        }
      }
    });
    this.reset();
  }

  zerosTensor(dims, type) {
    const n = numel(dims);
    if (type === "int64") return new this.ort.Tensor("int64", new BigInt64Array(n), dims);
    if (type === "bool") return new this.ort.Tensor("bool", new Uint8Array(n).fill(1), dims);
    if (type === "float16") return new this.ort.Tensor("float16", new Uint16Array(n), dims);
    return new this.ort.Tensor("float32", new Float32Array(n), dims);
  }

  reset() {
    /** @type {Map<string, import("onnxruntime-web").Tensor>} */
    this.states = new Map();
    for (const name of this.stateNames) {
      const m = this.meta.get(name);
      this.states.set(name, this.zerosTensor(m.initDims, m.type));
    }
  }

  snapshot() {
    const snap = new Map();
    for (const [name, t] of this.states) {
      snap.set(name, new this.ort.Tensor(t.type, t.data.slice(), [...t.dims]));
    }
    return snap;
  }

  restore(snap) {
    this.states = new Map();
    for (const [name, t] of snap) {
      this.states.set(name, new this.ort.Tensor(t.type, t.data.slice(), [...t.dims]));
    }
  }

  /** @param {Record<string, import("onnxruntime-web").Tensor>} inputs */
  async run(inputs) {
    const feeds = { ...inputs };
    for (const [name, t] of this.states) feeds[name] = t;
    const out = await this.sess.run(feeds);
    const oldSteps = new Map();
    for (const [, d] of this.delta) {
      oldSteps.set(d.stepName, Number(this.states.get(d.stepName).data[0]));
    }
    for (const [key, val] of Object.entries(out)) {
      if (!key.startsWith("out_state_")) continue;
      const name = key.slice(4);
      if (!this.states.has(name)) continue;
      const d = this.delta.get(name);
      if (!d) {
        this.states.set(name, val);
        continue;
      }
      // delta scatter into persistent cache
      const cache = this.states.get(name);
      const dst = cache.data;
      const src = val.data;
      const full = cache.dims;
      const L = val.dims[d.axis];
      const oldStep = oldSteps.get(d.stepName);
      const newStepT = out["out_" + d.stepName];
      const inner = full.slice(d.axis + 1).reduce((a, b) => a * b, 1);
      const outer = full.slice(0, d.axis).reduce((a, b) => a * b, 1);
      const fullStride = CAPACITY * inner;
      const deltaStride = L * inner;
      let start = oldStep % CAPACITY;
      if (start < 0) start += CAPACITY;
      for (let o = 0; o < outer; o++) {
        if (start + L <= CAPACITY) {
          dst.set(src.subarray(o * deltaStride, (o + 1) * deltaStride), o * fullStride + start * inner);
        } else {
          const first = CAPACITY - start;
          dst.set(src.subarray(o * deltaStride, o * deltaStride + first * inner), o * fullStride + start * inner);
          dst.set(src.subarray(o * deltaStride + first * inner, (o + 1) * deltaStride), o * fullStride);
        }
      }
      void newStepT; // counter itself is fed back via its own out_state_
    }
    const result = {};
    for (const [key, val] of Object.entries(out)) {
      if (!key.startsWith("out_state_")) result[key] = val;
    }
    return result;
  }
}
