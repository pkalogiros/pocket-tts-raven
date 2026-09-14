// PocketTTS synthesis engine on onnxruntime-web — a faithful port of the
// orchestration in pocket_tts.cpp (LatentGen + stream()): voice conditioning
// with KV snapshots, merged-flow AR loop, chunked streaming decode with
// crossfades, and the leading-trim gate with the sustain/burst check.

import { controls, parseVectors, applyShift } from "./steering.js";
import { StateRunner } from "./states.js";
import { Tokenizer } from "./tokenizer.js";
import { splitSentences, prepareText, isTerminalBoundary } from "./text.js";

export const SR = 24000;
const FRAME = 1920; // samples per latent frame

// Emission constants (mirror stream())
const DECODER_XFADE = 120, SPLIT_XFADE = 960, END_TRIM = 480;
const SENTENCE_FADE = 720, SENTENCE_PAUSE = 6360, CHUNK_PAUSE = 3600;
const LEAD_WIN = 240, LEAD_DROP = 480, LEAD_PREROLL = 2, LEAD_MAX = 24000;
const LEAD_SUSTAIN = 15, LEAD_COLLAPSE = 3, LEAD_CONFIRM = 2;
const RMS_T = 0.006, PEAK_T = 0.025;

const smooth = (t) => t * t * (3 - 2 * t);

function randn(out, stddev, rand) {
  for (let i = 0; i < out.length; i += 1) {
    let u1 = rand();
    while (u1 <= 1e-10) u1 = rand();
    out[i] = stddev * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * rand());
  }
}

export function parseNpyF32(buf) {
  const dv = new DataView(buf);
  const headerLen = dv.getUint16(8, true);
  const header = new TextDecoder().decode(new Uint8Array(buf, 10, headerLen));
  const shape = header.match(/\(([^)]*)\)/)[1].split(",").map(s => parseInt(s)).filter(n => !isNaN(n));
  return { shape, data: new Float32Array(buf, 10 + headerLen) };
}

/** Streaming emitter: trim gate (with sustain/carry) + fades + crossfades. */
export class Emitter {
  constructor(onChunk) {
    this.onChunk = onChunk;
    this.pendingTail = new Float32Array(0);
    this.fadeTotal = SENTENCE_FADE;
    this.fadePos = 0;
    this.resetGate();
    this.emittedAny = false;
  }
  resetGate() {
    this.trimming = true;
    this.seen = 0; this.hits = 0;
    this.candidate = []; this.preroll = []; this.carry = new Float32Array(0);
    this.candWindows = 0; this.quietRun = 0;
  }
  fadeIn(data) {
    if (this.fadePos >= this.fadeTotal) return;
    const fade = Math.min(data.length, this.fadeTotal - this.fadePos);
    for (let i = 0; i < fade; i++) data[i] *= smooth((this.fadePos + i) / this.fadeTotal);
    this.fadePos += data.length;
  }
  emitRaw(data) {
    if (!data.length) return;
    this.fadeIn(data);
    this.emittedAny = true;
    this.onChunk(data);
  }
  openGate() {
    this.trimming = false;
    this.fadeTotal = SENTENCE_FADE;
    this.fadePos = 0;
  }
  gate(data) {
    if (!this.trimming) return data;
    let buf = data;
    if (this.carry.length) {
      buf = new Float32Array(this.carry.length + data.length);
      buf.set(this.carry); buf.set(data, this.carry.length);
      this.carry = new Float32Array(0);
    }
    let pos = 0;
    if (this.seen < LEAD_DROP) {
      const drop = Math.min(buf.length, LEAD_DROP - this.seen);
      this.seen += drop; pos += drop;
    }
    while (pos + LEAD_WIN <= buf.length && this.trimming) {
      let sumSq = 0, peak = 0;
      for (let i = 0; i < LEAD_WIN; i++) {
        const s = buf[pos + i], a = Math.abs(s);
        if (a > peak) peak = a;
        sumSq += s * s;
      }
      const speechy = Math.sqrt(sumSq / LEAD_WIN) >= RMS_T && peak >= PEAK_T;
      if (this.hits === 0 && !speechy) {
        this.preroll.push(buf.slice(pos, pos + LEAD_WIN));
        while (this.preroll.length > LEAD_PREROLL) this.preroll.shift();
      } else {
        if (this.hits === 0) {
          this.candidate = this.preroll.splice(0);
          this.candWindows = 0; this.quietRun = 0;
        }
        this.candidate.push(buf.slice(pos, pos + LEAD_WIN));
        this.candWindows++;
        if (speechy) { this.hits++; this.quietRun = 0; }
        else if (++this.quietRun >= LEAD_COLLAPSE) {
          this.candidate = []; this.hits = 0; this.candWindows = 0; this.quietRun = 0;
        }
        if (this.hits >= LEAD_CONFIRM && this.candWindows >= LEAD_SUSTAIN) {
          this.openGate();
          pos += LEAD_WIN; this.seen += LEAD_WIN;
          break;
        }
      }
      pos += LEAD_WIN; this.seen += LEAD_WIN;
      if (this.seen >= LEAD_MAX) { this.openGate(); break; }
    }
    if (this.trimming) {
      this.carry = buf.slice(pos);
      return new Float32Array(0);
    }
    if (this.candidate.length) {
      const cand = new Float32Array(this.candidate.reduce((a, c) => a + c.length, 0));
      let o = 0;
      for (const c of this.candidate) { cand.set(c, o); o += c.length; }
      this.candidate = [];
      this.emitRaw(cand);
    }
    return buf.slice(pos);
  }
  /** Decoded chunk from the decoder; final marks the sentence's last batch. */
  pushDecoded(chunk, final) {
    let data = this.gate(chunk);
    const n = data.length;
    if (!n) return;
    const keep = Math.min(final ? SPLIT_XFADE + END_TRIM : DECODER_XFADE, n);
    let start = 0;
    if (this.pendingTail.length) {
      const fade = Math.min(this.pendingTail.length, n);
      const x = new Float32Array(fade);
      for (let i = 0; i < fade; i++) {
        const t = smooth(fade > 1 ? i / (fade - 1) : 1);
        x[i] = this.pendingTail[i] * (1 - t) + data[i] * t;
      }
      this.emitRaw(x);
      start = fade;
    }
    const end = n - keep;
    if (end > start) this.emitRaw(data.slice(start, end));
    let tailEnd = n;
    if (final) tailEnd -= Math.min(END_TRIM, tailEnd - end);
    this.pendingTail = data.slice(Math.max(end, 0), tailEnd);
  }
  flushCandidate() {
    if (!this.trimming || !this.candidate.length) return;
    this.openGate();
    const cand = new Float32Array(this.candidate.reduce((a, c) => a + c.length, 0));
    let o = 0;
    for (const c of this.candidate) { cand.set(c, o); o += c.length; }
    this.candidate = [];
    this.emitRaw(cand);
  }
  finishSentence(terminal, lastOfText) {
    this.flushCandidate();
    if (this.pendingTail.length) {
      const t = this.pendingTail;
      const inv = t.length > 1 ? 1 / (t.length - 1) : 1;
      for (let i = 0; i < t.length; i++) t[i] *= 1 - smooth(i * inv);
      this.emitRaw(t);
      this.pendingTail = new Float32Array(0);
    }
    if (!lastOfText) {
      // Full pause at sentence ends; short clause pause at mid-sentence
      // split points so comma joins keep natural phrasing (matches C++).
      this.onChunk(new Float32Array(terminal ? SENTENCE_PAUSE : CHUNK_PAUSE));
      this.fadeTotal = SENTENCE_FADE;
      this.fadePos = 0;
      this.resetGate();
    }
  }
}

export class PocketTTS {
  /** @param {{ort: any, modelsUrl: string, threads?: number, buffers?: Record<string, Uint8Array>, loader?: {json:(u:string)=>Promise<any>, buffer:(u:string)=>Promise<ArrayBuffer>}}} opts */
  static async create({ ort, modelsUrl, threads = 4, buffers, loader, soura = false, vectorsUrl, keepCommas = true }) {
    ort.env.wasm.numThreads = threads;
    const load = loader ?? {
      json: async (u) => (await fetch(u)).json(),
      buffer: async (u) => (await fetch(u)).arrayBuffer(),
    };
    const self = new PocketTTS();
    self.ort = ort;
    self.soura = soura === true; self.keepCommas = keepCommas;
    self.shift = new Float32Array(1024);
    self.shiftTensor = new ort.Tensor("float32", self.shift, [1, 1, 1024]);
    const so = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
    const src = (name) => (buffers && buffers[name]) || `${modelsUrl}/${name}`;
    const [main, dec, txt, vocab, bosBuf] = await Promise.all([
      ort.InferenceSession.create(src(self.soura ? "flow_lm_main_delta_flow_int8_soura.onnx" : "flow_lm_main_delta_flow_int8.onnx"), so),
      ort.InferenceSession.create(src("mimi_decoder_delta_int8.onnx"), so),
      ort.InferenceSession.create(src("text_conditioner.onnx"), so),
      load.json(`${modelsUrl}/spm_vocab.json`),
      load.buffer(`${modelsUrl}/bos_before_voice.npy`),
    ]);
    if (self.soura) self.vectors = parseVectors(await load.buffer(vectorsUrl || `${modelsUrl}/soura_vectors.npy`));
    self.main = new StateRunner(main, ort);
    self.dec = new StateRunner(dec, ort);
    self.txt = txt;
    self.tok = new Tokenizer(vocab);
    self.bos = parseNpyF32(bosBuf); // [1,1,1024]
    self.encoder = null; // lazy — only needed for cloning
    self.modelsUrl = modelsUrl;
    self.voiceSnapshots = new Map();
    return self;
  }

  loadVectors(buffer) {
    if (!this.soura) throw new Error("Enable steering before loading vectors");
    this.vectors = parseVectors(buffer);
    this.shift.fill(0);
  }

  async loadEncoder() {
    if (!this.encoder) {
      this.encoder = await this.ort.InferenceSession.create(
        `${this.modelsUrl}/mimi_encoder.onnx`,
        { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    }
    return this.encoder;
  }

  /** 24kHz mono Float32Array -> voice embedding [1,T,1024] with BOS prefix. */
  async encodeVoice(audio24k) {
    const enc = await this.loadEncoder();
    const capped = audio24k.length > 30 * SR ? audio24k.subarray(0, 30 * SR) : audio24k;
    const input = new this.ort.Tensor("float32", capped, [1, 1, capped.length]);
    const out = await enc.run({ [enc.inputNames[0]]: input });
    const emb = out[enc.outputNames[0]];
    let dims = emb.dims.filter((d, i) => !(d === 1 && i < emb.dims.length - 2));
    if (dims.length < 3) dims = [1, ...dims];
    return this.withBos({ dims: [1, dims[dims.length - 2], 1024], data: emb.data });
  }

  hasBos(v) {
    const b = this.bos.data;
    if (v.data.length < b.length) return false;
    for (let i = 0; i < b.length; i++) if (v.data[i] !== b[i]) return false;
    return true;
  }

  withBos(v) {
    if (this.hasBos(v)) return v;
    const data = new Float32Array(this.bos.data.length + v.data.length);
    data.set(this.bos.data);
    data.set(v.data, this.bos.data.length);
    return { dims: [1, v.dims[1] + 1, 1024], data };
  }

  zeroX() { return new this.ort.Tensor("float32", new Float32Array(32), [1, 32]); }
  emptySeq() { return new this.ort.Tensor("float32", new Float32Array(0), [1, 0, 32]); }
  emptyText() { return new this.ort.Tensor("float32", new Float32Array(0), [1, 0, 1024]); }

  async condPass(data, dims) {
    await this.main.run({
      ...(this.soura ? { soura_shift: this.shiftTensor } : {}),
      sequence: this.emptySeq(),
      text_embeddings: new this.ort.Tensor("float32", data, dims),
      flow_x: this.zeroX(),
    });
  }

  /** Run voice conditioning once and cache the KV snapshot under `key`. */
  async prepareVoice(key, embedding) {
    this.shift.fill(0);
    this.main.reset();
    await this.condPass(embedding.data, embedding.dims);
    this.voiceSnapshots.set(key, this.main.snapshot());
  }

  hasVoice(key) { return this.voiceSnapshots.has(key); }

  /**
   * Synthesize text; onChunk receives Float32Array pieces as they're ready.
   * @param {{temperature?: number, maxFrames?: number, seed?: number, onProgress?: (f:number)=>void}} opts
   */
  cancel() { this._cancel = true; }

  async speak(text, voiceKey, onChunk, opts = {}) {
    const selected = controls(this.soura, opts);
    if (this.soura) applyShift(this.shift, this.vectors, selected.emotion, selected.intensity);
    this._cancel = false;
    const temperature = opts.temperature ?? 0.45;
    const maxFrames = opts.maxFrames ?? 500;
    const snap = this.voiceSnapshots.get(voiceKey);
    if (!snap) throw new Error(`voice not prepared: ${voiceKey}`);
    let s = opts.seed ?? (Date.now() & 0xffffffff);
    const rand = () => {
      s = (s + 0x6d2b79f5) | 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };

    const sentences = splitSentences(text);
    if (!sentences.length) sentences.push(text);
    const emitter = new Emitter(onChunk);
    const stddev = Math.sqrt(temperature);

    for (let si = 0; si < sentences.length; si++) {
      // Keep commas: the model's own clause phrasing beats synthetic pauses.
      const { text: prepared, eosExtra } = prepareText(this.keepCommas ? sentences[si] : sentences[si].replace(/[,;:]/g, " "));
      if (!prepared) continue;
      const lastChunk = si === sentences.length - 1;
      const terminal = lastChunk || isTerminalBoundary(sentences[si]);
      const chunkEosExtra = terminal ? eosExtra : 1; // continuation: 1 tail frame
      const ids = this.tok.encode(prepared);

      // conditioning: restore voice KV, run text conditioning pass
      this.main.restore(snap);
      const tokTensor = new this.ort.Tensor("int64", BigInt64Array.from(ids, BigInt), [1, ids.length]);
      const t = await this.txt.run({ [this.txt.inputNames[0]]: tokTensor });
      const temb = t[this.txt.outputNames[0]];
      const tdims = temb.dims.length === 2 ? [1, ...temb.dims] : [...temb.dims];
      await this.condPass(temb.data, tdims);
      this.dec.reset();

      // AR + decode loop
      const cl = new Float32Array(32).fill(NaN);
      const noise = new Float32Array(32);
      let eos = false, extra = 0, frames = 0;
      let pending = [];
      let done = false;
      let chunkTarget = 1;
      while (!done && frames < maxFrames && !this._cancel) {
        if (temperature > 0) randn(noise, stddev, rand);
        else noise.fill(0);
        const out = await this.main.run({
          ...(this.soura ? { soura_shift: this.shiftTensor } : {}),
          sequence: new this.ort.Tensor("float32", cl.slice(), [1, 1, 32]),
          text_embeddings: this.emptyText(),
          flow_x: new this.ort.Tensor("float32", noise.slice(), [1, 32]),
        });
        const eosLogit = out.eos_logit.data[0];
        if (!eos && eosLogit > -4.0) eos = true;
        if (eos && ++extra > chunkEosExtra) { done = true; }
        else {
          const latent = out.latent.data;
          cl.set(latent);
          pending.push(Float32Array.from(latent));
          frames++;
        }
        const isLast = done || frames >= maxFrames;
        if (pending.length >= chunkTarget || (isLast && pending.length)) {
          const T = pending.length;
          const lat = new Float32Array(T * 32);
          pending.forEach((p, i) => lat.set(p, i * 32));
          pending = [];
          const decOut = await this.dec.run({
            latent: new this.ort.Tensor("float32", lat, [1, T, 32]),
          });
          const audio = decOut[Object.keys(decOut)[0]].data;
          emitter.pushDecoded(Float32Array.from(audio), isLast);
          chunkTarget = Math.min(chunkTarget * 2, 8);
          opts.onProgress?.(frames);
        }
      }
      emitter.finishSentence(terminal, lastChunk);
      if (this._cancel) break;
    }
    emitter.flushCandidate();
    if (emitter.pendingTail.length) {
      const t = emitter.pendingTail;
      const inv = t.length > 1 ? 1 / (t.length - 1) : 1;
      for (let i = 0; i < t.length; i++) t[i] *= 1 - smooth(i * inv);
      emitter.emitRaw(t);
      emitter.pendingTail = new Float32Array(0);
    }
  }
}
