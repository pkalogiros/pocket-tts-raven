// Streaming silence squeeze: the model's comma/clause pauses (with commas
// preserved) run 270-400ms, which reads as slightly theatrical. Pass the
// first PASS_MS of any quiet run untouched, time-compress the excess 2:1,
// and never emit more than HARD_MS of one continuous pause.
//   270ms -> ~215ms   400ms -> ~280ms   sentence pause 265ms -> ~212ms
// Zero added latency: decisions are per 10ms window as audio flows through.

const SR = 24000;
const WIN = 240;            // 10ms
const RMS_T = 0.006;
const PASS_WINDOWS = 16;    // 160ms passed through untouched
const HARD_WINDOWS = 32;    // max 320ms of emitted quiet per run

export class SilenceSqueeze {
  constructor() {
    this.carry = new Float32Array(0);
    this.quietRun = 0;      // windows seen in the current quiet run
    this.emittedQuiet = 0;  // windows emitted in the current quiet run
  }

  /** @param {Float32Array} chunk @returns {Float32Array} possibly shorter */
  push(chunk) {
    let buf = chunk;
    if (this.carry.length) {
      buf = new Float32Array(this.carry.length + chunk.length);
      buf.set(this.carry);
      buf.set(chunk, this.carry.length);
    }
    const out = new Float32Array(buf.length);
    let o = 0;
    let pos = 0;
    for (; pos + WIN <= buf.length; pos += WIN) {
      let s2 = 0;
      for (let i = pos; i < pos + WIN; i++) s2 += buf[i] * buf[i];
      const quiet = Math.sqrt(s2 / WIN) < RMS_T;
      if (!quiet) {
        this.quietRun = 0;
        this.emittedQuiet = 0;
        out.set(buf.subarray(pos, pos + WIN), o);
        o += WIN;
        continue;
      }
      this.quietRun++;
      const emit =
        this.quietRun <= PASS_WINDOWS ||                      // head: verbatim
        (this.quietRun % 2 === 0 &&                           // excess: 2:1
         this.emittedQuiet < HARD_WINDOWS);                   // ceiling
      if (emit) {
        this.emittedQuiet++;
        out.set(buf.subarray(pos, pos + WIN), o);
        o += WIN;
      }
    }
    this.carry = buf.slice(pos);
    return out.subarray(0, o);
  }

  flush() {
    const rest = this.carry;
    this.carry = new Float32Array(0);
    return rest;
  }
}
