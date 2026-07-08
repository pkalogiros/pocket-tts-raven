// SentencePiece unigram tokenizer (inference only), matching the exact
// settings of PocketTTS's tokenizer.model: identity normalizer (no charsmap),
// add_dummy_prefix=true, escape_whitespaces=true, remove_extra_whitespaces=false.
// Verified against native sentencepiece output by test/tokenizer.test.mjs.

const SPACE = "▁"; // ▁

export class Tokenizer {
  /** @param {{pieces: [string, number, number][], add_dummy_prefix: boolean, unk_id: number}} vocab */
  constructor(vocab) {
    this.unkId = vocab.unk_id;
    this.addDummyPrefix = vocab.add_dummy_prefix;
    this.pieceScore = new Map();
    this.byteId = new Map(); // byte value -> piece id (byte-fallback pieces, type 6)
    this.maxPieceLen = 1;
    let minScore = 0;
    vocab.pieces.forEach(([piece, score, type], id) => {
      if (type === 6) {
        this.byteId.set(parseInt(piece.slice(1, 5), 16), id);
        return;
      }
      if (type !== 1 && type !== 4) return; // 1=NORMAL, 4=USER_DEFINED; skip control/unk
      this.pieceScore.set(piece, { id, score });
      if (piece.length > this.maxPieceLen) this.maxPieceLen = piece.length;
      if (score < minScore) minScore = score;
    });
    this.unkScore = minScore - 10.0; // sentencepiece kUnkPenalty
  }

  static async load(url) {
    const vocab = await (await fetch(url)).json();
    return new Tokenizer(vocab);
  }

  /** @param {string} text @returns {number[]} */
  encode(text) {
    if (this.addDummyPrefix) text = " " + text;
    const s = text.replaceAll(" ", SPACE);
    const chars = Array.from(s); // code points
    const n = chars.length;
    // Viterbi over code-point positions.
    const best = new Float64Array(n + 1).fill(-Infinity);
    const from = new Int32Array(n + 1).fill(-1);
    /** @type {(number[]|number)[]} */
    const pieceAt = new Array(n + 1).fill(0);
    best[0] = 0;
    for (let i = 0; i < n; i++) {
      if (best[i] === -Infinity) continue;
      let sub = "";
      for (let j = i; j < Math.min(n, i + this.maxPieceLen); j++) {
        sub += chars[j];
        const hit = this.pieceScore.get(sub);
        if (hit !== undefined) {
          const sc = best[i] + hit.score;
          if (sc > best[j + 1]) {
            best[j + 1] = sc;
            from[j + 1] = i;
            pieceAt[j + 1] = hit.id;
          }
        }
      }
      // single-char fallback: unk or byte pieces
      const sc = best[i] + this.unkScore;
      if (sc > best[i + 1]) {
        best[i + 1] = sc;
        from[i + 1] = i;
        const bytes = new TextEncoder().encode(chars[i]);
        if (this.byteId.size > 0) {
          pieceAt[i + 1] = Array.from(bytes, (b) => this.byteId.get(b) ?? this.unkId);
        } else {
          pieceAt[i + 1] = this.unkId;
        }
      }
    }
    const ids = [];
    for (let pos = n; pos > 0; pos = from[pos]) {
      const p = pieceAt[pos];
      if (Array.isArray(p)) ids.push(...p.slice().reverse());
      else ids.push(p);
    }
    return ids.reverse();
  }
}
