// Ports of prepare_text / split_sentences / soften_commas from pocket_tts.cpp.

const ABBREV = new Set(["mr", "mrs", "ms", "dr", "st", "jr", "sr", "vs", "etc",
  "inc", "ltd", "prof", "gen", "gov", "sgt", "cpl", "pvt", "capt", "lt", "col"]);
const MAX_CHUNK_CHARS = 300;

export function splitSentences(text) {
  const sentences = [];
  let current = "";
  const isAbbrev = (s, dot) => {
    let start = dot;
    while (start > 0 && /[a-zA-Z]/.test(s[start - 1])) start--;
    return ABBREV.has(s.slice(start, dot).toLowerCase());
  };
  for (let i = 0; i < text.length; i++) {
    current += text[i];
    const c = text[i];
    if (c === "." || c === "!" || c === "?") {
      if (c === "." && text[i + 1] === ".") continue;
      if (c === "." && text[i - 1] === ".") continue;
      if (c === "." && isAbbrev(text, i)) continue;
      if (i + 1 >= text.length || text[i + 1] === " " || text[i + 1] === '"' || text[i + 1] === "'") {
        const t = current.replace(/^[\s]+/, "");
        if (t) sentences.push(t);
        current = "";
      }
    }
  }
  const t = current.replace(/^[\s]+/, "");
  if (t) sentences.push(t);

  const chunks = [];
  for (const sentence of sentences) {
    let start = 0;
    while (start < sentence.length) {
      while (start < sentence.length && /\s/.test(sentence[start])) start++;
      if (sentence.length - start <= MAX_CHUNK_CHARS) {
        const piece = sentence.slice(start).trim();
        if (piece) chunks.push(piece);
        break;
      }
      const limit = Math.min(sentence.length, start + MAX_CHUNK_CHARS);
      let split = -1;
      for (let i = limit; i > start; i--) {
        const c = sentence[i - 1];
        if (c === "," || c === ";" || c === ":") { split = i; break; }
      }
      if (split < 0) {
        for (let i = limit; i > start; i--) {
          if (/\s/.test(sentence[i - 1])) { split = i - 1; break; }
        }
      }
      if (split < 0 || split <= start) split = limit;
      const piece = sentence.slice(start, split).trim();
      if (piece) chunks.push(piece);
      start = split;
      while (start < sentence.length && /[\s,;:]/.test(sentence[start])) start++;
    }
  }
  return chunks;
}

export function isTerminalBoundary(s) {
  const t = s.trimEnd();
  const c = t[t.length - 1];
  return c === "." || c === "!" || c === "?";
}

export function softenCommas(s) {
  return s.replace(/[,;:]/g, " ");
}

/** @returns {{text: string, eosExtra: number}} */
export function prepareText(raw, cfgEosExtra = -1) {
  let text = raw.replace(/["`“”]/g, "");
  while (text.startsWith("'") || text.startsWith("`") || text.startsWith("‘") || text.startsWith("’"))
    text = text.slice(1);
  while (text.endsWith("'") || text.endsWith("`") || text.endsWith("‘") || text.endsWith("’"))
    text = text.slice(0, -1);
  text = text.trim();
  if (!text) return { text: "", eosExtra: cfgEosExtra >= 0 ? cfgEosExtra : 3 };
  text = text.replace(/[\n\r]/g, " ");
  const nwords = (text.match(/\S+/g) || []).length;
  const eosExtra = cfgEosExtra >= 0 ? cfgEosExtra : (nwords <= 4 ? 5 : 3);
  if (/[a-z]/.test(text[0])) text = text[0].toUpperCase() + text.slice(1);
  if (/[a-zA-Z0-9]/.test(text[text.length - 1])) text += ".";
  if (nwords < 5) text = "        " + text; // 8 spaces, matching native
  return { text, eosExtra };
}
