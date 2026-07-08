// node webdemo/test/tokenizer.test.mjs
import { readFileSync } from "node:fs";
import { Tokenizer } from "../src/engine/tokenizer.js";

const dir = new URL(".", import.meta.url).pathname;
const vocab = JSON.parse(readFileSync(`${dir}/../models/spm_vocab.json`, "utf8"));
const vectors = JSON.parse(readFileSync(`${dir}/tokenizer_vectors.json`, "utf8"));

const tok = new Tokenizer(vocab);
let pass = 0;
for (const { text, ids } of vectors) {
  const got = tok.encode(text);
  const ok = JSON.stringify(got) === JSON.stringify(ids);
  if (ok) pass++;
  else {
    console.log(`FAIL: ${JSON.stringify(text)}`);
    console.log(`  want ${JSON.stringify(ids)}`);
    console.log(`  got  ${JSON.stringify(got)}`);
  }
}
console.log(`${pass}/${vectors.length} tokenizer vectors match`);
process.exit(pass === vectors.length ? 0 : 1);
