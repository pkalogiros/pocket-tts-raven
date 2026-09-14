// Disposable AR/text worker: returns a KV snapshot, then the caller terminates it.
import { Tokenizer } from "./engine/tokenizer.js";
function assetJoin(base, path) {
  base = String(base || "").replace(/\/+$/, "");
  path = String(path || "").replace(/^\/+/, "");
  return base ? `${base}/${path}` : "";
}

function pttModuleOptions(assetBase, mainScriptUrl) {
  return {
    mainScriptUrlOrBlob: mainScriptUrl,
    locateFile: (path) => assetJoin(assetBase, `vendor/ptt/${path}`) || path,
  };
}

async function importPttModule(assetBase, fileName, fallback) {
  const remote = assetJoin(assetBase, `vendor/ptt/${fileName}`);
  if (!remote) {
    const { default: create } = await import(fallback);
    return { create, options: {} };
  }

  const res = await fetch(remote);
  if (!res.ok) throw new Error(`fetch ${fileName}: ${res.status}`);
  const blobUrl = URL.createObjectURL(new Blob([await res.text()], { type: "text/javascript" }));
  const { default: create } = await import(blobUrl);
  return { create, options: pttModuleOptions(assetBase, blobUrl) };
}

self.onmessage = async ({ data }) => {
  const { assetBase, modelsUrl, emb, key, pool, soura = false } = data;
  try {
    const { create, options } = await importPttModule(assetBase, "pocket_tts_wasm_growth.mjs", "../vendor/ptt/pocket_tts_wasm_growth.mjs");
    const M = await create(options);
    M.FS.mkdir("/models"); M.FS.mkdir("/voices"); M.FS.mkdir("/voices/.cache");
    const main = "flow_lm_main_delta_attn_flow_int8.onnx";
    for (const name of [main, "text_conditioner.onnx", "bos_before_voice.npy"]) {
      const file = soura && name === main ? name.replace(".onnx", "_soura.onnx") : name;
      const response = await fetch(`${modelsUrl}/${file}`);
      if (!response.ok) throw new Error(`Cache preparation: ${file} HTTP ${response.status}`);
      M.FS.writeFile(`/models/${file}`, new Uint8Array(await response.arrayBuffer()));
      if (file !== name) M.FS.symlink(`/models/${file}`, `/models/${name}`);
    }
    // Neutral KV preparation only needs a valid zero shift; no vector download.
    if (soura) {
      const text = "{'descr': '<f4', 'fortran_order': False, 'shape': (6, 1024), }";
      const header = text.padEnd(117, ' ') + "\n";
      const bytes = new Uint8Array(128 + 6 * 1024 * 4);
      bytes.set([147,78,85,77,80,89,1,0,118,0]);
      bytes.set(new TextEncoder().encode(header), 10);
      M.FS.writeFile("/models/soura_vectors.npy", bytes);
    }
    const response = await fetch(`${modelsUrl}/spm_vocab.json`);
    if (!response.ok) throw new Error("Tokenizer unavailable");
    const tokenizer = new Tokenizer(await response.json());
    self.__pkttsTokenize = (text) => tokenizer.encode(text);
    M.FS.writeFile(`/voices/.cache/${key}.emb`, new Uint8Array(emb));
    M.cwrap("ptt_configure_pool", null, ["number","number"])(pool || 2, 1);
    const handle = M.cwrap("ptt_create_ex", "number", ["string","string","string","string","number","number","number","number"])(
      "/models", "/voices", "/models/tokenizer.model", "int8", .5, 1, pool || 2, 16 | (soura ? 8 : 0));
    if (!handle) throw new Error("Conditioning-only engine failed to start");
    if (M.cwrap("ptt_prepare_voice", "number", ["number","string"])(handle, `${key}.wav`) !== 0)
      throw new Error("Voice cache preparation failed");
    const kv = M.FS.readFile(`/voices/.cache/${key}.kv`);
    self.postMessage({ ok: true, kv: kv.buffer }, [kv.buffer]);
  } catch (error) {
    self.postMessage({ ok: false, message: error.message || String(error) });
  }
};
