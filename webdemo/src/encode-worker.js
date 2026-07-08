// Disposable voice-encode worker: its own small (growth-memory) WASM
// instance runs ONLY the mimi encoder, returns the .emb bytes, and is
// terminated by the caller — the one reliable way to actually free WASM
// memory. Keeps the encoder's ~0.5GB activation peak out of the main
// engine's heap entirely.

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

self.onmessage = async (e) => {
  const { assetBase, modelsUrl, wav, key, pool } = e.data;
  try {
    const { create, options } = await importPttModule(
      assetBase,
      "pocket_tts_wasm_growth.mjs",
      "../vendor/ptt/pocket_tts_wasm_growth.mjs"
    );
    const M = await create(options);
    M.FS.mkdir("/models"); M.FS.mkdir("/voices"); M.FS.mkdir("/voices/.cache");
    for (const f of ["mimi_encoder.onnx", "bos_before_voice.npy"]) {
      const res = await fetch(`${modelsUrl}/${f}`);
      if (!res.ok) throw new Error(`fetch ${f}: ${res.status}`);
      M.FS.writeFile(`/models/${f}`, new Uint8Array(await res.arrayBuffer()));
    }
    M.FS.writeFile(`/voices/${key}.wav`, new Uint8Array(wav));
    M.cwrap("ptt_configure_pool", null, ["number", "number"])(pool || 2, 1);
    const h = M.cwrap("ptt_create_ex", "number",
      ["string", "string", "string", "string", "number", "number", "number", "number"])(
      "/models", "/voices", "/models/tokenizer.model", "int8", 0.5, 1, pool || 2, 4 /* encoder_only */);
    if (!h) throw new Error("encoder-only create failed");
    const rc = M.cwrap("ptt_encode_voice", "number", ["number", "string"])(h, `${key}.wav`);
    if (rc !== 0) throw new Error("encode failed");
    const emb = M.FS.readFile(`/voices/.cache/${key}.emb`);
    self.postMessage({ ok: true, emb: emb.buffer }, [emb.buffer]);
  } catch (err) {
    self.postMessage({ ok: false, message: String(err && err.message || err) });
  }
};
