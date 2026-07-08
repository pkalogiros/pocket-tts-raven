# PocketTTS webdemo

PocketTTS running fully in the browser: clone a voice from an audio file or
your microphone, type text, hear it stream, download the result. Nothing
leaves the machine — models run on onnxruntime-web (WASM SIMD + threads).

Self-contained: nothing here touches the native runtime in the repo root.

## Run

```bash
python3 webdemo/serve.py            # http://localhost:8093/ and https://<lan-ip>:8094/
uv run --no-project --with brotli python webdemo/compress.py   # after model changes
```

The server sets the COOP/COEP headers WASM threads require, serves HTTPS with
an auto-generated self-signed cert (SharedArrayBuffer needs a secure context,
so LAN devices must use https — accept the cert warning once per device), and
serves pre-compressed `.br` files (native engine first load: ~66MB instead of
~135MB raw). Use plain `http://localhost:8093/` on the same machine; use
`https://<lan-ip>:8094/` from other devices.

With precompressed assets, first page load transfers roughly 65–67 MB over the
wire for the selected engine's runtime plus text/AR/decoder models (then
caches them in the browser). The first voice clone lazily fetches a further
~14 MB compressed Mimi encoder (~38 MB raw). The optional ts engine uses its
own AR graph, which `prepare_models.sh` does not produce — see
[Engines](#engines) below.

## Test

```bash
./tools/prepare_models.sh --web
uv run --no-project --with onnx --with onnxruntime --with sentencepiece \
  python webdemo/test/gen_reference.py
uv run --no-project --with onnx --with onnxruntime --with sentencepiece \
  python webdemo/test/gen_state_xfer.py
cd webdemo && npm install && npm test
```

- `tokenizer.test.mjs` — TS SentencePiece vs native sentencepiece: exact ids
- `xfer.test.mjs` — the correctness gate: python post-conditioning state is
  transplanted into the JS engine and one AR step must match <=1e-3
- `engine.test.mjs` — end-to-end sanity (frame count, duration, RMS)

The generated references are intentionally ignored by git: `xfer_states.bin`
is about 47 MB and `ref_*.f32` changes whenever the preset voice or web model
set changes. Regenerate them after model or preset changes:

```bash
uv run --no-project --with onnx --with onnxruntime --with sentencepiece \
  python webdemo/test/gen_reference.py
uv run --no-project --with onnx --with onnxruntime --with sentencepiece \
  python webdemo/test/gen_state_xfer.py
```

Note on cross-platform output: WASM and native MLAS differ at ~1e-6 per step;
fp16 KV rounding and autoregressive feedback amplify that into different (but
equally valid) speech trajectories — like a different temperature draw. Exact
waveform equality across runtimes is not expected and not a bug; the
single-step transplant test is what proves the port correct.

## Engines

Two interchangeable engines implement the same worker message protocol:

- **ts**: JS orchestration on onnxruntime-web. No build step. Kept as a
  readable reference implementation and as the correctness baseline the
  native engine was validated against.

  onnxruntime-web cannot run the AttentionTail custom op, so this engine
  needs a different AR model: `flow_lm_main_delta_flow_int8.onnx` (delta-KV
  + merged flow, without the custom-attention rewrite).
  `prepare_models.sh` does not currently produce that file, so from a fresh
  checkout `?engine=ts` will fail to load its AR model. To build it, run the
  AR chain from `tools/prepare_models.sh` with the AttentionTail step left
  out (delta-KV → dedup → merged flow):

  ```bash
  P=(uv run --no-project --with onnx --with onnxruntime --with numpy python)
  "${P[@]}" tools/make_delta_kv_onnx.py models/flow_lm_main_int8.onnx /tmp/ts1.onnx
  "${P[@]}" tools/make_dedup_positional_onnx.py /tmp/ts1.onnx /tmp/ts2.onnx
  "${P[@]}" tools/make_merged_flow_onnx.py /tmp/ts2.onnx \
      models/flow_lm_flow_int8.onnx webdemo/models/flow_lm_main_delta_flow_int8.onnx
  ```

  Everything else the ts engine fetches (decoder, conditioner, tokenizer
  assets) is the standard web set; the default native engine never fetches
  this extra model.
- **native** (default): the full `pocket_tts.cpp` compiled with Emscripten — real
  pthreads pipelining, portable-SIMD custom attention op, tokenization
  delegated to the validated JS tokenizer. Select with `?engine=native`
  (persisted in localStorage; `?engine=ts` switches back).

  Measured (M4 Max, Node/V8 ≈ Chrome, 8-thread pool): **up to ~14x
  realtime, first audio ~45–70ms** — vs ~7x / ~100–150ms for the ts engine.
  Two build details matter enormously: ORT must be compiled with
  `--disable_wasm_exception_catching --enable_wasm_api_exception_catching`
  (full exception catching wraps every call in JS trampolines and cost 2.2x
  on the decoder), and the custom-op decoder variants are intentionally NOT
  used here (single-threaded custom kernels lose to MLAS's threaded convs in
  WASM; only the AR AttentionTail custom op wins).

Building the native engine:

```bash
# once, from the repo root: ORT WASM static lib (~1h)
git clone --recursive --branch v1.27.0 \
  https://github.com/microsoft/onnxruntime ../ort-wasm
(cd ../ort-wasm && ./build.sh --config Release --build_wasm_static_lib \
  --enable_wasm_threads --enable_wasm_simd \
  --disable_wasm_exception_catching --enable_wasm_api_exception_catching \
  --skip_tests --parallel)

# then, from the repo root: PocketTTS wasm module -> vendor/ptt/
./webdemo/wasm/build.sh
```

The native engine initially fetches the runtime plus text, AR, decoder, and
voice-preset assets. The Mimi encoder is loaded by a disposable encode worker
only when cloning, so preset playback avoids that extra download and memory
peak. Cloned voices persist as source WAVs; the engine re-derives its
`.emb`/`.kv` caches in MEMFS on restore.

## Using the WASM module in your own page

`webdemo/vendor/ptt/pocket_tts_wasm.{mjs,wasm}` is a self-contained ES
module exposing the full C API (`include/pocket_tts.h`) via cwrap. It
needs cross-origin isolation (COOP/COEP headers) for threads, and models
written into its virtual filesystem:

```js
import createPocketTTS from "./vendor/ptt/pocket_tts_wasm.mjs";
import { Tokenizer } from "./src/engine/tokenizer.js";

const M = await createPocketTTS();
M.FS.mkdir("/models"); M.FS.mkdir("/voices"); M.FS.mkdir("/voices/.cache");
for (const f of ["flow_lm_main_delta_attn_flow_int8.onnx",
                 "mimi_decoder_delta_int8.onnx", "text_conditioner.onnx",
                 "bos_before_voice.npy"]) {
  M.FS.writeFile(`/models/${f}`, new Uint8Array(await (await fetch(`models/${f}`)).arrayBuffer()));
}
// tokenization is delegated to JS (see OPTIMIZATION_NOTES on why):
const vocab = await (await fetch("models/spm_vocab.json")).json();
const tok = new Tokenizer(vocab);
self.__pkttsTokenize = (t) => tok.encode(t);

M.cwrap("ptt_configure_pool", null, ["number","number"])(8, 1);
const create = M.cwrap("ptt_create_ex", "number",
  ["string","string","string","string","number","number","number","number"]);
const h = create("/models", "/voices", "/models/tokenizer.model", "int8",
                 0.7, 1, 8, 2 /* defer encoder: fetch it only for cloning */);

// stream: start, then poll without blocking the runtime thread
const start = M.cwrap("ptt_stream_start", "number", ["number","string","string"]);
const poll  = M.cwrap("ptt_stream_poll", "number", ["number","number","number"]);
const end   = M.cwrap("ptt_stream_end", null, ["number"]);
// ... see src/tts-worker-native.js for the full poll loop, cloning, and
// wasm/test-node.mjs for a minimal headless example.
```

A complete runnable version of this example lives in
[`examples/minimal/`](examples/minimal/) — two files, ~60 lines of JS;
with the dev server running, open `/examples/minimal/`.

Two things that will bite you if skipped: run it in a **worker** (the poll
loop must yield so pthread filesystem calls can be serviced — never call
the blocking `ptt_stream_read` on the module's main runtime thread), and
keep the `.mjs`/`.wasm` filenames paired (the wasm path is baked into the
JS glue; rename both or neither).

## Architecture

```text
index.html          light-mode UI
src/bus.js          PKTTS event bus (fireEvent/listenFor, AudioMass-style)
src/app.js          engine bridge, player, recorder, validation, export, OPFS
src/ui.js           DOM wiring: chips, waveform, clone modal
src/tts-worker.js   Worker owning the ONNX sessions (UI thread never blocks)
src/playback-worklet.js  streaming ring-buffer player
src/engine/         the engine: tokenizer, state runner, synth (a faithful
                    port of pocket_tts.cpp: delta-KV scatter, KV snapshots,
                    merged-flow AR loop, crossfades, trim gate + sustain)
models/             browser model set: custom-op merged AR for native WASM,
                    stock-ORT merged AR for the TS engine, and dedup delta
                    decoder — verified bitwise vs the native set by
                    tools/verify_model_equivalence.py
presets/alba.emb    preset Alba voice embedding (same EMB1 format as native cache)
vendor/             onnxruntime-web dist, recorder-worklet, and lame.js
audiomass/          optional local AudioMass checkout/build; "Edit in
                    AudioMass" hands the WAV over via postMessage
```

Cloned voices persist in OPFS (Chrome/Edge/Firefox; Safari falls back to
session-only). All app code follows the AudioMass observer pattern: modules
communicate only through `PKTTS.fireEvent` / `PKTTS.listenFor`.

## Optional AudioMass integration

AudioMass is not vendored in this repository. To enable the "Edit in
AudioMass" buttons, place an AudioMass checkout or build under
`webdemo/audiomass/` so `webdemo/audiomass/index.html` exists. The expected
same-origin bridge and defensive `PKAudioEditor.engine.LoadFile` check are
documented in [`audiomass/README.md`](audiomass/README.md).

## Measured performance (Node/V8 WASM, M4 Max — Chrome-equivalent)

```text
native engine   up to ~14x realtime, first audio ~45-70ms
ts engine       ~7x realtime, first audio ~100-150ms
voice clone     ~0.5-1s for a 5s sample (one-time per voice, then cached)
iPhone 16 Pro Safari estimate: ~3-4x realtime
```
