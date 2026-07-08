# PocketTTS.cpp Runbook

How to build, run, test, and verify this optimized PocketTTS fork.
Generic upstream usage lives in [../README.md](../README.md); the full
optimization history and model-regeneration details live in
[OPTIMIZATION_NOTES.md](OPTIMIZATION_NOTES.md). This runbook uses the public
repo layout: `tools/prepare_models.sh` writes active models directly into
`models/`.

Current performance on M4 Max with the recommended setup below:
**~31x realtime, ~22ms time-to-first-audio** (was 18x / 36ms before the
2026-07 optimization rounds).

## Build

```bash
cmake -B .build -DCMAKE_BUILD_TYPE=Release
cmake --build .build -j12
```

Outputs land in the repo root:

- `pocket-tts` — CLI + HTTP server binary
- `libptt_custom_ops.dylib` — the custom ORT ops as a loadable library,
  used only by python-side model verification (macOS only)

The attention custom ops (`AttentionTail`, `DecoderAttentionTail`) are
portable: Accelerate on Apple, WASM-SIMD in the browser, and compiler
vector extensions (NEON/SSE) on any other GCC/Clang target — so Linux and
clang-built Windows load the fast `flow_lm_main_delta_attn_flow` AR model
too. The conv ops (`DecoderConvTransposeOverlap`, `AccelConv`,
`AccelConvElu`) stay Apple-only: they need Accelerate/AMX-class GEMM to
beat ORT's MLAS convolutions, so other platforms automatically keep the
plain `mimi_decoder_delta` decoder — slower decoder, same audio. MSVC has
no vector extensions; an MSVC build falls back to no custom ops entirely
(use clang-cl on Windows). `-DPTT_FORCE_PORTABLE` builds the portable
backend on macOS for A/B testing.

## Run

### Recommended flags (M4 Max, alongside the game)

```bash
./pocket-tts \
  --low-latency --trim-leading --trim-confirm 2 --trim-sustain 15 \
  --threads-ar 3 --threads-dec 2 --threads-full 2 \
  --temperature 0.7 \
  --models-dir models \
  --tokenizer models/tokenizer.model \
  --voices-dir <voices dir> \
  "Text to speak" voice.wav out.wav
```

Notes:

- All model optimizations (fused ConvTranspose decoder, AccelConv,
  merged main+flow graph, custom attention) are **default-on**. No flags
  needed; they activate when the model files exist in `--models-dir`.
- `--threads-ar 3` is the measured sweet spot; drop to 2 if the game needs
  the core. More decoder threads do NOT help (the decoder is
  Accelerate-bound).
- `--trim-confirm 2` is the built-in default (listed for clarity).
- `--trim-sustain 15` keeps the leading-burst gate on. Do not pass
  `--trim-sustain 0` for production; it reproduced the leading blip in
  internal burst checks.
- `--temperature 0` makes output deterministic — use it for any comparison
  work, never for production speech.

### Server mode

```bash
./pocket-tts --server --port 8080 \
  --low-latency --trim-leading --trim-sustain 15 \
  --threads-ar 3 --threads-dec 2 --threads-full 2 \
  --temperature 0.7 \
  --models-dir models \
  --tokenizer models/tokenizer.model \
  --voices-dir <voices dir>
```

- `POST /tts` `{"text": "...", "voice": "...", "format": "s16le"}` — chunked
  streaming PCM (s16le halves payload size; omit for float32)
- `POST /v1/audio/speech` — OpenAI-compatible, `wav` or `pcm`
- `GET /health`

**Always restart a running server after rebuilding or swapping models** — a
live process keeps the old binary and old graphs forever. Sanity check: the
per-request log line `First chunk latency:` should read low-20s ms, not 30+.

### Opt-out flags (A/B testing and rollback)

The runtime has kill switches for isolating regressions:

```text
--no-decoder-convtr     use the portable delta decoder instead of the fused Apple decoder
--no-merged-flow        use a separate flow session per frame when a non-merged AR model is staged
--no-custom-attention   disable AttentionTail custom-op AR models
--no-delta-kv           disable all delta-KV models (slowest, original path)
```

The clean public model set installs the default fast path, not every
intermediate A/B artifact. If an intermediate model is absent, the runtime
falls back to the next available graph. Use `--no-delta-kv` at
`--temperature 0` for the original-model reference path; stage intermediate
models separately when isolating a single rewrite.

## Test

### Smoke test

```bash
./pocket-tts --profile --low-latency --trim-leading --trim-sustain 15 \
  --threads-ar 3 --threads-dec 2 --threads-full 2 --temperature 0 \
  --models-dir models \
  --tokenizer models/tokenizer.model \
  --voices-dir <voices dir> \
  "Beware: the nearest mirror hides a portal to the abyss—look, but don't step!" \
  <voice>.wav /private/tmp/ptt-smoke.wav
```

`--profile` prints the first-chunk stage trace and a per-session timing
table. Expected on M4 Max (temp 0): RTFx ≥ ~30x, first chunk ≤ ~25ms,
`run:mimi_decoder_delta_convtr_int8` total ≈ 135–145ms for this sentence.
Confirm the profiler rows show the optimized session names
(`flow_lm_main_delta_attn_flow_int8`, `mimi_decoder_delta_convtr_int8`) —
plain names mean a model file is missing and a fallback silently engaged
(run with `--verbose` to see which).

### Benchmarking a change

Rules that keep results honest:

1. **Use `--temperature 0`** so every run generates the identical latent
   sequence.
2. **Run n≥5 and compare medians** — thermal and scheduling noise on a busy
   Mac is ±5%.
3. **Pair your runs**: benchmark old and new back-to-back in the same
   session, not against numbers from another day.
4. **Never diff emitted WAVs sample-by-sample across variants.** Each
   decode-batch boundary consumes 120 samples in the 5ms crossfade, and
   batch boundaries depend on thread timing — files shift by multiples of
   120 samples even when models are mathematically identical. A temp-0 WAV
   hash is meaningful only when **equal** (equal ⇒ identical; different ⇒
   inconclusive).

Metrics to extract from `--profile` output: `RTFx`, `First chunk latency`,
the per-session `Total(ms)` rows, and the first-chunk stage trace
(`latent_gen_ready`, `first_decoder_done`, `first_callback`).

### Verifying a model rewrite (quality check)

Any change to an ONNX model should pass the lockstep verifier before
install. It runs both models with identical inputs and C++-faithful state
feedback (delta-KV scatter, wraparound, bool-true init) and reports the max
abs difference on every output and internal state — completely
timing-independent:

```bash
cmake --build .build -j12 --target ptt-custom-ops   # once, for the dylib

uv run --no-project --with onnx --with onnxruntime python \
  tools/verify_model_equivalence.py OLD.onnx NEW.onnx
```

Interpretation:

- **0.0** — bitwise-identical math (e.g. the dedup pass)
- **≤ ~1e-6 on audio outputs** — floating-point reordering only
  (Accelerate vs MLAS accumulation order); inaudible, fine to ship
- **≥ ~1e-3** — the rewrite changed actual behavior; investigate before
  shipping

Stage unproven models via a symlinked bundle instead of touching
production:

```bash
STAGE=/private/tmp/ptt-models-stage
mkdir -p $STAGE && ln -s $(pwd)/models/* $STAGE/
rm $STAGE/<model>.onnx && ln -s /path/to/new.onnx $STAGE/<model>.onnx
./pocket-tts --models-dir $STAGE ...
```

### Ear check

Render the reference path and the current path with identical settings and
listen:

```bash
./pocket-tts <flags> --temperature 0 --no-decoder-convtr --no-merged-flow \
  "..." voice.wav /tmp/ab-old.wav
./pocket-tts <flags> --temperature 0 "..." voice.wav /tmp/ab-new.wav
```

## Models

### Bundle layout

`models/` is the active public model directory. `tools/prepare_models.sh`
downloads the original Kyutai ONNX bundle (hash-verified) and installs the
optimized ONNX files there. Older internal notes mention an
`onnx/english_2026-04/` symlink bundle; that layout is not required for
this repository.

Active optimized models and what produced them:

```text
flow_lm_main_delta_attn_flow_int8.onnx  delta-KV + AttentionTail + dedup + inlined 1-step flow
mimi_decoder_delta_int8.onnx            delta-KV portable decoder
mimi_decoder_delta_convtr_int8.onnx     delta-KV + ConvTranspose fusion + AccelConv + dedup
```

Intermediate AR graphs such as `flow_lm_main_delta_attn_int8.onnx` are
created during the pipeline but are not installed by the clean public setup.

Regeneration chains and commands: see "Regenerating the current installed
models" in [OPTIMIZATION_NOTES.md](OPTIMIZATION_NOTES.md).

### Rollback

Pre-change backups sit next to the model files:

```text
models/*.before-fastkv.onnx
models/*.before-accelconv.onnx
models/*.before-dedup.onnx
```

Restore by copying a backup over the real target in `models/`, e.g.:

```bash
cp -p models/mimi_decoder_delta_convtr_int8.before-accelconv.onnx \
  models/mimi_decoder_delta_convtr_int8.onnx
```

(Runtime-level rollback needs no file changes — use the `--no-*` flags.)

### Voice caches

`<voices-dir>/.cache/` holds `.emb` (encoder output) and `.kv`
(voice-conditioned KV snapshot) files keyed by voice filename and mtime.
Safe to delete anytime; they regenerate on next use (first request per voice
becomes slower by a few hundred ms). Delete them after changing the AR model
if you want to be strict — stale `.kv` files are detected by mtime against
the voice WAV, not against the model.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| RTFx dropped to ~18x | Fallback to old decoder: `models/mimi_decoder_delta_convtr_int8.onnx` missing. Run with `--verbose`. |
| First chunk 30ms+ on server | Server process predates the rebuild — restart it. |
| `is not a registered function/op` loading a model in python | Register the ops: build `ptt-custom-ops` and pass the dylib (see verify section). |
| Session names in profile lack `_attn`/`_flow`/`_convtr` suffixes | Model file missing → silent fallback; check that `models/` contains the optimized files listed above. |
| Output length differs by ~120 samples between runs | Normal: decode-batch crossfade + thread timing. Not a bug, not a quality change. |
| Different temp-0 WAV hash after a model change | Inconclusive on its own — run `tools/verify_model_equivalence.py` to judge equivalence. |
| Vocal blip before the sentence starts | Spurious model fragment from the first decoder audio. Listen for short pre-speech fragments, or run a local leading-burst checker if you have one; short bursts are discarded by the gate's sustain check (`--trim-sustain 15`). Do not use `--trim-sustain 0` in production; it reproduced `BLIP before.wav (burst ends 105ms)`. |
| First chunk latency higher than expected by ~8ms | The sustain check observes up to 150ms of decoded audio before emitting. Lower values such as `--trim-sustain 8` trade blip protection for latency; `--trim-sustain 0` is legacy/unsafe for register-heavy voices. |
