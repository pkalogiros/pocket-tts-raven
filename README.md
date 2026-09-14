# PocketTTS-RAVEN ☠

**Faster-than-realtime voice cloning and text-to-speech — on CPU, in one C++ file, and in your browser.**

## ☠ [Live Demo → pantel.is/projects/pocket-tts-raven](https://pantel.is/projects/pocket-tts-raven/)

<br>

A heavily optimized runtime for [Kyutai's Pocket TTS](https://github.com/kyutai-labs/pocket-tts) on ONNX Runtime:

|                       | speed          | first audio | notes |
|-----------------------|----------------|-------------|-------|
| **Native** (M4 Max)   | ~33x realtime  | ~30 ms      | no Python, no GPU, no frameworks |
| **Browser** (desktop) | ~14x realtime  | ~70 ms      | WASM + SIMD + threads |
| **Browser** (iPhone)  | ~3–4x realtime | <250ms      | same build |

Voice cloning takes 6–15 seconds of clean speech and runs entirely
on-device — file, microphone, or FFI. Nothing is uploaded, ever.

![PocketTTS-RAVEN](webdemo/og-card.png)

## Responsible use

This runtime can clone voices. Only use voices you own or have explicit,
lawful consent to use. Do not use generated speech for impersonation,
deception, fraud, harassment, privacy-invasive content, or to present
synthetic audio as a genuine recording. Upstream Pocket TTS model access
also carries prohibited-use terms; this repository does not bypass them.

## Quickstart

```bash
# 1. build the native runtime
cmake -B .build -DCMAKE_BUILD_TYPE=Release && cmake --build .build -j

# 2. get models (one-time: downloads the original Kyutai ONNX bundle,
#    ~165 MB hash-verified, then applies the graph rewrites locally)
./tools/prepare_models.sh

# 3. generate speech with the bundled Alba sample voice
./pocket-tts "The road ahead is more dangerous than it looks." example.wav out.wav

# 4. clone a voice — cloning IS step 3: the voice argument is any wav/mp3
cp ~/my-recording.wav voices/me.wav      # 6-15s of one person speaking
./pocket-tts "Now I speak with your voice." me.wav cloned.wav
```

The embedding is computed on first use and cached (`voices/.cache/`), so
the second run with the same voice starts in milliseconds.
The bundled `voices/example.wav` and browser preset are generated from
Kyutai's Alba MacKenna `casual.wav` sample; see
`THIRD_PARTY_NOTICES.md` for attribution and license details.

## The web demo

```bash
# one-time: install the web model set into webdemo/models/ (+ brotli precompress)
./tools/prepare_models.sh --web
python3 webdemo/serve.py
#  → http://localhost:8093          (this machine)
#  → https://<your-ip>:8094         (other devices; accept the cert once)
```

The same C++ runtime compiled to WASM (see `webdemo/wasm/`): streaming
playback while generating, voice cloning from file or microphone with a
trim-selectable preview, optional round-trip editing through
[AudioMass](https://audiomass.co), and WAV/MP3 export. With precompressed assets, first visit
transfers roughly 65–67 MB for the selected engine's runtime plus
non-encoder models (cached by the browser); the first voice clone fetches a
further ~14 MB compressed Mimi encoder (~38 MB raw).
`webdemo/README.md` covers the engines, the worker message protocol,
**using the WASM module in your own page**, and how to rebuild it.

### Rebuilding the WASM engine (optional)

The compiled engine ships in `webdemo/vendor/ptt/`, so you never need to
build it to use or serve the demo. Rebuild only if you change
`src/pocket_tts.cpp` and want that change in the browser:

```bash
# once (~1h): ONNX Runtime v1.27.0 as a WASM static library, sibling to the repo
git clone --recursive --branch v1.27.0 \
    https://github.com/microsoft/onnxruntime ../ort-wasm
(cd ../ort-wasm && ./build.sh --config Release --build_wasm_static_lib \
    --enable_wasm_threads --enable_wasm_simd \
    --disable_wasm_exception_catching --enable_wasm_api_exception_catching \
    --skip_tests --parallel)

# then: rebuild the module (installs into webdemo/vendor/ptt/)
./webdemo/wasm/build.sh
```

The second step reuses the emsdk that ORT's build installs, so there is
nothing else to set up. `ORT_WASM_ROOT=/path/to/ort-wasm` overrides the
default sibling location.

## Requirements

- CMake 3.28+, a C++17 compiler (dependencies are fetched by CMake)
- [uv](https://docs.astral.sh/uv/) for the model-preparation scripts
- Any OS for the runtime; the custom attention op accelerates all
  GCC/Clang targets (ARM + x86-64), and the fused-conv custom ops
  additionally light up on Apple silicon

## Design: fast models, stock runtime

The speed comes from two clean layers — **no ONNX Runtime fork, no patches**:

1. **Offline graph rewrites** (`tools/`): the exported ONNX models are
   rewritten into faster but math-preserving graphs — delta-KV caching,
   cross-layer dedup, merged flow, custom-op injection. The delta-KV rewrites
   run built-in ORT equivalence checks; the lockstep verifier in `tools/`
   checks staged graph changes against their predecessor.
2. **A fast driver around stock ORT** (`src/pocket_tts.cpp`): persistent
   cache buffers, pipelined generate/decode threads, KV snapshots, and
   custom operators registered through ONNX Runtime's public custom-op
   API. The WASM build compiles unmodified ORT sources, choosing only
   official build flags.

Upgrading ONNX Runtime is a version bump, not a rebase.

## Models

`./tools/prepare_models.sh` downloads the original Kyutai Pocket TTS ONNX
bundle from Hugging Face (every file verified against a sha256 pinned in
the script) and applies the graph-rewrite pipeline locally — delta-KV
caches, custom attention ops, cross-layer dedup, merged flow. The pipeline is
deterministic; delta-KV steps run built-in ORT equivalence checks, and
`tools/verify_model_equivalence.py` is the lockstep checker for staged model
changes. `models/README.md` and `webdemo/models/README.md` list what lands
where. The rewrite scripts themselves live in `tools/` and are documented in
[docs/OPTIMIZATION_NOTES.md](docs/OPTIMIZATION_NOTES.md).

## CLI

```bash
./pocket-tts [OPTIONS] TEXT VOICE [OUTPUT]

# pipe raw PCM to a player
./pocket-tts --stdout "Hello." me.wav | ffplay -f f32le -ar 24000 -nodisp -autoexit -

# fastest first audio
./pocket-tts --low-latency --trim-leading "Hello." me.wav out.wav

# let the model voice clause pauses (nicer for prose)
./pocket-tts --keep-commas "First, we dig. Then, we descend." me.wav out.wav
```

### HTTP Server

```bash
./pocket-tts --server --port 8080
```

Endpoints:
- `POST /v1/audio/speech` — OpenAI-compatible TTS (JSON body: `{"input": "...", "voice": "..."}`)
- `POST /tts` — streaming TTS (JSON body: `{"text": "...", "voice": "..."}`)
- `GET /health` — health check

The `/v1/audio/speech` endpoint is compatible with the OpenAI TTS API. Any client that supports OpenAI's TTS (SillyTavern, Open WebUI, etc.) can use PocketTTS.cpp as a drop-in replacement by pointing the base URL to `http://localhost:8080`. The `model` and `speed` fields are accepted but ignored. Supported `response_format` values are `wav` (default) and `pcm`.

```bash
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "tts-1", "input": "Hello world!", "voice": "voice", "response_format": "wav"}' \
  --output speech.wav
```

The `/tts` endpoint streams raw chunked PCM (`audio/pcm;rate=24000;encoding=float;bits=32`) for low-latency applications.

### C API (FFI)

Public header: [`include/pocket_tts.h`](include/pocket_tts.h) — every
function documented. Build the shared library with
`-DBUILD_SHARED_LIB=ON`, or compile `src/pocket_tts.cpp` with
`PTT_SHARED_LIB`. Audio is mono float32 at 24 kHz throughout.

```c
void* tts = ptt_create("models", "voices", "models/tokenizer.model",
                       "int8", 0.7f, /*lsd_steps*/ 1, /*threads*/ 0);
void* s = ptt_stream_start(tts, "Hello there.", "me.wav");
float* chunk; int n;
while (ptt_stream_read(s, &chunk, &n) == 1) {   // blocks between chunks
    play(chunk, n);
    ptt_free_audio(chunk);
}
ptt_stream_end(s);
ptt_destroy(tts);
```

| group | functions |
|---|---|
| lifecycle | `ptt_create`, `ptt_create_ex` (decoder-only / deferred-encoder / encoder-only modes), `ptt_destroy`, `ptt_warmup` |
| streaming | `ptt_stream_start`, `_read` (blocking), `_poll` (non-blocking), `_stop` (graceful abort), `_end`, `ptt_free_audio` |
| tuning | `ptt_set_temperature`, `ptt_set_soften_commas`, `ptt_set_max_chunk`, `ptt_configure_pool` |
| voice | `ptt_encode_voice`, `ptt_load_encoder` (cloning is otherwise implicit via the voice argument) |
| split pipelines | `ptt_latents_start/poll/stop/end` (raw latent frames out), `ptt_decode`, `ptt_decoder_reset` |
| diagnostics | `ptt_set_profiling`, `ptt_print_profile`, `ptt_debug_bench` |

## Optional native emotion steering and cache preparation

The native runtime supports opt-in `--soura` emotion steering, per-request
emotion/intensity/seed controls, custom or reference-derived vectors, and
`--prepare-voice` for building voice caches in a disposable process. New native synthesis
behavior remains opt-in. The web UI defaults to Varkos with steering enabled
(use its checkbox or `?soura=0` to disable it). The rebuilt WASM engines also
expose steering, custom vectors, seeds, and optional disposable cache preparation;
see [browser controls](webdemo/OPTIONAL_FEATURES.md).

See [optional features](docs/OPTIONAL_FEATURES.md) for model preparation, flags,
examples, limitations, and regression checks.

## Caching

PocketTTS uses two layers of disk caching, both stored under `voices/.cache/`:

**Voice embeddings (`.emb`)** — The output of the Mimi encoder for each voice sample. Avoids re-encoding the same WAV file on every run. Generated automatically on first use.

**KV state snapshots (`.kv`)** — The transformer's internal KV cache state after voice conditioning. This is the expensive part — on a cold start, voice conditioning takes hundreds of milliseconds. A cached `.kv` file restores in ~4ms. For multi-sentence input, the KV snapshot is also held in memory so only the first sentence pays the disk load cost.

Cache files are invalidated automatically when the source WAV is modified. To clear all caches:

```bash
rm -rf voices/.cache/
```

To disable caching entirely, pass `--no-cache`.

For low-latency speech with leading-blip protection, use `--low-latency
--trim-leading` and keep the default `--trim-sustain 15`. `--trim-sustain 0`
restores legacy behavior and can emit an isolated leading burst on some
voices.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `-h`, `--help` | — | Show usage and all options |
| `--precision` | `int8` | Model precision (`int8` or `fp32`) |
| `--temperature` | `0.7` | Sampling temperature |
| `--lsd-steps` | `1` | Flow matching ODE solver steps |
| `--eos-threshold` | `-4.0` | EOS detection threshold (lower = later cutoff) |
| `--eos-extra` | `-1` | Extra frames after EOS (`-1` = auto from text length) |
| `--noise-clamp` | `0` | Clamp noise magnitude (`0` = disabled, matches upstream) |
| `--threads` | `0` | Total thread budget (`0` = half of available cores) |
| `--models-dir` | `models` | Path to ONNX model directory |
| `--voices-dir` | `voices` | Path to voice samples directory |
| `--tokenizer` | `models/tokenizer.model` | Path to SentencePiece tokenizer |
| `--low-latency` | — | Use shorter startup fade and skip the leading gate unless `--trim-leading` is set |
| `--trim-leading` | — | Keep the startup leading gate in low-latency mode |
| `--trim-confirm` | `2` | 10ms speech detector windows required before accepting speech |
| `--trim-sustain` | `15` | 10ms windows observed before emitting onset; keep at 15 for blip protection |
| `--trim-collapse` | `3` | Consecutive quiet windows that discard a spurious leading burst |
| `--no-cache` | — | Disable all disk caching (`.emb` and `.kv` files) |
| `--no-delta-kv` | — | Disable delta-KV `flow_lm_main` model when present |
| `--stdout` | — | Output raw f32le PCM to stdout |
| `--verbose` | — | Enable verbose output |
| `--profile` | — | Print per-operation timing report after generation |
| `--server` | — | Start HTTP server mode |
| `--port` | `8080` | Server port |

The table above covers the common flags; run `./pocket-tts --help` for
the complete list (thread overrides, leading-trim gate tuning, chunk
sizing, model-variant toggles).

## Acknowledgments

- [Kyutai Labs](https://github.com/kyutai-labs/pocket-tts) — Pocket TTS model and original Python implementation (MIT); see their model card for weight licensing
- [VolgaGerm/PocketTTS.cpp](https://github.com/VolgaGerm/PocketTTS.cpp) — prior C++ Pocket TTS work and reference implementation
- [Verylicious/pocket-tts-ungated](https://huggingface.co/Verylicious/pocket-tts-ungated) — ungated model weights and tokenizer (CC-BY-4.0)
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) (MIT) — inference engine, including the WASM build
- [AudioMass](https://audiomass.co) — optional same-origin audio editor integration
- [lame.js](https://github.com/zhuker/lamejs) (LGPL) — in-browser MP3 export
- [dr_libs](https://github.com/mackron/dr_libs) (public domain / MIT-0) — audio decoding

Third-party components keep their own licenses — see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) (notably lame.js, LGPL).

## License

MIT — see [LICENSE](LICENSE).
