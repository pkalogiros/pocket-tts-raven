# Optional native features

These changes port the reusable companion-app work into PocketTTS-RAVEN.
All new synthesis behavior is off by default. Existing int8, delta-KV,
custom-attention, merged-flow, decoder, punctuation and trimming defaults stay
unchanged. No character-specific voice files, DSP, app mood routing, or speculative
continuation pipeline are installed. The browser and C ABI now expose opt-in equivalents; their default behavior
is unchanged. See [browser features](../webdemo/OPTIONAL_FEATURES.md). The
commands below describe the native CLI and `/tts` server endpoint.

## Emotion steering

Steering adds a scaled 1024-float vector after the last transformer block, before
output normalization. It does not add another inference pass. The injection is
after KV-producing operations, so voice caches remain usable across emotions.

First prepare generated sibling graphs. This needs Python with `numpy` and `onnx`
(the same model-preparation dependencies as other graph tools):

```sh
python tools/make_soura_onnx.py --models models --vectors /path/to/vectors.npz \
  --source https://huggingface.co/Sourajit123/SouraTTS
```

Input can be an NPZ with named vectors or a float32 NPY of shape `[6,1024]`.
Row order is **neutral, angry, disgust, fear, happy, sad**. Values must be finite.
Custom vector magnitudes are preserved; the neutral row is ignored. Use the
source flag to record actual provenance for custom vectors. The tool writes
`*_soura.onnx`, `soura_vectors.npy`, and `soura.json`; original graphs are retained.
Running it again regenerates those derived files. No downloads happen implicitly.

```sh
# Off: existing models and existing behavior; no steering assets needed.
./pocket-tts 'Hello there.' example.wav plain.wav

# On: CLI synthesis.
./pocket-tts --soura --emotion happy --intensity 0.8 --seed 31 \
  'Hello there.' example.wav happy.wav

# On: request-controlled HTTP server. Requests default to neutral.
./pocket-tts --server --soura
curl http://localhost:8080/tts -H 'Content-Type: application/json' \
  -d '{"text":"Hello there.","voice":"example.wav","emotion":"happy","intensity":0.8,"seed":31}' \
  --output happy.f32

# Select your own vectors without replacing the installed defaults.
./pocket-tts --server --soura --soura-vectors /path/to/custom.npy
```

| Control | Default | Behavior |
|---|---|---|
| `--soura` | off | Select generated steering graphs; missing assets are an explicit error. |
| `--soura-vectors PATH` | `models/soura_vectors.npy` | Choose an alternate NPY; requires `--soura`. |
| `--emotion LABEL` / HTTP `emotion` | neutral | One of the six labels above. CLI control requires `--soura`. |
| `--intensity N` / HTTP `intensity` | 1 | Finite value from 0 through 1.2; zero bypasses the shift. |
| `--seed N` / HTTP `seed` | random / existing RNG progression | Optional integer from 0 through 2^53−1 for reproducible latent sampling. |

CLI emotion, intensity and seed flags are for one-shot synthesis, not server
startup defaults. Set controls per HTTP request. Omitted emotion resets to neutral;
requests cannot inherit the previous request's emotion. Invalid controls return
HTTP 400 before audio headers. A server without `--soura` rejects positive
non-neutral steering. `/health` reports `soura` and `sampling_seed` capabilities.
The OpenAI-compatible `/v1/audio/speech` endpoint stays neutral.

Remove `--soura` and its dependent flags to turn steering off. Keep `--soura`
with neutral/zero intensity to test the same graph with no shift. Those two
comparisons can differ slightly in float rounding. Fixed seeds reproduce latent
sampling; WAV bytes can still vary because decode batching/crossfades depend on
timing. Do not use exact WAV equality as a determinism test.

Historical M4 Max companion measurements found no consistent steering overhead
and first-PCM medians around 25–26 ms. These are not new benchmarks of this port
or microphone-to-speaker latency guarantees. Reference recordings usually carry
more of the perceived emotion than the steering vectors do.

## Reference-derived vectors (experimental, explicit opt-in)

Use cached voice embeddings to derive directions from your own recordings. This
requires `onnxruntime` as well as `numpy` and `onnx`. The command writes a separate
vector file and provenance JSON; it does not alter any runtime configuration.

```sh
./pocket-tts --prepare-voice neutral.wav
./pocket-tts --prepare-voice happy.wav
python tools/derive_soura_vectors.py \
  --model models/flow_lm_main_delta_attn_flow_int8.onnx \
  --ops-lib ./libptt_custom_ops.dylib \
  --neutral voices/.cache/neutral.emb \
  --reference happy=voices/.cache/happy.emb \
  --alpha 0.3 --output results/my-steering.npy
./pocket-tts --server --soura --soura-vectors results/my-steering.npy
```

If your runtime has `models/bos_before_voice.npy`, supply that same file with
`--bos`. Otherwise omit it. Supply your platform's custom-ops library for a
custom-attention graph, or use the original `flow_lm_main_int8.onnx` graph without
`--ops-lib`. Repeat `--reference` for additional labels. Unspecified labels remain
zero; this does not claim to learn absent emotions. Different labels may explicitly
share a recording. Alpha scales vector norm relative to the neutral activation
norm; it is distinct from the per-request intensity. Audition the results before
choosing them for your application. This is an experimental heuristic, not training.

## Prepare caches in disposable processes

```sh
./pocket-tts --prepare-voice example.wav
# For multiple voices, use one short-lived process per voice:
for voice in neutral.wav happy.wav angry.wav; do
  ./pocket-tts --prepare-voice "$voice"
done
./pocket-tts --server
```

`--prepare-voice` builds or validates the `.emb` and `.kv` caches, then exits. It
performs conditioning but generates no audio frames and writes no WAV. Existing
cache freshness checks are reused. A failed cache write returns a nonzero exit.
It cannot be combined with `--server`, `--no-cache`, or synthesis arguments.
Omit this command to keep the old on-demand cache behavior.

The benefit is process lifetime: large conditioning allocations are released
before the persistent server starts. The companion recorded memory growth from
437 MB to 3077 MB while building six uncached references in a persistent worker;
this command makes its disposable-process workaround available without a Node
service. It does not promise a particular memory reduction on every platform.

Punctuation (`--keep-commas`), leading-silence protection (`--trim-leading`), and
thread controls already existed and retain their current flags/defaults. App
sentence grouping, profile routing, and playback effects remain client concerns.

## Verification

```sh
python -m unittest discover -s tests -p 'test_*.py'
cmake -B .build -DCMAKE_BUILD_TYPE=Release -DPTT_BUILD_TESTS=ON
cmake --build .build -j
# Requires prepared *_soura.onnx and soura_vectors.npy:
ctest --test-dir .build --output-on-failure
python tests/test_steering_http.py --binary ./pocket-tts \
  --models models --voices voices --voice example.wav
```

`PTT_BUILD_TESTS` defaults off. Override `PTT_TEST_MODELS`, `PTT_TEST_VOICES`, and
`PTT_TEST_VOICE` in CMake for isolated test assets. The native test checks original
versus zero-steering latents, fixed-seed repeatability, all labels, emotion/cache
isolation, cancellation, and disabled-mode rejection. The HTTP test starts and
stops isolated workers; it also tests malformed controls and disconnect recovery.
