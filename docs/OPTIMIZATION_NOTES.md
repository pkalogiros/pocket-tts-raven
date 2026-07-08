# PocketTTS latency optimization notes

How PocketTTS-RAVEN got from 9.2x to 31x realtime natively (and 15x in the
browser): every optimization that worked, with its mechanism and measured
impact — and every experiment that didn't, with the lesson. Model rewrite
scripts live in `tools/`; `tools/prepare_models.sh` runs the whole pipeline.

## Current shipping stack

Public repo layout:

```text
pocket-tts                         native CLI / HTTP server
src/pocket_tts.cpp                 runtime driver and custom-op registration
models/                            active native model directory
webdemo/models/                    browser-downloaded model directory
webdemo/vendor/ptt/                prebuilt Emscripten runtime
webdemo/vendor/ort/                prebuilt onnxruntime-web runtime
```

The default optimized native path is:

```text
text_conditioner.onnx
flow_lm_main_delta_attn_flow_int8.onnx   delta-KV + AttentionTail + dedup + merged flow
mimi_decoder_delta_convtr_int8.onnx      delta-KV + ConvTranspose fusion + AccelConv + dedup
mimi_encoder.onnx                        voice cloning encoder
tokenizer.model                          SentencePiece tokenizer
```

The browser path uses the same AR/text/encoder assets, plus the portable
`mimi_decoder_delta_int8.onnx` decoder because the Apple-only decoder custom
ops do not help in WASM. Native Linux / clang-Windows follows the same
pattern since the 2026-07-08 round: the attention custom op is portable, so
those builds run `flow_lm_main_delta_attn_flow_int8.onnx` +
`mimi_decoder_delta_int8.onnx`.

## Glossary

| Term | Meaning |
|---|---|
| AR | Autoregressive transformer that emits one latent frame at a time. |
| Mimi | Kyutai audio codec: encoder turns reference speech into conditioning latents; decoder turns generated latents into waveform audio. |
| KV cache | Transformer key/value state reused across autoregressive steps. |
| Delta-KV | Graph/runtime contract where the model emits only the newly written KV slice instead of the full cache. |
| Flow | One-step flow-matching model that maps AR conditioning into a Mimi latent frame. |
| ORT session | One ONNX Runtime session loaded for a specific graph. Fewer sessions/runs usually means less boundary overhead. |
| AttentionTail | Custom op that computes only the newest-token attention tail used by the hot AR path. |

## Regenerating the current installed models

`tools/prepare_models.sh` runs the full export and rewrite pipeline. To rerun
individual installed rewrites during development:

Decoder chain (`delta -> convtr fusion -> AccelConv -> dedup`):

```text
uv run --no-project --with onnx python tools/make_decoder_convtr_custom_onnx.py \
  models/mimi_decoder_delta_int8.onnx /tmp/dec_convtr.onnx
uv run --no-project --with onnx python tools/make_decoder_accel_conv_onnx.py \
  /tmp/dec_convtr.onnx /tmp/dec_accel.onnx
uv run --no-project --with onnx python tools/make_dedup_positional_onnx.py \
  /tmp/dec_accel.onnx models/mimi_decoder_delta_convtr_int8.onnx
```

Main AR chain (`delta -> AttentionTail -> dedup -> merged flow`):

```text
uv run --no-project --with onnx --with onnxruntime python tools/make_delta_kv_onnx.py \
  models/flow_lm_main_int8.onnx /tmp/ar_delta.onnx --verify
uv run --no-project --with onnx python tools/make_attention_tail_custom_onnx.py \
  /tmp/ar_delta.onnx /tmp/ar_attn.onnx
uv run --no-project --with onnx python tools/make_dedup_positional_onnx.py \
  /tmp/ar_attn.onnx /tmp/ar_attn_dedup.onnx
uv run --no-project --with onnx python tools/make_merged_flow_onnx.py \
  /tmp/ar_attn_dedup.onnx models/flow_lm_flow_int8.onnx \
  models/flow_lm_main_delta_attn_flow_int8.onnx
```

Verify any regenerated model against its predecessor:

```text
uv run --no-project --with onnx --with onnxruntime python \
  tools/verify_model_equivalence.py OLD.onnx NEW.onnx
```

## Summary of what worked

| Optimization | Mechanism | Impact |
|---|---|---|
| Delta-KV rewrite | AR graph emits only the new KV slice; runtime scatters into a persistent cache | removes per-step cache copies |
| FastKV / shape-plumbing removal | delete graph work that only recomputed known dimensions | first big AR win |
| Cross-layer dedup (CSE) | RoPE/mask computed once, shared across layers | fewer ops per step |
| Merged flow | one-step flow ODE folded into the main graph | one session call less per frame |
| AttentionTail custom op | attention tail as one kernel (Accelerate/AMX native, SIMD128 in WASM) | ~1.4x AR native, ~1.3x WASM |
| Fused ConvTranspose decoder | polyphase rewrite + Accelerate GEMM | ~1.4x decoder (Apple silicon) |
| KV snapshot cache (.kv) | voice conditioning state restored from disk in ~4ms | cold-start killer |
| WASM: API-boundary exceptions | default ORT build wraps every call in JS trampolines | 2.2x on the decoder |
| WASM: fixed (non-growable) memory | enables bounds-check elimination in V8 | ~10% |
| WASM: pool tuning | 10 threads on wide desktops, 2 on phones | +4% / phones usable |

## Experiments that did NOT work (kept for the lessons)

| Experiment | Result | Lesson |
|---|---|---|
| Split AR/decoder into two WASM instances | 13.8x vs 15.8x single | serial conditioning (Amdahl) + in-process contention beat the mock's projection |
| Norm/GELU fusion (−203 graph nodes) | ±0% | ORT's per-op profiler overhead masqueraded as dispatch cost; engine is compute-bound |
| Static-shape AR graph | ±0% | the "glue" was runtime index math, not shape plumbing |
| Relaxed-SIMD (int8 SDOT kernels) | +2–3% | batch-1 GEMV is memory-bandwidth bound; int4 weights are the real lever |
| mimalloc | heap 369MB → 1.3GB | disqualifying for mobile |
| Bigger decode batches | −10–20% | decoder pool bursts stall the AR stage |

The sections below are detailed working notes, in roughly chronological order.
Some older entries mention the internal `onnx/english_2026-04` symlink bundle;
the public repo ships and runs from `models/`.

## Current recommended runtime flags

For local CLI latency tests:

```bash
pocket-tts \
  --profile \
  --low-latency \
  --trim-leading \
  --trim-confirm 2 \
  --trim-sustain 15 \
  --threads-ar 3 \
  --threads-dec 2 \
  --threads-full 2 \
  --temperature 0.7 \
  --models-dir models \
  --tokenizer models/tokenizer.model \
  --voices-dir voices \
  "Beware: the nearest mirror hides a portal to the abyss—look, but don't step!" \
  example.wav \
  /private/tmp/pockettts-test.wav
```

`--no-delta-kv` disables the optimized flow and decoder delta model paths. Do
not use it for latency benchmarks unless intentionally measuring the old path.

## What changed

### 1. Stage profiling

`pocket_tts.cpp` now reports first-audio stages under `--profile`, including:

```text
latent_gen_ready
first_latent_ready
first_decoder_done
trim_gate_open
first_callback
```

Why it matters: this showed the queue/thread code was not the bottleneck. The
remaining latency was model graph work, first decoder, and trim confirmation.

### 2. Per-session ONNX Runtime threads

The binary supports separate thread controls:

```text
--threads-ar
--threads-dec
--threads-full
```

The stable low-latency setting in the original sweep was `2 / 2 / 2`; later
sweeps moved the current recommendation to `3 / 2 / 2` when a third AR core is
available. More threads are not automatically better on M4 because AR,
decoder, ASR, LLM, and the game can contend for cores.

### 3. Trim confirmation

The binary supports:

```text
--trim-leading
--trim-confirm <n>
--trim-sustain <n>
```

Current default/recommended value is `--trim-confirm 2`. This does not speed up
the model. It reduces the wait between decoded audio and first useful emitted
speech while still guarding against initial silence/noise.

Current default/recommended sustain is `--trim-sustain 15`. It keeps the
leading-burst detector active. Do not use `--trim-sustain 0` in production:
the detector repro on 2026-07-06 was `BLIP before.wav (burst ends 105ms)` for
the sustain-0 command and `ok after.wav` with the default sustain gate.

### 4. Delta-KV ONNX model

The delta-KV model avoids outputting a full updated KV cache every AR step. The
model outputs only the newly computed packed K/V slice:

```text
out_state_N: [2, 1, delta_len, 16, 64]
```

C++ writes that delta slice into the persistent full cache buffer.

Why it works: full cache tensors are large. Avoiding full-cache output/copy keeps
the AR loop focused on new tokens.

### 5. FastKV graph rewrite

The latest model optimization is in `tools/make_delta_kv_onnx.py`.

Two passes were added:

```text
optimize_cache_update_shapes()
optimize_delta_valid_cache()
```

The important profile finding was that ONNX was doing expensive packed-cache
`Gather` work mostly to compute shapes, not because it needed cache data.

Old hot pattern:

```text
Gather(state, K_or_V) -> Slice(step:end) -> Shape -> Expand(new_kv)
```

New pattern:

```text
Sub(end, step) -> Concat([1], length, [16, 64]) -> Expand(new_kv)
```

The rewrite also avoids building a packed `valid_cache` only to gather K/V back
out for attention. Internally it builds split valid K and valid V tensors, but
the public model ABI remains the same packed delta output. No C++ state-layout
change was needed.

Why it works: it removes full-cache reads and packed-cache graph plumbing while
preserving the exact same model math.

### 6. Decoder delta-KV model

The Mimi decoder also had two active packed float32 caches:

```text
state_19: [2, 1, 8, 1000, 64]
state_22: [2, 1, 8, 1000, 64]
```

The original decoder repacked full updated caches with `ScatterND` and returned
the full cache tensors. `tools/make_decoder_delta_kv_onnx.py` rewrites the decoder so
those outputs contain only the new packed K/V delta:

```text
out_state_19: [2, 1, 8, delta_len, 64]
out_state_22: [2, 1, 8, delta_len, 64]
```

C++ now writes those slices into the preallocated full decoder cache, including
circular wraparound. The attention path still uses the same updated K/V tensors,
so this is quality-preserving. Exact ORT equivalence was verified, including a
wraparound case, and deterministic CLI WAV output matched byte-for-byte:

```text
ae26865cab60970912434e0c0c168a4db860309090513aa5b77869b0ed0c3a3f
```

## Impact measured

Installed optimized model compared against the backup pre-FastKV delta model,
same binary, voice, prompt, temperature, trim, and thread flags:

```text
backup model:
  first callback median        44ms
  latent_gen_ready median      14.6ms
  first_latent_ready median    18.3ms
  first_decoder_done median    25.2ms
  flow_lm_main avg/run         2.9ms
  RTF median                   16.0x

installed FastKV model:
  first callback median        39ms
  latent_gen_ready median      11.9ms
  first_latent_ready median    14.8ms
  first_decoder_done median    22.1ms
  flow_lm_main avg/run         2.0ms
  RTF median                   17.6x
```

Main AR model speed improved by about 30%.

ORT profile category shift:

```text
Before FastKV steady AR:
  shape/copy          ~39.9%
  KV gather/scatter   ~20.2%
  elementwise/reduce  ~16.1%

After FastKV steady AR:
  shape/copy          ~46.9%
  matmul/linear       ~24.0%
  elementwise/reduce  ~19.5%
  KV gather/scatter   ~5.4%
```

The visible first callback did not improve by 30% because trim confirmation and
first decoder now account for more of the user-visible latency.

Decoder delta impact, compared against the immediately previous binary that
used the original decoder model with single-buffered decoder caches:

```text
old decoder:
  first callback median          27.36ms
  first_decoder_done median      27.36ms
  first decoder duration median   6.42ms
  RTF median                     17.96x

decoder delta:
  first callback median          26.64ms
  first_decoder_done median      26.64ms
  first decoder duration median   5.95ms
  RTF median                     17.73x
```

ORT per-op profile confirmed the intended graph work was removed:

```text
old decoder all runs:
  ScatterND total          3.17ms
  KV gather/scatter total 15.36ms

decoder delta all runs:
  ScatterND total          0.18ms
  KV gather/scatter total 10.78ms
```

This is a smaller win than the flow FastKV rewrite because decoder first-chunk
time is dominated by `ConvTranspose`, `Conv`, and transformer matmuls. The
quality-preserving decoder graph waste was worth about half a millisecond on the
first decoder call in the paired benchmark.

Static AR-step experiment:

```text
Wanted shape:
  sequence        [1, 1, 32]
  text_embeddings [1, 0, 1024]
  cache outputs   fixed one-step delta
```

This was not kept. Freezing both `sequence` and `text_embeddings` changed ORT
outputs on valid hot-path states, so it was not quality-preserving:

```text
worst eos_logit diff     0.112986
worst conditioning diff  0.055140
```

A safer partial variant was exact:

```text
sequence        [1, 1, 32]
text_embeddings dynamic, runtime still passes [1, 0, 1024]
cache outputs   [2, 1, 1, 16, 64]
```

That exact partial model also was not kept because it did not improve runtime.
Model-level timing was noise, and full CLI A/B testing was worse:

```text
with static AR-step:
  first_callback median 42.62ms
  frame:main_model avg   2.01ms

without static AR-step:
  first_callback median 36.67ms
  frame:main_model avg   1.97ms
```

The installed static model and symlink were removed. Current active runtime uses
the normal `flow_lm_main_delta_int8.onnx` FastKV path.

State-only text/voice prefill experiment:

```text
Wanted model:
  flow_lm_main_prefill_delta_int8.onnx
  same inputs as flow_lm_main_delta_int8.onnx
  outputs only out_state_*
```

The goal was to prune `conditioning` and `eos_logit` during prefill, because
`cond_pass()` discards them and only needs updated state. The generated model
was exact on reconstructed state outputs:

```text
tools/make_prefill_state_onnx.py ... --verify
  verified exact state equivalence
```

It was not kept because paired CLI benchmarks did not improve latency:

```text
with state-only prefill:
  first_callback median 40.14ms
  latent_gen_ready median 11.91ms
  text_conditioning_pass median 10.18ms

without state-only prefill:
  first_callback median 35.32ms
  latent_gen_ready median 11.84ms
  text_conditioning_pass median 10.08ms
```

The prefill model and symlink were removed from the active bundle. Current
active runtime still uses the normal `flow_lm_main_delta_int8.onnx` FastKV path.

## Regenerating the optimized delta model

Use `uv run` so the required `onnx` and `onnxruntime` packages are available:

```bash
cd .

uv run --no-project --with onnx --with onnxruntime python \
  tools/make_delta_kv_onnx.py \
  onnx/english_2026-04/flow_lm_main_int8.onnx \
  /private/tmp/flow_lm_main_delta_int8.fastkv.onnx \
  --verify
```

Install it by writing the real target in `models/`:

```bash
cp -p models/flow_lm_main_delta_int8.onnx \
  onnx/english_2026-04/flow_lm_main_delta_int8.before-new-change.onnx

cp -p /private/tmp/flow_lm_main_delta_int8.fastkv.onnx \
  models/flow_lm_main_delta_int8.onnx
```

Do not delete or recreate this symlink:

```text
onnx/english_2026-04/flow_lm_main_delta_int8.onnx
```

It should continue to point to:

```text
../../models/flow_lm_main_delta_int8.onnx
```

## Regenerating the decoder delta model

Use the same Python dependency path as the flow rewrite:

```bash
cd .

uv run --no-project --with onnx --with onnxruntime \
  python tools/make_decoder_delta_kv_onnx.py \
  onnx/english_2026-04/mimi_decoder_int8.onnx \
  /private/tmp/mimi_decoder_delta_int8.onnx \
  --verify
```

Install it by writing the real target in `models/`:

```bash
cp -p /private/tmp/mimi_decoder_delta_int8.onnx \
  models/mimi_decoder_delta_int8.onnx
```

The bundle symlink should be:

```text
onnx/english_2026-04/mimi_decoder_delta_int8.onnx
  -> ../../models/mimi_decoder_delta_int8.onnx
```

## Restoring the pre-FastKV model

Restore by copying the backup into the real target:

```bash
cd .

cp -p onnx/english_2026-04/flow_lm_main_delta_int8.before-fastkv.onnx \
  models/flow_lm_main_delta_int8.onnx
```

Again, do not replace the bundle symlink itself.

## Sanity checks

Check symlink:

```bash
readlink onnx/english_2026-04/flow_lm_main_delta_int8.onnx
```

Expected:

```text
../../models/flow_lm_main_delta_int8.onnx
```

Check graph counts:

```bash
uv run --no-project --with onnx --with onnxruntime python - <<'PY'
from collections import Counter
import onnx

p = "onnx/english_2026-04/flow_lm_main_delta_int8.onnx"
m = onnx.load(p)
c = Counter(n.op_type for n in m.graph.node)
print("nodes", len(m.graph.node))
for op in ["Gather", "Slice", "Shape", "Concat"]:
    print(op, c[op])
onnx.checker.check_model(m)
print("onnx checker ok")
PY
```

Current FastKV model should be around:

```text
nodes 2355
Gather 95
Slice 49
Shape 136
Concat 123
```

The pre-FastKV backup was:

```text
nodes 2265
Gather 119
Slice 55
Shape 148
Concat 93
```

Check decoder symlink:

```bash
readlink onnx/english_2026-04/mimi_decoder_delta_int8.onnx
```

Expected:

```text
../../models/mimi_decoder_delta_int8.onnx
```

## Things that were tried and not kept

These did not produce a reliable quality-preserving latency win:

```text
SPSC/ring buffer replacement for deque/mutex
vDSP trim scanning
smaller first-chunk fast-start variants
combine-first-step
static AR-step ONNX side-session
state-only prefill ONNX model
decoder custom-attention ONNX model
```

The current useful remaining targets are:

```text
more shape/copy rewrites in flow_lm_main_delta_int8
careful trim-confirm experiments if accepting more aggressive speech onset
```

### Decoder custom-attention experiment

Tried after the AR custom-attention win:

```text
models/mimi_decoder_delta_attn_int8.onnx
tools/make_decoder_attention_tail_custom_onnx.py
```

It replaced the two decoder transformer attention tails with a custom ORT op
for decoder cache layout `[2, 1, heads, capacity, dim]`.

Result: not deployed. The active bundle intentionally does not contain this
symlink:

```text
onnx/english_2026-04/mimi_decoder_delta_attn_int8.onnx
```

Reason: it removed some ONNX shape/copy work, but the replacement custom op
cost about the same amount. Paired benchmark with AR custom attention still on:

```text
baseline decoder: first decoder median 5.78ms, decoder avg median 31.50ms
custom decoder:   first decoder median 5.76ms, decoder avg median 35.06ms
```

A hand NEON decoder path was also tried and removed; it was slower:

```text
baseline decoder: first decoder median 5.94ms, decoder avg median 31.62ms
NEON custom:      first decoder median 6.47ms, decoder avg median 71.97ms
```

Current active decoder remains:

```text
onnx/english_2026-04/mimi_decoder_delta_int8.onnx
  -> ../../models/mimi_decoder_delta_int8.onnx
```

### Direct s16le server transport

The `/tts` server endpoint now accepts:

```json
{"format":"s16le"}
```

When requested, PocketTTS still runs the model in float32, but converts each
emitted chunk to PCM s16le before sending:

```text
audio/pcm;rate=24000;encoding=signed-integer;bits=16
```

This does not make model compute faster. It reduces HTTP payload size by about
half and lets clients avoid doing their own per-chunk float32-to-int16
conversion.

Temporary port-8098 benchmark:

```text
float32 payload: ~415-447 KB
s16le payload:   ~199-238 KB
```

The running PocketTTS server must be restarted to get this transport win. Old
server processes will ignore `format: "s16le"` and continue returning float32.

### Decoder Conv/ConvTranspose profiling

Active decoder:

```text
onnx/english_2026-04/mimi_decoder_delta_int8.onnx
  -> ../../models/mimi_decoder_delta_int8.onnx
```

ORT profiling shows the decoder is now mostly convolution work, not attention
glue. For the first individual decoder invocation:

```text
total node time:      ~6.0-6.6ms
ConvTranspose:        ~1.8-1.9ms
Conv:                 ~0.9ms
slice/concat/add glue: ~0.5ms ceiling
```

The ConvTranspose layers have kernel = `2 * stride` and are followed by
streaming overlap-state logic. The first stride is added to previous state, the
middle is emitted downstream, and the last stride becomes next state. So a
simple crop/static-shape rewrite cannot skip much work without changing output.

CoreML decoder-only EP was tested as a disabled experiment and removed:

```text
MLComputeUnitsCPUAndGPU: rejected by this ORT build
CPUAndGPU + MLProgram: model plan compile failed
CPUAndGPU default model format: crashed during first delegated decoder run
```

Do not deploy CoreML decoder without a separate fixed model/export path.

### Native/autovec C++ build experiment

Added opt-in CMake knobs so side builds can be produced without replacing the
stable binary:

```text
-DPTT_OUTPUT_DIR=/private/tmp/pockettts-native
-DPTT_NATIVE_OPT=ON
```

`PTT_NATIVE_OPT` enables `-mcpu=native`, `-ffp-contract=fast`, and target LTO
when the compiler supports it. This affects PocketTTS C++ glue and custom ORT
kernels only; it does not rebuild the prebuilt ONNX Runtime CPU kernels.

Side binaries tested:

```text
/private/tmp/pockettts-native/pocket-tts   # -O3 + native/LTO/fp-contract
/private/tmp/pockettts-ofast/pocket-tts    # -Ofast + native/LTO/fp-contract
```

Result with `--low-latency --trim-leading --trim-confirm 2 --trim-sustain 15 --threads-ar 2
--threads-dec 2 --threads-full 2 --temperature 0.7`:

```text
stable:
  latent_gen_ready median       ~10.9-11.5ms
  first_decoder_done median     ~18.9-19.8ms

native:
  latent_gen_ready median       ~11.6ms
  first_decoder_done median     ~20.0ms

ofast:
  latent_gen_ready median       ~11.1ms
  first_decoder_done median     ~18.9ms
```

Conclusion: native/autovec flags alone do not materially improve the first
model-compute stages. The apparent callback/trim movement is waveform-onset
jitter, not a reliable compiler win. Keep the stable binary; use custom ORT
kernels, graph rewrites, or a custom-built ONNX Runtime if we want SIMD to hit
the actual hotspots.

### Decoder ConvTranspose fused custom op

Implemented a quality-preserving decoder ConvTranspose rewrite:

```text
pocket_tts_custom_attention.hpp
  DecoderConvTransposeOverlap custom ORT op

tools/make_decoder_convtr_custom_onnx.py
  rewrites the three large decoder ConvTranspose streaming blocks

models/mimi_decoder_delta_convtr_int8.onnx
onnx/english_2026-04/mimi_decoder_delta_convtr_int8.onnx
  -> ../../models/mimi_decoder_delta_convtr_int8.onnx
```

This is opt-in only:

```text
--decoder-convtr
```

Default behavior is unchanged. Without `--decoder-convtr`, runtime still uses:

```text
onnx/english_2026-04/mimi_decoder_delta_int8.onnx
  -> ../../models/mimi_decoder_delta_int8.onnx
```

What changed:

```text
ConvTranspose(kernel=2*stride)
  -> Slice(first stride) + previous state
  -> Concat(rest)
  -> Slice(current output) and Slice/Sub(next state)
```

became one custom op that uses the `kernel = 2 * stride` polyphase identity:

```text
current output[n, phase] =
  bias + current_input[n] * W_first[phase]
       + previous_input[n-1] * W_tail[phase]

current output[0, phase] uses previous streaming state instead of previous input
next state[phase] = last_input * W_tail[phase]
```

Validation with deterministic `--temperature 0`:

```text
same sample count: yes
max abs sample delta: 3.73e-7
RMSE: 4.48e-8
SNR: ~126.5 dB
```

Alternating benchmark, 10 stable + 10 fused, same flags:

```text
--low-latency --trim-leading --trim-confirm 2 --trim-sustain 15
--threads-ar 2 --threads-dec 2 --threads-full 2
--temperature 0.7
```

Results:

```text
stable:
  first_decoder_done median   19.8ms
  decoder_total median       228.5ms
  decoder_avg median          32.3ms
  RTFx median                 17.9x

fused decoder:
  first_decoder_done median   17.7ms
  decoder_total median       158.6ms
  decoder_avg median          22.8ms
  RTFx median                 24.1x
```

Observed gain:

```text
first decoder completion: ~1.12x faster (~10-12%)
total decoder runtime:    ~1.44x faster
overall RTFx:             ~1.35x higher
```

Fused-model ORT profile now shows the remaining decoder hotspots are mostly:

```text
Conv                         ~29%
DecoderConvTransposeOverlap  ~17%
MatMul/DynamicQuantizeMatMul ~21%
decoder attention cache ops  still visible
```

Next decoder targets are ordinary Conv nodes and decoder transformer
Scatter/Split/Gather attention-cache work.

## 2026-07-06 optimization round

Paired temp-0 benchmark (same flags as the standard CLI latency test), start
of round vs end of round:

```text
start (decoder-convtr still opt-in, so default = old decoder):
  RTFx median            18.45x
  first chunk median     28ms
  decoder total median   250.5ms
  AR total median        93.9ms   (+ flow session 13.3ms)

end (all defaults, temp 0):
  RTFx median            30.33x
  first chunk median     25ms
  decoder total median   137.7ms
  AR total median        113.2ms  (flow now inlined in main graph)

end (temp 0.7, the production setting):
  RTFx median            29.84x
  first chunk median     25ms
```

### 1. --decoder-convtr is now the default

The fused ConvTranspose decoder model is used by default when present.
`--no-decoder-convtr` restores the old path. All custom-op model variants
(including `flow_lm_main_delta_attn*`) are now gated on `kCustomOpsAvailable`
(`pocket_tts_custom_attention.hpp`), so non-Apple builds never try to load
models containing `pockettts.*` ops.

### 2. AccelConv decoder Conv rewrite (kept, installed)

`tools/make_decoder_accel_conv_onnx.py` replaces all nine stride-1/dilation-1/
group-1 decoder Conv nodes with `AccelConv` custom ops (Accelerate sgemm, one
GEMM per kernel tap on a shifted submatrix, weights pre-packed offline to
`[K, Cout, Cin]`, no im2col). Convs whose sole consumer is `Elu(alpha=1)` use
the `AccelConvElu` fused variant (vDSP/vvexpf).

Installed into the real target of:

```text
onnx/english_2026-04/mimi_decoder_delta_convtr_int8.onnx
```

Backup of the pre-AccelConv fused decoder:

```text
onnx/english_2026-04/mimi_decoder_delta_convtr_int8.before-accelconv.onnx
```

Impact (temp-0 paired): decoder total 173.5ms -> ~137ms, RTFx 25.2 -> ~30.3.
Lockstep model equivalence: audio outputs max abs diff <= 8e-7.

### 3. Dequantized decoder matmuls (tried, NOT kept)

`tools/make_dequant_matmul_onnx.py` converts the QOperator dynamic-quant chains
(DynamicQuantizeLinear + MatMulInteger + Cast + Mul) to plain f32 MatMul with
offline-dequantized weights. On the decoder transformer this was SLOWER:
decoder total 139ms -> 180ms, RTFx 30 -> 24.4. MLAS f32 GEMM loses to the
fused int8 DynamicQuantizeMatMul at these shapes. Script kept for reference;
model not installed.

### 4. Cross-layer positional dedup (kept, installed)

`tools/make_dedup_positional_onnx.py` does cross-layer CSE: every transformer layer
recomputes identical RoPE cos/sin tables and attention masks from its own
step-counter state, and the counters always hold equal values, so layers 1..N
are rewired to share layer 0's positional subgraph.

```text
flow_lm_main_delta_attn_int8.onnx   1767 -> 1165 nodes (bitwise-exact outputs)
mimi_decoder_delta_convtr_int8.onnx  872 -> 558 nodes (bitwise-exact outputs)
```

Backups: `*.before-dedup.onnx` in the bundle directory.

End-to-end this was NOT a measurable win (within run noise). Important
finding: the "shape/copy ~47%" category in ORT profiles is heavily inflated
by per-node profiling overhead on microsecond-scale ops. Removing a third of
the graph nodes moved steady-state AR time by less than noise, so the real
glue cost is small and deeper AR graph surgery (including a hand-rolled AR
step) would gain far less than the profile percentages suggest. Kept because
it is bitwise-exact and strictly less work.

### 5. Merged main+flow model (kept, installed)

`tools/make_merged_flow_onnx.py` inlines `flow_lm_flow` into the main AR graph for
the lsd_steps=1 case: `latent = flow_x + flow_dir(conditioning, s=0, t=1,
flow_x)`, with noise `flow_x` as a new input and `latent` as a new output.
One ONNX Run() per frame instead of two.

```text
models/flow_lm_main_delta_attn_flow_int8.onnx
onnx/english_2026-04/flow_lm_main_delta_attn_flow_int8.onnx (symlink)
```

Runtime uses it only when lsd_steps == 1; `--no-merged-flow` opts out. Wall
clock is neutral on macOS (IoBinding Run() overhead was already tiny) but the
temp-0 output is bitwise-identical and the hot loop drops to a single
session, which matters for a future WASM port (per-Run JS boundary cost).

### Verification infrastructure (new)

```text
ptt_custom_ops.cpp -> libptt_custom_ops.dylib   (cmake target ptt-custom-ops)
tools/verify_model_equivalence.py
```

The dylib exposes RegisterCustomOps so python onnxruntime can load models
containing pockettts.* ops. `tools/verify_model_equivalence.py` lockstep-runs two
stateful models with C++-faithful state feedback (bool states init true,
delta-KV scatter with wraparound) and reports max abs diff on every output
and state.

Pitfall discovered: do NOT compare emitted CLI WAVs sample-by-sample across
binaries/models. Each decode-batch boundary consumes 120 samples in the
5ms crossfade, and batch boundaries depend on thread timing, so emitted
length shifts by multiples of 120 samples between variants even when the
models are exact. Use the lockstep verifier, and treat temp-0 WAV hashes as
meaningful only when equal (different hash does not imply different audio).

### Runtime knob sweep (2026-07-06, post-AccelConv)

Paired temp-0 sweep against the 2/2/2 default (RTFx 30.3, first chunk 25ms):

```text
--threads-ar 3      RTFx 31.3  first 22ms  latent_ready 9.4ms   <- adopt when a core is spare
--threads-ar 4      RTFx 31.0  first 22ms  latent_ready 8.9ms   (no gain over 3)
--threads-dec 3     RTFx 29.5  (worse; decoder is Accelerate/AMX-bound, ORT threads don't help)
--max-chunk 30      RTFx 27.5  (worse)
--first-chunk 2     first 26ms (no latency win)
--trim-confirm 1    first 23ms (no win: gate wait is model silence, not confirmation windows)
VECLIB_MAX_THREADS=1  slightly worse; leave Accelerate threading alone
```

New recommended low-latency flags when the game can spare a third core for the
AR session: `--threads-ar 3 --threads-dec 2 --threads-full 2`. The old
"2/2/2, more is not better" guidance predates the Accelerate decoder ops; AR
threads now also speed the text-conditioning pass (latent_gen_ready 11.7 ->
9.4ms). Keep 2/2/2 under heavy game contention.

### Leading "blip" investigation (2026-07-06 evening)

Symptom: a tiny vocal blip at the start of most generated WAVs. Root cause
found and it is NOT a regression from the optimization round (opt-out flags
reproduce session-start output bit-for-bit) and NOT the BOS mechanism (BOS
verified active):

The model itself emits a spurious speech fragment before the sentence with
certain voice references — measured up to ~270ms at full speech energy,
followed by ~100ms of silence, then the real sentence. It appears in the raw
ungated decoder output. It is voice-conditioning dependent: mostly voice refs
that are themselves short synthetic PocketTTS test outputs, plus a few
production register voices with shorter (~60-120ms) bursts.

Disproven hypotheses (measured, paired temp-0 renders):

```text
LEADING_DROP_SAMPLES 480 vs 540 vs 720: identical output, complete no-op
  (the drop region is deep inside audio the gate trims anyway)
preroll 2 -> 1 -> 0: shifts what is emitted, does not remove the artifact
decoder AccelConv/convtr: burst present with them disabled too
```

Fix shipped: a sustain check in the trim gate. On first detection the gate
buffers instead of emitting; if energy collapses within the sustain window
(`--trim-collapse` consecutive quiet windows, default 3) the candidate is
discarded as a spurious burst and trimming resumes; if it sustains, the whole
buffer is emitted (identical audio content, delayed decision). Defaults:

```text
--trim-sustain 15    (150ms observation; 0 disables, legacy behavior)
--trim-collapse 3    (30ms of quiet = burst)
```

Validated repro command/result:

```text
./pocket-tts --low-latency --trim-leading --trim-sustain 0 \
  --temperature 0 --threads-ar 3 --threads-dec 2 --threads-full 2 \
  --models-dir models \
  --tokenizer models/tokenizer.model \
  --voices-dir <voices dir> \
  "Could you read this short sentence for a trimming check?" \
  voices/example.wav /tmp/before.wav

same command without --trim-sustain 0 -> /tmp/after.wav

internal leading-burst checker:
  BLIP before.wav (burst ends 105ms)
  ok   after.wav
```

Conclusion: keep `--trim-sustain 15`; `--trim-sustain 0` is legacy/unsafe for
the affected short custom voice references.

Also fixed while in there: the gate used to skip sub-window remainders at
decode-chunk boundaries, which could place a real discontinuity inside
emitted audio; a carry buffer now keeps detector windows contiguous.

Measured cost/benefit (temp 0, recommended flags):

```text
first chunk: 22ms -> 30ms (the sustain window is the price)
RTFx: unchanged (~31x)
all production voices scan clean at sustain 15
pockettts-test-* fixture voices with 200-400ms fragments still burst —
  those are unfixable at the gate (they are genuine speech-length audio);
  replace those voice refs instead
```

The internal leading-burst checker detects the artifact in WAVs as:
burst = envelope rises above 0.03, falls below 0.01 for >=30ms, then real
speech starts. Use an equivalent local checker or manual listening pass to vet
new voice references. Burst-prone refs found:
all `pockettts-test-*-clean-short*`/`mp3ref`/`rawref`/`tempo115` fixtures,
plus some short custom references below sustain 15.

### WASM/Emscripten port (2026-07-07)

`webdemo/wasm/` compiles the full pocket_tts.cpp (voice cloning included)
against an ONNX Runtime WASM static library. Final measured numbers (M4 Max,
Node/V8, 8-thread global pool): 11.6-13.9x realtime, first audio 47-68ms.

Hard-won findings, in causal order:

```text
1. ORT wasm threads build REQUIRES the env-global thread pool
   (DisablePerSessionThreads per session; get_ort_env emscripten branch).
2. Exception mode is worth 2.2x on the decoder: building ORT with default
   exception catching wraps every call in wasm->JS->wasm invoke_ trampolines.
   Correct recipe (matches ort-web releases): --disable_wasm_exception_catching
   --enable_wasm_api_exception_catching, and compile only pocket_tts.cpp
   (the catching boundary) with -sDISABLE_EXCEPTION_CATCHING=0.
3. Custom ops are now portable (pocket_tts_custom_attention.hpp has an
   Accelerate backend and a WASM-SIMD backend; native output verified
   bit-identical after the refactor). In WASM: AttentionTail WINS
   (AR step 6.0 -> 3.4ms) but the conv/convtr custom kernels LOSE (single-
   threaded custom kernel vs MLAS threaded convs) — the deployed wasm mix is
   attn AR model + plain delta decoder.
4. ptt_stream_poll (non-blocking) exists because the Emscripten runtime
   thread must never block: pthreads proxy filesystem syscalls through it.
5. Tokenization in wasm delegates to the JS tokenizer via
   MAIN_THREAD_EM_ASM (sentencepiece's bundled protobuf collides with ORT's).
6. ORT 1.23 vs 1.27 and relaxed SIMD: ~7% only. Pool size/spin: noise —
   until kernels were trampoline-free, after which pool 8 scaled to 13.9x.
```

### Split-module WASM experiment (2026-07-07)

Hypothesis: AR and decoder in separate WASM instances (own ORT global pools)
escape shared-pool contention -> mock (separate OS processes, continuous
co-run) projected 19.6x ceiling at 6/6 vs 15.8x single-instance.

Reality in one worker (?engine=split): 13.8x best (6/6, spin on). The mock
missed (a) Amdahl: per-sentence conditioning is serial on the AR side and
starves the decoder (~300ms idle per long text), and (b) in-process
cross-instance contention: raising decoder threads made co-run decode
SLOWER (715ms -> 928ms busy for the same frames). Desktop verdict: split
loses to single-instance pool-10; native engine stays the default.

Kept as ?engine=split (&arpool=&decpool=, mobile default 2/2) because the
mobile regime differs: a shared pool of 2 serializes AR/decode almost
fully, so 2/2 across instances may still win on phones. Infra added:
ptt_latents_* (AR-side latent tap), ptt_create_ex + ptt_decode +
ptt_decoder_reset (decoder-only instances) - all additive, native output
hash-verified bit-exact.

### Norm-fusion experiment: the profiler lied (2026-07-07)

ORT's per-op profile painted ~60% of WASM AR time as glue ops (Add/Mul/
Gather/Shape, ~500 tiny nodes/step). tools/make_fuse_norms_onnx.py fused 19
LayerNorms + 6 GELUs + 9 SiLUs (1627 -> 1424 nodes), verified equivalent
to 2.4e-7... and the AR step did not move (3.25 -> 3.24ms).

Lesson: ORT profiling instrumentation costs ~20us PER NODE EVENT in wasm;
with 1600 tiny nodes that overhead dominates the profile and masquerades
as op time. Real dispatch is ~0.1us/node. The AR step's time is genuinely
in the int8 matmuls (MLAS GEMV) - i.e., the engine is compute-bound, not
dispatch-bound. Static-shape freeze (tools/make_static_ar_step_onnx.py) showed
the same null result for the same reason.

The fusion script is kept (correct, verified, harmless); the fused model
is NOT shipped (no benefit, adds an opset-17 requirement). Remaining
desktop lever: a relaxed-SIMD ORT build variant behind feature detection
(int8 dot-product kernels), est. single-digit %.

### Relaxed-SIMD A/B: RESULT +2-3% — parked as not worth shipping (2026-07-07)

Unblocked (the "toolchain mismatch" was a renamed .mjs still loading the
baseline .wasm — pair renamed module files by patching the embedded wasm
filename). MLAS's relaxed int8 kernels (qgemm_kernel_wasmrelaxedsimd,
relaxed_dot_i8x16 -> ARM SDOT) are present and active: AR step 3.27 ->
3.20ms, end-to-end within noise.

Interpretation: batch-1 int8 GEMV is MEMORY-BANDWIDTH bound (~36GB/s of
weight streaming per step), so faster dot arithmetic barely moves it.
This both explains the small win and STRENGTHENS the int4 case:
MatMulNBits halves the bytes streamed — the one lever that attacks the
actual bound. Not shipping a third variant for 2-3%.

### (superseded) Relaxed-SIMD A/B: built, blocked, parked (2026-07-07)

The last untested desktop/mobile lever: MLAS relaxed-SIMD int8 dot kernels
target exactly the proven bottleneck (DynamicQuantizeMatMul ~85% of AR
step). Artifacts ready:
  - ORT lib: ort-wasm/build-relaxed/Release/libonnxruntime_webassembly.a
  - module:  webdemo/vendor/ptt/pocket_tts_wasm_relaxed.{mjs,wasm}
  - harness: PTT_VARIANT=relaxed PTT_BENCH_ONLY=1 node webdemo/wasm/test-node.mjs

BLOCKED: relaxed module dies at init with "ASM_CONSTS[code] is not a
function" - classic mixed-emscripten EM_ASM table mismatch. ORT's build
ran update=True/emsdk 4.0.23, so the relaxed objects may pair with a
different emcc than the (working) baseline lib. To resume: rebuild the
BASELINE static lib under the now-active emsdk, relink both, rerun the
A/B (baseline AR step to beat: ~3.0ms at pool 10). If relaxed wins >=15%,
ship as a third feature-detected variant:
  WebAssembly.validate(<i8x16.relaxed_dot probe>) ? relaxed : plain.

Beyond that, the remaining ranked levers are int4 MatMulNBits weights
(bandwidth + 40MB download, quality-gated) and a WebGPU decoder for
mobile (frees both P-cores for AR; browser-only iteration).

## 2026-07-08 optimization round — portable custom attention (ARM + x86-64)

Until this round the custom-op header was gated `__APPLE__ ||
__EMSCRIPTEN__`, so native Linux/Windows had `kCustomOpsAvailable = false`
and the model selector fell all the way back to the *base*
`flow_lm_main_int8.onnx` — every delta variant in the native set needs the
attention op. Non-Apple was paying for a fallback nobody had measured.

**Change.** Third backend in `pocket_tts_custom_attention.hpp` using
GCC/Clang vector extensions (`__attribute__((vector_size(16)))`): one
source path that compiles to NEON on ARM and SSE on x86-64, no per-arch
intrinsics. Availability split into `kCustomAttentionAvailable` (any
GCC/Clang target) and `kAccelConvAvailable` (Apple only) so non-Apple
builds load the fast AR model but keep MLAS for the decoder.
`-DPTT_FORCE_PORTABLE` builds the portable backend on macOS for A/B.

**Measured** (M4 Max, page text, temp 0, single runs — deltas far above
run noise):

```text
base-model fallback (= non-Apple before)      14.4x realtime
portable AttentionTail + MLAS delta decoder   24.6x
same with the NEON fast path disabled         23.4x   (x86 codegen proxy)
Apple stock (Accelerate + AMX conv ops)       31-38x  (unchanged)
portable backend driving the conv ops          4.6x   (why convs stay MLAS)
```

Numerics: the portable kernel matches Accelerate to ~1e-7 per step and
hand-written NEON within 1-2% on timing; the same code compiled as
x86_64 machine code passes the correctness check under Rosetta.

**Why the conv ops stay Apple-only.** `AccelConv`/`ConvTransposeOverlap`
win on Apple because Accelerate's sgemm runs on AMX. A portable GEMM at
those shapes (512x512, K-tap accumulation) loses badly to ORT's threaded
MLAS convolutions — same conclusion the WASM round reached. Off Apple,
the delta decoder through MLAS is the right call.

**Caveats / next levers for non-Apple:**

- All x86 numbers above are codegen proxies measured on ARM. Real
  x86 hardware (e.g. a Zen 3 box) still needs to confirm absolute perf;
  correctness is already proven.
- MSVC has no vector extensions — an MSVC build compiles with no custom
  ops (old behavior). Use clang-cl on Windows for the fast path.
- Ranked follow-ups if non-Apple needs more: fp16 KV cache in the
  attention op (halves the per-token cache read — the AR scan is
  bandwidth-bound), splitting the 16 attention heads across 2-4 threads,
  32-byte vectors under `__AVX2__`, and shipping x86 binaries as
  `-march=x86-64-v3`.
