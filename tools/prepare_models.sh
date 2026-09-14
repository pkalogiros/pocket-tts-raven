#!/bin/bash
# Build the optimized PocketTTS model set.
#
#   ./tools/prepare_models.sh          # download originals + optimize -> models/
#   ./tools/prepare_models.sh --web    # also copy the web set -> webdemo/models/
#
# Step 1 downloads the original Kyutai Pocket TTS ONNX bundle
# (english_2026-04, ~165 MB) from Hugging Face. Every file is verified
# against a pinned sha256, so the hosting mirror does not need to be
# trusted. Step 2 applies this repo's deterministic graph rewrites. The
# delta-KV rewrites run built-in ORT equivalence checks; use
# tools/verify_model_equivalence.py for lockstep checks when changing or
# staging other rewrites.
#
# Requires: curl, python3 with uv (https://docs.astral.sh/uv/).
#
# To re-export the base models from the original weights instead of
# downloading them, see export_onnx.py (developer path, needs torch).
set -euo pipefail
cd "$(dirname "$0")/.."

# Original Kyutai ONNX bundle (a mirror of the official pocket-tts web
# release; hash-pinned below). Override the source with BUNDLE_URL=...
BUNDLE_URL="${BUNDLE_URL:-https://huggingface.co/KevinAHM/pocket-tts-onnx/resolve/main/onnx/english_2026-04}"
BUNDLE_FILES=(
    "tokenizer.model:d461765ae179566678c93091c5fa6f2984c31bbe990bf1aa62d92c64d91bc3f6"
    "text_conditioner.onnx:4ecee995fb69f85c7a7493d11f7b5ee15d9950facc7ab3f5c9c49ef1e03847bb"
    "mimi_encoder.onnx:853e2ca623b8782d94c3745ec6133bfdff7ce33d9b11128bd29ea03f28d76e3d"
    "flow_lm_main_int8.onnx:f9bd8106b79a0192c1c43399ab938fb24900a95c1c599870d75a884e99000116"
    "flow_lm_flow_int8.onnx:3dd781ee5abee9e195320bf0106bebd6372a852b3b36352524ee78b40554635d"
    "mimi_decoder_int8.onnx:3630450a3297a101792a6ac66619ebc70ab916b265e6220c2afaef8b1673f925"
    "bos_before_voice.npy:f46edf4f7007b7ba4ea58831f49d003e59e167b4641c44bb3addfe9231a780b1"
)

PY_ONNX=(uv run --no-project --with onnx --with onnxruntime --with numpy python)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

step() { echo; echo "══ $1"; }
sha256() {
    if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | cut -d' ' -f1
    else sha256sum "$1" | cut -d' ' -f1; fi
}

# ── 1. Fetch the original Kyutai ONNX bundle (hash-verified) ───────────────
step "Original Kyutai ONNX bundle -> models/ (skips files already verified)"
mkdir -p models
for entry in "${BUNDLE_FILES[@]}"; do
    f="${entry%%:*}"; want="${entry##*:}"
    if [ -f "models/$f" ] && [ "$(sha256 "models/$f")" = "$want" ]; then
        echo "  $f — already present, hash OK"
        continue
    fi
    if [ -f "models/$f" ]; then
        echo "  $f — present but does not match the pinned bundle; keeping it as $f.bak"
        mv "models/$f" "models/$f.bak"
    fi
    echo "  $f"
    curl -fL --progress-bar "$BUNDLE_URL/$f" -o "models/$f.tmp"
    got=$(sha256 "models/$f.tmp")
    if [ "$got" != "$want" ]; then
        echo "ERROR: sha256 mismatch for $f"
        echo "  expected $want"
        echo "  got      $got"
        rm -f "models/$f.tmp"
        exit 1
    fi
    mv "models/$f.tmp" "models/$f"
done

# ── 2. AR model chain: delta-KV -> custom attention -> dedup -> merged flow ─
step "AR model: delta-KV rewrite (emit only new cache slices)"
"${PY_ONNX[@]}" tools/make_delta_kv_onnx.py \
    models/flow_lm_main_int8.onnx "$TMP/ar_delta.onnx" --verify

step "AR model: AttentionTail custom op"
"${PY_ONNX[@]}" tools/make_attention_tail_custom_onnx.py \
    "$TMP/ar_delta.onnx" "$TMP/ar_attn.onnx"

step "AR model: cross-layer dedup (RoPE/mask CSE)"
"${PY_ONNX[@]}" tools/make_dedup_positional_onnx.py \
    "$TMP/ar_attn.onnx" "$TMP/ar_attn_dedup.onnx"

step "AR model: merge one-step flow into the main graph"
"${PY_ONNX[@]}" tools/make_merged_flow_onnx.py \
    "$TMP/ar_attn_dedup.onnx" models/flow_lm_flow_int8.onnx \
    models/flow_lm_main_delta_attn_flow_int8.onnx

# ── 3. Decoder chain: delta-KV (portable) + fused ConvTranspose (Apple) ────
step "Decoder: delta-KV rewrite"
"${PY_ONNX[@]}" tools/make_decoder_delta_kv_onnx.py \
    models/mimi_decoder_int8.onnx "$TMP/dec_delta.onnx" --verify
cp "$TMP/dec_delta.onnx" models/mimi_decoder_delta_int8.onnx

step "Decoder: fused ConvTranspose + Accelerate conv + dedup (Apple silicon)"
"${PY_ONNX[@]}" tools/make_decoder_convtr_custom_onnx.py \
    "$TMP/dec_delta.onnx" "$TMP/dec_convtr.onnx"
"${PY_ONNX[@]}" tools/make_decoder_accel_conv_onnx.py \
    "$TMP/dec_convtr.onnx" "$TMP/dec_accel.onnx"
"${PY_ONNX[@]}" tools/make_dedup_positional_onnx.py \
    "$TMP/dec_accel.onnx" models/mimi_decoder_delta_convtr_int8.onnx

step "Native model set complete -> models/"
ls -lh models/*.onnx | awk '{print "  " $5 "\t" $9}'

# ── 4. Web set ──────────────────────────────────────────────────────────────
if [ "${1:-}" = "--web" ]; then
    step "Copying the web model set -> webdemo/models/"
    mkdir -p webdemo/models
    # The TS engine runs on stock onnxruntime-web, so it needs a merged-flow
    # graph without PocketTTS custom ops. The native WASM engine uses the
    # faster AttentionTail variant copied below.
    "${PY_ONNX[@]}" tools/make_merged_flow_onnx.py \
        "$TMP/ar_delta.onnx" models/flow_lm_flow_int8.onnx \
        webdemo/models/flow_lm_main_delta_flow_int8.onnx
    for f in flow_lm_main_delta_attn_flow_int8.onnx mimi_decoder_delta_int8.onnx \
             text_conditioner.onnx mimi_encoder.onnx tokenizer.model \
             bos_before_voice.npy; do
        cp models/$f webdemo/models/
    done
    step "Steering graphs for the default web UI (Varkos vectors ship in Git)"
    "${PY_ONNX[@]}" tools/make_soura_onnx.py --models webdemo/models \
        --vectors webdemo/models/soura_vectors.derived.npy \
        --source 'Varkos reference-derived vectors; see soura_vectors.derived.json'
    echo "  (spm_vocab.json and the default vector set ship with the repo)"
    step "Precompressing and checking the complete web asset set"
    uv run --no-project --with brotli python webdemo/compress.py
    uv run --no-project --with brotli python webdemo/check_assets.py
fi

echo
echo "Done. Try:  ./pocket-tts \"The crypt accepts your voice.\" example.wav out.wav"
