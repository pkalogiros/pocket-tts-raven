# models/

ONNX models for the native runtime live here (gitignored). A clean
`prepare_models.sh` run installs roughly 285MB of runtime models; local
development directories may be larger if you keep ignored experiment or
rollback ONNX files.

Populate with:

    ./tools/prepare_models.sh

This downloads the original Kyutai Pocket TTS ONNX bundle from Hugging
Face (english_2026-04, ~165 MB — every file verified against a sha256
pinned in the script) and then applies this repo's graph rewrites
locally. The rewrites are deterministic: the optimized models it
produces are byte-for-byte reproducible. The delta-KV rewrites run
built-in ORT equivalence checks; use `tools/verify_model_equivalence.py`
for lockstep checks when changing or staging other rewrites.

Required files:

    tokenizer.model                          sentencepiece tokenizer
    text_conditioner.onnx                    text -> conditioning embeddings
    mimi_encoder.onnx                        voice cloning encoder
    flow_lm_main_delta_attn_flow_int8.onnx   AR model (delta-KV + custom attention + merged flow)
    flow_lm_flow_int8.onnx                   standalone flow (only needed for --lsd-steps > 1)
    mimi_decoder_delta_int8.onnx             streaming decoder (delta-KV)
    mimi_decoder_delta_convtr_int8.onnx      fused-ConvTranspose decoder (Apple silicon)

## Models are a matched set

Never mix models from different sources or export runs: a conditioner
from one set paired with an AR model from another produces subtly wrong
embeddings (symptom: runaway generation — EOS never fires). The default
pipeline avoids this by construction (the bundle is hash-pinned), but if
you swap any model in, replace the whole set together.

## Rebuilding the base models from weights (optional)

`export_onnx.py` re-exports the base ONNX models from the original
safetensors weights and validates them against the PyTorch reference
(developer path — needs torch, a few GB of downloads). The fresh exports
are functionally equivalent, but the optimization rewriters in `tools/`
pattern-match the official bundle's graph structure, so
`prepare_models.sh` always builds the optimized set from the bundle.
