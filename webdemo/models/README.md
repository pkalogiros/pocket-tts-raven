# webdemo/models/

The seven required ONNX models ship as Brotli `.onnx.br` files (about 224 MB
total). Raw `.onnx` files stay ignored. `onnx-manifest.json` records hashes of
both forms. Both engine variants, steering on/off, and voice cloning are included;
a browser downloads only its selected graph.

Unpack after checkout for local tools and tests (no model downloads):

    uv run --no-project --with brotli python webdemo/unpack_models.py

To regenerate from the hash-pinned upstream bundle instead:

    ./tools/prepare_models.sh --web

When intentionally updating the bundled models, regenerate their compressed
files and update both hashes in `onnx-manifest.json` in the same commit.

Required files:

    flow_lm_main_delta_attn_flow_int8_soura.onnx  default native steering AR
    flow_lm_main_delta_flow_int8_soura.onnx       JS steering AR
    flow_lm_main_delta_attn_flow_int8.onnx   AR model (steering off)
    flow_lm_main_delta_flow_int8.onnx        stock-ORT AR model for ?engine=ts
    mimi_decoder_delta_int8.onnx             streaming decoder
    text_conditioner.onnx                    text conditioner
    mimi_encoder.onnx                        voice-clone encoder (fetched lazily)
    tokenizer.model                          (native CLI parity; not fetched by the page)

Already shipped with the repo (small, non-model assets):

    soura_vectors.derived.npy                Default Varkos vectors
    soura_vectors.derived.json               vector provenance and SHA256
    bos_before_voice.npy                     BOS conditioning tensor
    spm_vocab.json                           tokenizer vocab for the JS tokenizer

Then optionally precompress for serving:

    uv run --no-project --with brotli python webdemo/compress.py
