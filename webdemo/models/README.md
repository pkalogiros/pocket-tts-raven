# webdemo/models/

Model set the web demo downloads at runtime (gitignored). The full web model
set is roughly 240MB raw, or about 120MB with `.br` sidecars; the default
native engine's first load fetches only about 65–67MB of that.

Populate with:

    ./tools/prepare_models.sh --web

Required files:

    flow_lm_main_delta_attn_flow_int8.onnx   AR model
    flow_lm_main_delta_flow_int8.onnx        stock-ORT AR model for ?engine=ts
    mimi_decoder_delta_int8.onnx             streaming decoder
    text_conditioner.onnx                    text conditioner
    mimi_encoder.onnx                        voice-clone encoder (fetched lazily)
    tokenizer.model                          (native CLI parity; not fetched by the page)

Already shipped with the repo (small, non-model assets):

    bos_before_voice.npy                     BOS conditioning tensor
    spm_vocab.json                           tokenizer vocab for the JS tokenizer

Then optionally precompress for serving:

    uv run --no-project --with brotli python webdemo/compress.py
