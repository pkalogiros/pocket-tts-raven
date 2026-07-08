# Third-party notices

This repository vendors or fetches the following third-party software.
Each component remains under its own license.

## Vendored in this repository

### Alba MacKenna voice sample — `voices/example.wav`, `webdemo/presets/alba.emb`
Kyutai TTS voice sample. CC BY 4.0.
Source: https://huggingface.co/kyutai/tts-voices/tree/main/alba-mackenna
License text: https://creativecommons.org/licenses/by/4.0/

`webdemo/presets/alba.emb` is generated from the same upstream
`alba-mackenna/casual.wav` sample using the native PocketTTS-RAVEN cache
format.

### Pocket TTS model-derived web assets — `webdemo/models/bos_before_voice.npy`, `webdemo/models/spm_vocab.json`
Small runtime assets derived from the Pocket TTS model bundle. CC BY 4.0.
Sources:

- https://huggingface.co/kyutai/pocket-tts
- https://huggingface.co/KevinAHM/pocket-tts-onnx (mirror of the official
  Kyutai ONNX bundle; `tools/prepare_models.sh` verifies every downloaded
  file against a pinned sha256)
- https://huggingface.co/Verylicious/pocket-tts-ungated

The full ONNX model weights are intentionally ignored and are not distributed
with this repository. Upstream model access includes prohibited-use terms;
users remain responsible for complying with the model card and applicable law.

### lame.js — `webdemo/vendor/lame.js`
JavaScript MP3 encoder, a port of LAME. **LGPL.**
Source: https://github.com/zhuker/lamejs (the vendored file is the
unmodified distribution build; its source is available at that repository).
The vendored source carries LGPL-2.0-or-later/LGPL-2.1-or-later notices in
its headers; keep those notices intact when redistributing.
Used as-is via dynamic invocation for MP3 export; no modifications.

### ONNX Runtime (WebAssembly build) — `webdemo/vendor/ort/`
Microsoft ONNX Runtime, MIT License.
Source: https://github.com/microsoft/onnxruntime
License: https://github.com/microsoft/onnxruntime/blob/main/LICENSE

## Fetched at build time (not vendored)

- **ONNX Runtime** (native prebuilt release) — MIT, fetched by CMake with
  pinned sha256 hashes.
- **dr_libs** (dr_wav/dr_mp3/dr_flac) — public domain or MIT-0, at your
  option. https://github.com/mackron/dr_libs — pinned to a commit.
- **SentencePiece** — Apache-2.0. https://github.com/google/sentencepiece
  — pinned to v0.2.1 (native build only; the web build uses a JS tokenizer).

## Fetched at page load (web demo)

- **Press Start 2P** and **VT323** fonts — SIL Open Font License 1.1,
  served from Google Fonts (not vendored; the page degrades to system
  monospace without them).

## Model weights

Full ONNX model weights are **not** distributed with this repository. They are
downloaded from upstream sources by `tools/prepare_models.sh` — see the
Pocket TTS model card for weight licensing and prohibited-use terms. The small
model-derived web assets that are distributed are listed above.
