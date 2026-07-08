#!/usr/bin/env python3
"""Reference generator for the webdemo engine test.

Runs the exact webdemo model set (merged-flow AR + delta decoder) at
temperature 0 with the preset voice, mirroring the C++/JS orchestration, and
dumps latents + raw decoded audio for the Node engine to match.

    uv run --no-project --with onnx --with onnxruntime --with sentencepiece \
        python webdemo/test/gen_reference.py
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from verify_model_equivalence import StateRunner, make_session  # noqa: E402

import sentencepiece as spm  # noqa: E402

WEB = ROOT / "webdemo"
TEXT = "How are things over in Riverwood these days?"
EOS_EXTRA = 3
DECODE_CHUNKS = [1, 2, 4, 8]  # then 8 forever — must match the JS test


def load_emb(path: Path):
    b = path.read_bytes()
    magic, ndims = struct.unpack("<II", b[:8])
    assert magic == 0x31424D45, "not an EMB1 file"
    shape = struct.unpack(f"<{ndims}q", b[8:8 + ndims * 8])
    data = np.frombuffer(b, np.float32, offset=8 + ndims * 8).reshape(shape)
    return data


def main():
    emb = load_emb(WEB / "presets" / "alba.emb")
    bos = np.load(WEB / "models" / "bos_before_voice.npy")
    assert np.array_equal(emb.reshape(-1)[:bos.size], bos.reshape(-1)), "preset lacks BOS"

    sp = spm.SentencePieceProcessor(model_file=str(WEB / "models" / "tokenizer.model"))
    ids = sp.encode(TEXT)

    ops_lib = ROOT / "nonexistent.dylib"  # plain models, no custom ops needed
    main_r = StateRunner(make_session(WEB / "models" / "flow_lm_main_delta_flow_int8.onnx", ops_lib))
    dec_r = StateRunner(make_session(WEB / "models" / "mimi_decoder_delta_int8.onnx", ops_lib))
    txt_s = make_session(WEB / "models" / "text_conditioner.onnx", ops_lib)

    temb = txt_s.run(None, {txt_s.get_inputs()[0].name: np.array([ids], np.int64)})[0]
    if temb.ndim == 2:
        temb = temb[None]

    empty_seq = np.zeros((1, 0, 32), np.float32)
    empty_txt = np.zeros((1, 0, 1024), np.float32)
    zero_x = np.zeros((1, 32), np.float32)

    main_r.run({"sequence": empty_seq, "text_embeddings": emb.astype(np.float32), "flow_x": zero_x})
    main_r.run({"sequence": empty_seq, "text_embeddings": temb.astype(np.float32), "flow_x": zero_x})

    cl = np.full((1, 1, 32), np.nan, np.float32)
    latents = []
    eos = False
    extra = 0
    for _ in range(200):
        out = main_r.run({"sequence": cl, "text_embeddings": empty_txt, "flow_x": zero_x})
        if not eos and float(out["eos_logit"].reshape(-1)[0]) > -4.0:
            eos = True
        if eos:
            extra += 1
            if extra > EOS_EXTRA:
                break
        lat = out["latent"].reshape(-1).astype(np.float32)
        latents.append(lat)
        cl = lat.reshape(1, 1, 32)

    lat_arr = np.stack(latents)
    print(f"{len(latents)} latent frames")

    audio = []
    i = 0
    ci = 0
    while i < len(latents):
        take = min(DECODE_CHUNKS[min(ci, len(DECODE_CHUNKS) - 1)], len(latents) - i)
        chunk = lat_arr[i:i + take][None]  # [1,T,32]
        out = dec_r.run({"latent": chunk})
        audio.append(out[next(iter(out))].reshape(-1).astype(np.float32))
        i += take
        ci += 1
    full = np.concatenate(audio)
    print(f"{len(full)} audio samples")

    out_dir = WEB / "test"
    lat_arr.tofile(out_dir / "ref_latents.f32")
    full.tofile(out_dir / "ref_audio.f32")
    json.dump({"frames": len(latents), "samples": int(len(full)), "ids": list(ids)},
              open(out_dir / "ref_meta.json", "w"))
    print("reference written")


if __name__ == "__main__":
    main()
