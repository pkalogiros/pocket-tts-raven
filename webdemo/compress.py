#!/usr/bin/env python3
"""Pre-compress large webdemo assets to .br for serve.py.

    uv run --no-project --with brotli python webdemo/compress.py
"""
import os
import sys

import brotli

HERE = os.path.dirname(os.path.abspath(__file__))
TARGETS = ["models", "presets", "vendor/ptt", "vendor/ort"]
EXT = {".onnx", ".wasm", ".mjs", ".js", ".emb", ".npy", ".json", ".model"}

total_in = total_out = 0
for sub in TARGETS:
    root = os.path.join(HERE, sub)
    if not os.path.isdir(root):
        continue
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        stem, ext = os.path.splitext(name)
        if ext not in EXT or not os.path.isfile(path):
            continue
        out = path + ".br"
        if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(path):
            total_in += os.path.getsize(path); total_out += os.path.getsize(out)
            continue
        data = open(path, "rb").read()
        q = 11  # max quality: one-time cost, every visitor benefits
        comp = brotli.compress(data, quality=q)
        open(out, "wb").write(comp)
        total_in += len(data); total_out += len(comp)
        print(f"{sub}/{name}: {len(data)/1e6:.1f} -> {len(comp)/1e6:.1f} MB (q{q})")
print(f"total: {total_in/1e6:.0f} -> {total_out/1e6:.0f} MB")
