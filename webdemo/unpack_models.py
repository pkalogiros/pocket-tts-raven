#!/usr/bin/env python3
"""Restore the bundled browser ONNX files, verifying compressed and raw hashes.

    uv run --no-project --with brotli python webdemo/unpack_models.py
"""
import argparse
import hashlib
import json
from pathlib import Path
import brotli


def unpack(root):
    manifest = json.loads((root / 'onnx-manifest.json').read_text())
    for name, hashes in manifest.items():
        if Path(name).name != name or not name.endswith('.onnx'):
            raise ValueError(f'Invalid model name in manifest: {name}')
        path = root / name
        compressed = path.with_suffix('.onnx.br').read_bytes()
        if hashlib.sha256(compressed).hexdigest() != hashes['brotli_sha256']:
            raise ValueError(f'Compressed model hash mismatch: {name}.br')
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == hashes['sha256']:
            print(f'{name}: already unpacked, hash verified')
            continue
        data = brotli.decompress(compressed)
        if hashlib.sha256(data).hexdigest() != hashes['sha256']:
            raise ValueError(f'Uncompressed model hash mismatch: {name}')
        temporary = path.with_suffix('.onnx.tmp')
        try:
            temporary.write_bytes(data)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        print(f'{name}: unpacked and verified')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, default=Path(__file__).resolve().parent / 'models')
    args = parser.parse_args()
    unpack(args.models)
