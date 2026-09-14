#!/usr/bin/env python3
"""Check a prepared webdemo (or --root staging directory) before deployment.

Run with: uv run --no-project --with brotli python webdemo/check_assets.py
No files are modified. Brotli is only needed when .br sidecars are present.
"""
import argparse
import hashlib
import json
from pathlib import Path

REQUIRED = [
    'index.html', 'src/app.js', 'src/ui.js', 'src/bus.js',
    'src/tts-worker-native.js', 'src/tts-worker.js', 'src/tts-worker-split.js',
    'src/encode-worker.js', 'src/prepare-worker.js', 'src/playback-worklet.js',
    'src/engine/steering.js', 'src/engine/synth.js', 'src/engine/states.js',
    'src/engine/tokenizer.js', 'src/engine/text.js', 'src/engine/silence.js',
    'vendor/lame.js',
    'presets/varkos.emb', 'presets/alba.emb',
    'models/spm_vocab.json', 'models/bos_before_voice.npy',
    'models/soura_vectors.derived.npy', 'models/soura_vectors.derived.json',
    'models/flow_lm_main_delta_attn_flow_int8.onnx',
    'models/flow_lm_main_delta_flow_int8.onnx',
    'models/flow_lm_main_delta_attn_flow_int8_soura.onnx',
    'models/flow_lm_main_delta_flow_int8_soura.onnx',
    'models/mimi_decoder_delta_int8.onnx', 'models/text_conditioner.onnx',
    'models/mimi_encoder.onnx',
    'vendor/ptt/pocket_tts_wasm.mjs', 'vendor/ptt/pocket_tts_wasm.wasm',
    'vendor/ptt/pocket_tts_wasm_growth.mjs', 'vendor/ptt/pocket_tts_wasm_growth.wasm',
    'vendor/ort/ort.wasm.min.mjs', 'vendor/ort/ort-wasm-simd-threaded.mjs',
    'vendor/ort/ort-wasm-simd-threaded.wasm',
]


def check(root):
    errors = []
    for name in REQUIRED:
        path = root / name
        if not path.is_file() or not path.stat().st_size:
            errors.append(f'Missing or empty required asset: {name}')
    try:
        vectors = (root / 'models/soura_vectors.derived.npy').read_bytes()
        metadata = json.loads((root / 'models/soura_vectors.derived.json').read_text())
        if hashlib.sha256(vectors).hexdigest() != metadata['vectors_sha256']:
            errors.append('Default vector SHA256 does not match its provenance JSON')
    except (OSError, ValueError, KeyError) as error:
        errors.append(f'Cannot verify Default vector provenance: {error}')
    sidecars = [p for folder in ('models', 'presets', 'vendor', 'src')
                for p in (root / folder).rglob('*.br')]
    sidecars += list(root.glob('*.br'))
    if sidecars:
        try:
            import brotli
        except ImportError:
            errors.append('Install brotli to verify compressed assets (see command in this script)')
            return errors
        for path in sidecars:
            try:
                original = path.with_suffix('')
                if brotli.decompress(path.read_bytes()) != original.read_bytes():
                    errors.append(f'Stale compressed asset: {path.relative_to(root)}')
            except (OSError, brotli.error) as error:
                errors.append(f'Invalid compressed asset {path.relative_to(root)}: {error}')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    errors = check(args.root)
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'Web assets PASS: {len(REQUIRED)} required files, Default vector hash, all present Brotli sidecars')


if __name__ == '__main__':
    main()
