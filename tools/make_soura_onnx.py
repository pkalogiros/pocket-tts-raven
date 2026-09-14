#!/usr/bin/env python3
"""Prepare opt-in emotion-steering graphs, leaving all original ONNX files intact.

Accepts an NPZ with six named vectors, or a float32 NPY of shape (6, 1024).
Rows: neutral, angry, disgust, fear, happy, sad. Custom vector magnitudes are
preserved. Neutral always bypasses steering. Requires numpy and onnx.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper

LABELS = ['neutral', 'angry', 'disgust', 'fear', 'happy', 'sad']
TARGET = '/transformer/layers.5/Add_2_output_0'


def patch(model):
    if any(i.name == 'soura_shift' for i in model.graph.input):
        raise ValueError('Graph is already patched')
    matches = [(i, n) for i, n in enumerate(model.graph.node) if TARGET in n.output]
    consumers = [n for n in model.graph.node if TARGET in n.input]
    if (len(matches) != 1 or matches[0][1].op_type != 'Add' or not consumers
            or not all(n.name.startswith('/out_norm/') for n in consumers)):
        raise ValueError('Unsupported export: expected final layer-5 output before out_norm')
    i, node = matches[0]
    node.output[list(node.output).index(TARGET)] = TARGET + '_unsteered'
    model.graph.input.append(helper.make_tensor_value_info('soura_shift', onnx.TensorProto.FLOAT, [1, 1, 1024]))
    model.graph.node.insert(i + 1, helper.make_node('Add', [TARGET + '_unsteered', 'soura_shift'], [TARGET], name='SouraLayer5Shift'))
    onnx.checker.check_model(model)
    return model


def load_vectors(path):
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        with loaded:
            rows = np.stack([loaded[label] for label in LABELS])
    else:
        rows = loaded
    if rows.shape != (6, 1024) or rows.dtype != np.float32 or not np.isfinite(rows).all():
        raise ValueError('Expected finite float32 vectors of shape (6, 1024)')
    rows = rows.copy()
    rows[0] = 0
    return rows


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--vectors', type=Path, required=True)
    parser.add_argument('--source', default='user-supplied', help='Provenance URL or description for the vectors')
    args = parser.parse_args()
    rows = load_vectors(args.vectors)
    sources = sorted(p for p in args.models.glob('flow_lm_main*.onnx') if not p.stem.endswith('_soura'))
    if not sources:
        parser.error('No original flow_lm_main graphs found')
    # Validate all graphs before writing any generated files.
    for path in sources:
        patch(onnx.load(path))
    hashes = {}
    vectors_hash = sha256(args.vectors)
    for path in sources:
        destination = path.with_stem(path.stem + '_soura')
        onnx.save(patch(onnx.load(path)), destination)
        hashes[path.name] = sha256(path)
        print(destination)
    np.save(args.models / 'soura_vectors.npy', rows)
    (args.models / 'soura.json').write_text(json.dumps({
        'source': args.source, 'labels': LABELS, 'vectors_sha256': vectors_hash,
        'base_sha256': hashes, 'injection_layer': 5,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
