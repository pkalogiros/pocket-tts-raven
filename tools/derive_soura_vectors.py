#!/usr/bin/env python3
"""Experimental reference-derived steering. Requires numpy, onnx, onnxruntime.

First build each voice's .emb/.kv files with --prepare-voice. Pass cached .emb
paths as --neutral and repeated --reference EMOTION=PATH arguments. Unspecified
emotions get zero vectors. The output is a separate NPY, selected explicitly by
--soura-vectors; no runtime defaults or installed vectors are replaced.
"""
import argparse
import json
import struct
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper
from make_soura_onnx import LABELS, TARGET, patch, sha256
from verify_model_equivalence import StateRunner


def load_embedding(path):
    data = path.read_bytes()
    if len(data) < 32:
        raise ValueError(f'Truncated embedding: {path}')
    magic, ndim = struct.unpack('<Ii', data[:8])
    if magic != 0x31424D45 or ndim != 3:
        raise ValueError(f'Unsupported embedding: {path}')
    shape = struct.unpack('<qqq', data[8:32])
    if shape[0] != 1 or shape[1] <= 4 or shape[2] != 1024:
        raise ValueError(f'Expected [1, >4, 1024] embedding: {path}')
    values = np.frombuffer(data[32:], dtype='<f4').reshape(shape).copy()
    if not np.isfinite(values).all():
        raise ValueError(f'Non-finite embedding: {path}')
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True, help='Original or Soura flow_lm_main ONNX graph')
    parser.add_argument('--bos', type=Path, help='bos_before_voice.npy, if present in the runtime model directory')
    parser.add_argument('--ops-lib', type=Path, help='Custom-ops library, required for custom-attention graphs')
    parser.add_argument('--neutral', type=Path, required=True, help='Neutral reference .emb cache')
    parser.add_argument('--reference', action='append', default=[], metavar='EMOTION=EMB')
    parser.add_argument('--alpha', type=float, default=.3, help='Vector norm / neutral activation norm (default: .3)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.alpha) or not 0 < args.alpha <= 1:
        parser.error('--alpha must be in (0, 1]')
    references = {}
    for spec in args.reference:
        label, sep, path = spec.partition('=')
        if not sep or label not in LABELS[1:] or label in references:
            parser.error('References need a unique non-neutral EMOTION=EMB')
        references[label] = Path(path)
    if not references:
        parser.error('At least one --reference is required')
    bos = np.load(args.bos, allow_pickle=False).astype(np.float32) if args.bos else None
    if bos is not None and (bos.shape != (1, 1, 1024) or not np.isfinite(bos).all()):
        parser.error('Expected finite BOS of shape (1, 1, 1024)')
    model = onnx.load(args.model)
    if not any(i.name == 'soura_shift' for i in model.graph.input):
        patch(model)
    layer = TARGET + '_unsteered'
    model.graph.output.append(helper.make_tensor_value_info(layer, onnx.TensorProto.FLOAT, [1, 'n', 1024]))
    onnx.checker.check_model(model)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.log_severity_level = 3
    if args.ops_lib:
        options.register_custom_ops_library(str(args.ops_lib.resolve()))
    with tempfile.TemporaryDirectory(prefix='raven-steering-') as temp:
        capture = Path(temp) / 'capture.onnx'
        onnx.save(model, capture)
        session = ort.InferenceSession(str(capture), options, providers=['CPUExecutionProvider'])

        def activations(path):
            emb = load_embedding(path)
            if bos is not None and not np.allclose(emb[:, :1], bos, atol=1e-4):
                emb = np.concatenate([bos, emb], axis=1)
            runner = StateRunner(session)
            feeds = {'sequence': np.zeros((1, 0, 32), np.float32), 'text_embeddings': emb,
                     'soura_shift': np.zeros((1, 1, 1024), np.float32)}
            if 'flow_x' in runner.non_state:
                feeds['flow_x'] = np.zeros((1, 32), np.float32)
            return np.asarray(runner.run(feeds)[layer])[0, 4:]

        neutral = activations(args.neutral)
        mean = neutral.mean(0)
        norm = float(np.linalg.norm(neutral, axis=1).mean())
        rows = np.zeros((6, 1024), np.float32)
        stats = {}
        for label, path in references.items():
            direction = activations(path).mean(0) - mean
            magnitude = float(np.linalg.norm(direction))
            if not np.isfinite(magnitude) or magnitude < 1e-8:
                raise ValueError(f'{label}: reference has no usable direction relative to neutral')
            rows[LABELS.index(label)] = direction / magnitude * args.alpha * norm
            stats[label] = {'embedding': str(path), 'sha256': sha256(path), 'raw_norm': magnitude}
    if not np.isfinite(rows).all():
        raise ValueError('Non-finite derived vectors')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as output:
        np.save(output, rows)
    args.output.with_suffix('.json').write_text(json.dumps({
        'experimental': True, 'labels': LABELS, 'alpha': args.alpha,
        'neutral_activation_norm': norm, 'neutral_sha256': sha256(args.neutral),
        'model_sha256': sha256(args.model), 'references': stats,
    }, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
