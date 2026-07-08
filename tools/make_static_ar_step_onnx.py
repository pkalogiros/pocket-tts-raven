#!/usr/bin/env python3
"""
Create a static-shape AR-step variant of PocketTTS flow_lm_main.

This does not change weights or graph math. It constrains the hot autoregressive
step shape where exact ORT equivalence holds:

    sequence:        [1, 1, 32]
    text_embeddings: dynamic, but runtime passes [1, 0, 1024]
    out_state_KV:    [2, 1, 1, 16, 64]

The dynamic flow_lm_main_delta model is still used for voice/text prefill.
The C++ runtime can then run this static session for normal one-frame AR steps
while reusing the same state buffers.

Do not freeze text_embeddings to [1, 0, 1024] here. In combination with a
static sequence input, ORT changes the optimized graph enough to produce
different outputs on valid hot-path states.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto
import onnxruntime as ort


def _set_shape(value: onnx.ValueInfoProto, dims: list[int | str]) -> None:
    tensor_type = value.type.tensor_type
    del tensor_type.shape.dim[:]
    for dim in dims:
        d = tensor_type.shape.dim.add()
        if isinstance(dim, int):
            d.dim_value = dim
        else:
            d.dim_param = dim


def _elem_type(model: onnx.ModelProto, name: str) -> int | None:
    for value in model.graph.input:
        if value.name == name:
            return value.type.tensor_type.elem_type
    return None


def _input_shape(model: onnx.ModelProto, name: str) -> list[int | str] | None:
    for value in model.graph.input:
        if value.name != name:
            continue
        dims: list[int | str] = []
        for dim in value.type.tensor_type.shape.dim:
            if dim.dim_value:
                dims.append(dim.dim_value)
            elif dim.dim_param:
                dims.append(dim.dim_param)
            else:
                dims.append("?")
        return dims
    return None


def _numeric_suffix(name: str, prefix: str) -> int | None:
    if not name.startswith(prefix):
        return None
    try:
        return int(name[len(prefix):])
    except ValueError:
        return None


def _packed_cache_indices(model: onnx.ModelProto) -> list[int]:
    result: list[int] = []
    inputs = {value.name for value in model.graph.input}
    for name in inputs:
        idx = _numeric_suffix(name, "state_")
        if idx is None:
            continue
        if f"state_{idx + 2}" not in inputs:
            continue
        shape = _input_shape(model, name)
        if (
            _elem_type(model, name) == TensorProto.FLOAT
            and shape == [2, 1, 1000, 16, 64]
            and _elem_type(model, f"state_{idx + 2}") == TensorProto.INT64
        ):
            result.append(idx)
    return sorted(result)


def convert_model(input_path: Path, output_path: Path) -> list[int]:
    model = onnx.load(str(input_path))
    cache_indices = _packed_cache_indices(model)
    if not cache_indices:
        raise RuntimeError("No packed delta-KV cache inputs found")

    for value in model.graph.input:
        if value.name == "sequence":
            _set_shape(value, [1, 1, 32])

    for value in model.graph.output:
        if value.name.startswith("out_state_"):
            idx = _numeric_suffix(value.name, "out_state_")
            if idx in cache_indices:
                _set_shape(value, [2, 1, 1, 16, 64])

    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    return cache_indices


def _array_for_input(inp, rng: np.random.Generator, step: int) -> np.ndarray:
    shape = []
    for dim in inp.shape:
        if isinstance(dim, str) or dim is None:
            shape.append(0)
        else:
            shape.append(int(dim))

    if inp.name == "sequence":
        return rng.normal(size=(1, 1, 32)).astype(np.float32)
    if inp.name == "text_embeddings":
        return np.empty((1, 0, 1024), dtype=np.float32)
    if inp.type == "tensor(float)":
        if shape == [0]:
            return np.empty((0,), dtype=np.float32)
        arr = np.zeros(shape, dtype=np.float32)
        if inp.name.startswith("state_") and len(shape) == 5:
            arr[:, :, :step, :, :] = rng.normal(size=(2, 1, step, 16, 64)).astype(np.float32) * 0.01
        return arr
    if inp.type == "tensor(int64)":
        arr = np.zeros(shape, dtype=np.int64)
        if inp.name.startswith("state_"):
            arr[...] = step
        return arr
    if inp.type == "tensor(bool)":
        return np.ones(shape, dtype=np.bool_)
    raise RuntimeError(f"Unsupported input {inp.name}: {inp.type}")


def _make_feed(session: ort.InferenceSession, rng: np.random.Generator, step: int) -> dict[str, np.ndarray]:
    return {inp.name: _array_for_input(inp, rng, step) for inp in session.get_inputs()}


def verify_equivalent(input_path: Path, output_path: Path) -> None:
    def make_opts() -> ort.SessionOptions:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return opts

    base = ort.InferenceSession(str(input_path), sess_options=make_opts(), providers=["CPUExecutionProvider"])
    static = ort.InferenceSession(str(output_path), sess_options=make_opts(), providers=["CPUExecutionProvider"])
    base_names = [out.name for out in base.get_outputs()]
    static_names = [out.name for out in static.get_outputs()]
    if base_names != static_names:
        raise RuntimeError("Output names changed")

    rng = np.random.default_rng(20260702)
    for step in (0, 1, 12, 127, 999):
        feed = _make_feed(base, rng, step)
        base_outputs = dict(zip(base_names, base.run(None, feed)))
        static_outputs = dict(zip(static_names, static.run(None, {k: v.copy() for k, v in feed.items()})))
        for name in base_names:
            lhs = base_outputs[name]
            rhs = static_outputs[name]
            if lhs.dtype == np.bool_:
                ok = np.array_equal(lhs, rhs)
                diff = 0.0 if ok else 1.0
            elif lhs.size == 0:
                diff = 0.0 if rhs.size == 0 and lhs.shape == rhs.shape else 1.0
            else:
                diff = float(np.max(np.abs(lhs - rhs)))
            if diff != 0.0:
                raise RuntimeError(f"{name} mismatch for step={step}: {diff}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a static AR-step flow_lm_main ONNX variant")
    parser.add_argument("input", type=Path, help="flow_lm_main_delta ONNX file")
    parser.add_argument("output", type=Path, help="output static AR-step ONNX file")
    parser.add_argument("--verify", action="store_true", help="run exact ORT equivalence checks")
    args = parser.parse_args()

    cache_indices = convert_model(args.input, args.output)
    print(f"  wrote {args.output} ({len(cache_indices)} cache states)")
    if args.verify:
        verify_equivalent(args.input, args.output)
        print("  verified exact equivalence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
