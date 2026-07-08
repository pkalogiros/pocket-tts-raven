#!/usr/bin/env python3
"""
Create a state-only prefill variant of PocketTTS flow_lm_main_delta.

Prefill calls feed an empty latent sequence plus voice/text embeddings and only
need updated state outputs. The normal model also computes conditioning and
eos_logit, but those outputs are discarded by C++ cond_pass(). This rewrite
keeps the same inputs and the same state output ABI, then drops non-state graph
outputs so ONNX Runtime can prune unused output heads.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def _make_opts() -> ort.SessionOptions:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    return opts


def convert_model(input_path: Path, output_path: Path) -> int:
    model = onnx.load(str(input_path))
    state_outputs = [out for out in model.graph.output if out.name.startswith("out_state_")]
    if not state_outputs:
        raise RuntimeError("No out_state_* outputs found")
    del model.graph.output[:]
    model.graph.output.extend(state_outputs)
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    return len(state_outputs)


def _input_shapes(session: ort.InferenceSession) -> dict[str, tuple[int, ...]]:
    result: dict[str, tuple[int, ...]] = {}
    for inp in session.get_inputs():
        result[inp.name] = tuple(int(d) if isinstance(d, int) else 0 for d in inp.shape)
    return result


def _cache_indices(session: ort.InferenceSession) -> list[int]:
    result: list[int] = []
    for inp in session.get_inputs():
        if inp.name.startswith("state_") and inp.type == "tensor(float)" and inp.shape == [2, 1, 1000, 16, 64]:
            result.append(int(inp.name.removeprefix("state_")))
    return sorted(result)


def _zero_state(session: ort.InferenceSession) -> dict[str, np.ndarray]:
    shapes = _input_shapes(session)
    state: dict[str, np.ndarray] = {}
    for inp in session.get_inputs():
        if not inp.name.startswith("state_"):
            continue
        if inp.type == "tensor(float)":
            state[inp.name] = np.zeros(shapes[inp.name], dtype=np.float32)
        elif inp.type == "tensor(int64)":
            state[inp.name] = np.zeros(shapes[inp.name], dtype=np.int64)
        elif inp.type == "tensor(bool)":
            state[inp.name] = np.ones(shapes[inp.name], dtype=np.bool_)
        else:
            raise RuntimeError(f"Unsupported state input {inp.name}: {inp.type}")
    return state


def _write_delta(cache: np.ndarray, delta: np.ndarray, old_step: int, new_step: int) -> np.ndarray:
    out = cache.copy()
    length = new_step - old_step
    if delta.shape[2] != length:
        raise RuntimeError(f"delta length mismatch: {delta.shape[2]} != {length}")
    start = old_step % out.shape[2]
    if start + length <= out.shape[2]:
        out[:, :, start:start + length, :, :] = delta
    else:
        first = out.shape[2] - start
        out[:, :, start:, :, :] = delta[:, :, :first, :, :]
        out[:, :, :length - first, :, :] = delta[:, :, first:, :, :]
    return out


def _reconstruct_state(feed: dict[str, np.ndarray], outputs: dict[str, np.ndarray], cache_indices: list[int]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    cache_set = set(cache_indices)
    for out_name, value in outputs.items():
        if not out_name.startswith("out_state_"):
            continue
        state_name = out_name.removeprefix("out_")
        state_idx = int(state_name.removeprefix("state_"))
        if state_idx in cache_set:
            old_step = int(feed[f"state_{state_idx + 2}"][0])
            new_step = int(outputs[f"out_state_{state_idx + 2}"][0])
            result[state_name] = _write_delta(feed[state_name], value, old_step, new_step)
        else:
            result[state_name] = value.copy()
    return result


def _feed(session: ort.InferenceSession, rng: np.random.Generator, step: int, text_len: int) -> dict[str, np.ndarray]:
    state = _zero_state(session)
    caches = _cache_indices(session)
    for idx in caches:
        if step:
            state[f"state_{idx}"][:, :, :step, :, :] = (
                rng.normal(size=(2, 1, step, 16, 64)).astype(np.float32) * 0.01
            )
        state[f"state_{idx + 2}"][...] = step
    state["sequence"] = np.empty((1, 0, 32), dtype=np.float32)
    state["text_embeddings"] = rng.normal(size=(1, text_len, 1024)).astype(np.float32)
    return state


def verify_equivalent(input_path: Path, output_path: Path) -> None:
    base = ort.InferenceSession(str(input_path), sess_options=_make_opts(), providers=["CPUExecutionProvider"])
    prefill = ort.InferenceSession(str(output_path), sess_options=_make_opts(), providers=["CPUExecutionProvider"])
    base_names = [out.name for out in base.get_outputs()]
    prefill_names = [out.name for out in prefill.get_outputs()]
    expected_names = [name for name in base_names if name.startswith("out_state_")]
    if prefill_names != expected_names:
        raise RuntimeError("State-only output ABI does not match base out_state_* order")

    rng = np.random.default_rng(20260702)
    cache_indices = _cache_indices(base)
    worst = (0.0, "", 0, 0)
    for step, text_len in ((0, 1), (0, 5), (0, 32), (3, 5), (12, 17), (127, 32)):
        feed = _feed(base, rng, step, text_len)
        base_outputs = dict(zip(base_names, base.run(None, {k: v.copy() for k, v in feed.items()})))
        prefill_outputs = dict(zip(prefill_names, prefill.run(None, {k: v.copy() for k, v in feed.items()})))
        base_state = _reconstruct_state(feed, base_outputs, cache_indices)
        prefill_state = _reconstruct_state(feed, prefill_outputs, cache_indices)
        for name, lhs in base_state.items():
            rhs = prefill_state[name]
            if lhs.dtype == np.bool_:
                diff = 0.0 if np.array_equal(lhs, rhs) else 1.0
            elif lhs.size == 0:
                diff = 0.0 if rhs.size == 0 and lhs.shape == rhs.shape else 1.0
            else:
                diff = float(np.max(np.abs(lhs - rhs)))
            if diff > worst[0]:
                worst = (diff, name, step, text_len)
    if worst[0] != 0.0:
        raise RuntimeError(f"State-only prefill mismatch: {worst}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a state-only prefill flow_lm_main ONNX variant")
    parser.add_argument("input", type=Path, help="flow_lm_main_delta ONNX file")
    parser.add_argument("output", type=Path, help="output state-only prefill ONNX file")
    parser.add_argument("--verify", action="store_true", help="run exact ORT state equivalence checks")
    args = parser.parse_args()

    count = convert_model(args.input, args.output)
    print(f"  wrote {args.output} ({count} state outputs)")
    if args.verify:
        verify_equivalent(args.input, args.output)
        print("  verified exact state equivalence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
