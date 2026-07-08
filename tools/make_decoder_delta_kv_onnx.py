#!/usr/bin/env python3
"""
Create delta-KV variants of PocketTTS Mimi decoder ONNX models.

The english_2026-04 decoder keeps two active packed float32 caches:

    state_N:     float32 [2, 1, heads, capacity, dim]
    state_N+1:   int64 write position
    out_state_N: full updated cache

The original graph updates K/V slices, repacks the full cache with ScatterND,
then immediately gathers K/V back out for attention. This rewrite keeps the
same attention math, but changes out_state_N to contain only the newly computed
packed [K,V] delta slice. C++ writes that delta into its preallocated full cache.
"""

from __future__ import annotations

import argparse
import copy
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort


def _producer_map(model: onnx.ModelProto) -> dict[str, int]:
    return {output: i for i, node in enumerate(model.graph.node) for output in node.output}


def _consumer_map(model: onnx.ModelProto) -> dict[str, list[int]]:
    consumers: dict[str, list[int]] = defaultdict(list)
    for i, node in enumerate(model.graph.node):
        for input_name in node.input:
            consumers[input_name].append(i)
    return consumers


def _input_shape(model: onnx.ModelProto, name: str) -> tuple[int | str, ...] | None:
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
        return tuple(dims)
    return None


def _elem_type(model: onnx.ModelProto, name: str) -> int | None:
    for value in model.graph.input:
        if value.name == name:
            return value.type.tensor_type.elem_type
    return None


def _numeric_suffix(name: str, prefix: str) -> int | None:
    if not name.startswith(prefix):
        return None
    try:
        return int(name[len(prefix):])
    except ValueError:
        return None


def _constant_scalar_int(model: onnx.ModelProto, name: str) -> int | None:
    producers = _producer_map(model)
    node_idx = producers.get(name)
    if node_idx is None:
        return None
    node = model.graph.node[node_idx]
    if node.op_type != "Constant":
        return None
    for attr in node.attribute:
        if attr.name == "value_int":
            return int(attr.i)
        if attr.name == "value":
            arr = numpy_helper.to_array(attr.t)
            if arr.size == 1:
                return int(arr.reshape(-1)[0])
    return None


def _const_int64(name: str, values: list[int]) -> onnx.NodeProto:
    return helper.make_node(
        "Constant",
        [],
        [name],
        name=name + "_const",
        value=helper.make_tensor(name + "_value", TensorProto.INT64, [len(values)], values),
    )


def _decoder_cache_state_indices(model: onnx.ModelProto) -> list[int]:
    result: list[int] = []
    input_names = {value.name for value in model.graph.input}
    for name in input_names:
        idx = _numeric_suffix(name, "state_")
        if idx is None:
            continue
        step_name = f"state_{idx + 1}"
        if step_name not in input_names:
            continue
        if _elem_type(model, name) != TensorProto.FLOAT:
            continue
        if _elem_type(model, step_name) != TensorProto.INT64:
            continue
        shape = _input_shape(model, name)
        if (
            shape
            and len(shape) == 5
            and shape[0] == 2
            and shape[1] == 1
            and isinstance(shape[2], int)
            and shape[3] == 1000
            and isinstance(shape[4], int)
        ):
            result.append(idx)
    return sorted(result)


def _find_gather_by_scalar(model: onnx.ModelProto, consumers: dict[str, list[int]], nodes, tensor: str, scalar: int) -> int | None:
    for idx in consumers.get(tensor, []):
        node = nodes[idx]
        if node.op_type != "Gather" or len(node.input) < 2 or node.input[0] != tensor:
            continue
        axis = 0
        for attr in node.attribute:
            if attr.name == "axis":
                axis = int(attr.i)
        if axis == 0 and _constant_scalar_int(model, node.input[1]) == scalar:
            return idx
    return None


def _only_scatter_elements(consumers: dict[str, list[int]], nodes, tensor: str) -> int | None:
    matches = [idx for idx in consumers.get(tensor, []) if nodes[idx].op_type == "ScatterElements"]
    return matches[0] if len(matches) == 1 else None


def _replace_inputs(nodes, replacements: dict[str, str]) -> None:
    for node in nodes:
        for i, input_name in enumerate(node.input):
            if input_name in replacements:
                node.input[i] = replacements[input_name]


def _prune_dead_nodes(model: onnx.ModelProto) -> int:
    nodes = list(model.graph.node)
    producers = _producer_map(model)
    required_values = {output.name for output in model.graph.output}
    keep: set[int] = set()
    queue = deque(required_values)

    while queue:
        name = queue.popleft()
        idx = producers.get(name)
        if idx is None or idx in keep:
            continue
        keep.add(idx)
        for input_name in nodes[idx].input:
            if input_name:
                queue.append(input_name)

    rewritten = [node for i, node in enumerate(nodes) if i in keep]
    removed = len(nodes) - len(rewritten)
    if removed:
        del model.graph.node[:]
        model.graph.node.extend(rewritten)

    used = {input_name for node in model.graph.node for input_name in node.input if input_name}
    kept_initializers = [init for init in model.graph.initializer if init.name in used]
    if len(kept_initializers) != len(model.graph.initializer):
        del model.graph.initializer[:]
        model.graph.initializer.extend(kept_initializers)
    return removed


def convert_model(input_path: Path, output_path: Path) -> int:
    model = onnx.load(str(input_path))
    graph = model.graph
    nodes = list(graph.node)
    producers = _producer_map(model)
    consumers = _consumer_map(model)

    new_nodes_by_insert: dict[int, list[onnx.NodeProto]] = defaultdict(list)
    replacements: dict[str, str] = {}
    remove_indices: set[int] = set()
    converted = 0

    for state_idx in _decoder_cache_state_indices(model):
        state_name = f"state_{state_idx}"
        out_name = f"out_state_{state_idx}"
        scatter_1_idx = producers.get(out_name)
        if scatter_1_idx is None:
            continue
        scatter_1 = nodes[scatter_1_idx]
        if scatter_1.op_type != "ScatterND" or len(scatter_1.input) < 3:
            continue

        scatter_0_out = scatter_1.input[0]
        scatter_0_idx = producers.get(scatter_0_out)
        if scatter_0_idx is None:
            continue
        scatter_0 = nodes[scatter_0_idx]
        if scatter_0.op_type != "ScatterND" or scatter_0.input[0] != state_name:
            continue

        old_k_gather_idx = _find_gather_by_scalar(model, consumers, nodes, state_name, 0)
        old_v_gather_idx = _find_gather_by_scalar(model, consumers, nodes, scatter_0_out, 1)
        valid_k_gather_idx = _find_gather_by_scalar(model, consumers, nodes, out_name, 0)
        valid_v_gather_idx = _find_gather_by_scalar(model, consumers, nodes, out_name, 1)
        if None in (old_k_gather_idx, old_v_gather_idx, valid_k_gather_idx, valid_v_gather_idx):
            continue

        k_scatter_idx = _only_scatter_elements(consumers, nodes, nodes[old_k_gather_idx].output[0])
        v_scatter_idx = _only_scatter_elements(consumers, nodes, nodes[old_v_gather_idx].output[0])
        if k_scatter_idx is None or v_scatter_idx is None:
            continue
        k_scatter = nodes[k_scatter_idx]
        v_scatter = nodes[v_scatter_idx]
        if len(k_scatter.input) < 3 or len(v_scatter.input) < 3:
            continue

        k_update = k_scatter.input[2]
        v_update = v_scatter.input[2]
        k_full = k_scatter.output[0]
        v_full = v_scatter.output[0]

        prefix = f"/decoder_delta_kv/state_{state_idx}"
        unsqueeze_axes = f"{prefix}/unsqueeze_axes"
        k_delta_5d = f"{prefix}/k_delta_5d"
        v_delta_5d = f"{prefix}/v_delta_5d"

        new_nodes_by_insert[scatter_1_idx].extend(
            [
                _const_int64(unsqueeze_axes, [0]),
                helper.make_node("Unsqueeze", [k_update, unsqueeze_axes], [k_delta_5d], name=f"{prefix}/UnsqueezeK"),
                helper.make_node("Unsqueeze", [v_update, unsqueeze_axes], [v_delta_5d], name=f"{prefix}/UnsqueezeV"),
                helper.make_node("Concat", [k_delta_5d, v_delta_5d], [out_name], name=f"{prefix}/ConcatDeltaKV", axis=0),
            ]
        )

        # K ScatterND only changed packed axis 0. V can read directly from the
        # original packed cache; attention can read from the already-updated
        # per-K/per-V ScatterElements outputs.
        replacements[scatter_0_out] = state_name
        replacements[nodes[valid_k_gather_idx].output[0]] = k_full
        replacements[nodes[valid_v_gather_idx].output[0]] = v_full
        remove_indices.update({scatter_0_idx, scatter_1_idx, valid_k_gather_idx, valid_v_gather_idx})

        for output in graph.output:
            if output.name == out_name:
                output.CopyFrom(
                    helper.make_tensor_value_info(out_name, TensorProto.FLOAT, [2, 1, 8, "delta_len", 64])
                )
                break
        converted += 1

    if converted == 0:
        return 0

    _replace_inputs(nodes, replacements)
    rewritten_nodes: list[onnx.NodeProto] = []
    for i, node in enumerate(nodes):
        rewritten_nodes.extend(new_nodes_by_insert.get(i, []))
        if i not in remove_indices:
            rewritten_nodes.append(node)

    del graph.node[:]
    graph.node.extend(rewritten_nodes)
    _prune_dead_nodes(model)
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    return converted


def _make_feed(session: ort.InferenceSession, seq_len: int, step: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    feed: dict[str, np.ndarray] = {}
    for inp in session.get_inputs():
        shape = []
        for dim in inp.shape:
            if isinstance(dim, str):
                shape.append(seq_len if dim == "seq_len" else 0)
            elif dim is None:
                shape.append(0)
            else:
                shape.append(int(dim))

        if inp.name == "latent":
            feed[inp.name] = rng.normal(size=(1, seq_len, 32)).astype(np.float32)
        elif inp.type == "tensor(float)":
            arr = np.zeros(shape, dtype=np.float32)
            if inp.name in ("state_19", "state_22"):
                arr = rng.normal(size=shape).astype(np.float32) * 0.01
            feed[inp.name] = arr
        elif inp.type == "tensor(int64)":
            arr = np.zeros(shape, dtype=np.int64)
            if inp.name in ("state_20", "state_21", "state_23", "state_24"):
                arr[...] = step
            feed[inp.name] = arr
        elif inp.type == "tensor(bool)":
            feed[inp.name] = np.ones(shape, dtype=np.bool_)
        else:
            raise RuntimeError(f"Unsupported input {inp.name}: {inp.type}")
    return feed


def _write_delta(cache: np.ndarray, delta: np.ndarray, old_step: int, new_step: int) -> np.ndarray:
    out = cache.copy()
    length = new_step - old_step
    if delta.shape[3] != length:
        raise RuntimeError(f"delta length mismatch: {delta.shape[3]} != {length}")
    capacity = out.shape[3]
    start = old_step % capacity
    if start + length <= capacity:
        out[:, :, :, start:start + length, :] = delta
    else:
        first = capacity - start
        out[:, :, :, start:, :] = delta[:, :, :, :first, :]
        out[:, :, :, :length - first, :] = delta[:, :, :, first:, :]
    return out


def verify_equivalent(input_path: Path, output_path: Path) -> None:
    base = ort.InferenceSession(str(input_path), providers=["CPUExecutionProvider"])
    delta = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    base_names = [out.name for out in base.get_outputs()]
    delta_names = [out.name for out in delta.get_outputs()]
    rng = np.random.default_rng(1234)

    for seq_len, step in ((1, 0), (4, 0), (1, 17), (4, 940), (4, 990)):
        feed = _make_feed(base, seq_len, step, rng)
        base_outputs = dict(zip(base_names, base.run(None, feed)))
        delta_outputs = dict(zip(delta_names, delta.run(None, copy.deepcopy(feed))))

        for name in base_names:
            if name in ("out_state_19", "out_state_22"):
                step_state = "out_state_20" if name == "out_state_19" else "out_state_23"
                new_step = int(base_outputs[step_state][0])
                reconstructed = _write_delta(feed[name.replace("out_", "")], delta_outputs[name], step, new_step)
                max_diff = float(np.max(np.abs(base_outputs[name] - reconstructed)))
            elif base_outputs[name].dtype == np.bool_:
                max_diff = 0.0 if np.array_equal(base_outputs[name], delta_outputs[name]) else 1.0
            else:
                max_diff = float(np.max(np.abs(base_outputs[name] - delta_outputs[name]))) if base_outputs[name].size else 0.0
            if max_diff != 0.0:
                raise RuntimeError(f"{name} mismatch for seq_len={seq_len}, step={step}: {max_diff}")


def convert_path(path: Path, output: Path | None, verify: bool) -> int:
    if path.is_dir():
        total = 0
        candidates = [
            ("mimi_decoder.onnx", "mimi_decoder_delta.onnx"),
            ("mimi_decoder_int8.onnx", "mimi_decoder_delta_int8.onnx"),
        ]
        for src_name, dst_name in candidates:
            src = path / src_name
            if src.exists():
                total += convert_path(src, path / dst_name, verify)
        return total

    output_path = output
    if output_path is None:
        stem = path.stem
        if stem.endswith("_int8"):
            output_path = path.with_name(stem[:-5] + "_delta_int8.onnx")
        else:
            output_path = path.with_name(stem + "_delta.onnx")

    converted = convert_model(path, output_path)
    if converted == 0:
        print(f"  skipped {path}: no decoder KV ScatterND pattern found")
        return 0

    print(f"  wrote {output_path} ({converted} cache states)")
    if verify:
        verify_equivalent(path, output_path)
        print("  verified exact equivalence")
    return converted


def main() -> int:
    parser = argparse.ArgumentParser(description="Create delta-KV PocketTTS Mimi decoder ONNX variants")
    parser.add_argument("input", type=Path, help="mimi_decoder ONNX file or bundle/model directory")
    parser.add_argument("output", type=Path, nargs="?", help="Output ONNX file when input is a file")
    parser.add_argument("--verify", action="store_true", help="Run exact ORT equivalence checks")
    args = parser.parse_args()

    total = convert_path(args.input, args.output, args.verify)
    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())
