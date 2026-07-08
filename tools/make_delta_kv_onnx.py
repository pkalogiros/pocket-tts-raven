#!/usr/bin/env python3
"""
Create delta-KV variants of PocketTTS flow_lm_main ONNX models.

This is a post-export/post-quantization graph rewrite for the packed cache
layout used by english_2026-04:

    state_N:     float32 [2, 1, capacity, heads, dim]
    state_N+1:   current_end, usually empty
    state_N+2:   int64 step [1]
    out_state_N: full updated cache

The rewrite removes the two ScatterND cache-update nodes per layer, outputs
only the newly computed packed K/V slice, and rewires attention to use:

    valid_cache = concat(old_cache[:, :, :step], new_kv_delta, axis=2)

C++ then writes the delta slice into its preallocated full cache. This keeps
model math equivalent while avoiding full-cache ScatterND work in ORT.
"""

from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort


def _input_shape(model: onnx.ModelProto, name: str) -> tuple[int | str, ...] | None:
    for value in model.graph.input:
        if value.name != name:
            continue
        tensor_type = value.type.tensor_type
        dims: list[int | str] = []
        for dim in tensor_type.shape.dim:
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


def _producer_map(model: onnx.ModelProto) -> dict[str, int]:
    return {output: i for i, node in enumerate(model.graph.node) for output in node.output}


def _consumer_map(model: onnx.ModelProto) -> dict[str, list[int]]:
    consumers: dict[str, list[int]] = defaultdict(list)
    for i, node in enumerate(model.graph.node):
        for input_name in node.input:
            consumers[input_name].append(i)
    return consumers


def _const_int64(name: str, values: list[int]) -> onnx.NodeProto:
    return helper.make_node(
        "Constant",
        [],
        [name],
        name=name + "_const",
        value=helper.make_tensor(name + "_value", TensorProto.INT64, [len(values)], values),
    )


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


def _packed_cache_state_indices(model: onnx.ModelProto) -> list[int]:
    result: list[int] = []
    input_names = {value.name for value in model.graph.input}
    for name in input_names:
        idx = _numeric_suffix(name, "state_")
        if idx is None:
            continue
        step_name = f"state_{idx + 2}"
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
            and shape[3] != "?"
            and shape[4] != "?"
        ):
            result.append(idx)
    return sorted(result)


def _prune_unused_initializers(model: onnx.ModelProto) -> int:
    used = {input_name for node in model.graph.node for input_name in node.input if input_name}
    kept = [init for init in model.graph.initializer if init.name in used]
    removed = len(model.graph.initializer) - len(kept)
    if removed:
        del model.graph.initializer[:]
        model.graph.initializer.extend(kept)
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

    for state_idx in _packed_cache_state_indices(model):
        state_name = f"state_{state_idx}"
        step_name = f"state_{state_idx + 2}"
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
        if (
            scatter_0.op_type != "ScatterND"
            or len(scatter_0.input) < 3
            or scatter_0.input[0] != state_name
        ):
            continue

        slice_idx = None
        for consumer_idx in consumers.get(out_name, []):
            if nodes[consumer_idx].op_type == "Slice":
                slice_idx = consumer_idx
                break
        if slice_idx is None:
            continue
        slice_node = nodes[slice_idx]
        if len(slice_node.input) < 5:
            continue

        # The first ScatterND writes K into cache; the second writes V into the
        # K-updated cache. Their update tensors are the new packed K/V slices.
        k_update = scatter_0.input[2]
        v_update = scatter_1.input[2]
        old_slice = f"/delta_kv/state_{state_idx}/old_cache_slice"
        valid_cache = f"/delta_kv/state_{state_idx}/valid_cache"

        new_nodes_by_insert[slice_idx].extend(
            [
                helper.make_node(
                    "Slice",
                    [state_name, slice_node.input[1], step_name, slice_node.input[3], slice_node.input[4]],
                    [old_slice],
                    name=f"/delta_kv/state_{state_idx}/SliceOldCache",
                ),
                helper.make_node(
                    "Concat",
                    [k_update, v_update],
                    [out_name],
                    name=f"/delta_kv/state_{state_idx}/ConcatNewKV",
                    axis=0,
                ),
                helper.make_node(
                    "Concat",
                    [old_slice, out_name],
                    [valid_cache],
                    name=f"/delta_kv/state_{state_idx}/ConcatValidCache",
                    axis=2,
                ),
            ]
        )

        replacements[slice_node.output[0]] = valid_cache
        # Nodes that shape the V update only need the original cache shape/slice,
        # not the K-updated cache tensor.
        replacements[scatter_0_out] = state_name
        remove_indices.update({scatter_0_idx, scatter_1_idx, slice_idx})

        for output in graph.output:
            if output.name == out_name:
                output.CopyFrom(
                    helper.make_tensor_value_info(out_name, TensorProto.FLOAT, [2, 1, "delta_len", 16, 64])
                )
                break
        converted += 1

    if converted == 0:
        return 0

    for node in nodes:
        for i, input_name in enumerate(node.input):
            if input_name in replacements:
                node.input[i] = replacements[input_name]

    rewritten_nodes: list[onnx.NodeProto] = []
    for i, node in enumerate(nodes):
        rewritten_nodes.extend(new_nodes_by_insert.get(i, []))
        if i not in remove_indices:
            rewritten_nodes.append(node)

    del graph.node[:]
    graph.node.extend(rewritten_nodes)
    optimize_cache_update_shapes(model)
    optimize_delta_valid_cache(model)
    _prune_unused_initializers(model)
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    return converted


def optimize_delta_valid_cache(model: onnx.ModelProto) -> int:
    """Avoid building packed valid_cache and gathering K/V back out.

    The delta graph keeps the public cache ABI as packed [K,V] so the C++
    runner can write one delta output per layer. Internally, attention only
    needs split K and V tensors. This rewrite changes:

        packed_old = Slice(state)
        packed_new = Concat(new_k, new_v) -> graph output
        packed_valid = Concat(packed_old, packed_new)
        k_valid = Gather(packed_valid, 0)
        v_valid = Gather(packed_valid, 1)

    into:

        old_k = Squeeze(Slice(state, k axis + old seq range))
        old_v = Squeeze(Slice(state, v axis + old seq range))
        k_valid = Concat(old_k, Squeeze(new_k), axis=1)
        v_valid = Concat(old_v, Squeeze(new_v), axis=1)

    The graph output remains the packed new [K,V] delta, so no C++ ABI changes
    are required.
    """

    graph = model.graph
    nodes = list(graph.node)
    producers = _producer_map(model)
    consumers = _consumer_map(model)

    new_nodes_by_insert: dict[int, list[onnx.NodeProto]] = defaultdict(list)
    replacements: dict[str, str] = {}
    remove_indices: set[int] = set()
    converted = 0

    for state_idx in _packed_cache_state_indices(model):
        state_name = f"state_{state_idx}"
        step_name = f"state_{state_idx + 2}"
        out_name = f"out_state_{state_idx}"
        old_cache_name = f"/delta_kv/state_{state_idx}/old_cache_slice"
        valid_cache_name = f"/delta_kv/state_{state_idx}/valid_cache"

        old_cache_idx = producers.get(old_cache_name)
        valid_cache_idx = producers.get(valid_cache_name)
        new_delta_idx = producers.get(out_name)
        if old_cache_idx is None or valid_cache_idx is None or new_delta_idx is None:
            continue

        old_cache_node = nodes[old_cache_idx]
        valid_cache_node = nodes[valid_cache_idx]
        new_delta_node = nodes[new_delta_idx]
        if (
            old_cache_node.op_type != "Slice"
            or valid_cache_node.op_type != "Concat"
            or new_delta_node.op_type != "Concat"
            or len(new_delta_node.input) != 2
            or old_cache_node.input[0] != state_name
            or valid_cache_node.input[0] != old_cache_name
            or valid_cache_node.input[1] != out_name
        ):
            continue

        valid_gathers: dict[int, int] = {}
        for consumer_idx in consumers.get(valid_cache_name, []):
            node = nodes[consumer_idx]
            if node.op_type != "Gather" or len(node.input) < 2 or node.input[0] != valid_cache_name:
                continue
            axis = 0
            for attr in node.attribute:
                if attr.name == "axis":
                    axis = int(attr.i)
            if axis != 0:
                continue
            scalar = _constant_scalar_int(model, node.input[1])
            if scalar in (0, 1):
                valid_gathers[scalar] = consumer_idx

        if 0 not in valid_gathers or 1 not in valid_gathers:
            continue

        k_update = new_delta_node.input[0]
        v_update = new_delta_node.input[1]
        prefix = f"/delta_kv_fast/state_{state_idx}"
        k_starts = f"{prefix}/old_k_starts"
        k_end_head = f"{prefix}/old_k_end_head"
        k_ends = f"{prefix}/old_k_ends"
        v_starts = f"{prefix}/old_v_starts"
        v_end_head = f"{prefix}/old_v_end_head"
        v_ends = f"{prefix}/old_v_ends"
        axes = f"{prefix}/old_slice_axes"
        steps = f"{prefix}/old_slice_steps"
        squeeze_axis = f"{prefix}/squeeze_axis"
        old_k_5d = f"{prefix}/old_k_5d"
        old_v_5d = f"{prefix}/old_v_5d"
        old_k = f"{prefix}/old_k"
        old_v = f"{prefix}/old_v"
        new_k = f"{prefix}/new_k"
        new_v = f"{prefix}/new_v"
        valid_k = f"{prefix}/valid_k"
        valid_v = f"{prefix}/valid_v"

        new_nodes_by_insert[old_cache_idx].extend(
            [
                _const_int64(k_starts, [0, 0]),
                _const_int64(k_end_head, [1]),
                helper.make_node("Concat", [k_end_head, step_name], [k_ends], name=f"{prefix}/OldKEnds", axis=0),
                _const_int64(v_starts, [1, 0]),
                _const_int64(v_end_head, [2]),
                helper.make_node("Concat", [v_end_head, step_name], [v_ends], name=f"{prefix}/OldVEnds", axis=0),
                _const_int64(axes, [0, 2]),
                _const_int64(steps, [1, 1]),
                _const_int64(squeeze_axis, [1]),
                helper.make_node("Slice", [state_name, k_starts, k_ends, axes, steps], [old_k_5d],
                                 name=f"{prefix}/SliceOldK"),
                helper.make_node("Slice", [state_name, v_starts, v_ends, axes, steps], [old_v_5d],
                                 name=f"{prefix}/SliceOldV"),
                helper.make_node("Squeeze", [old_k_5d, squeeze_axis], [old_k], name=f"{prefix}/SqueezeOldK"),
                helper.make_node("Squeeze", [old_v_5d, squeeze_axis], [old_v], name=f"{prefix}/SqueezeOldV"),
                helper.make_node("Squeeze", [k_update, squeeze_axis], [new_k], name=f"{prefix}/SqueezeNewK"),
                helper.make_node("Squeeze", [v_update, squeeze_axis], [new_v], name=f"{prefix}/SqueezeNewV"),
                helper.make_node("Concat", [old_k, new_k], [valid_k], name=f"{prefix}/ConcatValidK", axis=1),
                helper.make_node("Concat", [old_v, new_v], [valid_v], name=f"{prefix}/ConcatValidV", axis=1),
            ]
        )

        k_gather = nodes[valid_gathers[0]]
        v_gather = nodes[valid_gathers[1]]
        replacements[k_gather.output[0]] = valid_k
        replacements[v_gather.output[0]] = valid_v
        remove_indices.update({old_cache_idx, valid_cache_idx, valid_gathers[0], valid_gathers[1]})
        converted += 1

    if converted == 0:
        return 0

    for node in nodes:
        for i, input_name in enumerate(node.input):
            if input_name in replacements:
                node.input[i] = replacements[input_name]

    rewritten_nodes: list[onnx.NodeProto] = []
    for i, node in enumerate(nodes):
        rewritten_nodes.extend(new_nodes_by_insert.get(i, []))
        if i not in remove_indices:
            rewritten_nodes.append(node)

    del graph.node[:]
    graph.node.extend(rewritten_nodes)
    onnx.checker.check_model(model)
    return converted


def optimize_cache_update_shapes(model: onnx.ModelProto) -> int:
    """Replace cache data reads used only to compute update shapes.

    The exported packed-cache graph does this twice per layer:

        Gather(state, k_or_v) -> Slice(step:end) -> Shape -> Expand(new_kv)

    The Slice output is not used as data; only its shape is used. Since the
    slice always has shape [1, end - step, 16, 64], we can construct that
    shape directly and remove the full-cache Gather plus Slice.
    """

    graph = model.graph
    nodes = list(graph.node)
    consumers = _consumer_map(model)

    new_nodes_by_insert: dict[int, list[onnx.NodeProto]] = defaultdict(list)
    replacements: dict[str, str] = {}
    remove_indices: set[int] = set()
    converted = 0

    for state_idx in _packed_cache_state_indices(model):
        state_name = f"state_{state_idx}"
        for gather_idx, gather in enumerate(nodes):
            if gather_idx in remove_indices:
                continue
            if gather.op_type != "Gather" or len(gather.input) < 2 or gather.input[0] != state_name:
                continue
            axis = 0
            for attr in gather.attribute:
                if attr.name == "axis":
                    axis = int(attr.i)
            if axis != 0:
                continue
            if _constant_scalar_int(model, gather.input[1]) not in (0, 1):
                continue

            gather_consumers = consumers.get(gather.output[0], [])
            if len(gather_consumers) != 1:
                continue
            slice_idx = gather_consumers[0]
            slice_node = nodes[slice_idx]
            if slice_node.op_type != "Slice" or len(slice_node.input) < 5:
                continue

            slice_consumers = consumers.get(slice_node.output[0], [])
            if len(slice_consumers) != 1:
                continue
            shape_idx = slice_consumers[0]
            shape_node = nodes[shape_idx]
            if shape_node.op_type != "Shape":
                continue

            shape_consumers = consumers.get(shape_node.output[0], [])
            if not shape_consumers:
                continue

            starts = slice_node.input[1]
            ends = slice_node.input[2]
            prefix = f"/delta_kv_fast/state_{state_idx}/shape_{converted}"
            one = f"{prefix}/one"
            tail = f"{prefix}/tail"
            length = f"{prefix}/length"
            direct_shape = f"{prefix}/direct"

            new_nodes_by_insert[shape_idx].extend(
                [
                    _const_int64(one, [1]),
                    _const_int64(tail, [16, 64]),
                    helper.make_node("Sub", [ends, starts], [length], name=f"{prefix}/Length"),
                    helper.make_node("Concat", [one, length, tail], [direct_shape],
                                     name=f"{prefix}/DirectShape", axis=0),
                ]
            )

            replacements[shape_node.output[0]] = direct_shape
            remove_indices.update({gather_idx, slice_idx, shape_idx})
            converted += 1

    if converted == 0:
        return 0

    for node in nodes:
        for i, input_name in enumerate(node.input):
            if input_name in replacements:
                node.input[i] = replacements[input_name]

    rewritten_nodes: list[onnx.NodeProto] = []
    for i, node in enumerate(nodes):
        rewritten_nodes.extend(new_nodes_by_insert.get(i, []))
        if i not in remove_indices:
            rewritten_nodes.append(node)

    del graph.node[:]
    graph.node.extend(rewritten_nodes)
    onnx.checker.check_model(model)
    return converted


def _model_inputs(session: ort.InferenceSession) -> dict[str, tuple[int, list[int | str]]]:
    return {inp.name: (inp.type, list(inp.shape)) for inp in session.get_inputs()}


def _make_feed(
    session: ort.InferenceSession,
    cache_indices: Iterable[int],
    step: int,
    text_len: int,
    seq_len: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    feed: dict[str, np.ndarray] = {
        "sequence": rng.normal(size=(1, seq_len, 32)).astype(np.float32),
        "text_embeddings": rng.normal(size=(1, text_len, 1024)).astype(np.float32),
    }
    inputs = _model_inputs(session)
    for idx in cache_indices:
        cache_shape = inputs[f"state_{idx}"][1]
        capacity = int(cache_shape[2])
        heads = int(cache_shape[3])
        dim = int(cache_shape[4])
        cache = np.zeros((2, 1, capacity, heads, dim), dtype=np.float32)
        if step:
            cache[:, :, :step, :, :] = rng.normal(size=(2, 1, step, heads, dim)).astype(np.float32)
        feed[f"state_{idx}"] = cache
        feed[f"state_{idx + 1}"] = np.zeros((0,), dtype=np.float32)
        feed[f"state_{idx + 2}"] = np.array([step], dtype=np.int64)
    return feed


def verify_equivalent(input_path: Path, output_path: Path, cache_indices: list[int]) -> None:
    base = ort.InferenceSession(str(input_path), providers=["CPUExecutionProvider"])
    delta = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    base_names = [out.name for out in base.get_outputs()]
    delta_names = [out.name for out in delta.get_outputs()]
    rng = np.random.default_rng(123)

    cases = [(0, 5, 0), (0, 0, 1), (3, 0, 1), (3, 2, 1), (12, 0, 1), (30, 0, 7)]
    for step, text_len, seq_len in cases:
        feed = _make_feed(base, cache_indices, step, text_len, seq_len, rng)
        base_outputs = dict(zip(base_names, base.run(None, feed)))
        delta_outputs = dict(zip(delta_names, delta.run(None, feed)))
        total_new = text_len + seq_len

        for name in ("conditioning", "eos_logit"):
            max_diff = float(np.max(np.abs(base_outputs[name] - delta_outputs[name])))
            if max_diff != 0.0:
                raise RuntimeError(f"{name} mismatch for case {(step, text_len, seq_len)}: {max_diff}")

        for idx in cache_indices:
            out_name = f"out_state_{idx}"
            reconstructed = feed[f"state_{idx}"].copy()
            delta_slice = delta_outputs[out_name]
            expected_shape = reconstructed[:, :, step:step + total_new, :, :].shape
            if delta_slice.shape != expected_shape:
                raise RuntimeError(f"{out_name} delta shape mismatch: {delta_slice.shape} != {expected_shape}")
            reconstructed[:, :, step:step + total_new, :, :] = delta_slice
            max_diff = float(np.max(np.abs(base_outputs[out_name] - reconstructed)))
            if max_diff != 0.0:
                raise RuntimeError(f"{out_name} reconstruction mismatch: {max_diff}")


def convert_path(path: Path, output: Path | None, verify: bool) -> int:
    if path.is_dir():
        total = 0
        candidates = [
            ("flow_lm_main.onnx", "flow_lm_main_delta.onnx"),
            ("flow_lm_main_int8.onnx", "flow_lm_main_delta_int8.onnx"),
        ]
        for src_name, dst_name in candidates:
            src = path / src_name
            if not src.exists():
                continue
            total += convert_path(src, path / dst_name, verify)
        return total

    output_path = output
    if output_path is None:
        stem = path.stem
        if stem.endswith("_int8"):
            output_path = path.with_name(stem[:-5] + "_delta_int8.onnx")
        else:
            output_path = path.with_name(stem + "_delta.onnx")

    original = onnx.load(str(path))
    cache_indices = _packed_cache_state_indices(original)
    converted = convert_model(path, output_path)
    if converted == 0:
        print(f"  skipped {path}: no packed flow KV ScatterND pattern found")
        return 0

    print(f"  wrote {output_path} ({converted} cache states)")
    if verify:
        verify_equivalent(path, output_path, cache_indices)
        print("  verified exact equivalence")
    return converted


def main() -> int:
    parser = argparse.ArgumentParser(description="Create delta-KV PocketTTS ONNX model variants")
    parser.add_argument("input", type=Path, help="flow_lm_main ONNX file or bundle/model directory")
    parser.add_argument("output", type=Path, nargs="?", help="Output ONNX file when input is a file")
    parser.add_argument("--verify", action="store_true", help="Run exact ORT equivalence checks")
    args = parser.parse_args()

    converted = convert_path(args.input, args.output, args.verify)
    return 0 if converted >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
