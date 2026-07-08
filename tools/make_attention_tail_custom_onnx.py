#!/usr/bin/env python3
"""
Create a custom-attention variant of flow_lm_main_delta*.onnx.

This is a post-export graph rewrite layered on top of delta-KV:
  - delta-KV keeps model outputs as new K/V slices only.
  - this rewrite removes ONNX's old-cache Slice/Squeeze/ConcatValidK/V
    attention materialization path.
  - a C++ custom ORT op reads packed cache state_N, new K/V slices, and
    state_N+2 step directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import helper


TAIL = [
    "Transpose_2",
    "Transpose_1",
    "Mul_7",
    "MatMul",
    "Add_6",
    "Softmax",
    "MatMul_1",
    "Transpose_3",
    "Reshape_3",
]


def _producer_map(model: onnx.ModelProto) -> dict[str, int]:
    return {output: i for i, node in enumerate(model.graph.node) for output in node.output}


def _prune_dead_nodes(model: onnx.ModelProto) -> int:
    needed = {value.name for value in model.graph.output}
    kept = []
    for node in reversed(model.graph.node):
        if any(output in needed for output in node.output):
            kept.append(node)
            needed.update(input_name for input_name in node.input if input_name)
    kept.reverse()
    removed = len(model.graph.node) - len(kept)
    del model.graph.node[:]
    model.graph.node.extend(kept)
    return removed


def _prune_unused_initializers(model: onnx.ModelProto) -> int:
    used = {input_name for node in model.graph.node for input_name in node.input if input_name}
    used.update(value.name for value in model.graph.output)
    kept = [init for init in model.graph.initializer if init.name in used]
    removed = len(model.graph.initializer) - len(kept)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    return removed


def rewrite(input_path: Path, output_path: Path, layers: int) -> None:
    model = onnx.load(str(input_path), load_external_data=False)
    nodes_by_name = {node.name: node for node in model.graph.node}
    producers = _producer_map(model)
    skip = set()
    inserts_by_index: dict[int, onnx.NodeProto] = {}

    for layer in range(layers):
        prefix = f"/transformer/layers.{layer}/self_attn/"
        names = {name: prefix + name for name in TAIL}
        missing = [name for name in names.values() if name not in nodes_by_name]
        if missing:
            raise RuntimeError(f"layer {layer}: missing expected tail nodes: {missing}")

        transpose_k = nodes_by_name[names["Transpose_2"]]
        transpose_v = nodes_by_name[names["Transpose_1"]]
        mul_k = nodes_by_name[names["Mul_7"]]
        matmul_qk = nodes_by_name[names["MatMul"]]
        add_mask = nodes_by_name[names["Add_6"]]
        reshape = nodes_by_name[names["Reshape_3"]]

        valid_k = transpose_k.input[0]
        valid_v = transpose_v.input[0]
        concat_k = model.graph.node[producers[valid_k]]
        concat_v = model.graph.node[producers[valid_v]]
        if concat_k.op_type != "Concat" or concat_v.op_type != "Concat":
            raise RuntimeError(f"layer {layer}: expected valid K/V concat producers")

        old_k = concat_k.input[0]
        new_k = concat_k.input[1]
        new_v = concat_v.input[1]
        old_k_squeeze = model.graph.node[producers[old_k]]
        new_k_squeeze = model.graph.node[producers[new_k]]
        new_v_squeeze = model.graph.node[producers[new_v]]
        old_k_slice = model.graph.node[producers[old_k_squeeze.input[0]]]
        if old_k_slice.op_type != "Slice":
            raise RuntimeError(f"layer {layer}: expected old K slice producer")

        state = old_k_slice.input[0]
        if not state.startswith("state_"):
            raise RuntimeError(f"layer {layer}: unexpected state input {state}")
        state_index = int(state[len("state_"):])

        custom = helper.make_node(
            "AttentionTail",
            [
                matmul_qk.input[0],
                state,
                new_k_squeeze.input[0],
                new_v_squeeze.input[0],
                f"state_{state_index + 2}",
                mul_k.input[1],
                add_mask.input[1],
            ],
            [reshape.output[0]],
            name=f"/pockettts/layers.{layer}/AttentionTail",
            domain="pockettts",
        )
        inserts_by_index[producers[transpose_k.output[0]]] = custom
        skip.update(names.values())

    rewritten = []
    for index, node in enumerate(model.graph.node):
        if index in inserts_by_index:
            rewritten.append(inserts_by_index[index])
        if node.name not in skip:
            rewritten.append(node)

    del model.graph.node[:]
    model.graph.node.extend(rewritten)

    if not any(op.domain == "pockettts" for op in model.opset_import):
        model.opset_import.extend([helper.make_opsetid("pockettts", 1)])

    dead = _prune_dead_nodes(model)
    unused = _prune_unused_initializers(model)
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    print(f"replaced {layers} attention tails")
    print(f"pruned dead nodes: {dead}")
    print(f"pruned unused initializers: {unused}")
    print(f"wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="flow_lm_main_delta ONNX file")
    parser.add_argument("output", type=Path, help="custom-attention ONNX output file")
    parser.add_argument("--layers", type=int, default=6)
    args = parser.parse_args()
    rewrite(args.input, args.output, args.layers)


if __name__ == "__main__":
    main()
