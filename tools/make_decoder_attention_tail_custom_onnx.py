#!/usr/bin/env python3
"""
Create a custom-attention variant of mimi_decoder_delta*.onnx.

This is layered on top of the decoder delta-KV rewrite. It keeps decoder cache
outputs as delta slices, but replaces each decoder transformer's attention tail
with a C++ custom ORT op that reads the packed cache directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import helper


TAIL = [
    "Transpose_5",
    "Mul_2",
    "MatMul",
    "Add_5",
    "Softmax",
    "IsNaN",
    "Where_4",
    "MatMul_1",
    "Transpose_6",
    "Reshape_10",
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
        prefix = f"/decoder_transformer/transformer/layers.{layer}/self_attn/"
        names = {name: prefix + name for name in TAIL}
        missing = [name for name in names.values() if name not in nodes_by_name]
        if missing:
            raise RuntimeError(f"layer {layer}: missing expected tail nodes: {missing}")

        mul_k = nodes_by_name[names["Mul_2"]]
        matmul_qk = nodes_by_name[names["MatMul"]]
        add_mask = nodes_by_name[names["Add_5"]]
        reshape = nodes_by_name[names["Reshape_10"]]

        scatter_k = nodes_by_name[prefix + "ScatterElements"]
        scatter_v = nodes_by_name[prefix + "ScatterElements_1"]
        gather_k = model.graph.node[producers[scatter_k.input[0]]]
        gather_v = model.graph.node[producers[scatter_v.input[0]]]
        if gather_k.op_type != "Gather" or gather_v.op_type != "Gather":
            raise RuntimeError(f"layer {layer}: expected packed K/V gather producers")
        if gather_k.input[0] != gather_v.input[0]:
            raise RuntimeError(f"layer {layer}: K/V gathers read different packed states")

        custom = helper.make_node(
            "DecoderAttentionTail",
            [
                matmul_qk.input[0],
                gather_k.input[0],
                scatter_k.input[2],
                scatter_v.input[2],
                scatter_k.input[1],
                mul_k.input[1],
                add_mask.input[1],
            ],
            [reshape.output[0]],
            name=f"/pockettts/decoder_layers.{layer}/DecoderAttentionTail",
            domain="pockettts",
        )
        inserts_by_index[producers[matmul_qk.output[0]]] = custom
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
    print(f"replaced {layers} decoder attention tails")
    print(f"pruned dead nodes: {dead}")
    print(f"pruned unused initializers: {unused}")
    print(f"wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="mimi_decoder_delta ONNX file")
    parser.add_argument("output", type=Path, help="custom-attention decoder ONNX output file")
    parser.add_argument("--layers", type=int, default=2)
    args = parser.parse_args()
    rewrite(args.input, args.output, args.layers)


if __name__ == "__main__":
    main()
