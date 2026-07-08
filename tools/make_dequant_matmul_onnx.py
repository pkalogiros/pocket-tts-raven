#!/usr/bin/env python3
"""Replace dynamic-quantized MatMul chains with plain f32 MatMul.

Targets the QOperator pattern the quantizer emits (which ORT fuses into
DynamicQuantizeMatMul at load time):

    DynamicQuantizeLinear(A) -> A_q, a_scale, a_zp
    MatMulInteger(A_q, B_q, a_zp, b_zp) -> int32
    Cast(int32 -> f32)
    Mul(cast, Mul(a_scale, b_scale))

becomes MatMul(A, W) with W = (B_q - b_zp) * b_scale dequantized offline.

This is NOT bit-identical to the quantized graph: it removes the activation
quantization error entirely (strictly closer to the fp32 reference). Weights
grow 4x on disk for the affected matmuls.

    uv run --no-project --with onnx python make_dequant_matmul_onnx.py \
        models/mimi_decoder_delta_convtr_int8.onnx out.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--only", default=None,
                    help="comma-separated substrings; only rewrite matching MatMulInteger nodes")
    args = ap.parse_args()
    only = args.only.split(",") if args.only else None

    model = onnx.load(args.input)
    graph = model.graph
    inits = {t.name: t for t in graph.initializer}
    producers = {o: n for n in graph.node for o in n.output}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    def sole_consumer(name: str, op_type: str):
        c = consumers.get(name, [])
        if len(c) == 1 and c[0].op_type == op_type:
            return c[0]
        return None

    remove: set[str] = set()
    new_nodes: list[onnx.NodeProto] = []
    add_inits: list[onnx.TensorProto] = []
    rewritten = 0

    for node in graph.node:
        if node.op_type != "MatMulInteger":
            continue
        if only is not None and not any(s in node.name for s in only):
            continue
        dql = producers.get(node.input[0])
        if dql is None or dql.op_type != "DynamicQuantizeLinear":
            print(f"skip {node.name}: A not from DynamicQuantizeLinear")
            continue
        if node.input[1] not in inits or node.input[3] not in inits:
            print(f"skip {node.name}: B/zp not initializers")
            continue
        cast = sole_consumer(node.output[0], "Cast")
        if cast is None:
            print(f"skip {node.name}: no sole Cast consumer")
            continue
        mul = sole_consumer(cast.output[0], "Mul")
        if mul is None:
            print(f"skip {node.name}: no sole Mul consumer")
            continue
        scale_input = mul.input[0] if mul.input[1] == cast.output[0] else mul.input[1]
        scales_mul = producers.get(scale_input)
        if scales_mul is None or scales_mul.op_type != "Mul":
            print(f"skip {node.name}: scale input not Mul(a_scale, b_scale)")
            continue
        a_scale_name = dql.output[1]
        b_scale_name = None
        for i in scales_mul.input:
            if i == a_scale_name:
                continue
            if i in inits:
                b_scale_name = i
        if b_scale_name is None or a_scale_name not in scales_mul.input:
            print(f"skip {node.name}: cannot identify b_scale")
            continue

        b_q = numpy_helper.to_array(inits[node.input[1]])
        b_zp = numpy_helper.to_array(inits[node.input[3]])
        b_scale = numpy_helper.to_array(inits[b_scale_name]).astype(np.float32)
        w = (b_q.astype(np.int32) - b_zp.astype(np.int32)).astype(np.float32) * b_scale
        w_name = f"pockettts{node.name}.dequant.w"
        add_inits.append(numpy_helper.from_array(np.ascontiguousarray(w), w_name))

        new_nodes.append(helper.make_node(
            "MatMul",
            inputs=[dql.input[0], w_name],
            outputs=[mul.output[0]],
            name=f"{node.name}/pockettts/DequantMatMul",
        ))
        # DQL may feed multiple MatMulIntegers; only remove when this was the
        # sole consumer of its quantized output.
        dql_users = [c for c in consumers.get(dql.output[0], [])]
        if len(dql_users) == 1:
            remove.add(dql.name)
        scales_users = [c for c in consumers.get(scales_mul.output[0], [])]
        if len(scales_users) == 1:
            remove.add(scales_mul.name)
        remove.update({node.name, cast.name, mul.name})
        rewritten += 1
        print(f"dequant {node.name}: W={list(w.shape)} scale={'per-col' if b_scale.ndim else 'scalar'}")

    if not rewritten:
        raise SystemExit("nothing rewritten")

    kept = [n for n in graph.node if n.name not in remove]
    del graph.node[:]
    graph.node.extend(kept)
    # Insert new MatMuls right before their consumers is unnecessary — ORT
    # topo-sorts at load; but onnx.checker wants topological order, so append
    # after producers: simplest is to re-emit new nodes at the position of the
    # removed Mul. Append + topo-fix via onnx shape inference is overkill;
    # instead, insert each new node where its inputs are already produced.
    graph.node.extend(new_nodes)
    graph.initializer.extend(add_inits)

    referenced = {name for node in graph.node for name in node.input}
    referenced.update(o.name for o in graph.output)
    keep_inits = [t for t in graph.initializer if t.name in referenced]
    removed_inits = len(graph.initializer) - len(keep_inits)
    del graph.initializer[:]
    graph.initializer.extend(keep_inits)

    # Topologically sort (new nodes were appended out of order).
    nodes = list(graph.node)
    produced = {t.name for t in graph.initializer}
    produced.update(i.name for i in graph.input)
    produced.update(o for n in nodes for o in n.output if n.op_type == "Constant")
    ordered, pending = [], nodes
    while pending:
        progress = []
        rest = []
        for n in pending:
            if all(i in produced or not i for i in n.input):
                progress.append(n)
                produced.update(n.output)
            else:
                rest.append(n)
        if not progress:
            raise SystemExit("topological sort stuck (cycle?)")
        ordered.extend(progress)
        pending = rest
    del graph.node[:]
    graph.node.extend(ordered)

    print(f"rewrote {rewritten} matmuls, dropped {removed_inits} initializers")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
