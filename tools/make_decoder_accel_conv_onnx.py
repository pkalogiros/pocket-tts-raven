#!/usr/bin/env python3
"""Replace the Mimi decoder's stride-1 Conv nodes with AccelConv custom ops.

Every Conv in the decoder is 1D with stride 1, dilation 1, group 1 and no
padding (streaming state is concatenated upstream), so each one is exactly

    out[co, t] = bias[co] + sum_k sum_ci W[co, ci, k] * x[ci, t + k]

which the AccelConv C++ op computes as K accumulated sgemm calls through
Accelerate. Weights are re-packed offline to [K, Cout, Cin]. When the Conv's
only consumer is Elu(alpha=1), the AccelConvElu variant fuses the activation.

Run on the fused-ConvTranspose delta decoder (the deployed default):

    uv run --no-project --with onnx python make_decoder_accel_conv_onnx.py \
        models/mimi_decoder_delta_convtr_int8.onnx \
        /private/tmp/mimi_decoder_delta_convtr_accel_int8.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import OperatorSetIdProto, helper, numpy_helper


def attr_value(node: onnx.NodeProto, name: str, default=None):
    for attr in node.attribute:
        if attr.name == name:
            return helper.get_attribute_value(attr)
    return default


def replace_initializer(graph: onnx.GraphProto, tensor: onnx.TensorProto) -> None:
    for i, existing in enumerate(graph.initializer):
        if existing.name == tensor.name:
            graph.initializer[i].CopyFrom(tensor)
            return
    graph.initializer.append(tensor)


def ensure_custom_opset(model: onnx.ModelProto) -> None:
    for opset in model.opset_import:
        if opset.domain == "pockettts":
            opset.version = max(opset.version, 1)
            return
    opset = OperatorSetIdProto()
    opset.domain = "pockettts"
    opset.version = 1
    model.opset_import.append(opset)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="mimi_decoder delta ONNX file")
    parser.add_argument("output", type=Path, help="AccelConv ONNX file")
    parser.add_argument("--only", default=None,
                        help="comma-separated substrings; only rewrite matching Conv nodes")
    args = parser.parse_args()
    only = args.only.split(",") if args.only else None

    model = onnx.load(args.input)
    graph = model.graph
    init = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    replacements: dict[str, onnx.NodeProto] = {}
    remove_names: set[str] = set()

    for node in graph.node:
        if node.op_type != "Conv":
            continue
        if only is not None and not any(s in node.name for s in only):
            continue
        strides = attr_value(node, "strides", [1])
        dilations = attr_value(node, "dilations", [1])
        group = attr_value(node, "group", 1)
        pads = attr_value(node, "pads", [0, 0])
        if list(strides) != [1] or list(dilations) != [1] or group != 1 or any(pads):
            print(f"skip {node.name}: unsupported attrs")
            continue
        if node.input[1] not in init:
            print(f"skip {node.name}: non-initializer weight")
            continue

        weight = init[node.input[1]].astype(np.float32, copy=False)
        if weight.ndim != 3:
            raise RuntimeError(f"{node.name}: expected weight [Cout,Cin,K]")
        cout, cin, kernel = weight.shape
        packed = np.ascontiguousarray(weight.transpose(2, 0, 1))
        packed_name = f"pockettts{node.name}.accelconv.w"
        replace_initializer(graph, numpy_helper.from_array(packed, packed_name))

        if len(node.input) > 2:
            bias_name = node.input[2]
        else:
            bias_name = f"pockettts{node.name}.accelconv.zero_bias"
            replace_initializer(
                graph, numpy_helper.from_array(np.zeros(cout, np.float32), bias_name))

        # Fuse a following Elu(alpha=1) when it is the sole consumer.
        op_type = "AccelConv"
        output_name = node.output[0]
        conv_consumers = consumers.get(node.output[0], [])
        if (len(conv_consumers) == 1 and conv_consumers[0].op_type == "Elu"
                and attr_value(conv_consumers[0], "alpha", 1.0) == 1.0
                and node.output[0] not in {o.name for o in graph.output}):
            elu = conv_consumers[0]
            op_type = "AccelConvElu"
            output_name = elu.output[0]
            remove_names.add(elu.name)

        custom = helper.make_node(
            op_type,
            inputs=[node.input[0], packed_name, bias_name],
            outputs=[output_name],
            name=f"{node.name}/pockettts/{op_type}",
            domain="pockettts",
        )
        replacements[node.name] = custom
        print(f"{op_type} {node.name}: Cout={cout} Cin={cin} K={kernel}")

    new_nodes: list[onnx.NodeProto] = []
    for node in graph.node:
        if node.name in replacements:
            new_nodes.append(replacements[node.name])
        elif node.name not in remove_names:
            new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)

    # Drop initializers no longer referenced (the original Conv weights).
    referenced = {name for node in graph.node for name in node.input}
    referenced.update(o.name for o in graph.output)
    keep = [t for t in graph.initializer if t.name in referenced]
    removed = len(graph.initializer) - len(keep)
    del graph.initializer[:]
    graph.initializer.extend(keep)
    if removed:
        print(f"removed {removed} unused initializers")

    ensure_custom_opset(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
