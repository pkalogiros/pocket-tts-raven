#!/usr/bin/env python3
"""Create a fused-ConvTranspose variant of mimi_decoder_delta*.onnx.

The Mimi decoder's three large transposed conv streaming blocks have this exact
shape:

    ConvTranspose(kernel=2*stride)
      -> Slice(first stride) + previous state
      -> Concat(with rest)
      -> Slice(current output) and Slice/Sub(next state)

For kernel=2*stride this is equivalent to a two-phase polyphase convolution:
each emitted phase depends on the current input frame and the previous input
frame, while the final tail becomes the next streaming state. The custom C++
op computes the emitted output and next state directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import OperatorSetIdProto, helper, numpy_helper


BLOCKS = [
    ("/decoder/model.2", "state_4", "out_state_4"),
    ("/decoder/model.5", "state_9", "out_state_9"),
    ("/decoder/model.8", "state_14", "out_state_14"),
]


def attr_ints(node: onnx.NodeProto, name: str) -> list[int]:
    for attr in node.attribute:
        if attr.name == name:
            value = helper.get_attribute_value(attr)
            if isinstance(value, int):
                return [value]
            return list(value)
    raise KeyError(f"{node.name} missing attr {name}")


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
    parser.add_argument("input", type=Path, help="mimi_decoder_delta ONNX file")
    parser.add_argument("output", type=Path, help="fused ConvTranspose ONNX file")
    args = parser.parse_args()

    model = onnx.load(args.input)
    graph = model.graph
    nodes_by_name = {node.name: node for node in graph.node}
    init = {tensor.name: numpy_helper.to_array(tensor) for tensor in graph.initializer}

    replacements: dict[str, onnx.NodeProto] = {}
    remove_names: set[str] = set()

    for prefix, state_name, state_out_name in BLOCKS:
        conv_name = f"{prefix}/convtr/ConvTranspose"
        conv = nodes_by_name[conv_name]
        strides = attr_ints(conv, "strides")
        kernel_shape = attr_ints(conv, "kernel_shape")
        groups = attr_ints(conv, "group")[0]
        if len(strides) != 1 or len(kernel_shape) != 1:
            raise RuntimeError(f"{conv_name}: expected 1D ConvTranspose")
        stride = strides[0]
        kernel = kernel_shape[0]
        if groups != 1 or kernel != 2 * stride:
            raise RuntimeError(f"{conv_name}: expected group=1 and kernel=2*stride")
        if len(conv.input) != 3:
            raise RuntimeError(f"{conv_name}: expected input, weight, bias")

        weight = init[conv.input[1]].astype(np.float32, copy=False)
        if weight.ndim != 3:
            raise RuntimeError(f"{conv_name}: expected weight [Cin,Cout,K]")
        cin, cout, k = weight.shape
        if k != kernel:
            raise RuntimeError(f"{conv_name}: weight kernel mismatch")

        w_first = np.ascontiguousarray(weight[:, :, :stride].transpose(2, 0, 1))
        w_tail = np.ascontiguousarray(weight[:, :, stride:].transpose(2, 0, 1))
        first_name = f"pockettts{prefix}.convtr.w_first"
        tail_name = f"pockettts{prefix}.convtr.w_tail"
        replace_initializer(graph, numpy_helper.from_array(w_first, first_name))
        replace_initializer(graph, numpy_helper.from_array(w_tail, tail_name))

        custom = helper.make_node(
            "DecoderConvTransposeOverlap",
            inputs=[conv.input[0], first_name, tail_name, conv.input[2], state_name],
            outputs=[f"{prefix}/Slice_3_output_0", state_out_name],
            name=f"{prefix}/pockettts/DecoderConvTransposeOverlap",
            domain="pockettts",
        )
        replacements[conv_name] = custom
        remove_names.update(
            {
                conv_name,
                f"{prefix}/Slice",
                f"{prefix}/Slice_1",
                f"{prefix}/Add",
                f"{prefix}/Concat",
                f"{prefix}/Slice_2",
                f"{prefix}/Sub",
                f"{prefix}/Slice_3",
            }
        )
        print(f"fused {conv_name}: Cin={cin} Cout={cout} stride={stride}")

    new_nodes: list[onnx.NodeProto] = []
    for node in graph.node:
        if node.name in replacements:
            new_nodes.append(replacements[node.name])
        elif node.name not in remove_names:
            new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)
    ensure_custom_opset(model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
