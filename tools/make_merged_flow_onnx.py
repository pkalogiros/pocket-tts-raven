#!/usr/bin/env python3
"""Merge flow_lm_flow into flow_lm_main so one Run() covers both.

With --lsd-steps 1 (the default), the runtime solves the flow ODE with a
single Euler step at s=0, t=1:

    latent = x + flow_dir(conditioning, 0, 1, x) * dt      (dt = 1)

so the flow MLP can be inlined into the main AR graph: its `c` input wires to
the main graph's `conditioning` output value, s/t become baked constants, the
noise `x` becomes a new graph input `flow_x`, and a new `latent` output is
appended. The C++ runtime uses the merged model only when lsd_steps == 1.

    uv run --no-project --with onnx python make_merged_flow_onnx.py \
        models/flow_lm_main_delta_attn_int8.onnx \
        models/flow_lm_flow_int8.onnx \
        /private/tmp/flow_lm_main_delta_attn_flow_int8.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import OperatorSetIdProto, helper, numpy_helper

PREFIX = "flowm/"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("main_model", type=Path)
    ap.add_argument("flow_model", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()

    main = onnx.load(args.main_model)
    flow = onnx.load(args.flow_model)
    g = main.graph
    fg = flow.graph

    out_names = [o.name for o in g.output]
    if "conditioning" not in out_names:
        raise SystemExit(f"main model outputs {out_names}, no 'conditioning'")
    flow_ins = [i.name for i in fg.input]
    if flow_ins != ["c", "s", "t", "x"]:
        raise SystemExit(f"unexpected flow inputs {flow_ins}")
    flow_out = fg.output[0].name

    # Name mapping for the inlined flow graph.
    rename = {
        "c": "conditioning",
        "s": PREFIX + "s_const",
        "t": PREFIX + "t_const",
        "x": "flow_x",
    }

    existing = {t.name for t in g.initializer}
    for t in fg.initializer:
        nt = onnx.TensorProto()
        nt.CopyFrom(t)
        nt.name = PREFIX + t.name
        rename[t.name] = nt.name
        if nt.name in existing:
            raise SystemExit(f"initializer collision {nt.name}")
        g.initializer.append(nt)

    g.initializer.append(numpy_helper.from_array(
        np.zeros((1, 1), np.float32), PREFIX + "s_const"))
    g.initializer.append(numpy_helper.from_array(
        np.ones((1, 1), np.float32), PREFIX + "t_const"))

    for node in fg.node:
        nn = onnx.NodeProto()
        nn.CopyFrom(node)
        nn.name = PREFIX + (node.name or node.op_type)
        for i, name in enumerate(nn.input):
            nn.input[i] = rename.get(name, PREFIX + name)
        for i, name in enumerate(nn.output):
            rename[name] = PREFIX + name
            nn.output[i] = PREFIX + name
        g.node.append(nn)

    g.node.append(helper.make_node(
        "Add", inputs=["flow_x", rename[flow_out]], outputs=["latent"],
        name=PREFIX + "euler_step"))

    g.input.append(helper.make_tensor_value_info(
        "flow_x", onnx.TensorProto.FLOAT, [1, 32]))
    g.output.append(helper.make_tensor_value_info(
        "latent", onnx.TensorProto.FLOAT, [1, 32]))

    # Merge opset imports (flow uses only the default domain).
    main_domains = {op.domain: op for op in main.opset_import}
    for op in flow.opset_import:
        if op.domain in main_domains:
            main_domains[op.domain].version = max(main_domains[op.domain].version, op.version)
        else:
            new = OperatorSetIdProto()
            new.domain = op.domain
            new.version = op.version
            main.opset_import.append(new)

    print(f"inlined {len(fg.node)} flow nodes, +2 io ({len(g.node)} total nodes)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(main, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
