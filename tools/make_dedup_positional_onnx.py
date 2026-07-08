#!/usr/bin/env python3
"""Cross-layer CSE for PocketTTS transformer graphs.

Every transformer layer carries its own int64 step-counter state, but the
runtime advances them in lockstep, so they always hold equal values. Each
layer therefore recomputes an identical positional subgraph per step: the
RoPE angle table (Range -> Exp -> Cos/Sin and its Mul/Unsqueeze/Concat glue)
and the additive attention mask (Range -> LessOrEqual -> Where). ORT cannot
CSE these because the counters are distinct graph inputs.

This pass canonicalizes all scalar-int64 "state_*" inputs to one token, then
does common-subexpression elimination in topological order: any node whose
(op_type, attributes, canonicalized inputs) matches an earlier node is
dropped and its consumers rewired to the survivor. Cache updates stay
per-layer automatically because each ScatterElements takes its own cache
state as input, which never canonicalizes equal.

Correctness rests on the counters being equal at every step; verify with
verify_model_equivalence.py (lockstep state feedback) plus a temp-0 CLI run.

    uv run --no-project --with onnx python make_dedup_positional_onnx.py \
        models/flow_lm_main_delta_attn_int8.onnx out.onnx
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import onnx
from onnx import numpy_helper


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()

    model = onnx.load(args.input)
    graph = model.graph

    inits = {t.name: t for t in graph.initializer}
    graph_inputs = {i.name: i for i in graph.input}
    graph_outputs = {o.name for o in graph.output}

    # Canonical token per value name. Step counters (scalar int64 states) all
    # map to one token; small initializers map to a content hash so identical
    # constants in different layers compare equal.
    canon: dict[str, str] = {}
    for i in graph.input:
        tt = i.type.tensor_type
        dims = [d.dim_value for d in tt.shape.dim]
        if (i.name.startswith("state_") and tt.elem_type == onnx.TensorProto.INT64
                and dims in ([1], [])):
            canon[i.name] = "@STEP"
        else:
            canon[i.name] = f"@in:{i.name}"
    for name, t in inits.items():
        raw = t.SerializeToString()
        if len(raw) <= 65536:
            canon[name] = "@init:" + hashlib.sha1(raw).hexdigest()
        else:
            canon[name] = f"@biginit:{name}"

    def node_key(node: onnx.NodeProto) -> str | None:
        ins = []
        for i in node.input:
            if i == "":
                ins.append("@none")
            elif i in canon:
                ins.append(canon[i])
            else:
                return None  # producer not canonicalized (should not happen in topo order)
        attrs = sorted((a.name, a.SerializeToString().hex()) for a in node.attribute)
        raw = f"{node.domain}:{node.op_type}|{attrs}|{'|'.join(ins)}"
        # Hash to keep keys fixed-size; raw keys otherwise grow with graph depth
        # (each key embeds its inputs' keys).
        return hashlib.sha1(raw.encode()).hexdigest()

    survivors: dict[str, onnx.NodeProto] = {}
    replaced: dict[str, str] = {}  # dropped output name -> surviving output name
    kept_nodes: list[onnx.NodeProto] = []
    dropped = 0

    for node in graph.node:
        # Rewire inputs of every node through the replacement map first.
        for idx, i in enumerate(node.input):
            if i in replaced:
                node.input[idx] = replaced[i]
        key = node_key(node)
        producing_graph_output = any(o in graph_outputs for o in node.output)
        if key is not None and not producing_graph_output and key in survivors:
            surv = survivors[key]
            for o_old, o_new in zip(node.output, surv.output):
                replaced[o_old] = o_new
                canon[o_old] = canon[o_new]
            dropped += 1
            continue
        if key is not None and key not in survivors and not producing_graph_output:
            survivors[key] = node
        for pos, o in enumerate(node.output):
            canon[o] = f"@node:{key}#{pos}" if key is not None else f"@opaque:{o}"
        kept_nodes.append(node)

    del graph.node[:]
    graph.node.extend(kept_nodes)

    referenced = {name for node in graph.node for name in node.input}
    referenced.update(graph_outputs)
    keep_inits = [t for t in graph.initializer if t.name in referenced]
    removed_inits = len(graph.initializer) - len(keep_inits)
    del graph.initializer[:]
    graph.initializer.extend(keep_inits)

    print(f"dropped {dropped} duplicate nodes, {removed_inits} unused initializers, "
          f"{len(kept_nodes)} nodes remain")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
