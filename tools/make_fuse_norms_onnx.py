#!/usr/bin/env python3
"""
Fuse decomposed normalization / activation patterns into single contrib ops.

The PyTorch exporter shatters every LayerNorm into ReduceMean/Sub/Pow/Sqrt/Div
chains (and RMSNorm, GELU similarly). Native ORT dispatch makes this almost
free; WASM dispatch costs ~microseconds per op, so ~300 glue nodes per AR step
are real money. ORT's runtime fusion passes don't match these patterns (quant
Casts interleaved), so we fuse offline:

    LayerNorm  : RM -> Sub -> Pow(2) -> RM -> Add(eps) -> Sqrt -> Div
                 [-> Mul(gamma) [-> Add(beta)]]      => com.microsoft LayerNormalization
    RMSNorm    : Mul(x,x)|Pow(2) -> RM -> Add(eps) -> Sqrt -> Div(x,.)
                 [-> Mul(gamma)]                     => com.microsoft SimplifiedLayerNormalization
    GELU (erf) : Div(x, sqrt2) -> Erf -> Add(1) -> Mul(x,.) -> Mul(0.5)
                                                     => com.microsoft Gelu

Only exact, full-precision-preserving matches are fused; anything ambiguous is
left alone. Verify with verify_model_equivalence.py afterwards.

    python make_fuse_norms_onnx.py in.onnx out.onnx
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

MSDOMAIN = "com.microsoft"


class G:
    def __init__(self, model: onnx.ModelProto):
        self.model = model
        self.g = model.graph
        self.inits = {t.name: t for t in self.g.initializer}
        self.reindex()

    def reindex(self):
        self.producers = {}
        self.consumers = defaultdict(list)
        for n in self.g.node:
            for o in n.output:
                self.producers[o] = n
            for i in n.input:
                self.consumers[i].append(n)
        self.graph_outputs = {o.name for o in self.g.output}

    def const_scalar(self, name):
        """Value of a scalar initializer or Constant node output, else None."""
        if name in self.inits:
            a = numpy_helper.to_array(self.inits[name])
            return float(a.reshape(-1)[0]) if a.size == 1 else None
        p = self.producers.get(name)
        if p is not None and p.op_type == "Constant":
            for attr in p.attribute:
                if attr.name == "value":
                    a = numpy_helper.to_array(attr.t)
                    return float(a.reshape(-1)[0]) if a.size == 1 else None
        return None

    def only_consumer(self, tensor, op_type):
        """The single consumer of `tensor` if it has exactly one and matches."""
        if tensor in self.graph_outputs:
            return None
        cs = self.consumers.get(tensor, [])
        if len(cs) != 1 or cs[0].op_type != op_type:
            return None
        return cs[0]

    def axes_is_last(self, node):
        for attr in node.attribute:
            if attr.name == "axes":
                return list(attr.ints) in ([-1], [2])
        # opset >= 18 takes axes as input
        if len(node.input) > 1:
            a = self.inits.get(node.input[1])
            if a is not None:
                return list(numpy_helper.to_array(a)) in ([-1], [2])
        return False


def fuse_layernorm(gi: G):
    """RM -> Sub -> {Pow2 -> RM -> Add eps -> Sqrt, Div} -> Mul(g) -> Add(b)"""
    fused = 0
    dead: set[int] = set()
    new_nodes = []
    for rm1 in list(gi.g.node):
        if id(rm1) in dead or rm1.op_type != "ReduceMean" or not gi.axes_is_last(rm1):
            continue
        x = rm1.input[0]
        sub = gi.only_consumer(rm1.output[0], "Sub")
        if sub is None or sub.input != [x, rm1.output[0]]:
            continue
        # Sub feeds exactly Pow and Div
        subs_cs = gi.consumers.get(sub.output[0], [])
        if sorted(c.op_type for c in subs_cs) != ["Div", "Pow"]:
            continue
        pow2 = next(c for c in subs_cs if c.op_type == "Pow")
        div = next(c for c in subs_cs if c.op_type == "Div")
        if gi.const_scalar(pow2.input[1]) != 2.0:
            continue
        rm2 = gi.only_consumer(pow2.output[0], "ReduceMean")
        if rm2 is None or not gi.axes_is_last(rm2):
            continue
        addeps = gi.only_consumer(rm2.output[0], "Add")
        if addeps is None:
            continue
        eps = gi.const_scalar(addeps.input[1]) or gi.const_scalar(addeps.input[0])
        if eps is None or eps > 1e-3:
            continue
        sqrt = gi.only_consumer(addeps.output[0], "Sqrt")
        if sqrt is None or div.input[1] != sqrt.output[0] or div.input[0] != sub.output[0]:
            continue
        # optional affine tail
        gamma = beta = None
        out_node = div
        mul = gi.only_consumer(div.output[0], "Mul")
        if mul is not None:
            gname = mul.input[1] if mul.input[0] == div.output[0] else mul.input[0]
            if gname in gi.inits:
                gamma = gname
                out_node = mul
                addb = gi.only_consumer(mul.output[0], "Add")
                if addb is not None:
                    bname = addb.input[1] if addb.input[0] == mul.output[0] else addb.input[0]
                    if bname in gi.inits:
                        beta = bname
                        out_node = addb
        if gamma is None:
            continue  # contrib LayerNormalization requires scale
        group = [rm1, sub, pow2, rm2, addeps, sqrt, div]
        if out_node is not div:
            group.append(mul)
        if beta is not None:
            group.append(out_node)  # addb
        ln_inputs = [x, gamma] + ([beta] if beta else [])
        ln = helper.make_node("LayerNormalization", ln_inputs, [out_node.output[0]],
                              name=f"fused_ln_{fused}",
                              axis=-1, epsilon=float(eps))
        new_nodes.append((out_node, ln))
        dead.update(id(n) for n in group)
        fused += 1
    apply_replacements(gi, dead, new_nodes)
    return fused


def fuse_rmsnorm(gi: G):
    """[Mul(x,x)|Pow2] -> RM -> Add eps -> Sqrt -> Div(x,.) -> Mul(g)"""
    fused = 0
    dead: set[int] = set()
    new_nodes = []
    for sq in list(gi.g.node):
        if id(sq) in dead:
            continue
        if sq.op_type == "Mul" and len(sq.input) == 2 and sq.input[0] == sq.input[1]:
            x = sq.input[0]
        elif sq.op_type == "Pow" and gi.const_scalar(sq.input[1]) == 2.0:
            x = sq.input[0]
        else:
            continue
        rm = gi.only_consumer(sq.output[0], "ReduceMean")
        if rm is None or not gi.axes_is_last(rm):
            continue
        addeps = gi.only_consumer(rm.output[0], "Add")
        if addeps is None:
            continue
        eps = gi.const_scalar(addeps.input[1]) or gi.const_scalar(addeps.input[0])
        if eps is None or eps > 1e-3:
            continue
        sqrt = gi.only_consumer(addeps.output[0], "Sqrt")
        if sqrt is None:
            continue
        div = gi.only_consumer(sqrt.output[0], "Div")
        if div is None or div.input != [x, sqrt.output[0]]:
            continue
        mul = gi.only_consumer(div.output[0], "Mul")
        if mul is None:
            continue
        gname = mul.input[1] if mul.input[0] == div.output[0] else mul.input[0]
        if gname not in gi.inits:
            continue
        group = [sq, rm, addeps, sqrt, div, mul]
        sln = helper.make_node("SimplifiedLayerNormalization", [x, gname], [mul.output[0]],
                               name=f"fused_rms_{fused}", domain=MSDOMAIN,
                               axis=-1, epsilon=float(eps))
        new_nodes.append((mul, sln))
        dead.update(id(n) for n in group)
        fused += 1
    apply_replacements(gi, dead, new_nodes)
    return fused


def fuse_gelu(gi: G):
    """Div(x, sqrt2) -> Erf -> Add(1) -> Mul -> Mul(0.5|x) => Gelu(x)"""
    fused = 0
    dead: set[int] = set()
    new_nodes = []
    for div in list(gi.g.node):
        if id(div) in dead or div.op_type != "Div":
            continue
        c = gi.const_scalar(div.input[1])
        if c is None or abs(c - float(np.sqrt(2.0))) > 1e-5:
            continue
        x = div.input[0]
        erf = gi.only_consumer(div.output[0], "Erf")
        if erf is None:
            continue
        add1 = gi.only_consumer(erf.output[0], "Add")
        if add1 is None or (gi.const_scalar(add1.input[1]) or gi.const_scalar(add1.input[0])) != 1.0:
            continue
        mulx = gi.only_consumer(add1.output[0], "Mul")
        if mulx is None or x not in mulx.input:
            continue
        mulh = gi.only_consumer(mulx.output[0], "Mul")
        if mulh is None or (gi.const_scalar(mulh.input[1]) or gi.const_scalar(mulh.input[0])) != 0.5:
            continue
        group = [div, erf, add1, mulx, mulh]
        gelu = helper.make_node("Gelu", [x], [mulh.output[0]],
                                name=f"fused_gelu_{fused}", domain=MSDOMAIN)
        new_nodes.append((mulh, gelu))
        dead.update(id(n) for n in group)
        fused += 1
    apply_replacements(gi, dead, new_nodes)
    return fused


def apply_replacements(gi: G, dead: set[int], new_nodes):
    if not dead:
        return
    replacement_at = {id(tail): node for tail, node in new_nodes}
    rebuilt = []
    for n in gi.g.node:
        if id(n) in replacement_at:
            rebuilt.append(replacement_at[id(n)])
        elif id(n) not in dead:
            rebuilt.append(n)
    del gi.g.node[:]
    gi.g.node.extend(rebuilt)
    gi.reindex()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    args = ap.parse_args()

    model = onnx.load(args.input)
    before = len(model.graph.node)
    gi = G(model)
    ln = fuse_layernorm(gi)
    rms = fuse_rmsnorm(gi)
    gelu = fuse_gelu(gi)
    lng = fuse_layernorm_generic(gi)
    silu = fuse_silu(gi)

    if not any(o.domain == MSDOMAIN for o in model.opset_import):
        model.opset_import.add(domain=MSDOMAIN, version=1)
    # standard LayerNormalization needs opset >= 17 (no signature changes
    # 14 -> 17 for any op in this graph)
    for o in model.opset_import:
        if o.domain in ("", "ai.onnx") and o.version < 17:
            o.version = 17

    after = len(model.graph.node)
    print(f"fused: {ln}+{lng} LayerNorm, {rms} RMSNorm, {gelu} Gelu, {silu} SiLU "
          f"({before} -> {after} nodes, -{before - after})")
    onnx.save(model, args.output)




# ── generic, numerically-validated LayerNorm matcher ─────────────────────────
# Exporters decompose norms several ways; instead of enumerating forms, take
# any Div whose denominator ends in Sqrt, reconstruct the candidate subgraph
# from x to the affine tail, execute it, and fuse only on exact agreement
# with LayerNormalization semantics.

def _ancestors_until(gi, out_name, stop):
    nodes, stack, seen = [], [out_name], set()
    ok = True
    while stack:
        t = stack.pop()
        if t in stop or t in gi.inits or t == "":
            continue
        p = gi.producers.get(t)
        if p is None:
            return None  # depends on a graph input other than x
        if id(p) in seen:
            continue
        seen.add(id(p))
        nodes.append(p)
        stack.extend(p.input)
    return nodes


def fuse_layernorm_generic(gi: G):
    import onnxruntime as ort_rt
    fused, dead, new_nodes = 0, set(), []
    for div in list(gi.g.node):
        if id(div) in dead or div.op_type != "Div":
            continue
        sqrt = gi.producers.get(div.input[1])
        if sqrt is None or sqrt.op_type != "Sqrt":
            continue
        # numerator must be x - mean(x): find x
        sub = gi.producers.get(div.input[0])
        if sub is None or sub.op_type != "Sub":
            continue
        x = sub.input[0]
        # affine tail: Mul(w) then Add(b), both initializers
        mul = gi.only_consumer(div.output[0], "Mul")
        if mul is None:
            continue
        w = mul.input[1] if mul.input[0] == div.output[0] else mul.input[0]
        addb = gi.only_consumer(mul.output[0], "Add")
        if addb is None or w not in gi.inits:
            continue
        b = addb.input[1] if addb.input[0] == mul.output[0] else addb.input[0]
        if b not in gi.inits:
            continue
        core = _ancestors_until(gi, addb.output[0], {x})
        if core is None or not (4 <= len(core) <= 16):
            continue
        if any(id(n) in dead for n in core):
            continue
        # find eps among small consts inside the core
        eps = None
        for n in core:
            if n.op_type == "Add":
                for i in n.input:
                    v = gi.const_scalar(i)
                    if v is not None and 0 < v <= 1e-3:
                        eps = v
        if eps is None:
            continue
        # numeric validation: run the extracted core vs numpy LayerNorm
        try:
            wa = numpy_helper.to_array(gi.inits[w]).astype(np.float32)
            ba = numpy_helper.to_array(gi.inits[b]).astype(np.float32)
            D = wa.shape[-1]
            xin = helper.make_tensor_value_info(x, TensorProto.FLOAT, [1, 3, D])
            xout = helper.make_tensor_value_info(addb.output[0], TensorProto.FLOAT, [1, 3, D])
            init_names = {t for n in core for t in n.input if t in gi.inits}
            sg = helper.make_graph(
                list(reversed(core)), "probe", [xin], [xout],
                [gi.inits[t] for t in init_names])
            sm = helper.make_model(sg, opset_imports=list(gi.model.opset_import))
            sm.ir_version = gi.model.ir_version
            sess = ort_rt.InferenceSession(sm.SerializeToString(),
                                           providers=["CPUExecutionProvider"])
            rng = np.random.default_rng(7)
            xa = rng.standard_normal((1, 3, D), dtype=np.float32) * 3
            got = sess.run(None, {x: xa})[0]
            mu = xa.mean(-1, keepdims=True)
            var = ((xa - mu) ** 2).mean(-1, keepdims=True)
            ref = (xa - mu) / np.sqrt(var + eps) * wa + ba
            if not np.allclose(got, ref, atol=2e-5):
                continue
        except Exception:
            continue
        ln = helper.make_node("LayerNormalization", [x, w, b], [addb.output[0]],
                              name=f"fused_lng_{fused}",
                              axis=-1, epsilon=float(eps))
        new_nodes.append((addb, ln))
        dead.update(id(n) for n in core)
        fused += 1
    apply_replacements(gi, dead, new_nodes)
    return fused


def fuse_silu(gi: G):
    """Mul(x, Sigmoid(x)) => com.microsoft QuickGelu(alpha=1)"""
    fused, dead, new_nodes = 0, set(), []
    for sg in list(gi.g.node):
        if id(sg) in dead or sg.op_type != "Sigmoid":
            continue
        x = sg.input[0]
        mul = gi.only_consumer(sg.output[0], "Mul")
        if mul is None or x not in mul.input:
            continue
        node = helper.make_node("QuickGelu", [x], [mul.output[0]],
                                name=f"fused_silu_{fused}", domain=MSDOMAIN,
                                alpha=1.0)
        new_nodes.append((mul, node))
        dead.update({id(sg), id(mul)})
        fused += 1
    apply_replacements(gi, dead, new_nodes)
    return fused

if __name__ == "__main__":
    main()
