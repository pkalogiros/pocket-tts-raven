#!/usr/bin/env python3
"""Lockstep-compare two stateful PocketTTS ONNX models on identical inputs.

Runs both models for N sequential chunks, feeding states back exactly like the
C++ runtime does (full-state outputs replace, delta-KV outputs scatter into a
persistent cache with circular wraparound, bool states init to true), and
reports the max abs difference across every non-state output and every state
at every step. Timing-independent, unlike CLI WAV comparison (the streaming
crossfade consumes 120 samples per decode-batch boundary, so emitted WAV
length varies with thread timing).

    uv run --no-project --with onnx --with onnxruntime python \
        verify_model_equivalence.py A.onnx B.onnx --steps 4 --frames 3 2 1 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

CAPACITY = 1000


def make_session(path: Path, ops_lib: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    if ops_lib.exists():
        so.register_custom_ops_library(str(ops_lib))
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


class StateRunner:
    """Mirrors StateBufferIO/StatefulRunner state feedback in numpy."""

    def __init__(self, sess: ort.InferenceSession):
        self.sess = sess
        self.states: dict[str, np.ndarray] = {}
        self.state_names: list[str] = []
        self.non_state: list[str] = []
        self.full_shapes: dict[str, list[int]] = {}
        for i in sess.get_inputs():
            if not i.name.startswith("state_"):
                self.non_state.append(i.name)
                continue
            self.state_names.append(i.name)
            shape = [d if isinstance(d, int) and d > 0 else 0 for d in i.shape]
            self.full_shapes[i.name] = shape
            numel_shape = [max(d, 1) for d in shape] if all(shape) else shape
            if i.type == "tensor(int64)":
                self.states[i.name] = np.zeros(numel_shape, np.int64)
            elif i.type == "tensor(bool)":
                self.states[i.name] = np.ones(numel_shape, bool)
            elif i.type == "tensor(float16)":
                self.states[i.name] = np.zeros(numel_shape, np.float16)
            else:
                if not all(shape):  # dynamic: start empty on the dynamic axis
                    numel_shape = [d if d else 0 for d in shape]
                self.states[i.name] = np.zeros(numel_shape, np.float32)

    def run(self, feeds: dict[str, np.ndarray]):
        all_feeds = dict(feeds)
        all_feeds.update(self.states)
        out_names = [o.name for o in self.sess.get_outputs()]
        outs = self.sess.run(None, all_feeds)
        result = {}
        old_states = self.states
        new_states = dict(old_states)
        for name, val in zip(out_names, outs):
            if not name.startswith("out_state_"):
                result[name] = val
                continue
            sname = name[4:]
            full = self.full_shapes[sname]
            cur = old_states[sname]
            if val.dtype == np.float32 and list(val.shape) != list(cur.shape) \
                    and len(full) == 5 and CAPACITY in full:
                # delta-KV output: scatter into persistent cache at old step
                axis = full.index(CAPACITY)
                idx = self.state_names.index(sname)
                step_name = None
                for j in (1, 2):
                    if idx + j < len(self.state_names):
                        cand = self.state_names[idx + j]
                        if old_states[cand].dtype == np.int64 and old_states[cand].size == 1:
                            step_name = cand
                            break
                assert step_name, f"no step counter for {sname}"
                old_step = int(old_states[step_name].reshape(-1)[0])
                L = val.shape[axis]
                dst = cur.copy()
                pos = [(old_step + t) % CAPACITY for t in range(L)]
                sl_dst = [slice(None)] * 5
                sl_src = [slice(None)] * 5
                for t, p in enumerate(pos):
                    sl_dst[axis] = p
                    sl_src[axis] = t
                    dst[tuple(sl_dst)] = val[tuple(sl_src)]
                new_states[sname] = dst
            else:
                new_states[sname] = val
        self.states = new_states
        return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_a", type=Path)
    ap.add_argument("model_b", type=Path)
    ap.add_argument("--ops-lib", type=Path,
                    default=Path(__file__).parent / "libptt_custom_ops.dylib")
    ap.add_argument("--frames", type=int, nargs="+", default=[1, 3, 15, 7],
                    help="latent frames per sequential chunk")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    a = StateRunner(make_session(args.model_a, args.ops_lib))
    b = StateRunner(make_session(args.model_b, args.ops_lib))
    if a.non_state != b.non_state:
        sys.exit(f"non-state inputs differ: {a.non_state} vs {b.non_state}")

    in_meta = {i.name: i for i in a.sess.get_inputs()}
    worst = 0.0
    for step, t in enumerate(args.frames):
        feeds = {}
        for name in a.non_state:
            shape = [d if isinstance(d, int) and d > 0 else t
                     for d in in_meta[name].shape]
            feeds[name] = rng.standard_normal(shape).astype(np.float32) * 0.5
        ra = a.run(feeds)
        rb = b.run(dict(feeds))
        for name in ra:
            d = float(np.max(np.abs(ra[name].astype(np.float64) - rb[name].astype(np.float64)))) if ra[name].size else 0.0
            worst = max(worst, d)
            print(f"step {step} T={t} out {name}: shape={ra[name].shape} max|Δ|={d:.3e}")
        for sname in a.state_names:
            sa, sb = a.states[sname], b.states[sname]
            if sa.shape != sb.shape:
                sys.exit(f"step {step}: state {sname} shape {sa.shape} vs {sb.shape}")
            if sa.size:
                d = float(np.max(np.abs(sa.astype(np.float64) - sb.astype(np.float64))))
                worst = max(worst, d)
                if d > args.tol:
                    print(f"step {step} state {sname}: max|Δ|={d:.3e}")

    print(f"\nworst abs diff anywhere: {worst:.3e}  tol={args.tol}")
    sys.exit(0 if worst <= args.tol else f"FAILED tol {args.tol}")


if __name__ == "__main__":
    main()
