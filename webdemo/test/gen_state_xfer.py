#!/usr/bin/env python3
"""Export post-conditioning states + one-step reference for cross-runtime check."""
import json, struct, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from verify_model_equivalence import StateRunner, make_session
import sentencepiece as spm

WEB = ROOT / "webdemo"
def load_emb(p):
    b = p.read_bytes()
    magic, ndims = struct.unpack("<II", b[:8])
    shape = struct.unpack(f"<{ndims}q", b[8:8+ndims*8])
    return np.frombuffer(b, np.float32, offset=8+ndims*8).reshape(shape)

emb = load_emb(WEB/"presets"/"alba.emb")
sp = spm.SentencePieceProcessor(model_file=str(WEB/"models"/"tokenizer.model"))
ids = sp.encode("How are things over in Riverwood these days?")
r = StateRunner(make_session(WEB/"models"/"flow_lm_main_delta_flow_int8.onnx", ROOT/"x.dylib"))
txt = make_session(WEB/"models"/"text_conditioner.onnx", ROOT/"x.dylib")
temb = txt.run(None, {txt.get_inputs()[0].name: np.array([ids], np.int64)})[0]
if temb.ndim == 2: temb = temb[None]
z = {"sequence": np.zeros((1,0,32), np.float32), "flow_x": np.zeros((1,32), np.float32)}
r.run({**z, "text_embeddings": emb.astype(np.float32)})
r.run({**z, "text_embeddings": temb.astype(np.float32)})

# export states
meta, blobs, off = [], [], 0
for name in r.state_names:
    a = r.states[name]
    raw = a.tobytes()
    meta.append({"name": name, "dtype": str(a.dtype), "dims": list(a.shape), "offset": off, "bytes": len(raw)})
    blobs.append(raw); off += len(raw)
(WEB/"test"/"xfer_states.bin").write_bytes(b"".join(blobs))
# one AR step from this state
cl = np.full((1,1,32), np.nan, np.float32)
out = r.run({"sequence": cl, "text_embeddings": np.zeros((1,0,1024), np.float32), "flow_x": np.zeros((1,32), np.float32)})
json.dump({"states": meta,
           "latent": [float(x) for x in out["latent"].reshape(-1)],
           "eos": float(out["eos_logit"].reshape(-1)[0])},
          open(WEB/"test"/"xfer_meta.json", "w"))
print("exported", len(meta), "states; step-0 eos:", float(out["eos_logit"].reshape(-1)[0]))
