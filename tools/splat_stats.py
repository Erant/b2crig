"""Splat hygiene: count, opaque splats buried inside the body, and floaters off every cage surface.

    .venv/bin/python tools/splat_stats.py work/<subject> PLY [PLY ...]
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2crig.io.splat import load  # noqa: E402
from b2crig.rig import layered  # noqa: E402
from b2crig.rig.cage import vertex_normals  # noqa: E402

S = Path(sys.argv[1]); d = np.load(S / "mhr.npz"); BV = d["verts_world"].astype(float); BN = vertex_normals(BV, d["faces"].astype(np.int64))
if not (S / "expr_motion.npy").exists():   # per-vertex motion of the jaw-open expression (tools/face_probe's table)
    import torch
    from b2crig.rig.mhr import MHRBody
    _b = MHRBody(S / "mhr.npz"); e = _b.expr.clone(); e[0, 24] += 1.0
    jaw = np.linalg.norm(_b.pose(expr=e).verts[0].cpu().numpy() - _b.pose().verts[0].cpu().numpy(), axis=1) * 1000
    tab = np.zeros((72, len(jaw)), np.float32); tab[24] = jaw; np.save(S / "expr_motion.npy", tab)
mot = np.load(S / "expr_motion.npy")[24]; mouth = cKDTree(BV[mot > 3])
allv = np.concatenate([BV] + [L["V"] for L in layered.load_layers(S)]); tb, ta = cKDTree(BV), cKDTree(allv)
for p in sys.argv[2:]:
    s = load(p); op = s.opacity > 0.2
    _, j = tb.query(s.xyz); sd = np.einsum("ij,ij->i", s.xyz - BV[j], BN[j]); dc, _ = ta.query(s.xyz); dm, _ = mouth.query(s.xyz)
    deep = op & (sd < -0.02) & (dm > 0.045)
    print(f"{Path(p).parent.name}/{Path(p).name}: {len(s.xyz)} splats, opaque {op.sum()}; opaque >2 cm inside the body (not mouth) "
          f"{deep.sum()} (>4 cm {(deep & (sd < -0.04)).sum()}); floaters >3 cm off the cage {(op & (dc > 0.03)).sum()} (>5 cm {(op & (dc > 0.05)).sum()})")
