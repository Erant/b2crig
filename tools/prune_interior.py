"""Remove the opaque splats buried inside the body (a shell behind the visible surface that no view ever sees, so
training never removes it), keeping the mouth cavity (teeth and tongue live inside the head, shown when the jaw opens).

    .venv/bin/python tools/prune_interior.py work/<subject> IN_PLY OUT_PLY [--depth 0.02] [--mouth-radius 0.045]
                                              [--evidence EV_PLY --max-ev 0.1 --float 0.03]

With --evidence (IN_PLY with ev_* from `b2ctrain render --dataset <capture> --confidence --write-evidence`), only the
splats the capture views never see (ev_w_all < max-ev) go: inside the body, or floating more than --float from every
cage surface. Without it the depth test alone decides, which also takes visible surface where the clothed figure is
slimmer than the MHR body.

Inside = signed distance to the canonical MHR body surface (nearest vertex, its normal) below -depth. Mouth cavity =
within mouth-radius of the body vertices the jaw-open expression (e24) moves by more than 3 mm.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2crig.io.splat import load, save_subset  # noqa: E402
from b2crig.rig.cage import vertex_normals  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("subject", type=Path)
ap.add_argument("inp", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--depth", type=float, default=0.02)
ap.add_argument("--mouth-radius", type=float, default=0.045)
ap.add_argument("--evidence", type=Path)
ap.add_argument("--max-ev", type=float, default=0.1)
ap.add_argument("--float", type=float, default=0.03)
a = ap.parse_args()
d = np.load(a.subject / "mhr.npz"); BV = d["verts_world"].astype(float); BF = d["faces"].astype(np.int64)
BN = vertex_normals(BV, BF)
s = load(a.inp)
_, j = cKDTree(BV).query(s.xyz)
sd = np.einsum("ij,ij->i", s.xyz - BV[j], BN[j])
mot = np.load(a.subject / "expr_motion.npy")[24]                      # mm at jaw-open +1
mouth_v = BV[mot > 3.0]
dm, _ = cKDTree(mouth_v).query(s.xyz)
drop = (sd < -a.depth) & (dm > a.mouth_radius)
if a.evidence:
    from plyfile import PlyData
    from b2crig.rig import layered
    ev = np.asarray(PlyData.read(str(a.evidence))["vertex"]["ev_w_all"])
    assert len(ev) == len(sd), "the evidence ply has to be IN_PLY with ev_* added"
    allv = np.concatenate([BV] + [L["V"] for L in layered.load_layers(a.subject)])
    dc, _ = cKDTree(allv).query(s.xyz)
    drop = (ev < a.max_ev) & (dm > a.mouth_radius) & ((sd < -a.depth) | (dc > a.float))
keep = np.nonzero(~drop)[0]
save_subset(a.inp, keep, a.out)
print(f"prune_interior: dropped {drop.sum()} of {len(drop)} splats ({(drop & (s.opacity > 0.2)).sum()} opaque); "
      f"-> {a.out}")
