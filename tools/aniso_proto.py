"""Prototype: would ANISOTROPIC splat deformation (covariance A S A^T with the triangle's in-plane deformation
gradient, clamped per axis) fill the translucent shell a shearing triangle leaves? b2ctrain's cage only rotates and
scales splats isotropically by sqrt(area ratio). For ONE frame of a clip, pre-deform the canonical splats so that
b2ctrain's rotation + isotropic scale reproduces A S A^T, then render as usual.

    .venv/bin/python tools/aniso_proto.py work/<subject> SPLAT CLIP FRAME OUT.ply [--gmax 2.0]

Binding is approximated (nearest triangle centroid within the splat's layer; b2ctrain's own binding differs a little).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2crig import b2ctrain as B  # noqa: E402
from b2crig import evaluate as E  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("subject", type=Path)
ap.add_argument("splat", type=Path)
ap.add_argument("clip", type=Path)
ap.add_argument("frame", type=int)
ap.add_argument("out", type=Path)
ap.add_argument("--gmax", type=float, default=2.0)
ap.add_argument("--cage", default="cage.b2ccage")
a = ap.parse_args()

cg = E.read_cage(a.clip / a.cage)
V0, P = cg.verts.astype(np.float64), cg.posed[a.frame].astype(np.float64)
ply = PlyData.read(str(a.splat)); vx = ply["vertex"]
xyz = np.stack([vx["x"], vx["y"], vx["z"]], 1).astype(np.float64)
lab = np.asarray(vx["seg_label"]).astype(int)


def frames(V, F):
    u, w = V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]
    nn = np.cross(u, w); e1 = u / np.linalg.norm(u, axis=1, keepdims=True); n = nn / np.linalg.norm(nn, axis=1, keepdims=True)
    e2 = np.cross(n, e1)
    R = np.stack([e1, e2, n], 2)   # columns
    uv = np.stack([np.stack([(u * e1).sum(1), (u * e2).sum(1)], 1), np.stack([(w * e1).sum(1), (w * e2).sum(1)], 1)], 2)   # 2x2, cols = edges
    return R, uv, np.sqrt(np.linalg.norm(nn, axis=1))


U_all = np.tile(np.eye(3), (len(xyz), 1, 1))
bound = np.zeros(len(xyz), bool)
owned = set()
for (vo, vc, fo, fc, bits) in cg.layers[1:]:
    owned |= {k for k in range(32) if bits & (1 << k)}
for li, (vo, vc, fo, fc, bits) in enumerate(cg.layers):
    F = cg.faces[fo:fo + fc].astype(np.int64)
    cls = [k for k in range(32) if bits & (1 << k)]
    sel = ~np.isin(lab, list(owned)) if li == 0 else np.isin(lab, cls)
    R0, E0, k0 = frames(V0, F); R1, E1, k1 = frames(P, F)
    M2 = E1 @ np.linalg.inv(E0)   # in-plane deformation in frame coordinates
    Uu, s, Vt = np.linalg.svd(M2)
    s = np.clip(s, 1 / a.gmax, a.gmax)
    M2 = Uu @ (s[:, :, None] * Vt)
    kc = np.clip(k1 / k0, 1 / B.CAGE_MAX_GROWTH, B.CAGE_MAX_GROWTH)   # what b2ctrain applies isotropically
    M = np.zeros((len(F), 3, 3)); M[:, :2, :2] = M2; M[:, 2, 2] = kc   # the normal keeps b2ctrain's scaling
    # U = R0 (M / kc) R0^T, then b2ctrain's R R0^T kc U S U^T = R M R0^T S ...
    U = R0 @ (M / kc[:, None, None]) @ R0.transpose(0, 2, 1)
    cen = V0[F].mean(1)
    idx = np.nonzero(sel)[0]
    _, fi = cKDTree(cen).query(xyz[idx])
    U_all[idx] = U[fi]; bound[idx] = True

q = np.stack([vx["rot_0"], vx["rot_1"], vx["rot_2"], vx["rot_3"]], 1).astype(np.float64)   # w x y z
S = np.exp(np.stack([vx["scale_0"], vx["scale_1"], vx["scale_2"]], 1).astype(np.float64))
Rm = Rotation.from_quat(np.concatenate([q[:, 1:], q[:, :1]], 1)).as_matrix()
C = Rm @ (S[:, :, None] ** 2 * np.eye(3)) @ Rm.transpose(0, 2, 1)
C2 = U_all @ C @ U_all.transpose(0, 2, 1)
w, Q = np.linalg.eigh(C2)
Q[np.linalg.det(Q) < 0, :, 2] *= -1
qn = Rotation.from_matrix(Q).as_quat()   # x y z w
d = vx.data.copy()
for k, c in enumerate(("rot_1", "rot_2", "rot_3")):
    d[c] = qn[:, k]
d["rot_0"] = qn[:, 3]
for k in range(3):
    d[f"scale_{k}"] = 0.5 * np.log(np.maximum(w[:, k], 1e-20))
from plyfile import PlyElement  # noqa: E402
PlyData([PlyElement.describe(d, "vertex")], text=False, comments=ply.comments).write(str(a.out))
print(f"aniso_proto: {bound.sum()} splats re-shaped for frame {a.frame} -> {a.out}")
