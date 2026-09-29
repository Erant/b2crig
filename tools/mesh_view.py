"""Shaded views of a clip's posed cage mesh (painter's algorithm, no GPU) -> a contact sheet: is a defect in the rig
(the posed mesh) or in the splats' binding?

    .venv/bin/python tools/mesh_view.py CLIP_DIR FRAME OUT.png [--joint r_elbow] [--radius 0.8] [--az -90,0,90,180]
        [--el 0] [--size 400] [--layer 0]
"""
import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2crig import cameras as C  # noqa: E402
from b2crig import evaluate as E  # noqa: E402
from b2crig.rig.mhr import MHRBody  # noqa: E402
from b2crig.rig.skeleton import JOINT  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("clip", type=Path)
ap.add_argument("frame", type=int)
ap.add_argument("out", type=Path)
ap.add_argument("--cage", default="cage.b2ccage")
ap.add_argument("--joint", default="r_elbow")
ap.add_argument("--radius", type=float, default=0.8)
ap.add_argument("--az", default="-90,0,90,180")
ap.add_argument("--el", type=float, default=0.0)
ap.add_argument("--size", type=int, default=400)
ap.add_argument("--layer", type=int, default=0)
a = ap.parse_args()

cg = E.read_cage(a.clip / a.cage)
vo, vc, fo, fc, _ = cg.layers[a.layer]
F = cg.faces[fo:fo + fc] - vo
V = cg.posed[a.frame][vo:vo + vc].astype(np.float64)
body = MHRBody(a.clip.parent.parent / "mhr.npz", device="cpu")
j0 = body.pose().joints[0, JOINT[a.joint]].numpy()
nb = cg.layers[0][1]
near = np.argsort(np.linalg.norm(cg.verts[:nb] - j0, axis=1))[:30]
tgt = j0 + (cg.posed[a.frame][near] - cg.verts[near]).mean(0)
S = a.size; f = S * 1.6
cols = []
for az in [float(x) for x in a.az.split(",")]:
    el, azr = math.radians(a.el), math.radians(az)
    pos = tgt + a.radius * np.array([math.cos(el) * math.sin(azr), math.sin(el), math.cos(el) * math.cos(azr)])
    R = C.look_at(pos, tgt)   # columns right, up, back (camera -> world)
    q = (V - pos) @ R   # camera coords: x right, y up, z back (visible z < 0)
    z = -q[:, 2]
    u = f * q[:, 0] / np.maximum(z, 1e-6) + S / 2; v = -f * q[:, 1] / np.maximum(z, 1e-6) + S / 2
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    lam = np.abs(n @ (R[:, 2] * 0.7 + R[:, 1] * 0.5 + R[:, 0] * 0.3) / np.linalg.norm([0.7, 0.5, 0.3]))
    zf = z[F].mean(1)
    im = np.full((S, S, 3), 90, np.uint8)
    for k in np.argsort(-zf):
        if zf[k] <= 0.05:
            continue
        pts = np.stack([u[F[k]], v[F[k]]], 1).round().astype(np.int32)
        c = int(40 + 200 * lam[k])
        cv2.fillConvexPoly(im, pts, (c, c, c), lineType=cv2.LINE_AA)
    cols.append(im)
cv2.imwrite(str(a.out), np.hstack(cols))
print(f"mesh_view: {a.out}")
