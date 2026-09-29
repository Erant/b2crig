"""Render the subject's splat through expression-deformed cages: one close-up frontal frame per MHR expression
dimension, to name the dimensions and to see how the face splats follow.

    .venv/bin/python tools/face_probe.py work/<subject> [--dims 24 56 ...] [--amp 1.0] [--out DIR]

Writes <out>/grid.png (neutral first, then each dimension at +amp) and the cage/cameras used.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2crig import b2ctrain as B  # noqa: E402
from b2crig import cameras as C  # noqa: E402
from b2crig.model import data as D  # noqa: E402
from b2crig.rig import layered  # noqa: E402
from b2crig.rig.mhr import MHRBody  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("subject", type=Path)
ap.add_argument("--dims", type=int, nargs="*")
ap.add_argument("--amp", type=float, default=1.0)
ap.add_argument("--out", type=Path)
ap.add_argument("--azimuth", type=float, default=0.0)
a = ap.parse_args()
S = a.subject; out = a.out or S / "face_probe"; out.mkdir(parents=True, exist_ok=True)
run = Path(json.loads((S.parent / "subjects.json").read_text())[S.name])
body = MHRBody(S / "mhr.npz"); lay = layered.load_layers(S)
if a.dims is None:
    d = np.load(S / "expr_motion.npy") if (S / "expr_motion.npy").exists() else None
    a.dims = list(np.argsort(-d.max(1))[:23]) if d is not None else list(range(23))
E = body.expr.repeat(len(a.dims) + 1, 1).clone()
for k, i in enumerate(a.dims):
    E[k + 1, i] += a.amp
P = body.pose(body.body0.expand(len(E), -1), expr=E)
posed = layered.pose_layers(body, P, lay).cpu().numpy()
names = ["neutral"] + [f"e{i:02d}" for i in a.dims]
B.write_cage(out / "cage.b2ccage", layered.cage_layers(body, lay), posed, names)
# close-up camera on the face: target between the eyes-ish (head joint, raised), 0.45 m away
head = P.joints[0, D.JOINT["head"]].cpu().numpy() + np.array([0.0, 0.05, 0.0])
K = C.Intrinsics(512, 512, 1100.0, 1100.0, 256.0, 256.0)
poses = C.circular_orbit(head, 0.45, 0.0, 1, a.azimuth, 0.0) * len(names)
C.write_cameras(out / "cameras.json", K, poses, names)
B.render(run / "ply" / "scene.ply", out / "cameras.json", out / "render", cage=out / "cage.b2ccage", background=(0.5, 0.5, 0.5))
tiles = []
for nm in names:
    im = cv2.imread(str(out / "render" / f"{nm}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
    im = (im[..., :3] * im[..., 3:] / 255 + 127 * (1 - im[..., 3:] / 255)).astype(np.uint8)
    im = cv2.resize(im, (200, 200)); cv2.putText(im, nm, (4, 16), 0, 0.5, (255, 255, 255), 1)
    tiles.append(im)
while len(tiles) % 6:
    tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(str(out / "grid.png"), np.vstack([np.hstack(tiles[r:r + 6]) for r in range(0, len(tiles), 6)]))
print(f"face_probe: {len(names)} frames -> {out / 'grid.png'}")
