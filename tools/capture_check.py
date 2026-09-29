"""Does a retrained splat still reproduce the subject's capture? Render every k-th capture view canonically with the old
and the new splat and compare with the capture images (L1 over the figure, and over the face via the capture's labels).

    .venv/bin/python tools/capture_check.py work/<subject> NEW_PLY [--every 8]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2crig import b2ctrain as B  # noqa: E402
from b2crig.visibility import qvec_to_R  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("subject", type=Path)
ap.add_argument("new", type=Path)
ap.add_argument("--every", type=int, default=8)
a = ap.parse_args()
S = a.subject; run = Path(json.loads((S.parent / "subjects.json").read_text())[S.name]); cap = run / "colmap"
p = [l.split() for l in (cap / "cameras.txt").read_text().splitlines() if l and not l.startswith("#")][0]
W, H, fx, fy, cx, cy = int(p[2]), int(p[3]), *map(float, p[4:8])
cams = []
for ln in (cap / "images.txt").read_text().splitlines():
    f = ln.split()
    if ln.startswith("#") or len(f) < 10:
        continue
    Rw2c = qvec_to_R(list(map(float, f[1:5]))); t = np.array(list(map(float, f[5:8])))
    cams.append({"name": Path(f[9]).stem, "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                 "rotation": (Rw2c.T @ np.diag([1., -1., -1.])).tolist(), "position": (-Rw2c.T @ t).tolist()})
cams = cams[::a.every]
out = a.new.parent / "capture_check"; out.mkdir(exist_ok=True)
(out / "cameras.json").write_text(json.dumps({"width": W, "height": H, "cameras": cams}))
res = {}
cage = S / "clips" / "capture" / "cage.b2ccage"   # a re-canonicalized subject sees its capture posed (tools/port_clips.py)
for k, ply in (("old", run / "ply" / "scene.ply"), ("new", a.new)):
    B.render(ply, out / "cameras.json", out / k, background=(0.5, 0.5, 0.5), cage=cage if cage.exists() else None)
    fig, face = [], []
    for c in cams:
        r = cv2.imread(str(out / k / f"{c['name']}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
        g = cv2.imread(str(cap / "images" / f"{c['name']}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
        rc = r[..., :3] * r[..., 3:] / 255 + 127 * (1 - r[..., 3:] / 255); gc = g[..., :3] * g[..., 3:] / 255 + 127 * (1 - g[..., 3:] / 255)
        e = np.abs(rc - gc).mean(-1)
        fig.append(e[g[..., 3] > 127].mean())
        lab = cap / "labels" / f"{c['name']}.png"
        if lab.exists():
            fm = np.isin(cv2.imread(str(lab), 0), (3, 24, 25))
            if fm.sum() > 200:
                face.append(e[fm].mean())
    res[k] = (np.mean(fig), np.mean(face) if face else np.nan)
print(f"capture views ({len(cams)}): L1 vs capture, figure / face: old {res['old'][0]:.2f} / {res['old'][1]:.2f}   "
      f"new {res['new'][0]:.2f} / {res['new'][1]:.2f}")
