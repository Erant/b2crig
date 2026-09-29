"""Split fitted cage displacements into along-view-ray and image-plane parts; compare clips per part.

    .venv/bin/python tools/view_decomp.py work/<subject> clipA clipB [...]

Monocular fits barely constrain displacement along each frame's camera ray. For a pair of clips it reports,
on co-visible vertices: the share of |d|^2 along the ray, and the correlation of (a) the full world displacement,
(b) the image-plane parts, (c) the component along n = rA x rB, which lies in both image planes (views only).
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2crig.evaluate import read_cage  # noqa: E402
from b2crig.model import data as D  # noqa: E402

S = Path(sys.argv[1]); names = sys.argv[2:]


def load(n):
    clip = S / "clips" / n
    fit, cage = D.layered_pair(clip)
    c = read_cage(clip / cage)
    T, V = c.posed.shape[:2]
    d = np.fromfile(clip / fit / "delta.f32", np.float32).reshape(T, V, 3)
    w = np.minimum(np.fromfile(clip / fit / "vis.f32", np.float32).reshape(T, V), 3.0) / 3.0
    cam = np.array([x["position"] for x in json.loads((clip / "cameras.json").read_text())["cameras"]])
    r = c.posed - cam[:, None]; r /= np.linalg.norm(r, axis=-1, keepdims=True)
    return d, w, r


def corr(a, b):
    return np.corrcoef(a.ravel(), b.ravel())[0, 1]


cl = {n: load(n) for n in names}
for n, (d, w, r) in cl.items():
    m = w > 0.3; along = (d * r).sum(-1)
    print(f"{n}: along-ray share of |d|^2 on visible verts {np.sum(along[m]**2) / np.sum(d[m]**2):.2f}")
for a, b in combinations(names, 2):
    (da, wa, ra), (db, wb, rb) = cl[a], cl[b]
    m = np.minimum(wa, wb) > 0.3
    pa = da - (da * ra).sum(-1, keepdims=True) * ra; pb = db - (db * rb).sum(-1, keepdims=True) * rb
    nrm = np.cross(ra, rb); s = np.linalg.norm(nrm, axis=-1)
    out = f"{a} vs {b}: full {corr(da[m], db[m]):.2f}, image-plane {corr(pa[m], pb[m]):.2f}"
    k = m & (s > 0.5)
    if k.mean() > 0.01:
        nn = nrm[k] / s[k][:, None]
        out += f", common-axis {corr((da[k] * nn).sum(-1), (db[k] * nn).sum(-1)):.2f} (n={k.sum()})"
    print(out)
