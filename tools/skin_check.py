"""Does the fitted displacement grow with joint articulation (a skinning/binding error signature)?

    .venv/bin/python tools/skin_check.py work/<subject> CLIP [CLIP ...]

Per visible vertex-frame: |fitted displacement| (the clip's layered fit) against the articulation angle of the
vertex's dominant named joint (its rotation relative to its parent, measured from the canonical pose). Folds and
appearance fixes should not care much about the angle; mis-skinned or mis-bound surface should grow with it.
Reported per angle bin, for garment/hair layers and the body layer, and for vertices whose joint never moves.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2crig.evaluate import read_cage  # noqa: E402
from b2crig.model import data as D  # noqa: E402
from b2crig.rig.mhr import MHRBody  # noqa: E402

S = Path(sys.argv[1]); names = sys.argv[2:]
body = MHRBody(S / "mhr.npz"); bi = D.cage_body_index(body, S)
NB = len(body.verts_canon)
Wn = D.named_influence(body, bi); dom = Wn.argmax(1)
R0 = D.relative_rotations(body.pose())[0]            # the subject's canonical pose
bins = [0, 5, 10, 20, 30, 45, 90]
acc = {k: [[] for _ in bins[:-1]] for k in ("layers", "body")}
still = {"layers": [], "body": []}
for n in names:
    clip = S / "clips" / n
    fit, cage = D.layered_pair(clip)
    T, V = read_cage(clip / cage).posed.shape[:2]
    d = np.linalg.norm(np.fromfile(clip / fit / "delta.f32", np.float32).reshape(T, V, 3), axis=-1) * 1000
    vis = np.fromfile(clip / fit / "vis.f32", np.float32).reshape(T, V) > 1.0
    p = torch.as_tensor(np.load(clip / "motion.npz")["body_params"], device=body.device)
    Rr = np.concatenate([D.relative_rotations(body.pose(p[i:i + 16])) for i in range(0, T, 16)])
    rel = np.einsum("tjab,jcb->tjac", Rr, R0)                            # R(t) R0^T
    ang = np.degrees(np.arccos(np.clip((np.trace(rel, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))   # [T, J]
    moves = ang.max(0) > 5
    va = ang[:, dom]                                                       # [T, V] angle of each vertex's joint
    for part, sl in (("body", slice(0, NB)), ("layers", slice(NB, None))):
        m, a, dd = vis[:, sl], va[:, sl], d[:, sl]
        mv = moves[dom[sl]][None].repeat(T, 0)
        still[part].append(dd[m & ~mv])
        for b in range(len(bins) - 1):
            k = m & mv & (a >= bins[b]) & (a < bins[b + 1])
            acc[part][b].append(dd[k])
print(f"clips: {', '.join(names)}")
for part in ("layers", "body"):
    s = np.concatenate(still[part])
    print(f"{part:6s} joint never moves: median |d| {np.median(s):.2f} mm (n={len(s)})")
    for b in range(len(bins) - 1):
        x = np.concatenate(acc[part][b])
        if len(x) > 1000:
            print(f"{part:6s} angle {bins[b]:2d}-{bins[b + 1]:2d} deg: median |d| {np.median(x):.2f} mm, p90 {np.percentile(x, 90):.2f} (n={len(x)})")
