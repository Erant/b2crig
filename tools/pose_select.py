"""Pick a small pose set that exercises the rig as much as a big pool does, for b2ctrain's pose containment loss
(tools/cage_train.py --contain poseset_<tag>).

    .venv/bin/python tools/pose_select.py work/<subject> CLIP[:every] ... --k 16 [--tag T] [--curve 4,8,16,32,64]

Every pool frame deforms each body-layer triangle by some amount d = max |log principal stretch| against the canonical
triangle (the shear/stretch that exposes the interior). Greedy max coverage picks frames one at a time to maximise
sum_t area_t * max_{f in set} d_t(f): each body region gets the most extreme deformation the pool has for it, and a
frame that repeats what the set already covers adds nothing. Coverage is reported as a fraction of the whole pool's.
The hands are ~4% of the body's area, so by area alone finger poses would never be picked: `--hand-share` (default
0.15) scales the hand triangles (rig/contain.hand_vertices) to that share of the weight; body and hand coverage are
reported apart.
Writes <subject>/clips/poseset_<tag>/cage.b2ccage (frames named <clip>__<frame>) and poses.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2crig import evaluate  # noqa: E402
from b2crig.rig.contain import hand_vertices  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("subject", type=Path)
ap.add_argument("clips", nargs="+")
ap.add_argument("--k", type=int, default=16)
ap.add_argument("--every", type=int, default=3)
ap.add_argument("--tag", default="")
ap.add_argument("--curve", default="4,8,16,32,64")
ap.add_argument("--hand-share", type=float, default=0.15, help="share of the coverage weight on the hand triangles (0: by area)")
a = ap.parse_args()
S = a.subject


def tri_frames(V, F):   # [..., T, 2, 2] edge matrix of each triangle in its own orthonormal in-plane frame
    e1, e2 = V[..., F[:, 1], :] - V[..., F[:, 0], :], V[..., F[:, 2], :] - V[..., F[:, 0], :]
    u = e1 / np.linalg.norm(e1, axis=-1, keepdims=True)
    n = np.cross(e1, e2); n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    w = np.cross(n, u)
    return np.stack([np.stack([(e1 * u).sum(-1), (e2 * u).sum(-1)], -1), np.stack([(e1 * w).sum(-1), (e2 * w).sum(-1)], -1)], -2)


cg0 = None; names, posed, D = [], [], []
for spec in a.clips:
    c, every = spec.split(":")[0], int(spec.split(":")[1]) if ":" in spec else a.every
    cgp = S / "clips" / c / ("cage_v2.b2ccage" if (S / "clips" / c / "cage_v2.b2ccage").exists() else "cage.b2ccage")
    cg = evaluate.read_cage(cgp)
    if cg0 is None:
        cg0 = cg; v_off, v_cnt, f_off, f_cnt, _ = cg.layers[0]
        BF = cg.faces[f_off:f_off + f_cnt] - v_off
        E0 = tri_frames(cg.verts[v_off:v_off + v_cnt].astype(np.float64), BF)
        ok = np.abs(np.linalg.det(E0)) > 1e-12; E0inv = np.linalg.inv(np.where(ok[:, None, None], E0, np.eye(2)))
        area = 0.5 * np.abs(np.linalg.det(E0)) * ok
    assert len(cg.verts) == len(cg0.verts), f"{c}: cage layout differs"
    for fi in range(0, len(cg.names), every):
        P = cg.posed[fi][v_off:v_off + v_cnt].astype(np.float64)
        s = np.linalg.svd(tri_frames(P, BF) @ E0inv, compute_uv=False)
        D.append(np.abs(np.log(np.clip(s, 1e-3, None))).max(-1).astype(np.float32))
        names.append(f"{c}__{cg.names[fi]}"); posed.append(cg.posed[fi])
D = np.stack(D)   # [frames, T]
hv = hand_vertices(S / "mhr.npz")
hand = np.isin(BF, np.concatenate([hv["r"], hv["l"]])).all(1)
if a.hand_share > 0 and hand.any():
    w_h = a.hand_share / (1 - a.hand_share) * area[~hand].sum() / area[hand].sum()
    print(f"hands: {hand.sum()} triangles, {area[hand].sum() / area.sum() * 100:.1f}% of the area, weighted x{w_h:.1f} "
          f"to {a.hand_share:.0%} of the coverage")
    area = np.where(hand, area * w_h, area)
full = (area * D.max(0)).sum()
cov = lambda cur, m: (area[m] * cur[m]).sum() / max((area[m] * D.max(0)[m]).sum(), 1e-12)
print(f"pool: {len(D)} frames from {len(a.clips)} clips, {D.shape[1]} body triangles; median per-triangle max stretch "
      f"{np.exp(np.median(D.max(0))):.2f}x, frame means {np.exp(D.mean(1).min()):.3f}-{np.exp(D.mean(1).max()):.3f}x")
cur = np.zeros(D.shape[1], np.float32); sel = []
kmax = max([a.k] + [int(x) for x in a.curve.split(",") if x])
for k in range(min(kmax, len(D))):
    gain = (area * np.maximum(D - cur, 0)).sum(1)
    j = int(gain.argmax()); sel.append(j); cur = np.maximum(cur, D[j])
    if str(k + 1) in a.curve.split(",") or k + 1 == a.k:
        print(f"  k={k + 1:3d}: coverage {(area * cur).sum() / full:.3f} (body {cov(cur, ~hand):.3f}, hands {cov(cur, hand):.3f})")
sel = sel[:a.k]
print("picked: " + ", ".join(names[j] for j in sel))
out = S / "clips" / f"poseset_{a.tag or a.k}"; out.mkdir(parents=True, exist_ok=True)
evaluate.write_cage_like(evaluate.Cage(cg0.layers, cg0.verts, cg0.faces, [names[j] for j in sel], None), out / "cage.b2ccage",
                         np.stack([posed[j] for j in sel]))
(out / "poses.json").write_text(json.dumps({"pool": a.clips, "every": a.every, "frames": [names[j] for j in sel],
                                            "coverage": float((area * cur).sum() / full), "hand_share": a.hand_share,
                                            "coverage_body": float(cov(cur, ~hand)), "coverage_hands": float(cov(cur, hand))}, indent=1))
print(f"pose_select: {out / 'cage.b2ccage'}")
