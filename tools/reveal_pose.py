"""Find a held pose that exposes the most body surface the subject's original capture never saw.

    .venv/bin/python tools/reveal_pose.py work/<subject> [--evals 400] [--out work/<subject>/reveal]

Seen = body vertices (canonical pose) visible in >= MIN_VIEWS of the capture's training cameras (colmap/images.txt,
the splat's own frame). A candidate pose (10 DOF: shoulders x/y/z, elbows z, hips x/z) is scored by the vertex area,
unseen before, that a helical held-pose orbit (elevations -10..35) sees in >= MIN_VIEWS views, minus a small penalty
on how far the pose is from canonical. Visibility: per-camera z-buffer of vertices splatted as 3x3 px discs at ~220 px image height.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2crig import cameras as C  # noqa: E402
from b2crig.gen import clips  # noqa: E402,F401
from b2crig.motion.procedural import AXES, RANDOM_HANDLES  # noqa: E402
from b2crig.rig.mhr import MHRBody  # noqa: E402
from b2crig.rig.skeleton import ROT, body_index  # noqa: E402

MIN_VIEWS = 3
DOF = [("r_shoulder", "x"), ("r_shoulder", "y"), ("r_shoulder", "z"), ("r_elbow", "z"),
       ("l_shoulder", "x"), ("l_shoulder", "y"), ("l_shoulder", "z"), ("l_elbow", "z"),
       ("r_hip", "z"), ("l_hip", "z")]
LIM = np.array([1.5 * RANDOM_HANDLES.get(d, 0.3) for d in DOF])


def qvec_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def colmap_cams(ds: Path):
    cam = open(ds / "cameras.txt").read().split("\n")
    p = [ln.split() for ln in cam if ln and not ln.startswith("#")][0]
    W, H, fx, fy, cx, cy = int(p[2]), int(p[3]), *map(float, p[4:8])
    out = []
    for ln in (ds / "images.txt").read_text().splitlines():
        f = ln.split()
        if ln.startswith("#") or len(f) < 10:
            continue
        Rw2c = qvec_to_R(list(map(float, f[1:5]))); t = np.array(list(map(float, f[5:8])))
        out.append((Rw2c, t))
    return (W, H, fx, fy, cx, cy), out


def visible_counts(V: np.ndarray, intr, cams, px: int = 1, target_h: int = 220) -> np.ndarray:
    """Views seeing each vertex. The z-buffer is built at ~target_h px tall, where the vertices tile the body without
    holes (at full resolution occluded vertices show through the gaps)."""
    W, H, fx, fy, cx, cy = intr
    sc = target_h / H
    W, H, fx, fy, cx, cy = int(W * sc), int(H * sc), fx * sc, fy * sc, cx * sc, cy * sc
    cnt = np.zeros(len(V), np.int32)
    for Rw2c, t in cams:
        q = V @ Rw2c.T + t
        z = q[:, 2]; ok = z > 0.05
        u = np.round(fx * q[:, 0] / z + cx).astype(int); v = np.round(fy * q[:, 1] / z + cy).astype(int)
        ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
        zb = np.full((H + 2 * px, W + 2 * px), np.inf)
        for du in range(-px, px + 1):
            for dv in range(-px, px + 1):
                np.minimum.at(zb, (v[ok] + dv + px, u[ok] + du + px), z[ok])
        vis = np.zeros(len(V), bool)
        vis[ok] = z[ok] <= zb[v[ok] + px, u[ok] + px] + 0.01
        cnt += vis
    return cnt


def vertex_area(V, F):
    a = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    out = np.zeros(len(V)); np.add.at(out, F.ravel(), np.repeat(a / 3, 3)); return out


def helical(target, radius, K, n=48):
    """(Rw2c, t) for a helical orbit: one turn, elevation -10..35 deg, cameras in b2crig's convention."""
    out = []
    for i in range(n):
        el = -10 + 45 * i / (n - 1); az = 360 * i / n
        (R, pos), = C.circular_orbit(target, radius, el, 1, az, 0.0)
        c2w_cv = np.asarray(R) @ np.diag([1.0, -1.0, -1.0]); Rw2c = c2w_cv.T
        out.append((Rw2c, -Rw2c @ pos))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("subject", type=Path)
    ap.add_argument("--evals", type=int, default=400)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    S = a.subject; out = a.out or S / "reveal"; out.mkdir(parents=True, exist_ok=True)
    run = Path(json.loads((S.parent / "subjects.json").read_text())[S.name])
    body = MHRBody(S / "mhr.npz"); F = np.load(S / "mhr.npz")["faces"].astype(np.int64)
    V0 = body.pose().verts[0].cpu().numpy()
    intr, cams = colmap_cams(run / "colmap")
    seen = visible_counts(V0, intr, cams) >= MIN_VIEWS
    area = vertex_area(V0, F)
    print(f"capture: {len(cams)} cameras; unseen body area {area[~seen].sum() * 1e4:.0f} cm^2 of {area.sum() * 1e4:.0f} "
          f"({(~seen).mean() * 100:.1f}% of vertices)")
    # orbit geometry of the clips (a clip's cameras.json holds the same target/radius the queue uses)
    cj = json.loads(next((S / "clips").glob("*/cameras.json")).read_text())
    P = np.array([c["position"] for c in cj["cameras"]]); target = P.mean(0) * [1, 0, 1] + [0, P[:, 1].mean() - 0, 0]
    target[1] = V0[:, 1].mean(); radius = float(np.median(np.linalg.norm((P - target)[:, [0, 2]], axis=1)))
    K = cj["cameras"][0]; intr2 = (cj["width"], cj["height"], K["fx"], K["fy"], K["cx"], K["cy"])
    orbit = helical(target, radius, K)

    def score(x):
        prm = body.body0.reshape(1, -1).clone()   # the subject's own canonical pose plus the candidate offsets
        for (h, ax), v in zip(DOF, x):
            prm[0, body_index(ROT[h][AXES[ax]])] += float(v)
        Vp = body.pose(prm).verts[0].cpu().numpy()
        c = visible_counts(Vp, intr2, orbit)
        gain = area[(~seen) & (c >= MIN_VIEWS)].sum()
        return gain - 0.002 * float(np.sum((x / LIM) ** 2)) * area[~seen].sum(), gain, Vp

    rng = np.random.default_rng(0)
    best = (score(np.zeros(len(DOF)))[0], np.zeros(len(DOF)))
    print(f"canonical pose exposes {score(np.zeros(len(DOF)))[1] * 1e4:.0f} cm^2 of it")
    for k in range(a.evals):
        sig = 0.6 if k < a.evals // 2 else 0.2 * (1 - k / a.evals) + 0.05
        x = np.clip(best[1] + rng.normal(0, sig, len(DOF)) * LIM, -LIM, LIM) if k >= a.evals // 4 else rng.uniform(-LIM, LIM)
        s = score(x)
        if s[0] > best[0]:
            best = (s[0], x)
            print(f"eval {k}: exposes {s[1] * 1e4:.0f} cm^2  pose {dict(zip([f'{h}.{ax}' for h, ax in DOF], np.round(x, 2)))}", flush=True)
    s, gain, Vp = score(best[1])
    np.savez(out / "pose.npz", x=best[1], dof=np.array([f"{h}.{ax}" for h, ax in DOF]), gain_cm2=gain * 1e4,
             unseen_cm2=area[~seen].sum() * 1e4, seen=seen)
    print(f"best: exposes {gain * 1e4:.0f} of {area[~seen].sum() * 1e4:.0f} cm^2 unseen; saved {out / 'pose.npz'}")


if __name__ == "__main__":
    main()
