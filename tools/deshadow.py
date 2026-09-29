"""Neutralise crease shading baked into the canonical splat: skin splats on body surface that the pose library
stretches a lot (an A-pose armpit crease opening as the arm lifts) take the colour of nearby skin that stays put.

    .venv/bin/python tools/deshadow.py work/<subject> IN_PLY OUT_PLY [--clips lib_* d_*] [--hi 1.6] [--lo 1.15]
        [--radius 0.06]

Per body vertex: the largest size ratio of its incident faces over every frame of the listed clips' cages. A skin
splat (Sapiens skin classes) within 3 cm of the body takes weight w = clip((s - lo') / (hi - lo'), 0, 1) with
s its nearest vertex's stretch and lo' = (lo + hi) / 2; its DC colour moves by w towards the median DC of the skin
splats within `radius` whose vertex stretch stays under `lo`, and its higher SH bands shrink by (1 - w).
"Background"-labelled splats on the body (the dark specks in a crease) are treated the same and their opacity also
scales by (1 - w).
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2crig import evaluate  # noqa: E402

SKIN = (3, 6, 7, 11, 15, 16, 20, 22)   # face/neck, hands, arms, torso (no legs: trousers)


def vertex_stretch(S: Path, clips: list[str], nb: int, F: np.ndarray) -> np.ndarray:
    def size(V):
        return np.sqrt(np.linalg.norm(np.cross(V[..., F[:, 1], :] - V[..., F[:, 0], :], V[..., F[:, 2], :] - V[..., F[:, 0], :]), axis=-1))
    best = None
    k0 = None
    for pat in clips:
        for c in sorted(glob.glob(str(S / "clips" / pat))):
            cp = Path(c) / "cage.b2ccage"
            if not cp.exists():
                continue
            cg = evaluate.read_cage(cp)
            if k0 is None:
                k0 = np.maximum(size(cg.verts[:nb]), 1e-9)
            r = (size(cg.posed[:, :nb]) / k0).max(0)
            best = r if best is None else np.maximum(best, r)
    vs = np.ones(nb)
    np.maximum.at(vs, F.ravel(), np.repeat(best, 3))
    return vs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("subject", type=Path)
    ap.add_argument("inp", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--clips", nargs="+", default=["lib_*", "d_*_S"])
    ap.add_argument("--hi", type=float, default=1.6)
    ap.add_argument("--lo", type=float, default=1.15)
    ap.add_argument("--radius", type=float, default=0.06)
    a = ap.parse_args()
    S = a.subject.resolve()
    md = np.load(S / "mhr.npz")
    V, F = md["verts_world"], md["faces"].astype(np.int64)
    vs = vertex_stretch(S, a.clips, len(V), F)
    print(f"deshadow: body vertices stretched > {a.lo}: {(vs > a.lo).sum()}, > {a.hi}: {(vs > a.hi).sum()}")
    ply = PlyData.read(str(a.inp))
    el = ply["vertex"]
    xyz = np.stack([el["x"], el["y"], el["z"]], 1)
    lab = np.asarray(el["seg_label"]).astype(int)
    d, vi = cKDTree(V).query(xyz)
    s = vs[vi]
    skin = np.isin(lab, SKIN) & (d < 0.03)
    junk = (lab == 0) & (d < 0.03)   # "Background"-labelled splats on the body: in a crease they are the dark specks
    lo2 = (a.lo + a.hi) / 2
    w0 = np.clip((s - lo2) / (a.hi - lo2), 0, 1)
    w = w0 * (skin | junk)
    src = skin & (s < a.lo)
    dc = np.stack([np.asarray(el[f"f_dc_{k}"]) for k in range(3)], 1)
    tree = cKDTree(xyz[src])
    tgt = np.nonzero(w > 0)[0]
    nbrs = tree.query_ball_point(xyz[tgt], a.radius)
    srcdc = dc[src]
    new = dc.copy()
    moved = 0
    for j, nb in zip(tgt, nbrs):
        if len(nb) >= 8:
            new[j] = dc[j] + w[j] * (np.median(srcdc[nb], 0) - dc[j])
            moved += 1
    for k in range(3):
        el[f"f_dc_{k}"] = new[:, k].astype(np.float32)
    op = np.asarray(el["opacity"]).astype(np.float64)
    pj = 1 / (1 + np.exp(-op)) * np.where(junk, 1 - w, 1.0)
    el["opacity"] = np.log(np.clip(pj, 1e-6, 1 - 1e-6) / (1 - np.clip(pj, 1e-6, 1 - 1e-6))).astype(np.float32)
    rest = [p.name for p in el.properties if p.name.startswith("f_rest_")]
    for p in rest:
        el[p] = (np.asarray(el[p]) * (1 - w)).astype(np.float32)
    ply.write(str(a.out))
    print(f"deshadow: {moved} of {len(tgt)} weighted skin splats recoloured (mean weight {w[tgt].mean():.2f}) -> {a.out}")


if __name__ == "__main__":
    main()
