"""Assign every splat an "opening" vertex-pair gate (cage_gate_a/_b/_s ply properties).

SUPERSEDED for --cage-open (2026-09-29): b2ctrain's --cage-open reads the per-vertex rotation gate that
b2crig/rig/open_gate.py writes into the cage (B2COPEN1), not these properties. b2ctrain still reads cage_gate_a/_b, but
only for filler / crease splats (ply cage_fill != 0, tools/armpit_fill.py): their fade follows the pair's distance
ratio. cage_gate_s is carried through training and export and read by nothing. The text below is the tool's original
design.

    .venv/bin/python tools/open_gates.py CAGE IN_PLY OUT_PLY [--gate-radius 0.06] [--min-opening 2.0] [--body-dist 0.08]

For each splat within --body-dist of the body cage (and not on the hair / face layers), take its nearest body-cage
vertex a and, among the body vertices within --gate-radius of a in the canonical pose (at least 1 cm away), the vertex b
whose distance to a grows most, relative to its canonical distance, over the cage's frames (the pose library, dances,
reveals ride along in a training cage). Where that pair's largest ratio reaches --min-ratio AND its largest separation
(posed - canonical distance) reaches --min-sep, the splat gets the gate (a, b) as `cage_gate_a/_b` plus `cage_gate_s` =
that largest separation; b2ctrain reads the pair's (posed - canonical) / cage_gate_s per frame, the fraction of the
pair's own opening, and blends the splat's learned open state in over --cage-open-start/-end of it (0.25 -> 0.75).
The fraction, not the ratio, is the gate because only pairs that touch in the canonical pose ever double their distance;
a torso point a few cm under the crease reaches 2x at most with the arm fully raised, but most of its own separation.
Everything else stays a plain splat (gate -1). Label-free by design: the walls of a closed crease (inner arm, side of
the torso) rotate apart without stretching, and this is what a vertex-pair distance sees; a hand resting on the hip
gates the hip splats it shadowed the same way.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2crig import evaluate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cage", type=Path)
    ap.add_argument("inp", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--gate-radius", type=float, default=0.12, help="pair a with the vertex within this (canonical, m) that separates most")
    ap.add_argument("--min-ratio", type=float, default=1.5, help="gate only where the pair distance grows at least this much (ratio) over the frames")
    ap.add_argument("--min-sep", type=float, default=0.05, help="... and separates by at least this (m)")
    ap.add_argument("--body-dist", type=float, default=0.08, help="splats farther than this from the body cage stay ungated")
    ap.add_argument("--layer-dist", type=float, default=0.02, help="splats within this of a hair / face layer stay ungated")
    ap.add_argument("--frame-stride", type=int, default=2)
    a = ap.parse_args()

    t0 = time.time()
    cage = evaluate.read_cage(a.cage)
    nvb = cage.layers[0][1]
    Vb = cage.verts[:nvb]
    P = np.ascontiguousarray(cage.posed[::a.frame_stride, :nvb])
    print(f"cage: {len(cage.layers)} layers, {nvb} body vertices, {len(P)} frames ({time.time() - t0:.1f}s)")
    # Hair / face layer surfaces (their splats stay plain; the label is not trusted: dark crease splats get voted anything).
    other = [cage.verts[L[0]:L[0] + L[1]] for L in cage.layers[2:]]
    ply = PlyData.read(str(a.inp))
    el = ply["vertex"]
    xyz = np.stack([el["x"], el["y"], el["z"]], 1).astype(np.float32)
    lab = np.asarray(el["seg_label"]).astype(int) if "seg_label" in el.data.dtype.names else np.zeros(len(xyz), int)
    vtree = cKDTree(Vb)
    dv, va = vtree.query(xyz)
    d_other = cKDTree(np.concatenate(other)).query(xyz)[0] if other else np.full(len(xyz), 1e9)
    elig = (dv < a.body_dist) & (d_other > a.layer_dist)
    print(f"splats: {len(xyz)}, eligible {elig.sum()} (near hair / face layers: {(d_other <= a.layer_dist).sum()}, farther than {a.body_dist} m from the body: {(dv >= a.body_dist).sum()})")

    # Per body vertex that some eligible splat is nearest to: the best partner (largest ratio), its ratio and separation.
    verts = np.unique(va[elig])
    partner = np.full(nvb, -1, int)
    ratio = np.ones(nvb, np.float32)
    sep = np.zeros(nvb, np.float32)
    balls = vtree.query_ball_point(Vb[verts], a.gate_radius)
    for a_, cand in zip(verts, balls):
        cand = np.asarray(cand)
        d0 = np.linalg.norm(Vb[cand] - Vb[a_], axis=1)
        keep = d0 > 0.01
        cand, d0 = cand[keep], d0[keep]
        if len(cand) == 0:
            continue
        dmax = np.linalg.norm(P[:, cand] - P[:, a_:a_ + 1], axis=2).max(0)
        r = dmax / d0
        k = int(np.argmax(r))
        partner[a_] = cand[k]
        ratio[a_] = r[k]
        sep[a_] = dmax[k] - d0[k]
    print(f"vertices: {len(verts)} scanned; ratio >= {a.min_ratio} at {(ratio[verts] >= a.min_ratio).sum()}, "
          f"and separation >= {a.min_sep} m at {((ratio[verts] >= a.min_ratio) & (sep[verts] >= a.min_sep)).sum()} ({time.time() - t0:.1f}s)")

    gate = np.full((len(xyz), 2), -1.0, np.float32)
    gate_s = np.zeros(len(xyz), np.float32)
    ok = elig & (partner[va] >= 0) & (ratio[va] >= a.min_ratio) & (sep[va] >= a.min_sep)
    gate[ok, 0] = va[ok]
    gate[ok, 1] = partner[va[ok]]
    gate_s[ok] = sep[va[ok]]
    op = ratio[va[ok]]
    print(f"gated splats: {ok.sum()} ({ok.mean() * 100:.1f}%); ratio median {np.median(op):.2f}x (>= 2x: {(op >= 2).sum()}, >= 3x: {(op >= 3).sum()}); "
          f"separation median {np.median(gate_s[ok]) * 100:.1f} cm; y range {xyz[ok, 1].min():.2f}..{xyz[ok, 1].max():.2f}")
    cls = np.bincount(lab[ok], minlength=32)
    print("gated by class: " + " ".join(f"{c}:{n}" for c, n in enumerate(cls) if n))

    names = [n for n in el.data.dtype.names if n not in ("cage_gate_a", "cage_gate_b", "cage_gate_s")]
    out = np.zeros(len(el), dtype=[(nm, "f4") for nm in names] + [("cage_gate_a", "f4"), ("cage_gate_b", "f4"), ("cage_gate_s", "f4")])
    for nm in names:
        out[nm] = el[nm]
    out["cage_gate_a"] = gate[:, 0]
    out["cage_gate_b"] = gate[:, 1]
    out["cage_gate_s"] = gate_s
    PlyData([PlyElement.describe(out, "vertex")], text=False, byte_order="<", comments=list(ply.comments)).write(str(a.out))
    print(f"-> {a.out} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
