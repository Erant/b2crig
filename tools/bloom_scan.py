"""Find blooms: bound splats that posing carries away from the surface. For sampled frames of clips, b2ctrain's own
posed splats (render --export-posed) are compared with the canonical ones: distance to the nearest cage vertex (all
layers), posed vs canonical, and the signed distance to the MHR body (normal side). A splat "blooms" when it is opaque
and ends up more than --grow metres further from the cage than it sits canonically.

    .venv/bin/python tools/bloom_scan.py work/<subject> SPLAT CLIP [CLIP ...] [--every 8] [--grow 0.02]
        [--save-worst OUT.npz]

Prints per clip the worst frames (bloom count, the 99th percentile of the distance growth) and overall which body
regions / labels / canonical depths the blooming splats come from. --save-worst keeps the worst frame's posed
positions and the bloom mask for a closer look.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2crig import b2ctrain as B  # noqa: E402
from b2crig import cameras as C  # noqa: E402
from b2crig import evaluate as E  # noqa: E402
from b2crig.io.splat import GOLIATH  # noqa: E402
from b2crig.rig.cage import vertex_normals  # noqa: E402
from b2crig.rig.skeleton import JOINT  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("subject", type=Path)
ap.add_argument("splat", type=Path)
ap.add_argument("clips", nargs="+")
ap.add_argument("--every", type=int, default=8)
ap.add_argument("--grow", type=float, default=0.02)
ap.add_argument("--save-worst", type=Path)
ap.add_argument("--extra", default="")
ap.add_argument("--tips", action="store_true", help="measure the splats' 2-sigma tips instead of their centres")
a = ap.parse_args()
S = a.subject

v = PlyData.read(str(a.splat))["vertex"].data
xyz0 = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
op = 1 / (1 + np.exp(-v["opacity"].astype(np.float64)))
lab = v["seg_label"].astype(int)
d = np.load(S / "mhr.npz")
BF = d["faces"].astype(np.int64)
sv, sj, sw = d["skin_vertex"], d["skin_joint"], d["skin_weight"]
W = np.zeros((len(d["verts_world"]), len(d["joint_parents"]))); np.add.at(W, (sv, sj), sw)
par = d["joint_parents"]; inv = {j: k for k, j in JOINT.items()}


def region(j):
    while j not in inv and j >= 0:
        j = par[j]
    return inv.get(j, "?")


dom = np.array([region(j) for j in W.argmax(1)])


def sdist(xyz, BV):
    BN = vertex_normals(BV, BF)
    _, j = cKDTree(BV).query(xyz)
    return np.einsum("ij,ij->i", xyz - BV[j], BN[j]), j


def tips(vv, centre):
    from scipy.spatial.transform import Rotation
    q = np.stack([vv["rot_1"], vv["rot_2"], vv["rot_3"], vv["rot_0"]], 1).astype(np.float64)
    R = Rotation.from_quat(q / np.linalg.norm(q, axis=1, keepdims=True)).as_matrix()
    sc = np.exp(np.stack([vv["scale_0"], vv["scale_1"], vv["scale_2"]], 1).astype(np.float64)) * 2.0
    ax = R * sc[:, None, :]   # columns = scaled axes
    return np.concatenate([centre[:, None] + ax.transpose(0, 2, 1), centre[:, None] - ax.transpose(0, 2, 1)], 1)   # [n, 6, 3]


tmp = Path(tempfile.mkdtemp(dir=S))
worst = (0, None)
agg = []
for c in a.clips:
    clip = S / "clips" / c
    cgp = clip / ("cage_v2.b2ccage" if (clip / "cage_v2.b2ccage").exists() else "cage.b2ccage")
    cg = E.read_cage(cgp)
    nb = cg.layers[0][1]
    dc0, _ = cKDTree(cg.verts).query(xyz0)
    if a.tips:
        tip0 = cKDTree(cg.verts).query(tips(v, xyz0).reshape(-1, 3))[0].reshape(len(xyz0), 6).max(1)
    sd0, _ = sdist(xyz0, cg.verts[:nb].astype(np.float64))
    cams = json.loads((clip / "cameras.json").read_text())
    rows = []
    for fi in range(0, len(cg.names), a.every):
        nm = cg.names[fi]
        cj = dict(cams); cj["cameras"] = [dict(cams["cameras"][min(fi, len(cams["cameras"]) - 1)], name=nm)]
        (tmp / "c.json").write_text(json.dumps(cj))
        subprocess.run([str(B.B2CTRAIN), "render", "--splat", str(a.splat), "--cameras", str(tmp / "c.json"), "--output-dir", str(tmp / "r"),
                        "--cage", str(cgp), "--cage-max-growth", f"{B.CAGE_MAX_GROWTH:g}", "--export-posed", str(tmp / "p.ply"),
                        "--export-frame", nm, *a.extra.split()], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        pv = PlyData.read(str(tmp / "p.ply"))["vertex"].data
        xyz = np.stack([pv["x"], pv["y"], pv["z"]], 1).astype(np.float64)
        P = cg.posed[fi].astype(np.float64)
        dc, _ = cKDTree(P).query(xyz)
        g = dc - dc0
        if a.tips:   # the splat's 2-sigma tips along each axis: an elongated splat rotated to point off the body
            tp = tips(pv, xyz); tree = cKDTree(P)
            dtp = tree.query(tp.reshape(-1, 3))[0].reshape(len(xyz), 6).max(1)
            g = dtp - tip0
        bl = (op > 0.3) & (g > a.grow)
        rows.append((int(bl.sum()), float(np.percentile(g[op > 0.3], 99.9)), fi))
        if bl.sum() > worst[0]:
            sdp, jj = sdist(xyz, P[:nb])
            worst = (int(bl.sum()), dict(clip=c, frame=fi, xyz=xyz, bloom=bl, sd0=sd0, sdp=sdp, g=g))
        agg.append((c, fi, bl, sd0))
    rows.sort(reverse=True)
    print(f"{c:18s} worst frames: " + "  ".join(f"#{fi} {n} ({g * 100:.1f} cm p99.9)" for n, g, fi in rows[:4]), flush=True)

subprocess.run(["rm", "-rf", str(tmp)])
allb = np.zeros(len(xyz0), int)
for c, fi, bl, sd0 in agg:
    allb += bl
ever = allb > 0
print(f"\nsplats that bloom (> {a.grow * 100:.0f} cm further off the cage) in any sampled frame: {ever.sum()} of {(op > 0.3).sum()} opaque")
_, j0 = cKDTree(d["verts_world"]).query(xyz0)
for name, key in (("region", dom[j0]), ("label", np.array([GOLIATH[x] if x < len(GOLIATH) else str(x) for x in lab]))):
    u, n = np.unique(key[ever], return_counts=True)
    print(f"by {name}: " + ", ".join(f"{uu} {nn}" for nn, uu in sorted(zip(n, u), reverse=True)[:10]))
sd0 = agg[0][3]
h, e = np.histogram(sd0[ever] * 100, bins=[-10, -4, -2, -1, 0, 1, 2, 4, 10])
print("canonical signed distance to the body (cm) of blooming splats: " + ", ".join(f"[{e[i]:g},{e[i + 1]:g}) {h[i]}" for i in range(len(h))))
h2, _ = np.histogram(sd0[op > 0.3] * 100, bins=e)
print("                                        all opaque splats: " + ", ".join(f"[{e[i]:g},{e[i + 1]:g}) {h2[i]}" for i in range(len(h2))))
if a.save_worst and worst[1] is not None:
    w = worst[1]
    np.savez(a.save_worst, **{k: w[k] for k in ("xyz", "bloom", "sd0", "sdp", "g")}, clip=w["clip"], frame=w["frame"])
    print(f"worst: {w['clip']} frame {w['frame']} ({worst[0]} blooming) -> {a.save_worst}")
