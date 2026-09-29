"""Score what a splat throws outside the figure when animated: posed clip frames, seen from capture directions
re-aimed at the posed body (as tools/cage_train.py --sil builds its silhouette views), against the allowed silhouette =
the dilated posed render of the REFERENCE splat minus its splats more than --depth inside the canonical MHR body.

    .venv/bin/python tools/sil_eval.py work/<subject> REF_PLY PLY [PLY ...] --clips theater walk_wave [--every 12]
        [--cams 2] [--dilate 5] [--hands] [--sheet OUT.png]

Per splat: outside = alpha mass outside the allowed region per frame (pixels at half the capture resolution), and the
alpha coverage inside it (a splat that only got thinner would score low outside too). `--hands`: close-ups of one
hand per view instead (the hand vertices, b2crig/rig/contain.hand_vertices, 0.3 m across), for finger poses.
`--mesh-ref` (with --hands): only each splat's HAND splats are rendered (nearest canonical body vertex in a hand), and
the allowed silhouette is the posed MHR body (cage layer 0, dilated) instead of the reference splat's own posed render,
which already contains its knuckle-straddling splats sheared out.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2crig import b2ctrain as B  # noqa: E402
from b2crig import evaluate  # noqa: E402
from b2crig.cameras import Intrinsics, look_at, write_cameras  # noqa: E402
from b2crig.io.splat import load, save_subset  # noqa: E402
from b2crig.rig.cage import vertex_normals  # noqa: E402
from b2crig.rig.contain import hand_vertices  # noqa: E402
from b2crig.visibility import colmap_cams  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("subject", type=Path)
ap.add_argument("ref", type=Path)
ap.add_argument("plys", type=Path, nargs="+")
ap.add_argument("--clips", nargs="+", required=True)
ap.add_argument("--every", type=int, default=12)
ap.add_argument("--cams", type=int, default=2)
ap.add_argument("--depth", type=float, default=0.02)
ap.add_argument("--dilate", type=int, default=5)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--mesh-ref", action="store_true", help="allowed region = the posed body mesh's silhouette")
ap.add_argument("--hands", action="store_true", help="hand close-ups (0.3 m across) instead of whole-body views")
ap.add_argument("--sheet", type=Path, help="worst frame of each clip: reference | splats, outside alpha in red")
a = ap.parse_args()
S = a.subject.resolve(); run = Path(json.loads((S.parent / "subjects.json").read_text())[S.name])
tmp = Path(tempfile.mkdtemp(dir=S))
md = np.load(S / "mhr.npz"); BV = md["verts_world"].astype(float); BN = vertex_normals(BV, md["faces"].astype(np.int64))
sp = load(a.ref); _, j = cKDTree(BV).query(sp.xyz); sd = np.einsum("ij,ij->i", sp.xyz - BV[j], BN[j])
src = tmp / "src.ply"; save_subset(a.ref, np.flatnonzero(sd >= -a.depth), src)
(W1, H1, fx1, fy1, cx1, cy1), ccams = colmap_cams(run / "colmap")
K2 = Intrinsics(W1 // 2, H1 // 2, fx1 / 2, fy1 / 2, cx1 / 2, cy1 / 2)
offs = np.array([-R.T @ t for R, t in ccams]) - BV.mean(0)
hv = hand_vertices(S / "mhr.npz") if a.hands else None
hand_dist = max(K2.fx, K2.fy) * 0.3 / min(K2.width, K2.height)

def mesh_mask(V, F, R, pos, K):   # silhouette of a mesh seen by a look_at camera (columns right, up, back)
    q = (V - pos) @ R; z = -q[:, 2]
    uv = np.stack([K.fx * q[:, 0] / z + K.cx, -K.fy * q[:, 1] / z + K.cy], 1)
    m = np.zeros((K.height, K.width), np.uint8)
    ok = (z[F] > 0.01).all(1)
    cv2.fillPoly(m, [t for t in np.round(uv[F[ok]] * 4).astype(np.int32)], 1, shift=2)
    return m


rng = np.random.default_rng(a.seed); ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.dilate + 1,) * 2)
res = {p: [] for p in a.plys}; sheet = []; iou = []
if a.mesh_ref:   # render the hand splats only (the mesh is the reference; hair or clothing in view is not the hand's)
    hvx = hand_vertices(S / "mhr.npz"); hand_v = np.zeros(len(BV), bool); hand_v[np.concatenate([hvx["r"], hvx["l"]])] = True
    tree = cKDTree(BV); rend = {}
    for i, p in enumerate(a.plys):
        _, jj = tree.query(load(p).xyz); rend[p] = tmp / f"hands_{i}.ply"; save_subset(p, np.flatnonzero(hand_v[jj]), rend[p])
    _, jj = tree.query(load(a.ref).xyz); save_subset(a.ref, np.flatnonzero(hand_v[jj] & (sd >= -a.depth)), src)
else:
    rend = {p: p for p in a.plys}
for c in a.clips:
    clip = S / "clips" / c
    cgp = clip / "cage_v2.b2ccage" if (clip / "cage_v2.b2ccage").exists() else clip / "cage.b2ccage"
    cg = evaluate.read_cage(cgp); nb = cg.layers[0][1]; fis = list(range(0, len(cg.names), a.every))
    worst = (-1, None)
    for k in range(a.cams):
        poses = []
        for fi in fis:
            ctr = cg.posed[fi][:nb].astype(float).mean(0); pos = ctr + offs[rng.integers(len(offs))]
            if hv is not None:
                o = pos - ctr; ctr = cg.posed[fi][hv["rl"[rng.integers(2)]]].astype(float).mean(0)
                pos = ctr + o / np.linalg.norm(o) * hand_dist
            poses.append((look_at(pos, ctr), pos))
        cj = tmp / f"{c}_{k}.json"; write_cameras(cj, K2, poses, [cg.names[fi] for fi in fis])
        if a.mesh_ref:
            v0, vc, f0, fc, _ = cg.layers[0]; BF = (cg.faces[f0:f0 + fc] - v0).astype(np.int64)
            BF = BF[hand_v[BF].all(1)]                         # the hands' triangles (the splats rendered are the hands')
            mm = {fi: mesh_mask(cg.posed[fi][v0:v0 + vc].astype(float), BF, *poses[i], K2) for i, fi in enumerate(fis)}
        B.render(src, cj, tmp / f"{c}_{k}_ref", cage=cgp)
        for p in a.plys:
            B.render(rend[p], cj, tmp / f"{c}_{k}_{a.plys.index(p)}", cage=cgp, background=(0.5, 0.5, 0.5))
        for fi in fis:
            nm = f"{cg.names[fi]}.png"
            ref_a = cv2.imread(str(tmp / f"{c}_{k}_ref" / nm), cv2.IMREAD_UNCHANGED)[..., 3] > 76
            if a.mesh_ref:
                mref = mm[fi] > 0
                iou.append((ref_a & mref).sum() / max((ref_a | mref).sum(), 1))
                ref_a = mref
            allowed = cv2.dilate(ref_a.astype(np.uint8), ker) > 0
            ims = []
            for i, p in enumerate(a.plys):
                x = cv2.imread(str(tmp / f"{c}_{k}_{i}" / nm), cv2.IMREAD_UNCHANGED).astype(np.float32)
                al = x[..., 3] / 255
                res[p].append((c, al[~allowed].sum(), al[allowed].mean()))
                ims.append(x)
            out0 = res[a.plys[0]][-1][1]
            if a.sheet and out0 > worst[0]:
                worst = (out0, (allowed, ims))
    if a.sheet and worst[1] is not None:
        allowed, ims = worst[1]; row = []
        for x in ims:
            al = x[..., 3:] / 255; rgb = x[..., :3] * al + 127 * (1 - al)
            o = (~allowed) & (al[..., 0] > 0.02); rgb[o] = rgb[o] * 0.3 + np.array([0, 0, 255]) * 0.7
            row.append(rgb.astype(np.uint8))
        sheet.append(np.hstack(row))
print(f"{'splat':40s} " + " ".join(f"{c + ' outside':>18s} {'cover':>6s}" for c in a.clips))
for p in a.plys:
    r = res[p]
    print(f"{str(p.parent.name + '/' + p.name):40s} " + " ".join(
        f"{np.mean([x[1] for x in r if x[0] == c]):18.1f} {np.mean([x[2] for x in r if x[0] == c]):6.3f}" for c in a.clips))
if iou:
    print(f"mesh vs reference splat silhouette IoU: median {np.median(iou):.3f}, min {np.min(iou):.3f}")
if a.sheet:
    cv2.imwrite(str(a.sheet), np.vstack(sheet)); print(f"sheet: {a.sheet}")
import shutil; shutil.rmtree(tmp)
