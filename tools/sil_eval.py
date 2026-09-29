"""Score what a splat throws outside the figure when animated: posed clip frames, seen from capture directions
re-aimed at the posed body (as tools/cage_train.py --sil builds its silhouette views), against the allowed silhouette =
the dilated posed render of the REFERENCE splat minus its splats more than --depth inside the canonical MHR body.

    .venv/bin/python tools/sil_eval.py work/<subject> REF_PLY PLY [PLY ...] --clips theater walk_wave [--every 12]
        [--cams 2] [--dilate 5] [--sheet OUT.png]

Per splat: outside = alpha mass outside the allowed region per frame (pixels at half the capture resolution), and the
alpha coverage inside it (a splat that only got thinner would score low outside too).
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
rng = np.random.default_rng(a.seed); ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.dilate + 1,) * 2)
res = {p: [] for p in a.plys}; sheet = []
for c in a.clips:
    clip = S / "clips" / c
    cgp = clip / "cage_v2.b2ccage" if (clip / "cage_v2.b2ccage").exists() else clip / "cage.b2ccage"
    cg = evaluate.read_cage(cgp); nb = cg.layers[0][1]; fis = list(range(0, len(cg.names), a.every))
    worst = (-1, None)
    for k in range(a.cams):
        poses = []
        for fi in fis:
            ctr = cg.posed[fi][:nb].astype(float).mean(0); pos = ctr + offs[rng.integers(len(offs))]
            poses.append((look_at(pos, ctr), pos))
        cj = tmp / f"{c}_{k}.json"; write_cameras(cj, K2, poses, [cg.names[fi] for fi in fis])
        B.render(src, cj, tmp / f"{c}_{k}_ref", cage=cgp)
        for p in a.plys:
            B.render(p, cj, tmp / f"{c}_{k}_{a.plys.index(p)}", cage=cgp, background=(0.5, 0.5, 0.5))
        for fi in fis:
            nm = f"{cg.names[fi]}.png"
            allowed = cv2.dilate((cv2.imread(str(tmp / f"{c}_{k}_ref" / nm), cv2.IMREAD_UNCHANGED)[..., 3] > 76).astype(np.uint8), ker) > 0
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
if a.sheet:
    cv2.imwrite(str(a.sheet), np.vstack(sheet)); print(f"sheet: {a.sheet}")
import shutil; shutil.rmtree(tmp)
