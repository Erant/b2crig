"""Fill the surface the capture never saw from a held "reveal" clip: learn those splats' opacity and colour.

    .venv/bin/python tools/reveal_fit.py CLIP_DIR [--iters N] [--tag T] [-- extra fit-cage args]

Mask: splats whose nearest canonical body vertex was unseen by the capture (tools/reveal_pose.py's pose.npz `seen`).
The clip's static frames are cut into 4 orbit blocks; blocks 0 and 2 fit (all tied to one cage frame, <rep>@<frame>),
1 and 3 are held out. fit-cage --fit-appearance --no-delta writes the new scene.ply; the held-out frames are rendered
with the old and the new splat and scored by L1 against the WAN frame, over the figure and inside the pixels whose
colour the fit changed (where the fill matters).
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2crig import b2ctrain as B  # noqa: E402
from b2crig import evaluate  # noqa: E402
from b2crig.io.splat import load  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("clip", type=Path)
ap.add_argument("--iters", type=int, default=6000)
ap.add_argument("--tag", default="")
ap.add_argument("--splats", action="store_true", help="learn the masked splats' geometry too (fit-cage --fit-splats), "
                "mask widened by 2 mesh rings plus the dual-binding candidates")
argv = sys.argv[1:]
extra = argv[argv.index("--") + 1:] if "--" in argv else []
a = ap.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
clip = a.clip.resolve(); S = clip.parents[1]
run = Path(json.loads((S.parent / "subjects.json").read_text())[S.name]); splat = run / "ply" / "scene.ply"
out = clip / f"reveal_fit{a.tag}"; out.mkdir(exist_ok=True)

s = load(splat)
BV = np.load(S / "mhr.npz")["verts_world"]; seen = np.load(S / "reveal" / "pose.npz")["seen"]
_, nv = cKDTree(BV).query(s.xyz)
unseen = ~seen
if a.splats:
    import scipy.sparse as sp_
    F = np.load(S / "mhr.npz")["faces"].astype(np.int64)
    A = sp_.coo_matrix((np.ones(F.size * 2), (np.r_[F.ravel(), np.roll(F, 1, 1).ravel()], np.r_[np.roll(F, 1, 1).ravel(), F.ravel()])),
                       shape=(len(BV), len(BV))).tocsr()
    for _ in range(2):
        unseen = unseen | (A @ unseen.astype(float) > 0)
mask = unseen[nv].astype(np.uint8)
if a.splats and (S / "binding" / "candidates.bin").exists():
    from b2crig.rig import binding as BD
    mask[BD.read_alt(S / "binding" / "candidates.bin")[0]] = 1
mask.tofile(out / "mask.u8")
print(f"reveal_fit: {mask.sum()} of {len(mask)} splats sit on surface the capture never saw")

p = np.load(clip / "motion.npz")["body_params"]; T = len(p)
static = [t for t in range(1, T) if np.abs(p[t] - p[t - 1]).max() < 1e-6]
blocks = np.array_split(np.array(static), 4)
fit_fr = sorted(np.concatenate([blocks[0], blocks[2]]).tolist()); held = sorted(np.concatenate([blocks[1], blocks[3]]).tolist())
rep = static[0]
ds = out / "ds"; shutil.rmtree(ds, ignore_errors=True)
for sub in ("images", "labels"):
    (ds / sub).mkdir(parents=True)
src = clip / "fit_ds"; shutil.copy(src / "cameras.txt", ds); (ds / "points3D.txt").write_text("")
keep = []
for ln in (src / "images.txt").read_text().splitlines():
    f = ln.split()
    if len(f) < 10 or int(Path(f[9]).stem) not in fit_fr:
        continue
    nm = f"{rep:04d}@{Path(f[9]).stem}.png"
    keep += [" ".join(f[:9] + [nm]), ""]
    for sub in ("images", "labels"):
        (ds / sub / nm).symlink_to((src / sub / f[9]).resolve())
(ds / "images.txt").write_text("\n".join(keep) + "\n")
subprocess.run([str(B.B2CTRAIN), "fit-cage", *B.fit_groups_args(), "--splat", str(splat), "--cage", str(clip / "cage.b2ccage"), "--dataset", str(ds),
                "--output", str(out), "--iters", str(a.iters), "--temporal", "0", "--no-delta", "--fit-splats" if a.splats else "--fit-appearance", str(out / "mask.u8"),
                "--photo-weight", "1.0", *extra], check=True)

cams = json.loads((clip / "cameras.json").read_text())
cams["cameras"] = [c for c in cams["cameras"] if int(c["name"]) in held]
(out / "held_cameras.json").write_text(json.dumps(cams))
renders = {}
for name, ply in (("old", splat), ("new", out / "scene.ply")):
    rd = out / f"held_{name}"
    B.render(ply, out / "held_cameras.json", rd, cage=clip / "cage.b2ccage", background=(0.5, 0.5, 0.5))
    renders[name] = rd
res = {"old": {"fig": [], "chg": []}, "new": {"fig": [], "chg": []}}
for c in cams["cameras"]:
    nm = c["name"]
    wan = cv2.imread(str(clip / "wan" / f"{nm}.png")).astype(np.float32)
    seg = cv2.imread(str(clip / "seg" / f"{nm}.png"), 0) > 0
    im = {k: cv2.imread(str(v / f"{nm}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) for k, v in renders.items()}
    rgb = {k: v[..., :3] * v[..., 3:] / 255 + 127 * (1 - v[..., 3:] / 255) for k, v in im.items()}
    changed = np.abs(rgb["new"] - rgb["old"]).max(-1) > 8
    for k in rgb:
        err = np.abs(rgb[k] - wan).mean(-1)
        res[k]["fig"].append(err[seg].mean()); res[k]["chg"].append(err[changed].mean() if changed.sum() > 50 else np.nan)
print(f"held-out frames ({len(held)}): L1 vs WAN over the figure / inside the changed pixels")
for k in ("old", "new"):
    print(f"  {k:4s} {np.nanmean(res[k]['fig']):7.2f} / {np.nanmean(res[k]['chg']):7.2f}")
