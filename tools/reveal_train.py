"""Fill the surface the capture never saw: retrain the splat on the capture views plus a held "reveal" clip.

    .venv/bin/python tools/reveal_train.py CLIP_DIR [--iters N] [--repeat K] [--tag T] [-- extra b2ctrain args]

One COLMAP dataset: the subject's capture views (camera 1, render canonical) and the reveal clip's fitting frames
(orbit blocks 0 and 2 of its static frames; camera 2, posed by the clip's cage via b2ctrain --cage), warm-started from
the subject's scene.ply (init.ply). The capture holds everything it saw; the reveal frames fill in what it did not.
Scores: the held-out reveal frames (blocks 1 and 3) by L1 vs the WAN frame, over the figure and inside the pixels the
retraining changed; and the capture views (canonical) by L1 vs their images, to see that nothing seen got worse.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2crig import b2ctrain as B  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("clip", type=Path)
ap.add_argument("--iters", type=int, default=6000)
ap.add_argument("--repeat", type=int, default=2, help="list each reveal frame this often (the capture has ~160 views)")
ap.add_argument("--tag", default="")
ap.add_argument("--frames", default="wan", help="the reveal frames' folder, e.g. wan_1080 (b2crunner's tools/upscale_clip); alpha and "
                "labels come from the clip's seg, resized, and camera 2 is scaled to the frames")
argv = sys.argv[1:]
extra = argv[argv.index("--") + 1:] if "--" in argv else []
a = ap.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
clip = a.clip.resolve(); S = clip.parents[1]
run = Path(json.loads((S.parent / "subjects.json").read_text())[S.name]); cap = run / "colmap"
out = clip / f"reveal_train{a.tag}"; ds = out / "ds"
shutil.rmtree(ds, ignore_errors=True)
for sub in ("images", "labels"):
    (ds / sub).mkdir(parents=True)

cam1 = [l for l in (cap / "cameras.txt").read_text().splitlines() if l and not l.startswith("#")][0].split()
cam2 = (clip / "fit_ds" / "cameras.txt").read_text().split()
if a.frames != "wan":   # upscaled frames: build their RGBA + labels, scale the intrinsics
    f0 = cv2.imread(str(next((clip / a.frames).glob("*.png"))))
    H2, W2 = f0.shape[:2]; sx, sy = W2 / int(cam2[2]), H2 / int(cam2[3])
    cam2 = cam2[:2] + [str(W2), str(H2), str(float(cam2[4]) * sx), str(float(cam2[5]) * sy), str(float(cam2[6]) * sx), str(float(cam2[7]) * sy)]
    up = out / "reveal_frames"
    for sub in ("images", "labels"):
        (up / sub).mkdir(parents=True, exist_ok=True)
    for f in sorted((clip / a.frames).glob("*.png")):
        lab = cv2.resize(cv2.imread(str(clip / "seg" / f.name), 0), (W2, H2), interpolation=cv2.INTER_NEAREST)
        al = cv2.GaussianBlur((lab > 0).astype(np.float32), (0, 0), 1.5 * sx)
        cv2.imwrite(str(up / "images" / f.name), np.dstack([cv2.imread(str(f)), np.clip(al * 255, 0, 255).astype(np.uint8)]))
        cv2.imwrite(str(up / "labels" / f.name), lab)
    reveal_src = up
else:
    reveal_src = clip / "fit_ds"
(ds / "cameras.txt").write_text(" ".join(["1"] + cam1[1:]) + "\n" + " ".join(["2"] + cam2[1:]) + "\n")
lines = []
for ln in (cap / "images.txt").read_text().splitlines():
    f = ln.split()
    if ln.startswith("#") or len(f) < 10:
        continue
    lines += [" ".join([str(len(lines) // 2 + 1), *f[1:8], "1", f[9]]), ""]
    for sub in ("images", "labels"):
        if (cap / sub / f[9]).exists():
            (ds / sub / f[9]).symlink_to((cap / sub / f[9]).resolve())
p = np.load(clip / "motion.npz")["body_params"]; T = len(p)
static = [t for t in range(1, T) if np.abs(p[t] - p[t - 1]).max() < 1e-6]
blocks = np.array_split(np.array(static), 4)
fit_fr = set(np.concatenate([blocks[0], blocks[2]]).tolist()); held = sorted(np.concatenate([blocks[1], blocks[3]]).tolist())
for ln in (clip / "fit_ds" / "images.txt").read_text().splitlines():
    f = ln.split()
    if len(f) < 10 or int(Path(f[9]).stem) not in fit_fr:
        continue
    for r in range(a.repeat):
        nm = f"{Path(f[9]).stem}@r{r}.png"   # the cage frame is the part before '@'
        lines += [" ".join([str(len(lines) // 2 + 1), *f[1:8], "2", nm]), ""]
        for sub in ("images", "labels"):
            (ds / sub / nm).symlink_to((reveal_src / sub / f[9]).resolve())
(ds / "images.txt").write_text("\n".join(lines) + "\n"); (ds / "points3D.txt").write_text("")
(ds / "init.ply").symlink_to((run / "ply" / "scene.ply").resolve())
print(f"reveal_train: {len(lines) // 2} views ({len(fit_fr) * a.repeat} reveal); held-out reveal frames {len(held)}")

subprocess.run([str(B.B2CTRAIN), str(ds), "--cage", str(clip / "cage.b2ccage"), "--total-train-iters", str(a.iters),
                "--export-path", str(out), "--export-name", "scene.ply", "--export-every", str(a.iters),
                "--sh-degree", "3", "--max-resolution", "1920", *extra], check=True)
new = out / "scene.ply"; old = run / "ply" / "scene.ply"

# held-out reveal frames (posed)
cams = json.loads((clip / "cameras.json").read_text()); cams["cameras"] = [c for c in cams["cameras"] if int(c["name"]) in held]
(out / "held_cameras.json").write_text(json.dumps(cams))
rd = {}
for k, ply in (("old", old), ("new", new)):
    rd[k] = out / f"held_{k}"
    B.render(ply, out / "held_cameras.json", rd[k], cage=clip / "cage.b2ccage", background=(0.5, 0.5, 0.5))


def rgb(pth):
    x = cv2.imread(str(pth), cv2.IMREAD_UNCHANGED).astype(np.float32)
    return x[..., :3] * x[..., 3:] / 255 + 127 * (1 - x[..., 3:] / 255)


res = {k: {"fig": [], "chg": []} for k in rd}
for c in cams["cameras"]:
    nm = c["name"]; wan = cv2.imread(str(clip / "wan" / f"{nm}.png")).astype(np.float32)
    seg = cv2.imread(str(clip / "seg" / f"{nm}.png"), 0) > 0
    im = {k: rgb(v / f"{nm}.png") for k, v in rd.items()}
    chg = np.abs(im["new"] - im["old"]).max(-1) > 12
    for k in im:
        e = np.abs(im[k] - wan).mean(-1); res[k]["fig"].append(e[seg].mean()); res[k]["chg"].append(e[chg].mean() if chg.sum() > 50 else np.nan)
print(f"held-out reveal frames: L1 vs WAN over the figure / inside the changed pixels")
for k in res:
    print(f"  {k:4s} {np.nanmean(res[k]['fig']):7.2f} / {np.nanmean(res[k]['chg']):7.2f}")
