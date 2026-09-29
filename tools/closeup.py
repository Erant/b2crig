"""Close-up renders of one frame of a clip's cage around a (posed) joint -> a contact sheet.

    .venv/bin/python tools/closeup.py CLIP_DIR FRAME OUT.png --splat PLY [--splat PLY2 ...] [--joint l_shoulder]
        [--radius 0.6] [--az -60,0,60,120] [--el 10] [--size 512] [--extra "..."] [--labels]

One row per splat, one column per azimuth (degrees about the joint, relative to +Z). --labels adds a row of the
first splat's seg-label map (b2ctrain --label-maps) coloured per class.
"""
import argparse
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2crig import b2ctrain as B  # noqa: E402
from b2crig import cameras as C  # noqa: E402
from b2crig import evaluate as E  # noqa: E402
from b2crig.rig.mhr import MHRBody  # noqa: E402
from b2crig.rig.skeleton import JOINT  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("clip", type=Path)
ap.add_argument("frame", type=int)
ap.add_argument("out", type=Path)
ap.add_argument("--splat", type=Path, action="append", required=True)
ap.add_argument("--cage", default="cage.b2ccage")
ap.add_argument("--joint", default="l_shoulder")
ap.add_argument("--radius", type=float, default=0.6)
ap.add_argument("--az", default="-60,0,60,120")
ap.add_argument("--el", type=float, default=10.0)
ap.add_argument("--size", type=int, default=512)
ap.add_argument("--extra", action="append", default=[], help="b2ctrain render args per splat row (repeat; one per --splat)")
ap.add_argument("--labels", action="store_true")
ap.add_argument("--names", default="")
a = ap.parse_args()

cage = E.read_cage(a.clip / a.cage)
i = a.frame
# the joint's posed position: nearest body cage vertex to the canonical joint, followed through the frame
subject = None
for p in [a.clip.parent.parent]:
    if (p / "mhr.npz").exists():
        subject = p
body = MHRBody(subject / "mhr.npz", device="cpu")
j0 = body.pose().joints[0, JOINT[a.joint]].numpy()
nb = cage.layers[0][1]
near = np.argsort(np.linalg.norm(cage.verts[:nb] - j0, axis=1))[:30]
tgt = j0 + (cage.posed[i, near] - cage.verts[near]).mean(0)

tmp = Path(tempfile.mkdtemp(dir=Path(__file__).resolve().parents[1] / "work"))
# one cage frame per camera
azs = [float(x) for x in a.az.split(",")]
names = [f"{k:02d}" for k in range(len(azs))]
E.write_cage_like(E.Cage(cage.layers, cage.verts, cage.faces, names, np.repeat(cage.posed[i:i + 1], len(azs), 0)),
                  tmp / "c.b2ccage", np.repeat(cage.posed[i:i + 1], len(azs), 0))
f = a.size * 1.6
K = C.Intrinsics(a.size, a.size, f, f, a.size / 2, a.size / 2)
poses = []
for az in azs:
    el, azr = math.radians(a.el), math.radians(az)
    pos = tgt + a.radius * np.array([math.cos(el) * math.sin(azr), math.sin(el), math.cos(el) * math.cos(azr)])
    poses.append((C.look_at(pos, tgt), pos))
C.write_cameras(tmp / "cams.json", K, poses, names)

rows = []
labels = a.names.split(",") if a.names else [p.parent.name + "/" + p.stem for p in a.splat]
for r, sp in enumerate(a.splat):
    od = tmp / f"r{r}"
    extra = a.extra[r].split() if r < len(a.extra) else []
    B.render(sp, tmp / "cams.json", od, cage=tmp / "c.b2ccage", background=(0.5, 0.5, 0.5),
             label_maps=(a.labels and r == 0), extra=extra)
    ims = []
    for nm in names:
        x = cv2.imread(str(od / f"{nm}.png"), cv2.IMREAD_UNCHANGED)
        if x.shape[2] == 4:
            al = x[..., 3:] / 255.0
            x = (x[..., :3] * al + 127 * (1 - al)).astype(np.uint8)
        ims.append(x)
    row = np.hstack(ims)
    cv2.putText(row, labels[r], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    rows.append(row)
    if a.labels and r == 0:
        rng = np.random.default_rng(3)
        pal = rng.integers(40, 255, (256, 3)).astype(np.uint8); pal[0] = 0
        lab = []
        for nm in names:
            cands = sorted(od.glob(f"{nm}*label*.png")) + sorted(od.glob(f"{nm}.labels.png"))
            l = cv2.imread(str(cands[0]), cv2.IMREAD_GRAYSCALE) if cands else np.zeros((a.size, a.size), np.uint8)
            lab.append(pal[l])
        rows.append(np.hstack(lab))
cv2.imwrite(str(a.out), np.vstack(rows))
subprocess.run(["rm", "-rf", str(tmp)])
print(f"closeup: {a.out}")
