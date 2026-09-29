"""Render a solved motion (tools/make_walk.py) of a subject's splat with a tracking camera -> frames + mp4.

    .venv/bin/python tools/render_final.py work/<subject> MOTION.npz OUT_DIR [--splat PLY] [--width 1080 --height 1920]
        [--az0 35 --az1 8 --el 4 --radius0 3.3 --radius1 2.7] [--hair k_tip=0.04,...] [--sh-degree 3]

The camera looks at the subject's pelvis (horizontally smoothed, fixed height), from azimuth az0 easing to az1 and
radius0 easing to radius1 over the clip (a slow pan round to the front and a push in for the smile and wave).
Writes OUT_DIR/cage.b2ccage, cameras.json, frames/NNNN.png and OUT_DIR.mp4.
"""
import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from b2crig import plyheader  # noqa: E402

from b2crig import b2ctrain as B  # noqa: E402
from b2crig import cameras as C  # noqa: E402
from b2crig.rig import layered  # noqa: E402
from b2crig.rig.mhr import MHRBody  # noqa: E402
from recage import hair_params  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("subject", type=Path)
ap.add_argument("motion", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--splat", type=Path)
ap.add_argument("--width", type=int, default=1080)
ap.add_argument("--height", type=int, default=1920)
ap.add_argument("--az0", type=float, default=35.0)
ap.add_argument("--az1", type=float, default=8.0)
ap.add_argument("--el", type=float, default=4.0)
ap.add_argument("--radius0", type=float, default=3.3)
ap.add_argument("--radius1", type=float, default=2.7)
ap.add_argument("--hair", default="")
ap.add_argument("--bg", default="0.5,0.5,0.5")
ap.add_argument("--label", default="")
ap.add_argument("--extra", default="", help="extra b2ctrain render arguments, one string (e.g. \"--cage-fade-start 1.5 --cage-fade-end 2.5\")")
a = ap.parse_args()
S = a.subject
run = Path(json.loads((S.parent / "subjects.json").read_text())[S.name])
splat = a.splat or run / "ply" / "scene.ply"
a.out.mkdir(parents=True, exist_ok=True)

from b2crig.motion import io as MI  # noqa: E402
mo = MI.load(a.motion); m = np.load(a.motion)
fps = mo.fps; T = len(mo)
body = MHRBody(S / "mhr.npz"); lay = layered.load_layers(S)
posed = layered.pose_motion(body, mo, lay, hair_params(a.hair))
names = [f"{i:04d}" for i in range(T)]
B.write_cage(a.out / "cage.b2ccage", layered.cage_layers(body, lay), posed, names)

try:
    K0, target0, _ = C.orbit_from_record(plyheader.read_orbit(run / "ply" / "scene.ply"))
except ValueError:   # no b2c.orbit header (the run died before embed_orbit_record; b2crunner's tools/recover_body)
    K0, target0, _ = C.orbit_from_colmap(run / "colmap")
s = a.height / K0.height
K = C.Intrinsics(a.width, a.height, K0.fx * s, K0.fy * s, a.width / 2, a.height / 2)
rw = (m["root_world"] if "root_world" in m.files else mo.root_t).copy(); rw[:, 1] = 0.0
k = int(round(0.6 * fps))
rw = np.stack([np.convolve(np.pad(rw[:, j], (k, k), mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), "valid") for j in range(3)], 1)
u = np.arange(T) / max(T - 1, 1)
e = u * u * (3 - 2 * u)
poses = []
for i in range(T):
    az = math.radians(a.az0 + (a.az1 - a.az0) * e[i]); el = math.radians(a.el)
    r = a.radius0 + (a.radius1 - a.radius0) * e[i]
    tgt = target0 + rw[i] + np.array([0.0, 0.08 * e[i], 0.0])   # drift up towards the face as the camera pushes in
    pos = tgt + r * np.array([math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)])
    poses.append((C.look_at(pos, tgt), pos))
C.write_cameras(a.out / "cameras.json", K, poses, names)
fr = a.out / "frames"
shutil.rmtree(fr, ignore_errors=True)
B.render(splat, a.out / "cameras.json", fr, cage=a.out / "cage.b2ccage",
         background=tuple(float(x) for x in a.bg.split(",")), extra=a.extra.split())
bg = np.array([float(x) for x in a.bg.split(",")])[::-1] * 255
comp = a.out / "comp"; comp.mkdir(exist_ok=True)
for nm in names:
    x = cv2.imread(str(fr / f"{nm}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
    al = x[..., 3:] / 255
    im = (x[..., :3] * al + bg * (1 - al)).astype(np.uint8)
    if a.label:
        cv2.putText(im, a.label, (24, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    cv2.imwrite(str(comp / f"{nm}.png"), im)
mp4 = a.out.with_suffix(".mp4")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{fps:g}", "-i", str(comp / "%04d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(mp4)], check=True)
print(f"render_final: {T} frames -> {mp4}")
