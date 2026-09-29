"""Add a facial performance to a motion: a smile that grows once the walk stops, eyes and brows that lift with the
wave, blinks every ~3 s (not during the smile's onset). Timing comes from the motion itself (root speed, right-wrist
height).

    .venv/bin/python tools/add_face.py work/<subject> IN.npz OUT.npz [--smile 0.8] [--device cpu]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2crig.motion import io as MI  # noqa: E402
from b2crig.motion.procedural import FACE_DIMS  # noqa: E402
from b2crig.rig.mhr import MHRBody  # noqa: E402
from b2crig.rig.skeleton import JOINT  # noqa: E402


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


ap = argparse.ArgumentParser()
ap.add_argument("subject", type=Path)
ap.add_argument("inp", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--smile", type=float, default=0.8)
ap.add_argument("--device", default="cpu")
a = ap.parse_args()
body = MHRBody(a.subject / "mhr.npz", device=a.device)
raw = dict(np.load(a.inp))
m = MI.Motion(raw)
T, fps = len(m), m.fps
t = np.arange(T) / fps
J = MI.pose(body, m, 0, T).joints.cpu().numpy()
pel = gaussian_filter1d(J[:, JOINT["pelvis"]], 0.2 * fps, axis=0, mode="nearest")
speed = np.linalg.norm(np.gradient(pel[:, [0, 2]], axis=0), axis=1) * fps
moving = speed > 0.15
t_stop = t[np.nonzero(moving)[0][-1]] if moving.any() else t[0]
wr = J[:, JOINT["r_wrist"]][:, 1] - J[:, JOINT["r_shoulder"]][:, 1]
up = wr > -0.05
t_wave = t[np.nonzero(up)[0][0]] if up.any() else t_stop + 0.5
t_smile = min(t_stop - 0.3, t_wave - 0.4)
print(f"add_face: stop at {t_stop:.2f} s, wave from {t_wave:.2f} s, smile from {t_smile:.2f} s")
ex = np.zeros((T, 72), np.float32)
s_on = smooth((t - t_smile) / 0.7)
for d in FACE_DIMS["smile"]:
    ex[:, d] += a.smile * s_on
for d in FACE_DIMS["grin"]:                                 # a little width; more smears the lip seam
    ex[:, d] += 0.3 * a.smile * s_on
b_on = smooth((t - t_wave + 0.2) / 0.4) * (1 - smooth((t - t_wave - 1.6) / 0.6))
for d in FACE_DIMS["brows"]:
    ex[:, d] += 0.25 * b_on
blinks = [tb for tb in np.arange(1.1, t[-1] - 0.2, 2.9) if abs(tb - t_smile) > 0.5]
for tb in blinks:
    pulse = np.exp(-0.5 * ((t - tb) / 0.06) ** 2)
    for d in FACE_DIMS["blink"]:
        ex[:, d] += 0.95 * pulse
raw["expr"] = (body.expr[0].cpu().numpy()[None] + ex).astype(np.float32)
np.savez(a.out, **raw)
print(f"add_face: {len(blinks)} blinks -> {a.out}")
