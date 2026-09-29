"""Solve the walk/stop/wave choreography for a subject and save it.

    .venv/bin/python tools/make_walk.py work/<subject> OUT.npz [--fps 30] [--iters 600] [--device cpu] [--no-wave]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2crig.motion import walk  # noqa: E402
from b2crig.rig.mhr import MHRBody  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("subject", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--fps", type=float, default=30.0)
ap.add_argument("--iters", type=int, default=600)
ap.add_argument("--device", default="cpu")
ap.add_argument("--steps", type=int, default=7)
ap.add_argument("--no-wave", action="store_true")
a = ap.parse_args()
body = MHRBody(a.subject / "mhr.npz", device=a.device)
t0 = time.time()
ch = walk.walk_stop_wave(body, fps=a.fps, n_steps=a.steps, wave=not a.no_wave)
print(f"choreo: {len(ch.t)} frames ({ch.t[-1]:.1f} s) in {time.time() - t0:.1f}s", flush=True)
m = walk.solve(body, ch, iters=a.iters)
a.out.parent.mkdir(parents=True, exist_ok=True)
np.savez(a.out, **m, ankle_r=ch.ankle["r"], ankle_l=ch.ankle["l"], toe_r=ch.toe["r"], toe_l=ch.toe["l"], pelvis=ch.pelvis)
print(f"wrote {a.out} in {time.time() - t0:.0f}s")
