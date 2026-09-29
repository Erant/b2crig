"""Cage-only clips of procedural motions (no render, no WAN): a pose library for b2ctrain's stretch regulariser
(tools/cage_train.py --library).

    .venv/bin/python tools/pose_library.py work/<subject> rand1 rand2 hold_rand1 ... [--device cpu]
-> <subject>/clips/lib_<motion>/cage.b2ccage
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2crig import b2ctrain as B  # noqa: E402
from b2crig.motion import procedural  # noqa: E402
from b2crig.rig import layered  # noqa: E402
from b2crig.rig.mhr import MHRBody  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("subject", type=Path)
ap.add_argument("motions", nargs="+")
ap.add_argument("--frames", type=int, default=81)
ap.add_argument("--device", default="cpu")
a = ap.parse_args()
body = MHRBody(a.subject / "mhr.npz", device=a.device); lay = layered.load_layers(a.subject)

for mn in a.motions:
    mot = procedural.get(mn)
    params = body.body0 + torch.as_tensor(mot.offsets(a.frames, 16.0), device=body.device)
    expr = body.expr + torch.as_tensor(mot.expr_offsets(a.frames, 16.0), device=body.device)
    posed = torch.cat([layered.pose_layers(body, body.pose(params[i:i + 16], expr=expr[i:i + 16]), lay)
                       for i in range(0, a.frames, 16)]).cpu().numpy()
    posed = layered.hair_dynamics(body, params, expr, None, posed, lay, 16.0)
    out = a.subject / "clips" / f"lib_{mn}"; out.mkdir(parents=True, exist_ok=True)
    B.write_cage(out / "cage.b2ccage", layered.cage_layers(body, lay), posed, [f"{i:04d}" for i in range(a.frames)])
    print(f"pose_library: {out}")
