"""Split the hand splats that straddle a finger joint into one child per bone (b2crig/rig/joint_split.py), as the
warm start of a pose containment fine-tune with finger poses.

    .venv/bin/python tools/split_joints.py work/<subject> IN_PLY OUT_PLY [--k-sigma 2] [--min-mass 0.05] [--passes 3]

then  tools/cage_train.py work/<subject> --init OUT_PLY --contain poseset_romh16:1 --contain-hands 0.3 --tag T
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2crig.rig.joint_split import split_joints  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("subject", type=Path)
ap.add_argument("splat", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--k-sigma", type=float, default=2.0, help="split when the plane is within this many sigma of the centre")
ap.add_argument("--min-mass", type=float, default=0.05, help="... and the smaller half holds at least this share")
ap.add_argument("--passes", type=int, default=3)
a = ap.parse_args()
rep = split_joints(a.splat, a.subject / "mhr.npz", a.out, k_sigma=a.k_sigma, min_mass=a.min_mass, passes=a.passes)
print(json.dumps(rep))
