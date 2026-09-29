"""Re-pose clips' saved motions with the subject's current layers (hair shell + dynamics included) -> cage_v2.b2ccage.

    .venv/bin/python tools/recage.py work/<subject> CLIP [CLIP ...] [--hair k_root=0.3,damping=4]

Frame names are kept, so cage_train / demo_video pick cage_v2 up in place of the clip's original cage.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2crig import b2ctrain as B  # noqa: E402
from b2crig import evaluate  # noqa: E402
from b2crig.rig import layered  # noqa: E402
from b2crig.rig.hair import HairParams  # noqa: E402
from b2crig.rig.mhr import MHRBody  # noqa: E402


def hair_params(spec: str):
    """None (the subject's hair.json / defaults) for an empty spec, else defaults overridden by k=v pairs."""
    if not spec:
        return None
    p = HairParams()
    for kv in filter(None, spec.split(",")):
        k, v = kv.split("=")
        setattr(p, k, type(getattr(p, k))(v))
    return p


def recage(S: Path, clip: Path, body: MHRBody, lay, hp: HairParams, out_name: str = "cage_v2.b2ccage") -> None:
    from b2crig.motion import io as MI
    m = MI.load(clip / "motion.npz")
    names = evaluate.read_cage(clip / "cage.b2ccage").names
    posed = layered.pose_motion(body, m, lay, hp)
    B.write_cage(clip / out_name, layered.cage_layers(body, lay), posed, names)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("subject", type=Path)
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--hair", default="")
    ap.add_argument("--out-name", default="cage_v2.b2ccage")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    body = MHRBody(a.subject / "mhr.npz", device=a.device)
    lay = layered.load_layers(a.subject)
    for c in a.clips:
        recage(a.subject, a.subject / "clips" / c, body, lay, hair_params(a.hair), a.out_name)
        print(f"recage: {c}")
