"""Re-fit finished clips with a (re)built cage: layered by default.

    .venv/bin/python tools/refit.py CLIP_DIR [CLIP_DIR ...] [--tag layered] [--body-only] [-- extra fit-cage args]

Rebuilds <clip>/cage_<tag>.b2ccage from the clip's motion.npz, runs fit-cage into <clip>/fit_<tag>, evaluates.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from b2crig import b2ctrain as B  # noqa: E402
from b2crig import evaluate  # noqa: E402
from b2crig.rig import layered  # noqa: E402
from b2crig.rig.mhr import MHRBody  # noqa: E402

_BODIES: dict = {}


def rebuild_cage(clip: Path, subject: Path, tag: str, use_layers: bool) -> Path:
    if subject not in _BODIES:
        _BODIES[subject] = MHRBody(subject / "mhr.npz")
    body = _BODIES[subject]
    lay = layered.load_layers(subject) if use_layers else []
    m = np.load(clip / "motion.npz")
    params = torch.as_tensor(m["body_params"], device=body.device)
    expr = torch.as_tensor(m["expr"], device=body.device) if "expr" in m.files else body.expr.expand(len(params), -1)
    posed = torch.cat([layered.pose_layers(body, body.pose(params[i:i + 16], expr=expr[i:i + 16]), lay)
                       for i in range(0, len(params), 16)]).cpu().numpy()
    names = evaluate.read_cage(clip / "cage.b2ccage").names
    out = clip / f"cage_{tag}.b2ccage"
    B.write_cage(out, layered.cage_layers(body, lay), posed, names)
    return out


def main() -> None:
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+", type=Path)
    ap.add_argument("--tag", default="layered")
    ap.add_argument("--body-only", action="store_true")
    ap.add_argument("--iters", default="8000")
    a = ap.parse_args(argv)
    for clip in a.clips:
        clip = clip.resolve()
        subject = clip.parent.parent
        run = Path(json.loads((ROOT / "work" / "subjects.json").read_text())[subject.name])
        splat = run / "ply" / "scene.ply"
        cage = rebuild_cage(clip, subject, a.tag, not a.body_only)
        subprocess.run([str(B.B2CTRAIN), "fit-cage", *B.fit_groups_args(), "--splat", str(splat), "--cage", str(cage), "--dataset", str(clip / "fit_ds"),
                        "--output", str(clip / f"fit_{a.tag}"), "--iters", a.iters, *extra], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        s = evaluate.evaluate(clip, splat, f"fit_{a.tag}", video=True, cage=cage.name)
        print(clip.name, {k: {g: round(v, 3) for g, v in d.items()} for k, d in s.items() if k != "lbs"}, flush=True)


if __name__ == "__main__":
    main()
