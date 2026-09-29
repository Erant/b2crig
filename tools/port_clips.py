"""Port clips of a subject to a re-canonicalized one (tools/recanonicalize.py): same frames, new cages.

    .venv/bin/python tools/port_clips.py work/<new> work/<old> CLIP [CLIP ...] [--capture]

Each clip dir under <new>/clips links the old clip's cameras.json, fit_ds, wan*, seg, motion.npz and gets a cage built
from its motion.npz (body params, expressions, root transform, hair dynamics: layered.pose_motion) with <new>'s layers. --capture adds a "capture" clip: the subject's
capture views (its run/colmap) as frames posed at the OLD canonical pose (<new>/old_canonical_params.npy), so the
capture trains the new canonical like any other posed clip.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2crig import b2ctrain as B  # noqa: E402
from b2crig import evaluate  # noqa: E402
from b2crig.motion import io as MI  # noqa: E402
from b2crig.rig import layered  # noqa: E402
from b2crig.rig.mhr import MHRBody  # noqa: E402
from b2crig.visibility import qvec_to_R  # noqa: E402

args = sys.argv[1:]; cap = "--capture" in args; args = [x for x in args if x != "--capture"]
N, O = Path(args[0]), Path(args[1]); names = args[2:]
body = MHRBody(N / "mhr.npz"); lay = layered.load_layers(N)


def write_cage(out: Path, params: np.ndarray, expr: np.ndarray | None, frames: list[str]) -> None:
    p = torch.as_tensor(params, device=body.device)
    e = torch.as_tensor(expr, device=body.device) if expr is not None else body.expr.expand(len(p), -1)
    posed = torch.cat([layered.pose_layers(body, body.pose(p[i:i + 16], expr=e[i:i + 16]), lay) for i in range(0, len(p), 16)]).cpu().numpy()
    B.write_cage(out, layered.cage_layers(body, lay), posed, frames)


for c in names:
    src, dst = O / "clips" / c, N / "clips" / c
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("cameras.json", "fit_ds", "wan", "wan_1080", "seg", "mask", "control", "wan.json", "motion.npz", "reference.png", "eval_fit.json"):
        if (src / f).exists() and not (dst / f).exists():
            (dst / f).symlink_to((src / f).resolve())
    # the clip builder's path (gen/clips.py): root transform (root_R/root_t, global_trans) + hair dynamics included
    posed = layered.pose_motion(body, MI.load(src / "motion.npz"), lay)
    B.write_cage(dst / "cage.b2ccage", layered.cage_layers(body, lay), posed, evaluate.read_cage(src / "cage.b2ccage").names)
    print(f"port_clips: {c}")
if cap:
    run = Path(json.loads((N.parent / "subjects.json").read_text())[N.name]); colmap = run / "colmap"
    dst = N / "clips" / "capture"; (dst / "fit_ds").mkdir(parents=True, exist_ok=True)
    for f in ("images", "labels", "cameras.txt", "images.txt", "points3D.txt"):
        if not (dst / "fit_ds" / f).exists():
            (dst / "fit_ds" / f).symlink_to((colmap / f).resolve())
    p = [l.split() for l in (colmap / "cameras.txt").read_text().splitlines() if l and not l.startswith("#")][0]
    W, H, fx, fy, cx, cy = int(p[2]), int(p[3]), *map(float, p[4:8])
    cams, frames = [], []
    for ln in (colmap / "images.txt").read_text().splitlines():
        f = ln.split()
        if ln.startswith("#") or len(f) < 10:
            continue
        Rw2c = qvec_to_R(list(map(float, f[1:5]))); t = np.array(list(map(float, f[5:8])))
        frames.append(Path(f[9]).stem)
        cams.append({"name": frames[-1], "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                     "rotation": (Rw2c.T @ np.diag([1., -1., -1.])).tolist(), "position": (-Rw2c.T @ t).tolist()})
    (dst / "cameras.json").write_text(json.dumps({"width": W, "height": H, "cameras": cams}))
    old = np.load(N / "old_canonical_params.npy").reshape(1, -1).repeat(len(frames), 0)
    np.savez(dst / "motion.npz", body_params=old, fps=16.0, motion="capture")
    write_cage(dst / "cage.b2ccage", old, None, frames)
    print(f"port_clips: capture ({len(frames)} views at the old canonical pose)")
