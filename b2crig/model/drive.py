"""Drive a subject with a motion through the trained model: motion -> cage (+ predicted displacement) -> b2ctrain render.

    python -m b2crig.model.drive --subject work/416580 --model work/416580/model/model.pt --motion rand42 --out work/416580/drive/rand42

Writes the LBS-only and the model-driven renders side by side as <out>/<motion>.mp4 (orbit camera as the clips use).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch


from .. import b2ctrain as B  # noqa: E402
from .. import cameras as C  # noqa: E402
from .. import plyheader  # noqa: E402
from ..motion import procedural  # noqa: E402
from ..rig import layered  # noqa: E402
from ..rig.mhr import MHRBody  # noqa: E402
from . import data as D  # noqa: E402
from .train import MotionToCage  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--motion", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--fps", type=float, default=16.0)
    ap.add_argument("--elevation", type=float, default=5.0)
    ap.add_argument("--sweep", type=float, default=360.0)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=848)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    run = Path(json.loads((a.subject.parent / "subjects.json").read_text())[a.subject.name])
    ply = run / "ply" / "scene.ply"
    K, target, radius = C.orbit_from_record(plyheader.read_orbit(ply))
    s = a.width / K.width
    K = C.Intrinsics(a.width, a.height, K.fx * s, K.fy * a.height / K.height, K.cx * s, K.cy * a.height / K.height)
    names = [f"{i:04d}" for i in range(a.frames)]
    C.write_cameras(a.out / "cameras.json", K, C.circular_orbit(target, radius, a.elevation, a.frames, 0.0, a.sweep), names)

    body = MHRBody(a.subject / "mhr.npz"); lay = layered.load_layers(a.subject); bi = D.cage_body_index(body, a.subject)
    ck = torch.load(a.model, weights_only=False)
    model = MotionToCage(ck["d_in"], ck["n_verts"], ck["rank"]).cuda(); model.load_state_dict(ck["state"]); model.eval()
    params = body.body0 + torch.as_tensor(procedural.get(a.motion).offsets(a.frames, a.fps), device="cuda")
    posed_v, rots, joints, frames = [], [], [], []
    for i in range(0, a.frames, 16):
        p = body.pose(params[i:i + 16])
        posed_v.append(layered.pose_layers(body, p, lay)); rots.append(p.rots.cpu().numpy()); joints.append(p.joints.cpu().numpy())
        frames.append(D.vertex_frames(body, p, bi))
    posed = torch.cat(posed_v); Rv = torch.cat(frames)
    feats = D.skeleton_features(np.concatenate(rots), np.concatenate(joints), a.fps)
    x = torch.as_tensor((D.windows(feats, ck["window"]) - ck["mu"]) / ck["sd"], device="cuda", dtype=torch.float32)
    with torch.no_grad():
        dl = model(x)
    delta = torch.einsum("tvij,tvj->tvi", Rv, dl)
    L = layered.cage_layers(body, lay)
    B.write_cage(a.out / "lbs.b2ccage", L, posed.cpu().numpy(), names)
    B.write_cage(a.out / "model.b2ccage", L, (posed + delta).cpu().numpy(), names)
    for tag in ("lbs", "model"):
        B.render(ply, a.out / "cameras.json", a.out / tag, cage=a.out / f"{tag}.b2ccage", background=(0.5, 0.5, 0.5))
    out = a.out / f"{a.motion}.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{2 * a.width}x{a.height}",
                           "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)], stdin=subprocess.PIPE)
    for nm in names:
        row = []
        for tag, title in (("lbs", "LBS only"), ("model", "LBS + model")):
            im = cv2.imread(str(a.out / tag / f"{nm}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
            al = im[..., 3:] / 255
            im = (im[..., :3] * al + 127 * (1 - al)).astype(np.uint8)
            cv2.putText(im, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            row.append(im)
        ff.stdin.write(np.concatenate(row, 1).tobytes())
    ff.stdin.close(); ff.wait()
    print(out, "mean |delta| mm", float(delta.norm(dim=-1).mean()) * 1000)


if __name__ == "__main__":
    main()
