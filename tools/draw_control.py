"""body2colmap control drawings of a posed body along a clip's cameras: b2crunner's `outline+skeleton` look (flat
faint-grey silhouette on #7F7F7F, DWPose-style OpenPose body25+hands skeleton, occlusion 0.12 m).

Runs in b2crunner's venv (it has body2colmap and pyrender)::

    PYOPENGL_PLATFORM=egl ~/Projects/b2crunner/.venv/bin/python tools/draw_control.py POSED.npz CAMERAS.json OUT_DIR
        [--frames 1-79] [--outline-strength 6.25]

POSED.npz: verts [T, V, 3], faces [F, 3], kps [T, 70, 3] (MHR70 keypoints), world frame. Writes OUT_DIR/NNNN.png (RGBA,
alpha 255: the drawing is the whole frame).
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from body2colmap.camera import Camera  # noqa: E402
from body2colmap.renderer import Renderer  # noqa: E402
from body2colmap.scene import Scene  # noqa: E402

BG = 0x7F


def grey(strength: float) -> float:
    v = round(BG * (1 - strength / 100.0))
    return (v + 0.5) / 255.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("posed", type=Path)
    ap.add_argument("cameras", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--frames", default="", help="a-b inclusive (default: all)")
    ap.add_argument("--outline-strength", type=float, default=6.25)
    ap.add_argument("--no-skeleton", action="store_true")
    a = ap.parse_args()
    d = np.load(a.posed)
    V, F, K = d["verts"], d["faces"].astype(np.int32), d["kps"]
    cj = json.loads(a.cameras.read_text())
    W, H = int(cj["width"]), int(cj["height"])
    cams = cj["cameras"]
    idx = range(len(cams))
    if a.frames:
        f0, f1 = (int(x) for x in a.frames.split("-"))
        idx = range(f0, f1 + 1)
    a.out.mkdir(parents=True, exist_ok=True)
    modes = {"outline": {"fg_color": (grey(a.outline_strength),) * 3, "bg_color": (grey(0.0),) * 3, "blur": 4}}
    if not a.no_skeleton:
        modes["skeleton"] = {"target_format": "openpose_body25_hands", "style": "dwpose", "joint_radius": 0.005,
                             "bone_radius": 0.005, "occlusion_tolerance": 0.12}
    for i in idx:
        c = cams[i]
        cam = Camera(focal_length=(c["fx"], c["fy"]), image_size=(W, H), principal_point=(c["cx"], c["cy"]),
                     position=np.asarray(c["position"], np.float32), rotation=np.asarray(c["rotation"], np.float32))
        sc = Scene(V[i].astype(np.float32), F, skeleton_joints=K[i].astype(np.float32), skeleton_format="mhr70")
        r = Renderer(sc, render_size=(W, H))
        img = r.render_composite(camera=cam, modes=modes)
        r.delete()
        out = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2BGR)
        out = np.dstack([out, np.full(out.shape[:2], 255, np.uint8)])
        cv2.imwrite(str(a.out / f"{c['name']}.png"), out)
    print(f"draw_control: {len(idx)} frames -> {a.out}")


if __name__ == "__main__":
    main()
