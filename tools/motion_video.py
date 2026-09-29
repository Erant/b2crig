"""A WAN clip that INVENTS the motion: frame 0 is the subject's canonical render (kept), every other frame is free
(grey control, mask 255) under a prompt describing the action. The result is a natural-looking motion of this very
character, which tools/video_fit.py turns into MHR poses (SAM-3D-Body) and tools/retarget_video.py into a motion file.

    .venv/bin/python tools/motion_video.py work/<subject> CLIP_ID --action "..." [--az 20 --el 4 --radius 4.2]
        [--seed 0] [--width 720 --height 1280] [--frames 81] [--first PNG]

Static camera. Then run WAN on it (clip_queue does, or tools/wan_clip.py in the wan22 venv) and tools/seg_clip.py.
`--first`: use this image as the kept frame instead of the canonical render (to chain clips: the last frame of the
previous one).
"""
import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.home() / "Projects" / "b2crunner"))
from pipeline.orbit_record import read_orbit_record  # noqa: E402

from b2crig import b2ctrain as B  # noqa: E402
from b2crig import cameras as C  # noqa: E402
from b2crig.gen.clips import NEGATIVE  # noqa: E402

PROMPT = ("A static camera on a tripod, fixed framing and focal length, films $SUBJECT_DESC$ in an empty studio with a "
          "plain grey background and a plain grey floor. {action} Natural, fluid, realistic human motion with correct "
          "weight shift and timing; the clothing and hair move naturally. Soft, fixed studio lighting. The face, "
          "clothing, colours and materials stay identical in every frame. Photorealistic, sharp focus, full body in frame.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("subject", type=Path)
    ap.add_argument("clip")
    ap.add_argument("--action", required=True)
    ap.add_argument("--az", type=float, default=20.0)
    ap.add_argument("--el", type=float, default=4.0)
    ap.add_argument("--radius", type=float, default=4.2)
    ap.add_argument("--target-dy", type=float, default=0.0, help="raise the look-at point (m)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=1280)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--first", type=Path)
    a = ap.parse_args()
    S = a.subject
    run = Path(json.loads((S.parent / "subjects.json").read_text())[S.name])
    out = S / "clips" / a.clip
    out.mkdir(parents=True, exist_ok=True)
    rec = read_orbit_record(run / "ply" / "scene.ply")
    K0, target, _ = C.orbit_from_record(rec)
    s = a.height / K0.height
    K = C.Intrinsics(a.width, a.height, K0.fx * s, K0.fy * s, a.width / 2, a.height / 2)
    target = target + np.array([0.0, a.target_dy, 0.0])
    az, el = math.radians(a.az), math.radians(a.el)
    pos = target + a.radius * np.array([math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)])
    names = [f"{i:04d}" for i in range(a.frames)]
    C.write_cameras(out / "cameras.json", K, [(C.look_at(pos, target), pos)] * a.frames, names)
    ctrl = out / "control"
    shutil.rmtree(ctrl, ignore_errors=True); ctrl.mkdir()
    (out / "mask").mkdir(exist_ok=True)
    if a.first:
        im = cv2.imread(str(a.first), cv2.IMREAD_COLOR)
        cv2.imwrite(str(ctrl / "0000.png"), np.dstack([im, np.full(im.shape[:2], 255, np.uint8)]))
    else:
        tmp = out / "first"
        C.write_cameras(out / "first_cam.json", K, [(C.look_at(pos, target), pos)], ["0000"])
        B.render(run / "ply" / "scene.ply", out / "first_cam.json", tmp, background=(0.5, 0.5, 0.5))
        shutil.copy(tmp / "0000.png", ctrl / "0000.png"); shutil.rmtree(tmp)
    blank = np.zeros((a.height, a.width, 4), np.uint8)
    for i, nm in enumerate(names):
        if i:
            cv2.imwrite(str(ctrl / f"{nm}.png"), blank)
        cv2.imwrite(str(out / "mask" / f"{nm}.png"), np.full((a.height, a.width), 0 if i == 0 else 255, np.uint8))
    # identity reference = the kept first frame itself (front.png's framing and purple border get pasted in by VACE)
    fr0 = cv2.imread(str(ctrl / "0000.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
    al = fr0[..., 3:] / 255.0
    cv2.imwrite(str(out / "reference.png"), (fr0[..., :3] * al + 127.5 * (1 - al)).astype(np.uint8))
    wan = {"background": [0.5, 0.5, 0.5], "subject_desc": rec.get("prompt"),
           "params": {"width": a.width, "height": a.height, "steps_high": 2, "steps_low": 2, "cfg": 1.0, "seed": a.seed,
                      "low_vram": True, "sampler_high": "euler", "sampler_low": "euler", "sampler_shift": 2.5,
                      "strength": [1.0, 1.0, 1.0, 1.0],
                      "prompt": PROMPT.format(action=a.action), "negative_prompt": NEGATIVE + ", camera motion, zoom, cut"}}
    (out / "wan.json").write_text(json.dumps(wan, indent=1, ensure_ascii=False))
    print(f"motion_video: {out}")


if __name__ == "__main__":
    main()
