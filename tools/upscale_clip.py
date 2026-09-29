"""SeedVR2 upscale of a clip's WAN frames, as b2crunner's helical workflow upscales its capture frames.

Runs in b2crunner's seedvr2 venv::

    ~/Projects/b2crunner/pipeline/envs/seedvr2/venv/bin/python tools/upscale_clip.py CLIP_DIR [--resolution 1080]

Reads wan/NNNN.png, writes wan_<resolution>/NNNN.png (shortest edge = resolution; the workflow's settings: batch 1,
tiled VAE encode and decode).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, "/opt/b2c_runner" if Path("/opt/b2c_runner/pipeline").exists() else str(Path.home() / "Projects" / "b2crunner"))
import pipeline.steps  # noqa: E402,F401
from pipeline.registry import get_step_class  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", type=Path)
    ap.add_argument("--resolution", type=int, default=1080)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    frames = sorted(f for f in (a.clip / "wan").glob("*.png") if f.stem.isdigit())
    cls = get_step_class("seedvr2")
    params = cls.resolve_params({"resolution": a.resolution, "batch_size": 1, "vae_encode_tiled": True,
                                 "vae_decode_tiled": True, "seed": a.seed})
    # the caller rescales its own intrinsics (tools/reveal_train.py): hand the step placeholders and skip its helper
    import pipeline.steps.seedvr2 as sv
    sv._fit_cameras_to_images = lambda cams, ims: (cams, (ims[0].shape[1], ims[0].shape[0]))
    step = cls()
    step.load(params)
    out = step.run({"images": [cv2.imread(str(f), cv2.IMREAD_COLOR) for f in frames], "cameras": [None] * len(frames)}, params)["images"]
    od = a.clip / f"wan_{a.resolution}"; od.mkdir(exist_ok=True)
    for f, im in zip(frames, out):
        cv2.imwrite(str(od / f.name), im)
    print(f"upscale_clip: {len(out)} frames -> {out[0].shape[1]}x{out[0].shape[0]} in {od}")


if __name__ == "__main__":
    main()
