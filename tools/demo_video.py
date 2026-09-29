"""Side-by-side video of a clip rendered with the old and a new splat (through the clip's own cage) next to its WAN
frames: old | new | WAN, labelled, 16 fps.

    .venv/bin/python tools/demo_video.py NEW_PLY CLIP_DIR [CLIP_DIR ...] [--out DIR] [--old PLY] [--height 640]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2crig import b2ctrain as B  # noqa: E402
from b2crig import evaluate  # noqa: E402


def cage_of(clip: Path) -> Path:
    """The clip's cage under the current rig (cage_v2, rebuilt from its motion) when there is one."""
    return clip / "cage_v2.b2ccage" if (clip / "cage_v2.b2ccage").exists() else clip / "cage.b2ccage"


ap = argparse.ArgumentParser()
ap.add_argument("new", type=Path)
ap.add_argument("clips", nargs="+", type=Path)
ap.add_argument("--old", type=Path)
ap.add_argument("--new-clips", type=Path, help="render the new splat through these clips' cages (a re-canonicalized "
                "subject's ported clips, tools/port_clips.py); the old one keeps its own")
ap.add_argument("--out", type=Path)
ap.add_argument("--height", type=int, default=640)
a = ap.parse_args()
out = a.out or a.new.parent / "demo"; out.mkdir(parents=True, exist_ok=True)


def rgb(p):
    x = cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(np.float32)
    return (x[..., :3] * x[..., 3:] / 255 + 127 * (1 - x[..., 3:] / 255)).astype(np.uint8)


for clip in a.clips:
    clip = clip.resolve(); S = clip.parents[1]
    run = Path(json.loads((S.parent / "subjects.json").read_text())[S.name])
    rd = {}
    for k, ply in (("old", a.old or run / "ply" / "scene.ply"), ("new", a.new)):
        rd[k] = out / f"{clip.name}_{k}"
        if not rd[k].exists():
            B.render(ply, clip / "cameras.json", rd[k], cage=(a.new_clips / clip.name / "cage.b2ccage") if (k == "new" and a.new_clips) else cage_of(clip), background=(0.5, 0.5, 0.5))
    frames_dir = out / f"{clip.name}_sbs"; frames_dir.mkdir(exist_ok=True)
    for nm in evaluate.read_cage(cage_of(clip)).names:
        ims = [rgb(rd["old"] / f"{nm}.png"), rgb(rd["new"] / f"{nm}.png"), cv2.imread(str(clip / "wan" / f"{nm}.png"))]
        h = a.height; ims = [cv2.resize(x, (int(x.shape[1] * h / x.shape[0]) // 2 * 2, h)) for x in ims]
        for x, t in zip(ims, ("old splat", "retrained", "WAN (target)")):
            cv2.putText(x, t, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(str(frames_dir / f"{nm}.png"), np.hstack(ims))
    mp4 = out / f"{clip.name}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "16", "-i", str(frames_dir / "%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(mp4)], check=True)
    print(f"demo_video: {mp4}")
