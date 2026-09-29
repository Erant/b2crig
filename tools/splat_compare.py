"""Does a retrained splat animate better? Render clips with the old and a new splat through each clip's own cage (LBS,
no fit) and score both against the clip's WAN frames.

    .venv/bin/python tools/splat_compare.py NEW_PLY CLIP_DIR [CLIP_DIR ...] [--old PLY]

Per clip: L1 vs WAN over the figure and inside the pixels where the two renders differ (> 12 levels), and the label
IoUs (upper, lower, hair, skin, silhouette) against the clip's segmentation.
"""
import argparse
import json
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
ap.add_argument("--old-clips", type=Path, help="as --new-clips, for the old splat")
ap.add_argument("--new-clips", type=Path, help="render the new splat through these clips' cages (a re-canonicalized "
                "subject's ported clips, tools/port_clips.py); the old one keeps its own")
a = ap.parse_args()
G = ["upper", "lower", "hair", "skin", "silhouette"]
tag = a.new.parent.name


def rgb(p):
    x = cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(np.float32)
    return x[..., :3] * x[..., 3:] / 255 + 127 * (1 - x[..., 3:] / 255)


print(f"{'clip':16s} {'L1 fig old/new':>16s} {'L1 chg old/new':>16s}   IoU old -> new ({', '.join(G)})")
tot = {"fo": [], "fn": [], "co": [], "cn": []}
for clip in a.clips:
    clip = clip.resolve(); S = clip.parents[1]
    run = Path(json.loads((S.parent / "subjects.json").read_text())[S.name])
    old = a.old or run / "ply" / "scene.ply"
    rd = {}
    for k, ply in (("old", old), ("new", a.new)):
        rd[k] = clip / f"cmp_{tag}_{k}"
        B.render(ply, clip / "cameras.json", rd[k], cage=(a.new_clips / clip.name / "cage.b2ccage") if (k == "new" and a.new_clips) else (a.old_clips / clip.name / "cage.b2ccage") if (k == "old" and a.old_clips) else cage_of(clip), label_maps=True, background=(0.5, 0.5, 0.5))
    fig = {"old": [], "new": []}; chg = {"old": [], "new": []}; iou = {"old": [], "new": []}; ghost = {"old": [], "new": []}
    for nm in evaluate.read_cage(cage_of(clip)).names:
        wan = cv2.imread(str(clip / "wan" / f"{nm}.png")).astype(np.float32)
        seg = cv2.imread(str(clip / "seg" / f"{nm}.png"), 0)
        im = {k: rgb(v / f"{nm}.png") for k, v in rd.items()}
        c = np.abs(im["new"] - im["old"]).max(-1) > 12
        for k in im:   # ghost: rendered where WAN shows background (left-behind pieces), as a share of WAN's silhouette
            al = cv2.imread(str(rd[k] / f"{nm}.png"), cv2.IMREAD_UNCHANGED)[..., 3] > 127
            ghost[k].append((al & (cv2.dilate((seg > 0).astype(np.uint8), np.ones((7, 7), np.uint8)) == 0)).sum() / max((seg > 0).sum(), 1))
            e = np.abs(im[k] - wan).mean(-1)
            fig[k].append(e[seg > 0].mean()); chg[k].append(e[c].mean() if c.sum() > 50 else np.nan)
            iou[k].append(evaluate.ious(cv2.imread(str(rd[k] / f"{nm}.labels.png"), 0), seg))
    m = {k: [np.nanmean([x[g] for x in iou[k]]) for g in G] for k in iou}
    print(f"{clip.name:16s} {np.mean(fig['old']):7.2f}/{np.mean(fig['new']):7.2f} {np.nanmean(chg['old']):7.2f}/{np.nanmean(chg['new']):7.2f}   "
          + "  ".join(f"{o:.3f}->{n:.3f}" for o, n in zip(m["old"], m["new"]))
          + f"   ghost% {100 * np.mean(ghost['old']):.2f}->{100 * np.mean(ghost['new']):.2f}", flush=True)
    tot["fo"].append(np.mean(fig["old"])); tot["fn"].append(np.mean(fig["new"])); tot["co"].append(np.nanmean(chg["old"])); tot["cn"].append(np.nanmean(chg["new"]))
print(f"{'mean':16s} {np.mean(tot['fo']):7.2f}/{np.mean(tot['fn']):7.2f} {np.mean(tot['co']):7.2f}/{np.mean(tot['cn']):7.2f}")
