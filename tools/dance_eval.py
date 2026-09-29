"""Score splat-animation methods against WAN clips: render each method through each clip's cage and compare with the
clip's WAN frames (frames WAN was told to keep, mask all 0, are skipped).

    .venv/bin/python tools/dance_eval.py CLIP_DIR [CLIP_DIR ...] --m NAME=PLY[,cage=FILE][,extra=ARGS] ... [--json OUT]

`cage=FILE` renders through CLIP_DIR/FILE instead of the clip's cage (e.g. a learned corrective's cage);
`extra=ARGS` is passed to b2ctrain render (space-separated, e.g. "--alt-binding b.bin").
Per clip and method:
  fig / face / skin   L1 vs WAN (0-255) over WAN's figure / face (Sapiens face, lips, teeth, tongue) / other skin
  grey         L1 where the control was greyed (grey_unseen clips) but the splat renders: the surface WAN redrew
               freely (stretched creases, unseen surface)
  iou          upper, lower, hair, skin, silhouette label IoUs vs the clip's segmentation
  ghost%       rendered (alpha > .5) where WAN shows background (7 px dilation), share of WAN's figure
  miss%        WAN figure the render leaves empty (alpha < .5, 7 px erosion), share of WAN's figure
  haze%        blooms without WAN: faint coverage (alpha .04-.5) outside the 7 px-dilated opaque region, share of
               the opaque area
Renders are cached in CLIP_DIR/ev_<NAME>/ (delete to re-render; --clean deletes them after scoring).
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2crig import b2ctrain as B  # noqa: E402
from b2crig import evaluate  # noqa: E402

FACE = (3, 24, 25, 26, 27, 28)
SKIN = tuple(c for c in evaluate.GROUPS["skin"] if c not in FACE)   # arms, legs, torso skin: where unseen surface shows
G = ["upper", "lower", "hair", "skin", "silhouette"]
K7 = np.ones((7, 7), np.uint8)


def parse_method(s: str) -> dict:
    name, _, rest = s.partition("=")
    parts = rest.split(",")
    m = {"name": name, "ply": parts[0], "cage": None, "extra": []}
    for p in parts[1:]:
        k, _, v = p.partition("=")
        m[k] = v.split() if k == "extra" else v
    return m


def rgb(im):
    x = im.astype(np.float32)
    return x[..., :3] * x[..., 3:] / 255 + 127 * (1 - x[..., 3:] / 255)


def score_clip(clip: Path, m: dict) -> dict:
    cage = clip / m["cage"] if m["cage"] else (clip / "cage_v2.b2ccage" if (clip / "cage_v2.b2ccage").exists() else clip / "cage.b2ccage")
    rd = clip / f"ev_{m['name']}"
    names = evaluate.read_cage(cage).names
    if not (rd / f"{names[-1]}.labels.png").exists():
        B.render(m["ply"], clip / "cameras.json", rd, cage=cage, label_maps=True, background=(0.5, 0.5, 0.5), extra=m["extra"])
    acc = {k: [] for k in ("fig", "face", "skin", "grey", "ghost", "miss", "haze")}
    ious = []
    for nm in names:
        mk = cv2.imread(str(clip / "mask" / f"{nm}.png"), 0)
        if mk is not None and mk.max() == 0:
            continue
        wan = cv2.imread(str(clip / "wan" / f"{nm}.png")).astype(np.float32)
        seg = cv2.imread(str(clip / "seg" / f"{nm}.png"), 0)
        im = cv2.imread(str(rd / f"{nm}.png"), cv2.IMREAD_UNCHANGED)
        al = im[..., 3].astype(np.float32) / 255
        fg = seg > 0
        e = np.abs(rgb(im) - wan).mean(-1)
        face = np.isin(seg, FACE)
        acc["fig"].append(e[fg].mean())
        acc["face"].append(e[face].mean() if face.sum() > 100 else np.nan)
        ct = cv2.imread(str(clip / "control" / f"{nm}.png"), cv2.IMREAD_UNCHANGED)
        gr = fg & (al > 0.5) & (ct[..., 3] < 128) if ct is not None and ct.shape[2] == 4 else np.zeros_like(fg)
        acc["grey"].append(e[gr].mean() if gr.sum() > 50 else np.nan)
        sk = np.isin(seg, SKIN)
        acc["skin"].append(e[sk].mean() if sk.sum() > 100 else np.nan)
        n = max(fg.sum(), 1)
        acc["ghost"].append(((al > 0.5) & (cv2.dilate(fg.astype(np.uint8), K7) == 0)).sum() / n)
        acc["miss"].append(((al < 0.5) & (cv2.erode(fg.astype(np.uint8), K7) > 0)).sum() / n)
        op = al > 0.5
        acc["haze"].append(((al > 0.04) & (al <= 0.5) & (cv2.dilate(op.astype(np.uint8), K7) == 0)).sum() / max(op.sum(), 1))
        ious.append(evaluate.ious(cv2.imread(str(rd / f"{nm}.labels.png"), 0), seg))
    out = {k: float(np.nanmean(v)) for k, v in acc.items()}
    for k in ("ghost", "miss", "haze"):
        out[k] *= 100
    out["iou"] = {g: float(np.nanmean([x[g] for x in ious])) for g in G}
    out["frames"] = len(ious)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+", type=Path)
    ap.add_argument("--m", action="append", required=True, help="NAME=PLY[,cage=FILE][,extra=ARGS]")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--clean", action="store_true", help="delete the renders after scoring (disk)")
    a = ap.parse_args()
    ms = [parse_method(s) for s in a.m]
    res = {}
    print(f"{'clip':22s} {'method':14s} {'fig':>6s} {'face':>6s} {'skin':>6s} {'grey':>6s} {'ghost%':>7s} {'miss%':>6s} {'haze%':>6s}  IoU " + " ".join(G))
    for clip in a.clips:
        clip = clip.resolve()
        res[clip.name] = {}
        for m in ms:
            r = score_clip(clip, m)
            if a.clean:
                shutil.rmtree(clip / f"ev_{m['name']}", ignore_errors=True)
            res[clip.name][m["name"]] = r
            print(f"{clip.name:22s} {m['name']:14s} {r['fig']:6.2f} {r['face']:6.2f} {r['skin']:6.2f} {r['grey']:6.2f} {r['ghost']:7.2f} {r['miss']:6.2f} {r['haze']:6.2f}  "
                  + " ".join(f"{r['iou'][g]:.3f}" for g in G), flush=True)
    if len(a.clips) > 1:
        for m in ms:
            v = [res[c][m["name"]] for c in res]
            print(f"{'MEAN':22s} {m['name']:14s} {np.mean([x['fig'] for x in v]):6.2f} {np.nanmean([x['face'] for x in v]):6.2f} {np.nanmean([x['skin'] for x in v]):6.2f} {np.nanmean([x['grey'] for x in v]):6.2f} "
                  f"{np.mean([x['ghost'] for x in v]):7.2f} {np.mean([x['miss'] for x in v]):6.2f} {np.mean([x['haze'] for x in v]):6.2f}  "
                  + " ".join(f"{np.nanmean([x['iou'][g] for x in v]):.3f}" for g in G))
    if a.json:
        a.json.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
