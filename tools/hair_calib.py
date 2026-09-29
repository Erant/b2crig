"""Calibrate the hair dynamics (rig/hair.py) against WAN clips: for each parameter set, re-pose the clips' hair shell,
render labels through the cage and score the hair region against the Sapiens2 labels of the WAN frames.

    .venv/bin/python tools/hair_calib.py work/<subject> CLIP [CLIP ...] --sets "rigid:k_root=1,k_tip=1" "soft:k_tip=0.015" ...

Scores per set: hair IoU over the whole frame and over a band around the hair outline (where dynamics show), averaged
over the clips' frames after the first 8.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from b2crig import b2ctrain as B  # noqa: E402
from b2crig import evaluate  # noqa: E402
from b2crig.rig import layered  # noqa: E402
from b2crig.rig.mhr import MHRBody  # noqa: E402
from recage import hair_params, recage  # noqa: E402
from b2crig.rig.hair import HairParams  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("subject", type=Path)
ap.add_argument("clips", nargs="+")
ap.add_argument("--sets", nargs="+", required=True)
ap.add_argument("--band", type=int, default=20)
ap.add_argument("--splat", type=Path)
a = ap.parse_args()
S = a.subject
run = Path(json.loads((S.parent / "subjects.json").read_text())[S.name])
splat = a.splat or run / "ply" / "scene.ply"
body = MHRBody(S / "mhr.npz"); lay = layered.load_layers(S)
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.band + 1, 2 * a.band + 1))
res = {}
for spec in a.sets:
    tag, _, kv = spec.partition(":")
    sc = {"iou": [], "band": []}
    for c in a.clips:
        clip = S / "clips" / c
        recage(S, clip, body, lay, hair_params(kv) or HairParams(), f"cage_hair_{tag}.b2ccage")
        od = clip / f"hair_{tag}"
        shutil.rmtree(od, ignore_errors=True)
        B.render(splat, clip / "cameras.json", od, cage=clip / f"cage_hair_{tag}.b2ccage", label_maps=True)
        for i, nm in enumerate(evaluate.read_cage(clip / "cage.b2ccage").names):
            if i < 8:
                continue
            p = cv2.imread(str(od / f"{nm}.labels.png"), 0) == 4
            g = cv2.imread(str(clip / "seg" / f"{nm}.png"), 0) == 4
            u = (p | g).sum()
            sc["iou"].append((p & g).sum() / max(u, 1))
            band = (cv2.dilate(p.astype(np.uint8), k) - cv2.erode(p.astype(np.uint8), k)) | \
                   (cv2.dilate(g.astype(np.uint8), k) - cv2.erode(g.astype(np.uint8), k))
            band = band > 0
            ub = ((p | g) & band).sum()
            sc["band"].append(((p & g) & band).sum() / max(ub, 1))
        shutil.rmtree(od)
        (clip / f"cage_hair_{tag}.b2ccage").unlink()
    res[tag] = {k2: float(np.mean(v)) for k2, v in sc.items()}
    print(f"{tag:12s} {kv:40s} hair IoU {res[tag]['iou']:.4f}  band IoU {res[tag]['band']:.4f}", flush=True)
(S / "hair_calib.json").write_text(json.dumps(res, indent=1))
