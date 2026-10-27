"""Self-distilled training targets for the rig's fine-tune (cage_train --distill).

The containment fine-tune (cage_train --contain) retrains the splat it rigs, warm, with a fresh optimiser. On the
capture frames it re-learns the colours from them without what the delivered splat's own training had (b2crunner's
alignment loop, its per-view body rig, its detail passes), and that costs: b2crunner 8180cc, body sharpness at views
between the capture cameras 18.3 -> 12.2, 30.1 dB against the delivered splat; the face detailer's head 98 -> 54.
With the delivered splat's own renders as the targets, at the same cameras, the fine-tune only does what the
containment loss asks of it: 16.2, 36.3 dB; the head 92. Containment holds: leak outside the posed silhouette on two
held-out motions 7.0 / 4.2 without the fine-tune, 0.1 / 0.1 on the capture's frames, 0.5 / 0.5 distilled; hand
close-ups 1019 / 710 / 551.

`render_targets(ds, splat, out, alpha)` renders `splat` at every image of the COLMAP model `ds` (each with its own lens)
and writes out/<image name>: RGBA, straight colour (rendered on black, un-premultiplied), the alpha the view image's
own matte (`alpha="image"`, where the image has one: b2crunner's rmbg is tighter than the splat's fringe, 0.5 against
0.7 leak) or the render's.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

from .. import b2ctrain as B
from ..cameras import Intrinsics, write_cameras
from ..visibility import colmap_lenses


def render_targets(ds: Path, splat: Path, out: Path, alpha: str = "image") -> int:
    lenses, images = colmap_lenses(ds)
    out.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[int, int], list] = {}
    for cid, Rw2c, t, name in images:
        groups.setdefault(lenses[cid][:2], []).append((cid, Rw2c, t, name))
    for gi, ((W, H), views) in enumerate(groups.items()):
        tmp = out / f"_render{gi}"
        tmp.mkdir()
        per = [Intrinsics(*lenses[cid]) for cid, *_ in views]
        poses = [(Rw2c.T @ np.diag([1.0, -1.0, -1.0]), -Rw2c.T @ t) for _, Rw2c, t, _ in views]
        write_cameras(tmp / "cameras.json", per[0], poses, [Path(n).stem for *_, n in views], per_view=per)
        B.render(splat, tmp / "cameras.json", tmp, background=(0.0, 0.0, 0.0))
        for *_, name in views:
            rgba = cv2.imread(str(tmp / f"{Path(name).stem}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
            a = rgba[..., 3:] / 255.0
            rgb = np.where(a > 0, rgba[..., :3] / np.maximum(a, 1e-6), 0.0)
            al = rgba[..., 3:]
            if alpha == "image":   # the view's own matte (b2crunner's rmbg) where it has one
                own = cv2.imread(str(ds / "images" / name), cv2.IMREAD_UNCHANGED)
                if own is not None and own.ndim == 3 and own.shape[2] == 4:
                    al = own[..., 3:].astype(np.float32)
            cv2.imwrite(str(out / Path(name).with_suffix(".png").name),
                        np.dstack([np.clip(rgb + 0.5, 0, 255), al]).astype(np.uint8))
        shutil.rmtree(tmp)
    return len(images)
