"""Label targets that carry only what WAN changed, not the labeller's conventions.

Sapiens2 draws class boundaries differently from the splat's baked seg_label (on the LBS render itself it agrees with
the rendered labels only to IoU ~0.93 upper / 0.91 hair), so fitting to Sapiens-on-WAN also fits that convention gap,
which is view-dependent. The change-only target is the rendered LBS labels, replaced by Sapiens-on-WAN only where
Sapiens itself changes its mind between the LBS render (seg_control/) and the WAN frame (seg/).

    python -m b2crig.relabel CLIP_DIR [...]      -> <clip>/seg_rel/NNNN.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2


def change_only(clip: Path, out: str = "seg_rel") -> Path:
    o = clip / out
    o.mkdir(exist_ok=True)
    for f in sorted((clip / "seg").glob("*.png")):
        wan = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        ctrl = cv2.imread(str(clip / "seg_control" / f.name), cv2.IMREAD_GRAYSCALE)
        rend = cv2.imread(str(clip / "eval_lbs" / f"{f.stem}.labels.png"), cv2.IMREAD_GRAYSCALE)
        tgt = rend.copy()
        ch = wan != ctrl
        tgt[ch] = wan[ch]
        cv2.imwrite(str(o / f.name), tgt)
    return o


if __name__ == "__main__":
    for c in sys.argv[1:]:
        print(change_only(Path(c)))
