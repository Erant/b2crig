"""Contact sheet of frames (RGBA composited on grey): tools/sheet.py OUT.jpg DIR [DIR ...] [--frames 0,10,20] [--h 400]
Each DIR is a row."""
import argparse
from pathlib import Path

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("dirs", nargs="+", type=Path)
ap.add_argument("--frames", default="0,10,20,30,40,50,60,70,80")
ap.add_argument("--h", type=int, default=400)
a = ap.parse_args()
rows = []
for d in a.dirs:
    ims = []
    for f in a.frames.split(","):
        p = d / f"{int(f):04d}.png"
        x = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if x is None:
            x = np.zeros((a.h, a.h // 2, 3), np.uint8)
        if x.shape[2] == 4:
            al = x[..., 3:] / 255.0
            x = (x[..., :3] * al + 127 * (1 - al)).astype(np.uint8)
        x = cv2.resize(x, (int(x.shape[1] * a.h / x.shape[0]), a.h))
        cv2.putText(x, f"{d.parent.name}/{d.name} {f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        ims.append(x)
    rows.append(np.hstack(ims))
w = max(r.shape[1] for r in rows)
rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
cv2.imwrite(a.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
