"""A face-crop identity reference for close-up clips: Sapiens2 face/hair box of the subject's front.png, squared with a
margin and upscaled (Lanczos) to 768 px. Runs in b2crunner's wan22 venv::

    ~/Projects/b2crunner/pipeline/envs/wan22/venv/bin/python tools/face_reference.py work/<subject>   -> <subject>/face_ref.png
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path.home() / "Projects" / "b2crunner"))
import pipeline.steps  # noqa: E402,F401
from pipeline.registry import get_step_class  # noqa: E402

S = Path(sys.argv[1])
run = Path(json.loads((S.parent / "subjects.json").read_text())[S.name])
img = cv2.imread(str(run / "ply" / "front.png"), cv2.IMREAD_COLOR)
cls = get_step_class("sapiens2_seg"); params = cls.resolve_params({"dtype": "bfloat16"}); step = cls(); step.load(params)
lab = step.run({"images": [img]}, params)["labels"][0]
ys, xs = np.nonzero(np.isin(lab, (3, 4, 24, 25)))          # face, hair, lips
face_y = np.nonzero(np.isin(lab, (3,)))[0]
y0, y1 = ys.min(), face_y.max()                              # hair top to the chin
x0, x1 = xs.min(), xs.max()
side = int(max(y1 - y0, x1 - x0) * 1.35); cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
top, left = max(cy - side // 2, 0), max(cx - side // 2, 0)
crop = img[top:top + side, left:left + side]
ref = cv2.resize(crop, (768, 768), interpolation=cv2.INTER_LANCZOS4)
cv2.imwrite(str(S / "face_ref.png"), ref)
print(f"face_reference: {side}px box at ({left},{top}) -> {S / 'face_ref.png'}")
