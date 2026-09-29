"""Turn a generated clip into a b2ctrain fit dataset (COLMAP text + RGBA frames + labels/ sidecar).

The WAN frames are on a flat grey ground; their alpha comes from the Sapiens2
segmentation (non-background classes), feathered by a small blur so the
silhouette term is not a hard staircase. Frame names are the cage frame names.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """(w, x, y, z) of a rotation matrix."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1) * 2
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


def write_colmap(cameras_json: Path, out: Path, image_ext: str = ".png") -> None:
    cams = json.loads(cameras_json.read_text())
    W, H = cams["width"], cams["height"]
    c0 = cams["cameras"][0]
    (out / "cameras.txt").write_text(f"1 PINHOLE {W} {H} {c0['fx']} {c0['fy']} {c0['cx']} {c0['cy']}\n")
    lines = []
    for i, c in enumerate(cams["cameras"]):
        c2w_cv = np.asarray(c["rotation"], float) @ np.diag([1.0, -1.0, -1.0])
        Rw2c = c2w_cv.T
        t = -Rw2c @ np.asarray(c["position"], float)
        q = rot_to_quat(Rw2c)
        lines.append(f"{i + 1} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} 1 {c['name']}{image_ext}\n\n")
    (out / "images.txt").write_text("".join(lines))
    (out / "points3D.txt").write_text("")


def build_dataset(clip: Path, feather_px: float = 1.5, seg: str = "seg", out: str = "fit_ds") -> Path:
    """`seg`: the label folder the alpha and labels/ come from (seg_rel: b2crig.relabel's change-only target)."""
    ds = clip / out
    if ds.exists():
        shutil.rmtree(ds)
    (ds / "images").mkdir(parents=True)
    (ds / "labels").mkdir()
    write_colmap(clip / "cameras.json", ds)
    for f in sorted((clip / "wan").glob("*.png")):
        rgb = cv2.imread(str(f), cv2.IMREAD_COLOR)
        lab = cv2.imread(str(clip / seg / f.name), cv2.IMREAD_GRAYSCALE)
        a = (lab > 0).astype(np.float32)
        if feather_px > 0:
            a = cv2.GaussianBlur(a, (0, 0), feather_px)
        cv2.imwrite(str(ds / "images" / f.name), np.dstack([rgb, np.clip(a * 255, 0, 255).astype(np.uint8)]))
        cv2.imwrite(str(ds / "labels" / f.name), lab)
    return ds
