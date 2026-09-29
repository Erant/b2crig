"""Read b2ctrain/brush splat PLYs (with the b2crunner `seg_*` and `ev_*` extras)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from plyfile import PlyData

# Sapiens2 Goliath classes (b2crunner output/segsplat/common.py).
GOLIATH = (
    "Background", "Apparel", "Eyeglass", "Face_Neck", "Hair",
    "Left_Foot", "Left_Hand", "Left_Lower_Arm", "Left_Lower_Leg", "Left_Shoe", "Left_Sock",
    "Left_Upper_Arm", "Left_Upper_Leg", "Lower_Clothing",
    "Right_Foot", "Right_Hand", "Right_Lower_Arm", "Right_Lower_Leg", "Right_Shoe", "Right_Sock",
    "Right_Upper_Arm", "Right_Upper_Leg", "Torso", "Upper_Clothing",
    "Lower_Lip", "Upper_Lip", "Lower_Teeth", "Upper_Teeth", "Tongue",
)
LOOSE_CANDIDATES = (1, 4, 13, 23)   # Apparel, Hair, Lower_Clothing, Upper_Clothing


@dataclass
class Splat:
    xyz: np.ndarray          # [N, 3]
    scale_log: np.ndarray    # [N, 3]
    rot: np.ndarray          # [N, 4] wxyz, unnormalised (brush order rot_0..3)
    opacity_logit: np.ndarray
    seg_label: np.ndarray | None
    seg_conf: np.ndarray | None
    comments: list[str] = field(default_factory=list)

    @property
    def opacity(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.opacity_logit))


def load(path: str | Path) -> Splat:
    ply = PlyData.read(str(path))
    v = ply["vertex"].data
    names = v.dtype.names
    col = lambda *ks: np.stack([np.asarray(v[k], np.float32) for k in ks], 1)
    has_seg = "seg_label" in names
    return Splat(
        xyz=col("x", "y", "z"),
        scale_log=col("scale_0", "scale_1", "scale_2"),
        rot=col("rot_0", "rot_1", "rot_2", "rot_3"),
        opacity_logit=np.asarray(v["opacity"], np.float32),
        seg_label=np.rint(np.asarray(v["seg_label"])).astype(np.int32) if has_seg else None,
        seg_conf=np.asarray(v["seg_conf"], np.float32) if has_seg else None,
        comments=list(ply.comments),
    )


def save_subset(src: str | Path, idx: np.ndarray, dst: str | Path) -> None:
    """Write the vertices `idx` of a binary little-endian splat .ply (every property a float) to `dst`, header kept."""
    raw = Path(src).read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    head = raw[:end].decode("latin-1").splitlines()
    n_props = sum(1 for l in head if l.startswith("property"))
    n = int(next(l for l in head if l.startswith("element vertex")).split()[-1])
    data = np.frombuffer(raw, "<f4", n * n_props, end).reshape(n, n_props)[np.asarray(idx)]
    head = [f"element vertex {len(data)}" if l.startswith("element vertex") else l for l in head]
    Path(dst).write_bytes(("\n".join(head) + "\n").encode("latin-1") + data.astype("<f4").tobytes())
