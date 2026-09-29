"""Circular orbit cameras in body2colmap's cameras.json terms.

`rotation` is OpenGL camera-to-world (columns: right, up, back; the camera looks
down -Z), `position` the centre, intrinsics in pixels. b2ctrain's
`camera_from_json` reads exactly this. World up is +Y (the splats' frame).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


def look_at(position: np.ndarray, target: np.ndarray, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    back = position - target
    back = back / np.linalg.norm(back)
    right = np.cross(np.asarray(up, float), back)
    right = right / np.linalg.norm(right)
    true_up = np.cross(back, right)
    return np.stack([right, true_up, back], 1)


def circular_orbit(target: Sequence[float], radius: float, elevation_deg: float, n: int,
                   azimuth0_deg: float = 0.0, sweep_deg: float = 360.0) -> list[tuple[np.ndarray, np.ndarray]]:
    """`n` (rotation, position) pairs on a fixed-elevation circle; azimuth 0 looks from +Z (the front)."""
    t = np.asarray(target, float)
    el = math.radians(elevation_deg)
    out = []
    for i in range(n):
        az = math.radians(azimuth0_deg + sweep_deg * i / max(n - 1 if sweep_deg < 360 else n, 1))
        pos = t + radius * np.array([math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)])
        out.append((look_at(pos, t), pos))
    return out


def helical_orbit(target: Sequence[float], radius: float, el0_deg: float, el1_deg: float, n: int,
                  azimuth0_deg: float = 0.0, sweep_deg: float = 360.0) -> list[tuple[np.ndarray, np.ndarray]]:
    """`n` poses on one turn whose elevation goes linearly from el0 to el1 (more view directions per surface point
    than a circle: constrains the colour's view dependence out of the orbit plane)."""
    out = []
    for i in range(n):
        el = el0_deg + (el1_deg - el0_deg) * i / max(n - 1, 1)
        az = azimuth0_deg + sweep_deg * i / max(n - 1 if sweep_deg < 360 else n, 1)
        out += circular_orbit(target, radius, el, 1, az, 0.0)
    return out


def write_cameras(path: str | Path, K: Intrinsics, poses: Sequence[tuple[np.ndarray, np.ndarray]],
                  names: Sequence[str]) -> None:
    cams = [{"name": nm, "fx": K.fx, "fy": K.fy, "cx": K.cx, "cy": K.cy,
             "rotation": np.asarray(R).tolist(), "position": np.asarray(p).tolist()}
            for (R, p), nm in zip(poses, names)]
    Path(path).write_text(json.dumps({"width": K.width, "height": K.height, "cameras": cams}))


def orbit_from_record(record: dict) -> tuple[Intrinsics, np.ndarray, float]:
    """Intrinsics, orbit target and radius of the subject's own orbit cameras (b2c.orbit header)."""
    oc = record["orbit_cameras"]
    W, H = (int(x) for x in np.asarray(oc["image_size"]))
    fx, fy, cx, cy = (float(x) for x in np.asarray(oc["intrinsics"])[0])
    target = np.asarray(record["extras"]["orbit_target"], float)
    radius = float(np.median(np.linalg.norm(np.asarray(oc["position"]) - target, axis=1)))
    return Intrinsics(W, H, fx, fy, cx, cy), target, radius


def orbit_from_colmap(ds: Path) -> tuple[Intrinsics, np.ndarray, float]:
    """The same from the run's COLMAP model, for a scene.ply without the b2c.orbit header: the target is the point
    nearest (least squares) to every camera's optical axis."""
    from .visibility import colmap_cams
    (W, H, fx, fy, cx, cy), cams = colmap_cams(Path(ds))
    A, b, C = np.zeros((3, 3)), np.zeros(3), []
    for Rw2c, t in cams:
        c, d = -Rw2c.T @ t, Rw2c[2]          # centre, viewing direction (OpenCV +z)
        P = np.eye(3) - np.outer(d, d)
        A += P; b += P @ c; C.append(c)
    target = np.linalg.solve(A, b)
    radius = float(np.median(np.linalg.norm(np.asarray(C) - target, axis=1)))
    return Intrinsics(W, H, fx, fy, cx, cy), target, radius
