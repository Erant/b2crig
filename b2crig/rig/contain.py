"""Inputs of b2ctrain's pose containment loss (--pose-contain-weight, b2ctrain src/gpu/pose_contain.h).

b2ctrain renders the warm-start splat posed by a cage frame, with the splats deep inside the body hidden, and trains the
alpha outside that silhouette to 0. Which poses, from where, and which splats count as interior are decisions about the
rig, so they are made here (they used to be made inside b2ctrain) and handed over as two files:

  contain_cameras   cameras.json (body2colmap terms): n views over the cage's frames, spread evenly, each seen from one
                    of the capture cameras' offsets from the canonical body centre, re-applied to the posed body centre
                    (the body = cage layer 0, the MHR body). Named "<frame>@c<k>": b2ctrain poses each by <frame>.
  deep_mask         uint8 per splat of the warm start: 1 = more than `depth` behind the nearest body vertex along its
                    normal (area-weighted vertex normals of layer 0).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

from .. import cameras as C
from ..evaluate import Cage
from ..visibility import colmap_cams


def _body(cg: Cage) -> tuple[int, int, np.ndarray]:
    v_off, v_count, f_off, f_count, _ = cg.layers[0]
    F = cg.faces[f_off:f_off + f_count].astype(np.int64) - v_off
    F = F[((F >= 0) & (F < v_count)).all(1)]
    return v_off, v_count, F


def body_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n = np.zeros_like(V)
    for k in range(3):
        np.add.at(n, F[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)


def deep_mask(splat: str | Path, cg: Cage, depth: float = 0.02) -> np.ndarray:
    """uint8 [n]: splats more than `depth` behind the nearest body vertex (signed along its normal)."""
    v_off, v_count, F = _body(cg)
    V = cg.verts[v_off:v_off + v_count].astype(np.float32)
    N = body_normals(V, F).astype(np.float32)
    vx = PlyData.read(str(splat))["vertex"]
    xyz = np.stack([vx["x"], vx["y"], vx["z"]], 1).astype(np.float32)
    _, j = cKDTree(V).query(xyz)
    sd = ((xyz - V[j]) * N[j]).sum(1)
    return (sd < -depth).astype(np.uint8)


def contain_cameras(cg: Cage, capture: Path, n: int = 256, res: int = 960, seed: int = 49) -> tuple[C.Intrinsics, list, list[str]]:
    """(intrinsics, [(rotation, position)], names) of `n` containment views; `capture` is the capture's COLMAP dir
    (its first camera's intrinsics, scaled to `res` on the long side)."""
    (W, H, fx, fy, cx, cy), cams = colmap_cams(Path(capture))
    sc = res / max(W, H)
    K = C.Intrinsics(max(16, round(W * sc)), max(16, round(H * sc)), fx * sc, fy * sc, cx * sc, cy * sc)
    v_off, v_count, _ = _body(cg)
    c0 = cg.verts[v_off:v_off + v_count].astype(np.float64).mean(0)
    offs = [(-R.T @ t) - c0 for R, t in cams]       # capture camera centres relative to the canonical body
    rng = np.random.default_rng(seed)
    poses, names = [], []
    for k in range(n):
        fr = k * len(cg.names) // n
        tg = cg.posed[fr, v_off:v_off + v_count].astype(np.float64).mean(0)
        pos = tg + offs[rng.integers(len(offs))]
        poses.append((C.look_at(pos, tg), pos))
        names.append(f"{cg.names[fr]}@c{k}")
    return K, poses, names


def write_contain(out: Path, cg: Cage, splat: str | Path, capture: Path, n: int = 256, res: int = 960,
                  depth: float = 0.02, seed: int = 49) -> list[str]:
    """Write <out>/contain_cameras.json and <out>/contain_exclude.u8; returns b2ctrain's flags for them."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    K, poses, names = contain_cameras(cg, capture, n, res, seed)
    C.write_cameras(out / "contain_cameras.json", K, poses, names)
    m = deep_mask(splat, cg, depth)
    m.tofile(out / "contain_exclude.u8")
    print(f"contain: {n} views ({K.width}x{K.height}) over {len(cg.names)} cage frames; {int(m.sum())}/{len(m)} splats "
          f"> {depth} m inside the body left out of the targets")
    return ["--pose-contain-cameras", str(out / "contain_cameras.json"), "--pose-contain-exclude", str(out / "contain_exclude.u8")]
