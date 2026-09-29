"""Which body surface a subject's capture saw, and where that surface lands in a posed clip frame.

z-buffers of vertices splatted as small discs at ~220 px image height (at full resolution occluded vertices show
through the gaps between them). Used by tools/reveal_pose.py and by clip building (grey_unseen: the control frames
blank the surface the capture never saw, so WAN inpaints it instead of elaborating the leftover gaussians).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MIN_VIEWS = 3


def qvec_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def colmap_cams(ds: Path):
    """(W, H, fx, fy, cx, cy), [(R_w2c, t)] of a COLMAP text model with one PINHOLE camera."""
    p = [ln.split() for ln in (ds / "cameras.txt").read_text().splitlines() if ln and not ln.startswith("#")][0]
    intr = (int(p[2]), int(p[3]), *map(float, p[4:8]))
    out = []
    for ln in (ds / "images.txt").read_text().splitlines():
        f = ln.split()
        if ln.startswith("#") or len(f) < 10:
            continue
        out.append((qvec_to_R(list(map(float, f[1:5]))), np.array(list(map(float, f[5:8])))))
    return intr, out


def json_cams(cameras_json: Path):
    """The same from a b2crig cameras.json (rotation = camera->world with columns right, up, back; position)."""
    cj = json.loads(Path(cameras_json).read_text())
    c0 = cj["cameras"][0]
    intr = (cj["width"], cj["height"], c0["fx"], c0["fy"], c0["cx"], c0["cy"])
    out = []
    for c in cj["cameras"]:
        Rw2c = (np.asarray(c["rotation"]) @ np.diag([1.0, -1.0, -1.0])).T
        out.append((Rw2c, -Rw2c @ np.asarray(c["position"])))
    return intr, out


def visible(V: np.ndarray, intr, cam, px: int = 1, target_h: int = 220) -> np.ndarray:
    """bool [V]: vertices visible from one camera."""
    W, H, fx, fy, cx, cy = intr
    sc = target_h / H
    W, H, fx, fy, cx, cy = int(W * sc), int(H * sc), fx * sc, fy * sc, cx * sc, cy * sc
    Rw2c, t = cam
    q = V @ Rw2c.T + t
    z = q[:, 2]; ok = z > 0.05
    u = np.round(fx * q[:, 0] / np.where(ok, z, 1) + cx).astype(int); v = np.round(fy * q[:, 1] / np.where(ok, z, 1) + cy).astype(int)
    ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
    zb = np.full((H + 2 * px, W + 2 * px), np.inf)
    for du in range(-px, px + 1):
        for dv in range(-px, px + 1):
            np.minimum.at(zb, (v[ok] + dv + px, u[ok] + du + px), z[ok])
    vis = np.zeros(len(V), bool)
    vis[ok] = z[ok] <= zb[v[ok] + px, u[ok] + px] + 0.01
    return vis


def visible_counts(V: np.ndarray, intr, cams, **kw) -> np.ndarray:
    return np.sum([visible(V, intr, c, **kw) for c in cams], axis=0).astype(np.int32)


def capture_seen(subject: Path, run: Path, body_verts_canon: np.ndarray) -> np.ndarray:
    """bool per body vertex: seen by >= MIN_VIEWS of the capture's training cameras (cached in <subject>/seen.npy)."""
    f = subject / "seen.npy"
    if f.exists():
        return np.load(f)
    intr, cams = colmap_cams(run / "colmap")
    seen = visible_counts(body_verts_canon, intr, cams) >= MIN_VIEWS
    np.save(f, seen)
    return seen


def unseen_pixels(V: np.ndarray, seen: np.ndarray, intr, cam, radius_px: int = 6) -> np.ndarray:
    """uint8 [H, W] mask (255) of the pixels where visible, never-captured body surface lands."""
    import cv2
    W, H, fx, fy, cx, cy = intr
    m = np.zeros((H, W), np.uint8)
    vis = visible(V, intr, cam) & ~seen
    if not vis.any():
        return m
    Rw2c, t = cam
    q = V[vis] @ Rw2c.T + t
    u = np.round(fx * q[:, 0] / q[:, 2] + cx).astype(int); v = np.round(fy * q[:, 1] / q[:, 2] + cy).astype(int)
    for x, y in zip(u, v):
        if 0 <= x < W and 0 <= y < H:
            cv2.circle(m, (int(x), int(y)), radius_px, 255, -1)
    return m


def colmap_to_cameras_json(ds: Path, out: Path) -> None:
    """A COLMAP text model (one PINHOLE camera) as a b2crig cameras.json, named like its images (b2ctrain render
    --cameras of the capture views, e.g. for --write-evidence)."""
    from .cameras import Intrinsics, write_cameras
    (W, H, fx, fy, cx, cy), cams = colmap_cams(ds)
    names = [ln.split()[9] for ln in (ds / "images.txt").read_text().splitlines() if not ln.startswith("#") and len(ln.split()) >= 10]
    poses = [(Rw2c.T @ np.diag([1.0, -1.0, -1.0]), -Rw2c.T @ t) for Rw2c, t in cams]
    write_cameras(out, Intrinsics(W, H, fx, fy, cx, cy), poses, [Path(n).stem for n in names])
