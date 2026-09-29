"""A loose-garment shell (a skirt): a tube around the garment's vertical axis, on the outer surface of its splats, with
its own smoothed skinning weights.

An offset layer (rig/cage.py:offset_layer) copies body vertices, so a skirt becomes two sleeves hugging the thighs:
the splats spanning the gap between the legs bind to whichever thigh is nearer and the skirt tears apart as the legs
spread. The shell spans the gap; each vertex takes the weights of its nearest body vertex, diffused around and along
the tube (the waist row pinned), so the front and back between the legs follow both thighs and the pelvis.
The layer npz (cages/<name>_offset.npz) carries skin_idx / skin_w next to V, F, body_index; rig/layered poses it
with those weights.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from . import lbs


def build(xyz: np.ndarray, body, n_theta: int = 72, n_rows: int = 14, q_out: float = 0.95, smooth_iters: int = 60,
          k: int = 8, min_gap: float = 0.004) -> dict:
    """`xyz`: the garment's splats (canonical). Returns V [n_rows * n_theta, 3], F, skin_idx / skin_w [V, k],
    body_index (nearest body vertex), and the grid (theta, y) for inspection."""
    # the garment's contiguous vertical band around its median height (other pieces of the same class, e.g. stocking
    # bands labelled lower clothing, sit below an empty gap)
    h, e = np.histogram(xyz[:, 1], bins=np.arange(xyz[:, 1].min(), xyz[:, 1].max() + 0.02, 0.02))
    full = h > 0.002 * len(xyz)
    b0 = b1 = int(np.clip(np.searchsorted(e, np.median(xyz[:, 1])) - 1, 0, len(h) - 1))
    while b0 > 0 and full[b0 - 1]:
        b0 -= 1
    while b1 < len(h) - 1 and full[b1 + 1]:
        b1 += 1
    xyz = xyz[(xyz[:, 1] >= e[b0]) & (xyz[:, 1] <= e[b1 + 1])]
    c = np.median(xyz[:, [0, 2]], 0)
    y0, y1 = np.percentile(xyz[:, 1], 0.5), np.percentile(xyz[:, 1], 99.5)
    ys = np.linspace(y1, y0, n_rows)   # row 0 = waist
    rel = xyz[:, [0, 2]] - c
    th = np.arctan2(rel[:, 0], rel[:, 1]); rr = np.linalg.norm(rel, axis=1)
    ti = np.floor((th + np.pi) / (2 * np.pi) * n_theta).astype(int) % n_theta
    dy = (y1 - y0) / (n_rows - 1)
    R = np.full((n_rows, n_theta), np.nan)
    for r in range(n_rows):
        m = np.abs(xyz[:, 1] - ys[r]) < 0.75 * dy
        for t in range(n_theta):
            s = rr[m & (ti == t)]
            if len(s) >= 3:
                R[r, t] = np.quantile(s, q_out)
    for _ in range(200):   # fill empty bins from their neighbours (periodic in theta)
        if not np.isnan(R).any():
            break
        P = np.pad(R, ((1, 1), (0, 0)), constant_values=np.nan)
        nb = np.stack([np.roll(R, 1, 1), np.roll(R, -1, 1), P[:-2], P[2:]])
        fill = np.nanmean(nb, 0)
        R = np.where(np.isnan(R), fill, R)
    tc = (np.arange(n_theta) + 0.5) / n_theta * 2 * np.pi - np.pi
    V = np.stack([c[0] + R * np.sin(tc)[None], np.repeat(ys[:, None], n_theta, 1), c[1] + R * np.cos(tc)[None]], -1)
    # never inside the body: push out along the radial direction to min_gap beyond the nearest body surface point
    BV = np.asarray(body.verts_canon)
    V = V.reshape(-1, 3)
    tree = cKDTree(BV)
    d, bi = tree.query(V)
    radial = V - np.array([c[0], 0.0, c[1]]); radial[:, 1] = 0
    radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-9)
    inside = ((BV[bi] - V) * radial).sum(1) > -min_gap
    V[inside] = BV[bi[inside]] + radial[inside] * min_gap
    _, bi = tree.query(V)
    sv, sj, sw = body.skin
    W = lbs.dense_weights(sv, sj, sw, len(BV), len(body.parents))[bi].reshape(n_rows, n_theta, -1)
    W0 = W[0].copy()
    for _ in range(smooth_iters):   # Jacobi diffusion on the tube grid (periodic in theta), the waist row pinned
        P = np.concatenate([W[:1], W, W[-1:]], 0)
        W = (np.roll(W, 1, 1) + np.roll(W, -1, 1) + P[:-2] + P[2:]) / 4
        W[0] = W0
    idx, w = lbs.topk_weights(W.reshape(n_rows * n_theta, -1), k)
    F = []
    for r in range(n_rows - 1):
        for t in range(n_theta):
            a, b = r * n_theta + t, r * n_theta + (t + 1) % n_theta
            c2, d2 = a + n_theta, b + n_theta
            F += [(a, c2, b), (b, c2, d2)]
    return {"V": V.astype(np.float32), "F": np.asarray(F, np.int64), "skin_idx": idx, "skin_w": w,
            "body_index": bi, "grid": np.array([n_rows, n_theta])}


def save(subject: Path, name: str, sh: dict) -> None:
    np.savez(Path(subject) / "cages" / f"{name}_offset.npz", V=sh["V"], F=sh["F"], body_index=sh["body_index"],
             skin_idx=sh["skin_idx"], skin_w=sh["skin_w"], grid=sh["grid"])
