"""Hair: a shell mesh rigid to the head plus position-based secondary dynamics.

The offset-layer cage (rig/cage.py:offset_layer) copies body vertices, so hair hanging beside the neck or on the
shoulders would be skinned to the neck and clavicles. Hair is attached to the scalp only, so here:

  * shell: the hair splats' outer surface in spherical coordinates about a point inside the skull (a grid of polar
    angle x azimuth cells, each at the 90th-percentile radius of its hair splats), meshed where cells are filled;
  * rest motion: rigid with the head joint;
  * secondary motion: a PBD simulation (verlet, substeps) with the scalp pinned, per-vertex shape matching towards
    the head-rigid rest shape whose stiffness falls with the hair length below the pin line, edge-length
    constraints, gravity relative to the head (the rest shape already hangs, so only a tilted head feels gravity),
    and push-out collisions against the posed body.

The hair splats bind to the shell's triangles through b2ctrain's cage binding (layer classes = Hair), like any layer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .skeleton import JOINT

HEAD = JOINT["head"]


@dataclass
class HairParams:
    pin_deg: float = 72.0        # polar angle (from up) above which the hair is pinned to the scalp
    blend_deg: float = 18.0      # ramp from pinned to free over this many degrees
    k_root: float = 0.35         # shape-matching stiffness per 1/60 s at the pin line ...
    k_tip: float = 0.04          # ... and at the tips (exponential in the hair length below the pin line)
    length_scale: float = 0.10   # m
    damping: float = 3.0         # 1/s velocity damping
    gravity: float = 9.81
    substeps: int = 4
    edge_iters: int = 4
    collide_margin: float = 0.006
    extra: dict = field(default_factory=dict)


def head_frame(body) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonical (centre, up, fwd, side) basis around the skull: centre 6 cm above the head joint."""
    from ..motion.walk import subject_frame
    fr = subject_frame(body)
    c = body.joints_canon[HEAD] + fr.up * 0.06
    return c, np.stack([fr.side, fr.up, fr.fwd], 1)   # columns: x=side(left), y=up, z=fwd


def build_shell(xyz: np.ndarray, body, d_theta: float = 5.0, d_phi: float = 5.0, theta_max: float = 170.0,
                min_count: int = 3, pct: float = 90.0) -> dict:
    """Hair splat centres [N, 3] (canonical world) -> shell {V, F, theta, phi, r}."""
    c, B = head_frame(body)
    loc = (xyz - c) @ B
    r = np.linalg.norm(loc, axis=1)
    theta = np.degrees(np.arccos(np.clip(loc[:, 1] / np.maximum(r, 1e-9), -1, 1)))
    phi = np.degrees(np.arctan2(loc[:, 0], loc[:, 2])) % 360.0
    nt, nph = int(theta_max / d_theta), int(360 / d_phi)
    ti = np.clip((theta / d_theta).astype(int), 0, nt - 1)
    pi = np.clip((phi / d_phi).astype(int), 0, nph - 1)
    R = np.full((nt, nph), np.nan)
    cnt = np.zeros((nt, nph), int)
    np.add.at(cnt, (ti, pi), 1)
    order = np.lexsort((pi, ti))
    key = ti[order] * nph + pi[order]
    starts = np.searchsorted(key, np.arange(nt * nph)); ends = np.searchsorted(key, np.arange(nt * nph), side="right")
    for k in np.nonzero(cnt.ravel() >= min_count)[0]:
        R.flat[k] = np.percentile(r[order[starts[k]:ends[k]]], pct)
    valid = ~np.isnan(R)
    # fill single-cell holes and smooth (azimuth wraps)
    for _ in range(2):
        nb = np.stack([np.roll(R, s, 1) for s in (-1, 1)] + [np.roll(R, s, 0) for s in (-1, 1)])
        nb[2][0] = np.nan; nb[3][-1] = np.nan     # no wrap in theta
        nv = (~np.isnan(nb)).sum(0)
        fill = (~valid) & (nv >= 3)
        R[fill] = np.nanmean(nb[:, fill], 0)
        valid = ~np.isnan(R)
    for _ in range(2):
        nb = np.stack([R, np.roll(R, 1, 1), np.roll(R, -1, 1), np.vstack([R[:1], R[:-1]]), np.vstack([R[1:], R[-1:]])])
        sm = np.nanmedian(nb, 0)
        R = np.where(valid, sm, np.nan)
    # drop tiny islands: keep the component connected to the crown
    lab = -np.ones((nt, nph), int)
    stack = [(i, j) for i in range(min(3, nt)) for j in range(nph) if valid[i, j]]
    for s in stack:
        lab[s] = 0
    while stack:
        i, j = stack.pop()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, (j + dj) % nph
            if 0 <= a < nt and valid[a, b] and lab[a, b] < 0:
                lab[a, b] = 0; stack.append((a, b))
    valid &= lab == 0
    vid = -np.ones((nt, nph), int)
    th = (np.arange(nt) + 0.5) * d_theta; ph = (np.arange(nph) + 0.5) * d_phi
    V, TH, PH, RR = [], [], [], []
    for i in range(nt):
        for j in range(nph):
            if valid[i, j]:
                vid[i, j] = len(V)
                t, p = math.radians(th[i]), math.radians(ph[j])
                d = np.array([math.sin(t) * math.sin(p), math.cos(t), math.sin(t) * math.cos(p)])
                V.append(c + B @ (d * R[i, j])); TH.append(th[i]); PH.append(ph[j]); RR.append(R[i, j])
    F = []
    for i in range(nt - 1):
        for j in range(nph):
            a, b, cc, d = vid[i, j], vid[i, (j + 1) % nph], vid[i + 1, j], vid[i + 1, (j + 1) % nph]
            if min(a, b, cc, d) >= 0:
                F += [(a, cc, b), (b, cc, d)]
    # crown cap
    ring = [vid[0, j] for j in range(nph) if vid[0, j] >= 0]
    if len(ring) == nph:
        top = len(V)
        V.append(c + B[:, 1] * float(np.nanmean(R[0]))); TH.append(0.0); PH.append(0.0); RR.append(float(np.nanmean(R[0])))
        F += [(top, vid[0, j], vid[0, (j + 1) % nph]) for j in range(nph)]
    V = np.asarray(V); F = np.asarray(F, np.int64)
    # drop vertices no face uses: b2ctrain binds a splat through its nearest vertex's faces, so a bare vertex would
    # leave its splats unbound (static)
    used = np.unique(F)
    remap = -np.ones(len(V), np.int64); remap[used] = np.arange(len(used))
    V, F = V[used], remap[F]
    TH, PH, RR = np.asarray(TH)[used], np.asarray(PH)[used], np.asarray(RR)[used]
    # outward orientation: normals away from the centre
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    if np.mean(np.einsum("ij,ij->i", fn, V[F].mean(1) - c)) < 0:
        F = F[:, ::-1].copy()
    return {"V": V.astype(np.float32), "F": F.astype(np.int32), "theta": np.asarray(TH, np.float32),
            "phi": np.asarray(PH, np.float32), "r": np.asarray(RR, np.float32)}


def edges_of(F: np.ndarray) -> np.ndarray:
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.sort(e, 1)
    return np.unique(e, axis=0)


def hair_length(theta: np.ndarray, r: np.ndarray, p: HairParams) -> np.ndarray:
    """Arc length (m) below the pin line along the polar direction; <= 0 on the pinned crown."""
    return np.radians(theta - p.pin_deg) * r


def simulate(V0: np.ndarray, F: np.ndarray, theta: np.ndarray, r: np.ndarray, head_R: np.ndarray, head_t: np.ndarray,
             fps: float, p: HairParams | None = None, body_verts: np.ndarray | None = None,
             body_normals: np.ndarray | None = None, preroll: int = 0) -> np.ndarray:
    """Head transforms (canonical -> posed) per frame: head_R [T, 3, 3], head_t [T, 3] (x_posed = R x + t).
    body_verts / body_normals [T, Nb, 3]: the posed body for collisions (a subset near the hair is enough).
    Returns [T, V, 3] posed shell vertices."""
    p = p or HairParams()
    T, Vn = len(head_R), len(V0)
    tgt = np.einsum("tij,vj->tvi", head_R, V0) + head_t[:, None]
    s = hair_length(theta, r, p)
    w_free = np.clip(np.radians(theta - p.pin_deg) / np.radians(p.blend_deg), 0, 1)   # 0 pinned .. 1 free
    k60 = p.k_tip + (p.k_root - p.k_tip) * np.exp(-np.clip(s, 0, None) / p.length_scale)
    E = edges_of(F)
    L0 = np.linalg.norm(V0[E[:, 0]] - V0[E[:, 1]], axis=1)
    deg = np.bincount(E.ravel(), minlength=Vn).astype(float)
    dt = 1.0 / (fps * p.substeps)
    alpha = 1.0 - (1.0 - k60) ** (dt * 60.0)
    g = np.array([0.0, -p.gravity, 0.0])
    x = tgt[0].copy(); v = np.zeros_like(x)
    out = np.zeros((T, Vn, 3))
    # rest signed distance to the body (collision floor), measured at frame 0
    tree0 = None
    if body_verts is not None:
        tree0 = cKDTree(body_verts[0])
        _, i0 = tree0.query(V0)
        d0 = np.einsum("ij,ij->i", V0 - body_verts[0][i0], body_normals[0][i0])
        floor = np.minimum(d0, p.collide_margin)
    frames = [0] * preroll + list(range(T))
    for n, f in enumerate(frames):
        f0 = frames[n - 1] if n > 0 else f
        tree = cKDTree(body_verts[f]) if body_verts is not None else None
        for k in range(p.substeps):
            u = (k + 1) / p.substeps
            tg = tgt[f0] * (1 - u) + tgt[f] * u
            Rh = head_R[f]
            g_rel = g - Rh @ (head_R[0].T @ g)   # the rest shape already hangs under gravity in the canonical pose
            v = v + dt * g_rel[None] * w_free[:, None]
            v *= math.exp(-p.damping * dt)
            y = x + v * dt
            y += (alpha * w_free + (1 - w_free))[:, None] * (tg - y)
            for _ in range(p.edge_iters):
                d = y[E[:, 1]] - y[E[:, 0]]
                L = np.linalg.norm(d, axis=1) + 1e-9
                corr = ((L - L0) / L)[:, None] * d * 0.5
                acc = np.zeros_like(y)
                np.add.at(acc, E[:, 0], corr); np.add.at(acc, E[:, 1], -corr)
                y += 0.8 * acc / np.maximum(deg, 1)[:, None] * w_free[:, None]
            if tree is not None:
                _, ii = tree.query(y)
                sd = np.einsum("ij,ij->i", y - body_verts[f][ii], body_normals[f][ii])
                pen = np.minimum(sd - floor, 0.0)
                y -= pen[:, None] * body_normals[f][ii]
            v = (y - x) / dt
            x = y
        if n >= preroll:
            out[f] = x
    return out.astype(np.float32)


def save_shell(subject: Path, shell: dict, body) -> None:
    """<subject>/cages/hair_offset.npz in the layer format (V, F, body_index) plus the shell's own fields; the layer
    is marked rigid to the head so rig/layered.pose_layers poses it with the head joint."""
    _, bi = cKDTree(body.verts_canon).query(shell["V"])
    np.savez(Path(subject) / "cages" / "hair_offset.npz", V=shell["V"], F=shell["F"], body_index=bi,
             theta=shell["theta"], phi=shell["phi"], r=shell["r"], rigid_joint=np.int64(HEAD))
