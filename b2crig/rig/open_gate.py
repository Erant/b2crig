"""The opening gate of two-state splats (b2ctrain --cage-open, src/gpu/cage_open.h): per cage vertex and frame, how far a
joint next to the vertex has turned.

theta(a) = max over a's partners b of the angle between dq_a (v0_b - v0_a) and (v_b - v_a), in degrees: dq_a is the
rotation of a's incident face (its first-edge frame, as b2ctrain's cage.cu builds it, posed times canonical^T), the
partners the canonical vertices within `radius` of a and beyond `min_dist`, every stride-th of them in index order so
at most MAX_PARTNERS remain. Zero at the canonical pose and wherever the neighbourhood moves rigidly with the vertex;
large where a joint between the vertex and something nearby has turned (the torso side under a raised arm, the inner
arm, a hip a hand has left). A joint-angle measure that needs no skeleton, pose library or labels. Pair distances and
neighbour counts were tried first and fail on the dark armpit splats (they ride torso vertices next to the shoulder,
whose distance to the arm barely changes as it rises; measured 2026-09-27).

This used to run inside b2ctrain; it is a decision about the rig, so it lives here and b2ctrain reads the result from
the cage file: an optional section after the posed vertices,
    char tag[8] = "B2COPEN1"; float32 radius, min_dist; float16 theta[n_frames][n_verts]   (degrees)
b2ctrain maps theta to the blend weight with the start / end angles the splat was trained with (its ply header). A
cage without the section cannot drive a ply that carries open_* states (b2ctrain renders it plain and says so).

    .venv/bin/python -m b2crig.rig.open_gate CAGE [CAGE ...]      # add (or replace) the section in place
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TAG = b"B2COPEN1"
MAX_PARTNERS = 64      # b2ctrain's OPEN_K when the gate ran there; kept so trained open states see the same gate
RADIUS, MIN_DIST = 0.12, 0.03


def vertex_faces(F: np.ndarray, nv: int) -> np.ndarray:
    """One incident face per vertex (the lowest face index; -1 when none)."""
    vf = np.full(nv, np.iinfo(np.int64).max, np.int64)
    fi = np.repeat(np.arange(len(F)), 3)
    np.minimum.at(vf, F.reshape(-1), fi)
    vf[vf == np.iinfo(np.int64).max] = -1
    return vf


def partners(V0: np.ndarray, radius: float = RADIUS, min_dist: float = MIN_DIST, max_k: int | None = MAX_PARTNERS) -> np.ndarray:
    """[nv, K] partner vertex indices, -1 padded (K = max_k, or the largest count when None)."""
    from scipy.spatial import cKDTree
    V0 = np.asarray(V0, np.float32)
    lists = cKDTree(V0).query_ball_point(V0, radius * 1.001)
    out = []
    for a, bs in enumerate(lists):
        bs = np.sort(np.asarray(bs, np.int64)); bs = bs[bs != a]
        d = V0[bs] - V0[a]; d2 = (d * d).sum(1)
        bs = bs[(d2 <= np.float32(radius * radius)) & (d2 >= np.float32(min_dist * min_dist))]
        if max_k is not None and len(bs):
            stride = max(1, -(-len(bs) // max_k))
            bs = bs[::stride][:max_k]
        out.append(bs)
    K = max_k if max_k is not None else max((len(b) for b in out), default=0)
    P = np.full((len(V0), max(K, 1)), -1, np.int64)
    for a, bs in enumerate(out):
        P[a, :len(bs)] = bs
    return P


def _frames(V, F):
    """Triangle frames as cage.cu tri_frame: columns (e1 along the first edge, n x e1, n)."""
    import torch
    u = V[..., F[:, 1], :] - V[..., F[:, 0], :]; w = V[..., F[:, 2], :] - V[..., F[:, 0], :]
    e1 = torch.nn.functional.normalize(u, dim=-1, eps=1e-12)
    n = torch.nn.functional.normalize(torch.cross(u, w, dim=-1), dim=-1, eps=1e-12)
    return torch.stack([e1, torch.cross(n, e1, dim=-1), n], -1)


def prepare(V0: np.ndarray, F: np.ndarray, radius: float = RADIUS, min_dist: float = MIN_DIST,
            max_k: int | None = MAX_PARTNERS) -> dict:
    """The canonical part of the gate (partners, incident faces, canonical frames), computed once per cage."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    V0 = np.asarray(V0, np.float32); F = np.asarray(F, np.int64); nv = len(V0)
    vf = vertex_faces(F, nv); P = partners(V0, radius, min_dist, max_k)
    ok = vf >= 0
    a_idx = torch.as_tensor(np.nonzero(ok)[0], device=dev)
    Pt = torch.as_tensor(P[ok], device=dev)
    Ft = torch.as_tensor(F[vf[ok]], device=dev)            # the incident face of each gated vertex
    V0t = torch.as_tensor(V0, device=dev)
    return {"nv": nv, "dev": dev, "a": a_idx, "valid": Pt >= 0, "P": Pt.clamp(min=0), "F": Ft, "R0T": _frames(V0t, Ft).transpose(-1, -2),
            "off0": V0t[Pt.clamp(min=0)] - V0t[a_idx][:, None]}


def theta_prepared(prep: dict, posed: np.ndarray, chunk: int = 8) -> np.ndarray:
    """float32 [n_frames, nv] gate angles (degrees) of the posed frames `posed` [n_frames, nv, 3] (or [nv, 3] -> [nv])."""
    import torch
    posed = np.asarray(posed, np.float32); single = posed.ndim == 2
    if single:
        posed = posed[None]
    a_idx, valid, Pc, Ft = prep["a"], prep["valid"], prep["P"], prep["F"]
    out = np.zeros((len(posed), prep["nv"]), np.float32)
    for s in range(0, len(posed), chunk):
        V = torch.as_tensor(posed[s:s + chunk], device=prep["dev"])                  # [t, nv, 3]
        dq = _frames(V, Ft) @ prep["R0T"][None]                                        # [t, m, 3, 3] posed frame x canonical^T
        pred = torch.einsum("tmij,mkj->tmki", dq, prep["off0"])
        d = V[:, Pc] - V[:, a_idx][:, :, None]
        nn = pred.norm(dim=-1) * d.norm(dim=-1)
        c = torch.where(valid[None] & (nn > 1e-12), (pred * d).sum(-1) / nn.clamp(min=1e-30), torch.ones_like(nn))
        th = torch.zeros(V.shape[0], prep["nv"], device=prep["dev"])
        th[:, a_idx] = torch.rad2deg(torch.arccos(c.min(-1).values.clamp(-1, 1)))
        out[s:s + chunk] = th.cpu().numpy()
    return out[0] if single else out


def theta(V0: np.ndarray, F: np.ndarray, posed: np.ndarray, radius: float = RADIUS, min_dist: float = MIN_DIST,
          max_k: int | None = MAX_PARTNERS) -> np.ndarray:
    """float32 [n_frames, nv] gate angles (degrees) of the posed frames `posed` [n_frames, nv, 3]."""
    return theta_prepared(prepare(V0, F, radius, min_dist, max_k), posed)


def _body_end(raw: bytes) -> tuple[int, int, int, int]:
    """(offset where the posed vertices end, n_verts, n_faces, n_frames) of a B2CCAGE1 file."""
    assert raw[:8] == b"B2CCAGE1", "not a b2ctrain cage v1"
    nl, nv, nf, nfr = np.frombuffer(raw[8:24], np.int32)
    return int(24 + 20 * nl + 12 * nv + 12 * nf + 64 * nfr + 12 * nfr * nv), int(nv), int(nf), int(nfr)


def read_theta(path: str | Path) -> np.ndarray | None:
    raw = Path(path).read_bytes(); end, nv, _, nfr = _body_end(raw)
    if raw[end:end + 8] != TAG:
        return None
    return np.frombuffer(raw, np.float16, nfr * nv, end + 16).reshape(nfr, nv).astype(np.float32)


def has_open_gate(path: str | Path) -> bool:
    raw = Path(path).read_bytes(); end, *_ = _body_end(raw)
    return raw[end:end + 8] == TAG


def add_open_gate(path: str | Path, radius: float = RADIUS, min_dist: float = MIN_DIST) -> np.ndarray:
    """Compute the gate of every frame of the cage file and write it as its B2COPEN1 section (replacing one there)."""
    from ..evaluate import read_cage
    cg = read_cage(Path(path))
    th = theta(cg.verts, cg.faces, cg.posed, radius, min_dist)
    raw = Path(path).read_bytes(); end, *_ = _body_end(raw)
    with open(path, "wb") as f:
        f.write(raw[:end]); f.write(TAG); f.write(np.array([radius, min_dist], np.float32).tobytes())
        f.write(th.astype(np.float16).tobytes())
    return th


if __name__ == "__main__":
    for p in sys.argv[1:]:
        th = add_open_gate(p)
        print(f"{p}: open gate over {th.shape[0]} frames x {th.shape[1]} vertices, max {th.max():.1f} deg")
