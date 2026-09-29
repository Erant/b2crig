"""Split the hand splats that straddle a finger joint, one child per bone.

A splat binds to one cage triangle, so a splat that reaches across a knuckle follows one phalanx while half of it
covers the next: when the finger bends, that half shears out of the finger (b24be4: 52% of the opaque hand splats
cross a joint's cut plane within 2 sigma, by 5 mm median, 15-17 mm p95). The pose containment loss can only thin such
a splat. Here each one is replaced by its two halves on either side of the joint's cut plane, so each child binds to
its own phalanx and the containment fine-tune (tools/cage_train.py --contain ... --contain-hands) refines them.

Cut planes: at every articulated finger joint (rig/skeleton.FINGERS: knuckle, PIP, DIP; the thumb's CMC, second CMC,
MCP and IP) and at each wrist, through the joint, normal to the bisector of the bone into it and the bone out of it
(where a hinge's two segments meet when it bends). A plane only cuts the splats in its bone's tube: centre within
`tube` of the joint and on the hand (nearest body vertex skinned to that hand, rig/contain.hand_vertices), so the
neighbouring fingers' planes leave a finger alone.

The halves are the exact moments of the Gaussian truncated to each side: with u = n.(x - p) ~ N(t, s^2),
s^2 = n^T S n, a = -t / s, lam = phi(a) / (1 - Phi(a)) for the + side,
  mean  = mu + S n / s^2 * (E[u | u > 0] - t),   E[u | u > 0] = t + s lam
  cov   = S - S n n^T S / s^2 * (1 - Var[u | u > 0] / s^2),   Var = s^2 (1 + a lam - lam^2)
(the - side mirrored). Opacity keeps each half's share of the mass: o_c = o * m_c * sqrt(det S / det S_c), capped at
0.99. Colour, SH and labels are copied. A splat is split while any plane cuts it with both halves >= `min_mass` of
its mass and its 2 sigma extent across the plane; a second plane may cut a child again (up to `passes`), never
a plane that made it or its parent (the moment-matched half still spills ~9% across its own cut).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.special import ndtr

from .contain import hand_vertices
from .skeleton import FINGERS, JOINT

PHI = lambda x: np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)


def cut_planes(joints: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, str, float]]:
    """(point, unit normal, hand side, tube radius) per articulated finger joint and wrist, canonical pose."""
    unit = lambda v: v / np.linalg.norm(v)
    out = []
    for name, ch in FINGERS.items():
        side = name[0]
        wrist = JOINT[f"{side}_wrist"] + 1                     # the hand joint the fingers hang off
        chain = [wrist] + list(ch)
        for i in range(1, len(chain) - 1):                     # every joint with a bone in and a bone out
            j_prev, j, j_next = chain[i - 1], chain[i], chain[i + 1]
            d_in, d_out = unit(joints[j] - joints[j_prev]), unit(joints[j_next] - joints[j])
            n = unit(d_in + d_out)
            seg = min(np.linalg.norm(joints[j] - joints[j_prev]), np.linalg.norm(joints[j_next] - joints[j]))
            out.append((joints[j], n, side, max(0.012, 0.8 * seg)))
    for side, mid in (("r", "r_middle"), ("l", "l_middle")):
        w = JOINT[f"{side}_wrist"] + 1
        n = unit(joints[FINGERS[mid][0]] - joints[w])
        out.append((joints[w], n, side, 0.05))
    return out


def _cov(scale_log: np.ndarray, rot_wxyz: np.ndarray) -> np.ndarray:
    q = rot_wxyz / np.linalg.norm(rot_wxyz, axis=1, keepdims=True)
    R = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    s2 = np.exp(2 * scale_log)
    return np.einsum("nij,nj,nkj->nik", R, s2, R)


def _decompose(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[N, 3, 3] covariances -> (log scales [N, 3], wxyz quaternions [N, 4])."""
    w, V = np.linalg.eigh(C)
    V[np.linalg.det(V) < 0, :, 0] *= -1
    q = Rotation.from_matrix(V).as_quat()[:, [3, 0, 1, 2]]
    return 0.5 * np.log(np.clip(w, 1e-14, None)), q


def split_once(f: dict, planes, hand_side: np.ndarray, used: np.ndarray, k_sigma: float = 2.0,
               min_mass: float = 0.05) -> tuple[dict, np.ndarray, int]:
    """One pass: split every splat a plane cuts (the most even cut). `f`: PLY fields; `hand_side` [N] 'r' / 'l' / '';
    `used` [N]: bitmask of the planes that made each splat (its lineage), never used on it again. Returns
    (fields, used, n)."""
    X = np.stack([f["x"], f["y"], f["z"]], 1).astype(np.float64)
    sl = np.stack([f["scale_0"], f["scale_1"], f["scale_2"]], 1).astype(np.float64)
    rq = np.stack([f["rot_0"], f["rot_1"], f["rot_2"], f["rot_3"]], 1).astype(np.float64)
    C = _cov(sl, rq)
    N = len(X)
    cut = np.full(N, -1)
    best = np.zeros(N)
    for pi, (p, n, side, tube) in enumerate(planes):
        on = (hand_side == side) & (np.linalg.norm(X - p, axis=1) < tube) & ((used >> pi) & 1 == 0)
        if not on.any():
            continue
        idx = np.nonzero(on)[0]
        t = (X[idx] - p) @ n
        s = np.sqrt(np.einsum("i,nij,j->n", n, C[idx], n))
        m_pos = ndtr(t / s)
        ok = (np.abs(t) < k_sigma * s) & (np.minimum(m_pos, 1 - m_pos) >= min_mass)
        score = np.where(ok, np.minimum(m_pos, 1 - m_pos), 0)   # the most even cut wins when two planes cut it
        take = score > best[idx]
        cut[idx[take]] = pi; best[idx[take]] = score[take]
    split = np.nonzero(cut >= 0)[0]
    if len(split) == 0:
        return f, used, 0
    P = np.stack([planes[c][0] for c in cut[split]]); Nn = np.stack([planes[c][1] for c in cut[split]])
    mu, S = X[split], C[split]
    t = ((mu - P) * Nn).sum(1)
    Sn = np.einsum("nij,nj->ni", S, Nn)
    s2 = (Sn * Nn).sum(1); s = np.sqrt(s2)
    op = 1 / (1 + np.exp(-f["opacity"][split].astype(np.float64)))
    children = []
    for sign in (1.0, -1.0):
        a = -sign * t / s                                      # the cut in the side's standard units
        mass = 1 - ndtr(a)
        lam = PHI(a) / np.maximum(mass, 1e-12)
        eu = sign * (sign * t + s * lam)                       # E[u | this side]
        var = s2 * (1 + a * lam - lam * lam)
        m_c = mu + Sn / s2[:, None] * (eu - t)[:, None]
        C_c = S - np.einsum("ni,nj->nij", Sn, Sn) / s2[:, None, None] * (1 - var / s2)[:, None, None]
        sl_c, q_c = _decompose(C_c)
        o_c = np.clip(op * mass * np.sqrt(np.linalg.det(S) / np.linalg.det(C_c)), 1e-4, 0.99)
        children.append((m_c, sl_c, q_c, np.log(o_c / (1 - o_c))))
    keep = np.ones(N, bool); keep[split] = False
    out = {}
    for k, v in f.items():
        parts = [v[keep]]
        for m_c, sl_c, q_c, ol_c in children:
            c = v[split].copy()
            if k in ("x", "y", "z"):
                c = m_c[:, "xyz".index(k)].astype(v.dtype)
            elif k.startswith("scale_"):
                c = sl_c[:, int(k[-1])].astype(v.dtype)
            elif k.startswith("rot_"):
                c = q_c[:, int(k[-1])].astype(v.dtype)
            elif k == "opacity":
                c = ol_c.astype(v.dtype)
            parts.append(c)
        out[k] = np.concatenate(parts)
    child = used[split] | (np.int64(1) << cut[split].astype(np.int64))
    return out, np.concatenate([used[keep], child, child]), len(split)


def split_joints(splat: str | Path, mhr_npz: str | Path, out: str | Path, *, k_sigma: float = 2.0,
                 min_mass: float = 0.05, passes: int = 3, min_opacity: float = 0.05) -> dict:
    """Split `splat`'s knuckle-straddling hand splats into `out`. Returns counts."""
    d = np.load(mhr_npz)
    BV, J = d["verts_world"].astype(np.float64), d["joints_world"].astype(np.float64)
    hv = hand_vertices(mhr_npz)
    vside = np.full(len(BV), "", dtype="<U1")
    vside[hv["r"]] = "r"; vside[hv["l"]] = "l"
    planes = cut_planes(J)
    ply = PlyData.read(str(splat))
    el = ply["vertex"]
    f = {k: np.asarray(el[k]) for k in el.data.dtype.names}
    n0 = len(f["x"])
    tree = cKDTree(BV)
    report = {"splats_in": n0, "passes": []}
    used = np.zeros(n0, np.int64)
    assert len(planes) <= 63
    for _ in range(passes):
        X = np.stack([f["x"], f["y"], f["z"]], 1)
        _, j = tree.query(X)
        side = vside[j].copy()
        op = 1 / (1 + np.exp(-f["opacity"].astype(np.float64)))
        side[op < min_opacity] = ""                           # near-transparent splats are not worth splitting
        f, used, n = split_once(f, planes, side, used, k_sigma, min_mass)
        report["passes"].append(n)
        if n == 0:
            break
    report["splats_out"] = len(f["x"])
    arr = np.empty(len(f["x"]), dtype=el.data.dtype)
    for k in el.data.dtype.names:
        arr[k] = f[k]
    PlyData([PlyElement.describe(arr, "vertex")], text=False, comments=list(ply.comments),
            obj_info=list(ply.obj_info)).write(str(out))
    return report
