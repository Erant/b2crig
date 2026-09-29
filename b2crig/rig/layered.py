"""Layered cages: the MHR body plus offset garment/hair layers (rig/cage.py:offset_layer), posed per frame."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .. import b2ctrain as B
from . import lbs
from .mhr import MHRBody, Posed

N_CLASSES = 29


def layer_classes(subject: Path) -> dict:
    """The subject's garment layers {name: Sapiens2 classes}: <subject>/layers.json when present (e.g. an "apparel"
    layer of its own for straps and harnesses), else cage.DEFAULT_LAYERS."""
    from .cage import DEFAULT_LAYERS
    f = Path(subject) / "layers.json"
    return {k: tuple(v) for k, v in json.loads(f.read_text()).items()} if f.exists() else dict(DEFAULT_LAYERS)


def load_layers(subject: Path, names=None) -> list[dict]:
    classes = layer_classes(subject)
    out = []
    for n in (names or classes):
        p = subject / "cages" / f"{n}_offset.npz"
        if p.exists():
            d = np.load(p)
            L = {"name": n, "V": d["V"], "F": d["F"], "bi": d["body_index"], "classes": classes[n]}
            for k in ("rigid_joint", "theta", "r", "skin_idx", "skin_w"):   # a hair shell (rig/hair.py) / skirt shell (rig/skirt.py)
                if k in d.files:
                    L[k] = d[k]
            if "theta" in L and (subject / "hair.json").exists():   # calibrated dynamics (tools/hair_calib.py)
                L["hair_params"] = json.loads((subject / "hair.json").read_text())
            out.append(L)
    return out


def stable_faces(body: MHRBody, amp: float = 0.6, max_stretch: float = 2.5) -> np.ndarray:
    """bool per body face: its size (sqrt 2 area) stays within [1/max_stretch, max_stretch] of the canonical one under
    every MHR expression dimension at +-amp. The rest are the closed mouth's lip-seam and mouth-corner triangles, which
    grow up to ~12x when the jaw opens: a splat bound there would take that factor on its offset and scales (a blob).
    Leaving them out of the cage only moves such splats' binding to a stable neighbour."""
    if hasattr(body, "_stable_faces"):
        return body._stable_faces
    cache = body.npz_path.parent / "stable_faces.npy"   # frozen per subject: CPU and GPU forwards differ at the threshold
    if cache.exists():
        body._stable_faces = np.load(cache)
        return body._stable_faces
    F = body.faces.astype(np.int64)

    def size(V):
        return np.sqrt(np.linalg.norm(np.cross(V[:, F[:, 1]] - V[:, F[:, 0]], V[:, F[:, 2]] - V[:, F[:, 0]]), axis=-1))

    k0 = size(body.pose().verts.cpu().numpy())[0]
    n = body.expr.shape[1]
    ok = np.ones(len(F), bool)
    for sgn in (1.0, -1.0):
        E = body.expr.repeat(n, 1) + sgn * amp * torch.eye(n, device=body.device)
        for i in range(0, n, 8):
            V = body.pose(body.body0.expand(len(E[i:i + 8]), -1), expr=E[i:i + 8]).verts.cpu().numpy()
            r = size(V) / np.maximum(k0, 1e-9)
            ok &= ((r < max_stretch) & (r > 1 / max_stretch)).all(0)
    body._stable_faces = ok
    return ok


def cage_layers(body: MHRBody, layers: list[dict], face_safe: bool = True) -> list[B.CageLayer]:
    """`face_safe`: leave the expression-unstable body faces (stable_faces) out of the cage."""
    owned = {c for L in layers for c in L["classes"]}
    BF = body.faces[stable_faces(body)] if face_safe else body.faces
    out = [B.CageLayer("body", body.verts_canon, BF, [c for c in range(N_CLASSES) if c not in owned])]
    stable = {tuple(sorted(f)) for f in body.faces[stable_faces(body)].tolist()} if face_safe else None
    for L in layers:
        F = L["F"]
        if face_safe and "rigid_joint" not in L and "skin_w" not in L:   # a layer face is as (un)stable as the body face its vertices copy
            bf = np.sort(np.asarray(L["bi"])[F], axis=1)
            F = F[np.array([tuple(r) in stable for r in bf.tolist()], bool)]
        out.append(B.CageLayer(L["name"], L["V"], F, L["classes"]))
    return out


def pose_layers(body: MHRBody, posed: Posed, layers: list[dict]) -> np.ndarray:
    """[B, n_verts_total, 3]: body vertices from the MHR forward, each layer vertex = its body vertex's posed position
    plus its canonical offset carried by the LBS transform of that body vertex."""
    canon = body.pose()
    Rr, tr = lbs.relative_transforms(posed, canon)
    sv, sj, sw = body.skin
    if not hasattr(body, "_topk"):
        W = lbs.dense_weights(sv, sj, sw, len(body.verts_canon), len(body.parents))
        body._topk = tuple(torch.as_tensor(a, device=body.device) for a in lbs.topk_weights(W, 8))
    idx, w = body._topk
    parts = [posed.verts]
    for L in layers:
        V = torch.as_tensor(L["V"], device=body.device)
        if "rigid_joint" in L:   # rigid with one joint (the hair shell with its head); dynamics come after (hair_dynamics)
            j = int(L["rigid_joint"])
            parts.append((Rr[:, j, None] @ V[None, :, :, None])[..., 0] + tr[:, j, None])
            continue
        if "skin_w" in L:   # its own (smoothed) weights: a loose-garment shell (rig/skirt.py)
            parts.append(lbs.skin(V, torch.as_tensor(L["skin_idx"], device=body.device),
                                  torch.as_tensor(L["skin_w"], device=body.device), Rr, tr))
            continue
        bi = torch.as_tensor(L["bi"], device=body.device)
        B0 = torch.as_tensor(body.verts_canon, device=body.device)[bi]
        moved = lbs.skin(V, idx[bi], w[bi], Rr, tr) - lbs.skin(B0, idx[bi], w[bi], Rr, tr)
        parts.append(posed.verts[:, bi] + moved)
    return torch.cat(parts, 1)


def hair_dynamics(body: MHRBody, params, expr, gtrans, posed: np.ndarray, layers: list[dict], fps: float, hp=None,
                  preroll: int = 8, pose_fn=None) -> np.ndarray:
    """Replace the rigid hair shell block of `posed` [T, n_verts_total, 3] by its secondary-motion simulation
    (rig/hair.py). params/expr/gtrans: the per-frame MHR inputs `posed` came from, or `pose_fn(i0, i1) -> Posed`
    (a motion with a root transform, motion/io.pose)."""
    from . import hair as H
    from .cage import vertex_normals
    names = [L for L in layers if "rigid_joint" in L and "theta" in L]
    if not names:
        return posed
    nb = len(body.verts_canon)
    canon = body.pose()
    Rs, ts = [], []
    T = len(posed)
    for i in range(0, T, 16):
        P = pose_fn(i, min(i + 16, T)) if pose_fn else \
            body.pose(params[i:i + 16], expr=expr[i:i + 16], global_trans=None if gtrans is None else gtrans[i:i + 16])
        Rr, tr = lbs.relative_transforms(P, canon)
        Rs.append(Rr.cpu().numpy()); ts.append(tr.cpu().numpy())
    Rs = np.concatenate(Rs); ts = np.concatenate(ts)
    BF = body.faces.astype(np.int64)
    off = nb
    out = posed.copy()
    for L in layers:
        n = len(L["V"])
        if L in names:
            j = int(L["rigid_joint"])
            # collide against the body vertices near the shell
            from scipy.spatial import cKDTree
            d, _ = cKDTree(L["V"]).query(body.verts_canon)
            near = np.nonzero(d < 0.12)[0]
            bv = posed[:, :nb]
            bn = np.stack([vertex_normals(bv[f].astype(np.float64), BF)[near] for f in range(T)])
            p = hp
            if p is None and "hair_params" in L:
                p = H.HairParams(**L["hair_params"])
            out[:, off:off + n] = H.simulate(L["V"].astype(np.float64), L["F"], L["theta"], L["r"], Rs[:, j], ts[:, j], fps,
                                             p, body_verts=bv[:, near].astype(np.float64), body_normals=bn, preroll=preroll)
        off += n
    return out


def pose_motion(body: MHRBody, m, layers: list[dict], hp=None) -> np.ndarray:
    """[T, n_verts_total, 3] cage vertices of a motion (motion/io.Motion): layers posed, hair simulated."""
    from ..motion import io as MI
    T = len(m)
    posed = torch.cat([pose_layers(body, MI.pose(body, m, i, min(i + 16, T)), layers) for i in range(0, T, 16)]).cpu().numpy()
    return hair_dynamics(body, None, None, None, posed, layers, m.fps, hp, pose_fn=lambda a, b: MI.pose(body, m, a, b))
