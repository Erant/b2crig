"""Garment / hair cage layers from the splat's seg labels.

For each loose layer (a set of Sapiens2 classes):
  1. write the splats of those classes to their own PLY;
  2. b2ctrain probe --depth over an orbit of views, then b2ctrain mesh-fuse: the layer's outer surface;
  3. decimate to a few thousand vertices (quadric, fast_simplification);
  4. skin: copy the MHR body's weights at confident closest points (near and similarly oriented), then
     inpaint the rest with a Laplacian solve over the cage (Abdrashitov et al., Robust Skin Weights
     Transfer via Weight Inpainting, SIGGRAPH Asia 2023), so e.g. the fabric between the legs gets smooth
     weights instead of a left/right split.

The body layer is the MHR mesh itself (posed by the MHR forward, correctives included).
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

from .. import b2ctrain as B
from .. import cameras as C

DEFAULT_LAYERS = {
    "upper": (23, 1),     # Upper_Clothing, Apparel
    "lower": (13,),       # Lower_Clothing
    "hair": (4,),
    "face": (3, 24, 25),  # Face_Neck, lips: the face splats sit ~6 mm (p90 20 mm) off the MHR face; binding them to the
                          # body with that offset makes them swing and balloon under expressions (the layer's vertices
                          # follow their body vertex's posed position, expression included)
}

# Body regions (dominant named joint of a body vertex) a layer may copy. Garment splats vote for their nearest body
# vertex; in the canonical pose a hand resting at the pocket is nearer than the thigh, so without this the trouser
# layer took in hand vertices - and their skinning: trouser patches that travel with the hand.
LAYER_EXCLUDE = {
    "upper": ("r_hip", "l_hip", "r_knee", "l_knee", "r_ankle", "l_ankle"),
    "lower": ("r_clavicle", "l_clavicle", "r_shoulder", "l_shoulder", "r_elbow", "l_elbow", "r_wrist", "l_wrist",
              "neck", "head"),
    "hair": ("r_elbow", "l_elbow", "r_wrist", "l_wrist"),
    "face": ("r_clavicle", "l_clavicle", "r_shoulder", "l_shoulder", "r_elbow", "l_elbow", "r_wrist", "l_wrist",
             "r_hip", "l_hip", "r_knee", "l_knee", "r_ankle", "l_ankle", "pelvis", "spine0", "spine1", "spine2"),
}


@dataclass
class SkinnedLayer:
    layer: B.CageLayer
    idx: np.ndarray       # [V, k] joints
    w: np.ndarray         # [V, k] weights
    confident: np.ndarray  # [V] bool, weights copied (not inpainted)


def write_class_splat(src: Path, dst: Path, classes: Sequence[int], min_conf: float = 0.5) -> int:
    ply = PlyData.read(str(src))
    v = ply["vertex"].data
    keep = np.isin(np.rint(v["seg_label"]).astype(int), list(classes)) & (v["seg_conf"] >= min_conf)
    PlyData([PlyElement.describe(v[keep], "vertex")], text=False, comments=[c for c in ply.comments if not c.startswith("b2c.")]).write(str(dst))
    return int(keep.sum())


def fuse_layer(splat: Path, work: Path, K: C.Intrinsics, target, radius: float, voxel: float = 0.005) -> tuple[np.ndarray, np.ndarray]:
    work.mkdir(parents=True, exist_ok=True)
    poses, names = [], []
    for el in (-20.0, 5.0, 35.0, 70.0):
        for p in C.circular_orbit(target, radius, el, 12, azimuth0_deg=15.0 * (el > 0)):
            poses.append(p); names.append(f"e{int(el)}_{len(names):03d}")
    C.write_cameras(work / "cams.json", K, poses, names)
    subprocess.run([str(B.B2CTRAIN), "probe", "--splat", str(splat), "--cameras", str(work / "cams.json"),
                    "--output-dir", str(work / "probe"), "--depth", "--images", "--tau", "0.5"], check=True, capture_output=True)
    xyz = np.stack([PlyData.read(str(splat))["vertex"][k] for k in "xyz"], 1)
    lo, hi = np.percentile(xyz, 0.5, 0) - 0.05, np.percentile(xyz, 99.5, 0) + 0.05
    subprocess.run([str(B.B2CTRAIN), "mesh-fuse", "--output", str(work / "fused.ply"),
                    "--bbox", ",".join(f"{x:.4f}" for x in (*lo, *hi)),
                    "--views", str(work / "cams.json"), str(work / "probe"),
                    "--carve", str(work / "cams.json"), str(work / "probe"),
                    "--voxel", str(voxel)], check=True, capture_output=True)
    m = PlyData.read(str(work / "fused.ply"))
    V = np.stack([m["vertex"][k] for k in "xyz"], 1).astype(np.float64)
    F = np.stack(m["face"]["vertex_indices"]).astype(np.int64)
    return V, F


def outer_sheet(V: np.ndarray, F: np.ndarray, body_V: np.ndarray, body_F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep the faces of a fused garment slab that face away from the body.

    TSDF fusion of a thin layer of garment splats gives a closed slab: an outer sheet (the garment as seen) and an
    inner one facing the body. Splats sit on the outer sheet; the inner one would only double the cage and bind
    nothing, so it is dropped (faces whose normal points towards the nearest body point), and so are the vertices
    left unreferenced."""
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    cen = V[F].mean(1)
    _, i = cKDTree(body_V).query(cen)
    keep = np.einsum("ij,ij->i", fn, cen - body_V[i]) > 0
    F = F[keep]
    used = np.unique(F)
    remap = -np.ones(len(V), np.int64); remap[used] = np.arange(len(used))
    return V[used], remap[F]


def decimate(V: np.ndarray, F: np.ndarray, target_verts: int) -> tuple[np.ndarray, np.ndarray]:
    import fast_simplification
    if len(V) <= target_verts:
        return V, F
    Vd, Fd = fast_simplification.simplify(V.astype(np.float32), F.astype(np.int32), target_reduction=1.0 - target_verts / len(V))
    return Vd.astype(np.float64), Fd.astype(np.int64)


def vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n = np.zeros_like(V)
    for k in range(3):
        np.add.at(n, F[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def uniform_laplacian(n: int, F: np.ndarray) -> sp.csr_matrix:
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    d = np.asarray(A.sum(1)).ravel()
    return (sp.diags(d) - A).tocsr()


def transfer_weights(V: np.ndarray, F: np.ndarray, body_V: np.ndarray, body_F: np.ndarray, body_W: np.ndarray,
                     max_dist: float = 0.06, min_cos: float = 0.3, k: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense body weights [Vb, J] -> top-k weights on the cage, with Laplacian inpainting of unconfident vertices."""
    # Closest body sample (vertices plus face centroids and edge midpoints) with its interpolated weights.
    bn = vertex_normals(body_V, body_F)
    cen = body_V[body_F].mean(1)
    samples = np.concatenate([body_V, cen])
    s_w = np.concatenate([body_W, body_W[body_F].mean(1)])
    s_n = np.concatenate([bn, vertex_normals(body_V, body_F)[body_F].mean(1)])
    s_n /= np.maximum(np.linalg.norm(s_n, axis=1, keepdims=True), 1e-12)
    d, i = cKDTree(samples).query(V)
    cn = vertex_normals(V, F)
    cos = np.einsum("ij,ij->i", cn, s_n[i])
    confident = (d < max_dist) & (cos > min_cos)
    W = s_w[i].astype(np.float64)
    if (~confident).any() and confident.any():
        L = uniform_laplacian(len(V), F)
        Q = (L.T @ L).tocsr()
        u = np.nonzero(~confident)[0]
        c = np.nonzero(confident)[0]
        Quu = Q[u][:, u].tocsc()
        rhs = -(Q[u][:, c] @ W[c])
        solve = spla.factorized(Quu + 1e-8 * sp.eye(len(u), format="csc"))
        W[u] = np.stack([solve(rhs[:, j]) for j in range(W.shape[1])], 1) if W.shape[1] else W[u]
    W = np.clip(W, 0, None)
    W /= np.maximum(W.sum(1, keepdims=True), 1e-12)
    idx = np.argsort(-W, axis=1)[:, :k]
    w = np.take_along_axis(W, idx, 1)
    w /= np.maximum(w.sum(1, keepdims=True), 1e-12)
    return idx.astype(np.int64), w.astype(np.float32), confident


def offset_layer(body_V: np.ndarray, body_F: np.ndarray, splat_xyz: np.ndarray, splat_label: np.ndarray,
                 splat_ok: np.ndarray, classes: Sequence[int], min_count: int = 3, min_frac: float = 0.5,
                 dilate_rings: int = 1, smooth_iters: int = 10, max_offset: float = 0.12,
                 allowed: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A garment layer as an offset copy of the body region it covers.

    Every (confident, opaque) splat votes for its nearest body vertex; a vertex belongs to the layer when at least
    `min_count` splats of `classes` chose it and they are at least `min_frac` of all its splats; the region is grown by
    `dilate_rings` and kept where whole faces are inside. Each layer vertex is pushed along the body normal to the
    median signed distance of its garment splats (vertices without any take their neighbours' offset, by Jacobi
    smoothing), so the layer lies on the garment. `allowed` (bool per body vertex): the garment splats vote only among
    these, and the layer stays inside them (see LAYER_EXCLUDE). Returns (V, F, body_index): the layer's vertices, faces (local
    indices) and, per vertex, the body vertex it copies (whose skin weights it takes, exactly).
    """
    bn = vertex_normals(body_V, body_F)
    tree = cKDTree(body_V)
    _, near = tree.query(splat_xyz[splat_ok])
    lab = splat_label[splat_ok]
    xyz = splat_xyz[splat_ok]
    mine = np.isin(lab, list(classes))
    if allowed is not None:                  # garment splats vote among the allowed body vertices only
        idx = np.nonzero(allowed)[0]
        _, na = cKDTree(body_V[idx]).query(xyz[mine])
        near = near.copy(); near[np.nonzero(mine)[0]] = idx[na]
    n_all = np.bincount(near, minlength=len(body_V))
    n_mine = np.bincount(near[mine], minlength=len(body_V))
    inside = (n_mine >= min_count) & (n_mine >= min_frac * np.maximum(n_all, 1))
    L = uniform_laplacian(len(body_V), body_F)
    adj = (sp.diags(L.diagonal()) - L).tocsr()
    for _ in range(dilate_rings):
        inside = inside | (adj @ inside.astype(float) > 0)
    if allowed is not None:
        inside &= allowed
    fmask = inside[body_F].all(1)
    F = body_F[fmask]
    used = np.unique(F)
    remap = -np.ones(len(body_V), np.int64); remap[used] = np.arange(len(used))
    # signed normal distance of each garment splat to its body vertex, median per vertex
    sd = np.einsum("ij,ij->i", xyz[mine] - body_V[near[mine]], bn[near[mine]])
    off = np.full(len(body_V), np.nan)
    order = np.argsort(near[mine]); nm = near[mine][order]; sdo = sd[order]
    starts = np.searchsorted(nm, used); ends = np.searchsorted(nm, used, side="right")
    for u, s0, e0 in zip(used, starts, ends):
        if e0 > s0:
            off[u] = np.median(sdo[s0:e0])
    o = off[used]
    known = ~np.isnan(o)
    o = np.where(known, o, 0.0)
    A = adj[used][:, used]
    deg = np.maximum(np.asarray(A.sum(1)).ravel(), 1)
    for _ in range(smooth_iters * 5):   # fill the unknown ones
        o = np.where(known, o, (A @ o) / deg)
    for _ in range(smooth_iters):       # then smooth everything a little
        o = 0.5 * o + 0.5 * (A @ o) / deg
    o = np.clip(o, -0.02, max_offset)
    V = body_V[used] + bn[used] * o[:, None]
    return V, remap[F], used
