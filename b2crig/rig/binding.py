"""Dual-binding candidates: splats whose limb the seg label cannot decide, with a second triangle on the other limb.

In the canonical pose limbs touch (a hand resting at the pocket, an arm against the torso); there Sapiens2 labels are
unreliable (the glove at the pocket reads Lower_Clothing) and nearest-triangle binding picks a limb by proximity. A
splat is ambiguous when cage vertices of two different limb groups lie within `radius` of it. Its binding A is what
b2ctrain picks (the layer owning its class, nearest vertex); candidate B is the nearest triangle of another group, in
any layer. b2ctrain fit-cage --fit-binding then learns, per splat, which one the video's motion supports.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

GROUP = {"pelvis": "torso", "spine0": "torso", "spine1": "torso", "spine2": "torso", "spine3": "torso",
         "neck": "head", "head": "head",
         "r_clavicle": "torso", "r_shoulder": "r_arm", "r_elbow": "r_arm", "r_wrist": "r_arm",
         "l_clavicle": "torso", "l_shoulder": "l_arm", "l_elbow": "l_arm", "l_wrist": "l_arm",
         "r_hip": "r_leg", "r_knee": "r_leg", "r_ankle": "r_leg", "l_hip": "l_leg", "l_knee": "l_leg", "l_ankle": "l_leg"}


def candidates(xyz: np.ndarray, label: np.ndarray, conf: np.ndarray, opacity: np.ndarray, cage_V: np.ndarray,
               cage_F: np.ndarray, vert_layer: np.ndarray, layer_classes: list, vert_group: np.ndarray,
               radius: float = 0.03, min_conf: float = 0.5, min_opacity: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """(splat indices, face B per splat). `vert_group` [V]: limb-group name of each cage vertex."""
    owner = np.full(len(label), -1)
    for L, cls in enumerate(layer_classes):
        owner[np.isin(label, list(cls)) & (conf >= min_conf)] = L
    body = 0
    owner[(owner < 0) & (conf >= min_conf)] = body               # unowned classes: the body layer
    # binding A's group: nearest vertex of the owning layer (any layer for low confidence)
    gA = np.empty(len(label), object)
    for L in np.unique(owner):
        m = owner == L
        pool = np.arange(len(cage_V)) if L < 0 else np.nonzero(vert_layer == L)[0]
        _, j = cKDTree(cage_V[pool]).query(xyz[m]); gA[m] = vert_group[pool[j]]
    # ambiguous: a vertex of another group within the radius
    tree = cKDTree(cage_V)
    near = tree.query_ball_point(xyz, radius)
    face_group = np.array([max(set(g), key=list(g).count) for g in vert_group[cage_F]])
    cent = cage_V[cage_F].mean(1)
    ftree = {g: (np.nonzero(face_group == g)[0], cKDTree(cent[face_group == g])) for g in np.unique(face_group)}
    idx, fb = [], []
    for i in np.nonzero(opacity > min_opacity)[0]:
        other = {vert_group[k] for k in near[i]} - {gA[i]}
        if not other:
            continue
        best = None
        for g in other:
            fi, t = ftree[g]
            d, j = t.query(xyz[i])
            if best is None or d < best[0]:
                best = (d, fi[j])
        idx.append(i); fb.append(best[1])
    return np.array(idx, np.int32), np.array(fb, np.int32)


def write_alt(path: Path, splat: np.ndarray, face_b: np.ndarray, w: np.ndarray | float = 0.5) -> None:
    w = np.broadcast_to(np.asarray(w, np.float32), splat.shape)
    rec = np.zeros(len(splat), dtype=[("s", "<i4"), ("f", "<i4"), ("w", "<f4")])
    rec["s"], rec["f"], rec["w"] = splat, face_b, w
    with open(path, "wb") as f:
        f.write(b"B2CALT01"); f.write(np.array([len(splat)], "<i4").tobytes()); f.write(rec.tobytes())


def read_alt(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = Path(path).read_bytes()
    assert raw[:8] == b"B2CALT01"
    n = int(np.frombuffer(raw, "<i4", 1, 8)[0])
    rec = np.frombuffer(raw, dtype=[("s", "<i4"), ("f", "<i4"), ("w", "<f4")], count=n, offset=12)
    return rec["s"].copy(), rec["f"].copy(), rec["w"].copy()
