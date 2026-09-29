"""Training data for the motion -> cage-displacement model.

Per clip and frame t:
  * input: a causal window of the driving skeleton — for each of the named joints (rig.skeleton.JOINT) its rotation
    relative to its named parent as 6D (first two columns), plus the root's orientation relative to gravity and its
    linear velocity / acceleration in the root frame, each frame of the window; and finite-difference angular
    velocity / acceleration of the joints (per second, so fps-independent) at t.
  * target: the fitted displacement of every cage vertex, rotated into that vertex's LBS frame at t (so a
    displacement that "rides" the limb is constant), with the per-vertex visibility of frame t as the weight.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..evaluate import read_cage
from ..rig import lbs
from ..rig.mhr import MHRBody
from ..rig.skeleton import JOINT

NAMED = list(JOINT)          # joint order of the features
PARENT = {"pelvis": None, "spine0": "pelvis", "spine1": "spine0", "spine2": "spine1", "spine3": "spine2", "neck": "spine3",
          "head": "neck", "r_clavicle": "spine3", "r_shoulder": "r_clavicle", "r_elbow": "r_shoulder", "r_wrist": "r_elbow",
          "l_clavicle": "spine3", "l_shoulder": "l_clavicle", "l_elbow": "l_shoulder", "l_wrist": "l_elbow",
          "r_hip": "pelvis", "r_knee": "r_hip", "r_ankle": "r_knee", "l_hip": "pelvis", "l_knee": "l_hip", "l_ankle": "l_knee"}
UP = np.array([0.0, 1.0, 0.0])


@dataclass
class ClipData:
    name: str
    feats: np.ndarray    # [T, D] per-frame skeleton features
    delta: np.ndarray    # [T, V, 3] local-frame displacement
    weight: np.ndarray   # [T, V]
    fps: float


def skeleton_features(posed_rots: np.ndarray, posed_joints: np.ndarray, fps: float) -> np.ndarray:
    """[T, D] from world joint rotations [T, J, 3, 3] and positions [T, J, 3]."""
    T = len(posed_rots)
    cols = []
    for n in NAMED:
        j = JOINT[n]; p = PARENT[n]
        R = posed_rots[:, j]
        if p is not None:
            R = np.swapaxes(posed_rots[:, JOINT[p]], -1, -2) @ R
        cols.append(R[..., :, :2].reshape(T, 6))
    rot6 = np.concatenate(cols, 1)                                   # [T, 6J]
    root_R = posed_rots[:, JOINT["pelvis"]]
    up_root = np.einsum("tji,j->ti", root_R, UP)                    # gravity direction in the root frame
    pos = posed_joints[:, JOINT["pelvis"]]
    vel = np.gradient(pos, 1 / fps, axis=0); acc = np.gradient(vel, 1 / fps, axis=0)
    vel_r = np.einsum("tji,tj->ti", root_R, vel); acc_r = np.einsum("tji,tj->ti", root_R, acc)
    w = np.gradient(rot6, 1 / fps, axis=0); a = np.gradient(w, 1 / fps, axis=0)
    return np.concatenate([rot6, up_root, vel_r, acc_r, 0.1 * w, 0.01 * a], 1).astype(np.float32)


def vertex_frames(body: MHRBody, posed, cage_layers_bi: np.ndarray) -> torch.Tensor:
    """[T, V, 3, 3] blended (then orthonormalised) LBS rotation of every cage vertex; cage vertex -> body vertex map
    `cage_layers_bi` (body vertices map to themselves)."""
    canon = body.pose()
    Rr, _ = lbs.relative_transforms(posed, canon)
    if not hasattr(body, "_topk"):
        W = lbs.dense_weights(*body.skin, len(body.verts_canon), len(body.parents))
        body._topk = tuple(torch.as_tensor(x, device=body.device) for x in lbs.topk_weights(W, 8))
    idx, w = body._topk
    bi = torch.as_tensor(cage_layers_bi, device=body.device)
    M = (w[bi][None, ..., None, None] * Rr[:, idx[bi]]).sum(2)     # [T, V, 3, 3]
    U, _, Vh = torch.linalg.svd(M)
    return U @ Vh


def load_clip(clip: Path, body: MHRBody, fit: str = "fit_layered", cage: str = "cage_layered.b2ccage",
              layer_bi: np.ndarray | None = None) -> ClipData:
    c = read_cage(clip / cage)
    T, V = len(c.names), len(c.verts)
    d = np.fromfile(clip / fit / "delta.f32", np.float32).reshape(T, V, 3)
    vis = np.fromfile(clip / fit / "vis.f32", np.float32).reshape(T, V)
    m = np.load(clip / "motion.npz")
    fps = float(m["fps"])
    params = torch.as_tensor(m["body_params"], device=body.device)
    feats, dl = [], []
    for i in range(0, T, 16):
        posed = body.pose(params[i:i + 16])
        feats.append((posed.rots.cpu().numpy(), posed.joints.cpu().numpy()))
        Rv = vertex_frames(body, posed, layer_bi)
        dl.append(torch.einsum("tvji,tvj->tvi", Rv, torch.as_tensor(d[i:i + 16], device=body.device)).cpu().numpy())
    rots = np.concatenate([f[0] for f in feats]); joints = np.concatenate([f[1] for f in feats])
    return ClipData(clip.name, skeleton_features(rots, joints, fps), np.concatenate(dl), np.minimum(vis, 3.0) / 3.0, fps)


def layered_pair(clip: Path) -> tuple[str, str]:
    """(fit dir, cage file) of the clip's layered fit: the refit's *_layered pair, else the queue's own when layered."""
    if (clip / "cage_layered.b2ccage").exists() and (clip / "fit_layered").exists():
        return "fit_layered", "cage_layered.b2ccage"
    if len(read_cage(clip / "cage.b2ccage").layers) > 1:
        return "fit", "cage.b2ccage"
    raise FileNotFoundError(f"{clip}: no layered fit (run tools/refit.py)")


def cage_body_index(body: MHRBody, subject: Path) -> np.ndarray:
    from ..rig import layered
    parts = [np.arange(len(body.verts_canon))] + [L["bi"] for L in layered.load_layers(subject)]
    return np.concatenate(parts)


def windows(feats: np.ndarray, K: int) -> np.ndarray:
    """[T, K*D]: frame t sees frames t-K+1..t (the first frame repeated before the clip starts)."""
    T = len(feats)
    idx = np.clip(np.arange(T)[:, None] - np.arange(K)[None, ::-1], 0, T - 1)
    return feats[idx].reshape(T, -1)


def named_influence(body: MHRBody, layer_bi: np.ndarray) -> np.ndarray:
    """[V, len(NAMED)] skinning weight of every cage vertex per named joint: each MHR joint counts for its nearest
    named ancestor (cage vertices take their body vertex's weights through `layer_bi`)."""
    parents = np.asarray(body.parents)
    named = {JOINT[n]: k for k, n in enumerate(NAMED)}

    def anc(j: int) -> int:
        while j >= 0 and j not in named:
            j = int(parents[j])
        return named.get(j, named[JOINT["pelvis"]])

    jmap = np.array([anc(j) for j in range(len(parents))])
    W = lbs.dense_weights(*body.skin, len(body.verts_canon), len(parents))
    Wn = np.zeros((len(NAMED), len(body.verts_canon)), np.float32)
    np.add.at(Wn, jmap, W.T)
    return Wn.T[layer_bi]


def relative_rotations(posed) -> np.ndarray:
    """[T, len(NAMED), 3, 3] rotation of each named joint relative to its named parent (world for the pelvis)."""
    R = posed.rots.cpu().numpy()
    out = []
    for n in NAMED:
        Rj = R[:, JOINT[n]]; p = PARENT[n]
        out.append(Rj if p is None else np.swapaxes(R[:, JOINT[p]], -1, -2) @ Rj)
    return np.stack(out, 1)
