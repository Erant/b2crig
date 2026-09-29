"""Linear blend skinning of arbitrary points (cages) by the MHR skeleton.

The body itself is posed by `mhr_model.pt` (it has pose correctives LBS lacks);
cage layers are skinned here with per-vertex weights transferred from the body.
Joint transforms come from `MHRBody.forward`: world positions and world frame
orientations, so the relative transform of joint j is
    G_j = [Q_j Q0_j^T | c_j - Q_j Q0_j^T c0_j].
"""
from __future__ import annotations

import numpy as np
import torch

from .mhr import Posed


def relative_transforms(posed: Posed, canon: Posed) -> tuple[torch.Tensor, torch.Tensor]:
    """[B, J, 3, 3] rotations and [B, J, 3] translations taking canonical to posed."""
    Rrel = posed.rots @ canon.rots.transpose(-1, -2)
    trel = posed.joints - (Rrel @ canon.joints[..., None])[..., 0]
    return Rrel, trel


def dense_weights(skin_vertex, skin_joint, skin_weight, n_verts: int, n_joints: int) -> np.ndarray:
    W = np.zeros((n_verts, n_joints), np.float32)
    np.add.at(W, (skin_vertex, skin_joint), skin_weight)
    return W


def topk_weights(W: np.ndarray, k: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Top-k joints per vertex, renormalised: (idx [V, k] int64, w [V, k] float32)."""
    idx = np.argsort(-W, axis=1)[:, :k]
    w = np.take_along_axis(W, idx, 1)
    w = w / np.maximum(w.sum(1, keepdims=True), 1e-8)
    return idx.astype(np.int64), w.astype(np.float32)


def skin(points: torch.Tensor, idx: torch.Tensor, w: torch.Tensor,
         Rrel: torch.Tensor, trel: torch.Tensor) -> torch.Tensor:
    """points [V, 3] canonical, idx/w [V, k] -> [B, V, 3] posed."""
    R = Rrel[:, idx]                      # [B, V, k, 3, 3]
    t = trel[:, idx]                      # [B, V, k, 3]
    p = (R @ points[None, :, None, :, None])[..., 0] + t
    return (w[None, ..., None] * p).sum(-2)
