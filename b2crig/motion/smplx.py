"""SMPL-X forward kinematics for AMASS (MoSh++ stage-ii) motion files: joint positions and world rotations.

Only the skeleton is needed for retargeting (tools/retarget_smplx.py): rest joints from the shaped template
(J_regressor @ (v_template + shapedirs @ betas)), then plain FK. AMASS is Z-up; `load_amass` returns everything in
b2crig's Y-up frame (x, y, z) <- (x, z, -y).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

MODEL_DIR = Path.home() / "Downloads" / "models_lockedhead" / "smplx"
ZUP_TO_YUP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], np.float64)

# SMPL-X joint ids (body 0-21, jaw 22, eyes 23/24, left hand 25-39, right hand 40-54; per hand index1-3, middle1-3,
# pinky1-3, ring1-3, thumb1-3)
J = {"pelvis": 0, "l_hip": 1, "r_hip": 2, "spine1": 3, "l_knee": 4, "r_knee": 5, "spine2": 6, "l_ankle": 7,
     "r_ankle": 8, "spine3": 9, "l_foot": 10, "r_foot": 11, "neck": 12, "l_collar": 13, "r_collar": 14, "head": 15,
     "l_shoulder": 16, "r_shoulder": 17, "l_elbow": 18, "r_elbow": 19, "l_wrist": 20, "r_wrist": 21,
     "l_index1": 25, "l_middle1": 28, "l_pinky1": 31, "r_index1": 40, "r_middle1": 43, "r_pinky1": 46}


@dataclass
class Skel:
    joints: np.ndarray    # [T, 55, 3] world (Y-up, metres)
    rots: np.ndarray      # [T, 55, 3, 3] world joint orientations relative to the rest pose (rest = identity)
    fps: float
    rest_pelvis_height: float   # rest-pose pelvis above the soles (shaped template)
    rest_joints: np.ndarray     # [55, 3] shaped rest joints (template frame, pelvis at its template position)
    J: dict = field(default_factory=lambda: dict(J))   # joint name -> index (these SMPL-X names; motion/soma.py maps SOMA's)
    floor: float | None = None  # the floor height in `joints`, when the source knows it (else estimated from the feet)


def load_model(gender: str) -> dict:
    g = gender.upper() if gender.lower() in ("male", "female") else "NEUTRAL"
    return dict(np.load(MODEL_DIR / f"SMPLX_{g}.npz", allow_pickle=True))


def fk(parents: np.ndarray, rest: np.ndarray, local: np.ndarray, trans: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """local [T, J, 3, 3] -> (world rots [T, J, 3, 3], joints [T, J, 3]); root at rest[0] + trans."""
    T, nj = local.shape[:2]
    G = np.zeros_like(local)
    P = np.zeros((T, nj, 3))
    for j in range(nj):
        p = parents[j]
        if p < 0:
            G[:, j] = local[:, j]
            P[:, j] = rest[j] + trans
        else:
            G[:, j] = G[:, p] @ local[:, j]
            P[:, j] = P[:, p] + np.einsum("tij,j->ti", G[:, p], rest[j] - rest[p])
    return G, P


def load_amass(path: str | Path, fps_out: float = 30.0, t0: float = 0.0, dur: float | None = None) -> Skel:
    d = np.load(path, allow_pickle=True)
    model = load_model(str(d["gender"]))
    fps = float(d["mocap_frame_rate"])
    step = max(1, int(round(fps / fps_out)))
    i0 = int(round(t0 * fps))
    i1 = len(d["trans"]) if dur is None else min(len(d["trans"]), i0 + int(round(dur * fps)))
    sl = slice(i0, i1, step)
    poses, trans = d["poses"][sl], d["trans"][sl]
    nb = int(d["num_betas"]) if "num_betas" in d.files else len(d["betas"])
    betas = d["betas"][:nb]
    v = model["v_template"] + np.einsum("vcb,b->vc", model["shapedirs"][:, :, :nb], betas)
    rest = model["J_regressor"] @ v
    parents = model["kintree_table"][0].astype(np.int64)
    parents[0] = -1
    T = len(poses)
    local = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(T, 55, 3, 3)
    G, P = fk(parents, rest, local, trans)
    # to Y-up: x' = C x
    G = np.einsum("ij,tkjl->tkil", ZUP_TO_YUP, G)
    P = P @ ZUP_TO_YUP.T
    return Skel(joints=P, rots=G, fps=fps / step, rest_pelvis_height=float(rest[0, 1] - v[:, 1].min()), rest_joints=rest)
