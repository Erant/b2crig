"""Kimodo SOMA motions (NVIDIA Kimodo text-to-motion, `somaskel77`) as a retargeting source (tools/retarget_smplx.py).

A Kimodo NPZ holds `posed_joints` [T, 77, 3] and `global_rot_mats` [T, 77, 3, 3] (world orientations relative to
Kimodo's standard T-pose, so rest = identity) at 30 fps, Y-up, facing +Z, +X the subject's left, floor at y = 0:
already b2crig's frame, and the same shape as an AMASS skeleton (motion/smplx.Skel). The rest joints are the T-pose
`neutral_joints` (hips at the origin); FK over them reproduces `posed_joints` exactly.

The joints get the SMPL-X names the retargeting uses (pelvis, l_hip, ...; the hand's MCP knuckles are SMPL-X's
index1 / middle1 / pinky1) plus the finger chains: <side>_<finger>_mcp / _pip / _dip / _tip and
<side>_thumb_cmc / _mcp / _ip / _tip.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .smplx import Skel

FPS = 30.0
# somaskel77 joint order (kimodo/skeleton/definitions.py SOMASkeleton77)
NAMES = ["Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "HeadEnd", "Jaw", "LeftEye", "RightEye"]
for _side in ("Left", "Right"):
    NAMES += [f"{_side}Shoulder", f"{_side}Arm", f"{_side}ForeArm", f"{_side}Hand"]
    NAMES += [f"{_side}HandThumb{i}" for i in (1, 2, 3)] + [f"{_side}HandThumbEnd"]
    for _f in ("Index", "Middle", "Ring", "Pinky"):
        NAMES += [f"{_side}Hand{_f}{i}" for i in (1, 2, 3, 4)] + [f"{_side}Hand{_f}End"]
for _side in ("Left", "Right"):
    NAMES += [f"{_side}Leg", f"{_side}Shin", f"{_side}Foot", f"{_side}ToeBase", f"{_side}ToeEnd"]
IDX = {n: i for i, n in enumerate(NAMES)}


def joint_map() -> dict[str, int]:
    """b2crig/SMPL-X joint names -> somaskel77 indices."""
    J = {"pelvis": IDX["Hips"], "spine1": IDX["Spine1"], "spine2": IDX["Spine2"], "spine3": IDX["Chest"],
         "neck": IDX["Neck1"], "head": IDX["Head"]}
    for s, S in (("l", "Left"), ("r", "Right")):
        J |= {f"{s}_hip": IDX[f"{S}Leg"], f"{s}_knee": IDX[f"{S}Shin"], f"{s}_ankle": IDX[f"{S}Foot"],
              f"{s}_foot": IDX[f"{S}ToeBase"], f"{s}_toe_end": IDX[f"{S}ToeEnd"],
              f"{s}_collar": IDX[f"{S}Shoulder"], f"{s}_shoulder": IDX[f"{S}Arm"], f"{s}_elbow": IDX[f"{S}ForeArm"],
              f"{s}_wrist": IDX[f"{S}Hand"],
              # SMPL-X's index1 / middle1 / pinky1 are the MCP knuckles (SOMA's *2; SOMA's *1 is the metacarpal base)
              f"{s}_index1": IDX[f"{S}HandIndex2"], f"{s}_middle1": IDX[f"{S}HandMiddle2"],
              f"{s}_pinky1": IDX[f"{S}HandPinky2"]}
        for f, F in (("index", "Index"), ("middle", "Middle"), ("ring", "Ring"), ("pinky", "Pinky")):
            J |= {f"{s}_{f}_mcp": IDX[f"{S}Hand{F}2"], f"{s}_{f}_pip": IDX[f"{S}Hand{F}3"],
                  f"{s}_{f}_dip": IDX[f"{S}Hand{F}4"], f"{s}_{f}_tip": IDX[f"{S}Hand{F}End"]}
        J |= {f"{s}_thumb_cmc": IDX[f"{S}HandThumb1"], f"{s}_thumb_mcp": IDX[f"{S}HandThumb2"],
              f"{s}_thumb_ip": IDX[f"{S}HandThumb3"], f"{s}_thumb_tip": IDX[f"{S}HandThumbEnd"]}
    return J


def rest_joints() -> np.ndarray:
    """Kimodo's standard T-pose joints [77, 3] (hips at the origin), from its skeleton assets."""
    from kimodo.skeleton.definitions import SOMASkeleton77   # only in Kimodo's venv, so only when not cached
    return SOMASkeleton77().neutral_joints.cpu().numpy().astype(np.float64)


REST_CACHE = Path(__file__).with_name("somaskel77_rest.npy")


def load_rest() -> np.ndarray:
    if REST_CACHE.exists():
        return np.load(REST_CACHE)
    return rest_joints()


def load_kimodo(path: str | Path, fps_out: float = FPS, t0: float = 0.0, dur: float | None = None,
                fps_in: float = FPS) -> Skel:
    d = np.load(path)
    P, G = d["posed_joints"].astype(np.float64), d["global_rot_mats"].astype(np.float64)
    if P.shape[1] != len(NAMES):
        raise ValueError(f"{path}: {P.shape[1]} joints, expected somaskel77's {len(NAMES)} (a Kimodo-SOMA NPZ)")
    T = len(P)
    t_in = np.arange(T) / fps_in
    t1 = t_in[-1] if dur is None else min(t_in[-1], t0 + dur)
    t_out = np.arange(t0, t1 + 1e-9, 1.0 / fps_out)
    if len(t_out) != T or t0 != 0.0 or fps_out != fps_in:
        P = np.stack([np.stack([np.interp(t_out, t_in, P[:, j, c]) for c in range(3)], -1) for j in range(len(NAMES))], 1)
        G = np.stack([Slerp(t_in, Rotation.from_matrix(G[:, j]))(t_out).as_matrix() for j in range(len(NAMES))], 1)
    rest = load_rest()
    toe_end = [IDX["LeftToeEnd"], IDX["RightToeEnd"]]
    # soles: SOMA's bind mesh sits 7 mm under the toe-end joints (Kimodo puts that on its floor, y = 0)
    return Skel(joints=P, rots=G, fps=fps_out, rest_pelvis_height=float(rest[0, 1] - rest[toe_end, 1].min() + 0.007),
                rest_joints=rest, J=joint_map(), floor=0.0)


def is_kimodo(path: str | Path) -> bool:
    with np.load(path) as d:
        return "posed_joints" in d.files and "global_rot_mats" in d.files
