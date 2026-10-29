"""Mixamo FBX animations as a retargeting source (tools/retarget_smplx.py).

A Mixamo download (FBX Binary, any frame rate, with or without skin; the mesh is ignored) holds the 65-bone
`mixamorig:` skeleton: centimetres, Y-up, facing +Z with +X the character's left, toes on y = 0, bind pose a T-pose.
That is b2crig's frame, and the same shape as an AMASS skeleton (motion/smplx.Skel): io/fbx.py evaluates the bones'
world transforms (FBX's transform order, Pre/Post rotations), the rest is the static (bind) transform, and the world
rotations are taken relative to it (rest = identity).

The joints get the SMPL-X names the retargeting uses, plus the finger chains (motion/soma.py's names). Mixamo has no
metacarpals: Hand<Finger>1 is the knuckle (SMPL-X's index1 / middle1 / pinky1, the chain's MCP), 2-3 the PIP / DIP
and 4 the tip; the thumb's 1-4 are its CMC / MCP / IP / tip.

"In Place" downloads have the travel removed; leave it unticked to keep the root motion. Mixamo's loops repeat their
first pose as the last frame.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..io import fbx
from .smplx import Skel

FPS = 30.0


def joint_map(idx: dict[str, int]) -> dict[str, int]:
    """b2crig/SMPL-X joint names -> skeleton indices, from Mixamo bone name (without `mixamorig:`) -> index."""
    J = {"pelvis": idx["Hips"], "spine1": idx["Spine"], "spine2": idx["Spine1"], "spine3": idx["Spine2"],
         "neck": idx["Neck"], "head": idx["Head"]}
    for s, S in (("l", "Left"), ("r", "Right")):
        J |= {f"{s}_hip": idx[f"{S}UpLeg"], f"{s}_knee": idx[f"{S}Leg"], f"{s}_ankle": idx[f"{S}Foot"],
              f"{s}_foot": idx[f"{S}ToeBase"], f"{s}_toe_end": idx[f"{S}Toe_End"],
              f"{s}_collar": idx[f"{S}Shoulder"], f"{s}_shoulder": idx[f"{S}Arm"], f"{s}_elbow": idx[f"{S}ForeArm"],
              f"{s}_wrist": idx[f"{S}Hand"], f"{s}_index1": idx[f"{S}HandIndex1"],
              f"{s}_middle1": idx[f"{S}HandMiddle1"], f"{s}_pinky1": idx[f"{S}HandPinky1"]}
        for f, F in (("index", "Index"), ("middle", "Middle"), ("ring", "Ring"), ("pinky", "Pinky")):
            J |= {f"{s}_{f}_{k}": idx[f"{S}Hand{F}{i}"] for i, k in enumerate(("mcp", "pip", "dip", "tip"), 1)}
        J |= {f"{s}_thumb_{k}": idx[f"{S}HandThumb{i}"] for i, k in enumerate(("cmc", "mcp", "ip", "tip"), 1)}
    return J


def world(sc: fbx.Scene, t: np.ndarray | None) -> tuple[list[str], np.ndarray]:
    """Bone names (without the `mixamorig:` prefix) and world matrices [T, B, 4, 4] in metres at times t (s; None:
    the static bind transforms, one frame)."""
    ids = list(sc.bones)
    W = {}

    def go(i):
        if i not in W:
            b = sc.bones[i]
            M = fbx.local_matrices(b, t)
            W[i] = go(b.parent) @ M if b.parent in sc.bones else M
        return W[i]
    for i in ids:
        go(i)
    M = np.stack([W[i] for i in ids], 1)
    M[..., :3, 3] *= sc.unit_cm / 100.0
    return [sc.bones[i].name.split(":")[-1] for i in ids], M


def load_mixamo(path: str | Path, fps_out: float = FPS, t0: float = 0.0, dur: float | None = None) -> Skel:
    """The FBX's skeleton (world joints, rest-relative world rotations), sampled at fps_out from t0 for dur seconds
    (default: to the last key)."""
    sc = fbx.load(path)
    if sc.up_axis != 1:
        raise ValueError(f"{path}: up axis {sc.up_axis}; Mixamo exports are Y-up")
    names, M0 = world(sc, None)
    idx = {n: i for i, n in enumerate(names)}
    if "Hips" not in idx:
        raise ValueError(f"{path}: no mixamorig:Hips (not a Mixamo skeleton?)")
    t1 = sc.stop if dur is None else min(sc.stop, t0 + dur)
    _, M = world(sc, np.arange(t0, t1 + 1e-6, 1.0 / fps_out))
    rots = M[..., :3, :3] @ np.swapaxes(M0[0, :, :3, :3], -1, -2)[None]
    rest = M0[0, :, :3, 3]
    sole = float(rest[[idx["LeftToe_End"], idx["RightToe_End"]], 1].min())   # the bind pose stands its toes on y = 0
    return Skel(joints=M[..., :3, 3], rots=rots, fps=fps_out, rest_pelvis_height=float(rest[idx["Hips"], 1] - sole),
                rest_joints=rest, J=joint_map(idx), floor=sole)


def is_mixamo(path: str | Path) -> bool:
    return Path(path).suffix.lower() == ".fbx"
