"""Range-of-motion pose library: static MHR poses that push every body region to its anatomical extremes, for
b2ctrain's pose containment loss (tools/pose_library.py rom, tools/pose_select.py).

Purely synthetic, so it can ship with b2ctrain: joint ranges are textbook adult range of motion, not taken from any
motion-capture dataset. Poses are ABSOLUTE joint rotations over MHR's zero pose (standing, arms in a ~45 deg A), not
offsets from a subject's canonical pose, so the same library means the same poses on every subject.

Axis semantics (probed on MHR 2026-09-29; same sign on both sides unless noted):
  spine0-3  x twist, y lateral bend (+: towards the subject's right), z flexion (+: forward bend)
  neck      x twist, y lateral, z flexion;  head x turn, y tilt, z nod
  clavicle  x axial rotation, y elevation (+: shrug), z protraction (+: forward)
  wrist x/y/z, ankle x/y, neck x, clavicle x: not probed; ranges are textbook magnitudes, signs unverified
  shoulder  x humeral twist, y abduction (+: arm out/up, 0 = A pose), z flexion (+: arm forward/up)
  elbow z / knee z  flexion (+)
  hip       x twist, y adduction (+: across the midline, -: out to the side), z flexion (-: leg forward)
"""
from __future__ import annotations

import numpy as np

from ..rig.skeleton import ROT, body_index

AXES = {"x": 0, "y": 1, "z": 2}

# (min, max) radians. Four spine segments share the trunk's range.
RANGES = {
    **{f"spine{i}.{a}": r for i in range(4) for a, r in (("x", (-0.15, 0.15)), ("y", (-0.18, 0.18)), ("z", (-0.25, 0.35)))},
    "neck.x": (-0.5, 0.5), "neck.y": (-0.35, 0.35), "neck.z": (-0.4, 0.55),
    "head.x": (-0.9, 0.9), "head.y": (-0.4, 0.4), "head.z": (-0.4, 0.7),
    **{f"{s}_{j}": r for s in "rl" for j, r in (
        ("clavicle.x", (-0.8, 0.4)), ("clavicle.y", (-0.2, 0.55)), ("clavicle.z", (-0.5, 0.4)),
        ("shoulder.x", (-1.1, 1.1)), ("shoulder.y", (-0.8, 2.3)), ("shoulder.z", (-0.8, 2.6)),
        ("elbow.z", (-0.1, 2.4)), ("wrist.x", (-1.1, 0.8)), ("wrist.y", (-0.4, 0.6)), ("wrist.z", (-1.0, 1.0)),
        ("hip.x", (-0.6, 0.6)), ("hip.y", (-0.8, 0.35)), ("hip.z", (-2.0, 0.35)),
        ("knee.z", (-0.1, 2.3)), ("ankle.x", (-0.7, 0.3)), ("ankle.y", (-0.6, 0.6)))},
}
GROUPS = {
    "legs": [k for k in RANGES if k[2:5] in ("hip", "kne", "ank")],
    "arms": [k for k in RANGES if k[2:5] in ("sho", "elb", "cla", "wri")],
    "trunk": [k for k in RANGES if k.startswith(("spine", "neck", "head"))],
}
# DOFs whose sign flips under a left/right mirror (twists and lateral bends).
MIRROR_FLIP = {"x"}


def mirror(pose: dict) -> dict:
    out = {}
    for k, v in pose.items():
        h, ax = k.split(".")
        if h[:2] in ("r_", "l_"):
            h = ("l_" if h[0] == "r" else "r_") + h[2:]
            out[f"{h}.{ax}"] = -v if ax in MIRROR_FLIP else v
        else:
            out[k] = -v if ax in ("x", "y") else v
    return out


def _spine(z=0.0, y=0.0, x=0.0) -> dict:   # a whole-trunk bend spread over the four segments
    return {f"spine{i}.{a}": v / 4 for i in range(4) for a, v in (("x", x), ("y", y), ("z", z)) if v}


def _both(**kw) -> dict:                   # the same joint values on both sides: _both(hip_z=-1.0)
    return {f"{s}_{k.replace('_', '.', 1)}": v for s in "rl" for k, v in kw.items()}


# Composite poses: what dancers, athletes and everyday life do with several joints at once. One-sided poses get
# their mirror too (named_poses()).
NAMED = {
    "squat_deep": {**_both(hip_z=-1.9, knee_z=2.2, hip_y=-0.25, shoulder_z=1.4), **_spine(z=0.5)},
    "squat_half": {**_both(hip_z=-1.2, knee_z=1.3, shoulder_z=1.5), **_spine(z=0.3)},
    "sumo": {**_both(hip_z=-1.2, knee_z=1.4, hip_y=-0.7, hip_x=0.4)},
    "lunge": {"r_hip.z": -1.4, "r_knee.z": 1.6, "l_hip.z": 0.3, "l_knee.z": 0.4, **_spine(z=-0.1)},
    "lunge_side": {"r_hip.y": -0.6, "r_hip.z": -0.9, "r_knee.z": 1.4, "l_hip.y": -0.6, **_spine(y=0.2)},
    "knee_to_chest": {"r_hip.z": -2.0, "r_knee.z": 2.3, **_spine(z=0.3), "r_shoulder.z": 1.0, "r_elbow.z": 1.6},
    "high_kick": {"r_hip.z": -1.8, "r_knee.z": 0.1, "l_hip.z": 0.1, **_spine(z=-0.2)},
    "side_kick": {"r_hip.y": -0.8, "r_hip.z": -0.4, "r_hip.x": 0.4, "r_knee.z": 0.2, **_spine(y=-0.3)},
    "back_kick": {"r_hip.z": 0.35, "r_knee.z": 2.0, **_spine(z=-0.5), **_both(shoulder_y=1.2)},
    "splits_side": {**_both(hip_y=-0.8, hip_x=0.3), **_both(shoulder_y=1.0)},
    "kneel_sit": {**_both(hip_z=-0.3, knee_z=2.3), **_spine(z=0.1)},
    "sit_chair": {**_both(hip_z=-1.55, knee_z=1.55), **_spine(z=0.1)},
    "cross_legged": {**_both(hip_z=-1.4, knee_z=2.2, hip_y=-0.6, hip_x=0.6), **_spine(z=0.3)},
    "toe_touch": {**_both(hip_z=-1.3), **_spine(z=1.0), **_both(shoulder_z=1.7), "neck.z": 0.3},
    "back_arch": {**_spine(z=-0.6), "neck.z": -0.4, **_both(shoulder_z=2.4, shoulder_y=0.3), **_both(hip_z=0.3)},
    "side_bend": {**_spine(y=0.48), "neck.y": 0.2, "l_shoulder.y": 2.2, "l_elbow.z": 0.4, "r_shoulder.y": -0.5},
    "twist": {**_spine(x=0.6), "head.x": 0.8, "r_shoulder.z": 1.2, "l_shoulder.z": -0.5, "r_elbow.z": 1.0},
    "arms_overhead": {**_both(shoulder_y=2.3, clavicle_y=0.4)},
    "arms_forward_up": {**_both(shoulder_z=2.6, clavicle_y=0.3)},
    "arms_back": {**_both(shoulder_z=-0.8, shoulder_y=-0.2, clavicle_z=-0.3), **_spine(z=0.1)},
    "t_pose": {**_both(shoulder_y=0.85)},
    "hug_self": {**_both(shoulder_y=-0.5, shoulder_z=1.3, shoulder_x=0.6, elbow_z=2.0, clavicle_z=0.4), **_spine(z=0.2)},
    "arms_crossed": {**_both(shoulder_y=-0.4, shoulder_z=0.7, elbow_z=1.9, shoulder_x=0.5)},
    "hands_behind_head": {**_both(shoulder_y=1.9, shoulder_z=0.3, shoulder_x=-0.6, elbow_z=2.4, clavicle_y=0.3)},
    "hands_on_hips": {**_both(shoulder_y=0.3, shoulder_z=-0.4, shoulder_x=0.9, elbow_z=1.6)},
    "reach_across": {"r_shoulder.y": -0.7, "r_shoulder.z": 1.6, "r_clavicle.z": 0.4, **_spine(x=-0.4), "head.x": -0.5},
    "reach_up_one": {"r_shoulder.y": 2.3, "r_clavicle.y": 0.45, **_spine(y=-0.35), "l_shoulder.y": -0.6},
    "flex_biceps": {**_both(shoulder_y=0.9, shoulder_x=0.9, elbow_z=2.3)},
    "throw_windup": {"r_shoulder.y": 1.3, "r_shoulder.x": -1.0, "r_elbow.z": 1.6, "l_shoulder.z": 1.2, **_spine(x=-0.5, z=-0.1),
                     "l_hip.z": -0.5, "l_knee.z": 0.3},
    "punch": {"r_shoulder.z": 1.6, "r_shoulder.y": -0.3, "r_elbow.z": 0.1, "l_elbow.z": 2.2, "l_shoulder.z": 0.5, **_spine(x=0.5),
              "r_hip.z": -0.3, "r_knee.z": 0.5},
    "run_stride": {"r_hip.z": -1.2, "r_knee.z": 1.8, "l_hip.z": 0.35, "l_knee.z": 0.8, "r_shoulder.z": -0.7, "l_shoulder.z": 1.2,
                   **_both(elbow_z=1.6), **_spine(z=0.2)},
    "look_back": {"head.x": 0.9, "neck.y": 0.2, **_spine(x=0.5)},
    "chin_down": {"neck.z": 0.5, "head.z": 0.4},
    "chin_up": {"neck.z": -0.4, "head.z": -0.4},
    "shrug": {**_both(clavicle_y=0.45, shoulder_y=-0.3, elbow_z=1.4, shoulder_x=0.5)},
    "arabesque": {"l_hip.z": 0.35, "l_knee.z": 0.1, **_spine(z=0.6), "r_shoulder.z": 2.2, "l_shoulder.y": 1.2},
}


def named_poses() -> dict[str, dict]:
    out = {}
    for n, p in NAMED.items():
        out[n] = p
        m = mirror(p)
        if m != p:
            out[n + "_m"] = m
    return out


def random_pose(rng: np.random.Generator) -> dict:
    """A pose focused on one group (3-9 active DOFs) or the whole body (8-20), drawn towards their extremes
    (Beta(0.6, 0.6)), plus a lighter touch on the others."""
    group = rng.choice(["legs", "arms", "trunk", "all", "all"])
    keys = list(RANGES) if group == "all" else GROUPS[group]
    pose = {}
    n = int(rng.integers(8, 21)) if group == "all" else int(rng.integers(3, 10))   # whole-body poses move many joints
    for k in rng.choice(keys, size=min(len(keys), n), replace=False):
        lo, hi = RANGES[k]
        pose[str(k)] = float(lo + (hi - lo) * rng.beta(0.6, 0.6))
    for k in rng.choice(list(RANGES), size=4, replace=False):   # background variation
        if k not in pose:
            lo, hi = RANGES[k]
            pose[str(k)] = float(np.clip(rng.normal(0, 0.15 * (hi - lo)), lo, hi))
    return pose


def to_body(pose: dict, body0) -> "torch.Tensor":
    """[1, 130] body params: the subject's non-rotation params (bone lengths, extra DOFs) with every named joint
    rotation replaced by the pose (unnamed rotations at MHR zero)."""
    p = body0.clone()
    for h, v in ROT.items():
        for i in v:
            if i is not None:
                p[0, body_index(i)] = 0.0
    for k, val in pose.items():
        h, ax = k.split(".")
        p[0, body_index(ROT[h][AXES[ax]])] = val
    return p


# Self-collision: capsules between joints (MHR joint ids), radius from the subject's own skinned vertices.
SEGMENTS = {
    "torso": (1, 110), "head": (113, 113),
    "r_uarm": (39, 40), "r_farm": (40, 41), "l_uarm": (75, 76), "l_farm": (76, 77),
    "r_thigh": (18, 19), "r_shin": (19, 20), "l_thigh": (2, 3), "l_shin": (3, 4),
}
CHECK = [("torso", s) for s in ("r_farm", "l_farm", "r_uarm", "l_uarm", "r_shin", "l_shin")] + \
    [("head", s) for s in ("r_farm", "l_farm", "r_uarm", "l_uarm", "r_thigh", "l_thigh", "r_shin", "l_shin", "torso")] + \
    [("r_thigh", "l_thigh"), ("r_shin", "l_shin"), ("r_thigh", "l_shin"), ("l_thigh", "r_shin"),
     ("r_farm", "l_farm"), ("r_uarm", "l_farm"), ("l_uarm", "r_farm"),
     ("r_farm", "r_thigh"), ("l_farm", "l_thigh"), ("r_farm", "l_thigh"), ("l_farm", "r_thigh")]


def _seg_pts(a, b, n=12):
    t = np.linspace(0, 1, n)[:, None]
    return a + t * (b - a)


def _pt_seg_dist(P, a, b):
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=-1)


def capsule_radii(verts: np.ndarray, joints: np.ndarray, skin) -> dict:
    sv, sj, sw = (np.asarray(x) for x in skin)
    dom = np.full(len(verts), -1); best = np.zeros(len(verts))
    for v, j, w in zip(sv, sj, sw):
        if w > best[v]:
            best[v], dom[v] = w, j
    r = {}
    for n, (a, b) in SEGMENTS.items():
        m = dom == a if n != "torso" else np.isin(dom, [1, 34, 35, 36, 37])
        d = _pt_seg_dist(verts[m], joints[a], joints[b])
        r[n] = float(np.median(d)) if m.any() else 0.05
    return r


def collides(joints: np.ndarray, radii: dict, overlap: float = 0.8) -> str | None:
    """The first capsule pair that interpenetrates by more than `overlap` of its summed radii (contact is fine)."""
    for s1, s2 in CHECK:
        P = _seg_pts(joints[SEGMENTS[s1][0]], joints[SEGMENTS[s1][1]])
        d = _pt_seg_dist(P, joints[SEGMENTS[s2][0]], joints[SEGMENTS[s2][1]]).min()
        if d < (1 - overlap) * (radii[s1] + radii[s2]):
            return f"{s1}/{s2}"
    return None
