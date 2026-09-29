"""Named handles on MHR's model parameters.

Derived from `mhr_model.pt`'s parameter_transform (model param -> joint DOF,
DOFs 3/4/5 = local Euler x/y/z) and the joint tree/positions of an exported
subject (subject faces +Z, so -X is the subject's right). Indices are into the
full 204-wide model_params row; body params are row[6:136].

Joint ids: pelvis 1, spine 34-37, neck 110, head 113, right arm 38 (clavicle)
39 (shoulder) 40 (elbow) 41/42 (wrist), left arm 74-78, right leg 18 (hip)
19 (knee) 20/21 (ankle) 22-24 (foot), left leg 2-8. Params 68-121 are fingers
(held at the subject's canonical hands); 122-135 are extra hip/ankle DOFs and
bone-length translations (never animated).
"""
from __future__ import annotations

# name -> (x, y, z) model-param indices, None where the joint lacks that DOF.
ROT = {
    "spine0": (6, 8, 10), "spine1": (12, 14, 16), "spine2": (18, 19, 20), "spine3": (21, 22, 23),
    "neck": (24, 25, 26), "head": (27, 28, 29),
    "r_clavicle": (30, 31, 32), "r_shoulder": (33, 34, 35), "r_elbow": (None, None, 36), "r_wrist": (37, 38, 39),
    "l_clavicle": (40, 41, 42), "l_shoulder": (43, 44, 45), "l_elbow": (None, None, 46), "l_wrist": (47, 48, 49),
    "r_hip": (50, 51, 52), "r_knee": (None, None, 53), "r_ankle": (54, 55, None),
    "l_hip": (59, 60, 61), "l_knee": (None, None, 62), "l_ankle": (63, 64, None),
}
GLOBAL_TRANS = (0, 1, 2)   # x10 (metres * 10), raw frame
GLOBAL_ROT = (3, 4, 5)
JOINT = {"pelvis": 1, "spine0": 34, "spine1": 35, "spine2": 36, "spine3": 37, "neck": 110, "head": 113,
         "r_clavicle": 38, "r_shoulder": 39, "r_elbow": 40, "r_wrist": 41,
         "l_clavicle": 74, "l_shoulder": 75, "l_elbow": 76, "l_wrist": 77,
         "r_hip": 18, "r_knee": 19, "r_ankle": 20, "l_hip": 2, "l_knee": 3, "l_ankle": 4}


# Twist DOFs: near null spaces of bone-direction targets. With the arm straight, humeral twist
# (shoulder.x) and forearm twist (wrist.x) turn the palm alike (+1/-1 moves it 6 deg), so an IK must pull them to
# neutral or they drift apart into a candy-wrapped forearm. (handle, axis index into ROT)
TWISTS = [(f"spine{i}", 0) for i in range(4)] + [("neck", 0)] + \
    [(f"{s}_{j}", ax) for s in "rl" for j, ax in (("clavicle", 0), ("shoulder", 0), ("wrist", 0), ("hip", 0))]
# (Not the ankle roll: the ankle has two DOFs and the ankle->toe direction uses both; pulling one bent the other.)
# Hinge minimums (absolute body-param values): MHR's elbow is straight at -0.6 (0 is bent ~35 deg), its knee at 0.
HINGE_MIN = {"r_elbow": -0.75, "l_elbow": -0.75, "r_knee": -0.15, "l_knee": -0.15}


def body_index(model_index: int) -> int:
    """Index into the 130-wide body slice for a model-param index."""
    if not 6 <= model_index < 136:
        raise ValueError(f"model param {model_index} is not a body param")
    return model_index - 6
