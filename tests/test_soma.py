"""Kimodo SOMA source (motion/soma.py) and the finger retargeting it drives."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from b2crig.motion import io as MI
from b2crig.motion import soma
from b2crig.motion.smplx import J as SMPLX_J
from b2crig.rig.skeleton import FINGERS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import retarget_smplx as RS  # noqa: E402

SUBJECT = Path(__file__).resolve().parents[1] / "work" / "416580" / "mhr.npz"
needs_subject = pytest.mark.skipif(not SUBJECT.exists() or not torch.cuda.is_available(),
                                   reason="bring-up subject not exported / no GPU")


def test_joint_map_covers_the_retarget():
    J = soma.joint_map()
    for (a, b), _, _ in RS.bone_pairs(J):
        assert 0 <= a < 77 and 0 <= b < 77
    for name, *_ in RS.ORIENT:
        assert name in J
    assert all(n in J for n in RS.REST_PARENT.values())
    fp = RS.finger_pairs(J)
    assert len(fp) == 2 * (4 * 3 + 3)                       # every phalanx of both hands
    assert {m for _, pair, _ in fp for m in pair} <= {j for ch in FINGERS.values() for j in ch}
    assert RS.finger_pairs(SMPLX_J) == []                   # SMPL-X has no fingertips: canonical hands


def test_kimodo_file_loads_as_a_skeleton(tmp_path):
    """A Kimodo NPZ at rest (identity rotations, the T-pose joints): FK-consistent, resampled, floor known."""
    rest = soma.load_rest()
    T = 31
    P = np.repeat(rest[None] + [0, 1.0, 0], T, 0); P[:, :, 0] += np.linspace(0, 0.3, T)[:, None]
    G = np.repeat(np.eye(3)[None, None], T, 0).repeat(77, 1)
    f = tmp_path / "k.npz"
    np.savez(f, posed_joints=P, global_rot_mats=G, local_rot_mats=G)
    assert soma.is_kimodo(f)
    sk = soma.load_kimodo(f, fps_out=15.0)
    assert len(sk.joints) == 16 and sk.fps == 15.0 and sk.floor == 0.0
    np.testing.assert_allclose(sk.joints[-1], P[-1], atol=1e-9)
    assert 0.95 < sk.rest_pelvis_height < 1.05


@needs_subject
def test_hands_flag_poses_the_fingers():
    from b2crig.rig.mhr import MHRBody
    body = MHRBody(SUBJECT)
    hb = (body.hand_idx - 6).long()
    p = body.body0.clone(); p[0, hb] += 0.3
    canon, animated = body.model_params(p), body.model_params(p, hands=True)
    assert torch.equal(canon[0, body.hand_idx], body.model_params0[0, body.hand_idx])
    assert torch.allclose(animated[0, body.hand_idx], body.model_params0[0, body.hand_idx] + 0.3)
    d = {"body_params": p.cpu().numpy(), "expr": body.expr.cpu().numpy(), "fps": 30.0, "hands": True}
    m = MI.Motion(d)
    assert m.hands and m.save_fields()["hands"]
    J = MI.pose(body, m, 0, 1).joints
    J0 = body.pose(p).joints
    moved = (J - J0).norm(dim=-1)[0]
    assert moved[FINGERS["r_index"][-1]] > 5e-3 and moved[1] < 1e-6   # fingertips move, the pelvis does not
