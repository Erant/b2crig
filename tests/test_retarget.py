"""Retargeting IK regressions (skipped without the bring-up subject / GPU / the SMPL-X model files)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from b2crig.motion import smplx as SX
from b2crig.motion.retarget import ik_to_targets
from b2crig.rig.mhr import MHRBody
from b2crig.rig.skeleton import JOINT

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import retarget_smplx as RS  # noqa: E402

SUBJECT = Path(__file__).resolve().parents[1] / "work" / "416580" / "mhr.npz"
needs_subject = pytest.mark.skipif(not SUBJECT.exists() or not torch.cuda.is_available(),
                                   reason="bring-up subject not exported / no GPU")
needs_smplx = pytest.mark.skipif(not (SX.MODEL_DIR / "SMPLX_NEUTRAL.npz").exists(), reason="no SMPL-X model files")


@pytest.fixture(scope="module")
def body():
    return MHRBody(SUBJECT)


def yaw(th):
    return Rotation.from_euler("y", np.asarray(th)[:, None]).as_matrix()


@needs_subject
def test_root_follows_full_turns(body):
    """Two full turns: the root rotation must track through pi and 2 pi (a rotation-vector root spun the body round
    for ~4 frames at every half turn)."""
    T = 48
    th = np.linspace(0, 4 * np.pi, T)
    R0 = yaw(th)
    with torch.no_grad():
        Jz = body.pose(torch.zeros(1, 130, device="cuda")).joints[0].cpu().numpy()
    bl = [(JOINT["pelvis"], JOINT["neck"]), (JOINT["r_hip"], JOINT["l_hip"]), (JOINT["r_shoulder"], JOINT["l_shoulder"])]
    ma, mb = np.array([b[0] for b in bl]), np.array([b[1] for b in bl])
    d0 = Jz[mb] - Jz[ma]; d0 /= np.linalg.norm(d0, axis=1, keepdims=True)
    dS = torch.as_tensor(np.einsum("tij,bj->tbi", R0, d0), dtype=torch.float32, device="cuda")
    Rm0 = body.pose(torch.zeros(1, 130, device="cuda")).rots[0]
    oj = [JOINT["pelvis"]]
    Rt = torch.as_tensor(R0, dtype=torch.float32, device="cuda")[:, None] @ Rm0[oj][None]
    m = ik_to_targets(body, dS, (torch.as_tensor(ma, device="cuda"), torch.as_tensor(mb, device="cuda"),
                                 torch.ones(len(bl), device="cuda")), Rt, (oj, torch.ones(1, device="cuda")),
                      np.repeat(body.joints_canon[JOINT["pelvis"]][None], T, 0), R0, 30.0, 150, verbose=False,
                      neutral=True, w_smooth=1.0)
    err = np.degrees(Rotation.from_matrix(np.einsum("tji,tjk->tik", R0, m.root_R)).magnitude())
    assert err.max() < 10, f"root left the turn by {err.max():.0f} deg at frame {err.argmax()}"


@needs_subject
@needs_smplx
def test_rest_relative_targets_turn_with_the_body(body):
    """A source rigidly yawed through a half turn: the torso / neck / clavicle targets must be MHR's rest directions
    yawed alike (a world-frame shortest arc flipped them over the top at 180 deg: pinched shoulder blades)."""
    model = SX.load_model("neutral")
    rest = model["J_regressor"] @ model["v_template"]
    th = np.linspace(0, np.pi, 13)
    R = yaw(th)
    sk = SX.Skel(joints=np.einsum("tij,kj->tki", R, rest), rots=np.repeat(R[:, None], 55, 1), fps=30.0,
                 rest_pelvis_height=float(rest[0, 1] - model["v_template"][:, 1].min()), rest_joints=rest)
    tg = RS.build_targets(body, sk, verbose=False)
    dS = tg["dS"].cpu().numpy(); ma, mb, _ = (x.cpu().numpy() for x in tg["bones"])
    with torch.no_grad():
        Jz = body.pose(torch.zeros(1, 130, device="cuda")).joints[0].cpu().numpy()
    Y0 = RS.yaw_to_front(sk.rots[:, SX.J["pelvis"]])   # build_targets faces the window's mean heading to +Z
    for i in RS.REST_RELATIVE:
        m0 = Jz[mb[i]] - Jz[ma[i]]; m0 /= np.linalg.norm(m0)
        want = np.einsum("ij,tjk,k->ti", Y0, R, m0)
        ang = np.degrees(np.arccos(np.clip((want * dS[:, i]).sum(-1), -1, 1)))
        assert ang.max() < 0.5, f"bone {i}: target off the rigid turn by {ang.max():.1f} deg"
