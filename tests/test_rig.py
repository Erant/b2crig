"""Rig checks against the exported bring-up subject (skipped when it has not been exported)."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from b2crig import b2ctrain as B
from b2crig.rig import lbs
from b2crig.rig.cage import transfer_weights, uniform_laplacian
from b2crig.rig.mhr import MHRBody

SUBJECT = Path(__file__).resolve().parents[1] / "work" / "416580" / "mhr.npz"
needs_subject = pytest.mark.skipif(not SUBJECT.exists() or not torch.cuda.is_available(),
                                   reason="bring-up subject not exported / no GPU")


@pytest.fixture(scope="module")
def body():
    return MHRBody(SUBJECT)


@needs_subject
def test_replay_is_canonical(body):
    v = body.pose().verts[0].cpu().numpy()
    assert np.abs(v - body.verts_canon).max() < 1e-5


@needs_subject
def test_lbs_identity_and_rigid_root(body):
    canon = body.pose()
    sv, sj, sw = body.skin
    W = lbs.dense_weights(sv, sj, sw, len(body.verts_canon), len(body.parents))
    idx, w = (torch.as_tensor(a, device="cuda") for a in lbs.topk_weights(W, 8))
    x0 = torch.as_tensor(body.verts_canon, device="cuda")
    R, t = lbs.relative_transforms(canon, canon)
    assert (lbs.skin(x0, idx, w, R, t)[0] - x0).abs().max() < 1e-5
    # A pure global rotation is rigid: LBS must agree with the MHR forward exactly.
    rot = body.model_params0[:, 3:6] + torch.tensor([[0.0, 0.7, 0.0]], device="cuda")
    posed = body.pose(global_rot=rot)
    R, t = lbs.relative_transforms(posed, canon)
    assert (lbs.skin(x0, idx, w, R, t)[0] - posed.verts[0]).norm(dim=-1).max() < 1e-4


def test_weight_inpainting_is_smooth():
    # A strip of vertices: the two ends are confident (joint 0 / joint 1), the middle is inpainted.
    n = 20
    V = np.stack([np.linspace(0, 1, n), np.zeros(n), np.zeros(n)], 1)
    V = np.concatenate([V, V + [0, 0.05, 0]])
    F = np.array([[i, i + 1, n + i] for i in range(n - 1)] + [[i + 1, n + i + 1, n + i] for i in range(n - 1)])
    L = uniform_laplacian(len(V), F)
    assert abs(L.sum()) < 1e-9
    bodyV = np.array([[0, 0, 0.0], [1, 0, 0.0], [0, 0.05, 0], [1, 0.05, 0]])
    bodyF = np.array([[0, 1, 2], [1, 3, 2]])
    bodyW = np.array([[1.0, 0], [0, 1.0], [1.0, 0], [0, 1.0]])
    idx, w, conf = transfer_weights(V, F, bodyV, bodyF, bodyW, max_dist=0.02, min_cos=-1, k=2)
    W = np.zeros((len(V), 2))
    np.put_along_axis(W, idx, w, 1)
    assert conf[0] and conf[n - 1] and not conf[n // 2]
    assert np.all(np.diff(W[:n, 1]) >= -1e-6)      # joint-1 weight rises monotonically along the strip


def test_cage_file_layout(tmp_path):
    L = B.CageLayer("a", np.zeros((3, 3)), np.array([[0, 1, 2]]), classes=[1, 4])
    M = B.CageLayer("b", np.ones((3, 3)), np.array([[0, 2, 1]]), classes=[23])
    posed = np.arange(2 * 6 * 3, dtype=np.float32).reshape(2, 6, 3)
    B.write_cage(tmp_path / "c.b2ccage", [L, M], posed, ["f0", "f1"])
    raw = (tmp_path / "c.b2ccage").read_bytes()
    assert raw[:8] == b"B2CCAGE1"
    assert struct.unpack("<4i", raw[8:24]) == (2, 6, 2, 2)
    l0 = struct.unpack("<4iI", raw[24:44]); l1 = struct.unpack("<4iI", raw[44:64])
    assert l0 == (0, 3, 0, 1, (1 << 1) | (1 << 4)) and l1 == (3, 3, 1, 1, 1 << 23)
    faces = np.frombuffer(raw, np.int32, 6, 64 + 6 * 12).reshape(2, 3)
    assert faces.tolist() == [[0, 1, 2], [3, 5, 4]]
    assert len(raw) == 64 + 6 * 12 + 2 * 12 + 2 * 64 + posed.nbytes
