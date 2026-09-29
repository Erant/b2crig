"""rig/joint_split: the halves of a split splat are the exact truncated moments of the parent."""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from b2crig.rig import joint_split as JS


def test_halves_recompose_the_parent():
    rng = np.random.default_rng(0)
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    f = {"x": np.array([0.003], np.float32), "y": np.zeros(1, np.float32), "z": np.zeros(1, np.float32),
         "scale_0": np.log([0.008]).astype(np.float32), "scale_1": np.log([0.004]).astype(np.float32),
         "scale_2": np.log([0.002]).astype(np.float32), "rot_0": q[:1].astype(np.float32),
         "rot_1": q[1:2].astype(np.float32), "rot_2": q[2:3].astype(np.float32), "rot_3": q[3:].astype(np.float32),
         "opacity": np.zeros(1, np.float32), "f_dc_0": np.ones(1, np.float32)}
    n = np.array([1.0, 0.0, 0.0])
    planes = [(np.zeros(3), n, "r", 0.05)]
    out, used, k = JS.split_once(f, planes, np.array(["r"]), np.zeros(1, np.int64))
    assert k == 1 and len(out["x"]) == 2 and (used == 1).all()
    mu = np.array([0.003, 0, 0.0])
    S = JS._cov(np.stack([f[f"scale_{i}"] for i in range(3)], 1).astype(float),
                np.stack([f[f"rot_{i}"] for i in range(4)], 1).astype(float))[0]
    s = np.sqrt(n @ S @ n); m_pos = ndtr(mu[0] / s)
    X = np.stack([out["x"], out["y"], out["z"]], 1).astype(float)
    C = JS._cov(np.stack([out[f"scale_{i}"] for i in range(3)], 1).astype(float),
                np.stack([out[f"rot_{i}"] for i in range(4)], 1).astype(float))
    w = np.array([m_pos, 1 - m_pos]) if X[0, 0] > X[1, 0] else np.array([1 - m_pos, m_pos])
    mean = (w[:, None] * X).sum(0)
    cov = sum(w[i] * (C[i] + np.outer(X[i] - mean, X[i] - mean)) for i in range(2))
    np.testing.assert_allclose(mean, mu, atol=1e-6)
    np.testing.assert_allclose(cov, S, atol=1e-8)
    assert (X[:, 0] > 0).sum() == 1                          # one child on each side of the cut
    assert out["f_dc_0"].tolist() == [1.0, 1.0]              # appearance copied
