"""b2crig/subject.py's model lookup (SPEC 7.3) and motion/rom.tour (the pipeline's clip)."""
import hashlib

import numpy as np
import pytest
import torch

from b2crig import subject
from b2crig.motion import rom


class _Doc:
    def __init__(self, sha):
        self.json = {"nodes": [{"name": "b2c_body", "extensions": {"B2C_mhr": {"model": {
            "repo": "facebook/sam-3d-body-dinov3", "file": "assets/mhr_model.pt", "sha256": sha}}}}]}

    def node_index(self, name):
        return 0

    def extension(self, key, obj=None):
        return (obj or self.json).get("extensions", {}).get(key)


def _hub(tmp_path, data=b"model"):
    p = tmp_path / "hub" / "models--facebook--sam-3d-body-dinov3" / "snapshots" / "abc" / "assets" / "mhr_model.pt"
    p.parent.mkdir(parents=True)
    p.write_bytes(data)
    return p, hashlib.sha256(data).hexdigest()


def test_mhr_model_from_the_hub_cache(tmp_path, monkeypatch):
    p, sha = _hub(tmp_path)
    monkeypatch.delenv("B2C_MHR_MODEL_DIR", raising=False)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    assert subject.mhr_model(_Doc(sha)) == p.resolve()


def test_mhr_model_dir_wins_and_the_hash_is_checked(tmp_path, monkeypatch):
    _hub(tmp_path)
    d = tmp_path / "models"; d.mkdir(); (d / "mhr_model.pt").write_bytes(b"other")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setenv("B2C_MHR_MODEL_DIR", str(d))
    assert subject.mhr_model(_Doc(hashlib.sha256(b"other").hexdigest())) == (d / "mhr_model.pt").resolve()
    with pytest.raises(ValueError, match="sha256"):
        subject.mhr_model(_Doc(hashlib.sha256(b"model").hexdigest()))


def test_mhr_model_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("B2C_MHR_MODEL_DIR", raising=False)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "none"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "none"))
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        subject.mhr_model(_Doc("0" * 64))


def test_tour_starts_and_ends_at_the_canonical_pose():
    body0 = torch.linspace(-0.1, 0.1, 130)[None]
    expr = torch.zeros(1, 72)
    m = rom.tour(body0, expr, names=("t_pose", "twist"), fps=10.0, move=1.0, hold=0.5)
    bp = m["body_params"]
    assert bp.shape == (1 + 3 * (10 + 5), 130) and m["expr"].shape == (len(bp), 72) and m["fps"] == 10.0
    np.testing.assert_allclose(bp[0], body0[0].numpy()); np.testing.assert_allclose(bp[-1], body0[0].numpy())
    np.testing.assert_allclose(bp[10], rom.to_body(rom.named_poses()["t_pose"], body0)[0].numpy(), atol=1e-6)
    assert np.abs(np.diff(bp, axis=0)).max() < 0.5   # blended, no jumps
