"""The MHR70 regressor derived from SAM-3D-Body's checkpoint (skipped without the checkpoint)."""
from __future__ import annotations

import numpy as np
import pytest

from b2crig.rig import mhr70


def _ckpt():
    try:
        return mhr70.checkpoint()
    except FileNotFoundError:
        return None


@pytest.mark.skipif(_ckpt() is None, reason="no SAM-3D-Body checkpoint")
def test_derived_regressor_matches_and_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("B2CRIG_CACHE", str(tmp_path))
    m = mhr70.mapping()
    assert m.shape == mhr70.SHAPE and m.dtype == np.float32
    np.testing.assert_allclose(m.sum(1), 1.0, atol=1e-5)
    assert mhr70.cache_path().exists()
    np.testing.assert_array_equal(mhr70.mapping(), m)   # the cached copy
