"""b2crig/plyheader.py against b2crunner's docs/ply-header-records.md (written from the spec; no b2crunner import)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from b2crig import plyheader as H

HEADER = [
    "ply", "format binary_little_endian 1.0", "comment Exported from Brush",
    "comment b2c.mhr.version 2",
    "comment b2c.mhr.frame world = ...; pose_params are SAM-3D-Body raw",
    "comment b2c.mhr.model facebook/sam-3d-body-dinov3 assets/mhr_model.pt",
    "comment b2c.mhr.world_from_raw.scale 1 0.9864",
    "comment b2c.mhr.world_from_raw.rotation 3x3 1 0 0 0 1 0 0 0 1",
    "comment b2c.mhr.pose_params.shape_params 3 0.5 -0.25 1",
    "comment b2c.mhr.joint_parents 3 -1 0 1",
    "comment b2c.mhr.joints 3x3 0 0 0 0 1 0 0 2 0",
    "comment b2c.orbit.version 1",
    "comment b2c.orbit.helix.n_frames 1 96",
    "comment b2c.orbit.helix.amplitude_deg 1 12.5",
    "comment b2c.orbit.orbit_cameras.image_size 2 540 960",
    "comment b2c.orbit.orbit_cameras.position 2x3 0 0 3 3 0 0",
    'comment b2c.orbit.extras json {"orbit_target":[0,0.9,0]}',
    "element vertex 0", "property float x", "end_header",
]


@pytest.fixture
def ply(tmp_path):
    p = tmp_path / "scene.ply"
    p.write_bytes(("\n".join(HEADER) + "\n").encode("ascii"))
    return p


def test_body_record(ply):
    b = H.read_body(ply)
    assert b["version"] == "2" and b["model"].endswith("mhr_model.pt") and b["frame"].startswith("world = ")
    assert b["world_from_raw"]["scale"] == pytest.approx(0.9864)
    assert b["world_from_raw"]["rotation"].shape == (3, 3)
    assert b["pose_params"]["shape_params"].dtype == np.float32
    assert b["joint_parents"].dtype == np.int64 and b["joint_parents"].tolist() == [-1, 0, 1]
    assert b["joints"].shape == (3, 3) and b["joints"][2, 1] == 2.0


def test_orbit_record(ply):
    o = H.read_orbit(ply)
    assert o["version"] == 1
    assert o["helix"]["n_frames"] == 96 and isinstance(o["helix"]["n_frames"], int)
    assert o["helix"]["amplitude_deg"] == 12.5
    assert o["orbit_cameras"]["image_size"].dtype == np.int64 and o["orbit_cameras"]["image_size"].tolist() == [540, 960]
    assert o["orbit_cameras"]["position"].shape == (2, 3)
    assert o["extras"] == json.loads('{"orbit_target":[0,0.9,0]}')


def test_missing_record_raises(tmp_path):
    p = tmp_path / "bare.ply"
    p.write_bytes(b"ply\nformat ascii 1.0\ncomment other\nend_header\n")
    with pytest.raises(ValueError):
        H.read_orbit(p)
    with pytest.raises(ValueError):
        H.read_body(p)
