"""Inputs b2crig computes for b2ctrain (BOUNDARY.md): the cage-open gate section and the pose containment files."""
from __future__ import annotations

import json

import numpy as np
from plyfile import PlyData, PlyElement

from b2crig import b2ctrain as B
from b2crig.evaluate import read_cage
from b2crig.rig import contain, open_gate


def _strip(n: int = 40, length: float = 0.4):
    """A flat strip of triangles along x (two rows of vertices 2 cm apart)."""
    x = np.linspace(0, length, n)
    V = np.concatenate([np.stack([x, np.zeros(n), np.zeros(n)], 1), np.stack([x, np.full(n, 0.02), np.zeros(n)], 1)])
    F = [(i, i + 1, n + i) for i in range(n - 1)] + [(i + 1, n + i + 1, n + i) for i in range(n - 1)]
    return V.astype(np.float32), np.asarray(F, np.int64)


def _bend(V: np.ndarray, deg: float) -> np.ndarray:
    """Rotate the half x > 0.2 about the z axis through x = 0.2 (a joint turning)."""
    P = V.copy(); m = P[:, 0] > 0.2; t = np.radians(deg); c, s = np.cos(t), np.sin(t)
    d = P[m] - [0.2, 0, 0]
    P[m] = np.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1], d[:, 2]], 1) + [0.2, 0, 0]
    return P


def test_gate_zero_canonical_and_rigid_positive_bent():
    V, F = _strip()
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], np.float32)
    th = open_gate.theta(V, F, np.stack([V, V @ R.T + [1, 2, 3], _bend(V, 60)]))
    assert th[0].max() < 0.1 and th[1].max() < 0.1          # canonical, and the whole strip moved rigidly
    near = np.abs(V[:, 0] - 0.2) < 0.1                        # vertices with partners across the joint
    assert th[2][near].max() > 45 and th[2][V[:, 0] < 0.05].max() < 0.1


def test_gate_section_round_trip(tmp_path):
    V, F = _strip()
    posed = np.stack([V, _bend(V, 30), _bend(V, 80)])
    p = tmp_path / "c.b2ccage"
    B.write_cage(p, [B.CageLayer("body", V, F, [])], posed, ["a", "b", "c"])
    assert not open_gate.has_open_gate(p)
    th = open_gate.add_open_gate(p); open_gate.add_open_gate(p)     # replaces, does not stack
    assert open_gate.has_open_gate(p)
    assert np.abs(open_gate.read_theta(p) - th).max() < 0.1         # float16 in the file
    assert np.array_equal(read_cage(p).posed, posed.astype(np.float32))   # the cage body is untouched


def test_contain_inputs(tmp_path):
    # body = a closed tetrahedron-ish box around the origin; one splat inside it, one outside
    V = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], np.float32) * 0.2
    F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6], [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3]])
    cg_path = tmp_path / "c.b2ccage"
    B.write_cage(cg_path, [B.CageLayer("body", V, F, [])], np.stack([V, V + [0, 0.5, 0]]), ["f0", "f1"])
    cg = read_cage(cg_path)
    d = np.zeros(2, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")]); d["x"] = [0.1, 0.5]; d["y"] = [0.1, 0.0]; d["z"] = [0.1, 0.0]
    PlyData([PlyElement.describe(d, "vertex")]).write(str(tmp_path / "s.ply"))
    assert contain.deep_mask(tmp_path / "s.ply", cg, depth=0.02).tolist() == [1, 0]
    cap = tmp_path / "cap"; cap.mkdir()
    (cap / "cameras.txt").write_text("1 PINHOLE 1080 1920 1000 1000 540 960\n")
    (cap / "images.txt").write_text("1 1 0 0 0 0 0 3 1 a.png\n\n")
    flags = contain.write_contain(tmp_path / "out", cg, tmp_path / "s.ply", cap, n=4, res=960)
    cams = json.loads((tmp_path / "out" / "contain_cameras.json").read_text())
    assert (cams["width"], cams["height"]) == (540, 960) and [c["name"].split("@")[0] for c in cams["cameras"]] == ["f0", "f0", "f1", "f1"]
    # the camera keeps the capture camera's offset from the (posed) body centre
    assert np.allclose(np.asarray(cams["cameras"][2]["position"]) - [0, 0.5, 0], [0, 0, -3], atol=1e-5)
    assert flags[0] == "--pose-contain-cameras" and (tmp_path / "out" / "contain_exclude.u8").stat().st_size == 2
