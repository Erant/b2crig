"""Mixamo FBX source (io/fbx.py, motion/mixamo.py). Mixamo's files may not be redistributed, so the FBX here is
written by the test."""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from b2crig.io import fbx
from b2crig.motion import mixamo
from b2crig.rig.skeleton import FINGERS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import retarget_smplx as RS  # noqa: E402

BONES = ["Hips", "Spine", "Spine1", "Spine2", "Neck", "Head"]
for _s in ("Left", "Right"):
    BONES += [f"{_s}{n}" for n in ("Shoulder", "Arm", "ForeArm", "Hand", "UpLeg", "Leg", "Foot", "ToeBase", "Toe_End")]
    BONES += [f"{_s}Hand{f}{i}" for f in ("Thumb", "Index", "Middle", "Ring", "Pinky") for i in (1, 2, 3, 4)]


def test_joint_map_covers_the_retarget():
    J = mixamo.joint_map({n: i for i, n in enumerate(BONES)})
    for (a, b), _, _ in RS.bone_pairs(J):
        assert 0 <= a < len(BONES) and 0 <= b < len(BONES)
    for name, *_ in RS.ORIENT:
        assert name in J
    fp = RS.finger_pairs(J)
    assert len(fp) == 2 * (4 * 3 + 3)                       # every phalanx of both hands
    assert {m for _, pair, _ in fp for m in pair} <= {j for ch in FINGERS.values() for j in ch}


# ---- a minimal binary FBX writer (7.4: 32-bit offsets)

def _prop(v) -> bytes:
    if isinstance(v, str):
        b = v.encode()
        return b"S" + struct.pack("<I", len(b)) + b
    if isinstance(v, bool):
        return b"C" + struct.pack("<?", v)
    if isinstance(v, int):
        return b"L" + struct.pack("<q", v)
    if isinstance(v, float):
        return b"D" + struct.pack("<d", v)
    a = np.asarray(v)
    code, dt = {"f": (b"f", "<f4"), "i": (b"l", "<i8")}[a.dtype.kind]
    data = a.astype(dt).tobytes()
    return code + struct.pack("<III", len(a), 0, len(data)) + data


def _node(name: str, props: list, children: list, at: int) -> bytes:
    pb = b"".join(_prop(p) for p in props)
    head = 13 + len(name) + len(pb)
    body = b""
    for c in children:
        body += _node(*c, at + head + len(body))
    if children:
        body += b"\0" * 13
    return struct.pack("<III", at + head + len(body), len(props), len(pb)) + bytes([len(name)]) + name.encode() + pb + body


def write_fbx(path: Path, nodes: list):
    out = b"Kaydara FBX Binary  \x00\x1a\x00" + struct.pack("<I", 7400)
    for n in nodes:
        out += _node(*n, len(out))
    path.write_bytes(out + b"\0" * 13)


def P(name: str, *v):
    return ("P", [name, "", "", "A", *v], [])


def test_fbx_transforms_and_curves(tmp_path):
    """Two bones in centimetres: the child's PreRotation and an animated Euler XYZ rotation compose as FBX's
    T * Rpre * R; the curves are sampled linearly and the units become metres."""
    tick = fbx.TICKS
    keys = np.array([0, tick // 30, 2 * tick // 30])
    nodes = [
        ("GlobalSettings", [], [("Properties70", [], [P("UpAxis", 1), P("UnitScaleFactor", 1.0), P("TimeMode", 6)])]),
        ("Objects", [], [
            ("Model", [1, "mixamorig:Hips\x00\x01Model", "LimbNode"], [("Properties70", [], [
                P("Lcl Translation", 0.0, 100.0, 0.0)])]),
            ("Model", [2, "mixamorig:Spine\x00\x01Model", "LimbNode"], [("Properties70", [], [
                P("PreRotation", 0.0, 0.0, 90.0), P("Lcl Translation", 0.0, 10.0, 0.0)])]),
            ("AnimationCurveNode", [3, "R", ""], []),
            ("AnimationCurve", [4, "", ""], [("KeyTime", [keys], []), ("KeyValueFloat", [np.float32([0, 30, 60])], [])]),
        ]),
        ("Connections", [], [("C", ["OO", 2, 1], []), ("C", ["OP", 3, 2, "Lcl Rotation"], []),
                             ("C", ["OP", 4, 3, "d|X"], [])]),
    ]
    f = tmp_path / "two.fbx"
    write_fbx(f, nodes)
    sc = fbx.load(f)
    assert sc.fps == 30.0 and abs(sc.stop - 2 / 30) < 1e-9
    names, M = mixamo.world(sc, np.array([0.0, 0.5 / 30]))
    assert names == ["Hips", "Spine"]
    np.testing.assert_allclose(M[0, 1, :3, 3], [0, 1.1, 0], atol=1e-9)    # 100 + 10 cm, in metres
    want = Rotation.from_euler("z", 90, degrees=True) * Rotation.from_euler("x", 15, degrees=True)   # halfway: 15 deg
    np.testing.assert_allclose(M[1, 1, :3, :3], want.as_matrix(), atol=1e-6)
