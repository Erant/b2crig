"""A minimal binary FBX reader: the node tree, the objects and their connections, and a skeleton's animation.

Enough for Mixamo downloads (motion/mixamo.py; no Blender or FBX SDK needed): `load(path)` returns the bones
(Model nodes of type LimbNode / Null / Root, with their static transform properties and parent), the animation
curves on their Lcl Translation / Rotation / Scaling, and the global settings (up axis, unit). `local_matrices`
evaluates a bone's local matrices at given times with FBX's transform order (pivots, Pre/Post rotation, Euler order);
curves are sampled linearly, which is exact for Mixamo's per-frame keys. Binary FBX 7.x only.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

TICKS = 46186158000   # FBX time units per second
# FBX's ETimeMode -> frames per second (14 is custom: GlobalSettings' CustomFrameRate)
TIME_MODES = {1: 120.0, 2: 100.0, 3: 60.0, 4: 50.0, 5: 48.0, 6: 30.0, 7: 30.0, 8: 29.97, 9: 29.97, 10: 25.0, 11: 24.0,
              12: 1000.0, 13: 23.976, 15: 96.0, 16: 72.0, 17: 59.94}


@dataclass
class Node:
    name: str
    props: list
    children: list = field(default_factory=list)

    def find(self, name: str) -> "Node | None":
        return next((c for c in self.children if c.name == name), None)

    def all(self, name: str) -> list["Node"]:
        return [c for c in self.children if c.name == name]


def _prop(b: bytes, o: int):
    t = chr(b[o]); o += 1
    if t in "YCIFDL":
        fmt = {"Y": "<h", "C": "<?", "I": "<i", "F": "<f", "D": "<d", "L": "<q"}[t]
        return struct.unpack_from(fmt, b, o)[0], o + struct.calcsize(fmt)
    if t in "fdlib":
        n, enc, clen = struct.unpack_from("<III", b, o); o += 12
        raw = b[o:o + clen]; o += clen
        if enc == 1:
            raw = zlib.decompress(raw)
        dt = {"f": "<f4", "d": "<f8", "l": "<i8", "i": "<i4", "b": "?"}[t]
        return np.frombuffer(raw, dt, n), o
    if t in "SR":
        n = struct.unpack_from("<I", b, o)[0]; o += 4
        raw = b[o:o + n]
        return (raw.decode("utf-8", "replace") if t == "S" else raw), o + n
    raise ValueError(f"FBX property type {t!r} at {o - 1}")


def _node(b: bytes, o: int, wide: bool) -> tuple[Node | None, int]:
    if wide:
        end, nprops, _ = struct.unpack_from("<QQQ", b, o); o += 24
    else:
        end, nprops, _ = struct.unpack_from("<III", b, o); o += 12
    nlen = b[o]; o += 1
    if end == 0:
        return None, o
    name = b[o:o + nlen].decode("ascii", "replace"); o += nlen
    props = []
    for _ in range(nprops):
        v, o = _prop(b, o); props.append(v)
    n = Node(name, props)
    while o < end:
        c, o = _node(b, o, wide)
        if c is None:
            break
        n.children.append(c)
    return n, end


def parse(path: str | Path) -> Node:
    b = Path(path).read_bytes()
    if not b.startswith(b"Kaydara FBX Binary"):
        raise ValueError(f"{path}: not a binary FBX (ASCII FBX is not supported)")
    ver = struct.unpack_from("<I", b, 23)[0]
    wide = ver >= 7500
    root, o = Node("", []), 27
    while o < len(b):
        n, o = _node(b, o, wide)
        if n is None:
            break
        root.children.append(n)
    return root


def props70(n: Node) -> dict:
    """A node's Properties70 as {name: value(s)}."""
    p = n.find("Properties70")
    out = {}
    for c in (p.children if p else []):
        v = c.props[4:]
        out[c.props[0]] = v[0] if len(v) == 1 else tuple(v)
    return out


@dataclass
class Bone:
    id: int
    name: str
    kind: str
    p: dict                       # Properties70
    parent: int | None = None     # object id
    curves: dict = field(default_factory=dict)   # "T" / "R" / "S" -> [3] of (times s, values) or None


@dataclass
class Scene:
    bones: dict                   # id -> Bone
    up_axis: int
    unit_cm: float                # UnitScaleFactor: centimetres per unit
    fps: float | None             # the file's frame rate (TimeMode), if it says
    stop: float                   # the animation's last key, s


def load(path: str | Path) -> Scene:
    root = parse(path)
    gs = props70(root.find("GlobalSettings"))
    objs = root.find("Objects")
    bones, cnodes, curves = {}, {}, {}
    for n in objs.children:
        oid = n.props[0]
        if n.name == "Model" and n.props[2] in ("LimbNode", "Null", "Root"):
            bones[oid] = Bone(oid, n.props[1].split("\x00")[0], n.props[2], props70(n))
        elif n.name == "AnimationCurveNode":
            cnodes[oid] = {"curves": [None, None, None], "default": props70(n)}
        elif n.name == "AnimationCurve":
            t = n.find("KeyTime").props[0].astype(np.float64) / TICKS
            v = n.find("KeyValueFloat").props[0].astype(np.float64)
            curves[oid] = (t, v)
    links = {"T": "Lcl Translation", "R": "Lcl Rotation", "S": "Lcl Scaling"}
    for c in root.find("Connections").all("C"):
        kind, child, parent = c.props[:3]
        prop = c.props[3] if len(c.props) > 3 else None
        if child in bones and parent in bones and kind == "OO":
            bones[child].parent = parent
        elif child in curves and parent in cnodes and prop in ("d|X", "d|Y", "d|Z"):
            cnodes[parent]["curves"]["XYZ".index(prop[-1])] = curves[child]
        elif child in cnodes and parent in bones:
            key = next((k for k, v in links.items() if v == prop), None)
            if key:
                bones[parent].curves[key] = cnodes[child]["curves"]
    mode = gs.get("TimeMode")
    fps_mode = float(gs.get("CustomFrameRate", 0)) or None if mode == 14 else TIME_MODES.get(mode)
    stop = max((c[0][-1] for b in bones.values() for cs in b.curves.values() for c in cs if c is not None), default=0.0)
    return Scene(bones, int(gs.get("UpAxis", 1)), float(gs.get("UnitScaleFactor", 1.0)), fps_mode, stop)


ORDERS = ["xyz", "xzy", "yzx", "yxz", "zxy", "zyx"]   # FBX EulerOrder 0..5 (eEulerXYZ: R = Rz Ry Rx)


def euler(deg: np.ndarray, order: int = 0) -> np.ndarray:
    """[..., 3] degrees (x, y, z) -> [..., 3, 3]; FBX's eEulerXYZ applies x first."""
    o = ORDERS[order]
    a = np.take(np.radians(deg), ["xyz".index(c) for c in o], axis=-1)
    return Rotation.from_euler(o, a.reshape(-1, 3)).as_matrix().reshape(*deg.shape[:-1], 3, 3)


def _v(p: dict, k: str, d=(0.0, 0.0, 0.0)) -> np.ndarray:
    return np.array(p.get(k, d), np.float64)


def _channel(b: Bone, key: str, prop: str, t: np.ndarray, d) -> np.ndarray:
    base = _v(b.p, prop, d)
    out = np.repeat(base[None], len(t), 0)
    for i, c in enumerate(b.curves.get(key, [None] * 3)):
        if c is not None:
            out[:, i] = np.interp(t, c[0], c[1])
    return out


def local_matrices(b: Bone, t: np.ndarray | None) -> np.ndarray:
    """[T, 4, 4] local transforms at times t (None: the static properties, one frame).
    FBX: T * Roff * Rp * Rpre * R * Rpost^-1 * Rp^-1 * Soff * Sp * S * Sp^-1."""
    tt = np.zeros(1) if t is None else t
    if t is None:
        T_, R_, S_ = (np.array([_v(b.p, k, d)]) for k, d in (
            ("Lcl Translation", (0, 0, 0)), ("Lcl Rotation", (0, 0, 0)), ("Lcl Scaling", (1, 1, 1))))
    else:
        T_ = _channel(b, "T", "Lcl Translation", tt, (0, 0, 0))
        R_ = _channel(b, "R", "Lcl Rotation", tt, (0, 0, 0))
        S_ = _channel(b, "S", "Lcl Scaling", tt, (1, 1, 1))
    order = int(b.p.get("RotationOrder", 0))
    pre, post = euler(_v(b.p, "PreRotation")), euler(_v(b.p, "PostRotation"))
    roff, rp = _v(b.p, "RotationOffset"), _v(b.p, "RotationPivot")
    soff, sp = _v(b.p, "ScalingOffset"), _v(b.p, "ScalingPivot")
    R = pre @ euler(R_, order) @ post.T
    M = np.zeros((len(tt), 4, 4)); M[:, 3, 3] = 1
    M[:, :3, :3] = R * S_[:, None, :]
    # translation: T + Roff + Rp + R (Soff + Sp - Rp - S Sp)  (the pivots and offsets are usually zero)
    M[:, :3, 3] = T_ + roff + rp + np.einsum("tij,tj->ti", R, soff + sp - rp - S_ * sp)
    return M
