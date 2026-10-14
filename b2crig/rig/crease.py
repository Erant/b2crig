"""The armpit-crease heuristic: which splats are crease wall.

When an arm rises, the two surfaces of the armpit crease (arm side, torso side) rotate apart. Splats that sit between
them were captured in shadow and are not skin; stretched by the posed cage they bloom into coloured wisps under the
arm. A splat is crease wall when it lies within `reach` of an arm vertex AND of a torso vertex of the canonical body
and the two vertex normals face each other. The value is the larger of the two distances (how deep in the crease it
sits); culling splats below a radius r drops those with both surfaces within r. Uses no segmentation labels and no
pose: only the canonical body and its skin. b2cviewer ports this (web/crease.js, same constants); keep them in step.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

REACH = 0.10       # m: beyond this from either surface a splat is not crease (the viewer's largest radius)
FACING = -0.2      # the two surface normals' dot product must be below this
ARM = ("l_uparm", "r_uparm", "l_lowarm", "r_lowarm")   # b2crunner skeleton joint-name prefixes of each side's surface
TORSO = ("c_spine", "root")
NONE = np.float32(1e9)   # the value of a splat that is not crease


def vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n = np.zeros_like(V)
    for k in range(3):
        np.add.at(n, F[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)


def crease_values(xyz: np.ndarray, body_V: np.ndarray, body_F: np.ndarray, joint_names: np.ndarray,
                  reach: float = REACH) -> np.ndarray:
    """[n] per splat: max(distance to the nearest arm vertex, to the nearest torso vertex) where both are within
    `reach` and the surfaces face each other, else NONE. `joint_names`: the dominant skin joint's name per body vertex."""
    arm = np.flatnonzero([str(j).startswith(ARM) for j in joint_names])
    tor = np.flatnonzero([str(j).startswith(TORSO) for j in joint_names])
    n = vertex_normals(np.asarray(body_V, np.float64), np.asarray(body_F, np.int64))
    da, ja = cKDTree(body_V[arm]).query(xyz); dt, jt = cKDTree(body_V[tor]).query(xyz)
    facing = np.einsum("ij,ij->i", n[arm[ja]], n[tor[jt]])
    ok = (da < reach) & (dt < reach) & (facing < FACING)
    return np.where(ok, np.maximum(da, dt), NONE).astype(np.float32)


def subject_crease(doc) -> np.ndarray:
    """crease_values of a rigged subject file's current splat (b2cgltf Document)."""
    from b2cgltf import read
    from b2cgltf.b2crig import rig as GR
    cg = GR.cage(doc)
    body = next(m for m in cg.layers if m["name"] == "body")
    V = cg.verts[body["vertexOffset"]:body["vertexOffset"] + body["vertexCount"]].astype(np.float64)
    F = cg.faces[body["faceOffset"]:body["faceOffset"] + body["faceCount"]] - body["vertexOffset"]
    idx, w = read.joints_weights(doc, GR.body_primitive(doc))
    names = np.array(read.skeleton(doc).names)[idx[np.arange(len(idx)), w.argmax(1)]]
    sp = read.splat(doc, read.current_splat(doc))
    return crease_values(np.asarray(sp["pos"], np.float64), V, F, names)
