"""b2crig's side of the b2c glTF format (~/Projects/b2cgltf, SPEC.md): compute the rig and clip arrays with b2crig's
own posing, and hand them to the format package (b2cgltf.b2crig), which owns the layout and the rules.

    rig(subject_glb, work/<s>, out)               SPEC 5: cage, b2ctrain's binding, preview skin, scene 1
    clip(rigged_glb, work/<s>, motion, name, out)  SPEC 6: one clip file (skeletal animation + residual + motion)

Everything here is in the b2crunner frame, which is b2crig's world frame (SPEC 2).
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

import b2cgltf
from b2cgltf import read
from b2cgltf.b2crig import clip as GC
from b2cgltf.b2crig import rig as GR

from .. import b2ctrain as B

TOOL = "b2crig export/gltf.py"


def _commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def write_ply(path: Path, fields: dict) -> None:
    d = np.zeros(len(fields["x"]), [(k, "f4") for k in fields])
    for k, v in fields.items():
        d[k] = v
    PlyData([PlyElement.describe(d, "vertex")], text=False,
            comments=["Exported from Brush", "Vertical axis: y"]).write(str(path))


def cage_layers(doc: b2cgltf.Document, body, lay) -> tuple[list[GR.Layer], list]:
    """b2crig's cage (layered.cage_layers) as b2cgltf layers. Layer 0 is the subject's body (its faces: the
    expression-stable ones); the others take the file body's weights at the body vertex they copy, as
    layered.pose_layers skins them."""
    from ..rig import layered
    cl = layered.cage_layers(body, lay)
    bprim = GR.body_primitive(doc)
    bidx, bw = read.joints_weights(doc, bprim)
    nb = doc.json["accessors"][bprim["attributes"]["POSITION"]]["count"]
    if nb != len(body.verts_canon):
        raise ValueError(f"the subject's body has {nb} vertices, b2crig's MHR body {len(body.verts_canon)}")
    out = [GR.Layer("body", list(cl[0].classes), np.asarray(cl[0].faces))]
    for L, CL in zip(lay, cl[1:]):
        n = len(L["V"])
        if "rigid_joint" in L:
            ji = np.full((n, 1), int(L["rigid_joint"])); jw = np.ones((n, 1))
        elif "skin_w" in L:
            ji, jw = np.asarray(L["skin_idx"]), np.asarray(L["skin_w"])
        else:
            ji, jw = bidx[L["bi"]], bw[L["bi"]]
        out.append(GR.Layer(L["name"], list(CL.classes), np.asarray(CL.faces), np.asarray(L["V"]), ji, jw,
                            np.asarray(L["bi"]) if "bi" in L and "rigid_joint" not in L and "skin_w" not in L else None))
    return out, cl


def b2ctrain_binding(splat_fields: dict, layers: list[GR.Layer], body_verts: np.ndarray, min_conf: float) -> GR.Binding:
    """b2ctrain's own binding (`render --export-binding`) of the splat to the canonical cage."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        write_ply(td / "s.ply", splat_fields)
        cl = [B.CageLayer(L.name, body_verts if i == 0 else L.verts, L.faces, L.classes) for i, L in enumerate(layers)]
        V0 = np.concatenate([c.verts for c in cl])
        B.write_cage(td / "c.b2ccage", cl, V0[None], ["canonical"])
        subprocess.run([str(B.B2CTRAIN), "render", "--splat", str(td / "s.ply"), "--cage", str(td / "c.b2ccage"),
                        "--cage-min-conf", str(min_conf), "--export-binding", str(td / "b.bin")], check=True, capture_output=True)
        raw = (td / "b.bin").read_bytes()
    n = int(np.frombuffer(raw, np.int32, 1, 8)[0])
    assert raw[:8] == b"B2CBIND1" and n == len(splat_fields["x"])
    o = 12
    face = np.frombuffer(raw, np.int32, n, o).copy(); o += 4 * n
    bary = np.frombuffer(raw, np.float32, 2 * n, o).reshape(n, 2).copy(); o += 8 * n
    off = np.frombuffer(raw, np.float32, 3 * n, o).reshape(n, 3).copy()
    return GR.Binding(face, bary, off)


def rig(subject_glb: Path, S: Path, out: Path, *, splat_ply: Path | None = None, min_conf: float = 0.5,
        render: dict | None = None) -> dict:
    """Enhance a subject file with b2crig's rig (SPEC 5). `splat_ply`: a splat b2crig retrained (then it supersedes
    b2crunner's); default = rig the delivered splat."""
    from ..rig import layered
    from ..rig.mhr import MHRBody
    doc = b2cgltf.load(subject_glb)
    w = doc.writer("b2crig", tool=TOOL, commit=_commit())
    body = MHRBody(S / "mhr.npz"); lay = layered.load_layers(S)
    layers, _ = cage_layers(doc, body, lay)
    bverts = read_body(doc)
    if splat_ply is not None:
        vx = PlyData.read(str(splat_ply))["vertex"]
        keep = [k for k in vx.data.dtype.names if not k.startswith(("ev_",))]
        fields = {k: np.asarray(vx[k], np.float32) for k in keep}
        extra = {}
        for k, name in (("open_r", "_B2CRIG_OPEN_R"), ("open_g", "_B2CRIG_OPEN_G"), ("open_b", "_B2CRIG_OPEN_B"),
                        ("open_dopacity", "_B2CRIG_OPEN_DOPACITY"), ("cage_fill", "_B2CRIG_CAGE_FILL"),
                        ("cage_gate_a", "_B2CRIG_GATE_A"), ("cage_gate_b", "_B2CRIG_GATE_B")):
            if k in fields:
                extra[name] = fields.pop(k)
        fields.pop("cage_gate_s", None)   # carried by b2ctrain but read by nothing (b2ctrain docs/b2crig-boundary.md)
    else:
        fields = read.to_trainer_ply_fields(read.splat(doc, doc.node_index("b2c_splat")))
        extra = {}
    bind = b2ctrain_binding(fields, layers, bverts, min_conf)
    rnd = {"maxGrowth": B.CAGE_MAX_GROWTH, "fadeStart": 0.0, "fadeEnd": 0.0, "fillStart": 0.0, "fillEnd": 0.0,
           "minConf": min_conf, **(render or {})}
    res = GR.enhance(doc, w, layers, bind, render=rnd, retrained_splat=fields if splat_ply is not None else None,
                     splat_extra=extra or None)
    w.save(out)
    return {"path": str(out), "splats": len(bind.face), "unbound": int((bind.face < 0).sum()),
            "cage_vertices": sum(m["vertexCount"] for m in res.layers), "layers": [m["name"] for m in res.layers]}


def read_body(doc: b2cgltf.Document) -> np.ndarray:
    return doc.accessor(GR.body_primitive(doc)["attributes"]["POSITION"]).astype(np.float64)


def local_trs(rots: np.ndarray, joints: np.ndarray, parents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World (b2crunner-frame) joint frames [T, J, 3, 3], [T, J, 3] -> local TRS relative to the parent joint or the
    skeleton root: float32 quaternions (x y z w) and translations."""
    R = rots.copy(); t = joints.copy()
    for j, p in enumerate(parents):
        if p >= 0:
            Rp = rots[:, p]
            R[:, j] = np.swapaxes(Rp, -1, -2) @ rots[:, j]
            t[:, j] = (np.swapaxes(Rp, -1, -2) @ (joints[:, j] - joints[:, p])[..., None])[..., 0]
    return read.mat_to_quat(R).astype(np.float32), t.astype(np.float32)


def clip(rigged_glb: Path, S: Path, motion_npz: Path, name: str, out: Path, *, residual: bool = True,
         subject_uri: str | None = None) -> dict:
    """One clip file (SPEC 6) for a motion (motion/io.py npz), posed by b2crig (layered.pose_motion)."""
    from ..motion import io as MI
    from ..rig import layered
    from ..rig.mhr import MHRBody
    doc = b2cgltf.load(rigged_glb)
    sk = read.skeleton(doc)
    body = MHRBody(S / "mhr.npz"); lay = layered.load_layers(S)
    m = MI.load(motion_npz)
    T = len(m)
    rots, joints = [], []
    for i in range(0, T, 16):
        P = MI.pose(body, m, i, min(i + 16, T))
        rots.append(P.rots.cpu().numpy()); joints.append(P.joints.cpu().numpy())
    q, t = local_trs(np.concatenate(rots).astype(np.float64), np.concatenate(joints).astype(np.float64), sk.parents)
    posed = layered.pose_motion(body, m, lay).astype(np.float64)
    cg = GR.cage(doc)
    if posed.shape[1] != len(cg.verts):
        raise ValueError(f"b2crig posed {posed.shape[1]} cage vertices, the file's cage has {len(cg.verts)}")
    plain = GC.lbs(doc, q.astype(np.float64), t.astype(np.float64), cg)
    res = posed - plain
    motion = {"body_params": m.body_params, "expr": m.expr}
    for k in ("global_trans", "root_R", "root_t", "root_c"):
        if getattr(m, k) is not None:
            motion[k] = getattr(m, k)
    rep = GC.write(out, doc, name, m.fps, q, t, residual=res if residual else None, motion=motion,
                   subject_uri=subject_uri, tool=TOOL, commit=_commit())
    e = np.linalg.norm(res, axis=-1) * 1e3
    rep.update({"lbs_only_mm_p99": float(np.percentile(e, 99)), "lbs_only_mm_max": float(e.max())})
    return rep
