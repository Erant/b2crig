"""b2crig's side of the b2c glTF format (~/Projects/b2cgltf, SPEC.md): compute the rig and clip arrays with b2crig's
own posing, and hand them to the format package (b2cgltf.b2crig), which owns the layout and the rules.

    rig(subject_glb, work/<s>, out)               SPEC 5: cage, b2ctrain's binding, preview skin, scene 1
    clip(rigged_glb, work/<s>, motion, name, out)  SPEC 6: one clip file (skeletal animation + residual + motion)
    levels(rigged_glb, work/<s>, out)              SPEC 4.7 / 5.3: LOD levels of the rigged splat, bound to its cage

Everything here is in the b2crunner frame, which is b2crig's world frame (SPEC 2).
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

import b2cgltf
from b2cgltf import lod, read
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


LAYER_REACH = 0.15   # m: how far from its layer a splat can be and still bind to it (offset layers hug the body, max ~12 cm)


def _gate_layer_labels(fields: dict, layers: list[GR.Layer], min_conf: float) -> dict:
    """b2ctrain binds a splat to the nearest face of the layer its segmentation class belongs to, however far that is.
    A segmenter that calls a one-piece suit's trousers "upper clothing" binds the legs to the torso layer's nearest
    face, a hand. Splats farther than LAYER_REACH from their layer take the label of their nearest body-class splat."""
    from scipy.spatial import cKDTree
    if "seg_label" not in fields:
        return fields
    lab = np.asarray(fields["seg_label"]).copy()
    pos = np.stack([fields["x"], fields["y"], fields["z"]], 1)
    conf = np.asarray(fields["seg_conf"])
    body = np.isin(lab, layers[0].classes) & (conf >= min_conf)
    stray = np.zeros(len(lab), bool)
    for L in layers[1:]:
        mine = np.isin(lab, L.classes) & (conf >= min_conf)
        if not mine.any() or not len(L.faces):
            continue
        d, _ = cKDTree(L.verts[np.unique(L.faces)]).query(pos[mine])
        stray[np.flatnonzero(mine)[d > LAYER_REACH]] = True
    if not stray.any():
        return fields
    _, nn = cKDTree(pos[body]).query(pos[stray])
    lab[stray] = lab[body][nn]
    print(f"binding: {int(stray.sum())} splats lie > {LAYER_REACH * 100:.0f} cm from their garment layer; bound to the body instead", flush=True)
    return {**fields, "seg_label": lab.astype(np.asarray(fields["seg_label"]).dtype)}


def b2ctrain_binding(splat_fields: dict, layers: list[GR.Layer], body_verts: np.ndarray, min_conf: float) -> GR.Binding:
    """b2ctrain's own binding (`render --export-binding`) of the splat to the canonical cage."""
    splat_fields = _gate_layer_labels(splat_fields, layers, min_conf)
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


LEVELS = ((100000, 960), (50000, 480))   # SPEC 4.7 defaults; every rigged delivery gets them (user, 2026-10-30)
LEVEL_K = 3                              # lod-merge --max-extent-faces (b24be4 gate, 2026-10-30)


def file_cage_layers(doc: b2cgltf.Document) -> tuple[list[GR.Layer], np.ndarray]:
    """The rigged file's own cage as b2ctrain_binding's layers (layer 0's verts are returned separately)."""
    c = GR.cage(doc)
    out = []
    for m in c.layers:
        vo, vc, fo, fc = m["vertexOffset"], m["vertexCount"], m["faceOffset"], m["faceCount"]
        out.append(GR.Layer(m["name"], list(m["classes"]), c.faces[fo:fo + fc] - vo, None if vo == 0 else c.verts[vo:vo + vc]))
    return out, c.verts[:c.layers[0]["vertexCount"]]


def levels(rigged_glb: Path, S: Path, out: Path, *, teacher_ply: Path | None = None, views: Path | None = None,
           levels: tuple = LEVELS, contain: str = "poseset_romh16:1", contain_hands: float = 0.3) -> dict:
    """Append LOD levels of the rigged splat to a rigged subject file (SPEC 4.7, 5.3). Per level: b2ctrain lod-merge
    with the file's cage (K = LEVEL_K, the binding's min conf), then cage_train's pose containment fine-tune distilled
    against the full rigged splat with its own alpha (--teacher --distill-alpha render), then b2ctrain's binding to the
    file's cage. `S` is the subject's work dir (cage_train's: mhr.npz, layers, clips/<contain>), rebuilt for this
    subject; its cage must be the file's. `teacher_ply`: the rigged splat as cage_train wrote it (default: the
    file's current splat). A level the base is not bigger than is skipped."""
    doc = b2cgltf.load(rigged_glb)
    rig = doc.extension(GR.RIG)
    if rig is None:
        raise ValueError(f"{rigged_glb}: not rigged (no B2CRIG_rig)")
    base = rig["splatNode"]
    if read.current_splat(doc) != base:
        raise ValueError(f"{rigged_glb}: the current splat is not b2crig_splat")
    bat = doc.json["meshes"][doc.json["nodes"][base]["mesh"]]["primitives"][0]["attributes"]
    state = [k for k in lod.B2CRIG_STATE if k in bat and not k.startswith("_B2CRIG_SEG_")]
    if state:
        raise NotImplementedError(f"the rigged splat carries {state}: levels would need them too (SPEC 4.7)")
    min_conf = float(rig["render"].get("minConf", 0.5))
    layers, bverts = file_cage_layers(doc)
    # S's cage is what cage_train contains against: it must be the file's
    from ..rig import layered
    from ..rig.mhr import MHRBody
    s_layers, _ = cage_layers(doc, MHRBody(S / "mhr.npz"), layered.load_layers(S))
    sv = np.concatenate([bverts] + [L.verts for L in s_layers[1:]])
    fv = GR.cage(doc).verts
    if sv.shape != fv.shape or np.abs(sv - fv).max() > 1e-3:
        raise ValueError(f"{S}'s cage is not {rigged_glb}'s ({sv.shape} vs {fv.shape} vertices): rebuild the work dir "
                         "from this subject")
    n_base = doc.json["accessors"][bat["POSITION"]]["count"]
    work = S / "lod"; work.mkdir(parents=True, exist_ok=True)
    if teacher_ply is None:
        teacher_ply = work / "teacher.ply"
        write_ply(teacher_ply, read.to_trainer_ply_fields(read.splat(doc, base)))
    vx = PlyData.read(str(teacher_ply))["vertex"]
    tf = {k: np.asarray(vx[k], np.float32) for k in vx.data.dtype.names if not k.startswith(("ev_", "open_", "cage_"))}
    write_ply(work / "merge_in.ply", _gate_layer_labels(tf, layers, min_conf))   # lod-merge takes labels as given
    cage_file = work / "cage.b2ccage"
    cl = [B.CageLayer(L.name, bverts if i == 0 else L.verts, L.faces, L.classes) for i, L in enumerate(layers)]
    B.write_cage(cage_file, cl, np.concatenate([c.verts for c in cl])[None], ["canonical"])
    w = doc.writer("b2crig", tool=TOOL, commit=_commit())
    made, report = [], {"base": n_base, "levels": []}
    for i, (n, h) in enumerate(levels):
        if n >= n_base:
            print(f"levels: the rigged splat has {n_base} splats, not more than {n}: no {n} level", flush=True)
            continue
        merged = work / f"merge_{n}.ply"
        subprocess.run([str(B.B2CTRAIN), "lod-merge", "--splat", str(work / "merge_in.ply"), "--count", str(n),
                        "--cage", str(cage_file), "--max-extent-faces", str(LEVEL_K), "--cage-min-conf", str(min_conf),
                        "--out", str(merged)], check=True)
        tag = f"lod{n}"
        subprocess.run([sys.executable, str(Path(__file__).resolve().parents[2] / "tools" / "cage_train.py"), str(S),
                        "--init", str(merged), "--teacher", str(Path(teacher_ply).resolve()), "--distill",
                        "--distill-alpha", "render", "--contain", contain, "--contain-hands", str(contain_hands),
                        "--tag", tag, *(["--views", str(Path(views).resolve())] if views else [])], check=True)
        lv = PlyData.read(str(S / "train" / tag / "scene.ply"))["vertex"]
        fields = {k: np.asarray(lv[k], np.float32) for k in lv.data.dtype.names if not k.startswith(("ev_", "open_", "cage_"))}
        bind = b2ctrain_binding(fields, layers, bverts, min_conf)
        node = GR.add_level(doc, w, bind, name=f"b2crig_splat_lod{len(made) + 1}", splat_fields=fields)
        made.append({"node": node, "maxScreenHeightPx": h,
                     "method": f"b2ctrain lod-merge K={LEVEL_K} + cage_train --distill --distill-alpha render --contain {contain}"})
        report["levels"].append({"count": len(fields["x"]), "maxScreenHeightPx": h, "unbound": int((bind.face < 0).sum())})
    if made:
        lod.add_lods(w, base, made, coverage=True)
    w.save(out)
    report["path"] = str(out)
    return report


def render_inputs(subject_glb: Path, clip_glb: Path | None, frame: int, out: Path, *, residual: bool = True
                  ) -> tuple[Path, Path | None, dict]:
    """b2ctrain's inputs for one frame of a clip file (None = the rest pose): the current splat as a trainer PLY and a
    one-frame .b2ccage (SPEC 7.2 steps 1-2), written to out/. Returns (ply, cage, the rig's render settings); a subject
    b2crig has not rigged gives no cage (the splat renders as delivered)."""
    doc = b2cgltf.load(subject_glb)
    out.mkdir(parents=True, exist_ok=True)
    write_ply(out / "splat.ply", read.to_trainer_ply_fields(read.splat(doc, read.current_splat(doc))))
    if doc.extension(GR.RIG) is None:
        return out / "splat.ply", None, {}
    cg = GR.cage(doc)
    V = cg.verts
    if clip_glb is not None:
        c = b2cgltf.load(clip_glb)
        GC.check(doc, c)
        q, t, _ = GC.channels(c)
        V = GC.lbs(doc, q[frame:frame + 1], t[frame:frame + 1], cg)[0]
        e = (c.json["animations"][0].get("extensions") or {}).get(GC.RES)
        if residual and e:
            V = V + c.accessor(e["accessor"], normalized=True).reshape(e["frames"], e["vertices"], 3)[frame] * e["scale"]
    layers = [B.CageLayer(m["name"], cg.verts[m["vertexOffset"]:m["vertexOffset"] + m["vertexCount"]],
                          cg.faces[m["faceOffset"]:m["faceOffset"] + m["faceCount"]] - m["vertexOffset"], m["classes"])
              for m in cg.layers]
    B.write_cage(out / "frame.b2ccage", layers, V[None], [f"{frame:04d}"])
    return out / "splat.ply", out / "frame.b2ccage", cg.render


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
