"""A subject straight from b2crunner's subject file (b2cgltf SPEC 4): the MHR body as mhr.npz, and the delivered
splat as a trainer PLY, so a pipeline can rig a run without `tools/export_mhr_subject` (sam3dbody env, PLY header).

    mhr_npz(subject_glb, out)       mhr.npz (b2crunner docs/tools.md) from the body node and its B2C_mhr
    splat_ply(subject_glb, out)     the current splat (SPEC 7.1) with its seg labels
    mhr_model(doc)                  SPEC 7.3: $B2C_MHR_MODEL_DIR, then the Hugging Face cache, sha256-checked
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

import b2cgltf
from b2cgltf import read
from b2cgltf.b2crig import rig as GR


def _hub_caches() -> list[Path]:
    out = []
    if os.environ.get("HF_HUB_CACHE"):
        out.append(Path(os.environ["HF_HUB_CACHE"]))
    if os.environ.get("HF_HOME"):
        out.append(Path(os.environ["HF_HOME"]) / "hub")
    out.append(Path.home() / ".cache" / "huggingface" / "hub")
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def mhr_model(doc: b2cgltf.Document) -> Path:
    """The `mhr_model.pt` B2C_mhr.model names (SPEC 7.3), local files only. Refuses a hash mismatch."""
    spec = doc.extension("B2C_mhr", doc.json["nodes"][doc.node_index("b2c_body")])["model"]
    cands = []
    if os.environ.get("B2C_MHR_MODEL_DIR"):
        cands.append(Path(os.environ["B2C_MHR_MODEL_DIR"]) / Path(spec["file"]).name)
    repo_dir = "models--" + spec["repo"].replace("/", "--")
    for hub in _hub_caches():
        cands += sorted((hub / repo_dir / "snapshots").glob(f"*/{spec['file']}"))
    for p in cands:
        if p.is_file():
            got = _sha256(p.resolve())
            if spec.get("sha256") and got != spec["sha256"]:
                raise ValueError(f"{p}: sha256 {got} is not the subject's model {spec['sha256']}")
            return p.resolve()
    raise FileNotFoundError(f"no {spec['repo']}/{spec['file']} in $B2C_MHR_MODEL_DIR or the Hugging Face cache "
                            f"({', '.join(str(h) for h in _hub_caches())})")


def mhr_npz(subject_glb: str | Path, out: str | Path, *, model: str | Path | None = None) -> Path:
    """mhr.npz from the subject file's body: the bind-pose mesh is the refitted body in the b2crunner frame (W is the
    identity, SPEC 2), the joints are the bind transforms (inverse IBMs), the sparse skin is JOINTS_n/WEIGHTS_n without
    the zero slots, and the model row and frame come from B2C_mhr."""
    doc = b2cgltf.load(subject_glb)
    node = doc.json["nodes"][doc.node_index("b2c_body")]
    e = doc.extension("B2C_mhr", node)
    prim = GR.body_primitive(doc)
    sk = read.skeleton(doc)
    if not np.allclose(sk.W, np.eye(4), atol=1e-6):
        raise ValueError("the subject's skeleton root is not the identity (SPEC 2); mhr.npz is in the b2crunner frame")
    bind = np.linalg.inv(sk.ibm)
    idx, w = read.joints_weights(doc, prim)
    keep = w > 0
    sv = np.nonzero(keep)[0].astype(np.int64)
    pp = e["pose_params"]
    acc = lambda k: doc.accessor(k)
    np.savez(
        out,
        model_params=acc(e["model_params"]).astype(np.float32).reshape(-1),
        shape_params=acc(pp["shape_params"]).astype(np.float32).reshape(-1),
        expr_params=acc(pp["expr_params"]).astype(np.float32).reshape(-1),
        hand_idx=acc(e["hand_idx"]).astype(np.int64).reshape(-1),
        wfr_scale=np.float64(e["world_from_raw"]["scale"]),
        wfr_rotation=acc(e["world_from_raw"]["rotation"]).astype(np.float64).reshape(3, 3),
        wfr_translation=np.asarray(e["world_from_raw"]["translation"], np.float64),
        flip=np.asarray(e["flip"], np.float64),
        verts_world=doc.accessor(prim["attributes"]["POSITION"]).astype(np.float32),
        joints_world=bind[:, :3, 3].astype(np.float32),
        faces=doc.accessor(prim["indices"]).astype(np.int32).reshape(-1, 3),
        joint_parents=sk.parents.astype(np.int32),
        skin_vertex=sv, skin_joint=idx[keep].astype(np.int32), skin_weight=w[keep].astype(np.float32),
        mhr_model=str(model or mhr_model(doc)),
    )
    return Path(out)


def splat_ply(subject_glb: str | Path, out: str | Path) -> Path:
    """The subject's current splat as a trainer PLY (seg_label / seg_conf kept: build_layers and the binding use them)."""
    from .export.gltf import write_ply
    doc = b2cgltf.load(subject_glb)
    fields = read.to_trainer_ply_fields(read.splat(doc))
    if "seg_label" not in fields:
        raise ValueError(f"{subject_glb}: the splat carries no seg labels (b2crunner's splat_labels setting); the cage "
                         "layers are built from them")
    write_ply(Path(out), fields)
    return Path(out)
