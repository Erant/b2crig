"""The MHR70 keypoint regressor: [70, 18566] weights over MHR's mesh vertices and joints (each row sums to 1, ~8
non-zeros) that give SAM-3D-Body's 70 body / hand / foot keypoints (sam_3d_body/metadata/mhr70.py names them), e.g.
for body2colmap's skeleton drawings.

It is Meta's model data, the first 70 of the 308 rows of `head_pose.keypoint_mapping` in SAM-3D-Body's checkpoint
(SAM License), so it is not kept in the repo: `mapping()` derives it from the checkpoint once, checks it against
SHA256, and caches it under $B2CRIG_CACHE (default ~/.cache/b2crig). The checkpoint is found at $SAM3DBODY_CKPT, else
in the Hugging Face cache, else downloaded with huggingface_hub if that is installed (the repo is gated: accept the
licence on huggingface.co and `hf auth login` first).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

REPO = "facebook/sam-3d-body-dinov3"
CKPT = "model.ckpt"
KEY = "head_pose.keypoint_mapping"
SHAPE = (70, 18566)
SHA256 = "5b2972769a1e93410189318bba74546484c684dccd8b03acf8e1315fee334893"   # of the float32 little-endian array bytes


def cache_path() -> Path:
    return Path(os.environ.get("B2CRIG_CACHE", Path.home() / ".cache" / "b2crig")) / "mhr70_mapping.npy"


def _digest(m: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(m, "<f4").tobytes()).hexdigest()


def checkpoint() -> Path:
    """SAM-3D-Body's model.ckpt: $SAM3DBODY_CKPT, the Hugging Face cache, or a download."""
    if os.environ.get("SAM3DBODY_CKPT"):
        return Path(os.environ["SAM3DBODY_CKPT"])
    hub = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
    found = sorted((hub / f"models--{REPO.replace('/', '--')}" / "snapshots").glob(f"*/{CKPT}"))
    if found:
        return found[-1]
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise FileNotFoundError(f"SAM-3D-Body's {CKPT} is not in the Hugging Face cache ({hub}) and huggingface_hub is "
                                f"not installed: `hf download {REPO} {CKPT}` (after accepting the licence), or set "
                                f"SAM3DBODY_CKPT") from None
    return Path(hf_hub_download(REPO, CKPT))


def derive(ckpt: Path) -> np.ndarray:
    """The regressor from the checkpoint (memory-mapped: only the one tensor is read)."""
    import torch
    sd = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=False)
    sd = sd.get("state_dict", sd)
    m = sd[KEY][:SHAPE[0]].float().numpy().astype(np.float32)
    if m.shape != SHAPE or _digest(m) != SHA256:
        raise ValueError(f"{ckpt}: {KEY} is not the MHR70 regressor this code expects (shape {m.shape}); "
                         f"a different SAM-3D-Body release?")
    return m


def mapping() -> np.ndarray:
    """[70, 18566] float32, cached after the first call."""
    path = cache_path()
    if path.exists():
        m = np.load(path)
        if m.shape == SHAPE and _digest(m) == SHA256:
            return m
    m = derive(checkpoint())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, m)
    tmp.replace(path)
    return m
