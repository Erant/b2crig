"""Pose a subject's MHR body in the splat's world frame.

Drives the TorchScript `mhr_model.pt` directly with the `model_params` row that
b2crunner's `tools/export_mhr_subject` exported (docs/tools.md there has the layout). A
motion edits the global and body slots; the hand slots keep the subject's
canonical hands, and the 68 scales never change.

World frame = world_from_raw.scale * (raw * FLIP) @ R.T + t, where raw is what
`mhr_forward` returns (MHR centimetres / 100). This is `pipeline/ply_meta.py`'s
frame note.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

N_BODY = 130
TRANS, ROT, BODY, SCALES = slice(0, 3), slice(3, 6), slice(6, 6 + N_BODY), slice(136, 204)


@dataclass
class Posed:
    verts: torch.Tensor    # [B, V, 3] world
    joints: torch.Tensor   # [B, J, 3] world
    rots: torch.Tensor     # [B, J, 3, 3] world-frame joint orientations


class MHRBody:
    def __init__(self, npz: str | Path, device: str = "cuda"):
        d = np.load(npz)
        self.npz_path = Path(npz)
        self.device = device
        self.mhr = torch.jit.load(str(d["mhr_model"]), map_location=device).eval()
        self.model_params0 = torch.as_tensor(d["model_params"], device=device)[None]   # [1, 204]
        self.shape = torch.as_tensor(d["shape_params"], device=device)[None]
        self.expr = torch.as_tensor(d["expr_params"], device=device)[None]
        self.hand_idx = torch.as_tensor(d["hand_idx"], device=device)
        self.faces = d["faces"]
        self.parents = d["joint_parents"]
        self.verts_canon = d["verts_world"]
        self.joints_canon = d["joints_world"]
        s, R, t = float(d["wfr_scale"]), d["wfr_rotation"], d["wfr_translation"]
        flip = d["flip"]
        # world = raw @ A.T + t with A = s * R @ diag(flip)
        self._A = torch.as_tensor(s * R @ np.diag(flip), dtype=torch.float32, device=device)
        self._Rw = torch.as_tensor(R @ np.diag(flip), dtype=torch.float32, device=device)
        self._t = torch.as_tensor(t, dtype=torch.float32, device=device)
        self.skin = (d["skin_vertex"], d["skin_joint"], d["skin_weight"])

    @property
    def body0(self) -> torch.Tensor:
        """Canonical body params [1, 130] (hand slots hold the canonical hands)."""
        return self.model_params0[:, BODY]

    def model_params(self, body: torch.Tensor | None = None, global_rot: torch.Tensor | None = None,
                     global_trans: torch.Tensor | None = None) -> torch.Tensor:
        """Batched [B, 204] rows. `body` [B, 130] replaces the body slots except the hand slots;
        `global_rot` [B, 3] (MHR xyz Euler, raw frame) and `global_trans` [B, 3] (metres, raw frame)."""
        B = max(x.shape[0] for x in (body, global_rot, global_trans) if x is not None) \
            if any(x is not None for x in (body, global_rot, global_trans)) else 1
        mp = self.model_params0.expand(B, -1).clone()
        if body is not None:
            mp[:, BODY] = body
            mp[:, self.hand_idx] = self.model_params0[:, self.hand_idx]
        if global_rot is not None:
            mp[:, ROT] = global_rot
        if global_trans is not None:
            mp[:, TRANS] = global_trans * 10
        return mp

    def forward(self, model_params: torch.Tensor, expr: torch.Tensor | None = None) -> Posed:
        """`expr` [B, 72]: MHR expression coefficients (default: the subject's fitted ones)."""
        B = model_params.shape[0]
        ex = self.expr.expand(B, -1) if expr is None else expr
        with torch.no_grad():
            verts, skel = self.mhr(self.shape.expand(B, -1), model_params, ex)
        coords, quats = skel[..., :3] / 100, skel[..., 3:7]
        verts = verts / 100
        return Posed(verts=verts @ self._A.T + self._t, joints=coords @ self._A.T + self._t,
                     rots=self._Rw @ quat_xyzw_to_mat(quats))

    def pose(self, body=None, global_rot=None, global_trans=None, expr=None) -> Posed:
        return self.forward(self.model_params(body, global_rot, global_trans), expr)


def quat_xyzw_to_mat(q: torch.Tensor) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True)
    x, y, z, w = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], -1).reshape(*q.shape[:-1], 3, 3)
