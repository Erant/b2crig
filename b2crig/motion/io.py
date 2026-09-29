"""Motion files (npz) and posing them.

Keys: body_params [T, 130], expr [T, 72], fps; optional global_trans [T, 3] (MHR raw frame, MHRBody.pose), and an
optional rigid root motion applied after the MHR forward: root_R [T, 3, 3], root_t [T, 3] (world) about the pivot
root_c [3] (world): x' = root_R (x - root_c) + root_c + root_t. The IK walk (motion/walk.py) uses global_trans;
retargeted video motions (tools/retarget_video.py) use root_R/root_t, which carry turning and leaning. Optional
`hands` (bool): body_params' finger slots are animated (retargeted fingers); otherwise posing keeps the subject's
canonical hands whatever those slots hold.
"""
from __future__ import annotations

import numpy as np
import torch

from ..rig.mhr import MHRBody, Posed


class Motion:
    def __init__(self, d, sel=None):
        g = (lambda k: np.asarray(d[k])[sel] if sel is not None else np.asarray(d[k]))
        self.body_params = g("body_params")
        self.expr = g("expr")
        self.fps = float(d["fps"])
        files = d.files if hasattr(d, "files") else list(d.keys())
        self.global_trans = g("global_trans") if "global_trans" in files else None
        self.root_R = g("root_R") if "root_R" in files else None
        self.root_t = g("root_t") if "root_t" in files else None
        self.root_c = np.asarray(d["root_c"]) if "root_c" in files else None
        self.hands = bool(d["hands"]) if "hands" in files else False

    def __len__(self):
        return len(self.body_params)

    def save_fields(self) -> dict:
        out = {"body_params": self.body_params, "expr": self.expr, "fps": self.fps}
        for k in ("global_trans", "root_R", "root_t", "root_c"):
            if getattr(self, k) is not None:
                out[k] = getattr(self, k)
        if self.hands:
            out["hands"] = True
        return out


def load(path, sel=None) -> Motion:
    return Motion(np.load(path), sel)


def pose(body: MHRBody, m: Motion, i0: int, i1: int) -> Posed:
    dev = body.device
    P = body.pose(torch.as_tensor(m.body_params[i0:i1], device=dev), expr=torch.as_tensor(m.expr[i0:i1], device=dev),
                  global_trans=None if m.global_trans is None else torch.as_tensor(m.global_trans[i0:i1], device=dev),
                  hands=m.hands)
    if m.root_R is None:
        return P
    R = torch.as_tensor(m.root_R[i0:i1], dtype=torch.float32, device=dev)
    t = torch.as_tensor(m.root_t[i0:i1], dtype=torch.float32, device=dev)
    c = torch.as_tensor(m.root_c, dtype=torch.float32, device=dev)
    tr = lambda x: ((x - c) @ R.transpose(-1, -2)) + c + t[:, None]
    return Posed(verts=tr(P.verts), joints=tr(P.joints), rots=R[:, None] @ P.rots)
