"""Shared retargeting: IK to bone directions / orientations / a pelvis path (ik_to_targets), and the foot-contact
clean-up (contact_cleanup: motion/walk.solve with the root rotation fixed).

Used by tools/retarget_video.py (SAM-3D-Body video fits), tools/retarget_smplx.py (SMPL-X mocap) and
tools/refit_wan.py (re-fitting a WAN clip's performance).
"""
from __future__ import annotations

import time

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from . import io as MI
from . import walk
from .walk import fps_scale
from ..rig.mhr import MHRBody, quat_xyzw_to_mat
from ..rig.skeleton import JOINT, ROT, body_index


def contact_targets(body: MHRBody, Jr: np.ndarray, fps: float, v_max: float = 0.35, h_max: float = 0.04,
                    verbose: bool = True) -> tuple[dict, dict, dict]:
    """Ankle/toe targets [T, 3] per foot: held at their mean position through each contact segment (toe low and
    slow), planted toes at their canonical height; the switches softened. Returns (ankle, toe, contact)."""
    J0 = body.joints_canon
    T = len(Jr)
    anc, toe, con = {}, {}, {}
    for f in ("r", "l"):
        A, Tt = Jr[:, walk.ANKLE[f]], Jr[:, walk.TOE[f]]
        h = Tt[:, 1] - Tt[:, 1].min()
        v = np.linalg.norm(np.gradient(Tt[:, [0, 2]], axis=0), axis=1) * fps
        contact = (v < v_max) & (h < h_max)
        contact = gaussian_filter1d(contact.astype(float), 1.0) > 0.5
        A2, T2 = A.copy(), Tt.copy()
        i = 0
        while i < T:
            if contact[i]:
                j = i
                while j < T and contact[j]:
                    j += 1
                A2[i:j] = A[i:j].mean(0); T2[i:j] = Tt[i:j].mean(0)
                lift = T2[i:j, 1].min() - (J0[walk.TOE[f]][1])   # planted toes sit at their canonical height
                A2[i:j, 1] -= lift; T2[i:j, 1] -= lift
                i = j
            else:
                i += 1
        anc[f] = gaussian_filter1d(A2, 1.0, axis=0, mode="nearest")
        toe[f] = gaussian_filter1d(T2, 1.0, axis=0, mode="nearest")
        con[f] = contact
        if verbose:
            print(f"foot {f}: contact {contact.mean() * 100:.0f}% of frames")
    return anc, toe, con


def contact_cleanup(body: MHRBody, m: MI.Motion, iters: int = 400, verbose: bool = True) -> dict:
    """Re-solve a root_R motion so planted feet stay put: every named DOF pulled to the motion's value (legs weakly,
    the feet decide), the pelvis follows the motion, the root rotation stays fixed."""
    T = len(m)
    b0 = body.body0[0].cpu().numpy()
    P = MI.pose(body, m, 0, T)
    Jr = P.joints.cpu().numpy()
    anc, toe, _ = contact_targets(body, Jr, m.fps, verbose=verbose)
    style = (m.body_params - b0[None]).astype(np.float32)
    w_style = np.full(130, 0.02, np.float32)
    for ax in ROT.values():
        for mi in ax:
            if mi is not None:
                w_style[body_index(mi)] = 2.0
    for n in ("r_hip", "l_hip", "r_knee", "l_knee", "r_ankle", "l_ankle"):   # legs: the feet decide
        for mi in ROT[n]:
            if mi is not None:
                w_style[body_index(mi)] = 0.2
    t = np.arange(T) / m.fps
    ch = walk.Choreo(m.fps, t, anc, toe, Jr[:, JOINT["pelvis"]], np.ones((T, 3)), style, w_style, {}, {},
                     (m.expr - body.expr[0].cpu().numpy()[None]).astype(np.float32), root_R=m.root_R.astype(np.float32))
    sol = walk.solve(body, ch, iters=iters, verbose=verbose)
    sol["ankle_r"], sol["ankle_l"] = anc["r"], anc["l"]
    return sol


def ik_to_targets(body: MHRBody, dS: torch.Tensor, bones: tuple, Rt: torch.Tensor, orient: tuple, pel_t: np.ndarray,
                  w0: np.ndarray, fps: float, iters: int, x0: np.ndarray | None = None, verbose: bool = True,
                  neutral: bool = False) -> MI.Motion:
    """Per-frame IK through the differentiable MHR model: the named rotation DOFs, a root rotation (rotvec, about the
    canonical pelvis, applied after the forward) and a root translation (world), to unit bone directions dS [T, P, 3]
    (bones = (a, b, weight) tensors: direction of joint b - joint a), world orientations Rt [T, O, 3, 3] of joints
    orient = (ids, weights), and the pelvis path pel_t [T, 3]. `x0` [T, 130]: initial body-param offsets.
    `neutral`: start from and regularise towards MHR's zero pose instead of the capture pose (an arms-up capture
    otherwise keeps its raised clavicles: lowering the arms from there moved the clavicles as much as the shoulders)."""
    dev = body.device
    T = len(pel_t)
    ma, mb, wb = bones
    oj, wo = orient
    J0 = body.joints_canon
    pel0 = J0[JOINT["pelvis"]]
    b0 = body.body0.detach()
    free = torch.zeros(130, dtype=torch.bool, device=dev)
    for ax in ROT.values():
        for mi in ax:
            if mi is not None:
                free[body_index(mi)] = True
    free_idx = torch.nonzero(free).flatten()
    x = torch.zeros(T, len(free_idx), device=dev)
    if x0 is not None:
        x[:] = torch.as_tensor(x0, dtype=torch.float32, device=dev)[:, free_idx]
    elif neutral:
        x[:] = -b0[0, free_idx]
    x_ref = -b0[0, free_idx] if neutral else torch.zeros(len(free_idx), device=dev)
    x.requires_grad_(True)
    w = torch.as_tensor(w0, dtype=torch.float32, device=dev).requires_grad_(True)
    pel0_t = torch.as_tensor(pel0, dtype=torch.float32, device=dev)
    pel_tt = torch.as_tensor(pel_t, dtype=torch.float32, device=dev)
    r = (pel_tt - pel0_t).clone().requires_grad_(True)
    opt = torch.optim.Adam([x, w, r], lr=0.03)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=0.03 * 0.05)
    fs = fps_scale(fps)
    t_start = time.time()
    for it in range(iters):
        p = b0.expand(T, -1).clone()
        p[:, free_idx] = p[:, free_idx] + x
        mp = body.model_params(p)
        _, skel = body.mhr(body.shape.expand(T, -1), mp, body.expr.expand(T, -1))
        RR = rotvec_to_mat(w)
        Jw = (skel[..., :3] / 100) @ body._A.T + body._t
        Jw = (Jw - pel0_t) @ RR.transpose(-1, -2) + pel0_t + r[:, None]
        Rj = RR[:, None] @ (body._Rw @ quat_xyzw_to_mat(skel[:, oj, 3:7]))
        d = Jw[:, mb] - Jw[:, ma]
        d = d / (d.norm(dim=-1, keepdim=True) + 1e-8)
        l_dir = (((d - dS) ** 2).sum(-1) * wb).mean()
        l_or = (((Rj - Rt) ** 2).sum((-1, -2)) * wo).mean()
        l_pel = ((Jw[:, JOINT["pelvis"]] - pel_tt) ** 2).sum(-1).mean()
        acc = x[2:] - 2 * x[1:-1] + x[:-2]
        wacc = w[2:] - 2 * w[1:-1] + w[:-2]
        l_s = (acc ** 2).sum(-1).mean() + (wacc ** 2).sum(-1).mean()
        l_reg = ((x - x_ref) ** 2).mean()
        loss = l_dir + 0.5 * l_or + 30 * l_pel + 10 * l_s * fs + 1e-3 * l_reg
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if verbose and (it % 100 == 0 or it == iters - 1):
            ang = torch.rad2deg(torch.acos((d * dS).sum(-1).clamp(-1, 1)))
            print(f"ik {it:4d}: bone dir err mean {ang.mean().item():.1f} deg p95 {ang.flatten().quantile(0.95).item():.1f}  "
                  f"orient {l_or.item():.4f}  pelvis {l_pel.sqrt().item() * 100:.1f} cm  ({time.time() - t_start:.0f}s)", flush=True)
    with torch.no_grad():
        p = b0.expand(T, -1).clone(); p[:, free_idx] = p[:, free_idx] + x
        RR = rotvec_to_mat(w)
    ex = body.expr[0].cpu().numpy()[None].repeat(T, 0)
    return MI.Motion({"body_params": p.cpu().numpy().astype(np.float32), "expr": ex.astype(np.float32), "fps": fps,
                      "root_R": RR.cpu().numpy().astype(np.float32), "root_t": r.detach().cpu().numpy().astype(np.float32),
                      "root_c": pel0.astype(np.float32)})


def rotvec_to_mat(w: torch.Tensor) -> torch.Tensor:
    th = w.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = w / th
    K = torch.zeros(*w.shape[:-1], 3, 3, device=w.device)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K = K - K.transpose(-1, -2)
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    return torch.eye(3, device=w.device) + s * K + (1 - c) * (K @ K)
