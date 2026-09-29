"""Walk / stop / wave choreography, solved by inverse kinematics through the MHR forward.

A choreography is a set of world-space targets per frame (ankle + toe of each foot, pelvis, optionally wrist and
elbow) plus soft per-DOF style targets (arm swing, spine counter-rotation, head). `solve` optimises every frame's
body params and root translation jointly (Adam through the TorchScript `mhr_model.pt`, which is differentiable) with
a temporal smoothness term, so planted feet stay planted and the rest of the body follows the style curves.

Frame conventions (the subject's canonical frame): world up +Y, the subject faces `fwd` (toe minus ankle, projected
on the ground plane), `side` points to the subject's left. The subject walks along `fwd`.

    m = walk.walk_stop_wave(body, fps=30)          # dict: body_params [T,130], trans [T,3] world, expr [T,72], ...
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from ..rig.mhr import MHRBody, TRANS
from ..rig.skeleton import HINGE_MIN, JOINT, ROT, body_index
from .procedural import FACE_DIMS

ANKLE = {"r": 20, "l": 4}
TOE = {"r": 24, "l": 8}
HEEL_BACK = 0.05   # heel pivot this far behind the ankle joint, on the ground


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def minjerk(x):
    x = np.clip(x, 0.0, 1.0)
    return x ** 3 * (10 - 15 * x + 6 * x * x)


@dataclass
class Frame3:
    up: np.ndarray
    fwd: np.ndarray
    side: np.ndarray   # subject's left


def subject_frame(body: MHRBody) -> Frame3:
    J = body.joints_canon
    up = np.array([0.0, 1.0, 0.0])
    f = (J[TOE["r"]] - J[ANKLE["r"]]) + (J[TOE["l"]] - J[ANKLE["l"]])
    f = f - up * (f @ up)
    f /= np.linalg.norm(f)
    side = np.cross(up, f)   # up x fwd = subject's left when fwd is +Z and up +Y: (0,1,0)x(0,0,1) = (1,0,0) = +X = left
    return Frame3(up, f, side)


def dof(name: str, axis: str) -> int:
    return body_index(ROT[name]["xyz".index(axis)])


# ------------------------------------------------------------------------------------------------ feet
@dataclass
class Step:
    foot: str          # "r" | "l"
    t_off: float       # lift-off (toe-off) time, s
    t_on: float        # landing (heel-strike) time, s
    fwd: float         # landing ankle position along fwd (m, relative to the canonical ankle)
    lat: float         # landing ankle lateral position along side (m, absolute, relative to the pelvis line)


def foot_tracks(steps: list[Step], t: np.ndarray, fr: Frame3, J0: np.ndarray, ground_y: float,
                lift: float = 0.07, strike_pitch: float = 0.25, off_pitch: float = 0.55, heel_rise: float = 0.18):
    """Per foot: ankle and toe world targets [T, 3] from its steps (heel strike -> flat -> heel off -> swing)."""
    out = {}
    for ft in ("r", "l"):
        a0, t0 = J0[ANKLE[ft]], J0[TOE[ft]]
        h_a = (a0 - ground_y * fr.up) @ fr.up - 0.0      # ankle height over the ground plane
        toe_vec = t0 - a0                                  # canonical ankle -> toe
        my = [s for s in steps if s.foot == ft]
        # the foot's planted positions: canonical first, then each landing
        lat0 = (a0 - J0[1]) @ fr.side
        plants = [(0.0, lat0)] + [(s.fwd, s.lat) for s in my]
        A = np.zeros((len(t), 3)); T = np.zeros((len(t), 3))

        def planted(k):
            f, l = plants[k]
            base = a0 + fr.fwd * f + fr.side * (l - lat0)
            return base

        for i, ti in enumerate(t):
            # which phase: find the last step with t_on <= ti, and whether we are inside a swing
            k = 0
            pitch = 0.0
            pos = None
            for j, s in enumerate(my):
                if ti >= s.t_on:
                    k = j + 1
            swing = next((j for j, s in enumerate(my) if s.t_off <= ti < s.t_on), None)
            if swing is not None:
                s = my[swing]
                u = (ti - s.t_off) / (s.t_on - s.t_off)
                p0, p1 = planted(swing), planted(swing + 1)
                pos = p0 + (p1 - p0) * minjerk(u) + fr.up * (lift * math.sin(math.pi * min(1.0, u * 1.1)) ** 1.5)
                # pitch: toe-down at lift-off, back through flat to toe-up at strike
                pitch = -off_pitch * (1 - smooth(u / 0.45)) + strike_pitch * smooth((u - 0.5) / 0.5)
                # during swing the ankle is also raised by the heel-rise it had at lift-off, fading out
                pos = pos + fr.up * heel_rise * 0.5 * (1 - smooth(u / 0.4))
            else:
                p = planted(k)
                pos = p
                # heel strike rolls to flat over 0.12 s after landing (pivot at the heel)
                if k > 0 and ti - my[k - 1].t_on < 0.12:
                    pitch = strike_pitch * (1 - smooth((ti - my[k - 1].t_on) / 0.12))
                    heel = p - fr.fwd * HEEL_BACK - fr.up * h_a
                    R = rot_about(fr.side, -pitch)
                    pos = heel + R @ (p - heel)
                # heel off before the next lift-off (pivot at the toe)
                nxt = my[k] if k < len(my) else None
                if nxt is not None and ti > nxt.t_off - 0.2:
                    u = smooth((ti - (nxt.t_off - 0.2)) / 0.2)
                    pitch = -off_pitch * u
                    toe = p + toe_vec - fr.up * ((p + toe_vec) @ fr.up - ground_y - 0.01)
                    R = rot_about(fr.side, -pitch)
                    pos = toe + R @ (p - toe)
            R = rot_about(fr.side, -pitch)
            A[i] = pos
            T[i] = pos + R @ toe_vec
        out[ft] = (A, T)
    return out


def rot_about(axis, ang):
    """Rotation by `ang` about unit `axis` (Rodrigues). Positive pitch (toe up) about the subject's left axis is a
    negative rotation there, hence callers pass -pitch."""
    a = np.asarray(axis, float); a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


# ------------------------------------------------------------------------------------------------ choreography
@dataclass
class Choreo:
    fps: float
    t: np.ndarray
    ankle: dict            # foot -> [T, 3]
    toe: dict
    pelvis: np.ndarray     # [T, 3]
    w_pelvis: np.ndarray   # [T, 3] per-axis weights (fwd/lat/up mixed into world)
    style: np.ndarray      # [T, 130] body-param style targets (offsets from canonical)
    w_style: np.ndarray    # [130] weights
    wrist: dict            # side -> ([T, 3] target, [T] weight)
    elbow: dict
    expr: np.ndarray       # [T, 72] offsets
    look: np.ndarray | None = None   # [T] 0..1 blend of the head turning to the camera (informational)
    root_R: np.ndarray | None = None  # [T, 3, 3] fixed root rotation about the canonical pelvis (retargeted video motion);
                                      # the solved translation is then a world offset after it (root_t)


def walk_stop_wave(body: MHRBody, fps: float = 30.0, n_steps: int = 7, step_len: float = 0.62, step_t: float = 0.56,
                   half_width: float = 0.085, stand_half_width: float = 0.11, pause: float = 0.6, wave_s: float = 2.6,
                   hold_s: float = 1.2, start_hold: float = 0.4, arm_swing: float = 0.32, wave: bool = True,
                   smile: bool = True, relax_s: float = 0.6) -> Choreo:
    """Stand (canonical A-pose) -> arms relax down -> walk `n_steps` steps along fwd (first and last are half steps)
    -> stop with the feet together -> smile and wave with the right hand -> lower the hand and hold."""
    fr = subject_frame(body)
    J0 = body.joints_canon
    ground_y = float(body.verts_canon[:, 1].min())
    pel0 = J0[JOINT["pelvis"]]

    # ---- footsteps. Step k lands at t_on; the foot lifts one step period earlier plus the double-support time.
    ds = 0.12 * step_t * 2 / 2       # double support per transition (~12 % of the step)
    t_first = start_hold + relax_s * 0.6
    steps, t_on = [], t_first + step_t
    fwd_pos = {"r": 0.0, "l": 0.0}
    lead = "r"
    for k in range(n_steps):
        ft = lead if k % 2 == 0 else ("l" if lead == "r" else "r")
        other = "l" if ft == "r" else "r"
        if k == 0:
            target = fwd_pos[other] + step_len * 0.5
        elif k == n_steps - 1:
            target = fwd_pos[other]                  # closing step: next to the other foot
        else:
            target = fwd_pos[other] + step_len
        lat = (-1 if ft == "r" else 1) * (stand_half_width if k == n_steps - 1 else half_width)
        dur = step_t if k not in (0, n_steps - 1) else step_t * 0.9
        t_off = t_on - dur + ds
        steps.append(Step(ft, t_off, t_on, target, lat))
        fwd_pos[ft] = target
        t_on += step_t if k < n_steps - 2 else step_t * 1.05
    t_stop = steps[-1].t_on + 0.35
    t_wave0 = t_stop + pause
    t_wave1 = t_wave0 + (wave_s if wave else 0.0)
    T_end = t_wave1 + hold_s + 0.6
    t = np.arange(int(round(T_end * fps))) / fps
    tracks = foot_tracks(steps, t, fr, J0, ground_y)

    # ---- pelvis: over the stance foot at mid-stance; spline through those support points.
    # support position along fwd = mean of the two planted feet, weighted to the stance foot in single support.
    def foot_fwd(ft, ti):
        A = tracks[ft][0]
        i = min(int(ti * fps), len(t) - 1)
        return (A[i] - J0[ANKLE[ft]]) @ fr.fwd
    pel_f = np.array([0.5 * (foot_fwd("r", ti) + foot_fwd("l", ti)) for ti in t])
    # a walking pelvis leads the feet midpoint: move it forward by a fraction of the current speed
    k = max(1, int(0.25 * fps))
    pel_f = np.convolve(np.pad(pel_f, (k, k), mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
    vel = np.gradient(pel_f) * fps
    pel_f = pel_f + 0.08 * vel
    # lateral sway towards the stance foot, vertical bob (low at double support, high in mid-stance)
    lat = np.zeros_like(t); bob = np.zeros_like(t)
    walking = np.zeros_like(t)
    for s in steps:
        walking = np.maximum(walking, ((t > s.t_off - 0.3) & (t < s.t_on + 0.3)).astype(float))
    walking = np.convolve(np.pad(walking, (k, k), mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
    for s in steps:
        other = "l" if s.foot == "r" else "r"
        c = 0.5 * (s.t_off + s.t_on)                        # the other foot's mid-stance
        w = np.exp(-0.5 * ((t - c) / (0.35 * step_t)) ** 2)
        lat += w * (0.022 if other == "l" else -0.022)
        bob += w * 0.012
        bob -= np.exp(-0.5 * ((t - s.t_on) / (0.15 * step_t)) ** 2) * 0.012
    # the pelvis also drops a little while walking (knees never fully straight)
    pel = (pel0[None] + fr.fwd[None] * pel_f[:, None] + fr.side[None] * lat[:, None]
           + fr.up[None] * (bob - 0.012 * walking)[:, None])
    w_pel = np.stack([np.full_like(t, 3.0), np.full_like(t, 3.0), np.full_like(t, 3.0)], 1)

    # ---- style targets (offsets from canonical)
    style = np.zeros((len(t), 130), np.float32)
    w_style = np.full(130, 0.02, np.float32)          # every DOF: weak pull to canonical+style
    relax = smooth((t - start_hold) / relax_s)          # arms come down from the A-pose
    # arm swing: right arm forward when the left leg is forward. Phase from the feet.
    fr_r = np.array([foot_fwd("r", ti) for ti in t]); fr_l = np.array([foot_fwd("l", ti) for ti in t])
    swing = np.clip((fr_l - fr_r) / step_len, -1, 1) * walking   # +1: left foot ahead -> right arm forward
    arms_down = arms_down_offsets(body)
    for hnd in ("r", "l"):
        for ax, v in arms_down[hnd].items():
            style[:, dof(f"{hnd}_shoulder", ax)] += relax * v
        sgn = 1.0 if hnd == "r" else -1.0
        style[:, dof(f"{hnd}_shoulder", "z")] += arm_swing * sgn * swing
        style[:, dof(f"{hnd}_elbow", "z")] += relax * 0.18 + 0.2 * np.clip(sgn * swing, 0, 1) * 1.0 + 0.05 * walking
        w_style[dof(f"{hnd}_shoulder", "x")] = w_style[dof(f"{hnd}_shoulder", "y")] = w_style[dof(f"{hnd}_shoulder", "z")] = 1.0
        w_style[dof(f"{hnd}_elbow", "z")] = 1.0
    # thorax counter-rotation and a slight forward lean while walking
    style[:, dof("spine2", "y")] += 0.05 * swing
    style[:, dof("spine1", "y")] += 0.03 * swing
    for n in ("spine1", "spine2", "spine3", "neck", "head", "spine0"):
        for ax in "xyz":
            w_style[dof(n, ax)] = 20.0
    # head stays level: counter the spine twist
    style[:, dof("neck", "y")] -= 0.06 * swing
    for d in ("r_wrist", "l_wrist"):
        for ax in "xyz":
            w_style[dof(d, ax)] = 0.3

    wrist, elbow = {}, {}
    expr = np.zeros((len(t), 72), np.float32)
    if wave:
        # right hand: upper arm out to the side and a little forward, forearm up, the hand waving side to side
        up = smooth((t - t_wave0) / 0.55) * (1 - smooth((t - (t_wave1 - 0.1)) / 0.7))
        S = J0[JOINT["r_shoulder"]]
        # subject's right is -side
        E = S - fr.side * 0.23 - fr.up * 0.06 + fr.fwd * 0.07
        osc = np.sin(2 * math.pi * 1.9 * np.clip(t - t_wave0 - 0.45, 0, None)) * smooth((t - t_wave0 - 0.35) / 0.3)
        W = (E[None] + fr.up[None] * 0.25 + fr.fwd[None] * 0.05
             - fr.side[None] * (0.03 + 0.07 * osc)[:, None] - fr.up[None] * (0.012 * np.abs(osc))[:, None])
        wp = wave_pose_offsets(body, E, E + fr.up * 0.25 + fr.fwd * 0.05 - fr.side * 0.03)
        ad = arms_down_offsets(body)["r"]
        for k, v in wp.items():
            if k.startswith("_"):
                continue
            n, ax = k.split(".")
            base = ad.get(ax, 0.0) * relax if n == "r_shoulder" else 0.0
            cur = style[:, dof(n, ax)] - (base if n == "r_shoulder" else 0.0) * 0   # walking style already there
            style[:, dof(n, ax)] = style[:, dof(n, ax)] * (1 - up) + v * up
            w_style[dof(n, ax)] = 1.0
        carry = pel - pel0[None]                      # the targets travel with the body
        # the IK targets act only near the full wave pose; the raise and the lowering follow the interpolated style
        # (partial IK weights there fight the style and the spine takes up the difference)
        wrist["r"] = (W + carry, 2.0 * up ** 4)
        elbow["r"] = (E[None] + carry, 0.5 * up ** 4)
        for ax in "xyz":   # while the IK drives the arm, the style targets step aside
            w_style[dof("r_shoulder", ax)] = 1.0
        # during the wave the arm style targets are released by down-weighting via the solve (see solve: style_mask)
    if smile:
        s_on = smooth((t - (t_stop - 0.2)) / 0.5) * (1 - smooth((t - (T_end - 0.2)) / 0.8) * 0.0)
        for d in FACE_DIMS["smile"]:
            expr[:, d] += 0.75 * s_on
        # eyes crinkle / cheeks: a touch of brow, a small jaw opening at the peak of the smile
        expr[:, 24] += 0.08 * s_on
        for d in FACE_DIMS["brows"]:
            expr[:, d] += 0.15 * s_on * (1 - smooth((t - t_wave0 - 0.8) / 0.8))
    # blinks, every ~2.5 s
    for tb in np.arange(1.3, T_end - 0.3, 2.7):
        pulse = np.exp(-0.5 * ((t - tb) / 0.06) ** 2)
        for d in FACE_DIMS["blink"]:
            expr[:, d] += 0.95 * pulse
    # head: look ahead while walking, at the camera (straight ahead, slightly down-left) when stopped: a small tilt
    look = smooth((t - t_stop + 0.3) / 0.6)
    style[:, dof("head", "z")] += 0.06 * look * (1 - smooth((t - t_wave1) / 1.0) * 0.5)
    style[:, dof("head", "x")] += -0.03 * look
    return Choreo(fps, t, {f: tracks[f][0] for f in tracks}, {f: tracks[f][1] for f in tracks}, pel, w_pel,
                  style, w_style, wrist, elbow, expr, look)


_ARMS_CACHE: dict = {}


def wave_pose_offsets(body: MHRBody, E: np.ndarray, W: np.ndarray, restarts: int = 6) -> dict:
    """Right-arm DOF offsets (clavicle/shoulder x,y,z, elbow z, wrist x,y,z) that put the elbow at E and the wrist at
    W (canonical frame), the rest of the body frozen; best of a few random starts (the arm has to rotate ~90 degrees
    up, and a single start from the hanging arm can stall)."""
    dev = body.device
    names = [("r_clavicle", a) for a in "xyz"] + [("r_shoulder", a) for a in "xyz"] + [("r_elbow", "z")]
    idx = [dof(n, a) for n, a in names]
    Et = torch.as_tensor(E, dtype=torch.float32, device=dev); Wt = torch.as_tensor(W, dtype=torch.float32, device=dev)
    best = (1e9, None)
    rng = np.random.default_rng(0)
    for r in range(restarts):
        x0 = rng.uniform(-1.2, 1.2, len(idx)) * (r > 0)
        x0[:3] *= 0.2   # the clavicle moves little
        x = torch.tensor(x0, dtype=torch.float32, device=dev, requires_grad=True)
        opt = torch.optim.Adam([x], lr=0.05)
        for it in range(250):
            p = body.body0.clone(); p[0, idx] = p[0, idx] + x
            _, sk = body.mhr(body.shape, body.model_params(p), body.expr)
            Jw = (sk[0, :, :3] / 100) @ body._A.T + body._t
            loss = ((Jw[JOINT["r_wrist"]] - Wt) ** 2).sum() + ((Jw[JOINT["r_elbow"]] - Et) ** 2).sum() \
                + 1e-3 * (x ** 2).sum() + 1e-2 * (x[:3] ** 2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        if loss.item() < best[0]:
            best = (loss.item(), x.detach().cpu().numpy())
    return {n + "." + a: float(v) for (n, a), v in zip(names, best[1])} | {"_err": math.sqrt(best[0])}


def arms_down_offsets(body: MHRBody) -> dict:
    """Shoulder x/y/z offsets that bring the A-pose arms down to hang beside the body (IK on the wrists: beside the hip,
    just in front of the thigh)."""
    key = id(body)
    if key in _ARMS_CACHE:
        return _ARMS_CACHE[key]
    fr = subject_frame(body)
    J0 = body.joints_canon
    dev = body.device
    idx = [dof(f"{h}_shoulder", a) for h in "rl" for a in "xyz"] + [dof("r_elbow", "z"), dof("l_elbow", "z")]
    x = torch.zeros(len(idx), device=dev, requires_grad=True)
    tgt = {}
    for h, sgn in (("r", -1), ("l", 1)):
        hip = J0[JOINT[f"{h}_hip"]]
        tgt[h] = torch.as_tensor(hip + fr.side * sgn * 0.17 - fr.up * 0.11 + fr.fwd * 0.03, dtype=torch.float32, device=dev)
    opt = torch.optim.Adam([x], lr=0.03)
    for it in range(200):
        p = body.body0.clone()
        p[0, idx] = p[0, idx] + x
        v, sk = body.mhr(body.shape, body.model_params(p), body.expr)
        Jw = (sk[0, :, :3] / 100) @ body._A.T + body._t
        loss = sum(((Jw[JOINT[f"{h}_wrist"]] - tgt[h]) ** 2).sum() for h in "rl") + 1e-3 * (x ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    xv = x.detach().cpu().numpy()
    out = {"r": dict(zip("xyz", xv[0:3])), "l": dict(zip("xyz", xv[3:6]))}
    _ARMS_CACHE[key] = out
    return out


# ------------------------------------------------------------------------------------------------ solver
def solve(body: MHRBody, ch: Choreo, iters: int = 600, lr: float = 0.02, w_smooth: float = 30.0,
          verbose: bool = True) -> dict:
    """IK over all frames: body param offsets [T, 130] (hand slots excluded) and root translation [T, 3] (world)."""
    dev = body.device
    T = len(ch.t)
    fr = subject_frame(body)
    b0 = body.body0.detach()
    hand = torch.zeros(130, dtype=torch.bool, device=dev)
    hand[(body.hand_idx - 6).clamp(0, 129)] = True
    free = ~hand
    free[116:] = False   # model params 122..135 (extra hip/ankle DOFs, bone lengths) stay canonical
    # only the named rotation DOFs (rig/skeleton.ROT) move: MHR's other spine DOFs are unregularised shortcuts the IK
    # otherwise bends the back with
    named = torch.zeros(130, dtype=torch.bool, device=dev)
    for ax in ROT.values():
        for mi in ax:
            if mi is not None:
                named[body_index(mi)] = True
    free &= named
    free_idx = torch.nonzero(free).flatten()
    style = torch.as_tensor(ch.style, device=dev)
    w_style = torch.as_tensor(ch.w_style, device=dev)
    # while an IK target drives an arm, its style pull fades
    wmask = torch.ones(T, 130, device=dev)
    for h, (W, w) in ch.wrist.items():
        f = torch.as_tensor(np.clip(np.asarray(w) / (np.max(w) + 1e-9), 0, 1), dtype=torch.float32, device=dev)
        for ax in "xyz":
            wmask[:, dof(f"{h}_shoulder", ax)] = 1 - 0.5 * f
            wmask[:, dof(f"{h}_wrist", ax)] = 1 - 0.9 * f
        wmask[:, dof(f"{h}_elbow", "z")] = 1 - 0.5 * f
    x = torch.zeros(T, len(free_idx), device=dev)
    x[:] = style[:, free_idx]
    x.requires_grad_(True)
    # root translation init: pelvis target minus canonical pelvis
    pel0 = torch.as_tensor(body.joints_canon[JOINT["pelvis"]], dtype=torch.float32, device=dev)
    pel_t = torch.as_tensor(ch.pelvis, dtype=torch.float32, device=dev)
    r = (pel_t - pel0).clone().requires_grad_(True)
    tgt = {}
    for f in ("r", "l"):
        tgt[f] = (torch.as_tensor(ch.ankle[f], dtype=torch.float32, device=dev),
                  torch.as_tensor(ch.toe[f], dtype=torch.float32, device=dev))
    Ainv = torch.linalg.inv(body._A)
    tr0 = body.model_params0[0, TRANS] / 10
    RR = None if ch.root_R is None else torch.as_tensor(ch.root_R, dtype=torch.float32, device=dev)
    hinge_idx = torch.as_tensor([body_index(ROT[h][2]) for h in HINGE_MIN], device=dev)
    hinge_min = torch.as_tensor(list(HINGE_MIN.values()), device=dev)
    opt = torch.optim.Adam([x, r], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr * 0.05)
    wr = {h: (torch.as_tensor(W, dtype=torch.float32, device=dev), torch.as_tensor(w, dtype=torch.float32, device=dev))
          for h, (W, w) in ch.wrist.items()}
    we = {h: (torch.as_tensor(W, dtype=torch.float32, device=dev), torch.as_tensor(w, dtype=torch.float32, device=dev))
          for h, (W, w) in ch.elbow.items()}
    for it in range(iters):
        p = b0.expand(T, -1).clone()
        p[:, free_idx] = p[:, free_idx] + x
        if RR is None:
            gt = tr0[None] + r @ Ainv.T
            mp = body.model_params(p, global_trans=gt)
        else:
            mp = body.model_params(p)
        _, sk = body.mhr(body.shape.expand(T, -1), mp, body.expr.expand(T, -1))
        Jw = (sk[..., :3] / 100) @ body._A.T + body._t
        if RR is not None:
            Jw = (Jw - pel0) @ RR.transpose(-1, -2) + pel0 + r[:, None]
        l_feet = sum(((Jw[:, ANKLE[f]] - tgt[f][0]) ** 2).sum(-1).mean() + ((Jw[:, TOE[f]] - tgt[f][1]) ** 2).sum(-1).mean()
                     for f in ("r", "l"))
        l_pel = ((Jw[:, JOINT["pelvis"]] - pel_t) ** 2).sum(-1).mean()
        off = p - b0
        l_style = ((off - style) ** 2 * w_style * wmask).sum(-1).mean()
        l_arm = 0.0
        for h, (W, w) in wr.items():
            l_arm = l_arm + (w * ((Jw[:, JOINT[f"{h}_wrist"]] - W) ** 2).sum(-1)).mean()
        for h, (E, w) in we.items():
            l_arm = l_arm + (w * ((Jw[:, JOINT[f"{h}_elbow"]] - E) ** 2).sum(-1)).mean()
        acc = x[2:] - 2 * x[1:-1] + x[:-2]
        racc = r[2:] - 2 * r[1:-1] + r[:-2]
        l_s = (acc ** 2).sum(-1).mean() + 10 * (racc ** 2).sum(-1).mean()
        # hinge guard (rig/skeleton.HINGE_MIN): planting a foot the leg cannot quite reach bent the knee backwards
        # (retargeted flamenco: -0.49 rad from an IK result that stayed above -0.15)
        l_hinge = (torch.relu(hinge_min - p[:, hinge_idx]) ** 2).sum(-1).mean()
        loss = 1000 * l_feet + 300 * l_pel + l_style + 1000 * l_arm + w_smooth * l_s * fps_scale(ch.fps) + 1000 * l_hinge
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if verbose and (it % 100 == 0 or it == iters - 1):
            print(f"ik {it:4d}: feet {math.sqrt(l_feet.item() / 2) * 100:.2f} cm  pelvis {math.sqrt(l_pel.item()) * 100:.2f} cm  "
                  f"arm {float(l_arm):.4f}  style {l_style.item():.4f}  smooth {l_s.item():.2e}", flush=True)
    with torch.no_grad():
        p = b0.expand(T, -1).clone(); p[:, free_idx] = p[:, free_idx] + x
        gt = tr0[None] + r @ Ainv.T
    if RR is not None:
        return {"body_params": p.cpu().numpy(), "root_R": ch.root_R.astype(np.float32), "root_t": r.detach().cpu().numpy(),
                "root_c": pel0.cpu().numpy(), "root_world": r.detach().cpu().numpy(),
                "expr": (body.expr + torch.as_tensor(ch.expr, device=dev)).cpu().numpy(), "fps": ch.fps, "t": ch.t}
    return {"body_params": p.cpu().numpy(), "global_trans": gt.cpu().numpy(), "root_world": r.detach().cpu().numpy(),
            "expr": (body.expr + torch.as_tensor(ch.expr, device=dev)).cpu().numpy(), "fps": ch.fps, "t": ch.t}


def fps_scale(fps: float) -> float:
    """Second differences scale with 1/fps^2; keep the smoothness term's meaning fixed across frame rates."""
    return (fps / 30.0) ** 4
