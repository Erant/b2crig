"""Retarget a SAM-3D-Body video fit (tools/video_fit.py) onto a subject, with foot-contact cleanup -> motion npz.

    .venv/bin/python tools/retarget_video.py work/<subject> CLIP OUT.npz [--fps-out 30] [--smooth 1.5] [--device cpu]

1. Frame alignment: frame 0 of a tools/motion_video.py clip IS the subject's canonical render, so a Kabsch fit of the
   frame-0 fitted joints to the subject's canonical joints maps SAM-3D-Body's camera frame into the splat's world.
2. Pose: body params = subject canonical + (video(t) - video(0)) (joint-angle deltas; frame 0 is exactly the splat's
   pose), smoothed in time; root = the pelvis's rotation and translation relative to frame 0, mapped to world.
3. Clean-up IK (motion/walk.solve with the root rotation fixed): feet planted where the retargeted toe is low and
   slow (contact segments held at their mean position), the pelvis follows the video, every named DOF pulled to
   the video's value.
Optionally resampled to --fps-out (the video is 16 fps).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2crig.motion.retarget import contact_cleanup  # noqa: E402
from b2crig.motion import io as MI  # noqa: E402
from b2crig.rig.mhr import MHRBody, BODY  # noqa: E402
from b2crig.rig.skeleton import JOINT  # noqa: E402

KABSCH_JOINTS = [JOINT[k] for k in ("pelvis", "spine1", "spine3", "neck", "head", "r_shoulder", "l_shoulder", "r_hip",
                                    "l_hip", "r_knee", "l_knee", "r_ankle", "l_ankle", "r_elbow", "l_elbow")]


def kabsch(P, Q):
    """R, s, t minimising |s R P + t - Q| (rows are points)."""
    mp, mq = P.mean(0), Q.mean(0)
    X, Y = P - mp, Q - mq
    U, S, Vt = np.linalg.svd(X.T @ Y)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    s = (S * np.diag(D)).sum() / (X ** 2).sum()
    return R, s, mq - s * R @ mp


def resample(x, t_src, t_dst):
    return np.stack([np.interp(t_dst, t_src, x[:, j]) for j in range(x.shape[1])], 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("subject", type=Path)
    ap.add_argument("clip", help="clip id, or several joined by '+': chained clips (tools/motion_video.py --first), "
                    "whose first frame repeats the previous clip's last one; concatenated with that frame dropped")
    ap.add_argument("out", type=Path)
    ap.add_argument("--fps-in", type=float, default=16.0)
    ap.add_argument("--fps-out", type=float, default=30.0)
    ap.add_argument("--smooth", type=float, default=1.2, help="Gaussian sigma (input frames) on params and root")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-ik", action="store_true")
    ap.add_argument("--fits-from", type=Path, help="read the clips' video_fit.npz from this subject (retarget another "
                    "subject's generated motion; same MHR skeleton)")
    ap.add_argument("--end", type=int, default=0, help="keep the concatenated input frames [0, end) only")
    a = ap.parse_args()
    S = a.subject
    parts = [np.load((a.fits_from or S) / "clips" / c / "video_fit.npz") for c in a.clip.split("+")]
    d = {k: np.concatenate([parts[0][k]] + [q[k][1:] for q in parts[1:]]) for k in parts[0].files}
    if a.end:
        d = {k: v[:a.end] for k, v in d.items()}
    body = MHRBody(S / "mhr.npz", device=a.device)
    J0 = body.joints_canon
    Jv = d["joints"]
    T = len(Jv)
    R, s, t = kabsch(Jv[0][KABSCH_JOINTS], J0[KABSCH_JOINTS])
    res = np.linalg.norm((s * Jv[0][KABSCH_JOINTS] @ R.T + t) - J0[KABSCH_JOINTS], axis=1)
    print(f"frame-0 alignment: scale {s:.3f}, joint residual mean {res.mean() * 100:.1f} cm max {res.max() * 100:.1f} cm")
    # body params: deltas from frame 0 on the canonical, smoothed
    vb = d["model_params"][:, BODY]
    vb = gaussian_filter1d(vb, a.smooth, axis=0, mode="nearest") if a.smooth > 0 else vb
    b0 = body.body0[0].cpu().numpy()
    bp = b0[None] + (vb - vb[:1])
    # root: pelvis rotation / position relative to frame 0, in world
    Rp = d["global_rots"][:, JOINT["pelvis"]]
    Rw = np.einsum("ij,tjk->tik", R, Rp)
    dR = np.einsum("tij,kj->tik", Rw, Rw[0])                  # Rw(t) Rw(0)^T
    pel = s * Jv[:, JOINT["pelvis"]] @ R.T + t
    dp = pel - pel[:1]
    if a.smooth > 0:
        dp = gaussian_filter1d(dp, a.smooth * 1.5, axis=0, mode="nearest")
        rv = Rotation.from_matrix(dR).as_rotvec()
        dR = Rotation.from_rotvec(gaussian_filter1d(rv, a.smooth, axis=0, mode="nearest")).as_matrix()
    ex = body.expr[0].cpu().numpy()[None].repeat(T, 0)
    # resample to the output rate
    t_in = np.arange(T) / a.fps_in
    t_out = np.arange(int(round(t_in[-1] * a.fps_out)) + 1) / a.fps_out
    bp = resample(bp, t_in, t_out); dp = resample(dp, t_in, t_out); ex = resample(ex, t_in, t_out)
    dR = Slerp(t_in, Rotation.from_matrix(dR))(t_out).as_matrix()
    To = len(t_out)
    c0 = J0[JOINT["pelvis"]]
    m = MI.Motion({"body_params": bp.astype(np.float32), "expr": ex.astype(np.float32), "fps": a.fps_out,
                   "root_R": dR.astype(np.float32), "root_t": dp.astype(np.float32), "root_c": c0.astype(np.float32)})
    if not a.no_ik:
        sol = contact_cleanup(body, m, iters=a.iters)
        np.savez(a.out, **sol)
    else:
        np.savez(a.out, **m.save_fields(), root_world=dp.astype(np.float32))
    print(f"retarget_video: {To} frames at {a.fps_out:g} fps -> {a.out}")


if __name__ == "__main__":
    main()
