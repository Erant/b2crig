"""Re-fit a WAN clip's performance: the pose WAN actually drew (SAM-3D-Body, tools/video_fit.py) instead of the pose
the clip was built from -> CLIP/refit_motion.npz and CLIP/cage_refit.b2ccage (the splat posed like the WAN frames).

    .venv/bin/python tools/refit_wan.py CLIP_DIR [--iters 400] [--no-cleanup]

For first-last-frame clips (clips.build control="flf") WAN only loosely follows the skeleton drawing; the frames it
keeps (0 and the last) are exactly our pose, so they calibrate SAM-3D-Body's bias: per joint, the world-orientation
and bone-direction offsets between SAM's fit and our pose at the two kept frames, blended linearly in between, and the
pelvis position offset likewise (the pelvis is placed on SAM's image ray at our motion's depth). SAM's camera-frame joints go to world through the clip's own cameras (OpenCV ->
OpenGL); its global_rots are in MHR's raw frame and take the flip first. IK (motion/retarget.ik_to_targets) starts from the clip's motion; then the foot-contact clean-up.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2crig import b2ctrain as B  # noqa: E402
from b2crig.motion import io as MI  # noqa: E402
from b2crig.motion.retarget import contact_cleanup, ik_to_targets  # noqa: E402
from b2crig.rig import layered  # noqa: E402
from b2crig.rig.mhr import MHRBody  # noqa: E402
from b2crig.rig.skeleton import JOINT  # noqa: E402

CV2GL = np.diag([1.0, -1.0, -1.0])
TOE = {"r": 24, "l": 8}
HAND = {"r": (42, 52, 44, 56), "l": (78, 88, 80, 92)}   # wrist, middle MCP, pinky MCP, index MCP


def bones() -> list[tuple[int, int, float]]:
    j = JOINT
    out = [(j["pelvis"], j["spine1"], 1.0), (j["spine1"], j["spine3"], 1.0), (j["spine3"], j["neck"], 1.0),
           (j["neck"], j["head"], 1.0), (j["r_hip"], j["l_hip"], 2.0), (j["r_shoulder"], j["l_shoulder"], 2.0)]
    for s in ("r", "l"):
        w, m, p, i = HAND[s]
        out += [(j[f"{s}_clavicle"], j[f"{s}_shoulder"], 0.5), (j[f"{s}_shoulder"], j[f"{s}_elbow"], 2.0),
                (j[f"{s}_elbow"], j[f"{s}_wrist"], 2.0), (w, m, 1.0), (p, i, 1.0),
                (j[f"{s}_hip"], j[f"{s}_knee"], 2.0), (j[f"{s}_knee"], j[f"{s}_ankle"], 2.0), (j[f"{s}_ankle"], TOE[s], 1.0)]
    return out


ORIENT = [(JOINT["pelvis"], 2.0), (JOINT["spine3"], 1.0), (JOINT["head"], 1.0)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", type=Path)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--cleanup-iters", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    clip = a.clip
    S = clip.parents[1]
    body = MHRBody(S / "mhr.npz", device=a.device)
    m = MI.load(clip / "motion.npz")
    T = len(m)
    P = MI.pose(body, m, 0, T)
    Jm, Rm = P.joints.cpu().numpy(), P.rots.cpu().numpy()
    vf = np.load(clip / "video_fit.npz")
    cams = json.loads((clip / "cameras.json").read_text())["cameras"]
    C = np.stack([np.asarray(c["rotation"]) @ CV2GL for c in cams])
    pos = np.stack([np.asarray(c["position"]) for c in cams])
    X = np.einsum("tij,tkj->tki", C, vf["joints"]) + pos[:, None]
    G = np.einsum("tij,tkjl->tkil", C @ CV2GL, vf["global_rots"])   # the rotations are in MHR's raw frame (the joints
                                                                     # are flipped to the camera's): flip them too
    al = (np.arange(T) / (T - 1))[:, None]
    ends = (0, T - 1)
    # bone directions, bias-corrected at the kept frames
    bl = bones()
    ia = np.array([b[0] for b in bl]); ib = np.array([b[1] for b in bl])
    unit = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
    dS, dM = unit(X[:, ib] - X[:, ia]), unit(Jm[:, ib] - Jm[:, ia])
    corr = [dM[e] - dS[e] for e in ends]
    dT = unit(dS + (1 - al)[:, :, None] * corr[0] + al[:, :, None] * corr[1])
    # orientations: G(t) B(t), B slerped between the kept frames' offsets G^T R_ours
    oj = [o[0] for o in ORIENT]
    Rt = np.zeros((T, len(oj), 3, 3))
    for k, j in enumerate(oj):
        Bs = Rotation.from_matrix(np.stack([G[e, j].T @ Rm[e, j] for e in ends]))
        Bt = Slerp([0.0, 1.0], Bs)(al[:, 0]).as_matrix()
        Rt[:, k] = G[:, j] @ Bt
    # pelvis path: on SAM's camera ray through its pelvis (the image position is what a single view measures well), at
    # the depth our own motion has there (SAM's depth is not); then the kept frames' residual offsets, blended
    pj = JOINT["pelvis"]
    pc = vf["joints"][:, pj]
    zm = np.einsum("tji,tj->ti", C, Jm[:, pj] - pos)[:, 2]           # our pelvis in camera coords (OpenCV z)
    Xp = np.einsum("tij,tj->ti", C, pc * (zm / pc[:, 2])[:, None]) + pos
    off = [Jm[e, pj] - Xp[e] for e in ends]
    pel_t = Xp + (1 - al) * off[0] + al * off[1]
    err0 = np.degrees(np.arccos(np.clip((dT * dM).sum(-1), -1, 1)))
    print(f"refit_wan: {T} frames; WAN pose vs the clip's own: bone dir diff mean {err0.mean():.1f} deg "
          f"(p95 {np.percentile(err0, 95):.1f}), pelvis {np.linalg.norm(pel_t - Jm[:, pj], axis=1).mean() * 100:.1f} cm")
    dev = body.device
    w0 = Rotation.from_matrix(m.root_R).as_rotvec()
    x0 = m.body_params - body.body0[0].cpu().numpy()[None]
    mo = ik_to_targets(body, torch.as_tensor(dT, dtype=torch.float32, device=dev),
                       (torch.as_tensor(ia, device=dev), torch.as_tensor(ib, device=dev),
                        torch.as_tensor([b[2] for b in bl], dtype=torch.float32, device=dev)),
                       torch.as_tensor(Rt, dtype=torch.float32, device=dev),
                       (oj, torch.as_tensor([o[1] for o in ORIENT], dtype=torch.float32, device=dev)),
                       pel_t, w0, m.fps, a.iters, x0=x0)
    mo.expr = m.expr
    sol = contact_cleanup(body, mo, iters=a.cleanup_iters) if a.cleanup_iters > 0 else {**mo.save_fields(), "root_world": mo.root_t}
    np.savez(clip / "refit_motion.npz", **sol)
    mf = MI.Motion(dict(np.load(clip / "refit_motion.npz")))
    lay = layered.load_layers(S)
    posed = layered.pose_motion(body, mf, lay)
    names = [c["name"] for c in cams]
    B.write_cage(clip / "cage_refit.b2ccage", layered.cage_layers(body, lay), posed, names)
    print(f"refit_wan: -> {clip / 'refit_motion.npz'}, {clip / 'cage_refit.b2ccage'}")


if __name__ == "__main__":
    main()
