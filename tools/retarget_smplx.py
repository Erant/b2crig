"""Retarget an AMASS SMPL-X motion (e.g. DanceDB) onto a subject's MHR body -> motion npz (root_R form).

    .venv/bin/python tools/retarget_smplx.py work/<subject> AMASS.npz OUT.npz [--t0 S] [--dur S] [--fps 30]
        [--also-fps 16] [--iters 500] [--device cuda]

1. SMPL-X FK (b2crig/motion/smplx.py), Y-up, yawed so the window's mean facing is +Z (the subject's front).
2. Per-frame IK through the differentiable MHR model: the named rotation DOFs, a root rotation (about the canonical
   pelvis, applied after the forward) and a root translation. Proportion-free targets: bone DIRECTIONS (limbs, hand
   direction and palm width line, hip and shoulder lines, neck, torso) plus rest-relative world orientations of the
   pelvis, chest and head (both rigs stand upright facing +Z at rest; the limbs differ, T- vs A-pose, so they only
   get directions). spine3->neck and neck->shoulder are left out: the rigs place those joints differently (constant
   42 / 17 deg offsets). The pelvis path is scaled by the ratio of rest pelvis heights.
3. Foot-contact clean-up (b2crig/motion/retarget.contact_cleanup).
`--also-fps 16` also writes OUT_16.npz, resampled for 16 fps WAN clips.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2crig.motion import io as MI  # noqa: E402
from b2crig.motion import smplx as SX  # noqa: E402
from b2crig.motion.retarget import contact_cleanup, ik_to_targets  # noqa: E402
from b2crig.rig.mhr import MHRBody  # noqa: E402
from b2crig.rig.skeleton import JOINT  # noqa: E402

M_TOE = {"r": 24, "l": 8}
M_HAND = {"r": {"wrist": 42, "middle": 52, "index": 56, "pinky": 44},
          "l": {"wrist": 78, "middle": 88, "index": 92, "pinky": 80}}


def bone_pairs() -> list[tuple[tuple[int, int], tuple[int, int], float]]:
    """((smplx a, b), (mhr a, b), weight): direction of b - a."""
    s, m = SX.J, JOINT
    out = [((s["pelvis"], s["neck"]), (m["pelvis"], m["neck"]), 2.0),
           ((s["neck"], s["head"]), (m["neck"], m["head"]), 1.0),
           ((s["r_hip"], s["l_hip"]), (m["r_hip"], m["l_hip"]), 2.0),
           ((s["r_shoulder"], s["l_shoulder"]), (m["r_shoulder"], m["l_shoulder"]), 2.0),
           ((s["spine3"], s["r_shoulder"]), (m["spine3"], m["r_shoulder"]), 2.0),   # the clavicles (rest-relative)
           ((s["spine3"], s["l_shoulder"]), (m["spine3"], m["l_shoulder"]), 2.0)]
    for f in ("r", "l"):
        out += [((s[f"{f}_hip"], s[f"{f}_knee"]), (m[f"{f}_hip"], m[f"{f}_knee"]), 2.0),
                ((s[f"{f}_knee"], s[f"{f}_ankle"]), (m[f"{f}_knee"], m[f"{f}_ankle"]), 2.0),
                ((s[f"{f}_ankle"], s[f"{f}_foot"]), (m[f"{f}_ankle"], M_TOE[f]), 1.0),
                ((s[f"{f}_shoulder"], s[f"{f}_elbow"]), (m[f"{f}_shoulder"], m[f"{f}_elbow"]), 2.0),
                ((s[f"{f}_elbow"], s[f"{f}_wrist"]), (m[f"{f}_elbow"], m[f"{f}_wrist"]), 2.0),
                ((s[f"{f}_wrist"], s[f"{f}_middle1"]), (M_HAND[f]["wrist"], M_HAND[f]["middle"]), 1.0),
                ((s[f"{f}_pinky1"], s[f"{f}_index1"]), (M_HAND[f]["pinky"], M_HAND[f]["index"]), 1.0)]
    return out


REST_RELATIVE = (0, 1, 4, 5)   # bone_pairs() indices: pelvis->neck, neck->head, spine3->shoulders


def min_rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """[T, 3, 3] rotations taking the unit vector a [3] to each unit b [T, 3] by the shortest arc."""
    v = np.cross(a, b); c = b @ a
    K = np.zeros((len(b), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0], K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -v[:, 2], v[:, 1], v[:, 2], -v[:, 0], -v[:, 1], v[:, 0]
    return np.eye(3) + K + K @ K / (1 + c)[:, None, None]


ORIENT = [(SX.J["pelvis"], JOINT["pelvis"], 2.0), (SX.J["spine3"], JOINT["spine3"], 1.0), (SX.J["head"], JOINT["head"], 1.0)]


def yaw_to_front(R: np.ndarray) -> np.ndarray:
    """Rotation about +Y taking the mean horizontal facing (pelvis z axis) of R [T, 3, 3] to +Z."""
    f = R[:, :, 2].mean(0)
    a = np.arctan2(f[0], f[2])
    return Rotation.from_rotvec([0, -a, 0]).as_matrix()


def solve_ik(body: MHRBody, sk: SX.Skel, iters: int, verbose: bool = True) -> MI.Motion:
    dev = body.device
    T = len(sk.joints)
    Y = yaw_to_front(sk.rots[:, SX.J["pelvis"]])
    Sj = sk.joints @ Y.T
    Sr = np.einsum("ij,tkjl->tkil", Y, sk.rots)
    # pelvis path: horizontal relative to frame 0, vertical above the floor; scaled by rest pelvis heights
    J0 = body.joints_canon
    pel0 = J0[JOINT["pelvis"]]
    floor_m = float(body.verts_canon[:, 1].min())
    k = (pel0[1] - floor_m) / sk.rest_pelvis_height
    ps = Sj[:, SX.J["pelvis"]]
    floor_s = float(np.percentile(Sj[:, [SX.J["l_foot"], SX.J["r_foot"]], 1].min(1), 2)) - 0.02
    pel_t = np.stack([pel0[0] + k * (ps[:, 0] - ps[0, 0]), floor_m + k * (ps[:, 1] - floor_s),
                      pel0[2] + k * (ps[:, 2] - ps[0, 2])], 1)
    print(f"retarget: {T} frames, pelvis scale {k:.3f}, floor {floor_s:.3f} m")
    pairs = bone_pairs()
    sa = np.array([p[0][0] for p in pairs]); sb = np.array([p[0][1] for p in pairs])
    ma = torch.as_tensor([p[1][0] for p in pairs], device=dev); mb = torch.as_tensor([p[1][1] for p in pairs], device=dev)
    wb = torch.as_tensor([p[2] for p in pairs], dtype=torch.float32, device=dev)
    dS = Sj[:, sb] - Sj[:, sa]
    dS = dS / np.linalg.norm(dS, axis=-1, keepdims=True)
    # The torso, neck and clavicle lines are rest-relative: the rigs place spine3 / neck / head / shoulders differently
    # (constant offsets), so world directions cannot be matched, and without a clavicle target the IK swung both
    # clavicles forward ~50 deg to aim the arms (the shoulder line does not see a symmetric protraction): hunched,
    # rolled-forward shoulders and a stretched neck. Target = the MHR REST (zero pose, neutral clavicles; not the
    # capture pose, whose raised arms lift them) direction turned by the source's minimal rotation from its rest
    # direction to the current one.
    with torch.no_grad():
        Jc = body.pose(torch.zeros(1, 130, device=dev)).joints[0].cpu().numpy()
    Rs = sk.rest_joints
    for i in REST_RELATIVE:
        d0 = Rs[sb[i]] - Rs[sa[i]]; d0 = d0 / np.linalg.norm(d0)
        m0 = Jc[int(mb[i])] - Jc[int(ma[i])]; m0 = m0 / np.linalg.norm(m0)
        dS[:, i] = min_rotation(d0, dS[:, i]) @ m0
    dS = torch.as_tensor(dS, dtype=torch.float32, device=dev)
    # rest-relative orientation targets: R_m(t) = R_s(t) R_m(canon)
    Rm0 = body.pose().rots[0]
    oj = [o[1] for o in ORIENT]
    Rt = torch.as_tensor(np.stack([Sr[:, o[0]] for o in ORIENT], 1), dtype=torch.float32, device=dev) @ Rm0[oj][None]
    wo = torch.as_tensor([o[2] for o in ORIENT], dtype=torch.float32, device=dev)

    w0 = Rotation.from_matrix(Sr[:, SX.J["pelvis"]]).as_rotvec()
    return ik_to_targets(body, dS, (ma, mb, wb), Rt, (oj, wo), pel_t, w0, sk.fps, iters, verbose=verbose, neutral=True)


def resample_motion(d: dict, fps_out: float) -> dict:
    fps = float(d["fps"])
    T = len(d["body_params"])
    t_in = np.arange(T) / fps
    t_out = np.arange(int(np.floor(t_in[-1] * fps_out)) + 1) / fps_out
    lin = lambda a: np.stack([np.interp(t_out, t_in, a.reshape(T, -1)[:, j]) for j in range(a.reshape(T, -1).shape[1])], 1).reshape(len(t_out), *a.shape[1:])
    out = {k: v for k, v in d.items()}
    for k in ("body_params", "expr", "root_t", "root_world", "ankle_r", "ankle_l"):
        if k in d:
            out[k] = lin(np.asarray(d[k])).astype(np.float32)
    out["root_R"] = Slerp(t_in, Rotation.from_matrix(d["root_R"]))(t_out).as_matrix().astype(np.float32)
    out["fps"] = fps_out
    out["t"] = t_out
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("subject", type=Path)
    ap.add_argument("amass", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=None)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--also-fps", type=float, default=0.0)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--cleanup-iters", type=int, default=400)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    body = MHRBody(a.subject / "mhr.npz", device=a.device)
    sk = SX.load_amass(a.amass, fps_out=a.fps, t0=a.t0, dur=a.dur)
    m = solve_ik(body, sk, a.iters)
    if a.cleanup_iters > 0:
        sol = contact_cleanup(body, m, iters=a.cleanup_iters)
    else:
        sol = {**m.save_fields(), "root_world": m.root_t}
    sol["source"] = str(a.amass); sol["t0"] = a.t0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **sol)
    print(f"retarget_smplx: {len(sol['body_params'])} frames at {a.fps:g} fps -> {a.out}")
    if a.also_fps:
        o2 = a.out.with_name(a.out.stem + f"_{a.also_fps:g}.npz")
        np.savez(o2, **resample_motion(dict(sol), a.also_fps))
        print(f"  + {o2}")


if __name__ == "__main__":
    main()
