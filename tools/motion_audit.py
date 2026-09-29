"""Audit retargeted motions for poses a body cannot make: mesh damage per region, joint-limit excess, capsule
self-collision and per-frame pops. A regression check for the retargeting IK (motion/retarget.py) and a finder for
frames worth looking at.

    .venv/bin/python tools/motion_audit.py work/<subject> MOTION.npz ... [--json OUT.json] [--sheet OUT.png --top 24]

Per frame, on the subject's MHR mesh (not the splat cage):
  stretch  area share of a region whose triangles stretch/squash beyond 1.6x against MHR's zero pose
  crease   area share of a region's triangles on a NEW sharp crease: an edge whose dihedral opens past 110 deg where
           the zero pose's is under 40 deg (a shoulder folded into the back, an elbow bent inside out)
  limit    radians beyond rom.RANGES (+ margin) summed over the DOFs (the ROM library's textbook ranges; the margin
           keeps it about gross violations, not the range bounds' exact values)
  collide  first capsule pair interpenetrating (rom.collides)
  pop      largest per-DOF step in rad/s (a flip or a jump the smoothness term lost)
  track    (SMPL-X retargets: the npz's `source`) per-bone angle between the final motion and retarget_smplx's IK
           targets, and the pelvis/chest/head orientation error: where the pose leaves the mocap, the IK (or the
           contact clean-up) made it up
Stretch/crease thresholds are per region, calibrated on the synthetic ROM library (rom.py: textbook-limited poses on
the same subject): a region is flagged when its share exceeds max(3%, the ROM poses' 97th percentile + 3%), i.e. worse
than what an anatomically limited pose does to it under LBS. A frame is FLAGGED when any region is, limit > 0.3 rad,
it collides, or pop > 15 rad/s.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from b2crig.motion import io as MI  # noqa: E402
from b2crig.motion import rom  # noqa: E402
from b2crig.rig.mhr import MHRBody  # noqa: E402
from b2crig.rig.skeleton import JOINT, ROT, body_index  # noqa: E402

REGION_OF = {1: "hips", 34: "abdomen", 35: "abdomen", 36: "chest", 37: "chest", 110: "neck", 113: "head",
             **{JOINT[f"{s}_{j}"]: f"{s}_{r}" for s in "rl" for j, r in (("clavicle", "shoulder"), ("shoulder", "uarm"),
                ("elbow", "farm"), ("wrist", "hand"), ("hip", "thigh"), ("knee", "shin"), ("ankle", "foot"))}}
TH_STRETCH, TH_SHARE, TH_LIMIT, TH_POP, LIMIT_MARGIN = math.log(1.6), 0.03, 0.3, 15.0, 0.15


def regions(body: MHRBody) -> tuple[np.ndarray, list[str]]:
    """Per-face region name index (dominant skinning joint of the face's vertices, walked up to a named joint)."""
    sv, sj, sw = (np.asarray(x) for x in body.skin)
    nv = len(body.verts_canon)
    dom, best = np.zeros(nv, np.int64), np.zeros(nv)
    np.maximum.at(best, sv, sw)
    dom[sv[sw >= best[sv]]] = sj[sw >= best[sv]]
    par = body.parents
    def up(j):
        while j not in REGION_OF and j > 0:
            j = int(par[j])
        return REGION_OF.get(j, "hips")
    vreg = np.array([up(int(j)) for j in dom])
    names = sorted(set(REGION_OF.values()))
    fr = vreg[body.faces]   # majority of the three
    freg = np.where(fr[:, 1] == fr[:, 2], fr[:, 1], fr[:, 0])
    return np.array([names.index(r) for r in freg]), names


def edge_pairs(F: np.ndarray) -> np.ndarray:
    """[E, 2] face index pairs sharing an edge."""
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); e.sort(1)
    fid = np.tile(np.arange(len(F)), 3)
    key = e[:, 0] * (F.max() + 1) + e[:, 1]
    o = np.argsort(key); k, f = key[o], fid[o]
    same = k[1:] == k[:-1]
    return np.stack([f[:-1][same], f[1:][same]], 1)


class Auditor:
    def __init__(self, body: MHRBody):
        self.body = body; dev = body.device
        self.F = torch.as_tensor(body.faces.astype(np.int64), device=dev)
        self.reg, self.names = regions(body)
        self.reg_t = torch.as_tensor(self.reg, device=dev)
        self.EP = torch.as_tensor(edge_pairs(body.faces), device=dev)
        with torch.no_grad():   # reference = MHR's neutral zero pose, not the capture canonical (often arms up)
            V0 = body.pose(rom.to_body({}, body.body0)).verts.float()[0]
        E0 = self._frames(V0[None])[0]
        det = torch.linalg.det(E0)
        self.ok = det.abs() > 1e-10
        self.E0inv = torch.linalg.inv(torch.where(self.ok[:, None, None], E0, torch.eye(2, device=dev)))
        self.area = (0.5 * det.abs() * self.ok).float()
        self.reg_area = torch.zeros(len(self.names), device=dev).index_add_(0, self.reg_t, self.area)
        self.dih0 = self._dihedral(V0[None])[0]
        self.radii = rom.capsule_radii(body.verts_canon, body.joints_canon, body.skin)
        self.lim = [(body_index(ROT[k.split(".")[0]][rom.AXES[k.split(".")[1]]]), lo, hi) for k, (lo, hi) in rom.RANGES.items()]
        self.lim_names = list(rom.RANGES)

    def _frames(self, V):
        F = self.F
        e1, e2 = V[:, F[:, 1]] - V[:, F[:, 0]], V[:, F[:, 2]] - V[:, F[:, 0]]
        u = e1 / e1.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        n = torch.linalg.cross(e1, e2); n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        w = torch.linalg.cross(n, u)
        return torch.stack([torch.stack([(e1 * u).sum(-1), (e2 * u).sum(-1)], -1),
                            torch.stack([(e1 * w).sum(-1), (e2 * w).sum(-1)], -1)], -2)

    def _dihedral(self, V):
        F = self.F
        n = torch.linalg.cross(V[:, F[:, 1]] - V[:, F[:, 0]], V[:, F[:, 2]] - V[:, F[:, 0]])
        n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return torch.acos((n[:, self.EP[:, 0]] * n[:, self.EP[:, 1]]).sum(-1).clamp(-1, 1))   # 0 = flat

    @torch.no_grad()
    def frames(self, m: MI.Motion, chunk: int = 32) -> dict:
        T = len(m); R = len(self.names)
        st, cr = np.zeros((T, R), np.float32), np.zeros((T, R), np.float32)
        joints = np.zeros((T, len(self.body.joints_canon), 3), np.float32)
        for i in range(0, T, chunk):
            P = MI.pose(self.body, m, i, min(T, i + chunk))
            V = P.verts.float(); joints[i:i + len(V)] = P.joints.cpu().numpy()
            s = torch.linalg.svdvals(self._frames(V) @ self.E0inv).clamp_min(1e-3).log().abs().amax(-1)
            bad = ((s > TH_STRETCH) & self.ok).float() * self.area
            st[i:i + len(V)] = (torch.zeros(len(V), R, device=V.device).index_add_(1, self.reg_t, bad) / self.reg_area).cpu().numpy()
            dh = self._dihedral(V)
            new = (dh > math.radians(110)) & (self.dih0 < math.radians(40))
            fb = torch.zeros(len(V), len(self.reg), device=V.device)
            fb.index_add_(1, self.EP[:, 0], new.float()); fb.index_add_(1, self.EP[:, 1], new.float())
            cr[i:i + len(V)] = (torch.zeros(len(V), R, device=V.device).index_add_(1, self.reg_t, (fb > 0).float() * self.area)
                                / self.reg_area).cpu().numpy()
        B = m.body_params
        exc = np.stack([np.maximum(0, lo - LIMIT_MARGIN - B[:, j]) + np.maximum(0, B[:, j] - hi - LIMIT_MARGIN)
                        for j, lo, hi in self.lim], 1)
        rot_idx = [body_index(i) for v in ROT.values() for i in v if i is not None]
        dB = np.abs(np.diff(B[:, rot_idx], axis=0)) * m.fps
        pop = np.concatenate([[0.0], dB.max(1)])
        col = [rom.collides(J, self.radii) for J in joints]
        return dict(stretch=st, crease=cr, limit=exc, pop=pop, collide=col)


def rom_motion(body: MHRBody, n_random: int = 1000) -> MI.Motion:
    """The ROM library's poses (named + random, collision-free) as a root-less motion, for calibration."""
    rng = np.random.default_rng(0)
    poses = list(rom.named_poses().values()) + [rom.random_pose(rng) for _ in range(n_random)]
    bp = torch.cat([rom.to_body(p, body.body0) for p in poses]).cpu().numpy()
    return MI.Motion({"body_params": bp, "expr": body.expr.cpu().numpy().repeat(len(bp), 0), "fps": 30.0})


def calibrate(au: "Auditor") -> tuple[np.ndarray, np.ndarray]:
    r = au.frames(rom_motion(au.body))
    ok = np.array([c is None for c in r["collide"]])
    th = lambda x: np.maximum(TH_SHARE, np.percentile(x[ok], 97, axis=0) + TH_SHARE)
    return th(r["stretch"]), th(r["crease"])


def tracking(body: MHRBody, p: Path, m: MI.Motion) -> dict | None:
    d = np.load(p, allow_pickle=True)
    if "source" not in d.files or not str(d["source"]).endswith("stageii.npz"):
        return None
    import retarget_smplx as RS
    from b2crig.motion import smplx as SX
    sk = SX.load_amass(str(d["source"]), fps_out=m.fps, t0=float(d["t0"]), dur=len(m) / m.fps)
    n = min(len(sk.joints), len(m))
    sk = SX.Skel(sk.joints[:n], sk.rots[:n], sk.fps, sk.rest_pelvis_height, sk.rest_joints)
    tg = RS.build_targets(body, sk, verbose=False)
    ma, mb, _ = (x.cpu().numpy() for x in tg["bones"])
    with torch.no_grad():
        P = MI.pose(body, m, 0, n)
    J, Rj = P.joints.cpu().numpy(), P.rots.cpu().numpy()
    dM = J[:, mb] - J[:, ma]; dM /= np.linalg.norm(dM, axis=-1, keepdims=True)
    bone = np.degrees(np.arccos(np.clip((dM * tg["dS"].cpu().numpy()).sum(-1), -1, 1)))
    oj = tg["orient"][0]; Rt = tg["Rt"].cpu().numpy()
    tr = np.einsum("tkij,tkij->tk", Rj[:, oj], Rt)   # trace(Rj^T Rt)
    ori = np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))
    names = [f"{JN.get(int(a), a)}>{JN.get(int(b), b)}" for a, b in zip(ma, mb)]
    return dict(bone=bone, orient=ori, bone_names=names, orient_names=[JN.get(j, j) for j in oj])


JN = {v: k for k, v in JOINT.items()} | {24: "r_toe", 8: "l_toe", 52: "r_mid", 88: "l_mid", 44: "r_pinky", 80: "l_pinky",
                                          56: "r_index", 92: "l_index", 42: "r_hand", 78: "l_hand"}


def print_tracking(tk: dict) -> None:
    b, o = tk["bone"], tk["orient"]
    worst = sorted(range(b.shape[1]), key=lambda k: -np.percentile(b[:, k], 95))[:6]
    print("   track bone deg (mean/p95/max): " + ", ".join(f"{tk['bone_names'][k]} {b[:, k].mean():.0f}/{np.percentile(b[:, k], 95):.0f}/{b[:, k].max():.0f}" for k in worst))
    print("   track orient deg (mean/p95/max): " + ", ".join(f"{n} {o[:, k].mean():.0f}/{np.percentile(o[:, k], 95):.0f}/{o[:, k].max():.0f}" for k, n in enumerate(tk["orient_names"])))


def summarise(name: str, r: dict, names: list[str], lim_names: list[str], th_st: np.ndarray, th_cr: np.ndarray) -> dict:
    st, cr, lim, pop, col = r["stretch"], r["crease"], r["limit"], r["pop"], r["collide"]
    f_st, f_cr = (st > th_st).any(1), (cr > th_cr).any(1)
    f_li, f_po, f_co = lim.sum(1) > TH_LIMIT, pop > TH_POP, np.array([c is not None for c in col])
    flag = f_st | f_cr | f_li | f_po | f_co
    score = np.maximum(st - th_st, 0).max(1) + np.maximum(cr - th_cr, 0).max(1) * 2 + lim.sum(1) * 0.1 + f_co * 0.05 + np.maximum(0, pop - TH_POP) * 0.002
    T = len(pop)
    print(f"{name}: {T} frames, flagged {flag.sum()} ({flag.mean() * 100:.1f}%): stretch {f_st.sum()} crease {f_cr.sum()} "
          f"limit {f_li.sum()} collide {f_co.sum()} pop {f_po.sum()}")
    for key, arr, th in (("stretch", st, th_st), ("crease", cr, th_cr)):
        worst = [(names[k], int((arr[:, k] > th[k]).sum()), float(arr[:, k].max())) for k in range(len(names))]
        worst = [w for w in sorted(worst, key=lambda w: -w[1]) if w[1]][:5]
        if worst:
            print(f"   {key:7s} by region (frames, max share): " + ", ".join(f"{n} {c} {mx:.2f}" for n, c, mx in worst))
    dofs = [(lim_names[k], int((lim[:, k] > 0).sum()), float(lim[:, k].max())) for k in range(lim.shape[1])]
    dofs = [d for d in sorted(dofs, key=lambda d: -d[2]) if d[1]][:6]
    if dofs:
        print("   limit by DOF (frames, max rad over): " + ", ".join(f"{n} {c} {mx:.2f}" for n, c, mx in dofs))
    cols = {}
    for c in col:
        if c:
            cols[c] = cols.get(c, 0) + 1
    if cols:
        print("   collide: " + ", ".join(f"{k} {v}" for k, v in sorted(cols.items(), key=lambda x: -x[1])))
    return dict(frames=T, flagged=int(flag.sum()), score=score.tolist(),
                why=[";".join(filter(None, [
                    ",".join(f"st:{names[k]}" for k in np.nonzero(st[t] > th_st)[0]),
                    ",".join(f"cr:{names[k]}" for k in np.nonzero(cr[t] > th_cr)[0]),
                    ",".join(f"{lim_names[k]}+{lim[t, k]:.2f}" for k in np.nonzero(lim[t] > 0.1)[0]),
                    f"col:{col[t]}" if col[t] else "", f"pop:{pop[t]:.0f}" if pop[t] > TH_POP else ""])) for t in range(T)])


def render(body: MHRBody, m: MI.Motion, t: int, size: int = 260, azs=(0, 90, 180, 270)) -> np.ndarray:
    """Shaded full-body views of frame t (painter's algorithm)."""
    P = MI.pose(body, m, t, t + 1)
    V = P.verts[0].cpu().numpy().astype(np.float64); F = body.faces
    c = (V.max(0) + V.min(0)) / 2; h = (V.max(0) - V.min(0)).max()
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]); n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    out = []
    for az in azs:
        a = math.radians(az); d = np.array([math.sin(a), 0, math.cos(a)])   # camera on +d looking at c
        right = np.array([math.cos(a), 0, -math.sin(a)]); up = np.array([0, 1.0, 0])
        q = V - c; u = (q @ right) / h * size * 0.9 + size / 2; v = -(q @ up) / h * size * 0.9 + size / 2; z = q @ d
        lam = np.abs(n @ (d * 0.7 + up * 0.5 + right * 0.3) / np.linalg.norm([0.7, 0.5, 0.3]))
        im = np.full((size, size, 3), 90, np.uint8)
        for k in np.argsort(z[F].mean(1)):
            col = int(40 + 200 * lam[k])
            cv2.fillConvexPoly(im, np.stack([u[F[k]], v[F[k]]], 1).round().astype(np.int32), (col, col, col), lineType=cv2.LINE_AA)
        out.append(im)
    return np.hstack(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("subject", type=Path)
    ap.add_argument("motions", nargs="+", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--sheet", type=Path, help="contact sheet of the worst frames across all motions")
    ap.add_argument("--top", type=int, default=24)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-track", action="store_true")
    a = ap.parse_args()
    body = MHRBody(a.subject / "mhr.npz", device=a.device)
    au = Auditor(body)
    th_st, th_cr = calibrate(au)
    print("calibrated (ROM p97 + 3%) stretch: " + " ".join(f"{n} {v:.2f}" for n, v in zip(au.names, th_st)))
    print("                          crease:  " + " ".join(f"{n} {v:.2f}" for n, v in zip(au.names, th_cr)))
    res, tot, fl = {}, 0, 0
    for p in a.motions:
        m = MI.load(p)
        res[str(p)] = summarise(p.stem, au.frames(m), au.names, au.lim_names, th_st, th_cr)
        if not a.no_track and (tk := tracking(body, p, m)) is not None:
            print_tracking(tk)
            res[str(p)]["track_bone_p95"] = {n: float(np.percentile(tk["bone"][:, k], 95)) for k, n in enumerate(tk["bone_names"])}
        tot += res[str(p)]["frames"]; fl += res[str(p)]["flagged"]
    print(f"TOTAL: {tot} frames, flagged {fl} ({fl / max(tot, 1) * 100:.1f}%)")
    if a.json:
        a.json.write_text(json.dumps(res))
    if a.sheet:
        cand = sorted(((s, p, t) for p, r in res.items() for t, s in enumerate(r["score"])), reverse=True)
        picked, seen = [], {}
        for s, p, t in cand:   # at most one frame per 10-frame window per motion
            if all(abs(t - u) >= 10 for u in seen.get(p, [])):
                picked.append((s, p, t)); seen.setdefault(p, []).append(t)
            if len(picked) == a.top:
                break
        rows = []
        for s, p, t in picked:
            im = render(body, MI.load(Path(p)), t)
            cv2.putText(im, f"{Path(p).stem} f{t} {res[p]['why'][t][:90]}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 120), 1)
            rows.append(im)
        cv2.imwrite(str(a.sheet), np.vstack(rows))
        print(f"sheet: {a.sheet}")


if __name__ == "__main__":
    main()
