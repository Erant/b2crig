"""Refine a subject's canonical MHR pose to its own splat: the capture-time body fit can be off at a joint (2bccb7's
right elbow: the rig bent 60 deg, the splat's arm 75), and LBS then leaves the difference in every pose as a kink.

    .venv/bin/python tools/refine_fit.py work/<subject> [--parts r_arm,l_arm,r_leg,l_leg] [--iters 300] [--dry]

Per part, the splats of its Sapiens classes (opaque, confident) are pulled to the posed MHR surface of the same part
(vertices whose dominant skinning joint is in the part's chain): ICP with point-to-plane distances, correspondences
re-found every 20 steps, a truncated (Geman-McClure-like) loss so crease / garment splats do not drag the limb, and an
L2 pull of the rotation DOFs towards the capture fit. Only the part chains' rotation DOFs move (clavicle, shoulder,
elbow, wrist / hip, knee, ankle). Writes <subject>/mhr.npz (the old one kept as mhr_capture.npz) with the new
model_params / verts_world / joints_world, and prints per-part distances before -> after. Afterwards rebuild the
subject's layers, stable faces and clip cages (they depend on the canonical mesh).
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json  # noqa: E402

from b2crig.io.splat import load  # noqa: E402
from b2crig.rig.mhr import MHRBody  # noqa: E402
from b2crig.rig.skeleton import ROT  # noqa: E402

PARTS = {   # classes (weight), skeleton chain (rotation DOFs), skinning joints owning the part's vertices
    "r_arm": ({20: 1.0, 16: 1.0, 15: 0.5}, ["r_clavicle", "r_shoulder", "r_elbow", "r_wrist"], "r_arm"),
    "l_arm": ({11: 1.0, 7: 1.0, 6: 0.5}, ["l_clavicle", "l_shoulder", "l_elbow", "l_wrist"], "l_arm"),
    "r_leg": ({21: 1.0, 17: 1.0, 19: 1.0}, ["r_hip", "r_knee", "r_ankle"], "r_leg"),
    "l_leg": ({12: 1.0, 8: 1.0, 10: 1.0}, ["l_hip", "l_knee", "l_ankle"], "l_leg"),
}


def descendants(parents: np.ndarray, root: int) -> set:
    out, grow = {root}, True
    while grow:
        grow = False
        for j, p in enumerate(parents):
            if p in out and j not in out:
                out.add(j); grow = True
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("subject", type=Path)
    ap.add_argument("--parts", default="r_arm,l_arm,r_leg,l_leg")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--trunc", type=float, default=0.02, help="m: residual scale of the robust loss")
    ap.add_argument("--reg", type=float, default=1e-4, help="L2 pull of the moved DOFs towards the capture fit (rad^-2)")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    S = a.subject
    src = S / "mhr_capture.npz" if (S / "mhr_capture.npz").exists() else S / "mhr.npz"
    body = MHRBody(src, device="cuda")
    run = Path(json.loads((S.parent / "subjects.json").read_text())[S.name])
    s = load(run / "ply" / "scene.ply")
    ok = (s.opacity > 0.5) & (s.seg_conf > 0.5)
    from b2crig.rig.skeleton import JOINT
    sv, sj, sw = body.skin
    nv = len(body.verts_canon)
    W = np.zeros((nv, len(body.parents)), np.float32)
    np.add.at(W, (sv, sj), sw)
    dom = W.argmax(1)
    F = torch.as_tensor(body.faces.astype(np.int64), device="cuda")

    parts = a.parts.split(",")
    idx, pts, wts, vsets = [], [], [], []
    for p in parts:
        cls, chain, _ = PARTS[p]
        m = ok & np.isin(s.seg_label, list(cls))
        pts.append(s.xyz[m]); wts.append(np.array([cls[c] for c in s.seg_label[m]], np.float32))
        joints = descendants(body.parents, JOINT[chain[1]])   # shoulder / hip down (the clavicle moves the shoulder)
        vsets.append(np.where(np.isin(dom, list(joints)))[0])
        for n in chain:
            idx += [i for i in ROT[n] if i is not None]
    idx = torch.as_tensor(sorted(set(idx)), device="cuda")
    mp0 = body.model_params0.clone()
    delta = torch.zeros(len(idx), device="cuda", requires_grad=True)
    opt = torch.optim.Adam([delta], lr=a.lr)
    P = [torch.as_tensor(x, dtype=torch.float32, device="cuda") for x in pts]
    Wt = [torch.as_tensor(x, device="cuda") for x in wts]

    def posed(dl):
        mp = mp0.clone(); mp[0, idx] = mp0[0, idx] + dl
        v, _ = body.mhr(body.shape, mp, body.expr)
        return ((v / 100) @ body._A.T + body._t)[0]

    def normals(V):
        n = torch.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]], dim=-1)
        vn = torch.zeros_like(V).index_add_(0, F[:, 0], n).index_add_(0, F[:, 1], n).index_add_(0, F[:, 2], n)
        return vn / vn.norm(dim=-1, keepdim=True).clamp_min(1e-9)

    def residuals(V, corr):
        N = normals(V)
        return [((Pk - V[ck]) * N[ck]).sum(-1) for Pk, ck in zip(P, corr)]

    def find_corr(V):
        Vn = V.detach().cpu().numpy()
        return [torch.as_tensor(vs[cKDTree(Vn[vs]).query(x)[1]], device="cuda") for x, vs in zip(pts, vsets)]   # global vertex ids

    def report(V, tag):
        corr = find_corr(V); r = residuals(V, corr)
        print(tag, "  ".join(f"{p}: med {r_.abs().median().item() * 1000:.1f} mm p90 {r_.abs().quantile(0.9).item() * 1000:.1f}"
                             for p, r_ in zip(parts, r)), flush=True)

    with torch.no_grad():
        report(posed(torch.zeros_like(delta)), "before")
    corr = None
    for it in range(a.iters):
        V = posed(delta)
        if it % 20 == 0:
            corr = find_corr(V)
        c2 = a.trunc ** 2
        loss = sum((wk * (rk ** 2 / (rk ** 2 + c2))).mean() for rk, wk in zip(residuals(V, corr), Wt)) + a.reg * (delta ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        V = posed(delta)
        report(V, "after ")
    names = {i: f"{n}.{'xyz'[k]}" for n, t in ROT.items() for k, i in enumerate(t) if i is not None}
    d = delta.detach().cpu().numpy()
    print("DOF changes (deg):", ", ".join(f"{names[int(i)]} {np.degrees(x):+.1f}" for i, x in zip(idx.cpu().numpy(), d) if abs(x) > np.radians(0.5)))
    if a.dry:
        return
    if not (S / "mhr_capture.npz").exists():
        shutil.copy(S / "mhr.npz", S / "mhr_capture.npz")
    dd = dict(np.load(src))
    mp = mp0.clone(); mp[0, idx] += delta.detach()
    dd["model_params"] = mp[0].cpu().numpy().astype(dd["model_params"].dtype)
    pz = body.forward(mp)
    dd["verts_world"] = pz.verts[0].cpu().numpy(); dd["joints_world"] = pz.joints[0].cpu().numpy()
    np.savez(S / "mhr.npz", **dd)
    print(f"refine_fit: wrote {S / 'mhr.npz'} (capture fit kept in mhr_capture.npz)")


if __name__ == "__main__":
    main()
