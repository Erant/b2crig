"""Leave-one-pose-out test of a static pose -> displacement model on the held-pose targets (b2crig.model.static).

    .venv/bin/python tools/static_lopo.py work/<subject> [--lam 1.0] [--gamma 0.5]

For each held pose p (from targets.npz) predict its local-frame displacement from the other poses by
  const: the count-weighted mean over the other poses,
  krr:   kernel ridge regression (RBF on the joints' parent-relative 6D rotations) per vertex coordinate,
  own:   p's own view-average (the ceiling of this extraction),
  lin:   per-vertex ridge on the parent-relative (R - I) of its 2 most influential named joints plus an unpenalised
         bias (the const), SMPL-pose-blendshape style but local, garment/hair layers only,
  *_lay: the same with the body layer (skin, face, shoes) left at zero - only the garment/hair layers move,
rotate into p's LBS frames, apply it to p's held frames of every clip that holds p, score those frames against the
clip's WAN segmentation (mean IoU per group over the held frames, averaged over poses).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2crig import evaluate  # noqa: E402
from b2crig.model import data as D  # noqa: E402
from b2crig.model.static import segments  # noqa: E402
from b2crig.rig.mhr import MHRBody  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("subject", type=Path)
ap.add_argument("--lam", type=float, default=1.0)
ap.add_argument("--gamma", type=float, default=0.5)
ap.add_argument("--targets", default="targets")
ap.add_argument("--lin-lam", type=float, nargs="+", default=[0.01, 0.1, 1.0])
a = ap.parse_args()
S = a.subject
run = Path(json.loads((S.parent / "subjects.json").read_text())[S.name]); splat = run / "ply" / "scene.ply"
t = np.load(S / "static" / f"{a.targets}.npz")
P, Y, C = t["params"], t["delta_local"], t["count"]
N, V = Y.shape[:2]
body = MHRBody(S / "mhr.npz"); bi = D.cage_body_index(body, S)
posed = body.pose(torch.as_tensor(P, device=body.device))
F = D.skeleton_features(posed.rots.cpu().numpy(), posed.joints.cpu().numpy(), 16.0)[:, :6 * len(D.NAMED)]
Rv = D.vertex_frames(body, posed, bi).cpu().numpy()                     # [N, V, 3, 3]
NB = len(body.verts_canon)
top2 = np.argsort(-D.named_influence(body, bi), 1)[:, :2]                  # [V, 2] named joints per cage vertex
Rrel = D.relative_rotations(posed) - np.eye(3)                                # [N, 21, 3, 3]
Xall = Rrel.reshape(N, len(D.NAMED), 9)[:, top2].reshape(N, -1, 18)            # [N, V, 18]
print(f"{N} poses; mean |static delta| {np.linalg.norm(Y, axis=-1).mean() * 1000:.1f} mm")


def predict(i: int) -> dict:
    tr = [j for j in range(N) if j != i]
    w = C[tr][..., None]
    const = (Y[tr] * w).sum(0) / np.maximum(w.sum(0), 1)
    d2 = ((F[tr][:, None] - F[tr][None]) ** 2).sum(-1); k = np.exp(-a.gamma * ((F[tr] - F[i]) ** 2).sum(-1))
    K = np.exp(-a.gamma * d2)
    Yc = np.where(C[tr][..., None] > 0, Y[tr], const[None]) - const[None]  # unseen -> the constant
    alpha = np.linalg.solve(K + a.lam * np.eye(len(tr)), Yc.reshape(len(tr), -1))
    krr = const + (k @ alpha).reshape(V, 3)
    lay = np.ones((V, 1), np.float32); lay[:NB] = 0                       # garment/hair layers only
    out = {}
    X = Xall[tr]                                                          # [n, V, 18]
    Xc = X - X.mean(0, keepdims=True); Yc2 = np.where(C[tr][..., None] > 0, Y[tr], const[None]) - const[None]
    cw = np.minimum(C[tr], 8.0)                                           # [n, V] view-count weights, capped
    A = np.einsum("nv,nvi,nvj->vij", cw, Xc, Xc); B = np.einsum("nv,nvi,nvk->vik", cw, Xc, Yc2)
    for lam in a.lin_lam:
        coef = np.linalg.solve(A + lam * cw.sum(0)[:, None, None] * np.eye(18) / len(tr) + 1e-9 * np.eye(18), B)
        out[f"lin{lam:g}"] = (const + np.einsum("vi,vik->vk", Xall[i] - X.mean(0), coef)) * lay
    return {**out,"const": const, "krr": krr, "own": Y[i], "const_lay": const * lay, "own_lay": Y[i] * lay}


groups = ["upper", "lower", "hair", "skin", "silhouette"]
agg: dict = {}
for i in range(N):
    preds = predict(i)
    for clip_name in str(t["clips"][i]).split("+"):
        clip = S / "clips" / clip_name
        fit, cage = D.layered_pair(clip)
        T = len(evaluate.read_cage(clip / cage).names)
        params = np.load(clip / "motion.npz")["body_params"]
        seg = next(s for s in segments(params) if np.abs(params[s[0]] - P[i]).max() < 1e-6)
        for name, loc in preds.items():
            world = np.einsum("vij,vj->vi", Rv[i], loc)
            x = np.zeros((T, V, 3), np.float32); x[seg] = world
            name = f"{name}_{a.targets}" if a.targets != "targets" else name
            o = clip / f"lopo_{name}"; o.mkdir(exist_ok=True)
            x.tofile(o / "delta.f32"); np.zeros((T, V), np.float32).tofile(o / "vis.f32")
            evaluate.evaluate(clip, splat, f"lopo_{name}", video=False, cage=cage)
            pf = json.loads((clip / f"eval_lopo_{name}.json").read_text())["per_frame"]
            for key in ("lbs", f"lopo_{name}"):
                agg.setdefault(key.replace("lopo_", ""), []).append([np.nanmean([pf[key][f][g] for f in seg]) for g in groups])
        print(f"pose {i} ({clip_name}): done")
print(f"{'held frames, LOPO':22s}" + "".join(f"{g:>11s}" for g in groups))
for k, v in agg.items():
    print(f"{k:22s}" + "".join(f"{x:11.3f}" for x in np.mean(v, 0)))
