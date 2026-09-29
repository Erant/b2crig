"""Make a posed export the subject's new canonical: a new subject whose rest pose is the export's pose.

    .venv/bin/python tools/recanonicalize.py work/<subject> CLIP FRAME NEW_NAME

Input: <subject>/../<NEW_NAME>/run/ply/posed_raw.ply (+ .frame.f32) from
`b2ctrain render --cage <clip>/cage.b2ccage --export-posed ... --export-frame FRAME`.
Writes work/<NEW_NAME>/: mhr.npz (the clip's body params of FRAME as the canonical pose; verts/joints posed),
run/ply/scene.ply (SH bands 1-3 rotated by each splat's triangle rotation, the original ply's b2c.* header comments),
run/ply/front.png and run/colmap (links), expr_motion.npy (copied); registers NEW_NAME in work/subjects.json.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2crig.rig.mhr import BODY, MHRBody  # noqa: E402

S = Path(sys.argv[1]); clip = S / "clips" / sys.argv[2]; frame = int(sys.argv[3]); new = S.parent / sys.argv[4]
subs = json.loads((S.parent / "subjects.json").read_text()); run = Path(subs[S.name])
nrun = new / "run"; (nrun / "ply").mkdir(parents=True, exist_ok=True)

# --- SH rotation: f'(d) = f(R^T d), per band, by least squares over sample directions
C1 = 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658,
      1.445305721320277, -0.5900435899266435]


def basis(d):
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    xx, yy, zz = x * x, y * y, z * z
    b1 = np.stack([-C1 * y, C1 * z, -C1 * x], -1)
    b2 = np.stack([C2[0] * x * y, C2[1] * y * z, C2[2] * (2 * zz - xx - yy), C2[3] * x * z, C2[4] * (xx - yy)], -1)
    b3 = np.stack([C3[0] * y * (3 * xx - yy), C3[1] * x * y * z, C3[2] * y * (4 * zz - xx - yy),
                   C3[3] * z * (2 * zz - 3 * xx - 3 * yy), C3[4] * x * (4 * zz - xx - yy), C3[5] * z * (xx - yy),
                   C3[6] * x * (xx - 3 * yy)], -1)
    return [b1, b2, b3]


rng = np.random.default_rng(0)
D = rng.normal(size=(96, 3)); D /= np.linalg.norm(D, axis=1, keepdims=True)
P = [np.linalg.pinv(b) for b in basis(D)]                     # [2l+1, Nd]
ply = PlyData.read(str(new / "run" / "ply" / "posed_raw.ply")); v = ply["vertex"].data
n = len(v)
q = np.fromfile(str(new / "run" / "ply" / "posed_raw.ply") + ".frame.f32", np.float32).reshape(n, 4)   # w, x, y, z
w, x, y, z = q.T
R = np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
              np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
              np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)   # [n, 3, 3]
rest = [f"f_rest_{k}" for k in range(45)]
F = np.stack([v[f] for f in rest], 1).reshape(n, 3, 15)          # channel-major: [n, rgb, 15 coeffs]
out = F.copy()
bands = [(0, 3), (3, 8), (8, 15)]
for i0 in range(0, n, 50_000):
    sl = slice(i0, min(i0 + 50_000, n))
    RtD = np.einsum("nji,dj->ndi", R[sl], D)                     # R^T d for every sample
    Bs = basis(RtD)
    for (a, b), Pb, Bb in zip(bands, P, Bs):
        M = np.einsum("kd,ndj->nkj", Pb, Bb)                       # [n, 2l+1, 2l+1]
        out[sl, :, a:b] = np.einsum("nkj,ncj->nck", M, F[sl, :, a:b])
out = out.reshape(n, 45)
nv = v.copy()
for k, f in enumerate(rest):
    nv[f] = out[:, k]
orig = PlyData.read(str(run / "ply" / "scene.ply"))
comments = [c for c in orig.comments]
PlyData([PlyElement.describe(nv, "vertex")], text=False, comments=comments).write(str(nrun / "ply" / "scene.ply"))
shutil.copy(run / "ply" / "front.png", nrun / "ply" / "front.png")
if not (nrun / "colmap").exists():
    (nrun / "colmap").symlink_to(run / "colmap")

# --- the body: FRAME's params as the canonical pose
d = dict(np.load(S / "mhr.npz"))
body = MHRBody(S / "mhr.npz")
params = torch.as_tensor(np.load(clip / "motion.npz")["body_params"][frame][None], device=body.device)
mp = body.model_params(params)
posed = body.forward(mp)
d["model_params"] = mp[0].cpu().numpy()
d["verts_world"] = posed.verts[0].cpu().numpy(); d["joints_world"] = posed.joints[0].cpu().numpy()
np.savez(new / "mhr.npz", **d)
for f in ("expr_motion.npy", "face_ref.png"):
    if (S / f).exists():
        shutil.copy(S / f, new / f)
np.save(new / "old_canonical_params.npy", body.body0.cpu().numpy())
subs[new.name] = str(nrun.resolve()); (S.parent / "subjects.json").write_text(json.dumps(subs, indent=1))
print(f"recanonicalize: {new} ({n} splats, canonical = {clip.name} frame {frame})")
