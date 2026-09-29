"""WAN's depth change per cage vertex from Sapiens2 pointmaps, and whether other views' fits predict it.

    .venv/bin/python tools/pointmap_check.py work/<subject> VIEW_CLIP [FIT=CLIP_OR_DIR ...]

Per frame of VIEW_CLIP: project the cage vertices it sees into its camera; fit z_true = s * z_pm + b per image (the
head predicts its scale per image) on the control (LBS) render and on the WAN frame; the control residual is the
noise floor, the difference dz is WAN's depth change along this view's rays. Then, for each FIT (a clip name
or a fit dir), correlate its displacement's component along this view's rays with dz on the vertices seen here.
Needs <clip>/pointmap_{wan,control}.npy (tools/pointmap_clip.py).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2crig.evaluate import read_cage  # noqa: E402
from b2crig.model.data import layered_pair  # noqa: E402

S = Path(sys.argv[1]); clip = S / "clips" / sys.argv[2]
fit, cg = layered_pair(clip)
cage = read_cage(clip / cg); T, V = cage.posed.shape[:2]
vis = np.fromfile(clip / fit / "vis.f32", np.float32).reshape(T, V)
cams = json.loads((clip / "cameras.json").read_text())
W, H = cams["width"], cams["height"]
pw = np.load(clip / "pointmap_wan.npy", mmap_mode="r"); pc = np.load(clip / "pointmap_control.npy", mmap_mode="r")


def load_fit(spec):
    p = S / "clips" / spec
    if p.exists():
        f, _ = layered_pair(p); p = p / f
    else:
        p = Path(spec)
    return np.fromfile(p / "delta.f32", np.float32).reshape(T, V, 3)


fits = {s.split("=")[0]: load_fit(s.split("=", 1)[1]) for s in sys.argv[3:]}
dz_all, floor, idx, ray_all = [], [], [], []
for t in range(T):
    c = cams["cameras"][t]; R = np.array(c["rotation"]); pos = np.array(c["position"])
    q = (cage.posed[t] - pos) @ R                      # right, up, back
    x, y, z = q[:, 0], -q[:, 1], -q[:, 2]              # OpenCV
    u = np.round(c["fx"] * x / z + c["cx"]).astype(int); v = np.round(c["fy"] * y / z + c["cy"]).astype(int)
    m = (vis[t] > 1.0) & (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    zc = pc[t, v[m], u[m], 2].astype(float); zw = pw[t, v[m], u[m], 2].astype(float)
    ok = np.isfinite(zc) & np.isfinite(zw) & (zc > 0) & (zw > 0)
    zt = z[m][ok]

    def calib(zp):
        """Robust z_true = s * zp + b (the head predicts its scale per image, so each image gets its own)."""
        A = np.stack([zp, np.ones(len(zp))], 1); k = np.ones(len(zp), bool)
        for _ in range(4):
            sb, *_ = np.linalg.lstsq(A[k], zt[k], rcond=None)
            r = zt - A @ sb; k = np.abs(r) < 3 * np.median(np.abs(r)) + 1e-6
        return A @ sb, r

    zc_m, rc = calib(zc[ok]); zw_m, rw = calib(zw[ok])
    floor.append(np.median(np.abs(rc)))
    dz = zw_m - zc_m                                     # + = WAN surface further from the camera than LBS
    vi = np.nonzero(m)[0][ok]
    ray = (cage.posed[t, vi] - pos); ray /= np.linalg.norm(ray, axis=1, keepdims=True)
    dz_all.append(dz); idx.append((t, vi)); ray_all.append(ray)
dz = np.concatenate(dz_all)
print(f"{clip.name}: {len(dz)} vertex-frames; pointmap noise floor (median |resid| on the LBS render) "
      f"{np.median(floor) * 1000:.1f} mm; WAN depth change |dz| median {np.median(np.abs(dz)) * 1000:.1f} mm, "
      f"p90 {np.percentile(np.abs(dz), 90) * 1000:.1f} mm")
clipv = np.abs(dz) < np.percentile(np.abs(dz), 99)
for name, d in fits.items():
    comp = np.concatenate([(d[t, vi] * ray).sum(1) for (t, vi), ray in zip(idx, ray_all)])
    cc = np.corrcoef(comp[clipv], dz[clipv])[0, 1]
    print(f"  {name:14s} along-ray component |.| median {np.median(np.abs(comp)) * 1000:.1f} mm, corr with dz {cc:+.2f}")
