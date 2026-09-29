"""Build a subject's hair shell (rig/hair.py) -> <subject>/cages/hair_offset.npz (replaces the offset hair layer).

    .venv/bin/python tools/build_hair.py work/<subject>
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2crig.io.splat import load  # noqa: E402
from b2crig.rig import hair as H  # noqa: E402
from b2crig.rig.mhr import MHRBody  # noqa: E402

S = Path(sys.argv[1])
run = Path(json.loads((S.parent / "subjects.json").read_text())[S.name])
s = load(run / "ply" / "scene.ply")
m = (s.seg_label == 4) & (s.opacity > 0.2) & (s.seg_conf > 0.5)
body = MHRBody(S / "mhr.npz", device="cpu")
sh = H.build_shell(s.xyz[m].astype(float), body)
H.save_shell(S, sh, body)
d, _ = cKDTree(sh["V"]).query(s.xyz[m])
print(f"hair shell: {len(sh['V'])} verts {len(sh['F'])} faces, theta max {sh['theta'].max():.0f} deg, r {sh['r'].min():.3f}..{sh['r'].max():.3f}; "
      f"hair splats to shell vertex p50 {np.median(d) * 100:.1f} p90 {np.percentile(d, 90) * 100:.1f} cm")
p = H.HairParams()
L = H.hair_length(sh["theta"], sh["r"], p)
print(f"free hair length max {L.max() * 100:.1f} cm; free verts {(L > 0).sum()}")
# a mesh to look at
V, F = sh["V"], sh["F"]
hdr = (f"ply\nformat ascii 1.0\nelement vertex {len(V)}\nproperty float x\nproperty float y\nproperty float z\n"
       f"element face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n")
(S / "cages" / "hair_shell.ply").write_text(hdr + "".join(f"{a} {b} {c}\n" for a, b, c in V) + "".join(f"3 {a} {b} {c}\n" for a, b, c in F))
