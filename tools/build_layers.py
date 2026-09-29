"""Build a subject's offset garment/hair layers (rig/cage.py:offset_layer).

    .venv/bin/python tools/build_layers.py work/<subject>     (run dir from work/subjects.json)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2crig.io.splat import load  # noqa: E402
from b2crig.model import data as D  # noqa: E402
from b2crig.rig import cage as CG  # noqa: E402
from b2crig.rig.mhr import MHRBody  # noqa: E402

S = Path(sys.argv[1])
run = Path(json.loads((S.parent / "subjects.json").read_text())[S.name])
s = load(run / "ply" / "scene.ply"); ok = (s.opacity > 0.2) & (s.seg_conf > 0.5)
d = np.load(S / "mhr.npz"); BV = d["verts_world"].astype(float); BF = d["faces"].astype(np.int64)
(S / "cages").mkdir(exist_ok=True)
body = MHRBody(S / "mhr.npz")
dom = np.array(D.NAMED)[D.named_influence(body, np.arange(len(BV))).argmax(1)]
from b2crig.rig.layered import layer_classes  # noqa: E402
for f in (S / "cages").glob("*_offset.npz"):   # layers the subject no longer has must not linger
    if f.stem[:-7] not in layer_classes(S):
        f.unlink()
for name, cls in layer_classes(S).items():
    allowed = ~np.isin(dom, CG.LAYER_EXCLUDE.get(name, ()))
    V, F, bi = CG.offset_layer(BV, BF, s.xyz.astype(float), s.seg_label, ok, cls, allowed=allowed)
    if len(F) == 0:
        print(f"{name}: empty, skipped"); continue
    off = np.linalg.norm(V - BV[bi], axis=1)
    np.savez(S / "cages" / f"{name}_offset.npz", V=V.astype(np.float32), F=F.astype(np.int32), body_index=bi)
    print(f"{name}: {len(V)} verts {len(F)} faces, offset cm p50 {np.median(off) * 100:.2f} p90 {np.percentile(off, 90) * 100:.2f} max {off.max() * 100:.1f}")
