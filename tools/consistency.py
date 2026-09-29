"""Agreement of fitted displacements between clips of the same motion (local vertex frames, co-visible vertices).

    .venv/bin/python tools/consistency.py work/416580 clipA clipB [clipC ...]
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2crig.model import data as D  # noqa: E402
from b2crig.rig.mhr import MHRBody  # noqa: E402

S = Path(sys.argv[1]); names = sys.argv[2:]
body = MHRBody(S / "mhr.npz"); bi = D.cage_body_index(body, S)
cl = {n: D.load_clip(S / "clips" / n, body, *D.layered_pair(S / "clips" / n), bi) for n in names}
nb = len(body.verts_canon)
for a, b in combinations(names, 2):
    A, B = cl[a], cl[b]
    w = np.minimum(A.weight, B.weight)
    for part, sl in (("body", slice(0, nb)), ("layers", slice(nb, None))):
        m = w[:, sl] > 0.3
        da, db = A.delta[:, sl][m], B.delta[:, sl][m]
        corr = np.corrcoef(da.ravel(), db.ravel())[0, 1]
        rel = np.linalg.norm(da - db, axis=-1).mean() / max(0.5 * (np.linalg.norm(da, axis=-1).mean() + np.linalg.norm(db, axis=-1).mean()), 1e-9)
        # the part that is common: projection of the per-vertex time-mean
        print(f"{a} vs {b} [{part}]: co-visible {m.mean() * 100:.0f}%, |d| {np.linalg.norm(da, axis=-1).mean() * 1e3:.2f}/{np.linalg.norm(db, axis=-1).mean() * 1e3:.2f} mm, "
              f"corr {corr:.2f}, rel diff {rel:.2f}")
