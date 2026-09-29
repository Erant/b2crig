"""Does a rig change bring plain LBS closer to what WAN draws? No fitting: LBS renders only.

    .venv/bin/python tools/rig_compare.py TAG CLIP_DIR [CLIP_DIR ...]

Rebuilds <clip>/cage_<TAG>.b2ccage from the clip's motion with the subject's current layers (tools/refit.py), renders
it with zero displacement, and scores it against the clip's WAN segmentation next to the clip's own LBS (cage.b2ccage).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from b2crig import evaluate  # noqa: E402
from refit import rebuild_cage  # noqa: E402

tag = sys.argv[1]
groups = ["upper", "lower", "hair", "skin", "silhouette"]
rows = {"old LBS": [], f"{tag} LBS": []}
for c in sys.argv[2:]:
    clip = Path(c).resolve(); subject = clip.parent.parent
    run = Path(json.loads((subject.parent / "subjects.json").read_text())[subject.name]); splat = run / "ply" / "scene.ply"
    cage = rebuild_cage(clip, subject, tag, True)
    T, V = evaluate.read_cage(cage).posed.shape[:2]
    z = clip / f"zero_{tag}"; z.mkdir(exist_ok=True)
    np.zeros((T, V, 3), np.float32).tofile(z / "delta.f32"); np.zeros((T, V), np.float32).tofile(z / "vis.f32")
    s = evaluate.evaluate(clip, splat, f"zero_{tag}", video=False, cage=cage.name)
    rows["old LBS"].append([s["lbs"][g] for g in groups]); rows[f"{tag} LBS"].append([s[f"zero_{tag}"][g] for g in groups])
    print(f"{clip.name:16s} old " + " ".join(f"{x:.3f}" for x in rows["old LBS"][-1]) + f" | {tag} " + " ".join(f"{x:.3f}" for x in rows[f"{tag} LBS"][-1]), flush=True)
print(f"{'mean':16s}" + "".join(f"{g:>11s}" for g in groups))
for k, v in rows.items():
    print(f"{k:16s}" + "".join(f"{x:11.3f}" for x in np.mean(v, 0)))
