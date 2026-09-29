"""Score cage fits on a clip they were not fitted to (same motion and cage, another camera / seed).

    .venv/bin/python tools/holdout_eval.py HOLDOUT_CLIP [--seg seg_rel] NAME=FIT_DIR [NAME=FIT_DIR ...]

Each FIT_DIR (delta.f32 + vis.f32) is linked into HOLDOUT_CLIP/x_<NAME> and evaluated against the clip's WAN seg.
The clip's own fit and LBS are printed alongside as the bounds.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2crig import evaluate  # noqa: E402
from b2crig.model.data import layered_pair  # noqa: E402

args = sys.argv[1:]
seg = "seg"
if "--seg" in args:
    i = args.index("--seg"); seg = args[i + 1]; del args[i:i + 2]
clip = Path(args[0]).resolve()
run = Path(json.loads((clip.parents[2] / "subjects.json").read_text())[clip.parents[1].name])
splat = run / "ply" / "scene.ply"
own_fit, cage = layered_pair(clip)
sfx = "" if seg == "seg" else "_" + seg
if not (clip / f"eval_{own_fit}{sfx}.json").exists():
    evaluate.evaluate(clip, splat, own_fit, video=False, cage=cage, seg=seg)
own = json.loads((clip / f"eval_{own_fit}{sfx}.json").read_text())["summary"]
rows = {"lbs": own["lbs"], "own fit": own[own_fit]}
for arg in args[1:]:
    name, src = arg.split("=", 1)
    dst = clip / f"x_{name}"
    dst.mkdir(exist_ok=True)
    for f in ("delta.f32", "vis.f32"):
        (dst / f).unlink(missing_ok=True)
        (dst / f).symlink_to(Path(src).resolve() / f)
    rows[name] = evaluate.evaluate(clip, splat, f"x_{name}", video=False, cage=cage, seg=seg)[f"x_{name}"]
groups = ["upper", "lower", "hair", "skin", "silhouette"]
print(f"{'on ' + clip.name + ' (' + seg + ')':22s}" + "".join(f"{g:>11s}" for g in groups))
for k, s in rows.items():
    print(f"{k:22s}" + "".join(f"{s[g]:11.3f}" for g in groups))
