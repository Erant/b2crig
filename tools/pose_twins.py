"""Within-generation view transfer: frames of one clip with the same pose seen from other orbit angles.

    .venv/bin/python tools/pose_twins.py CLIP_DIR [--seg seg] [-- extra fit-cage args]

Groups the clip's frames by identical pose (max |body_params| difference < 3e-3), holds out all but the first frame
of each group, fits the remaining frames, gives each held-out frame its group's first frame's displacement, and
scores the held-out frames only: LBS, the transferred fit, and the clip's own fit (which saw them).
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2crig import b2ctrain as B  # noqa: E402
from b2crig import evaluate  # noqa: E402
from b2crig.model.data import layered_pair  # noqa: E402

args = sys.argv[1:]
extra = args[args.index("--") + 1:] if "--" in args else []
args = args[:args.index("--")] if "--" in args else args
seg = "seg"
if "--seg" in args:
    i = args.index("--seg"); seg = args[i + 1]; del args[i:i + 2]
clip = Path(args[0]).resolve()
run = Path(json.loads((clip.parents[2] / "subjects.json").read_text())[clip.parents[1].name])
splat = run / "ply" / "scene.ply"
own_fit, cage = layered_pair(clip)
p = np.load(clip / "motion.npz")["body_params"]; T = len(p)
d = np.abs(p[:, None] - p[None]).max(-1)
partner, seen = {}, set()
for i in range(T):
    if i in seen:
        continue
    g = [j for j in range(T) if d[i, j] < 3e-3 and j not in seen]; seen |= set(g)
    for j in g[1:]:
        partner[j] = i
held = sorted(partner)
print(f"{clip.name}: {len(held)} held-out frames, orbit angle to partner (deg) "
      f"median {np.median([(j - partner[j]) * 360 / T for j in held]):.0f}")

src = clip / ("fit_ds" if seg == "seg" else "fit_ds_rel")
ds = clip / "twins_ds"; shutil.rmtree(ds, ignore_errors=True); shutil.copytree(src, ds, symlinks=True)
keep = []
for ln in (src / "images.txt").read_text().splitlines():
    f = ln.split()
    if len(f) >= 10 and int(Path(f[9]).stem) not in partner:
        keep += [ln, ""]
(ds / "images.txt").write_text("\n".join(keep) + "\n")
out = clip / "twins_fit"
subprocess.run([str(B.B2CTRAIN), "fit-cage", *B.fit_groups_args(), "--splat", str(splat), "--cage", str(clip / cage), "--dataset", str(ds),
                "--output", str(out), "--iters", "8000", *extra], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
V = evaluate.read_cage(clip / cage).posed.shape[1]
dl = np.fromfile(out / "delta.f32", np.float32).reshape(T, V, 3)
for j in held:
    dl[j] = dl[partner[j]]
tr = clip / "twins_xfer"; tr.mkdir(exist_ok=True)
dl.tofile(tr / "delta.f32"); shutil.copy(out / "vis.f32", tr / "vis.f32")
res = {}
for name in ("twins_xfer", own_fit):
    evaluate.evaluate(clip, splat, name, video=False, cage=cage, seg=seg)
    pf = json.loads((clip / f"eval_{name}{'' if seg == 'seg' else '_' + seg}.json").read_text())["per_frame"]
    res.setdefault("lbs", pf["lbs"]); res[name] = pf[name]
groups = ["upper", "lower", "hair", "skin", "silhouette"]
print(f"{'held-out frames':22s}" + "".join(f"{g:>11s}" for g in groups))
for k, v in res.items():
    print(f"{k:22s}" + "".join(f"{np.nanmean([v[j][g] for j in held]):11.3f}" for g in groups))
