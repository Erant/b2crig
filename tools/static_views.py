"""Is a fit 3D-consistent within one WAN generation? A held pose (motion hold_*) seen from the whole orbit.

    .venv/bin/python tools/static_views.py CLIP_DIR [--tied-iters N] [--tag T] [-- extra fit-cage args]

The static frames (pose unchanged from the previous frame) are cut into 4 orbit blocks; blocks 0 and 2 fit, 1 and 3
are held out. "tied": every fitting frame is named <rep>@<frame>, so one displacement explains all fitting views
(--temporal 0: its neighbouring cage frames are never fitted).
"untied": the fitting frames get per-frame fits; each held-out frame borrows the nearest fitting frame's. Scores on
the held-out frames: LBS, tied, untied, and the clip's own fit (which saw them).
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
tied_iters, tag = 3200, ""
if "--tied-iters" in args:
    i = args.index("--tied-iters"); tied_iters = int(args[i + 1]); del args[i:i + 2]
if "--tag" in args:
    i = args.index("--tag"); tag = args[i + 1]; del args[i:i + 2]
clip = Path(args[0]).resolve()
run = Path(json.loads((clip.parents[2] / "subjects.json").read_text())[clip.parents[1].name])
splat = run / "ply" / "scene.ply"
own_fit, cage = layered_pair(clip)
p = np.load(clip / "motion.npz")["body_params"]; T = len(p)
static = [t for t in range(1, T) if np.abs(p[t] - p[t - 1]).max() < 1e-6]
blocks = np.array_split(np.array(static), 4)
fit_fr = sorted(np.concatenate([blocks[0], blocks[2]]).tolist()); held = sorted(np.concatenate([blocks[1], blocks[3]]).tolist())
rep = static[0]
# fit-cage diverges past ~100-150 steps per cage frame (2026-09-25): untied gets 100 per fitted frame
print(f"{clip.name}: static frames {static[0]}..{static[-1]}; fitting {len(fit_fr)}, held out {len(held)}")

src = clip / "fit_ds"
lines = [ln.split() for ln in (src / "images.txt").read_text().splitlines() if len(ln.split()) >= 10]
V = evaluate.read_cage(clip / cage).posed.shape[1]
out = {}
for mode in ("tied", "untied"):
    iters = tied_iters if mode == "tied" else 100 * len(fit_fr)
    ds = clip / f"static_{mode}{tag}_ds"; shutil.rmtree(ds, ignore_errors=True)
    for sub in ("images", "labels"):
        (ds / sub).mkdir(parents=True)
    shutil.copy(src / "cameras.txt", ds); (ds / "points3D.txt").write_text("")
    keep = []
    for f in lines:
        t = int(Path(f[9]).stem)
        if t not in fit_fr:
            continue
        name = f"{rep:04d}@{t:04d}.png" if mode == "tied" else f[9]
        keep += [" ".join(f[:9] + [name]), ""]
        for sub in ("images", "labels"):
            (ds / sub / name).symlink_to((src / sub / f[9]).resolve())
    (ds / "images.txt").write_text("\n".join(keep) + "\n")
    fo = clip / f"static_{mode}{tag}_fit"
    subprocess.run([str(B.B2CTRAIN), "fit-cage", *B.fit_groups_args(), "--splat", str(splat), "--cage", str(clip / cage), "--dataset", str(ds),
                    "--output", str(fo), "--iters", str(iters),
                    *(["--temporal", "0"] if mode == "tied" else []), *extra], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    d = np.fromfile(fo / "delta.f32", np.float32).reshape(T, V, 3)
    x = np.zeros_like(d)
    for t in range(T):
        x[t] = d[rep] if mode == "tied" else d[min(fit_fr, key=lambda s: abs(s - t))]
    xo = clip / f"static_{mode}{tag}_x"; xo.mkdir(exist_ok=True)
    x.tofile(xo / "delta.f32"); shutil.copy(fo / "vis.f32", xo / "vis.f32")
    out[mode] = xo.name
res = {}
alts = {}
if "--fit-binding" in extra:                     # render each fit with the binding it learned (rounded)
    from b2crig.rig import binding as BD
    for mode in ("tied", "untied"):
        s_, f_, w_ = BD.read_alt(clip / f"static_{mode}{tag}_fit" / "alt_binding.bin")
        BD.write_alt(clip / f"static_{mode}{tag}_x" / "alt_rounded.bin", s_, f_, (w_ > 0.5).astype(np.float32))
        alts[out[mode]] = clip / f"static_{mode}{tag}_x" / "alt_rounded.bin"
for name in (out["tied"], out["untied"], own_fit):
    evaluate.evaluate(clip, splat, name, video=False, cage=cage, alt=alts.get(name))
    pf = json.loads((clip / f"eval_{name}.json").read_text())["per_frame"]
    res.setdefault("lbs", pf["lbs"]); res[name] = pf[name]
groups = ["upper", "lower", "hair", "skin", "silhouette"]
for label, frames in (("held-out frames", held), ("fitting frames", fit_fr)):
    print(f"{label:22s}" + "".join(f"{g:>11s}" for g in groups))
    for k, v in res.items():
        print(f"{k:22s}" + "".join(f"{np.nanmean([v[j][g] for j in frames]):11.3f}" for g in groups))
