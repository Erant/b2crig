"""Learn the dual binding (b2crig.rig.binding) from several motion clips at once, then score held-out clips.

    .venv/bin/python tools/fit_binding.py work/<subject> --train CLIP ... --test CLIP ... [--tag fix] [--iters N]

Concatenates the train clips' cages (cage_<tag>.b2ccage, all built from the same layers) into one cage whose frames
are "<clip>__<frame>", links their fit datasets under those names, runs b2ctrain fit-cage --fit-binding --no-delta,
and renders the test clips LBS-only with the fixed rig, with and without the learned binding (rounded), scoring
labels (IoU vs the WAN seg) and colour (L1 vs the WAN frame inside the union of both silhouettes).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2crig import b2ctrain as B  # noqa: E402
from b2crig import evaluate  # noqa: E402
from b2crig.rig import binding as BD  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("subject", type=Path)
ap.add_argument("--train", nargs="+", required=True)
ap.add_argument("--test", nargs="+", required=True)
ap.add_argument("--tag", default="fix")
ap.add_argument("--iters", type=int, default=8000)
ap.add_argument("--out", default="binding/fit")
ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
a = ap.parse_args()
S = a.subject.resolve()
run = Path(json.loads((S.parent / "subjects.json").read_text())[S.name]); splat = run / "ply" / "scene.ply"
out = S / a.out; ds = out / "ds"
for sub in ("images", "labels"):
    (ds / sub).mkdir(parents=True, exist_ok=True)

# one cage over all train clips
raw0 = None; names, posed, lines = [], [], []
for c in a.train:
    clip = S / "clips" / c
    cg = evaluate.read_cage(clip / f"cage_{a.tag}.b2ccage")
    raw = (clip / f"cage_{a.tag}.b2ccage").read_bytes()
    head = raw[:24 + 20 * len(cg.layers) + cg.verts.nbytes + cg.faces.nbytes]
    raw0 = raw0 or head
    assert head == raw0, f"{c}: cage layers differ from the first train clip's"
    names += [f"{c}__{n}" for n in cg.names]; posed.append(cg.posed)
    (ds / "cameras.txt").write_text((clip / "fit_ds" / "cameras.txt").read_text())
    for ln in (clip / "fit_ds" / "images.txt").read_text().splitlines():
        f = ln.split()
        if len(f) < 10:
            continue
        nm = f"{c}__{f[9]}"
        lines += [" ".join([str(len(lines) // 2 + 1), *f[1:9], nm]), ""]
        for sub in ("images", "labels"):
            d = ds / sub / nm
            if not d.exists():
                d.symlink_to((clip / "fit_ds" / sub / f[9]).resolve())
(ds / "images.txt").write_text("\n".join(lines) + "\n"); (ds / "points3D.txt").write_text("")
P = np.concatenate(posed).astype(np.float32)
counts = np.frombuffer(bytes(raw0[8:24]), np.int32).copy(); counts[3] = len(names)
with open(out / "cage.b2ccage", "wb") as f:
    f.write(raw0[:8]); f.write(counts.tobytes()); f.write(raw0[24:])
    for nm in names:
        b = nm.encode()[:63]; f.write(b + b"\0" * (64 - len(b)))
    f.write(P.tobytes())
print(f"fit_binding: {len(a.train)} clips, {len(names)} frames")

subprocess.run([str(B.B2CTRAIN), "fit-cage", *B.fit_groups_args(), "--splat", str(splat), "--cage", str(out / "cage.b2ccage"), "--dataset", str(ds),
                "--output", str(out), "--iters", str(a.iters), "--alt-binding", str(S / "binding" / "candidates.bin"),
                "--fit-binding", "--no-delta", "--photo-weight", "1.0", *a.extra], check=True)
s, fb, w = BD.read_alt(out / "alt_binding.bin")
print(f"learned: {np.mean(w > 0.5) * 100:.1f}% of {len(w)} candidates prefer B; w quantiles {np.quantile(w, [0.1, 0.5, 0.9]).round(2)}")
BD.write_alt(out / "alt_rounded.bin", s, fb, (w > 0.5).astype(np.float32))


def score(clip: Path, render_dir: Path) -> dict:
    res = {"iou": [], "l1": []}
    for nm in evaluate.read_cage(clip / f"cage_{a.tag}.b2ccage").names:
        gt = cv2.imread(str(clip / "seg" / f"{nm}.png"), 0)
        res["iou"].append(evaluate.ious(cv2.imread(str(render_dir / f"{nm}.labels.png"), 0), gt))
        r = cv2.imread(str(render_dir / f"{nm}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
        wan = cv2.imread(str(clip / "wan" / f"{nm}.png")).astype(np.float32)
        m = (r[..., 3] > 127) | (gt > 0)
        res["l1"].append(np.abs(r[..., :3] - wan)[m].mean())
    g = ["upper", "lower", "hair", "skin", "silhouette"]
    return {**{k: float(np.nanmean([x[k] for x in res["iou"]])) for k in g}, "L1": float(np.mean(res["l1"]))}


print(f"{'test (LBS only)':28s}" + "".join(f"{k:>11s}" for k in ["upper", "lower", "hair", "skin", "silhouette", "L1"]))
for c in a.test:
    clip = S / "clips" / c
    for label, alt in (("fixed rig", None), ("+ learned binding", out / "alt_rounded.bin")):
        rd = out / f"render_{c}_{'alt' if alt else 'base'}"
        B.render(splat, clip / "cameras.json", rd, cage=clip / f"cage_{a.tag}.b2ccage", label_maps=True, background=(0.5, 0.5, 0.5),
                 extra=["--alt-binding", str(alt)] if alt else [])
        sc = score(clip, rd)
        print(f"{c + ' ' + label:28s}" + "".join(f"{sc[k]:11.3f}" for k in ["upper", "lower", "hair", "skin", "silhouette", "L1"]), flush=True)
