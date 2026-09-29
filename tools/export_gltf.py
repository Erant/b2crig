"""b2crig's steps on the b2c glTF files (~/Projects/b2cgltf/SPEC.md; b2crig/export/gltf.py).

    .venv/bin/python tools/export_gltf.py rig work/<s> [--subject work/<s>/gltf/scene.glb] [--splat PLY] [--out OUT]
    .venv/bin/python tools/export_gltf.py clip work/<s> MOTION.npz[:NAME] ... [--subject RIGGED.glb] [--no-residual]
        [--check-cage NAME=CAGE ...]
    .venv/bin/python tools/export_gltf.py validate FILE ...

rig: enhances b2crunner's subject file (default work/<s>/gltf/scene.glb, written by b2crunner's export_glb) with
b2crig's rig, in place unless --out. --splat rigs a splat b2crig retrained (it supersedes the delivered one).
clip: writes work/<s>/gltf/<name>.clip.glb per motion. --check-cage compares b2crig's posed cage with an existing
clip cage file. validate: the Khronos validator check of SPEC 8 (B2C_GLTF_VALIDATOR).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import b2cgltf  # noqa: E402
from b2cgltf.validate import validate  # noqa: E402

from b2crig import evaluate  # noqa: E402
from b2crig.export import gltf  # noqa: E402

ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
r = sub.add_parser("rig"); r.add_argument("subject_dir", type=Path); r.add_argument("--subject", type=Path)
r.add_argument("--splat", type=Path); r.add_argument("--out", type=Path)
c = sub.add_parser("clip"); c.add_argument("subject_dir", type=Path); c.add_argument("motions", nargs="+")
c.add_argument("--subject", type=Path); c.add_argument("--no-residual", action="store_true")
c.add_argument("--check-cage", nargs="*", default=[])
v = sub.add_parser("validate"); v.add_argument("files", nargs="+", type=Path)
a = ap.parse_args()

if a.cmd == "rig":
    subj = a.subject or a.subject_dir / "gltf" / "scene.glb"
    print(json.dumps(gltf.rig(subj, a.subject_dir, a.out or subj, splat_ply=a.splat), indent=1))
elif a.cmd == "clip":
    subj = a.subject or a.subject_dir / "gltf" / "scene.glb"
    checks = dict(s.split("=", 1) for s in a.check_cage)
    for spec in a.motions:
        p, _, name = spec.partition(":")
        name = name or Path(p).stem
        out = subj.parent / f"{name}.clip.glb"
        rep = gltf.clip(subj, a.subject_dir, Path(p), name, out, residual=not a.no_residual,
                        subject_uri=os.path.relpath(subj, out.parent))
        if name in checks:
            from b2cgltf.b2crig import clip as GC
            posed = GC.pose(b2cgltf.load(subj), b2cgltf.load(out))
            ref = evaluate.read_cage(Path(checks[name])).posed.astype(np.float64)
            d = np.linalg.norm(posed - ref, axis=-1) * 1e3
            rep["vs_cage_mm_max"] = float(d.max())
        print(json.dumps(rep, indent=1))
else:
    for f in a.files:
        rep = validate(f)
        print(f"{f}: ok ({rep['issues']['numErrors']} allowed errors, {rep['issues']['numWarnings']} warnings, validator {rep['validatorVersion']})")
