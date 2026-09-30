"""Rig a b2crunner subject file in one go: the pipeline's entry point (b2crunner's `rig_subject` step).

    .venv/bin/python tools/rig_subject.py SUBJECT.glb CAPTURE_DIR OUT.glb [--work DIR] [--clip rom_tour|none]

CAPTURE_DIR is the run's COLMAP dataset (images/, labels/, cameras.txt, images.txt), the views the delivered splat was
trained on. The flow is the one that gave b24be4's spH16, each stage the tool of the same name:

1. the subject from the file: mhr.npz (b2crig/subject.py), the delivered splat as a trainer PLY, the frozen
   expression-stable faces;
2. build_layers: the default garment / hair / face offset layers from the splat's seg labels;
3. pose_library rom:1500 --hands, then pose_select --k 16 over it: the containment pose set (poseset_romh16);
4. split_joints: hand splats that straddle a finger joint, one child per bone;
5. cage_train --contain poseset_romh16:1 --contain-hands 0.3: the pose containment fine-tune of the split splat
   against the capture views (random background, no growth);
6. export_gltf rig --splat: OUT.glb = the subject file with the cage, b2ctrain's binding of the retrained splat and
   the preview skin (SPEC 5);
7. export_gltf clip: OUT's folder gets <clip>.clip.glb, a tour of the ROM library's named poses (motion/rom.tour),
   so a player has something to pose it with.

The work directory holds b2crig's usual subject layout (WORK/subjects.json, WORK/run -> the inputs, WORK/subject);
by default it is OUT's folder/rig_work and is left for inspection. Prints one JSON report line last.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("subject_glb", type=Path)
ap.add_argument("capture", type=Path, help="the run's COLMAP dataset (images/, labels/, cameras.txt, images.txt)")
ap.add_argument("out", type=Path)
ap.add_argument("--work", type=Path, help="work directory (default: OUT's folder/rig_work)")
ap.add_argument("--clip", default="rom_tour", help="the clip to export beside OUT ('none': no clip)")
ap.add_argument("--device", default="cuda")
a = ap.parse_args()

if not os.environ.get("B2CTRAIN") and not (Path.home() / "Projects" / "b2ctrain" / "build" / "b2ctrain").exists() \
        and shutil.which("b2ctrain"):
    os.environ["B2CTRAIN"] = shutil.which("b2ctrain")   # the pod's b2ctrain is on PATH, not in a checkout

import numpy as np  # noqa: E402

from b2crig import subject  # noqa: E402

cap = a.capture.resolve()
for f in ("cameras.txt", "images.txt", "images"):
    if not (cap / f).exists():
        raise SystemExit(f"rig_subject: {cap} has no {f}: not a COLMAP dataset")
work = (a.work or a.out.parent / "rig_work").resolve()
S, run = work / "subject", work / "run"
shutil.rmtree(work, ignore_errors=True)
(run / "ply").mkdir(parents=True); S.mkdir()
(run / "colmap").symlink_to(cap)
(run / "ply" / "scene.glb").symlink_to(a.subject_glb.resolve())
(work / "subjects.json").write_text(json.dumps({S.name: str(run)}))
report, t00 = {"stages": {}}, time.time()


def stage(name, fn):
    t0 = time.time()
    print(f"rig_subject: {name}", flush=True)
    out = fn()
    report["stages"][name] = round(time.time() - t0, 1)
    return out


def tool(*args):
    subprocess.run([sys.executable, str(ROOT / "tools" / args[0]), *map(str, args[1:])], check=True, cwd=ROOT)


def prepare():
    subject.mhr_npz(a.subject_glb, S / "mhr.npz")
    subject.splat_ply(a.subject_glb, run / "ply" / "scene.ply")
    from b2crig.rig.layered import stable_faces
    from b2crig.rig.mhr import MHRBody
    np.save(S / "stable_faces.npy", stable_faces(MHRBody(S / "mhr.npz", device=a.device)))   # frozen: every later tool reads it


stage("subject", prepare)
stage("build_layers", lambda: tool("build_layers.py", S))
stage("pose_library", lambda: tool("pose_library.py", S, "rom:1500", "--hands", "--device", a.device))
stage("pose_select", lambda: tool("pose_select.py", S, "lib_rom_hands:1", "--k", "16", "--tag", "romh16"))
split = S / "train" / "split.ply"
split.parent.mkdir(parents=True, exist_ok=True)
stage("split_joints", lambda: tool("split_joints.py", S, run / "ply" / "scene.ply", split))
stage("cage_train", lambda: tool("cage_train.py", S, "--init", split, "--contain", "poseset_romh16:1",
                                 "--contain-hands", "0.3", "--tag", "contain"))
trained = S / "train" / "contain" / "scene.ply"
a.out.parent.mkdir(parents=True, exist_ok=True)


def rig():
    from b2crig.export import gltf
    return gltf.rig(a.subject_glb, S, a.out, splat_ply=trained)


report["rig"] = stage("rig", rig)
if a.clip != "none":
    def clip():
        import torch
        from b2crig.export import gltf
        from b2crig.motion import rom
        from b2crig.rig.mhr import MHRBody
        if a.clip != "rom_tour":
            raise SystemExit(f"rig_subject: unknown clip {a.clip!r} (rom_tour, none)")
        body = MHRBody(S / "mhr.npz", device=a.device)
        with torch.no_grad():
            m = rom.tour(body.body0, body.expr)
        (S / "motions").mkdir(exist_ok=True)
        np.savez(S / "motions" / f"{a.clip}.npz", **m)
        out = a.out.parent / f"{a.clip}.clip.glb"
        return gltf.clip(a.out, S, S / "motions" / f"{a.clip}.npz", a.clip, out,
                         subject_uri=os.path.relpath(a.out, out.parent))
    report["clip"] = stage("clip", clip)
report["seconds"] = round(time.time() - t00, 1)
report["out"] = str(a.out)
print(json.dumps(report, default=str), flush=True)
