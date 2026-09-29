"""Clip queue: build -> WAN -> seg -> fit dataset -> fit-cage -> evaluate, one spec at a time.

    .venv/bin/python tools/clip_queue.py QUEUE.jsonl

Each line of QUEUE.jsonl is a spec: {"id", "subject", "run", "motion", and optional clip-builder kwargs
(elevation, sweep, azimuth0, seed, strength, loose_only, dilate_px, width, height, frames) and
"fit_args": [...extra fit-cage args], "fit": false to skip the per-clip fit}. The file is re-read after every clip, so specs can be appended while
it runs; a clip whose eval json exists is skipped. Clips go to <subject>/clips/<id>.
"""
from __future__ import annotations

import fcntl
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from b2crig.gen import clips  # noqa: E402
from b2crig.prep import build_dataset  # noqa: E402
from b2crig import evaluate  # noqa: E402
from b2crig import b2ctrain as B  # noqa: E402

GPU_LOCK = ROOT / "work" / "gpu.lock"   # held around the big-VRAM stages; side jobs: flock work/gpu.lock CMD


def gpu_lock():
    f = open(GPU_LOCK, "w")
    fcntl.flock(f, fcntl.LOCK_EX)
    return f


BUILD_KEYS = ("n_frames", "elevation", "elevation_end", "grey_unseen", "prompt_extra", "target_joint", "radius", "target_offset", "reference", "motion_text", "sweep", "azimuth0", "seed", "strength", "loose_only", "dilate_px", "width", "height", "splat", "control", "prompt", "stretch_grey", "grey_dilate", "grey_fill", "background", "local_mask", "rotate_grey", "stretch_mode")


def run_spec(spec: dict) -> None:
    subject = ROOT / spec["subject"]
    run = Path(spec["run"])
    clip = subject / "clips" / spec["id"]
    t0 = time.time()
    if not (clip / "wan" / "timing.json").exists():
        with gpu_lock():   # builds render the control on the GPU too
            clips.build(subject, run, spec["motion"], clip, **{k: spec[k] for k in BUILD_KEYS if k in spec})
            import torch
            torch.cuda.empty_cache()   # WAN needs all of the 12 GB card; this process keeps only its context
        with gpu_lock(), open(clip / "wan_log.txt", "w") as log:
            subprocess.run([str(clips.WAN_PYTHON), str(ROOT / "tools" / "wan_clip.py"), str(clip)], check=True,
                           stdout=log, stderr=subprocess.STDOUT,
                           env={**__import__("os").environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    t1 = time.time()
    if not (clip / "seg").exists():
        with gpu_lock():
            subprocess.run([str(clips.WAN_PYTHON), str(ROOT / "tools" / "seg_clip.py"), str(clip)], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    build_dataset(clip)
    if not spec.get("fit", True):   # WAN frames + seg + dataset only (for cage_train); no per-clip fit-cage
        (clip / "done.json").write_text(json.dumps({"wan_s": t1 - t0}))
        print(f"[{time.strftime('%H:%M:%S')}] {spec['id']}: wan {t1 - t0:.0f}s (no fit)", flush=True)
        return
    splat = run / "ply" / "scene.ply"
    lk = gpu_lock()
    subprocess.run([str(B.B2CTRAIN), "fit-cage", *B.fit_groups_args(), "--splat", str(splat), "--cage", str(clip / "cage.b2ccage"),
                    "--dataset", str(clip / "fit_ds"), "--output", str(clip / "fit"), "--iters", "8000", *spec.get("fit_args", [])],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    s = evaluate.evaluate(clip, splat, "fit", video=True)
    lk.close()
    t2 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] {spec['id']}: wan {t1 - t0:.0f}s, rest {t2 - t1:.0f}s; "
          f"IoU lbs->fit upper {s['lbs']['upper']:.3f}->{s['fit']['upper']:.3f} lower {s['lbs']['lower']:.3f}->{s['fit']['lower']:.3f} "
          f"hair {s['lbs']['hair']:.3f}->{s['fit']['hair']:.3f} sil {s['lbs']['silhouette']:.3f}->{s['fit']['silhouette']:.3f}", flush=True)


def main() -> None:
    qfile = Path(sys.argv[1])
    failed = set()
    while True:
        specs = [json.loads(l) for l in qfile.read_text().splitlines() if l.strip() and not l.startswith("#")]
        todo = [s for s in specs if s["id"] not in failed and not any((ROOT / s["subject"] / "clips" / s["id"] / f).exists() for f in ("eval_fit.json", "done.json"))]
        if not todo:
            print("queue empty", flush=True)
            return
        try:
            run_spec(todo[0])
        except Exception:
            traceback.print_exc()
            failed.add(todo[0]["id"])


if __name__ == "__main__":
    main()
