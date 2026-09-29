"""Merge several clips' fit datasets (same motion, different cameras) into one multi-view dataset for fit-cage.

    .venv/bin/python tools/merge_views.py OUT_DIR CLIP_DIR [CLIP_DIR ...]

Image NNNN.png of clip C becomes NNNN@C.png (fit-cage maps "<frame>@<view>" to cage frame <frame>); images and
labels are symlinked. All clips must share the intrinsics.
"""
import sys
from pathlib import Path

out = Path(sys.argv[1]); clips = [Path(c).resolve() for c in sys.argv[2:]]
cams = {(c / "fit_ds" / "cameras.txt").read_text().strip() for c in clips}
assert len(cams) == 1, "clips differ in intrinsics"
for sub in ("images", "labels"):
    (out / sub).mkdir(parents=True, exist_ok=True)
(out / "cameras.txt").write_text(cams.pop() + "\n")
(out / "points3D.txt").write_text("")
lines, k = [], 0
for c in clips:
    for ln in (c / "fit_ds" / "images.txt").read_text().splitlines():
        f = ln.split()
        if len(f) < 10:
            continue
        k += 1
        name = f"{Path(f[9]).stem}@{c.name}.png"
        lines += [" ".join([str(k), *f[1:9], name]), ""]
        for sub in ("images", "labels"):
            src = c / "fit_ds" / sub / f[9]
            dst = out / sub / name
            if src.exists() and not dst.exists():
                dst.symlink_to(src)
(out / "images.txt").write_text("\n".join(lines) + "\n")
print(f"merge_views: {k} images from {len(clips)} clips -> {out}")
