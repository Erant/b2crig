"""Re-render a b2cviewer screenshot with b2ctrain: the viewer's PNG (S / "⤓ PNG") carries a tEXt chunk "b2cviewer" with
the subject and clip files loaded, the frame shown, the render options and the camera. This renders exactly that view
(and the same view with other splats) and writes a side-by-side sheet: viewer | b2ctrain | --splat ...

    .venv/bin/python tools/viewer_shot.py [SHOT.png] [--splat PLY ...] [--names a,b] [--out OUT.png] [--extra "..."]
        [--meta]   (just print the embedded JSON)

Without SHOT.png: the newest ~/Downloads/b2cview_*.png. Paths in the metadata are relative to b2crig/work (serve.py's
root). b2ctrain gets the subject file's current splat and the frame's posed cage (export/gltf.py render_inputs, with or
without the clip's residual as the viewer showed it). Prints the frame and the camera so the view can be reused.
"""
import argparse
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from b2crig import b2ctrain as B  # noqa: E402
from b2crig.export.gltf import render_inputs  # noqa: E402


def read_meta(png: Path) -> dict:
    raw = png.read_bytes()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n", f"{png}: not a PNG"
    o = 8
    while o < len(raw):
        n, = struct.unpack(">I", raw[o:o + 4]); typ = raw[o + 4:o + 8]; body = raw[o + 8:o + 8 + n]
        if typ == b"tEXt" and body.startswith(b"b2cviewer\0"):
            return json.loads(body[len(b"b2cviewer\0"):].decode("latin-1"))
        o += 12 + n
    raise SystemExit(f"{png}: no b2cviewer metadata (saved with the viewer's S key?)")


ap = argparse.ArgumentParser()
ap.add_argument("shot", type=Path, nargs="?")
ap.add_argument("--splat", type=Path, action="append", default=[])
ap.add_argument("--names", default="")
ap.add_argument("--out", type=Path)
ap.add_argument("--extra", default="", help="extra b2ctrain render args for every b2ctrain column")
ap.add_argument("--meta", action="store_true")
a = ap.parse_args()
shot = a.shot or max((Path.home() / "Downloads").glob("b2cview_*.png"), key=lambda p: p.stat().st_mtime)
m = read_meta(shot)
if a.meta:
    print(json.dumps(m, indent=1)); sys.exit()
work = ROOT / "work"
if "subject_file" not in m:
    raise SystemExit("the shot is from the old viewer (PLY + .b2ccage): re-take it with the glTF viewer")
if str(m["subject_file"]).startswith("local:") or str(m.get("clip_file") or "").startswith("local:"):
    raise SystemExit("the shot shows dropped local files: pass --splat and render by hand")
opt = m["options"]
print(f"{shot.name}: {m['subject']} / {m['clip'] or 'rest pose'} frame {m['frame']}"
      f"{'' if opt.get('residual', True) else ' (plain skinning)'}, {m['width']}x{m['height']}, "
      f"camera at {np.round(m['camera']['position'], 3).tolist()}")
tmp = Path(tempfile.mkdtemp(dir=work))
ply, cage, _ = render_inputs(work / m["subject_file"], work / m["clip_file"] if m.get("clip_file") else None, m["frame"],
                             tmp, residual=opt.get("residual", True))
cam = {**m["camera"], "name": f"{m['frame']:04d}"}   # the name of render_inputs' one cage frame
(tmp / "cams.json").write_text(json.dumps({"width": m["width"], "height": m["height"], "cameras": [cam]}))
common = ["--sh-degree", str(opt.get("degree", 3))]
if cage and opt.get("fade") and opt["fade"][1] > opt["fade"][0] > 0:
    common += ["--cage-fade-start", str(opt["fade"][0]), "--cage-fade-end", str(opt["fade"][1])]
bg = tuple(opt.get("bg", [0.5, 0.5, 0.5]))
cols = [cv2.imread(str(shot))]
labels = ["viewer"]
names = a.names.split(",") if a.names else []
for k, sp in enumerate([ply] + a.splat):
    od = tmp / f"r{k}"
    growth = opt.get("maxGrowth") or B.CAGE_MAX_GROWTH
    subprocess.run([str(B.B2CTRAIN), "render", "--splat", str(sp), "--cameras", str(tmp / "cams.json"), "--output-dir", str(od),
                    "--background", ",".join(f"{x:g}" for x in bg),
                    *(["--cage", str(cage), "--cage-max-growth", f"{growth:g}"] if cage else []), *common, *a.extra.split()], check=True, stdout=subprocess.DEVNULL)
    x = cv2.imread(str(next(od.glob("*.png"))), cv2.IMREAD_UNCHANGED).astype(np.float32)
    if x.shape[2] == 4:
        al = x[..., 3:] / 255
        x = x[..., :3] * al + np.array(bg[::-1]) * 255 * (1 - al)
    cols.append(x.astype(np.uint8))
    labels.append(names[k] if k < len(names) else ("b2ctrain" if k == 0 else sp.parent.name + "/" + sp.stem))
H = min(c.shape[0] for c in cols)
cols = [cv2.resize(c, (round(c.shape[1] * H / c.shape[0]), H)) for c in cols]
for c, lb in zip(cols, labels):
    cv2.putText(c, lb, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
out = a.out or shot.with_name(shot.stem + "_cmp.png")
cv2.imwrite(str(out), np.hstack(cols))
subprocess.run(["rm", "-rf", str(tmp)])
print(f"viewer_shot: {out}")
