"""Armpit / opened-crease crops: on the frames of a clip where the most surface was greyed as stretched (G3 control
holes), crop around the largest hole and show WAN next to each method's render (grey background).

    .venv/bin/python tools/armpit_crops.py CLIP_DIR OUT.png --m NAME=PLY[,extra=ARGS] ... [--frames 4] [--size 256]
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2crig import b2ctrain as B  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dance_eval import parse_method, rgb  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--m", action="append", required=True)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--size", type=int, default=256)
    a = ap.parse_args()
    clip = a.clip.resolve()
    cams = json.loads((clip / "cameras.json").read_text())
    cage = clip / "cage_v2.b2ccage" if (clip / "cage_v2.b2ccage").exists() else clip / "cage.b2ccage"
    # Frames by greyed area (the control's alpha holes inside WAN's figure), skipping WAN's kept frames.
    scored = []
    for c in cams["cameras"]:
        nm = Path(c["name"]).stem
        ct = cv2.imread(str(clip / "control" / f"{nm}.png"), cv2.IMREAD_UNCHANGED)
        mk = cv2.imread(str(clip / "mask" / f"{nm}.png"), 0)
        seg = cv2.imread(str(clip / "seg" / f"{nm}.png"), 0)
        if ct is None or ct.shape[2] != 4 or seg is None or (mk is not None and mk.max() == 0):
            continue
        hole = ((ct[..., 3] < 128) & (seg > 0)).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(hole)
        if n < 2:
            continue
        k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        scored.append((int(st[k, cv2.CC_STAT_AREA]), nm, c, st[k, :4]))
    scored.sort(key=lambda t: -t[0])
    pick, used = [], []
    for s in scored:   # spread over the clip: at least 8 frames apart
        i = int(s[1].split("_")[-1]) if s[1].split("_")[-1].isdigit() else len(used)
        if all(abs(i - u) >= 8 for u in used):
            pick.append(s); used.append(i)
        if len(pick) == a.frames:
            break
    ms = [parse_method(s) for s in a.m]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cj = td / "cams.json"
        cj.write_text(json.dumps({**cams, "cameras": [p[2] for p in pick]}))
        for m in ms:
            B.render(m["ply"], cj, td / m["name"], cage=cage, background=(0.5, 0.5, 0.5), extra=m["extra"])
        rows = []
        for _, nm, _, (x, y, w, h) in pick:
            wan = cv2.imread(str(clip / "wan" / f"{nm}.png"))
            H, W = wan.shape[:2]
            side = int(max(w, h) * 1.6) + 40
            cx, cy = x + w // 2, y + h // 2
            x0, y0 = max(0, min(W - side, cx - side // 2)), max(0, min(H - side, cy - side // 2))
            side = min(side, W, H)
            tiles = [wan]
            for m in ms:
                im = cv2.imread(str(td / m["name"] / f"{nm}.png"), cv2.IMREAD_UNCHANGED)
                tiles.append(rgb(im).clip(0, 255).astype(np.uint8))
            row = [cv2.resize(t[y0:y0 + side, x0:x0 + side], (a.size, a.size), interpolation=cv2.INTER_AREA) for t in tiles]
            rows.append(np.hstack(row))
    hdr = np.full((28, rows[0].shape[1], 3), 255, np.uint8)
    for j, lab in enumerate(["WAN"] + [m["name"] for m in ms]):
        cv2.putText(hdr, lab, (j * a.size + 6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(a.out), np.vstack([hdr] + rows))
    print(a.out, [p[1] for p in pick])


if __name__ == "__main__":
    main()
