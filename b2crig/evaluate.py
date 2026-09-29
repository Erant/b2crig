"""Evaluate a cage fit against its WAN clip: label IoU per group, and a comparison video.

    python -m b2crig.evaluate CLIP_DIR [--fit fit] [--video]

Renders the clip's cage twice through b2ctrain (the LBS cage, and the cage plus
the fitted per-frame displacement), label-maps both, and compares each to the
Sapiens2 segmentation of the WAN frames.
"""
from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import b2ctrain as B

GROUPS = {"hair": (4,), "upper": (23, 1), "lower": (13,), "skin": (3, 6, 7, 11, 15, 16, 20, 22, 24, 25), "shoes": (9, 18)}


@dataclass
class Cage:
    layers: list[tuple[int, int, int, int, int]]
    verts: np.ndarray
    faces: np.ndarray
    names: list[str]
    posed: np.ndarray


def read_cage(path: Path) -> Cage:
    raw = Path(path).read_bytes()
    assert raw[:8] == b"B2CCAGE1"
    nl, nv, nf, nfr = struct.unpack("<4i", raw[8:24])
    o = 24
    layers = [struct.unpack("<4iI", raw[o + 20 * i:o + 20 * (i + 1)]) for i in range(nl)]
    o += 20 * nl
    V = np.frombuffer(raw, np.float32, nv * 3, o).reshape(nv, 3); o += nv * 12
    F = np.frombuffer(raw, np.int32, nf * 3, o).reshape(nf, 3); o += nf * 12
    names = [raw[o + 64 * i:o + 64 * (i + 1)].split(b"\0")[0].decode() for i in range(nfr)]; o += 64 * nfr
    P = np.frombuffer(raw, np.float32, nfr * nv * 3, o).reshape(nfr, nv, 3)
    return Cage(layers, V.copy(), F.copy(), names, P.copy())


def write_cage_like(c: Cage, path: Path, posed: np.ndarray) -> None:
    layers = []
    for (v_off, v_cnt, f_off, f_cnt, bits) in c.layers:
        classes = [k for k in range(32) if bits & (1 << k)]
        layers.append(B.CageLayer("", c.verts[v_off:v_off + v_cnt], c.faces[f_off:f_off + f_cnt] - v_off, classes))
    B.write_cage(path, layers, posed, c.names)


def load_delta(fit_dir: Path, c: Cage) -> tuple[np.ndarray, np.ndarray]:
    d = np.fromfile(fit_dir / "delta.f32", np.float32).reshape(len(c.names), len(c.verts), 3)
    v = np.fromfile(fit_dir / "vis.f32", np.float32).reshape(len(c.names), len(c.verts))
    return d, v


def group_map(lab: np.ndarray) -> np.ndarray:
    g = np.zeros(lab.shape, np.uint8)
    for k, cls in enumerate(GROUPS.values(), start=1):
        g[np.isin(lab, cls)] = k
    return g


def ious(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    out = {}
    pg, gg = group_map(pred), group_map(gt)
    for k, name in enumerate(GROUPS, start=1):
        a, b = pg == k, gg == k
        u = (a | b).sum()
        out[name] = float((a & b).sum() / u) if u else float("nan")
    a, b = pred > 0, gt > 0
    out["silhouette"] = float((a & b).sum() / max((a | b).sum(), 1))
    return out


def render_labels(splat: Path, cameras: Path, cage: Path, out: Path, alt: Path | None = None) -> None:
    if out.exists():
        shutil.rmtree(out)
    B.render(splat, cameras, out, cage=cage, label_maps=True, background=(0.5, 0.5, 0.5),
             extra=["--alt-binding", str(alt)] if alt else [])


def evaluate(clip: Path, splat: Path, fit: str = "fit", video: bool = True, cage: str = "cage.b2ccage",
             seg: str = "seg", alt: Path | None = None) -> dict:
    """`alt`: a dual-binding file (b2crig.rig.binding) the fitted render uses."""
    c = read_cage(clip / cage)
    d, vis = load_delta(clip / fit, c)
    fitted = clip / f"{fit}_cage.b2ccage"
    write_cage_like(c, fitted, c.posed + d)
    if not (clip / "eval_lbs").exists():
        render_labels(splat, clip / "cameras.json", clip / "cage.b2ccage", clip / "eval_lbs")
    render_labels(splat, clip / "cameras.json", fitted, clip / f"eval_{fit}", alt)
    res = {"lbs": [], fit: []}
    for nm in c.names:
        gt = cv2.imread(str(clip / seg / f"{nm}.png"), cv2.IMREAD_GRAYSCALE)
        for key, d_ in (("lbs", "eval_lbs"), (fit, f"eval_{fit}")):
            res[key].append(ious(cv2.imread(str(clip / d_ / f"{nm}.labels.png"), cv2.IMREAD_GRAYSCALE), gt))
    summary = {k: {g: float(np.nanmean([r[g] for r in v])) for g in list(GROUPS) + ["silhouette"]} for k, v in res.items()}
    summary["delta_mm"] = {"mean": float(np.linalg.norm(d, axis=-1).mean() * 1000),
                           "p99": float(np.percentile(np.linalg.norm(d, axis=-1), 99) * 1000)}
    (clip / f"eval_{fit}{'' if seg == 'seg' else '_' + seg}.json").write_text(json.dumps({"summary": summary, "per_frame": res}, indent=1))
    if video:
        make_video(clip, fit)
    return summary


def _rgba_on(path: Path, bg: int = 127) -> np.ndarray:
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    al = a[..., 3:] / 255
    return (a[..., :3] * al + bg * (1 - al)).astype(np.uint8)


def make_video(clip: Path, fit: str = "fit") -> Path:
    c_names = sorted(p.stem for p in (clip / "wan").glob("*.png"))
    pal = np.random.default_rng(3).integers(40, 255, (32, 3)).astype(np.uint8); pal[0] = 0
    out = clip / f"{clip.name}_{fit}.mp4"
    frames = []
    for nm in c_names:
        lbs = _rgba_on(clip / "eval_lbs" / f"{nm}.png")
        fitr = _rgba_on(clip / f"eval_{fit}" / f"{nm}.png")
        wan = cv2.imread(str(clip / "wan" / f"{nm}.png"))
        gt = cv2.imread(str(clip / "seg" / f"{nm}.png"), cv2.IMREAD_GRAYSCALE)
        pl = cv2.imread(str(clip / f"eval_{fit}" / f"{nm}.labels.png"), cv2.IMREAD_GRAYSCALE)
        # disagreement overlay: WAN seg group vs fitted render group; red = mismatch
        dis = (group_map(gt) != group_map(pl))
        over = (wan * 0.6).astype(np.uint8); over[dis] = (0, 0, 255)
        row = np.concatenate([lbs, fitr, wan, over], 1)
        for x, t in zip((10, lbs.shape[1] + 10, 2 * lbs.shape[1] + 10, 3 * lbs.shape[1] + 10),
                        ("LBS (rig only)", "fitted cage", "WAN target", "label mismatch (fit)")):
            cv2.putText(row, t, (x, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        frames.append(row)
    H, W = frames[0].shape[:2]
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                          "-r", "16", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)],
                         stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close(); p.wait()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", type=Path)
    ap.add_argument("--splat", type=Path, required=True)
    ap.add_argument("--fit", default="fit")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--cage", default="cage.b2ccage")
    a = ap.parse_args()
    print(json.dumps(evaluate(a.clip, a.splat, a.fit, not a.no_video, a.cage), indent=1))


if __name__ == "__main__":
    main()
