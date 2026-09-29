"""Static targets: the view-averaged cage displacement of each held pose (clips of motion hold_*).

A per-frame fit is dominated by its camera view (same-view fits of two WAN seeds correlate ~0.64, one clip's fits
71 deg apart ~0.1), but averaging a vertex over every view of a held pose cancels that: the averages of two seeds
agree at ~0.74 and transfer across generations. Per held segment (>= MIN_FRAMES identical poses):

    delta_world [V, 3]  mean over the views that saw the vertex (vis > VIS_MIN)
    delta_local [V, 3]  the same in the vertex's LBS frame (model/data.py's target convention)
    count       [V]     how many views saw it (the weight)

Segments of different clips with the same pose (seeds) are merged count-weighted.

    python -m b2crig.model.static --subject work/416580 [--clips hold_arms_s0 ...]   -> <subject>/static/targets.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ..evaluate import read_cage
from ..rig.mhr import MHRBody
from . import data as D

MIN_FRAMES = 8
VIS_MIN = 1.0


def segments(params: np.ndarray) -> list[list[int]]:
    """Maximal runs of frames whose pose equals the previous frame's (the run includes that first frame)."""
    same = np.abs(np.diff(params, axis=0)).max(1) < 1e-6
    runs, cur = [], []
    for t, s in enumerate(same, start=1):
        if s:
            cur = cur or [t - 1]
            cur.append(t)
        elif cur:
            runs.append(cur); cur = []
    if cur:
        runs.append(cur)
    return [r for r in runs if len(r) >= MIN_FRAMES]


def extract(clip: Path, body: MHRBody, layer_bi: np.ndarray, fit_name: str | None = None) -> list[dict]:
    fit, cage = D.layered_pair(clip)
    fit = fit_name or fit
    T, V = read_cage(clip / cage).posed.shape[:2]
    d = np.fromfile(clip / fit / "delta.f32", np.float32).reshape(T, V, 3)
    w = (np.fromfile(clip / fit / "vis.f32", np.float32).reshape(T, V) > VIS_MIN).astype(np.float32)
    params = np.load(clip / "motion.npz")["body_params"]
    out = []
    for seg in segments(params):
        cnt = w[seg].sum(0)
        mean = (d[seg] * w[seg][..., None]).sum(0) / np.maximum(cnt, 1)[:, None]
        posed = body.pose(torch.as_tensor(params[seg[0]][None], device=body.device))
        Rv = D.vertex_frames(body, posed, layer_bi)[0].cpu().numpy()          # [V, 3, 3]
        local = np.einsum("vji,vj->vi", Rv, mean)
        out.append({"clip": clip.name, "frames": seg, "params": params[seg[0]], "delta_world": mean, "delta_local": local,
                    "count": cnt})
    return out


def merge(items: list[dict]) -> list[dict]:
    """Count-weighted merge of segments with the same pose."""
    merged: list[dict] = []
    for it in items:
        for m in merged:
            if np.abs(m["params"] - it["params"]).max() < 1e-6:
                c = m["count"] + it["count"]
                for k in ("delta_world", "delta_local"):
                    m[k] = (m[k] * m["count"][:, None] + it[k] * it["count"][:, None]) / np.maximum(c, 1)[:, None]
                m["count"] = c; m["clip"] += "+" + it["clip"]
                break
        else:
            merged.append(dict(it))
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=Path, required=True)
    ap.add_argument("--clips", nargs="*", help="default: every clip whose motion holds a pose")
    ap.add_argument("--fit", help="fit dir name in each clip (default: its layered fit)")
    ap.add_argument("--out", default="targets", help="-> <subject>/static/<out>.npz")
    a = ap.parse_args()
    body = MHRBody(a.subject / "mhr.npz"); bi = D.cage_body_index(body, a.subject)
    names = a.clips or sorted(p.name for p in (a.subject / "clips").iterdir()
                              if (p / "eval_fit.json").exists() and (a.fit is None or (p / a.fit).exists()))
    items = []
    for n in names:
        segs = extract(a.subject / "clips" / n, body, bi, a.fit)
        items += segs
        if segs:
            print(f"{n}: {len(segs)} held segments, frames {[f'{s[0]}..{s[-1]}' for s in (x['frames'] for x in segs)]}")
    m = merge(items)
    (a.subject / "static").mkdir(exist_ok=True)
    np.savez(a.subject / "static" / f"{a.out}.npz", clips=np.array([x["clip"] for x in m]), params=np.stack([x["params"] for x in m]),
             delta_world=np.stack([x["delta_world"] for x in m]).astype(np.float32),
             delta_local=np.stack([x["delta_local"] for x in m]).astype(np.float32), count=np.stack([x["count"] for x in m]))
    print(f"{len(m)} static poses from {len(items)} segments -> {a.subject / 'static' / (a.out + '.npz')}")


if __name__ == "__main__":
    main()
