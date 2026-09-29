"""Train the motion -> cage-displacement model on fitted clips; predict held-out clips.

    python -m b2crig.model.train --subject work/416580 --train c001 arm_s0 ... --val rand7 rand8 --out work/416580/model

The model: an MLP over a causal window of skeleton features -> a code z (r dims) -> per-vertex displacement in the
vertex's LBS frame through a learned basis (a linear layer, i.e. a data-driven PCA). Weighted MSE by visibility.
Held-out clips get <clip>/pred_<tag>/{delta.f32, vis.f32} in fit-cage's layout (world-frame displacement), so
b2crig.evaluate scores them against the WAN segmentation exactly like a fit.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..rig.mhr import MHRBody
from . import data as D


class MotionToCage(nn.Module):
    def __init__(self, d_in: int, n_verts: int, rank: int = 64, hidden: int = 512):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, rank))
        self.basis = nn.Linear(rank, n_verts * 3)
        nn.init.zeros_(self.basis.weight); nn.init.zeros_(self.basis.bias)
        self.n_verts = n_verts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.basis(self.enc(x)).view(-1, self.n_verts, 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=Path, required=True)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fit", default="fit_layered")
    ap.add_argument("--cage", default="cage_layered.b2ccage")
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--no-motion", action="store_true", help="ablation: input zeros (predicts the mean displacement)")
    a = ap.parse_args()
    dev = "cuda"
    body = MHRBody(a.subject / "mhr.npz")
    bi = D.cage_body_index(body, a.subject)
    clips = {n: D.load_clip(a.subject / "clips" / n, body, *D.layered_pair(a.subject / "clips" / n), bi) for n in a.train + a.val}

    def xs(cd):
        x = D.windows(cd.feats, a.window)
        return np.zeros_like(x) if a.no_motion else x
    Xtr = np.concatenate([xs(clips[n]) for n in a.train]); Ytr = np.concatenate([clips[n].delta for n in a.train])
    Wtr = np.concatenate([clips[n].weight for n in a.train])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-3
    X = torch.as_tensor((Xtr - mu) / sd, device=dev); Y = torch.as_tensor(Ytr, device=dev); Wt = torch.as_tensor(Wtr, device=dev)
    model = MotionToCage(X.shape[1], Y.shape[1], a.rank).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    val = {n: (torch.as_tensor((xs(clips[n]) - mu) / sd, device=dev), torch.as_tensor(clips[n].delta, device=dev),
               torch.as_tensor(clips[n].weight, device=dev)) for n in a.val}

    def wmse(p, y, w):
        return ((w[..., None] * (p - y) ** 2).sum() / (3 * w.sum().clamp_min(1e-6))).item()

    t0 = time.time(); bs = 32
    for ep in range(a.epochs):
        perm = torch.randperm(len(X), device=dev)
        for k in range(0, len(X), bs):
            i = perm[k:k + bs]
            p = model(X[i])
            loss = (Wt[i][..., None] * (p - Y[i]) ** 2).sum() / (3 * Wt[i].sum().clamp_min(1e-6))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 100 == 0 or ep == a.epochs - 1:
            with torch.no_grad():
                tr = wmse(model(X), Y, Wt)
                vl = {n: wmse(model(x), y, w) for n, (x, y, w) in val.items()}
                zero = {n: wmse(torch.zeros_like(y), y, w) for n, (x, y, w) in val.items()}
            print(f"epoch {ep}: train rmse {np.sqrt(tr) * 1000:.2f} mm; val rmse " +
                  ", ".join(f"{n} {np.sqrt(v) * 1000:.2f} (zero {np.sqrt(zero[n]) * 1000:.2f})" for n, v in vl.items()) +
                  f"  [{time.time() - t0:.0f}s]", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "mu": mu, "sd": sd, "window": a.window, "rank": a.rank,
                "d_in": X.shape[1], "n_verts": Y.shape[1], "train": a.train}, a.out / f"{a.tag}.pt")
    # held-out predictions in fit-cage layout (world frame)
    from ..evaluate import read_cage
    for n, (x, y, w) in val.items():
        clip = a.subject / "clips" / n
        with torch.no_grad():
            pl = model(x)
        params = torch.as_tensor(np.load(clip / "motion.npz")["body_params"], device=dev)
        world = []
        for i in range(0, len(pl), 16):
            Rv = D.vertex_frames(body, body.pose(params[i:i + 16]), bi)
            world.append(torch.einsum("tvij,tvj->tvi", Rv, pl[i:i + 16]).cpu().numpy())
        od = clip / f"pred_{a.tag}"; od.mkdir(exist_ok=True)
        np.concatenate(world).astype(np.float32).tofile(od / "delta.f32")
        (np.fromfile(clip / D.layered_pair(clip)[0] / "vis.f32", np.float32)).tofile(od / "vis.f32")
    (a.out / f"{a.tag}.json").write_text(json.dumps({"train": a.train, "val": a.val}))


if __name__ == "__main__":
    main()
