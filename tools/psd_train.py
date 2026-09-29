"""Neural pose-space correctives for the cage, learned from WAN clips, scored against plain LBS on held-out clips.

    .venv/bin/python tools/psd_train.py work/<subject> --train CLIP ... --test CLIP ... [--fit-tag fit_psd] [--tag psd]
        [--rank 32] [--hidden 256] [--epochs 3000] [--vel] [--device cuda]

1. Per train clip, b2ctrain fit-cage (body layer frozen, |delta| <= 3 cm, temporal smoothing) unless
   CLIP/<fit-tag>/delta.f32 exists: per-frame cage displacements that make the LBS-posed splat match the WAN frames.
2. A small model maps the pose (the named rotation DOFs of motion.npz; with --vel also their velocities, for lag) to
   a displacement of every cage vertex, through a learned low-rank basis (rank R): delta = B tanh-MLP(pose). Loss:
   visibility-weighted squared error against the fitted displacements (single-view fits are noisy: the weights are
   fit-cage's per-vertex visibility, and the low rank + weight decay keep the model from learning view noise).
   Baseline "const": the visibility-weighted mean displacement (pose-independent).
3. For every test clip writes CLIP/cage_<tag>.b2ccage and CLIP/cage_<tag>_const.b2ccage (LBS posed + prediction):
   score them with tools/dance_eval.py (cage=...) against LBS.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2crig import b2ctrain as B  # noqa: E402
from b2crig import evaluate  # noqa: E402
from b2crig.rig.skeleton import ROT, body_index  # noqa: E402

NAMED = sorted({body_index(mi) for ax in ROT.values() for mi in ax if mi is not None})


def features(clip: Path, vel: bool) -> np.ndarray:
    m = np.load(clip / "motion.npz")
    x = m["body_params"][:, NAMED]
    if "root_R" in m.files:   # the root's tilt matters for gravity (lean, bend); its yaw does not
        up = m["root_R"][:, :, 1]
        x = np.concatenate([x, up[:, [0, 2]]], 1)
    if vel:
        v = np.gradient(x, axis=0) * float(m["fps"]) / 16.0
        x = np.concatenate([x, v], 1)
    return x.astype(np.float32)


def fit_clip(clip: Path, splat: Path, tag: str, iters: int) -> None:
    if (clip / tag / "delta.f32").exists():
        return
    subprocess.run([str(B.B2CTRAIN), "fit-cage", *B.fit_groups_args(), "--splat", str(splat), "--cage", str(clip / "cage.b2ccage"),
                    "--dataset", str(clip / "fit_ds"), "--output", str(clip / tag), "--iters", str(iters),
                    "--freeze-layers", "0", "--max-disp", "0.03", "--temporal", "0.3"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class PSD(torch.nn.Module):
    def __init__(self, n_in: int, n_out: int, rank: int, hidden: int):
        super().__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(n_in, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, hidden),
                                       torch.nn.Tanh(), torch.nn.Linear(hidden, rank))
        self.basis = torch.nn.Parameter(torch.zeros(rank, n_out))
        self.bias = torch.nn.Parameter(torch.zeros(n_out))

    def forward(self, x):
        return self.mlp(x) @ self.basis + self.bias


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("subject", type=Path)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--splat", type=Path, help="default: the subject's run scene.ply")
    ap.add_argument("--fit-tag", default="fit_psd")
    ap.add_argument("--fit-iters", type=int, default=8000)
    ap.add_argument("--tag", default="psd")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--vel", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    S = a.subject.resolve()
    run = Path(json.loads((S.parent / "subjects.json").read_text())[S.name])
    splat = a.splat or run / "ply" / "scene.ply"
    X, D, W = [], [], []
    for c in a.train:
        clip = S / "clips" / c
        fit_clip(clip, splat, a.fit_tag, a.fit_iters)
        cg = evaluate.read_cage(clip / "cage.b2ccage")
        d, v = evaluate.load_delta(clip / a.fit_tag, cg)
        mk = [cv for cv in range(len(cg.names))]
        X.append(features(clip, a.vel)[mk]); D.append(d[mk]); W.append(v[mk])
        print(f"psd: {c}: {len(mk)} frames, |delta| mean {np.linalg.norm(d, axis=-1).mean() * 1000:.2f} mm", flush=True)
    X, D, W = np.concatenate(X), np.concatenate(D), np.concatenate(W)
    nv = D.shape[1]
    const = (D * W[..., None]).sum(0) / np.maximum(W.sum(0), 1e-6)[:, None]
    mu, sd = X.mean(0), X.std(0) + 1e-3
    dev = a.device
    xt = torch.as_tensor((X - mu) / sd, device=dev)
    dt = torch.as_tensor(D.reshape(len(D), -1), device=dev)
    wt = torch.as_tensor(np.repeat(W, 3, axis=1), device=dev)
    model = PSD(X.shape[1], nv * 3, a.rank, a.hidden).to(dev)
    with torch.no_grad():
        model.bias.copy_(torch.as_tensor(const.reshape(-1), device=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=a.wd)
    for ep in range(a.epochs):
        idx = torch.randint(0, len(xt), (256,), device=dev)
        e = (model(xt[idx]) - dt[idx]) ** 2 * wt[idx]
        loss = e.sum() / wt[idx].sum()
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 500 == 0 or ep == a.epochs - 1:
            print(f"psd: epoch {ep}: weighted rmse {loss.sqrt().item() * 1000:.3f} mm "
                  f"(const {np.sqrt((((D - const) ** 2).sum(-1) * W).sum() / W.sum() / 3) * 1000:.3f})", flush=True)
    model.eval()
    for c in a.test:
        clip = S / "clips" / c
        cg = evaluate.read_cage(clip / "cage.b2ccage")
        with torch.no_grad():
            pred = model(torch.as_tensor((features(clip, a.vel) - mu) / sd, device=dev)).cpu().numpy().reshape(len(cg.names), nv, 3)
        evaluate.write_cage_like(cg, clip / f"cage_{a.tag}.b2ccage", cg.posed + pred)
        evaluate.write_cage_like(cg, clip / f"cage_{a.tag}_const.b2ccage", cg.posed + const[None])
        print(f"psd: {c}: predicted |delta| mean {np.linalg.norm(pred, axis=-1).mean() * 1000:.2f} mm", flush=True)
    torch.save({"model": model.state_dict(), "mu": mu, "sd": sd, "const": const, "args": vars(a)}, S / f"{a.tag}.pt")


if __name__ == "__main__":
    main()
