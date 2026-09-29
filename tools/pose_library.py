"""Cage-only clips of procedural motions (no render, no WAN): a pose library for b2ctrain's stretch regulariser
(tools/cage_train.py --library).

    .venv/bin/python tools/pose_library.py work/<subject> rand1 rand2 hold_rand1 ... [--device cpu]
-> <subject>/clips/lib_<motion>/cage.b2ccage

`rom[:N]` is the static range-of-motion library (b2crig/motion/rom.py: the named poses, their mirrors and N random
extreme poses, default 1500), one frame per pose, self-colliding poses dropped; poses.json holds the joint angles.
`--hands` also exercises the fingers -> clips/lib_rom_hands: the hand shapes (rom.hand_poses) and random hands on
every random pose (from their own generator, so the body poses are lib_rom's); posed with the hand slots animated.
"""
import json
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2crig import b2ctrain as B  # noqa: E402
from b2crig.motion import procedural, rom  # noqa: E402
from b2crig.rig import layered  # noqa: E402
from b2crig.rig.mhr import MHRBody  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("subject", type=Path)
ap.add_argument("motions", nargs="+")
ap.add_argument("--frames", type=int, default=81)
ap.add_argument("--device", default="cpu")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--hands", action="store_true", help="rom: random hands and the hand shapes too (-> lib_rom_hands)")
a = ap.parse_args()
body = MHRBody(a.subject / "mhr.npz", device=a.device); lay = layered.load_layers(a.subject)


def rom_clip(n_random: int):
    poses = rom.named_poses() | (rom.hand_poses() if a.hands else {})
    rng = np.random.default_rng(a.seed)
    hrng = np.random.default_rng(a.seed + 1)
    for i in range(n_random):
        poses[f"rand{i:04d}"] = rom.random_pose(rng)
        if a.hands:
            poses[f"rand{i:04d}"] |= rom.random_hand(hrng, "r") | rom.random_hand(hrng, "l")
    P0 = body.pose(rom.to_body({}, body.body0))
    radii = rom.capsule_radii(P0.verts[0].cpu().numpy(), P0.joints[0].cpu().numpy(), body.skin)
    names, dropped, chunks = list(poses), {}, []
    keep = []
    for i in range(0, len(names), 16):
        nb = names[i:i + 16]
        P = body.pose(torch.cat([rom.to_body(poses[n], body.body0) for n in nb]), hands=a.hands)
        for n, J in zip(nb, P.joints.cpu().numpy()):
            hit = rom.collides(J, radii)
            (dropped.__setitem__(n, hit) if hit else keep.append(n))
        ok = [j for j, n in enumerate(nb) if n not in dropped]
        if ok:
            chunks.append(layered.pose_layers(body, body.forward(body.model_params(
                torch.cat([rom.to_body(poses[nb[j]], body.body0) for j in ok]), hands=a.hands)), lay).cpu().numpy())
    named = [n for n in dropped if not n.startswith("rand")]
    print(f"rom: {len(keep)} poses kept, {len(dropped)} self-colliding dropped"
          + (f" (named: {', '.join(f'{n}:{dropped[n]}' for n in named)})" if named else ""))
    out = a.subject / "clips" / ("lib_rom_hands" if a.hands else "lib_rom"); out.mkdir(parents=True, exist_ok=True)
    B.write_cage(out / "cage.b2ccage", layered.cage_layers(body, lay), np.concatenate(chunks), keep)
    (out / "poses.json").write_text(json.dumps({n: poses[n] for n in keep}, indent=0))
    print(f"pose_library: {out}")


for mn in a.motions:
    if mn.split(":")[0] == "rom":
        rom_clip(int(mn.split(":")[1]) if ":" in mn else 1500)
        continue
    mot = procedural.get(mn)
    params = body.body0 + torch.as_tensor(mot.offsets(a.frames, 16.0), device=body.device)
    expr = body.expr + torch.as_tensor(mot.expr_offsets(a.frames, 16.0), device=body.device)
    posed = torch.cat([layered.pose_layers(body, body.pose(params[i:i + 16], expr=expr[i:i + 16]), lay)
                       for i in range(0, a.frames, 16)]).cpu().numpy()
    posed = layered.hair_dynamics(body, params, expr, None, posed, lay, 16.0)
    out = a.subject / "clips" / f"lib_{mn}"; out.mkdir(parents=True, exist_ok=True)
    B.write_cage(out / "cage.b2ccage", layered.cage_layers(body, lay), posed, [f"{i:04d}" for i in range(a.frames)])
    print(f"pose_library: {out}")
