"""The b2ctrain side: cage files and the binary's subcommands.

Everything that rasterises runs in b2ctrain; this module only writes its inputs
and invokes it. The cage layout is documented in b2ctrain's src/gpu/cage.h.
Where the line between the two repos runs: BOUNDARY.md at this repo's root.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

B2CTRAIN = Path(os.environ.get("B2CTRAIN", Path.home() / "Projects" / "b2ctrain" / "build" / "b2ctrain"))
# Bound splats follow their cage triangle's size change only within 1/G..G: LBS stretches armpit / elbow triangles 2-4x
# and unclamped splats bloom off the skin there (2026-09-26). Every cage render and cage training uses it.
CAGE_MAX_GROWTH = float(os.environ.get("B2C_CAGE_MAX_GROWTH", "1.15"))
# fit-cage's three label groups (Sapiens2 classes): hair | upper clothing + apparel | lower clothing. b2ctrain has no
# default: which classes mean what is the rig's business.
FIT_GROUPS = ((4,), (23, 1), (13,))


def fit_groups_args(groups: Sequence[Sequence[int]] = FIT_GROUPS) -> list[str]:
    """`--groups` for b2ctrain fit-cage."""
    return ["--groups", ";".join(",".join(str(int(c)) for c in g) for g in groups)]


def ply_has_open_states(splat: str | Path) -> bool:
    """The splat carries --cage-open's open_* states (its cage then needs the opening gate, rig/open_gate.py)."""
    with open(splat, "rb") as f:
        head = f.read(1 << 16)
    return b"property float open_r" in head.split(b"end_header")[0]


@dataclass
class CageLayer:
    name: str
    verts: np.ndarray        # [V, 3] canonical, world
    faces: np.ndarray        # [F, 3] local vertex indices
    classes: Sequence[int]   # Sapiens2 classes this layer owns


def write_cage(path: str | Path, layers: Sequence[CageLayer], posed: np.ndarray, names: Sequence[str]) -> None:
    """`posed` [n_frames, n_verts_total, 3]: every layer's vertices, concatenated in layer order."""
    nv = sum(len(L.verts) for L in layers)
    nf = sum(len(L.faces) for L in layers)
    posed = np.asarray(posed, np.float32)
    if posed.shape[1:] != (nv, 3) or len(names) != len(posed):
        raise ValueError(f"write_cage: posed is {posed.shape}, expected ({len(names)}, {nv}, 3)")
    with open(path, "wb") as f:
        f.write(b"B2CCAGE1")
        f.write(np.array([len(layers), nv, nf, len(posed)], np.int32).tobytes())
        v_off = f_off = 0
        for L in layers:
            bits = 0
            for c in L.classes:
                bits |= 1 << int(c)
            f.write(np.array([v_off, len(L.verts), f_off, len(L.faces)], np.int32).tobytes())
            f.write(np.array([bits], np.uint32).tobytes())
            v_off += len(L.verts)
            f_off += len(L.faces)
        f.write(np.concatenate([np.asarray(L.verts, np.float32) for L in layers]).tobytes())
        off, faces = 0, []
        for L in layers:
            faces.append(np.asarray(L.faces, np.int32) + off)
            off += len(L.verts)
        f.write(np.concatenate(faces).astype(np.int32).tobytes())
        for nm in names:
            b = nm.encode()[:63]
            f.write(b + b"\0" * (64 - len(b)))
        f.write(posed.tobytes())


def render(splat: str | Path, cameras: str | Path, out_dir: str | Path, *, cage: str | Path | None = None,
           class_mask: Sequence[int] | None = None, label_maps: bool = False,
           background: Sequence[float] = (1.0, 1.0, 1.0), extra: Sequence[str] = ()) -> None:
    cmd = [str(B2CTRAIN), "render", "--splat", str(splat), "--cameras", str(cameras), "--output-dir", str(out_dir),
           "--background", ",".join(f"{x:g}" for x in background)]
    if cage:
        cmd += ["--cage", str(cage), "--cage-max-growth", f"{CAGE_MAX_GROWTH:g}"]
        if ply_has_open_states(splat):   # two-state splats: the gate is ours to compute, b2ctrain only reads it
            from .rig import open_gate
            if not open_gate.has_open_gate(cage):
                open_gate.add_open_gate(cage)
    if class_mask:
        cmd += ["--class-mask", ",".join(str(int(c)) for c in class_mask)]
    if label_maps:
        cmd += ["--label-maps"]
    cmd += list(extra)
    subprocess.run(cmd, check=True)
