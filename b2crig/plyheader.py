"""Read the records b2crunner writes into a delivered splat's PLY header: the refitted body (`b2c.mhr.*`) and the
orbit its frames were made on (`b2c.orbit.*`).

The format is specified in b2crunner's docs/ply-header-records.md; this is an independent reader of that spec (b2crig
does not import b2crunner). Values come back as numpy arrays (scalars as Python numbers), dotted keys nested one level
(`record["orbit_cameras"]["position"]`), JSON keys decoded.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MHR = "b2c.mhr."
ORBIT = "b2c.orbit."

# Per the spec: which keys are integers, which carry text or JSON; everything else numeric is a float.
_MHR_STRINGS = {"version", "model", "frame"}
_MHR_INTS = {"joint_parents"}
_ORBIT_INTS = {"version", "helix.n_frames", "helix.n_loops", "pass_frames", "anchor_frame_index",
               "orbit_cameras.image_size", "final_cameras.image_size",
               "extension.before", "extension.after", "extension.overlap_before", "extension.overlap_after"}


def read_comments(path: str | Path) -> list[str]:
    """Every header `comment` line's text (prefix stripped), in file order."""
    out = []
    with open(path, "rb") as f:
        if not f.readline().startswith(b"ply"):
            raise ValueError(f"{path} is not a PLY file")
        for raw in f:
            line = raw.decode("ascii", "replace").rstrip("\r\n")
            if line == "end_header":
                return out
            if line.startswith("comment "):
                out.append(line[len("comment "):])
    raise ValueError(f"{path}: header has no end_header")


def parse_record(comments, prefix: str, ints=frozenset(), strings=frozenset()) -> dict:
    """The lines with `prefix` as a dict (empty when there are none)."""
    out: dict = {}
    for text in comments:
        if not text.startswith(prefix):
            continue
        key, _, payload = text[len(prefix):].partition(" ")
        if key in strings:
            value = payload
        elif key == "version":   # `<prefix>version <n>`, no shape (the orbit record's is an int)
            value = int(payload)
        else:
            shape_s, _, values = payload.partition(" ")
            if shape_s == "json":
                value = json.loads(values)
            else:
                shape = () if shape_s == "1" else tuple(int(d) for d in shape_s.split("x"))
                if key in ints:
                    arr = np.array([int(v) for v in values.split()], np.int64).reshape(shape)
                else:
                    arr = np.array([float(v) for v in values.split()], np.float64).reshape(shape)
                value = arr.item() if arr.ndim == 0 else arr
        group, _, sub = key.partition(".")
        if sub:
            out.setdefault(group, {})[sub] = value
        else:
            out[key] = value
    if prefix == MHR:   # the body's pose parameters are float32 (they replay through the model as such)
        for k, v in out.get("pose_params", {}).items():
            out["pose_params"][k] = np.asarray(v, np.float32)
    return out


def read_body(path: str | Path) -> dict:
    """The `b2c.mhr.*` record. `global_rots` agree with `joints` from version 2 on; version 1 headers (before
    2026-09-29) wrote them without the raw frame's flip (the spec's caveat)."""
    rec = parse_record(read_comments(path), MHR, ints=_MHR_INTS, strings=_MHR_STRINGS)
    if not rec:
        raise ValueError(f"{path} carries no {MHR}* record")
    return rec


def read_orbit(path: str | Path) -> dict:
    """The `b2c.orbit.*` record (cameras and extras; the images beside the .ply are not read or checked)."""
    rec = parse_record(read_comments(path), ORBIT, ints=_ORBIT_INTS)
    if not rec:
        raise ValueError(f"{path} carries no {ORBIT}* record")
    return rec
