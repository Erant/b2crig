"""Procedural MHR motions: offsets from the subject's canonical pose, at a fixed fps.

A motion is a sum of smooth per-DOF curves (radians) on the named handles of
`rig.skeleton.ROT`, eased in from zero so frame 0 is the canonical pose (the
splat's own pose, which is what the WAN anchor frame shows).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..rig.skeleton import ROT, body_index

AXES = {"x": 0, "y": 1, "z": 2}


@dataclass
class Wave:
    handle: str          # e.g. "r_shoulder"
    axis: str            # x | y | z
    amp: float           # radians
    freq: float          # Hz
    phase: float = 0.0   # radians
    bias: float = 0.0    # radians, reached after the ease-in


@dataclass
class Motion:
    name: str
    waves: list[Wave] = field(default_factory=list)
    ease_s: float = 0.5

    def offsets(self, n_frames: int, fps: float) -> np.ndarray:
        """[n_frames, 130] body-param offsets."""
        t = np.arange(n_frames) / fps
        ease = np.clip(t / self.ease_s, 0, 1) if self.ease_s > 0 else np.ones_like(t)
        ease = ease * ease * (3 - 2 * ease)
        out = np.zeros((n_frames, 130), np.float32)
        for w in self.waves:
            mi = ROT[w.handle][AXES[w.axis]]
            if mi is None:
                raise ValueError(f"{w.handle} has no {w.axis} DOF")
            # sin(phase) is subtracted so the curve starts at zero; the ease keeps the start smooth.
            curve = w.amp * (np.sin(2 * math.pi * w.freq * t + w.phase) - math.sin(w.phase)) + w.bias
            out[:, body_index(mi)] += (ease * curve).astype(np.float32)
        return out

    def expr_offsets(self, n_frames: int, fps: float) -> np.ndarray:
        """[n_frames, 72] MHR expression offsets (none for a body motion)."""
        return np.zeros((n_frames, 72), np.float32)


def library() -> dict[str, Motion]:
    return {
        "arm_swing": Motion("arm_swing", [
            Wave("r_shoulder", "z", 0.5, 0.6), Wave("l_shoulder", "z", 0.5, 0.6, math.pi),
            Wave("r_elbow", "z", 0.3, 0.6), Wave("l_elbow", "z", 0.3, 0.6, math.pi)]),
        "twist": Motion("twist", [
            Wave("spine1", "y", 0.25, 0.4), Wave("spine2", "y", 0.25, 0.4), Wave("neck", "y", 0.2, 0.4, 0.5)]),
        "bend": Motion("bend", [Wave("spine1", "x", 0.2, 0.3), Wave("spine2", "x", 0.2, 0.3)]),
        "sway": Motion("sway", [
            Wave("spine1", "z", 0.15, 0.5), Wave("spine2", "z", 0.15, 0.5),
            Wave("r_hip", "z", 0.1, 0.5), Wave("l_hip", "z", 0.1, 0.5)]),
        "head_shake": Motion("head_shake", [Wave("head", "y", 0.4, 1.2), Wave("neck", "y", 0.2, 1.2)]),
        # Held poses: eased in over 1 s, then frozen for the rest of the orbit - one garment state seen from all
        # sides, the within-generation test of whether a fit is 3D-consistent (tools/static_views.py).
        "hold_arms": Motion("hold_arms", [   # mid-stride: one arm forward, one back, elbows bent
            Wave("r_shoulder", "z", 0.0, 0.0, bias=0.6), Wave("l_shoulder", "z", 0.0, 0.0, bias=-0.6),
            Wave("r_elbow", "z", 0.0, 0.0, bias=0.5), Wave("l_elbow", "z", 0.0, 0.0, bias=-0.5)], ease_s=1.0),
        "hold_twist": Motion("hold_twist", [
            Wave("spine1", "y", 0.0, 0.0, bias=0.2), Wave("spine2", "y", 0.0, 0.0, bias=0.2),
            Wave("r_hip", "x", 0.0, 0.0, bias=-0.25), Wave("r_shoulder", "x", 0.0, 0.0, bias=-0.4)], ease_s=1.0),
    }


# Handles a random motion may drive, with the largest amplitude (radians) each takes.
RANDOM_HANDLES = {
    ("r_shoulder", "x"): 0.6, ("r_shoulder", "y"): 0.4, ("r_shoulder", "z"): 0.7, ("r_elbow", "z"): 0.8,
    ("l_shoulder", "x"): 0.6, ("l_shoulder", "y"): 0.4, ("l_shoulder", "z"): 0.7, ("l_elbow", "z"): 0.8,
    ("spine1", "x"): 0.2, ("spine1", "y"): 0.3, ("spine1", "z"): 0.15,
    ("spine2", "x"): 0.2, ("spine2", "y"): 0.3, ("spine2", "z"): 0.15,
    ("neck", "y"): 0.3, ("head", "x"): 0.25, ("head", "y"): 0.5,
    ("r_hip", "x"): 0.35, ("l_hip", "x"): 0.35, ("r_hip", "z"): 0.15, ("l_hip", "z"): 0.15,
}


def random_motion(seed: int, n_waves: tuple[int, int] = (2, 6), freq: tuple[float, float] = (0.3, 1.5)) -> Motion:
    """A random sum of waves over RANDOM_HANDLES. Legs appear in pairs in antiphase (a stepping pattern) so the
    body does not drift into implausible splits; everything else is independent."""
    rng = np.random.default_rng(seed)
    keys = list(RANDOM_HANDLES)
    waves = []
    for _ in range(int(rng.integers(n_waves[0], n_waves[1] + 1))):
        h, ax = keys[int(rng.integers(len(keys)))]
        amp = float(rng.uniform(0.3, 1.0) * RANDOM_HANDLES[(h, ax)])
        f = float(rng.uniform(*freq)); ph = float(rng.uniform(0, 2 * math.pi))
        waves.append(Wave(h, ax, amp, f, ph))
        if h in ("r_hip", "l_hip", "r_knee", "l_knee") and ax in ("x", "z"):
            other = ("l_" if h.startswith("r_") else "r_") + h[2:]
            waves.append(Wave(other, ax, amp, f, ph + math.pi))
    return Motion(f"rand{seed}", waves)


@dataclass
class HeldPoses(Motion):
    """A sequence of random poses, each reached by a smoothstep over `trans_s` and then held: several garment states
    per orbit, each seen from a wide arc (the static half of the static/dynamic split)."""
    poses: list[dict] = field(default_factory=list)   # [{(handle, axis): radians}]
    trans_s: float = 0.6

    def offsets(self, n_frames: int, fps: float) -> np.ndarray:
        out = np.zeros((n_frames, 130), np.float32)
        seg = n_frames / len(self.poses)
        prev = np.zeros(130, np.float32)
        for k, pose in enumerate(self.poses):
            tgt = np.zeros(130, np.float32)
            for (h, ax), v in pose.items():
                tgt[body_index(ROT[h][AXES[ax]])] += v
            a, b = int(round(k * seg)), int(round((k + 1) * seg))
            e = np.clip((np.arange(b - a) / fps) / self.trans_s, 0, 1) if self.trans_s > 0 else np.ones(b - a)
            e = e * e * (3 - 2 * e)
            out[a:b] = prev + e[:, None] * (tgt - prev)
            prev = tgt
        return out


def arm_lift() -> HeldPoses:
    """Arms lifted from the A pose to a Y, then swung slowly between a T and a Y (and a little above): the armpit
    crease opening and closing through an orbit (moving arms keep WAN drawing skin where a held pose did not)."""
    ys = [1.9, 1.1, 2.1, 1.3]
    return HeldPoses("arm_lift", poses=[{("r_shoulder", "y"): v, ("l_shoulder", "y"): v} for v in ys], trans_s=1.0)


def held_random(seed: int, n_poses: int = 2) -> HeldPoses:
    """`n_poses` random poses over RANDOM_HANDLES (3-7 handles each, legs mirrored as in random_motion)."""
    rng = np.random.default_rng(10_000 + seed)
    keys = list(RANDOM_HANDLES)
    poses = []
    for _ in range(n_poses):
        pose: dict = {}
        for _ in range(int(rng.integers(3, 8))):
            h, ax = keys[int(rng.integers(len(keys)))]
            v = float(rng.choice((-1.0, 1.0)) * rng.uniform(0.5, 1.0) * RANDOM_HANDLES[(h, ax)])
            pose[(h, ax)] = v
            if h in ("r_hip", "l_hip") and ax in ("x", "z"):
                pose[(("l_" if h.startswith("r_") else "r_") + h[2:], ax)] = -v
        poses.append(pose)
    return HeldPoses(f"hold_rand{seed}", poses=poses)


def held_from_file(path: str, trans_s: float = 1.0) -> HeldPoses:
    """One pose from an npz with `dof` ("handle.axis") and `x` (radians), e.g. tools/reveal_pose.py's pose.npz; eased
    in over `trans_s` (0: held from the first frame)."""
    d = np.load(path)
    pose = {tuple(str(k).split(".")): float(v) for k, v in zip(d["dof"], d["x"])}
    return HeldPoses(f"hold_file:{path}", poses=[pose], trans_s=trans_s)


# MHR expression dimensions (tools/face_probe.py, 2026-09-25): all activations, used >= 0.
FACE_DIMS = {
    "jaw": [24],
    "lips": [26, 27, 30, 31, 32, 33, 38, 39, 40, 54, 55, 58, 59, 60, 61, 62, 63, 64, 65],
    "smile": [32, 33],      # lip corners up, closed lips (face-probed 2026-09-26; 56/57 are a pucker, not a smile)
    "grin": [70, 71],       # lip corners out with parted lips
    "pucker": [56, 57],
    "brows": [0, 1],
    "blink": [12, 13],
}


@dataclass
class FaceMotion(Motion):
    """Facial animation: smooth activations of visemes (jaw + lip shapes), smiles and brow raises, blinks as short
    pulses on both eyelids, and a little head motion (the waves). Close-up clips (b2crig face rig)."""
    expr_waves: list = field(default_factory=list)     # (dim, amp, freq Hz, phase)
    blinks: list = field(default_factory=list)          # blink times, seconds

    def expr_offsets(self, n_frames: int, fps: float) -> np.ndarray:
        t = np.arange(n_frames) / fps
        ease = np.clip(t / self.ease_s, 0, 1) if self.ease_s > 0 else np.ones_like(t)
        ease = ease * ease * (3 - 2 * ease)
        out = np.zeros((n_frames, 72), np.float32)
        for d, amp, f, ph in self.expr_waves:
            out[:, d] += (ease * amp * 0.5 * (1 - np.cos(2 * math.pi * f * t + ph))).astype(np.float32)
        for tb in self.blinks:
            pulse = np.exp(-0.5 * ((t - tb) / 0.07) ** 2)
            for d in FACE_DIMS["blink"]:
                out[:, d] += (0.9 * pulse).astype(np.float32)
        return out


def face_random(seed: int, talk: float = 1.0) -> FaceMotion:
    rng = np.random.default_rng(20_000 + seed)
    ew = [(24, float(rng.uniform(0.2, 0.45) * talk), float(rng.uniform(1.5, 3.0)), float(rng.uniform(0, 6.28)))]
    for d in rng.choice(FACE_DIMS["lips"], size=4, replace=False):
        ew.append((int(d), float(rng.uniform(0.2, 0.5) * talk), float(rng.uniform(0.8, 2.5)), float(rng.uniform(0, 6.28))))
    for d in FACE_DIMS["smile"]:
        ew.append((d, float(rng.uniform(0.2, 0.6)), float(rng.uniform(0.15, 0.4)), float(rng.uniform(0, 6.28))))
    b = float(rng.uniform(0.3, 0.7))
    for d in FACE_DIMS["brows"]:
        ew.append((d, b, 0.25, float(rng.uniform(0, 6.28))))
    blinks = sorted(rng.uniform(1.5, 3.5, size=int(rng.integers(2, 4))).tolist())   # while the orbit faces the subject
    head = [Wave("neck", "y", float(rng.uniform(0.05, 0.12)), float(rng.uniform(0.2, 0.4)), float(rng.uniform(0, 6.28))),
            Wave("head", "x", float(rng.uniform(0.03, 0.08)), float(rng.uniform(0.3, 0.6)), float(rng.uniform(0, 6.28)))]
    return FaceMotion(f"face{seed}", waves=head, expr_waves=ew, blinks=blinks)


def get(name: str) -> Motion:
    if name.startswith("face"):
        return face_random(int(name[4:]))
    if name.startswith("hold_file:"):
        return held_from_file(name[len("hold_file:"):])
    if name == "arm_lift":
        return arm_lift()
    if name.startswith("hold_now:"):   # held from frame 0 (an armpit reveal: the pose itself is the point)
        return held_from_file(name[len("hold_now:"):], trans_s=0.0)
    if name.startswith("hold_rand"):
        return held_random(int(name[9:]))
    if name.startswith("rand"):
        return random_motion(int(name[4:]))
    return library()[name]


def describe(m: Motion) -> str:
    """Words for the WAN prompt: the handles that move, loosely."""
    if isinstance(m, FaceMotion):
        return "talking and making facial expressions: speaking, smiling, blinking and raising the eyebrows, moving the head slightly"
    if isinstance(m, HeldPoses):
        return "holding a still pose" if len(m.poses) == 1 else "holding still poses, changing pose once"
    if all(w.amp == 0 for w in m.waves):
        return "holding a still pose"
    parts = sorted({w.handle.replace("r_", "right ").replace("l_", "left ").replace("spine1", "torso").replace("spine2", "torso") for w in m.waves})
    return "moving the " + ", ".join(parts) + " rhythmically"
