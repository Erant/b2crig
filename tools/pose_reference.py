"""A pose reference for the capture's image model: a generic MHR body (zero shape, zero bone-length scales, neutral
expression) in a requested capture pose, rendered with body2colmap as two 9:16 frames joined side by side, a character turnaround's front | back:
--layout skel-fb (DWPose skeletons, body2colmap style="dwpose", OpenPose body25+hands, on black), mesh-fb (the plain
shaded mesh), or skel-mesh (skeleton | mesh, both from the front).

    .venv/bin/python tools/pose_reference.py work/<subject> OUT.png [--arm-elev 45] [--elbow 5] [--width 720 --height 1280]
        [--mesh-bg 0,0,0] [--fill 0.9]

The pose starts from the subject's capture pose (legs, feet, hands, spine) and re-solves the arms by IK through the
differentiable MHR model: both arms straight (elbows bent --elbow degrees), raised --arm-elev degrees above horizontal
in the frontal plane, palms facing forward (the palm normal's sign taken from the capture). The cameras look at the
body from the front (+Z) and from behind (-Z), framed to fill --fill of the frame's limiting dimension.
Rendering runs in b2crunner's venv (body2colmap + pyrender), like tools/draw_control.py.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
M_HAND = {"r": {"wrist": 42, "index": 56, "pinky": 44}, "l": {"wrist": 78, "index": 92, "pinky": 80}}


def solve(a) -> dict:
    import torch
    sys.path.insert(0, str(ROOT))
    from b2crig.rig.mhr import MHRBody, SCALES
    from b2crig.rig.skeleton import JOINT, ROT
    body = MHRBody(a.subject / "mhr.npz", device="cpu")
    shape = torch.zeros_like(body.shape)
    expr = torch.zeros_like(body.expr)
    mp0 = body.model_params0.clone()
    if not a.subject_shape:
        mp0[:, SCALES] = 0.0

    def fwd(mp):
        v, skel = body.mhr(shape, mp, expr)
        return (v / 100) @ body._A.T + body._t, (skel[..., :3] / 100) @ body._A.T + body._t

    with torch.no_grad():
        _, J0 = fwd(mp0)
    J0 = J0[0]

    def palm(J, s):
        h = M_HAND[s]
        return torch.linalg.cross(J[h["index"]] - J[h["wrist"]], J[h["pinky"]] - J[h["wrist"]])

    sgn = {s: float(torch.sign(palm(J0, s)[2])) for s in "rl"}   # the capture's palms face forward (+Z)
    idx = [i for s in "rl" for n in (f"{s}_shoulder", f"{s}_elbow", f"{s}_wrist") for i in ROT[n] if i is not None]
    idx = torch.as_tensor(idx)
    delta = torch.zeros(len(idx), requires_grad=True)
    opt = torch.optim.Adam([delta], lr=0.02)
    e = np.radians(a.arm_elev)
    tgt = {"r": torch.tensor([-np.cos(e), np.sin(e), 0.0], dtype=torch.float32),
           "l": torch.tensor([np.cos(e), np.sin(e), 0.0], dtype=torch.float32)}
    z = torch.tensor([0.0, 0.0, 1.0])
    half = np.radians(a.elbow) / 2   # the upper arm and forearm each a half-bend off the arm line
    for it in range(a.iters):
        mp = mp0.clone(); mp[0, idx] = mp0[0, idx] + delta
        _, J = fwd(mp); J = J[0]
        loss = 0.0
        for s in "rl":
            sh, el, wr = (J[JOINT[f"{s}_{k}"]] for k in ("shoulder", "elbow", "wrist"))
            u = (el - sh) / (el - sh).norm(); f = (wr - el) / (wr - el).norm()
            loss = loss + (u @ tgt[s] - np.cos(half)) ** 2 + (f @ tgt[s] - np.cos(half)) ** 2
            n = palm(J, s); n = n / n.norm()
            loss = loss + 0.5 * (1 - sgn[s] * (n @ z))
            loss = loss + 0.5 * (u[2] ** 2 + f[2] ** 2)   # in the frontal plane
        loss = loss + 1e-4 * (delta ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        mp = mp0.clone(); mp[0, idx] = mp0[0, idx] + delta
        V, J = fwd(mp); V, J = V[0].numpy(), J[0].numpy()
    from b2crig.rig import mhr70
    K = mhr70.mapping()
    kps = K @ np.concatenate([V, J], 0)
    for s in "rl":
        sh, el, wr = (J[JOINT[f"{s}_{k}"]] for k in ("shoulder", "elbow", "wrist"))
        u, f = el - sh, wr - el
        bend = np.degrees(np.arccos(u @ f / np.linalg.norm(u) / np.linalg.norm(f)))
        elev = np.degrees(np.arctan2((wr - sh)[1], abs((wr - sh)[0])))
        n = np.cross(J[M_HAND[s]["index"]] - J[M_HAND[s]["wrist"]], J[M_HAND[s]["pinky"]] - J[M_HAND[s]["wrist"]])
        print(f"{s} arm: elbow bend {bend:.1f} deg, raised {elev:.1f} deg, palm normal z {sgn[s] * n[2] / np.linalg.norm(n):+.2f} (1 = forward)")
    return {"verts": V.astype(np.float32), "faces": body.faces.astype(np.int32), "kps": kps.astype(np.float32)}


def render(a) -> None:
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import cv2
    from body2colmap.camera import Camera
    from body2colmap.renderer import Renderer
    from body2colmap.scene import Scene
    d = np.load(a.posed)
    V, F, K = d["verts"], d["faces"], d["kps"]
    W, H = a.width, a.height
    lo, hi = V.min(0), V.max(0)
    c = (lo + hi) / 2
    fx = 20.0 * H / 2   # ~6 deg vertical field of view, far away: near-orthographic, so front and back match in scale
    dist = max((hi[0] - lo[0]) / 2 * fx / (a.fill * W / 2), (hi[1] - lo[1]) / 2 * fx / (a.fill * H / 2))
    sc = Scene(V, F, skeleton_joints=K, skeleton_format="mhr70")
    r = Renderer(sc, render_size=(W, H))
    bg = tuple(float(x) for x in a.mesh_bg.split(","))
    mc = None if a.mesh_color == "default" else tuple(float(x) for x in a.mesh_color.split(","))

    def rgb(img, bgc):
        x = img[..., :3].astype(np.float32); al = img[..., 3:].astype(np.float32) / 255
        return (x * al + np.array(bgc) * 255 * (1 - al)).astype(np.uint8)

    def view(side):   # camera -> world, columns right, up, back; front looks down -Z from +Z, back down +Z from -Z
        s = 1.0 if side == "front" else -1.0
        pos = np.array([c[0], c[1], c[2] + s * dist], np.float32)
        rot = np.diag([s, 1.0, s]).astype(np.float32)
        return Camera(focal_length=(fx, fx), image_size=(W, H), principal_point=(W / 2, H / 2), position=pos, rotation=rot)

    def skeleton(side):   # DWPose drops what its detector could not see: joints behind the body, the face from behind
        return rgb(r.render_skeleton(view(side), target_format="openpose_body25_hands", style="dwpose", joint_radius=0.005,
                                     bone_radius=0.005, bg_color=(0.0, 0.0, 0.0), face_mode="points", occlusion_tolerance=0.12),
                   (0, 0, 0))

    def mesh(side):
        return rgb(r.render_mesh(view(side), mesh_color=mc, bg_color=bg), bg)

    panels = {"skel-mesh": (skeleton("front"), mesh("front")), "skel-fb": (skeleton("front"), skeleton("back")),
              "mesh-fb": (mesh("front"), mesh("back"))}[a.layout]
    r.delete()
    out = np.hstack(panels)
    cv2.imwrite(str(a.out), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print(f"pose_reference: {a.out} ({out.shape[1]}x{out.shape[0]})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("subject", type=Path, nargs="?")
    ap.add_argument("out", type=Path)
    ap.add_argument("--arm-elev", type=float, default=45.0)
    ap.add_argument("--elbow", type=float, default=5.0)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=1280)
    ap.add_argument("--fill", type=float, default=0.9)
    ap.add_argument("--mesh-bg", default="0,0,0")
    ap.add_argument("--mesh-color", default="0.8,0.8,0.8", help="r,g,b in 0-1, or 'default' (body2colmap's)")
    ap.add_argument("--subject-shape", action="store_true", help="keep the subject's bone-length scales (not generic)")
    ap.add_argument("--layout", default="skel-fb", choices=["skel-fb", "mesh-fb", "skel-mesh"],
                    help="left | right panels: skeleton front | back, mesh front | back, or skeleton | mesh (front)")
    ap.add_argument("--posed", type=Path, help="(internal) render this posed npz")
    a = ap.parse_args()
    if a.posed:
        render(a); return
    posed = a.out.with_suffix(".posed.npz")
    np.savez(posed, **solve(a))
    sys.path.insert(0, str(ROOT))
    from b2crig.b2crunner import MAIN_PYTHON
    subprocess.run([str(MAIN_PYTHON), __file__, a.out, "--posed", str(posed), "--width", str(a.width), "--height",
                    str(a.height), "--fill", str(a.fill), "--mesh-bg", a.mesh_bg, "--mesh-color", a.mesh_color, "--layout", a.layout], check=True)


if __name__ == "__main__":
    main()
