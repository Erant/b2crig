"""Build one training clip: motion + circular orbit -> cage -> b2ctrain control render -> WAN inputs.

    python -m b2crig.gen.clips --subject work/416580 --run <b2crunner run dir> --motion arm_swing --out clips/0001

Frame 0 is the canonical pose rendered exactly and is KEPT by WAN (mask 0), so the
clip starts from the splat's own appearance; every other frame is generated
under the control render (mask 255), optionally only inside the dilated loose
region (--loose-only), where clothing and hair may deviate from the rigid bind.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.home() / "Projects" / "b2crunner"))
from pipeline.orbit_record import read_orbit_record  # noqa: E402

from .. import b2ctrain as B  # noqa: E402
from .. import cameras as C  # noqa: E402
from ..io.splat import LOOSE_CANDIDATES  # noqa: E402
from ..motion import procedural  # noqa: E402
from ..rig.mhr import MHRBody  # noqa: E402
from ..rig import layered  # noqa: E402

WAN_PYTHON = Path.home() / "Projects" / "b2crunner" / "pipeline" / "envs" / "wan22" / "venv" / "bin" / "python"
TOOLS = Path(__file__).resolve().parents[2] / "tools"

PROMPT = (
    "The camera orbits smoothly at a constant height and fixed focal length around $SUBJECT_DESC$, "
    "who is standing in place in an empty studio with a plain grey background and is moving: {motion}. "
    "The clothing responds to the movement with natural weight and inertia: fabric swings, folds and "
    "settles, loose straps and hems sway and lag behind the body, hair bounces and follows the head. "
    "Soft, fixed studio lighting. The face, clothing, colours and materials stay identical in every frame. "
    "Photorealistic, sharp focus."
)
NEGATIVE = ("static, frozen, still image, blurry, low quality, jpeg artifacts, deformed, extra limbs, "
            "extra fingers, changing clothes, changing colours, flicker, text, watermark, busy background")
MOTION_TEXT = {
    "arm_swing": "swinging both arms back and forth", "twist": "twisting the upper body from side to side",
    "bend": "bending forward at the waist and straightening up", "sway": "swaying the hips and torso from side to side",
    "head_shake": "shaking the head from side to side",
}


def rotation_prep(CV: np.ndarray, CF: np.ndarray, radius: float = 0.12, min_dist: float = 0.03) -> dict:
    """The canonical-pose part of rotated_vertices (vertex pairs, frames), computed once per cage.

    The same measure as b2crig/rig/open_gate.py (the gate b2ctrain --cage-open reads from the cage), over EVERY partner
    as flat pairs (open_gate keeps b2ctrain's 64 per vertex): grey_unseen only thresholds it per frame."""
    from scipy.spatial import cKDTree
    n = len(CV)
    first = np.full(n, -1, np.int64)           # one incident triangle per vertex (for its frame)
    for k in range(3):
        idx = CF[:, k]; m = first[idx] < 0; first[idx[m]] = np.nonzero(m)[0]
    has = first >= 0
    pairs = cKDTree(CV).query_pairs(radius, output_type="ndarray")
    d = np.linalg.norm(CV[pairs[:, 0]] - CV[pairs[:, 1]], axis=1); pairs = pairs[d > min_dist]
    pairs = np.concatenate([pairs, pairs[:, ::-1]])                                    # both directions
    a, b = pairs[:, 0], pairs[:, 1]; ok = has[a]; a, b = a[ok], b[ok]
    tri = CF[first[has]]
    return {"n": n, "tri": tri, "ids": np.nonzero(has)[0], "a": a, "b": b, "F0T": np.transpose(_tri_frames(CV, tri), (0, 2, 1)),
            "off": CV[b] - CV[a]}


def _tri_frames(V: np.ndarray, tri: np.ndarray) -> np.ndarray:
    e1 = V[tri[:, 1]] - V[tri[:, 0]]; e2 = V[tri[:, 2]] - V[tri[:, 0]]
    x = e1 / np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-9)
    z = np.cross(x, e2); z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-9)
    return np.stack([x, np.cross(z, x), z], -1)                                        # [m, 3, 3] columns


def rotated_vertices(prep: dict, P: np.ndarray, deg: float) -> np.ndarray:
    """bool [n_verts]: cage vertices whose neighbours (canonical distance min_dist..radius) moved by more than `deg`
    degrees relative to the vertex's own triangle frame between the canonical cage and the posed cage P (the
    b2ctrain --cage-open gate: a joint between them has turned)."""
    R = _tri_frames(P, prep["tri"]) @ prep["F0T"]                                      # canonical frame -> posed frame
    a, b = prep["a"], prep["b"]
    Ra = R[np.searchsorted(prep["ids"], a)]
    pred = np.einsum("nij,nj->ni", Ra, prep["off"]); act = P[b] - P[a]
    cos = (pred * act).sum(1) / np.maximum(np.linalg.norm(pred, axis=1) * np.linalg.norm(act, axis=1), 1e-9)
    out = np.zeros(prep["n"], np.uint8)
    np.maximum.at(out, a, (cos < np.cos(np.radians(deg))).astype(np.uint8))
    return out.astype(bool)


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    import cv2
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask, k)


def build(subject: Path, run: Path, motion: str, out: Path, *, n_frames: int = 81, fps: float = 16.0,
          elevation: float = 5.0, sweep: float = 360.0, azimuth0: float = 0.0, width: int = 480, height: int = 848,
          loose_only: bool = False, dilate_px: int = 12, strength: list[float] | None = None, seed: int = 0,
          layers: bool = True, elevation_end: float | None = None, grey_unseen: bool = False,
          prompt_extra: str = "", target_joint: str | None = None, radius: float | None = None,
          target_offset: tuple = (0.0, 0.0, 0.0), reference: str = "front", motion_text: str = "",
          hair_params=None, splat: str | None = None, control: str = "splat", prompt: str | None = None,
          stretch_grey: float = 1.35, grey_dilate: int = 4, grey_fill: str = "grey", background=(0.5, 0.5, 0.5),
          local_mask: bool = False, rotate_grey: float = 30.0, stretch_mode: str = "area") -> None:
    """`reference`: WAN's identity reference - "front" (the run's front.png) or "face" (<subject>/face_ref.png,
    tools/face_reference.py: a face crop, better for close-ups).
    `target_joint` / `radius` / `target_offset`: orbit that joint's canonical position (plus the offset, metres) at that
    distance instead of the subject's own orbit - e.g. a face close-up ("head", 0.5, (0, 0.05, 0)).
    `elevation_end`: a helical orbit from `elevation` to it (default: circular at `elevation`).
    `grey_unseen`: grey out (0.5, which VACE reads as a region to fill) what the capture does not vouch for, so WAN
    inpaints it rather than elaborating the leftover gaussians: the control renders with b2ctrain's per-splat confidence
    gate (stretched haze of weakly supported splats), and the pixels where body surface the capture never saw lands are
    blanked on top (confident but occluded surface, e.g. under the hand resting at the pocket). grey_unseen="unseen":
    only the unseen-surface blanking, on the plain render (the confidence gate greys whole limbs in dance poses and
    WAN then moves them). grey_unseen="stretch": that, plus the cage surface (every layer) LBS stretches beyond `stretch_grey` x
    its canonical triangle size (an A-pose armpit crease opening as the arm lifts: its baked shadow smears into a blotch
    that WAN elaborates into hair). Under "stretch" the kept frame 0 is greyed too and its mask opens exactly there.
    grey_unseen="rotate": "stretch" plus the surface whose cage neighbourhood (3-12 cm) rotated by more than `rotate_grey`
    degrees relative to its own triangle since the canonical pose (crease walls that turn apart without stretching).
    `stretch_mode`: "area" (triangle size ratio) or "principal" (largest principal stretch; catches shear).
    `grey_fill`: "grey" (0.5, VACE's fill convention) or "inpaint" (cv2 Telea from the surrounding render colour).
    `background`: the colour WAN's control is composited on (holes = alpha 0 take it too).
    `local_mask`: under grey_unseen, every frame's VACE mask opens only on the holes (pure inpainting).
    `grey_dilate`: px the greyed region grows by (a fringe of the defect left around a small hole regrows it).
    `splat`: the splat rendered into the control (default: the run's scene.ply), e.g. a retrained one.
    `control`: "splat" = the cage-posed splat render in every frame (frame 0 kept); "flf" = first and last frames are
    the splat render, KEPT, and the frames between are body2colmap outline+skeleton drawings of the posed body
    (tools/draw_control.py, b2crunner's capture look) that WAN fills in freely; "draw" = frame 0 kept, drawings after.
    Under flf/draw the splat render of every frame is kept in lbs/ (for comparisons).
    `prompt`: replaces the whole PROMPT (its {motion} is still filled)."""
    out.mkdir(parents=True, exist_ok=True)
    ply = Path(splat) if splat else run / "ply" / "scene.ply"
    rec = read_orbit_record(run / "ply" / "scene.ply")
    K, target, radius0 = C.orbit_from_record(rec)
    s = width / K.width
    K = C.Intrinsics(width, height, K.fx * s, K.fy * height / K.height, K.cx * s, K.cy * height / K.height)
    names = [f"{i:04d}" for i in range(n_frames)]
    body = MHRBody(subject / "mhr.npz")
    if target_joint:
        from ..rig.skeleton import JOINT
        target = body.pose().joints[0, JOINT[target_joint]].cpu().numpy() + np.asarray(target_offset, float)
    radius = float(radius) if radius else radius0
    orbit = (C.circular_orbit(target, radius, elevation, n_frames, azimuth0, sweep) if elevation_end is None
             else C.helical_orbit(target, radius, elevation, elevation_end, n_frames, azimuth0, sweep))

    gtrans = None
    if motion.startswith("npz:"):   # a solved motion (tools/make_walk.py): npz:<path>[@start frame], its fps must match
        path, _, st = motion[4:].partition("@")
        d = np.load(path); st = int(st or 0)
        sel = np.minimum(np.arange(st, st + n_frames), len(d["body_params"]) - 1)
        from ..motion import io as MI
        mfile = MI.Motion(d, sel)
        params = torch.as_tensor(mfile.body_params, device=body.device)
        expr = torch.as_tensor(mfile.expr, device=body.device)
        gtrans = None if mfile.global_trans is None else torch.as_tensor(mfile.global_trans, device=body.device)
        # the camera follows the root (horizontally, smoothed), so the subject walks on the spot in frame
        rw = (d["root_world"][sel] if "root_world" in d.files else mfile.root_t).copy(); rw[:, 1] = 0.0
        k = 5; rw = np.stack([np.convolve(np.pad(rw[:, j], (k, k), mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), "valid") for j in range(3)], 1)
        orbit = [(rot, pos + rw[i]) for i, (rot, pos) in enumerate(orbit)]
        mot = None
    else:
        mot = procedural.get(motion)
        params = body.body0 + torch.as_tensor(mot.offsets(n_frames, fps), device=body.device)
        expr = body.expr + torch.as_tensor(mot.expr_offsets(n_frames, fps), device=body.device)
    C.write_cameras(out / "cameras.json", K, orbit, names)
    lay = layered.load_layers(subject) if layers else []
    if mot is None:
        posed = layered.pose_motion(body, mfile, lay, hair_params)
    else:
      posed = torch.cat([layered.pose_layers(body, body.pose(params[i:i + 16], expr=expr[i:i + 16],
                                                           global_trans=None if gtrans is None else gtrans[i:i + 16]), lay)
                       for i in range(0, n_frames, 16)]).cpu().numpy()
      posed = layered.hair_dynamics(body, params, expr, gtrans, posed, lay, fps, hair_params)
    if mot is None:
        np.savez(out / "motion.npz", motion=motion, **mfile.save_fields())
    else:
        np.savez(out / "motion.npz", body_params=params.cpu().numpy(), expr=expr.cpu().numpy(), fps=fps, motion=motion)
    B.write_cage(out / "cage.b2ccage", layered.cage_layers(body, lay), posed, names)

    ctrl = out / "control"
    if ctrl.exists():
        shutil.rmtree(ctrl)
    B.render(ply, out / "cameras.json", ctrl, cage=out / "cage.b2ccage", class_mask=LOOSE_CANDIDATES, label_maps=True)
    if grey_unseen is True:   # the frames from the confidence-gated render (its own pass: it needs the feature channel)
        cdir = out / "control_conf"
        B.render(ply, out / "cameras.json", cdir, cage=out / "cage.b2ccage", extra=["--confidence"])
        for nm in names:
            shutil.copy(cdir / f"{nm}.png", ctrl / f"{nm}.png")
        shutil.rmtree(cdir)

    import cv2
    keep = {0}
    if control in ("flf", "draw"):
        draw_frames(body, mfile if mot is None else None, params, expr, gtrans, out, n_frames)
        lbs = out / "lbs"
        if lbs.exists():
            shutil.rmtree(lbs)
        ctrl.rename(lbs)
        ctrl.mkdir()
        keep = {0, n_frames - 1} if control == "flf" else {0}
        for nm in names:
            for f in lbs.glob(f"{nm}.*"):
                if f.name != f"{nm}.png":
                    shutil.copy(f, ctrl / f.name)   # label/mask sidecars stay with the frame
        for i, nm in enumerate(names):
            shutil.copy((lbs if i in keep else out / "drawing") / f"{nm}.png", ctrl / f"{nm}.png")
    (out / "mask").mkdir(exist_ok=True)
    for i, nm in enumerate(names):
        if i in keep:
            m = np.zeros((height, width), np.uint8)
        elif loose_only:
            m = dilate((cv2.imread(str(ctrl / f"{nm}.mask.png"), cv2.IMREAD_GRAYSCALE) > 64).astype(np.uint8) * 255, dilate_px)
        else:
            m = np.full((height, width), 255, np.uint8)
        cv2.imwrite(str(out / "mask" / f"{nm}.png"), m)
    if grey_unseen:
        from .. import visibility as VI
        seen = VI.capture_seen(subject, run, body.pose().verts[0].cpu().numpy())
        intr, cams = VI.json_cams(out / "cameras.json")
        nb = len(body.verts_canon)
        cl = layered.cage_layers(body, lay)
        offs = np.cumsum([0] + [len(L.verts) for L in cl])
        CV = np.concatenate([L.verts for L in cl])
        CF = np.concatenate([L.faces.astype(np.int64) + o for L, o in zip(cl, offs) if L.name not in ("hair", "face")])
        # the hair shell stretches with its dynamics all the time and its splats are not crease shading; the face must
        # never be redrawn (identity), and neck triangles stretch as the head turns: only surface below the neck counts
        from ..rig.skeleton import JOINT
        # "below the neck" = outside the head region (above the neck AND within 15 cm of it horizontally): a plain height
        # test drops the raised arms of an arms-up canonical (2bccb7)
        nk = body.joints_canon[JOINT["neck"]]
        below_neck = (CV[:, 1] < nk[1] - 0.03) | (np.linalg.norm((CV - nk)[:, [0, 2]], axis=1) > 0.15)
        tri_size_g = lambda V: np.sqrt(np.linalg.norm(np.cross(V[CF[:, 1]] - V[CF[:, 0]], V[CF[:, 2]] - V[CF[:, 0]]), axis=-1))
        ck0 = np.maximum(tri_size_g(CV), 1e-9)
        if stretch_mode == "principal":   # the largest principal stretch of each triangle's 2D deformation gradient:
            # a shoulder whose arm comes down from an arms-up capture shears (1.8x one way, 0.4x across) at ~constant
            # area, which the size ratio misses
            e1, e2 = CV[CF[:, 1]] - CV[CF[:, 0]], CV[CF[:, 2]] - CV[CF[:, 0]]
            n0 = np.cross(e1, e2); n0 /= np.maximum(np.linalg.norm(n0, axis=1, keepdims=True), 1e-12)
            u0 = e1 / np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-12); w0 = np.cross(n0, u0)
            X = np.stack([np.stack([(e1 * u0).sum(1), (e1 * w0).sum(1)], 1), np.stack([(e2 * u0).sum(1), (e2 * w0).sum(1)], 1)], 2)
            Xi = np.linalg.pinv(X)

            def stretch_of(V):
                Y = np.stack([V[CF[:, 1]] - V[CF[:, 0]], V[CF[:, 2]] - V[CF[:, 0]]], 2)
                return np.linalg.svd(Y @ Xi, compute_uv=False)[:, 0]
        else:
            stretch_of = lambda V: tri_size_g(V) / ck0
        rprep = rotation_prep(CV, CF) if grey_unseen == "rotate" else None
        for i, nm in enumerate(names):
            if i == 0 and grey_unseen not in ("stretch", "rotate"):
                continue
            if grey_unseen in ("stretch", "rotate"):   # also the surface LBS stretches: crease shading baked into it smears
                bad = np.zeros(len(CV), bool)   # every cage layer (a bra edge dragged into an opening armpit too)
                bad[CF[stretch_of(posed[i]) > stretch_grey].ravel()] = True
                if grey_unseen == "rotate":   # and the surface whose neighbourhood turned relative to it (a joint between
                    # them opened: the crease WALLS, which rotate apart without stretching -- the two-state gate's signal)
                    bad |= rotated_vertices(rprep, posed[i], rotate_grey)
                bad &= below_neck
                # unseen surface counts below the neck only as well: a few never-captured head vertices (inner ear,
                # nostrils) dilated by a big grey_dilate blank the whole head and WAN redraws the face
                ok = np.concatenate([seen | ~below_neck[:nb], np.ones(len(CV) - nb, bool)]) & ~bad
                um = dilate(VI.unseen_pixels(posed[i], ok, intr, cams[i]), grey_dilate)
            else:
                um = dilate(VI.unseen_pixels(posed[i, :nb], seen, intr, cams[i]), grey_dilate)
            im = cv2.imread(str(ctrl / f"{nm}.png"), cv2.IMREAD_UNCHANGED)
            if local_mask:   # holes only where the figure is: a hole reaching into empty background makes WAN paint
                # translucent "wings" there (the mask asks for content, the control shows none)
                um = um & dilate((im[..., 3] > 127).astype(np.uint8) * 255, 3)
            if grey_fill == "inpaint":   # a smooth continuation of the surrounding colour instead of grey (a big grey
                # patch over a shoulder reads as grey fabric: WAN painted puffy grey sleeves)
                hole = (um > 0) & (im[..., 3] > 0)
                src = (im[..., 3] > 200) & ~hole   # sources: opaque body pixels only (under alpha 0 the RGB is white)
                filled = cv2.inpaint(im[..., :3], (~src).astype(np.uint8) * 255, 9, cv2.INPAINT_TELEA)
                im[..., :3][hole] = filled[hole]
            else:
                im[..., 3][um > 0] = 0
            cv2.imwrite(str(ctrl / f"{nm}.png"), im)
            if i == 0 or local_mask:   # the kept first frame (or every frame with local_mask): WAN regenerates only the
                # greyed pixels (else it carries the defect on); local_mask = pure per-frame inpainting of the holes
                cv2.imwrite(str(out / "mask" / f"{nm}.png"), dilate(um, 8))
    shutil.copy(subject / "face_ref.png" if reference == "face" else run / "ply" / "front.png", out / "reference.png")
    wan = {"background": [float(c) for c in background], "subject_desc": rec.get("prompt"),
           "params": {"width": width, "height": height, "steps_high": 2, "steps_low": 2, "cfg": 1.0, "seed": seed,
                      "low_vram": True, "sampler_high": "euler", "sampler_low": "euler", "sampler_shift": 2.5,
                      "strength": strength or [1.0, 0.7, 0.5, 0.5],
                      "prompt": (prompt or PROMPT).format(motion=motion_text or MOTION_TEXT.get(motion) or (procedural.describe(mot) if mot else "walking")) + (" " + prompt_extra if prompt_extra else ""), "negative_prompt": NEGATIVE}}
    (out / "wan.json").write_text(json.dumps(wan, indent=1, ensure_ascii=False))


B2CRUNNER_PYTHON = Path.home() / "Projects" / "b2crunner" / ".venv" / "bin" / "python"
MHR70 = Path(__file__).resolve().parents[1] / "rig" / "mhr70_mapping.npy"


def draw_frames(body: MHRBody, mfile, params, expr, gtrans, out: Path, n_frames: int) -> None:
    """out/drawing/NNNN.png: body2colmap outline+skeleton drawings of the posed MHR body (tools/draw_control.py)."""
    from ..motion import io as MI
    K = torch.as_tensor(np.load(MHR70), device=body.device)
    V, P70 = [], []
    for i in range(0, n_frames, 16):
        j = min(i + 16, n_frames)
        P = MI.pose(body, mfile, i, j) if mfile is not None else body.pose(
            params[i:j], expr=expr[i:j], global_trans=None if gtrans is None else gtrans[i:j])
        V.append(P.verts.cpu().numpy())
        P70.append(torch.einsum("kn,bnc->bkc", K, torch.cat([P.verts, P.joints], 1)).cpu().numpy())
    np.savez(out / "posed_body.npz", verts=np.concatenate(V).astype(np.float32), faces=body.faces,
             kps=np.concatenate(P70).astype(np.float32))
    import subprocess
    subprocess.run([str(B2CRUNNER_PYTHON), str(TOOLS / "draw_control.py"), str(out / "posed_body.npz"),
                    str(out / "cameras.json"), str(out / "drawing")], check=True, stdout=subprocess.DEVNULL)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--motion", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--elevation", type=float, default=5.0)
    ap.add_argument("--sweep", type=float, default=360.0)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=848)
    ap.add_argument("--loose-only", action="store_true")
    ap.add_argument("--wan", action="store_true", help="also run WAN (in the wan22 venv)")
    a = ap.parse_args()
    build(a.subject, a.run, a.motion, a.out, n_frames=a.frames, elevation=a.elevation, sweep=a.sweep,
          width=a.width, height=a.height, loose_only=a.loose_only)
    if a.wan:
        import subprocess
        subprocess.run([str(WAN_PYTHON), str(TOOLS / "wan_clip.py"), str(a.out)], check=True)


if __name__ == "__main__":
    main()
