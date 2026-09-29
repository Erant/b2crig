"""Retrain the subject's splat on its capture views plus generated clips posed by their cages (b2ctrain --cage),
then score held-out clips old vs new.

    .venv/bin/python tools/cage_train.py work/<subject> --train CLIP ... --test CLIP ... [--iters N] [--tag T]
                                          [--every K] [--repeat R] [-- extra b2ctrain args]

Train clips' cages are concatenated into one cage (frames "<clip>__<frame>", the clips must share the cage layout,
i.e. be built from the same layers) and every K-th of their frames joins the capture views (listed R times). Scores per
test clip, rendered through its own cage with the old and the new splat: L1 vs its WAN frames over the figure and over
the face (Sapiens classes face/lips/teeth/tongue), and the label IoUs.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2crig import b2ctrain as B  # noqa: E402
from b2crig import evaluate  # noqa: E402

FACE = (3, 24, 25, 26, 27, 28)
argv = sys.argv[1:]
extra = argv[argv.index("--") + 1:] if "--" in argv else []
ap = argparse.ArgumentParser()
ap.add_argument("subject", type=Path)
ap.add_argument("--train", nargs="*", default=[])
ap.add_argument("--test", nargs="+", default=[])
ap.add_argument("--iters", type=int, default=6000)
ap.add_argument("--library", nargs="*", default=[], help="clips whose cage frames ride along without views: a pose library "
                "for b2ctrain's stretch regulariser (--cage-stretch-weight)")
ap.add_argument("--every", type=int, default=1)
ap.add_argument("--repeat", type=int, default=1)
ap.add_argument("--tag", default="cage_train")
ap.add_argument("--init", type=Path, help="start from this splat instead of the run's scene.ply (a warm start)")
ap.add_argument("--capture-repeat", type=int, default=1, help="list every capture view this often (keeps its weight up as clips are added)")
ap.add_argument("--frames", default="wan", help="train clips' frame folder, e.g. wan_1080 (b2crunner's tools/upscale_clip); clips "
                "without it use their own frames")
ap.add_argument("--anchor-size", type=int, default=512, help="face anchor crop size, px")
ap.add_argument("--anchor-drop", type=float, default=0.0, help="crop centre this far below head+5cm, metres (e.g. 0.12: "
                "neck, collar and pendant too)")
ap.add_argument("--hollow", type=float, default=0.0, help="b2ctrain's hollow loss at this weight against the subject's "
                "MHR body (posed per view by the cage): empties the body interior")
ap.add_argument("--hollow-margin", type=float, default=0.04)
ap.add_argument("--anchor-clip", default="", help="pose the face anchors by this train clip's frames (named like the "
                "capture images: a re-canonicalized subject's 'capture' clip)")
ap.add_argument("--face-anchors", type=int, default=0, help="add 512x512 face crops of the capture views (camera 90+), this "
                "many times each: they hold the capture's face against close-up clips that redraw it")
ap.add_argument("--contain", nargs="*", default=[], help="pose clips CLIP[:every] for b2ctrain's pose containment loss "
                "(--pose-contain-weight): their frames join the cage without views; b2ctrain renders the posed splat from "
                "capture directions and pushes its alpha outside the silhouette of its own surface (deep interior left out) to 0")
ap.add_argument("--contain-every", type=int, default=6)
ap.add_argument("--contain-weight", type=float, default=1.0)
ap.add_argument("--contain-views", type=int, default=300)
ap.add_argument("--contain-res", type=int, default=960, help="long side of the containment views, px")
ap.add_argument("--contain-depth", type=float, default=0.02, help="splats more than this (m) inside the body do not "
                "define the containment silhouette")
ap.add_argument("--contain-fraction", type=float, default=0.65)
ap.add_argument("--background", default="0.5,0.5,0.5:0.5", help="training background COLOR:NOISE (b2ctrain "
                "--background-color / --background-noise-strength). Default = b2crunner's random_background, which its final "
                "training uses: against b2ctrain's near-black default (0,0,0:0.1) a soft silhouette edge is fitted by opaque "
                "DARK splats (dark streaks on the tops of raised arms), so a fine-tune on black puts them back")
ap.add_argument("--grow", action="store_true", help="let b2ctrain grow and screen-size-split splats (off by default: "
                "the warm start is b2crunner's culled splat, and growth refills what the culling removed; hand close-ups "
                "alone split ~38k body splats)")
ap.add_argument("--contain-hands", type=float, default=0.0, help="share of the containment views that are hand close-ups "
                "(b2crig/rig/contain.py), for finger poses (pose_library --hands)")
a = ap.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
assert a.train or a.contain, "nothing to train on: --train and/or --contain"
S = a.subject.resolve(); run = Path(json.loads((S.parent / "subjects.json").read_text())[S.name]); cap = run / "colmap"
out = S / "train" / a.tag; ds = out / "ds"; shutil.rmtree(ds, ignore_errors=True)
for sub in ("images", "labels"):
    (ds / sub).mkdir(parents=True)

cam_lines = ["1 " + " ".join([l for l in (cap / "cameras.txt").read_text().splitlines() if l and not l.startswith("#")][0].split()[1:])]
lines = []
for ln in (cap / "images.txt").read_text().splitlines():
    f = ln.split()
    if ln.startswith("#") or len(f) < 10:
        continue
    for r in range(a.capture_repeat):
        nm = f[9] if r == 0 else f"{Path(f[9]).stem}__c{r}.png"
        lines += [" ".join([str(len(lines) // 2 + 1), *f[1:8], "1", nm]), ""]
        for sub in ("images", "labels"):
            if (cap / sub / f[9]).exists():
                (ds / sub / nm).symlink_to((cap / sub / f[9]).resolve())
if a.face_anchors:
    from b2crig.visibility import colmap_cams
    from b2crig.rig.skeleton import JOINT
    (W1, H1, fx1, fy1, cx1, cy1), ccams = colmap_cams(cap)
    if a.anchor_clip and (S / "old_canonical_params.npy").exists():   # the capture shows the OLD canonical pose
        import torch
        from b2crig.rig.mhr import MHRBody
        _b = MHRBody(S / "mhr.npz")
        hj = _b.pose(torch.as_tensor(np.load(S / "old_canonical_params.npy").reshape(1, -1), device=_b.device)).joints[0, JOINT["head"]].cpu().numpy()
    else:
        hj = np.load(S / "mhr.npz")["joints_world"][JOINT["head"]]
    hj = hj + np.array([0.0, 0.05 - a.anchor_drop, 0.0])
    AS = a.anchor_size; AH = AS // 2
    capnames = [ln.split()[9] for ln in (cap / "images.txt").read_text().splitlines() if not ln.startswith("#") and len(ln.split()) >= 10]
    fd = out / "face_anchor"; (fd / "images").mkdir(parents=True, exist_ok=True); (fd / "labels").mkdir(exist_ok=True)
    fimgs = [l for l in (cap / "images.txt").read_text().splitlines() if not l.startswith("#") and len(l.split()) >= 10]
    n_anchor = 0
    for ln in fimgs:
        f = ln.split(); Rw2c = __import__("b2crig.visibility", fromlist=["x"]).qvec_to_R(list(map(float, f[1:5])))
        t = np.array(list(map(float, f[5:8]))); q = Rw2c @ hj + t
        if q[2] <= 0:
            continue
        u, v = fx1 * q[0] / q[2] + cx1, fy1 * q[1] / q[2] + cy1
        x0, y0 = int(round(u)) - AH, int(round(v)) - AH
        if x0 < 0 or y0 < 0 or x0 + AS > W1 or y0 + AS > H1:
            continue
        lab = cv2.imread(str(cap / "labels" / f[9]), 0)
        if lab is None or np.isin(lab[y0:y0 + AS, x0:x0 + AS], (3, 24, 25)).sum() < 5000:   # the face has to be visible
            continue
        im = cv2.imread(str(cap / "images" / f[9]), cv2.IMREAD_UNCHANGED)
        nm0 = f"face__{Path(f[9]).stem}.png"
        cv2.imwrite(str(fd / "images" / nm0), im[y0:y0 + AS, x0:x0 + AS]); cv2.imwrite(str(fd / "labels" / nm0), lab[y0:y0 + AS, x0:x0 + AS])
        cid = 100 + n_anchor; n_anchor += 1
        cam_lines.append(f"{cid} PINHOLE {AS} {AS} {fx1} {fy1} {cx1 - x0} {cy1 - y0}")
        for r in range(a.face_anchors):
            nm = (f"{a.anchor_clip}__{Path(f[9]).stem}@a{r}.png" if a.anchor_clip else f"face__{Path(f[9]).stem}__r{r}.png")
            lines += [" ".join([str(len(lines) // 2 + 1), *f[1:8], str(cid), nm]), ""]
            for sub in ("images", "labels"):
                (ds / sub / nm).symlink_to((fd / sub / nm0).resolve())
    print(f"cage_train: {n_anchor} capture face crops as anchors (x{a.face_anchors})")
head0, names, posed = None, [], []
for ci, c in enumerate(a.train):
    parts = c.split(":")   # name[:repeat[:every]]
    c, rep, every = parts[0], int(parts[1]) if len(parts) > 1 else a.repeat, int(parts[2]) if len(parts) > 2 else a.every
    clip = S / "clips" / c
    cgp = clip / "cage_v2.b2ccage" if (clip / "cage_v2.b2ccage").exists() else clip / "cage.b2ccage"   # the current rig
    cg = evaluate.read_cage(cgp); raw = cgp.read_bytes()
    head = raw[:24 + 20 * len(cg.layers) + cg.verts.nbytes + cg.faces.nbytes]
    head0 = head0 or head
    assert head[:20] + head[24:] == head0[:20] + head0[24:], f"{c}: its cage layout differs from {a.train[0]}'s"   # not the frame count
    names += [f"{c}__{n}" for n in cg.names]; posed.append(cg.posed)
    cam = next(l for l in (clip / "fit_ds" / "cameras.txt").read_text().splitlines() if l.strip() and not l.startswith("#")).split()
    src = clip / "fit_ds"
    if a.frames != "wan" and (clip / a.frames).exists():   # upscaled frames: RGBA + labels from the seg, intrinsics scaled
        f0 = cv2.imread(str(next((clip / a.frames).glob("*.png"))))
        H2, W2 = f0.shape[:2]; sx, sy = W2 / int(cam[2]), H2 / int(cam[3])
        cam = cam[:2] + [str(W2), str(H2), str(float(cam[4]) * sx), str(float(cam[5]) * sy), str(float(cam[6]) * sx), str(float(cam[7]) * sy)]
        src = out / f"frames_{c}"
        for sub in ("images", "labels"):
            (src / sub).mkdir(parents=True, exist_ok=True)
        for fr in sorted((clip / a.frames).glob("*.png")):
            lab = cv2.resize(cv2.imread(str(clip / "seg" / fr.name), 0), (W2, H2), interpolation=cv2.INTER_NEAREST)
            al = cv2.GaussianBlur((lab > 0).astype(np.float32), (0, 0), 1.5 * sx)
            cv2.imwrite(str(src / "images" / fr.name), np.dstack([cv2.imread(str(fr)), np.clip(al * 255, 0, 255).astype(np.uint8)]))
            cv2.imwrite(str(src / "labels" / fr.name), lab)
    cid = ci + 2; cam_lines.append(" ".join([str(cid)] + cam[1:]))
    k_img = -1
    for ln in (clip / "fit_ds" / "images.txt").read_text().splitlines():
        f = ln.split()
        if ln.startswith("#") or len(f) < 10:
            continue
        k_img += 1
        if k_img % every:
            continue
        for r in range(rep):
            nm = f"{c}__{Path(f[9]).stem}@r{r}.png"
            lines += [" ".join([str(len(lines) // 2 + 1), *f[1:8], str(cid), nm]), ""]
            for sub in ("images", "labels"):
                (ds / sub / nm).symlink_to((src / sub / f[9]).resolve())
for c in a.library:   # frames no view is named after
    clip = S / "clips" / c
    cgp = clip / "cage_v2.b2ccage" if (clip / "cage_v2.b2ccage").exists() else clip / "cage.b2ccage"
    cg = evaluate.read_cage(cgp); raw = cgp.read_bytes()
    head = raw[:24 + 20 * len(cg.layers) + cg.verts.nbytes + cg.faces.nbytes]
    assert head[:20] + head[24:] == head0[:20] + head0[24:], f"library {c}: cage layout differs"
    names += [f"lib_{c}__{n}" for n in cg.names]; posed.append(cg.posed)
for spec in a.contain:   # pose frames for the containment loss (no views are named after them)
    c, every = spec.split(":")[0], int(spec.split(":")[1]) if ":" in spec else a.contain_every
    clip = S / "clips" / c
    cgp = clip / "cage_v2.b2ccage" if (clip / "cage_v2.b2ccage").exists() else clip / "cage.b2ccage"
    cg = evaluate.read_cage(cgp); raw = cgp.read_bytes()
    head = raw[:24 + 20 * len(cg.layers) + cg.verts.nbytes + cg.faces.nbytes]
    head0 = head0 or head
    assert head[:20] + head[24:] == head0[:20] + head0[24:], f"contain {c}: cage layout differs"
    names += [f"pc_{c}__{n}" for n in cg.names[::every]]; posed.append(cg.posed[::every])
(ds / "cameras.txt").write_text("\n".join(cam_lines) + "\n")
(ds / "images.txt").write_text("\n".join(lines) + "\n"); (ds / "points3D.txt").write_text("")
(ds / "init.ply").symlink_to((a.init or run / "ply" / "scene.ply").resolve())
counts = np.frombuffer(head0[8:24], np.int32).copy(); counts[3] = len(names)
with open(out / "cage.b2ccage", "wb") as fh:
    fh.write(head0[:8]); fh.write(counts.tobytes()); fh.write(head0[24:])
    for nm in names:
        b = nm.encode()[:63]; fh.write(b + b"\0" * (64 - len(b)))
    fh.write(np.concatenate(posed).astype(np.float32).tobytes())
print(f"cage_train: {len(lines) // 2} views, {len(names)} cage frames from {len(a.train)} clips (+ {len(a.contain)} containment pose clips)")
if a.contain:   # the containment views and the interior mask are ours to choose (b2crig/rig/contain.py); b2ctrain renders them
    from b2crig.rig import contain
    extra = ["--pose-contain-weight", str(a.contain_weight), "--pose-contain-fraction", str(a.contain_fraction),
             *contain.write_contain(out, evaluate.read_cage(out / "cage.b2ccage"), (a.init or run / "ply" / "scene.ply").resolve(), cap,
                                    n=a.contain_views, res=a.contain_res, depth=a.contain_depth,
                                    hands=contain.hand_vertices(S / "mhr.npz") if a.contain_hands > 0 else None,
                                    hand_views=a.contain_hands)] + extra
if "--cage-open" in extra:   # two-state splats: the opening gate goes into the cage (b2crig/rig/open_gate.py)
    from b2crig.rig import open_gate
    open_gate.add_open_gate(out / "cage.b2ccage")
if a.hollow > 0:   # the canonical body as the hollow proxy; the trainer poses it with each cage frame
    md = np.load(S / "mhr.npz"); V = md["verts_world"].astype("<f4"); F = md["faces"].astype("<i4")
    hdr = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(V)}\nproperty float x\nproperty float y\nproperty float z\n"
           f"element face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n").encode()
    fr = np.zeros(len(F), dtype=[("n", "u1"), ("v", "<i4", (3,))]); fr["n"] = 3; fr["v"] = F
    (ds / "mesh.ply").write_bytes(hdr + V.tobytes() + fr.tobytes())
    extra = ["--hollow-weight", str(a.hollow), "--hollow-margin", str(a.hollow_margin), "--hollow-proxy", "mesh",
             "--mesh", str(ds / "mesh.ply")] + extra
subprocess.run([str(B.B2CTRAIN), str(ds), "--cage", str(out / "cage.b2ccage"), "--cage-max-growth", f"{B.CAGE_MAX_GROWTH:g}", "--total-train-iters", str(a.iters),
                "--export-path", str(out), "--export-name", "scene.ply", "--export-every", str(a.iters),
                "--sh-degree", "3", "--max-resolution", "1920", "--background-color", a.background.split(":")[0],
                "--background-noise-strength", a.background.split(":")[1],
                *([] if a.grow else ["--growth-stop-iter", "0", "--split-at-screen-size", "0"]), *extra], check=True, stdout=open(out / "train.log", "w"), stderr=subprocess.STDOUT)
new = out / "scene.ply"; old = run / "ply" / "scene.ply"


def rgb(p):
    x = cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(np.float32)
    return x[..., :3] * x[..., 3:] / 255 + 127 * (1 - x[..., 3:] / 255)


G = ["upper", "lower", "hair", "skin", "silhouette"]
print(f"{'test clip':14s} {'L1 fig old/new':>15s} {'L1 face old/new':>16s}  IoU old->new ({', '.join(G)})")
for c in a.test:
    clip = S / "clips" / c
    rd = {}
    for k, ply in (("old", old), ("new", new)):
        rd[k] = out / f"test_{c}_{k}"
        B.render(ply, clip / "cameras.json", rd[k], cage=clip / "cage_v2.b2ccage" if (clip / "cage_v2.b2ccage").exists() else clip / "cage.b2ccage",
                 label_maps=True, background=(0.5, 0.5, 0.5))
    L = {k: {"fig": [], "face": []} for k in rd}; I = {k: [] for k in rd}; SH = {"old": [], "new": [], "wan": []}
    for nm in evaluate.read_cage(clip / "cage.b2ccage").names:   # frame names are the same in cage_v2
        wan = cv2.imread(str(clip / "wan" / f"{nm}.png")).astype(np.float32); seg = cv2.imread(str(clip / "seg" / f"{nm}.png"), 0)
        face = np.isin(seg, FACE)
        if face.sum() > 100:   # sharpness: mean |Laplacian| of the grey image over the face
            for k, im in (("old", rgb(rd["old"] / f"{nm}.png")), ("new", rgb(rd["new"] / f"{nm}.png")), ("wan", wan)):
                SH[k].append(np.abs(cv2.Laplacian(cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2GRAY), cv2.CV_32F))[face].mean())
        for k in rd:
            e = np.abs(rgb(rd[k] / f"{nm}.png") - wan).mean(-1)
            L[k]["fig"].append(e[seg > 0].mean()); L[k]["face"].append(e[face].mean() if face.sum() > 100 else np.nan)
            I[k].append(evaluate.ious(cv2.imread(str(rd[k] / f"{nm}.labels.png"), 0), seg))
    m = {k: [np.nanmean([x[g] for x in I[k]]) for g in G] for k in I}
    print(f"{c:14s} {np.mean(L['old']['fig']):6.2f}/{np.mean(L['new']['fig']):6.2f}   {np.nanmean(L['old']['face']):6.2f}/{np.nanmean(L['new']['face']):6.2f}   "
          + "  ".join(f"{o:.3f}->{n:.3f}" for o, n in zip(m["old"], m["new"])), flush=True)
    if SH["wan"]:
        print(f"{'':14s} face sharpness (mean |Laplacian|): old {np.mean(SH['old']):.2f}  new {np.mean(SH['new']):.2f}  WAN {np.mean(SH['wan']):.2f}")
