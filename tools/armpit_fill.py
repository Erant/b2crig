"""Cover an opening crease with filler splats: coverage the capture never gave (2026-09-27, "coverage, not colour").

    .venv/bin/python tools/armpit_fill.py work/<subject> IN_PLY OUT_PLY [--clips lib_* d_*_G3] [--stretch 1.5]
        [--spacing 0.006] [--radius 0.06] [--shade 1.0]

An A-pose armpit is a closed, shaded crease; when the arm lifts, LBS stretches its body-cage triangles 2-4x and the
crease splats spread into a sparse dark web over skin no capture view ever showed. b2ctrain's render-time stretch fade
(--cage-fade-start/-end) removes the web; this tool adds what should be underneath: on every body-layer triangle that
some listed clip stretches past `--stretch`, and whose neighbourhood is skin, it scatters new splats (posed spacing
`--spacing` at the triangle's largest stretch), coloured with the median colour of the nearby UNstretched skin
(higher SH bands zero), lying at the nearby skin splats' offset from the mesh, sized for the stretched triangle
(b2ctrain clamps growth at 1.15x, so the canonical size is the posed one / 1.15). They carry `cage_fill = 1`:
b2ctrain fades them IN over the same stretch range the crease splats fade out (--cage-fill-start/-end to change it),
so the canonical / capture pose renders exactly as before. No WAN, no training.

Two-state crease (--crease-radius > 0, default): the crease WALLS (inner arm, torso side) only rotate apart when the
arm lifts, they barely stretch, but they were captured in the closed crease's shadow, so their baked shading is the
dark core of the blotch. Every skin splat within `--crease-radius` of a membrane triangle becomes a crease splat
(`cage_fill = -1`) whose fade reads how far its crease has opened: the distance between its own cage vertex and the
vertex nearby that separates most over the clips (the opposite wall; `cage_gate_a/_b`, posed / canonical), and gets
a clone (`cage_fill = +1`, same gate) relit to the luminance of the unstretched skin around (SH bands scaled alike)
that fades in as the crease opens (--cage-fill-start/-end on that distance ratio, e.g. 1.5 -> 3). The membrane
fillers read the same kind of gate.
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2crig import evaluate  # noqa: E402
from b2crig.b2ctrain import CAGE_MAX_GROWTH  # noqa: E402

SKIN = (3, 6, 7, 11, 15, 16, 20, 22)   # face/neck, hands, arms, torso (no legs: trousers)
SH_C0 = 0.28209479177387814


def tri_size(V, F):
    return np.sqrt(np.linalg.norm(np.cross(V[..., F[:, 1], :] - V[..., F[:, 0], :], V[..., F[:, 2], :] - V[..., F[:, 0], :]), axis=-1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("subject", type=Path)
    ap.add_argument("inp", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--clips", nargs="+", default=["lib_*", "d_*_G3"])
    ap.add_argument("--stretch", type=float, default=1.5, help="fill triangles some clip stretches past this ratio")
    ap.add_argument("--unstretched", type=float, default=1.15, help="colour source: skin whose stretch stays under this")
    ap.add_argument("--spacing", type=float, default=0.006, help="filler spacing (m) at the triangle's largest stretch")
    ap.add_argument("--radius", type=float, default=0.06, help="colour source radius (m)")
    ap.add_argument("--shade", type=float, default=1.0, help="multiply the filler colour (1 = the neighbours' colour)")
    ap.add_argument("--opacity", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--crease-radius", type=float, default=0.05, help="body splats within this of a membrane triangle are crease splats (0 = off)")
    ap.add_argument("--gate-radius", type=float, default=0.06, help="a gated splat pairs its cage vertex with the one within this that separates most over the clips")
    ap.add_argument("--min-opening", type=float, default=2.0, help="gate only where that pair distance grows at least this much")
    ap.add_argument("--all-skinlike", action="store_true", help="relight every crease clone, bra-labelled ones included")
    ap.add_argument("--hide-crease", action="store_true", help="debug: crease originals invisible")
    ap.add_argument("--hide-clones", action="store_true", help="debug: clones and fillers invisible")
    ap.add_argument("--debug-cats", action="store_true", help="debug: hide crease + clones, paint ungated splats by why they were skipped")
    ap.add_argument("--debug", action="store_true", help="paint clones and fillers magenta")
    ap.add_argument("--debug2", action="store_true", help="--debug, plus: crease originals invisible, ungated splats within 6 cm of the membrane cyan")
    a = ap.parse_args()
    if a.debug2:
        a.debug = True
    S = a.subject.resolve()

    # Body-layer cage triangles and their largest stretch over the listed clips.
    cage = None
    best = None
    posed_frames = []
    for pat in a.clips:
        for c in sorted(glob.glob(str(S / "clips" / pat))):
            cp = Path(c) / "cage.b2ccage"
            if not cp.exists():
                continue
            cg = evaluate.read_cage(cp)
            if cage is None:
                cage = cg
                nfb = cg.layers[0][3]
                Fb = cg.faces[:nfb]
                k0 = np.maximum(tri_size(cg.verts, Fb), 1e-9)
            r = (tri_size(cg.posed, Fb) / k0).max(0)
            best = r if best is None else np.maximum(best, r)
            posed_frames.append(cg.posed[::5, :cage.layers[0][1]].copy())
    if cage is None:
        sys.exit("no clip cages found")
    V, F = cage.verts, Fb
    nvb = cage.layers[0][1]
    P = np.concatenate(posed_frames)   # [frames, body verts, 3]
    Vb = V[:nvb]
    vtree = cKDTree(Vb)
    bra = cage.layers[1] if len(cage.layers) > 1 else None
    bra_cen = V[cage.faces[bra[2]:bra[2] + bra[3]]].mean(1) if bra else None
    cen = V[F].mean(1)
    e1 = V[F[:, 1]] - V[F[:, 0]]
    nrm = np.cross(e1, V[F[:, 2]] - V[F[:, 0]])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-12)
    e2 = np.cross(nrm, e1)

    ply = PlyData.read(str(a.inp))
    el = ply["vertex"]
    xyz = np.stack([el["x"], el["y"], el["z"]], 1)
    lab = np.asarray(el["seg_label"]).astype(int)
    dc = np.stack([np.asarray(el[f"f_dc_{k}"]) for k in range(3)], 1)
    d, fi = cKDTree(cen).query(xyz)
    on_body = d < 0.03
    skin = np.isin(lab, SKIN) & on_body
    sdist = np.einsum("ij,ij->i", xyz - cen[fi], nrm[fi])   # signed offset of each splat from its nearest face's plane
    src = skin & (best[fi] < a.unstretched)                  # colour source: skin that stays put

    # Candidate faces: stretched, in a skin neighbourhood (not under a garment layer), below the neck.
    cand = np.nonzero(best > a.stretch)[0]
    body_tree = cKDTree(xyz[on_body])
    lab_b, sd_b = lab[on_body], sdist[on_body]
    keep, height, colour, label = [], [], [], []
    src_tree = cKDTree(xyz[src])
    dc_src = dc[src]
    lab_src = lab[src]
    for f in cand:
        nb = body_tree.query_ball_point(cen[f], 0.03)
        if len(nb) < 5:
            continue
        ln = lab_b[nb]
        if np.isin(ln, SKIN).mean() < 0.5:
            continue
        cs = src_tree.query_ball_point(cen[f], a.radius)
        if len(cs) < 10:
            cs = src_tree.query_ball_point(cen[f], 2 * a.radius)
            if len(cs) < 10:
                continue
        keep.append(f)
        height.append(np.median(sd_b[nb][np.isin(ln, SKIN)]))
        colour.append(np.median(dc_src[cs], 0))
        label.append(np.bincount(lab_src[cs], minlength=32).argmax())
    keep = np.array(keep, int)
    if len(keep) == 0:
        sys.exit("no fill faces")
    height = np.clip(np.array(height), -0.015, 0.015)
    colour = np.array(colour)
    label = np.array(label)
    ratio = best[keep]
    print(f"fill faces: {len(keep)} of {len(cand)} stretched > {a.stretch} (body layer); stretch median {np.median(ratio):.2f} "
          f"max {ratio.max():.2f}; splat offset from mesh median {np.median(height) * 100:.2f} cm; "
          f"y range {cen[keep, 1].min():.2f}..{cen[keep, 1].max():.2f}")

    # Scatter: n per face from its posed area at max stretch; canonical position on the triangle + the skin offset.
    rng = np.random.default_rng(a.seed)
    area_posed = 0.5 * (k0[keep] * ratio) ** 2
    n_per = np.maximum(np.ceil(area_posed / a.spacing ** 2).astype(int), 1)
    fidx = np.repeat(keep, n_per)
    j = np.repeat(np.arange(len(keep)), n_per)
    u, v = rng.random(len(fidx)), rng.random(len(fidx))
    su = np.sqrt(u)
    b0, b1, b2 = 1 - su, su * (1 - v), su * v
    pos = V[F[fidx, 0]] * b0[:, None] + V[F[fidx, 1]] * b1[:, None] + V[F[fidx, 2]] * b2[:, None] + nrm[fidx] * height[j][:, None]
    # Sizes: posed in-plane sigma ~ 0.8 spacing at the largest stretch; b2ctrain scales a bound splat by min(ratio, growth).
    s_posed = 0.8 * a.spacing
    s_can = s_posed / np.minimum(ratio[j], CAGE_MAX_GROWTH)
    log_scale = np.stack([np.log(s_can), np.log(s_can), np.log(0.25 * s_can)], 1)
    R = np.stack([e1[fidx], e2[fidx], nrm[fidx]], 2)          # columns: local axes in world
    q = Rotation.from_matrix(R).as_quat()                        # x, y, z, w
    quat = np.stack([q[:, 3], q[:, 0], q[:, 1], q[:, 2]], 1)
    dcf = (colour[j] * SH_C0 + 0.5) * a.shade
    dcf = (dcf - 0.5) / SH_C0
    m = len(fidx)
    print(f"fillers: {m} splats ({m / len(keep):.1f} per face), canonical sigma median {np.median(s_can) * 1000:.1f} mm")

    # Gate per splat: its nearest body vertex a, paired with the vertex b within --gate-radius (canonical) whose distance
    # to a grows most over the clips (the opposite wall of the crease). Returns pairs [n, 2] and each pair's largest ratio.
    mtree = cKDTree(cen[keep])   # membrane triangles: the crease region is what lies within --crease-radius of them
    pair_cache = {}
    def gate_of(pts):
        _, va = vtree.query(pts)
        pairs = np.full((len(pts), 2), -1, int)
        opening = np.ones(len(pts))
        for i, a_ in enumerate(va):
            if a_ not in pair_cache:
                cand = np.array(vtree.query_ball_point(Vb[a_], a.gate_radius))
                d0 = np.linalg.norm(Vb[cand] - Vb[a_], axis=1)
                cand, d0 = cand[d0 > 0.01], d0[d0 > 0.01]
                if len(cand) == 0:
                    pair_cache[a_] = (-1, 1.0)
                else:
                    r = (np.linalg.norm(P[:, cand] - P[:, a_:a_ + 1], axis=2) / d0).max(0)
                    k = int(np.argmax(r))
                    pair_cache[a_] = (int(cand[k]), float(r[k]))
            b_, o = pair_cache[a_]
            pairs[i] = (a_, b_)
            opening[i] = o
        ok = opening >= a.min_opening
        pairs[~ok] = -1
        return pairs, opening
    gate_f, op_f = gate_of(pos)
    print(f"filler gates: opening median {np.median(op_f):.1f}x, {(gate_f[:, 0] >= 0).mean() * 100:.0f}% gated")

    # Crease splats: skin on the body within --crease-radius of a membrane triangle; each fades out on its gate and gets a
    # relit clone (luminance matched to the unstretched skin around it) that fades in.
    fill_old = np.zeros(len(el), np.float32)
    gate_old = np.full((len(el), 2), -1.0, np.float32)
    clones = None
    if a.crease_radius > 0:
        dm, _ = mtree.query(xyz)
        # Crease region, label-free: every splat within 3 cm of the body or a garment-layer surface (hair / face layers
        # excluded by their owning classes) whose surroundings open >= --min-opening over the clips. --crease-radius only
        # bounds the search (the membrane triangles are where the opening is largest).
        EXCL = tuple(k for L in cage.layers[2:] for k in range(32) if L[4] >> k & 1)   # hair, face layers
        layer_cen = [cen] + ([bra_cen] if bra_cen is not None else [])
        d_layer = np.stack([cKDTree(c_).query(xyz)[0] for c_ in layer_cen], 1)
        which = d_layer.argmin(1)                                   # geometric layer: 0 body, 1 garment
        near = (d_layer.min(1) < 0.03) & (dm < 2 * a.crease_radius) & ~np.isin(lab, EXCL)
        idx_near = np.nonzero(near)[0]
        g_all, o_all = gate_of(xyz[idx_near])
        ok = g_all[:, 0] >= 0
        never_opens = idx_near[~ok]
        print(f"near the membrane: {len(idx_near)} splats, {(~ok).sum()} skipped (pair opening < {a.min_opening})")
        idx_near, g_all, o_all = idx_near[ok], g_all[ok], o_all[ok]
        # Relight only what sits on the body surface (the garment keeps its own look); the label is not trusted here: a
        # shadowed crease splat is often voted as the dark garment.
        skinlike = (which[idx_near] == 0) | a.all_skinlike
        print(f"crease splats on the body surface: {skinlike.sum()}, on the garment: {(~skinlike).sum()} "
              f"(labelled garment but on the body: {(skinlike & (lab[idx_near] == 23)).sum()})")
        fill_old[idx_near] = -1.0
        gate_old[idx_near] = g_all
        crease = idx_near
        g_c = g_all
        # Reference colour: body-surface splats far from any opening (their own pair never opens), median within 2-4x radius.
        far_idx = np.nonzero((d_layer[:, 0] < 0.03) & ~np.isin(lab, EXCL + (23,)) & (best[fi] < a.unstretched))[0]
        far_gate, far_open = gate_of(xyz[far_idx])
        far_idx = far_idx[far_open < 1.3]
        ftree = cKDTree(xyz[far_idx])
        rgb_all = dc * SH_C0 + 0.5
        ref = rgb_all[crease].copy()
        for i, s_ in enumerate(crease):
            if not skinlike[i]:
                continue
            nb = ftree.query_ball_point(xyz[s_], 2 * a.radius)
            if len(nb) < 10:
                nb = ftree.query_ball_point(xyz[s_], 4 * a.radius)
            if len(nb) >= 10:
                ref[i] = np.median(rgb_all[far_idx][nb], 0)
        # Blend by how much the splat's surroundings open: fully relit deep in the crease, the original at its rim.
        w = np.clip((o_all - a.min_opening) / a.min_opening, 0, 1) * skinlike
        clones = (crease, ref, w, g_c)
        print(f"crease: {len(idx_near)} splats fade out (pair opening median {np.median(o_all):.1f}x); "
              f"{skinlike.sum()} clones relit ({(w >= 1).sum()} fully)")

    names = [p.name for p in el.properties]
    nc = 0 if clones is None else len(clones[0])
    out = np.zeros(len(el) + m + nc, dtype=[(nm, "f4") for nm in names] + [("cage_fill", "f4"), ("cage_gate_a", "f4"), ("cage_gate_b", "f4")])
    for nm in names:
        out[nm][:len(el)] = el[nm]
    out["cage_fill"][:len(el)] = fill_old
    if a.debug2 or a.hide_crease:
        out["opacity"][:len(el)][fill_old < 0] = -10.0
    if a.debug_cats:
        out["opacity"][:len(el)][fill_old < 0] = -10.0
        out["opacity"][len(el):] = -10.0
        dmm, _ = mtree.query(xyz)
        ung = fill_old == 0
        cats = {"off-body (>3cm from a body face)": (ung & ~on_body & (dmm < 0.10), (1, 0, 0)),
                "5-8 cm from the membrane": (ung & on_body & (dmm >= a.crease_radius) & (dmm < 0.08), (0, 1, 0)),
                "8-12 cm": (ung & on_body & (dmm >= 0.08) & (dmm < 0.12), (0, 0, 1)),
                "excluded label": (ung & on_body & (dmm < a.crease_radius) & np.isin(lab, (4, 13, 1, 24, 25)), (1, 1, 0)),
                "pair never opens": (np.isin(np.arange(len(el)), never_opens), (0, 1, 1))}
        for nm_, (msk, col) in cats.items():
            print(f"debug-cats {nm_}: {msk.sum()}")
            for k in range(3):
                out[f"f_dc_{k}"][:len(el)][msk] = (col[k] - 0.5) / SH_C0
            for nm in names:
                if nm.startswith("f_rest_"):
                    out[nm][:len(el)][msk] = 0
    if a.hide_clones:
        out["opacity"][len(el):] = -10.0
    if a.debug2:
        dmm, _ = mtree.query(xyz)
        cyan = (fill_old == 0) & (dmm < 0.06)
        for k, c in enumerate((0.0, 1.0, 1.0)):
            out[f"f_dc_{k}"][:len(el)][cyan] = (c - 0.5) / SH_C0
        for nm in names:
            if nm.startswith("f_rest_"):
                out[nm][:len(el)][cyan] = 0
        print(f"debug2: {cyan.sum()} ungated splats within 6 cm of the membrane painted cyan")
    out["cage_gate_a"][:len(el)] = gate_old[:, 0]
    out["cage_gate_b"][:len(el)] = gate_old[:, 1]
    if clones is not None:
        ci, ref, w, g_c = clones
        cl = out[len(el) + m:]
        for nm in names:
            cl[nm] = el[nm][ci]
        for k in range(3):
            rgb = np.asarray(el[f"f_dc_{k}"])[ci] * SH_C0 + 0.5
            rgb = w * ref[:, k] + (1 - w) * rgb
            cl[f"f_dc_{k}"] = (rgb - 0.5) / SH_C0
        for nm in names:
            if nm.startswith("f_rest_"):
                cl[nm] = el[nm][ci] * (1 - w)
        if a.debug:
            cl["f_dc_0"] = (1.0 - 0.5) / SH_C0; cl["f_dc_1"] = (0.0 - 0.5) / SH_C0; cl["f_dc_2"] = (1.0 - 0.5) / SH_C0
            for nm in names:
                if nm.startswith("f_rest_"):
                    cl[nm] = 0
        cl["cage_fill"] = 1.0
        cl["cage_gate_a"] = g_c[:, 0]
        cl["cage_gate_b"] = g_c[:, 1]
    new = out[len(el):len(el) + m]
    new["cage_gate_a"] = gate_f[:, 0]
    new["cage_gate_b"] = gate_f[:, 1]
    new["x"], new["y"], new["z"] = pos.T
    if a.debug:
        dcf = (np.array([[1.0, 0.0, 1.0]]) - 0.5) / SH_C0 * np.ones((m, 1))
    for k in range(3):
        new[f"scale_{k}"] = log_scale[:, k]
        new[f"f_dc_{k}"] = dcf[:, k]
    for k in range(4):
        new[f"rot_{k}"] = quat[:, k]
    new["opacity"] = np.log(a.opacity / (1 - a.opacity))
    new["seg_label"] = label[j]
    new["seg_conf"] = 1.0
    if "ev_views" in names:   # evidence: copy a plausible value so an evidence prune does not drop them by default
        new["ev_w_in"] = 1.0; new["ev_w_all"] = 1.0; new["ev_views"] = 1.0
    new["cage_fill"] = 1.0
    PlyData([PlyElement.describe(out, "vertex")], text=False, byte_order="<").write(str(a.out))
    print(f"wrote {a.out} ({len(out)} splats)")


if __name__ == "__main__":
    main()
