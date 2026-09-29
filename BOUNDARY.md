# b2crig ↔ b2ctrain: what lives where

b2ctrain (`~/Projects/b2ctrain`) is the CUDA splat trainer and renderer. b2crig animates a trained subject by driving
it. The counterpart note is `b2ctrain/docs/b2crig-boundary.md`; keep the two in step.

## The rule

**b2crig owns every decision about the rig**, including:
- the body model (MHR) and cage layer 0 = the body;
- what each Sapiens2 class means (layers, groups, exclusions);
- cage building and skinning, motions and retargeting, pose libraries and pose selection;
- the cameras for virtual or evaluation views;
- which splats are interior or ambiguous;
- the geometric measures that gate pose-dependent behaviour.

**b2ctrain owns the differentiable loop and the rasteriser.** It holds posing splats by the cage, losses on posed
renders, per-splat learned state, and anything else that needs gradients or runs per pixel.

b2crig hands its decisions to b2ctrain as **files and flags**, and never reimplements rendering or training. b2ctrain
never makes a rig decision on its own. Test for new code: *would changing the rig (a new body model, other garment
classes, another pose library) require editing it?* If yes, it belongs here. If it is maths on splats, pixels or
gradients, it belongs in b2ctrain.

Every call into b2ctrain goes through the binary (`B2CTRAIN`, default `~/Projects/b2ctrain/build/b2ctrain`). The
file writers and shared constants live in `b2crig/b2ctrain.py`.

## What b2crig computes for b2ctrain

| Input | Where in b2crig | b2ctrain side |
|---|---|---|
| Cage file `B2CCAGE1` (layers, canonical and posed vertices) | `b2ctrain.write_cage`, `rig/layered.py`, `rig/cage.py` | `--cage`; format in `src/gpu/cage.h` |
| Opening gate θ (`B2COPEN1` section of the cage) | `rig/open_gate.py` (`python -m b2crig.rig.open_gate CAGE`); `b2ctrain.render` adds it when the ply has `open_*` | `--cage-open`, render |
| Dual binding `B2CALT01` | `rig/binding.py`, `tools/fit_binding.py` | `--alt-binding`, `fit-cage --fit-binding` |
| Containment cameras + interior mask | `rig/contain.py` (`write_contain`), used by `tools/cage_train.py --contain` | `--pose-contain-cameras`, `--pose-contain-exclude` |
| fit-cage label groups | `b2ctrain.FIT_GROUPS` / `fit_groups_args()`; every fit-cage call passes them | `fit-cage --groups` (b2ctrain has no default) |
| Pose sets for the containment and stretch losses | `tools/pose_select.py`, `motion/` | the cage's extra frames |
| Filler/crease splats `cage_fill`, `cage_gate_a/_b` | `tools/armpit_fill.py` | `pose_bound` fade/fill |
| Size clamp | `b2ctrain.CAGE_MAX_GROWTH` → `--cage-max-growth` | train, render |

b2ctrain returns: the trained ply (`seg_label`, `open_*`, `b2c.*` header), the `.app` appearance MLP, fit-cage's
`delta.f32` / `vis.f32` / `fit.json`, and `render --export-posed`.

## What b2crig delivers

One glTF file per subject in the b2c format (`~/Projects/b2cgltf/SPEC.md`; the package `b2cgltf` owns the layout
and rules). b2crunner writes the subject file; `tools/export_gltf.py rig` enhances it in place with b2crig's cage,
b2ctrain's binding (`render --export-binding`) and a preview skin (SPEC 5), and `tools/export_gltf.py clip` writes one
`<name>.clip.glb` per motion: skeletal animation plus the residual from plain skinning to b2crig's posed cage (SPEC 6).
b2crig's side of the code is `b2crig/export/gltf.py`. b2cviewer reads these files.

## Moved here from b2ctrain on 2026-09-29

- **Pose containment's poses, cameras and interior mask:** `rig/contain.py`. It is a port of what the trainer did:
  - cage frames spread evenly over N views;
  - each view uses a capture camera's offset from the canonical layer-0 centroid, re-applied to the posed centroid;
  - the interior mask marks splats more than `depth` behind the nearest layer-0 vertex along its normal.

  Parity on b24be4: identical excluded count (3488), equal silhouette leak.
- **The cage-open gate:** `rig/open_gate.py`. It is the CUDA partner selection (in index order, every stride-th, at
  most 64 per vertex) and the relative rotation of the incident first-edge triangle frame. `--cage-open-debug`
  renders match the previous in-trainer gate to 1/255.
  - `gen/clips.py rotated_vertices` (grey_unseen) is the same measure over every partner as flat pairs. It is kept
    separate because padding ~1.8k partners per vertex does not fit on the GPU.
- **fit-cage's label groups** (`4;23,1;13`) now come from b2crig.

## Grey areas and known issues

- **Two copies of the posing maths.** `b2cviewer/web/rig.js` is a JS port of b2ctrain's `cage.cu` posing and of
  `cage_app`. b2crig should hand the viewer data, not new posing rules. The viewer lacks stretch fade/fill, `open_*`
  and dual binding.
- **`tools/open_gates.py`** (pair gates `cage_gate_*`) is superseded for `--cage-open`. b2ctrain uses
  `cage_gate_a/_b` only for filler/crease splats and ignores `cage_gate_s`.
- **`tools/aniso_proto.py`** prototypes a change to b2ctrain's posing maths by pre-deforming a ply. If it becomes a
  feature, the maths goes into b2ctrain's `cage.cu` (and `rig.js`), not here.
- **Garment-edge "hairs"** (long splats posed rigidly by one triangle; b2ctrain `out/needles`): attaching both ends of
  a long splat to their own triangles would be computed here (a binding file like `B2CALT01`) and posed in b2ctrain.
  A bend-aware split/loss would be a b2ctrain training feature fed by pose frames chosen here.
