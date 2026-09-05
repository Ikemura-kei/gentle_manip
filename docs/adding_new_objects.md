# Adding a new object (mesh → configs → smoke → collect)

Everything object-agnostic is SHARED and must stay identical across objects; only the blocks marked
OBJECT-SPECIFIC change. Do not touch the synthesis/execution code (`docs/grasp_synthesis_final.md`).

## 1. Get a mesh
Any route that yields a closed surface of the object at rest:
- **Photos → TripoSG** (`docs/mesh_from_photos.md`; env `envs/triposg_arrhenius`, GPU): several
  views × seeds, keep candidates that pass the topology gates (genus 0, watertight, one component);
  outputs `obj_meshes/<name>/clean.obj` + `report.json`. This produced the mushrooms/tomatoes/berries.
- **Web / CAD** (`.obj`/`.stl`): fine; extruded shapes (letters) came this way.
Requirements: **metres**, a single component, roughly watertight (small holes are repaired in memory),
any face count (scans are decimated to 2000 faces at load; CAD meshes are remeshed to 4 mm).

## 2. Post-process (once, offline) — size, thickness, centring
Real objects are what the sim must match; check `trimesh.load(mesh).extents` against a ruler.
- **Scale to the real extents** (uniform). Keep the mesh in metres.
- **Thickness matters more than length**: an object < ~25 mm tall on the board is hard to grasp under
  the 15 mm TCP floor (banana_chunk 20 → 26.5 mm, letters 6 → 25 mm were thickened by scaling the thin
  axis about the centroid). Aim ≥ 25 mm resting height unless the real object truly is flat.
- **Centre the mesh** on its centroid (the registry `default_pos`/spawn is the centroid).
- **FEM-aware check** — the planner's FEM must equal the mesh (no dilation). Run
  `smgrasp.finger_grasp_final.build_grasp_fem(path)` and compare extents; `meta["direct_tet"]` must
  be `True` and the extent ratio ≈ 1.00 (see the 2026-09-05 table in the DEVLOG: 30/31 objects direct).
  If it prints `[fem] ... voxel remesh fallback (dilates)` the surface self-intersects: fix the mesh.
- Put it in `gentle_manip/assets/objects/<name>.obj`. Registry meshes are the only files you add.

## 3. Registry (`gentle_manip/assets/registry.py`) — ADDITIVE
```python
"<name>": ObjectDef("<name>", MATERIALS["<material>"], object_type="soft",
                    size=(sx, sy, sz),                 # extents (m), for bookkeeping/spawn
                    default_pos=(0.47, 0.0, sz/2 + 0.001),   # resting on the TABLE: half height + 1 mm
                    mesh_path=str(_OBJ_DIR / "<name>.obj")),
```
Never edit an existing entry (parallel branches merge on this file); add a new name. Material: pick an
existing one (`tofu` 50 kPa, `mushroom` 300 kPa, `soft_shape` …) or add one in `materials.py`
(E, ν, ρ, von-Mises yield). Yield is the gentleness reference and is NOT randomised.

## 4. Configs — copy the tofu trio, change only the OBJECT-SPECIFIC block
`tasks/single_lift_<name>_soft.yaml`, `dr/soft_orientation_realws_<name>.yaml`,
`experiments/single_lift_<name>_soft_abs_action_armfocus_7d_realws.yaml` (from the tofu files; the
experiment yaml just renames `task:`/`dr:`). Keep the 3-line header (`# [type]`, `# Used by`, `# Status`).

**SHARED — do not change** (task): success band `success_z_min/max`, `hold_steps`, `success_scale`,
board (13.8 mm, size, centre, colours), camera (`cam_fov/pos/lookat/up` = the calibrated D435i),
rewards except `lift.grasp_gate_dist`. (DR): `object_pos_x/y`, `object_nominal_xy`, `robot_init_pos_xyz`,
`object_yaw_deg`, `object_pitch_roll_deg`, `object_flip_*`, `coup_friction`, `start_modes`,
`disturbance_prob`. `object_nominal_xy` MUST equal the registry `default_pos` xy (0.47, 0) — a mismatch
pushes objects out of the MPM box and slices them.

**OBJECT-SPECIFIC — choose** (task):
- `object_name`, `object_type: soft`.
- `sim_substeps` / `mpm_grid_density`: 235 / 250 (tofu, ~4 mm cells) for objects ≥ 25 mm; the letters
  used 470/500 only while 6 mm thin — cost ∝ density⁴, so keep 250 unless a feature is < 3 cells.
- `object_spawn_z` (**the burying knob**): spawn height of the centroid, object must start RESTING or
  above, never inside the board (particles inside a rigid fixture get kicked at spawn):
  `spawn_z = max(board + h/2·s_max + 20 mm, board + (diag/2)·s_max + 5 mm)` with board 0.0138,
  h = resting thickness, diag = bbox diagonal, s_max = max `object_scale` (tilted/flipped spawns must
  clear the board). Drop from a bit high is fine (it settles); a buried spawn is not.
- `lift.grasp_gate_dist` (size dependent, ~ half diagonal + finger reach; tofu 0.079).
(DR): `object_scale` (e.g. [0.8, 1.4]; smaller range for objects whose real size is known),
`object_axis_scale`, `object_bend_deg`/`twist`/`taper` (0 for rigid-ish shapes, small for produce),
`object_E`/`object_nu`/`object_rho` ranges around the registry material.

`--table-z 0.0138` on the command line = the board top (the planner's table plane); leave it.

## 5. Smoke with visualisation, then a rough success rate
```bash
# 1 env, viewer + step-through of every synthesis stage (q advances), object particles vs FEM overlay
bash gentle_manip/scripts/dev_synth.sh            # set obj=<name> inside; MUJOCO_GL=glfw
# 20 episodes x 10 envs, videos, stats: success / ever-lifted / sub-yield / max stress-yield / speed
bash gentle_manip/scripts/final/profile_demo_collection.sh   # add <name> to objects=(...)
```
Read in the log: `[proj]` (planned width vs FEM span vs MPM particle span — the sim body is ~1 mm/side
inside the mesh, normal), `SYNTH FAILED` count (planner found no grasp: object too flat/small under the
15 mm floor, or the mesh is bad), `[grasp]` force at close (0 = fingers not touching), `Success rate`,
`Ever lifted`, `sub-yield`. Expect ≥ 90 % for compact objects ≥ 25 mm tall.

## 6. If success is low — only two levers
1. **Mesh**: thicker (scale the thin axis), or larger overall; re-check §2. Fallbacks (`SYNTH FAILED`)
   almost always mean "too flat". Lifts-then-drops on a wide non-convex shape = the planner's
   two-extremity preference (letters) — not fixable per object; report it.
2. **Material**: stiffer E raises grip per mm of indentation (tofu 50 kPa → mushroom 300 kPa helped
   grip force, not the grasp style); yield sets what counts as gentle.
Do NOT change planner constants, speeds, standoff, DR shared blocks, or the action box.

## 7. Collect
`gentle_manip/scripts/final/collect_demo_template.sh`: set `obj`, `n_episodes`, `SEED` (different per
parallel job). It stamps the run, tees the log, and copies the resolved experiment + leaf yamls into
`<run>/config/`. Runs land in `dataset/demos/single_lift_<name>_soft/<yy-mm-dd>-<abc>/` (random suffix:
parallel-safe). Register the object in the DEVLOG (mesh source, real extents, chosen knobs, smoke
success) — that entry is what the next person reads.
