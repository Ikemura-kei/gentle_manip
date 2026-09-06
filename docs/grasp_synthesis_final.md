# Grasp synthesis + execution — the frozen pipeline (2026-09-05)

What a collector run does, per object, so you can reason about results without reading the code.
Code: `grasp_synthesis/smgrasp/finger_grasp_final.py` (planner), `grasp_synthesis/collect_demos_synth_v4.py`
(executor + data), configs under `gentle_manip/configs/`. **The synthesis logic is frozen**: add objects,
tune their meshes/materials/configs, do not change the planner or executor.

## 1. One experiment config drives everything
`configs/experiments/<name>.yaml` names five leaves: `task` (object, spawn, board, camera, MPM grid),
`action` (`abs_pose_euler_abs_gripper_z15`: absolute 7-D euler actions, TCP box x 0.26–0.55, y ±0.225,
z 0.015–0.50), `dr` (pose/shape/material randomisation + start modes), `augmentation` (`d435i_noise`),
`obs` (`superset_soft_armfocus`). Collection, training and eval all load the same file
(`Experiment.load`). Naming: `single_lift_<object>_soft_abs_action_armfocus_7d_realws`.

## 2. Per batch (10 envs, one Genesis scene)
1. **Scene DR** (every batch): one mesh variant — size (`object_scale`), shape (bend/twist/taper/axis
   scale), material (E, ν, ρ; yield is NOT randomised: registry material) — shared by the 10 envs.
2. **Reset**: object dropped at a random pose (xy in the DR box, yaw ±180°, pitch/roll ±45°, 25 % flips),
   arm at home (0.45, 0, 0.20) with ±2 cm jitter; settle.
3. **FEM build** (once per mesh): the object mesh is repaired and decimated IN MEMORY to ≤2000 uniform
   faces, coarse CAD meshes are isotropically remeshed to 4 mm, then tetrahedralised directly (no
   voxel remesh: that dilated every scan by ~2 mm/side). Nothing is written to disk.
4. **Grasp planning** (per env, ~5–10 s, GPU FEM): a 7-DOF TCP grasp `[x,y,z,roll,pitch,yaw,width]`.
   - Seeds: 2600 antipodal surface pairs + 500 medial-axis points × 4 widths, width measured on the
     LOCAL stroke inside the pad footprint (non-convex objects), fingertip height random within the object.
   - Filters: fingers ≥ 2 mm above the board, rotation box (roll ±30°, pitch ±20°, yaw ±60° about
     top-down), finger–object penetration ≤ 10 mm, TCP z ≥ 15 mm (= the action box / real EE clip).
   - Score (batched GPU FEM, displacement-controlled contact on the pad footprint): gates for table,
     penetration, indentation ≤ 10 mm, force holdability 2µN ≥ m(g+9.81), torsion, yield; score =
     −top-10 % von-Mises stress − 0.1·contact pressure. Top-6 seeds (`TOP_K`) → CMA-ES (400 evals each, steps
     2 mm / 5° / 2 mm) → ±3 mm width refine → argmax.
   - No feasible grasp → "SYNTH FAILED" fallback (45 mm top-down); such episodes are never saved.
5. **Start condition** (`dr.start_modes`, per env): `home` (default 60 %), or teleport to `in_air`
   (random workspace pose), `above_object`, `mid_approach` (on the home→grasp line), gripper part-closed
   20–80 mm and re-opened at 2.2 mm/step. Fingers ≥ 3 cm above the object, inside the action box −5 mm.
6. **Execution** (scripted, open loop, recorded through the SAME perception/action pipelines as real):
   approach in two legs at 2.4 mm/step — to a **standoff** on the grasp's approach axis at the start's own axial distance clamped to 4–10 cm (no up-then-down from a start already near the axis), then
   straight along the axis into the grasp (open fingers straddle the object: no diagonal collisions) —
   settle 1 → close at **2.2 mm/step** (the measured real teleop rate) to the planned width **− 0.8 mm**
   → dwell 2 → lift 0.2 m → hold 20 (FROZEN 2026-09-06; the trailing hold is never trimmed since 2026-09-06 — it is the only
   supervision for "arrived: keep commanding this pose, gripper closed"; 12 trimmed to 4 caused mid-air reopens). `disturbance_prob` (default 10 %; CONDITIONAL on the start mode not being `above_object` — the two never
   combine, 2026-09-06): a 4-step lateral drag on the
   OBJECT during the approach; after 16 settle steps the grasp is re-targeted by the object's xy
   displacement and re-approached via the new standoff (recovery demos).
7. **Saved**: successes only (object above half lift height at the end), `data.pkl` shards, per-attempt
   `dr_params.csv` (pose/scene/material DR, start mode/width, drag/retarget, success/ever-lifted, max
   stress/yield, planned grasp), `config.yaml` (experiment name, DR, control knobs, git commit),
   `stats.yaml` (success/ever/sub-yield/stress, per-stage timings), videos + final-grasp PNG per
   recorded episode (`--record-video N`).

## 3. Numbers to expect (2026-09-05 profiling, 20 eps × 10 envs)
tofu 100 % success / 100 % sub-yield / 0.4 min per saved episode; strawberry 90 %; banana_chunk 52 %
(planner fallbacks on a 20 mm-tall object under the 15 mm TCP floor → mesh thickened). Execution
dominates wall time (16–23 s per env at 3–4 FPS); synthesis 3–8 s per env. Letters (non-convex):
planner prefers wide two-extremity grasps that pivot out → low success — known, frozen for now.

## 4. What is NOT tunable (frozen) vs what is
Frozen: everything in §2.4–2.6 (seeds, gates, score, CMA budget, speeds, standoff, drag). Tunable per
object: mesh (size/thickness), registry material, and the object-specific config block
(`docs/adding_new_objects.md`). Diagnostics: `dev_synth.sh` (viewer + `--dev-viz` step-through of
seeds/filter/score/CMA/refine/final + a particles-vs-FEM overlay), `[proj]`/`[grasp]`/`[start]`/`[drag]`
log lines, `gentle_manip/scripts/final/profile_demo_collection.sh` for success/gentleness/speed.
