# Grasp synthesis + execution — the frozen pipeline (v4.2, 2026-09-06)

What a collector run does, per object, so you can reason about results without reading the code.
Code: `grasp_synthesis/smgrasp/finger_grasp_final.py` (planner), `grasp_synthesis/collect_demos_synth_v4.py`
(executor + data), configs under `gentle_manip/configs/`. **The synthesis and execution logic is frozen at
commit `e7ca088` (v4.2)** — the cluster campaign collects on it. Add objects, tune their meshes/materials/
configs, do not change the planner or executor. Changes since the 2026-09-05 freeze are marked **[v4.2]**.

## 1. One experiment config drives everything
`configs/experiments/<name>.yaml` names five leaves: `task` (object, spawn, board, camera, MPM grid),
`action` (`abs_pose_euler_abs_gripper_z15`: absolute 7-D euler actions, TCP box x 0.26–0.55, y ±0.225,
z 0.015–0.50), `dr` (pose/shape/material randomisation + start modes + `disturbance_prob`),
`augmentation` (`d435i_noise` — sensor noise only, the SIM-ROLLOUT/EVAL default; the collector does NOT
apply it: demos are recorded clean, train-time augmentation lives in the training loss — see
`docs/training_plan_sim2real_2026-09.md`), `obs` (`superset_soft_armfocus_board`: crop z ≥ 19 mm = the
real deploy crop **[v4.2]**). Collection, training and eval all load the same file (`Experiment.load`).
Naming: `single_lift_<object>_soft_abs_action_armfocus_7d_realws`.

## 1b. Worked example: what a tofu collection run actually uses (and does NOT)
`single_lift_tofu_soft_abs_action_armfocus_7d_realws.yaml` → the collector reads exactly four leaves
(`exp.task_cfg`, `exp.collection_obs()`, `exp.action_config`, `exp.dr`); `augmentation:` is never read.

| leaf | file | what the collector takes from it |
|---|---|---|
| task | `tasks/single_lift_tofu_soft.yaml` | object `tofu`, spawn z 0.062 on the 13.8 mm board (centre 0.41, 0), MPM grid 250 / 235 substeps, `hold_steps` 30, camera at the **2026-09-05 recalibrated extrinsic**: pos (0.780, −0.004, 0.289), lookat (0.259, 0.012, 0.014), fov 43.15° |
| obs | `obs/superset_soft_armfocus_board.yaml` | crop z ≥ 19 mm (= real deploy), 1024 points, voxel outlier filter (1 cm / 23 neighbours), object focus (z < 12 cm or within 11 cm of the EE), quat jitter 0.003; privileged stress/object-pos/contact labels. **No `ground_residual` block** (that filter is real-only, in `point_cloud_1cam_armfocus.yaml`) |
| action | `action/abs_pose_euler_abs_gripper_z15.yaml` | TCP box x 0.26–0.55, y ±0.225, z 0.015–0.50; 10-D rot6d actions are RECORDED (gripper last), converted to 7-D euler at training time with this yaml + `abs_pose_abs_gripper_z15.yaml` as the source |
| dr | `dr/soft_orientation_realws_tofu.yaml` | object xy x 0.30–0.46, y ±0.12; yaw ±180°, pitch/roll ±45°, 25 % flips; scale 0.8–1.4, taper ±0.01; E 3–8e4, ν 0.28–0.38, ρ 900–1100, coupling friction 3.5–4.5; home jitter 2 cm; start modes home 0.6 / in_air 0.15 / above_object 0.15 / mid_approach 0.1; `disturbance_prob` 0.1 (never with above_object) |

**Not used by collection (train-time or real-side only):** the `d435i_noise*` augmentation yamls (sensor
noise, leaked-residue clusters, occlusion patch), the rigid cloud offset, the encoder consistency and
paired real–sim terms, the hold-tail post-processing (`augment_hold_tail`; the collector records the hold
natively), and the real deploy's `ground_residual` filter. The collector and planner reference none of
these symbols (checked 2026-09-06). Recorded demos are therefore CLEAN sim clouds through the deploy-
identical crop/filters; every robustness measure is applied afterwards, in the loss or on the rig.

## 2. Per batch (10 envs, one Genesis scene)
1. **Scene DR** (every batch): one mesh variant — size (`object_scale`), shape (bend/twist/taper/axis
   scale), material (E, ν, ρ; yield is NOT randomised: registry material) — shared by the 10 envs.
2. **Reset**: object placed at a random pose (xy in the DR box, yaw ±180°, pitch/roll ±45°, 25 % flips),
   arm at home (0.45, 0, 0.20) with ±2 cm jitter; settle.
3. **FEM build** (once per mesh): the object mesh is repaired and decimated IN MEMORY to ≤2000 uniform
   faces, coarse CAD meshes are isotropically remeshed to 4 mm, then tetrahedralised directly (no
   voxel remesh: that dilated every scan by ~2 mm/side). Nothing is written to disk.
4. **Grasp planning** (per env, ~5–10 s nominal, GPU FEM): a 7-DOF TCP grasp `[x,y,z,roll,pitch,yaw,width]`.
   - Seeds: 2600 antipodal surface pairs + 500 medial-axis points × 4 widths (4600 seeds), width measured
     on the LOCAL stroke inside the pad footprint (non-convex objects), fingertip height random within
     the object.
   - Search box **[v4.2, cluster]**: xy = 1.3 × the object's bbox, centred on the **bbox centre** (was the
     COM: a bent object's extremities fell outside the box). Rotation box roll ±30°, pitch ±20°, yaw ±60°
     about top-down; TCP z ≥ 15 mm (= the action box / real EE clip); width 10–88 mm.
   - Filters: fingers ≥ 2 mm above the board, rotation box, finger–object penetration ≤ 10 mm.
   - Score (batched GPU FEM, displacement-controlled contact on the pad footprint): gates for table,
     penetration (> 5 mm), indentation, force holdability 2µN ≥ m(g + 9.81), torsion, yield;
     score = −top-10 % von-Mises stress − 0.1·contact pressure. Top-6 seeds (`TOP_K`) → CMA-ES (400 evals
     each, steps 2 mm / 5° / 2 mm; seeds are clipped into the box and a seed whose CMA init fails is
     dropped **[v4.2, cluster]**) → ±3 mm width refine → argmax.
   - **Relaxation tiers [v4.2]** — only when a tier yields NO holdable grasp (draws that succeed at tier 0
     are bit-identical to the pre-tier planner):
     | tier | rotation box | penetration (filter + scorer) | pressure weight | yield gate |
     |---|---|---|---|---|
     | 0 nominal | ±30 / ±20 / ±60° | 10 mm / 5 mm | 0.1 | on |
     | 1 | roll/pitch +10°, yaw ±80° | 20 mm / 20 mm | 0.2 (×2, anti-pinch) | on |
     | 2 | as tier 1 | as tier 1 | 0.2 | **off** (last resort) |
     | 3 fallback | the filter survivor nearest the COM (`fallback_seed`); the episode runs and simply fails to lift if it cannot hold |
     The tier is recorded per episode (`synth_tier` in `dr_params.csv`, histogram `synth_tiers` in
     `stats.yaml`). Diagnosis of WHY a draw needs relaxation: `scripts/final/synth_diagnose.sh`
     (`--skip-execution`: per-env seeds by generator, rejections per filter, scored statuses, holdable
     counts, CMA statuses, tier → `synth_stats.csv`). On banana_chunk the binding constraint was the
     penetration cap on width (friction gate `2µN ≥ m(g+a)` fails at the allowed indentation).
5. **Start condition** (`dr.start_modes`, per env): `home` (default 60 %), or teleport to `in_air`
   (random workspace pose), `above_object`, `mid_approach` (on the home→grasp line), gripper part-closed
   20–80 mm and re-opened at 2.2 mm/step. Fingers ≥ 3 cm above the object, inside the action box −5 mm.
6. **Execution** (scripted, open loop, recorded through the SAME perception/action pipelines as real):
   approach in two legs at 2.4 mm/step — to a **standoff** on the grasp's approach axis at the start's
   own axial distance clamped to 4–10 cm (no up-then-down from a start already near the axis), then
   straight along the axis into the grasp (open fingers straddle the object: no diagonal collisions) —
   settle 1 → close at **2.2 mm/step** (the measured real teleop rate) to the planned width **− 0.8 mm**
   → dwell 2 → lift 0.2 m → **hold 10 [v4.2]**. The trailing hold is never trimmed (it is the only
   supervision for "arrived: keep commanding this pose, gripper closed"; the old 12-trimmed-to-4 caused
   mid-air reopens, 60 overwrote re-open-on-empty-grasp); mid-episode identical-command runs > 8 are
   still collapsed to 4. `disturbance_prob` (default 10 %): a 4-step lateral drag on the OBJECT during
   the approach; after 16 settle steps the grasp is re-targeted by the object's xy displacement and
   re-approached via the new standoff (recovery demos). **[v4.2]** The drag is never drawn for an
   `above_object` start — `disturbance_prob` is the probability conditional on the other start modes.
7. **Saved**: successes only (object above half lift height at the end), `data.pkl` shards, per-attempt
   `dr_params.csv` (pose/scene/material DR, start mode/width, drag/retarget, success/ever-lifted, max
   stress/yield, planned grasp, `synth_tier`), `config.yaml` (experiment name, DR, control knobs, git
   commit), `stats.yaml` (success/ever/sub-yield/stress, per-stage timings, `synth_tiers`), videos +
   final-grasp PNG per recorded episode (`--record-video N`).

## 3. Numbers to expect
2026-09-05 profiling (20 eps × 10 envs, pre-tier planner): tofu 100 % success / 100 % sub-yield / 0.4 min
per saved episode; strawberry 90 %; banana_chunk 52 % (mesh since thickened). Execution dominates wall
time (16–23 s per env at 3–4 FPS); synthesis 3–8 s per env nominal, up to ~3× on draws that need tier
1–2 (each tier re-runs filter + score + CMA). Letters (non-convex): planner prefers wide two-extremity
grasps that pivot out → low success — known, frozen. **[v4.2]** banana_chunk seed 0, 10 draws with the
tiers: 10/10 solutions (6 at tier 0, the rest relaxed; before: roughly 1 in 3 draws had a solution).

## 4. What is NOT tunable (frozen) vs what is
Frozen: everything in §2.4–2.6 (seeds, search box, gates, score, tiers, CMA budget, speeds, standoff,
drag, hold). Tunable per object: mesh (size/thickness), registry material, and the object-specific
config block (`docs/adding_new_objects.md`). Diagnostics: `dev_synth.sh` (viewer + `--dev-viz`
step-through of seeds/filter/score/CMA/refine/final — the final panel is shown once, after the tiers —
plus a particles-vs-FEM overlay), `scripts/final/synth_diagnose.sh` (synthesis-only statistics, nothing
executed), `[proj]`/`[grasp]`/`[start]`/`[drag]`/`[synth]` log lines,
`gentle_manip/scripts/final/profile_demo_collection.sh` for success/gentleness/speed.
