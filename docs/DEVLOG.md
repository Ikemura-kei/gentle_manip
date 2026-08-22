# Development Log — Gentle Manipulation Sim2Real

**Mission.** Develop a sim2real policy-learning framework for gentle manipulation of
real-world food, where demonstrations come from a **stress-aware scripted demonstrator with
grasp synthesis** (CMA-ES over grasp poses + von-Mises-stress-aware execution on soft MPM
bodies). End goal: a policy that handles **multi-category objects** (tofu, mushroom,
fruits, …). Current stage: a **specialist** policy (mushroom only) used to nail down the
training/data foundation; the **generalist** data-collection/training recipe is a later
investigation.

Subpages (detail documents):
- [Action-space ablation & follow-ups — final report](action_space_ablation_final_report.md) — all runs, curves, numbers
- [hwo dataset investigation](hwo_dataset_investigation.md) — collection-recipe forensics (R1/R2)
- [Abs-action derivation debugging](debug_partC_euler_action_anomaly.md) — the euler-seam and fixed-point-stall root causes
- [Campaign instructions (original)](cluster_experiment_action_space_ablation.md)

---

## Current foundation (adopted 2026-08-23, pending real-robot validation)

The following stack is the recommended baseline for further development. **Every element
is backed by the sim experiments in the final report; none of it is yet validated on the
real robot — that is the next gate.**

| element | choice | why |
|---|---|---|
| point cloud | **arm-focus filter** (1024 pts) | matches the real rig's processed cloud; costs ≈0 in sim (R1 0.76 = R2 0.76) |
| rotation obs | **quaternion** (canonicalized sign) | obs_dim 8; rot6d obs adds nothing for the EE's bounded range |
| action space | **7d absolute** (pos3 + euler3 + grip1, frame-offset euler) | ≈ delta in sim (0.76 vs 0.75); absolute re-anchors every step, so it does NOT accumulate drift in real, where the sim2real gap would otherwise compound delta errors; ≈ 10d rot6d (0.84 vs 0.88) at 3 fewer dims |
| action derivation | **recorded COMMANDED targets** (`--derive-source-action`), +K=4 lookahead for slow real teleop | achieved-pose derivation stalls at a closed-loop fixed point (see debug doc); commanded supervision also gives stable checkpoint curves. NOTE: commanded targets carry the closed-loop lead INHERENTLY (controller tracking lag, 5-10 mm) — natively recording 7d-euler commands at collection time would be bit-equivalent (decode→re-encode is lossless, ~1e-7) and needs no lookahead; we record 10d + derive only to keep one collection convertible to ANY action space. Lookahead is a patch for lead-deficient sources only (achieved poses; slow real teleop at ±2.6 mm) |
| workspace DR | **real-workspace absolute spawn box** x [0.29, 0.48] × y [−0.11, 0.11] | matches the physical table; 0.71 sim success on the 4.6×-larger region |
| demo collection | **v3 collector, hwo recipe**: phases 77/30, `grasp_extra_close=5 mm` (+2 mm soft-firm) | the recipe (not the cloud, not luck) explains the 0.62→0.76+ policy gap and 85%→94% collection success |
| trainer | DPPO PointNet diffusion (big net) | DP3-vs-DPPO comparison in flight (see Open questions) |
| checkpoint selection | sweep every checkpoint with the canonical harness; peak is at 100-300/600 | every stable run peaks early; never ship the final epoch |

### Reproduction recipe (exact commands)

**1. Collect** (GH200 node, ~6 h / 650 episodes, ~93-94% demonstrator success):
```bash
EXPERIMENT=single_lift_mushroom_soft_abs_action_armfocus_realws N_EPISODES=650 N_ENVS=8 \
MAXFEVALS=1145 SEED=0 SCENE_DR_EVERY=1 N_HOME_TO_PRE=77 N_GRASP=30 GRASP_EXTRA_CLOSE=0.005 \
sbatch gentle_manip/scripts/arrhenius/collect_demos_synth.sbatch   # runs the v3 collector
```
(non-realws variant: experiment `single_lift_mushroom_soft_abs_action_armfocus`)

**2. Convert** (commanded-euler 7d; obs = ee_pos + ee_quat + gripper_width + cloud):
```bash
uv run --project envs/dppo_arrhenius --no-sync python -m gentle_manip.dppo.convert_demos \
  <collection>/data.pkl --out dataset/dppo/<name> \
  --obs-keys ee_pos ee_quat gripper_width --point-cloud \
  --derive-action gentle_manip/configs/action/abs_pose_euler_abs_gripper.yaml \
  --derive-source-action gentle_manip/configs/action/abs_pose_abs_gripper.yaml
# real teleop pkls (recorded delta actions): --derive-source-action <delta cfg> --derive-lookahead 4
```
Pre-flight gate before training: euler dim3 seam-free (0 jumps) AND commanded lead
p75 ≥ 5 mm (see the verify blocks in the scratchpad chain scripts / debug doc).

**3. Train** (DPPO, jfhlu config: horizon 4 / cond 2 / denoise 20, visual_feature_dim 512,
mlp [1024,1024,1024], 600 epochs, checkpoint every 100):
```bash
DATA_ENV=<name> N_EPOCHS=600 SAVE_FREQ=100 \
CFG_DIR=$PWD/gentle_manip/dppo/cfg/single_lift_mushroom_soft_abs_pcd_hwo CFG_NAME=pre_diffusion_pointnet \
GM_EXTRA_OVERRIDES="action_dim=7 experiment=single_lift_mushroom_soft_abs_action_armfocus_7d_realws" \
sbatch gentle_manip/scripts/arrhenius/dppo_pretrain.sbatch
```

**4. Evaluate** every checkpoint with the canonical harness (200 eps, 5 envs, seed 42,
scene_group_size 4, 300 sim steps, per-episode video + DR audit) via
`dppo_eval.sbatch` with the SAME experiment; pick the peak checkpoint.

**5. Deploy**: pair the checkpoint with ITS dataset's `normalization.npz`; checkout must
include commits `76f5efa` (euler frame offset) + `9938b40` (7d deploy warmup); the deploy
experiment must reference `abs_pose_euler_abs_gripper`. For real: create
`single_lift_mushroom_real_abs_7d` experiment config (TODO). Real-table object placement:
inside x [0.29, 0.48] × y [−0.11, 0.11] (robot-base frame).

---

## Conclusions to date (2026-08-23)

1. **7d euler-abs ≈ 10d rot6d** in sim (0.84 vs 0.88) once the euler encoding carries a
   frame offset and commanded-target derivation → the compact action is (nearly) free.
2. **Abs ≈ delta in sim**; abs chosen for real deployment because it re-anchors to an
   absolute pose every step — no drift accumulation under sim2real execution error,
   unlike delta where within-chunk open-loop drift and target-vs-actual accumulation
   compound.
3. **Demonstrator grasp firmness is a first-order lever**: the v3 recipe's extra squeeze
   (5 mm + soft-firm) moved trained-policy success 0.62-0.70 → 0.76+ and demonstrator
   success 85% → 94%. Collection reproduces deterministically (3 collections within 0.6%).
4. **Arm-focus cloud is free in sim** and matches the real rig → strictly preferable.
5. **BC'd absolute actions must be supervised with targets that LEAD the pose**
   (commanded or ≥4-step lookahead); achieved next-pose targets create a closed-loop
   fixed-point stall (0% success with perfect training loss). Two encoding bugs (euler ±π
   seam; the stall) cost the first two rounds — both now have automated pre-flight gates.
6. **Co-training with the 50 real demos: no harm in sim, and plain concat (no
   oversampling) is the best variant** — confirmed: no-oversample peaks 0.785/0.76 (std
   box) and 0.685/0.585 (realws) vs ×4-oversample 0.745/0.65 and 0.65/0.23. Whether it
   HELPS in real is untested (the whole point of adding real data).
7. **Early checkpoints win**: every stable run peaked at 100-300 of 600 epochs (DPPO) and
   real-data val loss bottomed at ~epoch 600-1000 of 6000 — sweep-and-pick is mandatory,
   and DP3 must retain all periodic checkpoints (`checkpoint.topk.k=12`; its default k=1
   with no sim score silently keeps only the most-overfit checkpoint).
8. **Best current deployment candidates** (sim-ranked; real value untested):
   `nmbtz/state_500` (pure-sim realws 0.71), `afucm/state_400` (realws + real 0.685),
   `wyigy/state_100` (std box + real 0.785), `vdmtb/state_200` (pure-sim std 0.76),
   `qjzsf/state_500-1000` (real-only DPPO abs).

## Open questions (gates before further building)

- **Real-robot validation of the whole foundation** — nothing above is real-verified yet.
- **Does real co-training help in real?** (sim says it's free; real says nothing yet)
- **DP3 vs DPPO codebase** — harness evals of DP3 now exist (real-data DP3-abs already
  shows 0.40 sim success through the real→sim gap in a smoke); a DP3-on-sim arm
  (armfocus_firm data) is training for the clean comparison vs DPPO's 0.76. **We keep
  whichever trains better.**
- Residual ~0.08 original-hwo vs fresh-collection gap: run variance vs the genesis
  submodule bump — only matters if it reappears.

## Roadmap / TODO

**Evaluation & analysis**
- [ ] **OOD generalization test**: eval sets with out-of-training-range size/shape
  (scale > 1.5, bend > 25°, novel taper/axis combinations) — current tests are all
  in-domain. Also zero-shot other categories (tofu block, fruit meshes) as a generalist
  preview.
- [ ] **Real-vs-sim demo trajectory analysis**: compare the scripted demonstrator's
  kinematics (speeds, pauses, approach angles, grasp-close timing) against the real human
  teleop demos; match the scripted trajectory properties to human execution where they
  differ (a likely sim2real lever on the DATA side).
- [ ] **Gentleness-aware model selection**: the harness already records per-episode stress
  — rank checkpoints by success AND stress percentiles, not success alone (the mission is
  gentle manipulation, not just lift success).

**Demonstrator improvements**
- [ ] **Occlusion-aware grasp synthesis**: penalize grasp poses where the fingers occlude
  the object from cam_ext in the CMA-ES cost — fewer finger-occluded observations should
  help the visual policy.
- [ ] Failure+recovery demonstrations (induced slip + regrasp — ideas 2/3 in the CLAUDE.md
  retry brainstorm) so the policy sees recovery behavior.

**Policy / architecture**
- [ ] **DP horizon ablation**: prediction horizon 8 with execution steps 4 (current DPPO:
  4/4; DP3 default: 16 predict / 8 execute) — re-planning at half-horizon is the standard
  diffusion-policy sweet spot.
- [ ] **Memory in the policy** (RNN/GRU or temporal transformer over obs history): retain
  information from early frames where the object is fully visible — through later
  occlusion by the gripper, and for committing to a grasp width from the first clear
  view. Alternative lightweight version: append the FIRST frame's cloud (or its encoding)
  as a persistent context token.
- [ ] Aux objectives on the real data too (object-pos head currently sim-only — a real
  object-detector label could supervise it).

**Scaling toward the generalist**
- [ ] Multi-category collections (tofu/fruit meshes + per-category material presets from
  `assets/materials.py`); category-mixed training; measure specialist-vs-generalist gap.
- [ ] Data-scale study (650 → 2k episodes; collection is cheap at ~6 h/650 and
  deterministic).
- [ ] Camera-pose DR (real extrinsics drift between calibrations).

**Bookkeeping**
- [ ] `single_lift_mushroom_real_abs_7d` experiment config + deploy-script entries for the
  shortlist checkpoints (deploy-side).
- [ ] Port the pre-flight dataset gates (seam + lead) from the scratchpad chain scripts
  into a repo script (`gentle_manip/scripts/verify_derived_dataset.py`).

---

## Log

**2026-08-23 — DP3 harness integration + two eval-comparability bugs (user-caught).**
DP3 checkpoints now evaluate through the canonical harness (`eval_dp3_harness.py` +
`dp3_eval.sbatch`). The user noticed from renders that DP3 evals' MPM particles looked
randomly sampled where DPPO's looked regular — real bug: `GM_MPM_SAMPLER=regular` is
pinned in `dppo_eval.sbatch` but was missing from the DP3 launcher (genesis silently
falls back to the random sampler → different physics discretization). Second bug found
while verifying: a venv horizon that isn't exactly `policy_steps × act_steps` skips the
per-batch truncation auto-reset and shifts the DR RNG stream → pose scenarios diverge
from batch 1 on (DP3's 37×8=296 vs DPPO's 75×4=300). Both fixed; batch-0 scenario params
verified bit-identical across stacks. **Eval-protocol rules adopted:** (1) any new eval
launcher must replicate ALL of `dppo_eval.sbatch`'s pinned GM_* env (sampler, substeps
discipline) — diff the sim-server logs' `[scene_builder] GM overrides` line when in
doubt; (2) real-demo-trained policies are evaluated with the `_realws` experiments (the
real workspace box is their data's home turf), sim-trained policies with the experiment
matching their collection's DR.

**2026-08-20 → 08-23 — Action-space ablation campaign + follow-ups.** Parts A/B/C
(DPPO vs DP3 × abs vs delta × real/sim data), two derivation-bug hunts (euler seam,
fixed-point stall), commanded-target derivation, hwo recipe forensics (R1/R2), the realws
workspace box, 8-run co-training matrix, DP3 canonical-harness adapter + retrains with
full checkpoint retention. Details: the three subpages above. In flight as of this entry:
DP3 harness evals (4 existing ckpts), DP3 Part A retrains (12-ckpt), DP3-on-sim arm
(train + sweep).
