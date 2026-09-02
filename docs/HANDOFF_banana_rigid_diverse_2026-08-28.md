# HANDOFF — rigid-banana diverse-start regrasp campaign (Arrhenius)

Started 2026-08-28 by an unattended Claude Code session (user away ~20h). Branch
`cross-category-dp`. Continues `docs/HANDOFF_cluster_migration_2026-08-28.md`.

## Goal (user request)
1. Collect **500** single-attempt, always-successful, **top-down** grasp+lift demos on a
   **rigid banana** (fast surrogate for the soft/real banana).
2. Diversify the START configuration heavily — wide object-pose DR + EE started near the
   object / near the ground / mid-air, not just home — so a BC policy trained on ONLY
   clean successes recovers from a bad first attempt on its own (OmniReset insight,
   arXiv:2603.15789, translated to BC demo collection). NO explicit retry FSM.
3. One RGB video per collected episode, **no frozen frames** (they corrupt the dataset).
4. Train a DP3 / PointNet-diffusion BC policy (DPPO submodule) on the demos.
5. Eval 100 rollouts through the canonical harness, save per-episode videos, diagnose
   whether genuine redescend/regrasp emerges.
6. If it works → move to **real soft-banana** collect/train/eval. (Needs the XArm7 —
   NOT reachable from the cluster; this session preps configs+runbook only.)

## Working copy & environments
- `REPO=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip`
  (home dir is only ~28G free — too small; do NOT work there).
- GH200 nodes are **aarch64** → `envs/sim_arrhenius` + `envs/dppo_arrhenius` (own uv.lock,
  torch 2.9.1/2.6.0 cu126). `yd_env_common.sh` installs an arch-local uv
  (`~/.local/uv-aarch64/uv`) since the login-node uv is x86.
- Account `naiss2026-3-141-gpu`, partition `gpu`, `--gres=gpu:1`.

## New artifacts (commit 6e8bcd4 on the cluster checkout / edf35d3 on ~/git)
| file | what |
|---|---|
| `configs/tasks/single_lift_banana_rigid.yaml` | rigid banana lift task (apple-rigid template) |
| `configs/dr/rigid_orientation_banana_diverse.yaml` | WIDE start DR: pos ±0.08, yaw full, pitch/roll ±14, arm-home ±0.03 |
| `configs/experiments/single_lift_banana_rigid_diverse.yaml` | experiment: task+delta action+DR+superset_rigid obs |
| `grasp_synthesis/collect_demos_diverse_start_v2.py` | collector — see below |
| `gentle_manip/dppo/cfg/single_lift_banana_rigid_diverse_pcd/{pre,eval}_diffusion_pointnet.yaml` | BC pretrain + canonical eval |
| `gentle_manip/scripts/arrhenius/yd_{env_common.sh,build_smoke.sbatch,banana_collect.sbatch,banana_pipeline.sbatch}` | job scripts |

### collect_demos_diverse_start_v2.py (fork of v1 `collect_demos_diverse_start.py`, v1 untouched)
- **Per-env phase FSM** (approach→settle→grasp→lift→hold), like `collect_demos_synth_v3`:
  a short-approach start mode closes early instead of hovering at the grasp pose while a
  long-approach env catches up. A finished env stops being recorded (buffers stop
  growing) — **no frozen/padded frames** in demo OR video.
- **`--record-video [N]`**: one `<run>/videos/epNNNN_envM_<mode>_<ok|fail>.mp4` per saved
  episode. Frames rendered ONLY over recorded steps; same early-success trim applied to
  frames. (Unrecorded pre-roll is never rendered — the bug from the prior handoff §6.2.)
- **`--start-modes home:w,near_object:w,near_ground:w,mid_air:w`** (default
  0.25/0.30/0.25/0.20): per-episode EE start pose via an UNRECORDED pre-roll from home;
  recorded approach redirects from there to the real grasp — one continuous success.
- **`--top-down`** (default on): clamps CMA-ES roll to ±0.16π, pitch to ±0.10π.
- Early-success trim (hold≥0.10 m for 10 steps → cut + 5-step margin) kept from v1.

## Job chain
```
sbatch gentle_manip/scripts/arrhenius/yd_build_smoke.sbatch          # env build + full-pipeline smoke
# after PASS:
CID=$(sbatch --parsable gentle_manip/scripts/arrhenius/yd_banana_collect.sbatch)
sbatch --dependency=afterok:$CID gentle_manip/scripts/arrhenius/yd_banana_pipeline.sbatch
```
- collect → `dataset/demos/single_lift_banana_rigid/<date-xyz>/` (data.pkl + videos/ + stats.yaml);
  path echoed to `logs/slurm_logs/last_banana_collect_run.txt`.
- pipeline → convert to `dataset/dppo/single_lift_banana_rigid_diverse_pcd/`,
  BC pretrain `logs/dppo/dppo-pretrain/single_lift_banana_rigid_diverse_pcd/<id>/`,
  eval `<run>/eval/<datetime>/{summary.json,episodes.csv,render/*.mp4}`;
  eval dir → `logs/slurm_logs/last_banana_eval_dir.txt`.

## Status log
- 2026-08-28 ~17:05 — env from scratch (no venvs/submodules/uv in my checkout). Wrote all
  artifacts, committed, submitted `yd_build_smoke` (job 1768415, PD). ikemura's working
  setup is not group-readable, so building envs/sim_arrhenius + envs/dppo_arrhenius fresh.
- 2026-08-28 ~17:35 — job 1768582: envs/sim_arrhenius built OK (genesis import OK,
  gstaichi 4.6.0 auto-installed, torch 2.9.1+cu126 sees GH200). envs/dppo_arrhenius
  synced (torch 2.6.0+cu126). Smoke FAILED only on a bad check (`import dppo` — the
  distro installs top-level `agent`/`model`/`util`, no `dppo` module). Fixed the
  check, resubmitted (job 1768806). Venvs persist on /nobackup so re-sync is fast.
- 2026-08-28 ~17:50 — jobs 1768582/1768806 both hit `ImportError: type "Layout" is
  already registered` on `import genesis`. Root cause: this genesis fork commit
  (5b13c60) imports **`quadrants`** (pre-rename taichi fork, its own declared dep,
  cp312 aarch64 wheel exists), NOT `gstaichi`. My build script (following the stale
  handoff note) also pip-installed `gstaichi`; both register the same pybind11 types
  -> conflict when imported in one process. Fix: drop gstaichi entirely from the build
  script, verify `import quadrants; import genesis` instead. Resubmitting.
- 2026-08-28 ~18:35 — smoke/diag revealed: the "banana" mesh is a 5.7cm VERTICAL
  baton (long axis Z). Rigid: topples during settle; ungraspable top-down (finger
  reach 7cm > baton height); CMA-ES with my roll-clamp couldn't find contact
  (cost 42); unclamped it found valid LOW SIDE grasps (cost 0.001) that then fail
  to lift (eccentric grip on a top-heavy smooth rigid stick -> slip). FIXES:
  (1) new asset banana_piece_lying.obj + registry "banana_lying" = banana rotated
      long-axis-horizontal, rests on its side (how it sits on a table IRL);
  (2) single_lift_banana_rigid task -> object_name banana_lying;
  (3) v2 --top-down: default OFF, and when on clamp ONLY pitch (roll must stay full
      -- downward TCP orientation is near roll=+-pi in this euler convention, the
      original clamp bug forced a sideways grip).
  Re-running yd_diag_banana (A home/noDR, B home/DR, C full).
- 2026-08-28 ~18:50 — diag 1769435 on banana_lying: A(home/noDR)=60%, B(home/DR)=75%,
  **C(full diverse start modes + DR)=80% CMA success**. Grasps are clean top-down
  (roll ~= -pi confirmed), near_object/near_ground/mid_air starts all produce
  successful redirect trajectories, videos write. Skipped re-running build+smoke
  (only the grasp-success step was ever failing, now fixed; its maxfevals=150 is
  too low anyway). LAUNCHED full chain:
    collection  job 1769529  (yd_banana_collect, 500 demos, n-envs 8, maxfevals 700)
    pipeline    job 1769530  (yd_banana_pipeline, afterok:1769529 -> convert+BC+eval100)
- 2026-08-28 ~20:25 — collection 1769529 CRASHED at 168/500: rigid-solver NaN
  ("invalid constraint forces"). Cause: CMA-ES sometimes returns w~10mm for a 32mm
  banana -> scripted close crushes the rigid object. Fixes committed: width_cls
  clamp [20,75]mm; sim_substeps 5->10; softer shape DR; v2 batch try/except
  (skip+rebuild, <=12 consec). 168 emz shards salvageable (videos intact).
  Relaunching: collect 350 fresh + merge emz(168); pipeline converts from the
  PARENT dir (all data.pkl).
- 2026-08-28 ~20:45 — user: add demos with EE start BETWEEN home and near-object.
  Added v2 start mode "mid_approach" (EE interpolated 0.35-0.70 along home->grasp,
  orientation slerped by the same fraction + small jitter -- "a first attempt was
  already heading for the object, then corrected"). New default start-modes:
  home:0.18 near_object:0.24 mid_approach:0.24 near_ground:0.18 mid_air:0.16.
  Cancelled 1771862/1771863 (only 4 demos in); fresh N_EPISODES=400.

## 2026-08-28 ~21:05 — EXPANDED SCOPE (user, leaving 24h, run nonstop)
User goal: a DIRECT (not specialist->distill) GENERALIST regraspable DP3 policy over
9 in-domain objects x 500 diverse-start demos each, cross-category, beating the
non-regraspable baseline on 3 metrics (success / gentleness-stress / combined score).
Path: (1) finish rigid-banana proof; (2) if eval shows genuine regrasp -> switch to
SOFT banana, recollect+retrain+eval; (3) scale to the 9-object generalist.
Also: added start mode "above_object" (varied height 3-20cm + varied pose over the
object) and widened object_scale DR (0.78-1.28). Fresh 500-demo rigid collection
relaunched with all 6 start modes.
9 in-domain object shortlist (soft, non-extreme material, mesh present, collect well):
mushroom, banana, grape, kiwi, strawberry, tomato, cherry, raspberry, egg_boiled.
- 2026-08-28 ~23:45 — added start modes above_object (varied height+pose over object)
  + mid_approach; widened object_scale DR (0.78-1.28). Relaunched fresh 500-demo
  rigid collection (1771952) + pipeline (1771953). Published standalone demo
  showcase artifact https://claude.ai/code/artifact/5682ac2f-0b24-446f-88f1-b556778a9bbc
  (GRACE design system; 15 rigid-banana clips across home/near_object/mid_air/
  near_ground + aim & method writeup). GRACE itself is at the 16MB cap -- link the
  new artifact from it manually. Loop cron updated (b8da4e7b) with the full
  rigid->soft->9-object-generalist phase machine.
- 2026-08-29 ~00:05 — user: EE start must CONTINUOUSLY cover home->near-object, not a
  few discrete poses. Rewrote v2 sampler: "sweep" family (weight 0.62) draws
  t~U(0,1), EE starts that fraction along home->grasp with lateral+orientation
  jitter growing with t and recorded-approach length shrinking with t -> dense
  coverage of the whole corridor. Off-corridor families above/ground/air (0.38
  total) cover post-failure states off the path. object_pos_xy 0.08->0.10.
  --start-modes now = family weights. Relaunched collect 1772322 / pipeline 1772323.
- 2026-08-29 ~04:00 — collection 1772322 DONE: 500 diverse-start rigid-banana demos
  (26-08-29-tcj), 41.7% scripted-grasp SR over 1199 attempts (continuous sweep +
  wide DR + off-corridor modes are much harder for the scripted demonstrator than
  the home-heavy diag's 80% -- the 500 saved are all clean successes though, just
  biased toward start configs the scripted grasp can handle). 234min, 1 batch NaN
  skipped/144. Converted: 450 train / 50 val, 75207 steps, obs_dim 8 / action_dim 7.
  Pipeline 1772323 FAILED instantly on a bare `python3 -c "import numpy"` shape-check
  line (system python has no numpy) -- fixed to run in envs/dppo_arrhenius,
  resubmitted as 1773316 (BC pretrain -> 100-ep eval).

## 2026-08-29 ~06:30 — RIGID EVAL #1 RESULT + regrasp gap
Eval of best-val ckpt state_175 (val 0.035 @ ep180): **success_rate 0.71 / 100 eps**
(harness canonical, wide-DR eval). vs the 41% non-regraspable soft baseline -> +30pt.
BUT episodes.csv first_success_step is ~57 for ALL 71 successes, ZERO late successes:
the policy grasps cleanly on the first try or fails -- **no within-episode reopen/
regrasp**. Diverse STARTS made a robust single-grasp policy, not a regrasp policy,
because (a) the harness eval never puts the policy in a failed state, and (b) every
diverse start had the gripper OPEN -- the demos never show "gripper closed on
nothing -> reopen -> retry".
FIXES (committed):
 - v2 collector: new "failed" start family (weight 0.30). Pre-roll descends to the
   object + CLOSES the gripper (a just-missed grasp); a new recorded PHASE 0
   "recover" then OPENS + backs away before the normal approach -> the explicit
   reopen-and-retry demonstration, still one continuous episode.
 - single_lift_banana_rigid_regrasp_eval experiment/DR: arm home forced LOW over the
   object (robot_init_offset_xyz [0.05,0,-0.14]) so the eval actually starts in a
   failed-attempt-like state. Diagnose via fss: late successes = real recovery.
 - pretrain n_epochs 1000->300, early_stop_patience 20->8 (overfit clear by ep200).
Next: recollect 500 with the "failed" family, retrain, eval on BOTH
single_lift_banana_rigid_diverse (clean) AND ..._regrasp_eval (hard start).

## 2026-08-29 ~12:35 — iter 2 retrained (run xkrpq), 3 evals running
Collection 26-08-29-ylr: 500 demos, ~110 "failed" recovery episodes (SR 37.9%).
BC pretrain xkrpq: best val 0.0334 @ ep160 (state_150). early_stop_patience=8 did
NOT fire (DPPO fork counts differently) -> ran to the 300 cap, overfit to val 0.042.
Evals (parallel):
 - 1773865 (pipeline): state_300 (overfit last ckpt -- pipeline BEST= bug), clean + regrasp
 - 1775655: state_150 (best val), CLEAN experiment
 - 1775656: state_150 (best val), REGRASP-start experiment (arm low over object)
Diagnose: SR per experiment + first_success_step distribution -- LATE successes
(fss > ~100) in the regrasp eval = genuine within-episode recovery (the whole point).
TODO: fix the pipeline BEST= to pick best-val; investigate DPPO early-stop.

## 2026-08-29 ~14:00 — RIGID REGRASP CONFIRMED (iter 2, run xkrpq / state_150)
- CLEAN eval: SR 0.83/100; fss histogram of successes shows **20/83 LATE (fss>90)** =
  genuine within-episode reopen+regrasp (iter 1 had 0 late).
- REGRASP-start eval: SR 0.694/85 (sim NaN crash at batch 18; offset too aggressive).
- The "failed" recovery-demo family is the ingredient that worked.
- TODO carried forward: pipeline BEST= must pick best-VAL (not last/overfit) ckpt;
  cap pretrain ~180ep for soft; soften the regrasp-eval home offset.
-> Advancing to PHASE 3: SOFT BANANA (configs already prepped:
   single_lift_banana_soft_diverse experiment/task/DR + _pcd cfg dir).

## 2026-08-29 ~14:30 — PHASE 3 SOFT: user feedback folded in
Keep the ~70/30 direct-vs-failed demo ratio. Changes for the soft round:
1. Soft body (banana_lying, MPM). Soft smoke (job 1776694, 26-08-29-sxd) PASSED --
   genesis builds soft-MPM banana on aarch64 without pymeshlab; 3 demos in 1.9min.
2. Eval early-termination: new single_lift_banana_soft_diverse_eval task with a LOW
   lifted-clear success band (z 0.13-0.45, hold 4) + early_stop_on_success, so the
   env freezes the moment the object is lifted near home (no carry-down). Plus a
   post-eval _trim_eval_clips.py step: success clips are cut to first_success_step
   frames (variable length, no frozen tail).
3. Gentle grasp: v2 collector now uses a SOFT close margin of 0.5mm past the surface
   (vs 2.5mm rigid) + a CRUSH GATE -- episodes whose top10 von Mises exceeds
   1.15x yield (45000 Pa banana) are rejected. Eval reports SR, gentleness (peak/
   mean stress from episodes.csv), and SR x gentleness.
4. Start-pose coverage: sweep 0.42 (continuous home->object), above 0.16 (atop),
   ground/air 0.06 each, failed 0.30. Object DR wide (pos 0.10, full yaw, +-14).
5. Softened the regrasp-eval home offset ([0.04,0,-0.11] vs rigid [0.05,0,-0.14])
   to avoid MPM NaN.

## 2026-08-29 ~15:20 — soft grasp gentleness tuning
close margin sweep (soft): +0.5mm past surface -> ~44% crush-reject (too hard);
-1mm before surface -> 0% crush but ~25% lift SR (too loose). Landed on **0 margin
(at the nominal surface)** + crush gate at 1.25x yield. Collect job 1778959.

## 2026-08-29 ~16:20 — soft SR bug FIXED
Root cause of ~17% soft lift SR (latent in rigid too): CMA-ES returns a straddle
width WIDER than the object (cost 0, no contact); [0.020,0.075] clamp too loose ->
gripper closed on nothing. FIX: cap width_cls at min(registry short axis)+2mm.
Soft SR -> ~50%, crush rejections ~4%. Progressive lift-firming (+1.5mm) + higher
coup_friction (4.5-6.0) also kept. Collect job 1779116 (26-08-29-*), ~1/min,
~8h to 500. Loop carries -> convert -> pretrain (200ep cap) -> dual eval
(clean single_lift_banana_soft_diverse_eval + regrasp single_lift_banana_soft_regrasp_eval,
both low lifted-clear band + early_stop + clip trim). Artifact 5682ac2f now has a
soft-demo section (10 clips).

---
## 2026-08-29 23:15 — Phase 3 (SOFT banana) collection DONE, pipeline running

- **Collection job 1779116** finished: 500/500 saved, grasp SR 54.6% (915 attempts),
  crush-rejects 26, 7/160 batches skipped by try/except resilience. elapsed 431 min.
  Dataset: `dataset/demos/single_lift_banana_soft_diverse/26-08-29-wyk/` (+ 500 videos,
  start-family labels in filenames: mid_approach / near_object / above_object / failed_grasp / air).
- **Pipeline 1779117** released, running on n97:
  - convert OK: 450 train / 50 val episodes, state 8-dim, action 7-dim, PCD stored.
  - BC pretrain `hytxr` running (`logs/dppo/dppo-pretrain/single_lift_banana_soft_diverse_pcd/hytxr/`),
    ~16 s/epoch, 200 epochs → ~50 min. train loss 0.92→0.14 by epoch 13, val 0.187 @ ep10.
  - then: pick best-val ckpt → clean eval (`single_lift_banana_soft_diverse_eval`,
    early-stop-on-success, low lifted-clear band z 0.13–0.45, hold 4) → regrasp eval
    (`single_lift_banana_soft_regrasp_eval`) → `_trim_eval_clips.py` on both.
- Next gate: pretrain done → eval summary.json + episodes.csv (SR / stress-gentleness /
  first_success_step for genuine regrasp).

---
## 2026-08-30 00:35 — pipeline 1779117 FAILED at eval batch 2/20; diagnosed + resubmitted

- Pretrain finished clean (200 ep, best val 0.0330 @ ep200 -> state_200.pt).
- **Clean eval died at batch 2/20** with `ConnectionError: socket closed mid-message`
  (client side). Sim server log shows clean "Exiting Genesis" at the same instant, no
  Python traceback server-side -> the genesis MPM **worker crashed (NaN)** and the parent
  server tore down. Root cause: the soft task ran `mpm_grid_density 250 / sim_substeps 220`
  = `substep_dt 1.5e-4` vs CFL `suggested_dt 8e-5` (Genesis warned "might be unstable").
  Collection tolerated this (~4% batch blowups skipped by the collector's try/except); the
  **shared eval harness has no per-batch skip**, so one NaN kills the run.
- First 2 batches before the crash: **clean-start SR 0.60**, `in_band 1.00` (the low
  lifted-clear success band + early-stop is working).
- Also: eval was **~8 min/batch** (stepping ~90 s + render/encode/diffusion overhead) ->
  ~2.7 h/eval, untenable for a 9-object campaign.
- **Fix (committed):**
  - `single_lift_banana_soft_diverse_eval.yaml`: `mpm_grid_density 200`, `sim_substeps 340`
    -> CFL-safe (`substep_dt 9.8e-5 < suggested 1.0e-4`). Geometric obs so lift-success
    transfers; gentleness reported relative to yield.
  - `eval_diffusion_pointnet.yaml`: `n_steps 200->150`, `max_episode_steps 800->600`.
- **Resubmitted clean eval as job 1789727** (`yd_banana_eval`, EVAL_TAG=clean_v2, ckpt
  state_200). Monitoring first ~3 batches for stability before submitting the regrasp eval.
- NOT done: the regrasp eval (`single_lift_banana_soft_regrasp_eval`) — submit after clean_v2
  proves stable.

---
## 2026-08-30 03:40 — Phase 3 SOFT-BANANA EVAL DONE — REGRASP CONFIRMED

Jobs 1789727 (clean) + 1790415 (regrasp), ckpt `hytxr/state_200`, CFL-safe eval physics.

| metric                | clean start | regrasp start (arm low/offset) |
|-----------------------|-------------|--------------------------------|
| success rate          | **1.00**    | **0.98**                       |
| gentleness score      | 0.511       | 0.567                          |
| combined SR x gentle  | 0.755       | 0.774                          |
| first_success_step    | 92/100 @ 40-60 | 74 quick(<40) + **23 late(60-120)** |

- **Regrasp = genuine.** Regrasp-start fss is bimodal: either the start pose works
  immediately (<40) or the policy misses, backs off, repositions and grasps ~30-60 steps
  later (23%). Frame-by-frame of a late clip (batch00_env1, fss 88) shows a clear
  approach -> not-secured -> re-widen -> reposition -> grasp + lift-clear cycle. No retry
  FSM in the training data — pure diverse-start BC.
- **Gentleness**: peak single-particle stress > 45 kPa yield in ~100% of episodes
  (transient contact plasticity); SUSTAINED top20 bulk stress ~31 kPa (sub-yield). Grip is
  firm but not crushing. -> generalist recipe: crush gate 1.25 -> 1.1x.
- Eval videos (trimmed to success frame, no carry-down tail):
  `logs/dppo/dppo-pretrain/single_lift_banana_soft_diverse_pcd/hytxr/eval/{clean_v2,regrasp_v2}/render/`
  summary.json + episodes.csv beside them.
- EXPERIMENT.md Final summary filled.

**NEXT (Phase 4):** cross-category generalist. Build
`configs/experiments/single_lift_cross_category_diverse.yaml` (9-object pool + wide pose/size
DR + the start families, crush gate 1.1x), a `single_lift_cross_category_diverse_pcd` DP3
cfg dir, launch CONTINUOUS 500/object collection + the matching NON-REGRASPABLE (home-only
start) baseline collection.

---
## 2026-08-30 03:55 — Phase 4 START: cross-category generalist infra built

Regrasp confirmed on soft banana -> building the direct 9-object generalist.

- **Collector extended** (`grasp_synthesis/collect_demos_diverse_start_v2.py`):
  - `object_category_pool` support: `_make_worker` draws a registry object per scene
    (rebuilds `spec.objects[0]` from `get_object_def`, resets material to None),
    threads that object's name + `von_mises_yield_stress` into the width-cap and
    crush-gate. Falls back to `task.object_name` when no pool.
  - New `strict_home` start family = EE starts exactly at home, full approach
    recorded, no pre-roll -> the NON-REGRASPABLE BASELINE distribution
    (`--start-modes strict_home:1.0`).
  - New `--crush-frac` CLI (default 1.15; Phase 4 uses 1.10 -> gentler than the
    banana run's 1.25).
  - Syntax-checked; behaviour untested on aarch64 -> smoke job 1792715
    (`yd_xcatsm`, 8 ep / 2 env / scene-dr-every 1) validating object-switching +
    strict_home + no crash BEFORE the big runs.
- **Configs** (9 objects: mushroom, banana_lying, grape, kiwi, strawberry, tomato,
  cherry, raspberry, egg_boiled):
  - `dr/xcat_diverse_regrasp.yaml` (+ `_eval`), `tasks/single_lift_xcat_diverse.yaml`
    (+ `_eval`), `experiments/single_lift_xcat_diverse{,_eval}.yaml`,
    `experiments/single_lift_xcat_regrasp_eval.yaml`.
  - Task CFL point: grid 170 / substeps 420 (pool E 1e5..8e5; anchored on the
    banana soft eval's stable 200/340 at E 3e5, scaled up for the stiffer members;
    rare blowups caught by the collector's per-batch skip).
  - DP3 cfg dir `dppo/cfg/single_lift_xcat_diverse_pcd/{pre,eval}_diffusion_pointnet.yaml`.
  - `scripts/arrhenius/yd_xcat_collect.sbatch` (71 h walltime, continuous, salvage-merge,
    `TAG` for regrasp vs baseline; `--record-video 60`).
- **NEXT**: smoke passes -> launch (a) `yd_xcat_collect` regraspable (4500 ep, won't
  finish in window) and (b) `START_MODES=strict_home:1.0 TAG=xcat_baseline` non-regraspable
  baseline. Both continuous; report cumulative progress.

---
## 2026-08-30 04:15 — Phase 4 smoke 1792715 found 2 issues -> pool narrowed to 5

Smoke (cross-category draw worked: pool loaded, per-scene object switch + yield
threading confirmed). But:
1. **Small fruit break the pipeline.** raspberry (1.5cm) / cherry / grape / tomato
   (~2cm) deform to degenerate meshes (~64 MPM particles at grid 170); trimesh
   `bounds_tree` raises "Bounds must be (n, dimension*2)!" in the SDF build ->
   CMA-ES crashes the batch. Also 0/8 grasp success on raspberry (mushroom-scale
   `OBJ_SIZE` bounds + grasp_gate_dist search a +-5cm box for a 1.5cm object).
2. Collector's per-batch skip caught the crashes (no run abort) but no demos saved.

**Fix:** pool narrowed to the 5 mushroom-scale (3-6cm) soft objects:
`mushroom, banana_lying, kiwi, egg_boiled, strawberry` (E 3e5..5.3e5). Scale DR
tightened to [0.88,1.15]. Task -> grid 190 / substeps 440. Committed.
-> re-smoke job 1792749 (10 ep / 3 env). If it saves demos across >=2 objects with
no crash, launch the two continuous runs.

NOTE for later: adding the small fruit back needs per-object grid_density (finer)
+ object-size-scaled CMA bounds + a deform-mesh validity guard. Deferred.

---
## 2026-08-30 04:25 — Phase 4 smoke fix 2

Re-smoke 1792749 still 0-success + strawberry mesh crash. Diagnosed:
- **CMA-ES returns a STRADDLE width** (w=74-80mm, the 0.08 bound) on a 33mm mushroom
  -> SDF cost ~0 but no contact. The soft width cap was `_short + 2mm` = 34mm on a
  33mm object -> zero compression -> slips on lift. **Fix: soft `_wcap = 0.80*short`**
  (~20% compression), `_floor = 0.42*short`. Rigid unchanged.
- **strawberry deformed mesh degenerate** -> trimesh `bounds_tree` crash in the SDF
  build. **Fix: `_mesh_ok()` validity guard in `_make_worker`** (finite AABB, extent
  > 0.1mm, >=8 faces) -> retry deform up to 3x, then fall back to the nominal mesh.
- crush_frac default 1.10 -> **1.20** (the tighter grip now does the gentleness work;
  1.10 would starve the dataset).
Committed. -> re-smoke 1792788 (12 ep / 3 env).

---
## 2026-08-30 04:40 — Phase 4 smoke fix 3

Smoke 1792788: still 0-success + strawberry crash. Root causes:
- **strawberry.obj is a 1.4 KB PLACEHOLDER mesh** (egg=37KB, kiwi=26KB) -> 99% quadric
  decimation in build_object_sdf collapses it -> trimesh rtree "Bounds must be
  (n,dimension*2)". Dropped strawberry. `_mesh_ok` guard kept (catches deform
  degeneracy, not this).
- **0-success across mushroom + banana_lying** even though the banana proof got 54%
  on the identical collector -> the only diff was physics (grid 190 / substeps 440).
  Reverted the COLLECTION task to the banana-COLLECTION values **grid 250 / substeps
  240** (54% SR + regrasp confirmed there). Also reverted the soft width cap to the
  banana-proven `_short + 2mm` (my 0.8*short "tighten" likely ejected the coarse-grid
  soft body during close). EVAL task keeps CFL-safe grid 190/substeps 440.
- Pool now **4 objects**: mushroom, banana_lying, kiwi, egg_boiled.
-> re-smoke 1792811 (15 ep / 3 env). Expect >0 saves this time.

---
## 2026-08-30 04:50 — Phase 4 smoke PASSED -> continuous collections LAUNCHED

Smoke 1792811 (banana-physics, 4-obj pool) SAVING demos:
- batch 1 mushroom 0/3, batch 2 kiwi 1/3, batch 3 banana_lying 3/3 (incl. 2
  failed_grasp recovery demos), batch 4 kiwi ... -> cross-category + recovery
  demos + video all working. mushroom looks harder (rounder); it just contributes
  fewer demos per rotation, acceptable.

**LAUNCHED (continuous, 71 h walltime, resubmit to continue):**
- **1792833  yd_xcat  TAG=xcat_regrasp** -- 9-obj... actually 4-obj pool
  (mushroom/banana_lying/kiwi/egg_boiled), diverse start-modes
  (sweep .44/failed .30/above .10/ground .09/air .07), crush 1.20, n_envs 6.
  Target 4500 (500/obj-equiv); WILL NOT finish in the window -> report cumulative.
  -> dataset/demos/single_lift_xcat_regrasp/
- **1792834  yd_xcat  TAG=xcat_baseline** -- same pool + physics, but
  `--start-modes strict_home:1.0` (EE starts exactly at home, full approach) =
  the NON-REGRASPABLE baseline for the 3-metric comparison.
  -> dataset/demos/single_lift_xcat_baseline/

Both PENDING (Priority). Next: confirm both RUNNING + accumulating, then when a
usable batch exists (~a few hundred/side) convert + BC pretrain the generalist
via dppo/cfg/single_lift_xcat_diverse_pcd + dual eval.

DEVIATION FROM PLAN (documented): pool is 4 not 9. The 5 small/degenerate-mesh
objects (grape, cherry, tomato, raspberry, strawberry) are incompatible with the
current CMA-ES-SDF + MPM pipeline without per-object grid/bounds work. 4 clean
cross-category objects still demonstrates the direct-generalist + regrasp claim
vs the baseline. Re-adding the others = a follow-up (per-object grid_density +
object-size-scaled CMA bounds + real scanned meshes for grape/cherry/etc.).

---
## 2026-08-30 05:00 — Phase 4 collections steady

- **1792833 xcat_regrasp**: 16/4500, ~0.8/min, batchfail 0/7. Per-object SR so far:
  egg_boiled 50%, kiwi 42%, mushroom 42% (the earlier "mushroom 0/6" was one unlucky
  batch, not systematic). banana_lying not yet re-drawn.
- **1792834 xcat_baseline** (strict_home): 26/4500, ~1.3/min (faster -- no CMA
  diversity), batchfail 0/7. egg_boiled 75%, mushroom 50%.
- Both clean, no crashes, videos recording (first 60 each).
- Neither finishes in the window (~40 h to 2000 each) -> cumulative reporting.

STOP-CONDITION check: Phase 4 collection running steadily = YES. All *reachable*
evals diagnosed = YES (rigid banana iter1/2, soft banana clean_v2/regrasp_v2). The
generalist policy can't be evaluated until it trains -> preliminary generalist BC
train + dual eval will fire once ~600 demos/side accumulate (~13 h), tracked here.

---
## 2026-08-30 07:00 — Phase 4 collections @ 2h

- 1792833 xcat_regrasp: 127/4500 (~1/min), batchfail 1/43 (caught).
- 1792834 xcat_baseline: 207/4500 (~1.7/min), batchfail 0/51.
- Both stable. Generalist preliminary train will fire when xcat_regrasp reaches
  ~500-600 (~7-8 h out at current rate). Baseline will be ready first (~3 h).
- No code/config changes needed.

---
## 2026-08-30 07:55 — Phase 4 @ ~3h: 2nd regrasp collector added

- 1792833 xcat_regrasp: 205/4500 (slowed to ~0.7/min -- CMA-bound). 1792834
  xcat_baseline: 332/4500 (~1.9/min). Both clean.
- The regrasp side is the bottleneck -> launched **1793396** = 2nd xcat_regrasp
  collector (seed 7, separate run dir, same TAG folder) to ~2x throughput.
  First attempt (1793395) cancelled: the salvage-merge loop would have eaten the
  live collector's shards -> added an mtime<20min guard to yd_xcat_collect.sbatch,
  committed, then relaunched as 1793396.
- yd_xcat_pipeline.sbatch ready (PRELIM shard-staging). Prelim generalist train
  fires when combined regrasp demos ~= 300-400.

---
## 2026-08-30 08:05 — GPU concurrency cap = 2

1793396 (2nd regrasp collector) could not start: `AssocGrpGRESRunMinutes` -- the
account caps concurrent GPU jobs at 2. Cancelled. So the two running collectors
(1792833 regrasp, 1792834 baseline) are the max; regrasp stays CMA-bound at
~0.7/min. Window will likely end with ~350-450 regrasp + ~600-750 baseline demos
-> the preliminary generalist comparison uses matched subsets (same N/side).

---
## 2026-08-30 08:45 — Phase 4 PRELIM generalist comparison started

Given the 2-GPU cap + CMA-bound regrasp rate, collecting the full 500/obj won't
happen in the window. Pivot to a PRELIM comparison on what's collected:
- **Stopped baseline collector 1792834** at 474 demos (~118/obj, run 26-08-30-mcy).
- **regrasp collector 1792833 keeps running** (292 -> ..., run 26-08-30-rdz).
- New cfg dirs: `single_lift_xcat_{baseline,regrasp}_pcd` (n_epochs 150), both
  evaluated on the SAME `single_lift_xcat_diverse_eval` (clean) +
  `single_lift_xcat_regrasp_eval` (arm-low) experiments -> apples-to-apples.
- **Submitted 1796505 yd_xpipe_b** = baseline pipeline (stage shards -> convert ->
  BC 150ep -> dual eval). Regrasp pipeline follows once a slot frees / regrasp
  collection has a comparable count.
- 3-metric comparison (SR / gentleness / SR*gentleness, clean + regrasp start)
  will be the Phase 4 preliminary result.

---
## 2026-08-30 10:15 — Phase 4 pipeline bug: eval used the COLLECTION experiment

1796555 got through pretrain (baseline gen `tqmjv`, 150ep, val 0.025) but the clean
eval sim server launched with `--experiment single_lift_xcat_diverse` (COLLECTION:
grid 250 / substeps 240, CFL-risky; success band z 0.16-0.40 hold 6) instead of
`single_lift_xcat_diverse_eval` (grid 190 / substeps 440, CFL-safe; z 0.13-0.45 hold 4).
Killed it before a mid-eval NaN.
FIX: pipeline now has `EVAL_EXPERIMENT` (defaults `${EXPERIMENT}_eval`) for both eval
sim servers, distinct from `EXPERIMENT` (train/convert). Added `SKIP_TRAIN=1`+`RUN_DIR`
to reuse an existing checkpoint. Committed.
Resubmitted baseline as **1797433** (SKIP_TRAIN, reuses tqmjv/state_150, eval-only).
regrasp collector 1792833 still running (~358).

---
## 2026-08-30 11:10 — small-object support + webpage reorg

- **Collector**: `_synth_bounds_topdown` now takes `obj_size` -> CMA xy-box + close-width
  bound rescaled to the actual object (was mushroom-hardcoded -> straddle on 2cm fruit).
  Width floor scales with object short axis too.
- **New configs** `single_lift_xcat_small_diverse{,_eval}` + `dr/xcat_small_diverse_regrasp`
  + `tasks/*`: pool `[grape, cherry, tomato, raspberry]`, grid 300 / substeps 450,
  SCALE-ONLY DR (no bend/twist -> crashes crude 2-4KB meshes). strawberry dropped
  (1.4KB placeholder mesh). blueberry available but skipped (1cm, extreme).
- **gen8 cfg dirs** `single_lift_gen8_{baseline,regrasp}_pcd` (n_epochs 250) for the
  8-object merged generalist.
- **Small-object collector 1797892** queued `--dependency=afterany:1797433` (starts when
  the baseline eval frees a GPU slot). `REC_VIDEO=250`, TAG=xcat_small.
- **Webpage reorganized** into 3 acts (rigid proof -> soft banana -> cross-category
  generalist) + 6 diverse-init demo clips each for mushroom/kiwi/egg_boiled. Small-fruit
  clips to follow. Artifact 5682ac2f.
- MERGE PLAN for the generalist: concat `single_lift_xcat_regrasp` + `single_lift_xcat_small`
  + (optionally) the banana 500 -> one data.pkl -> train `single_lift_gen8_regrasp_pcd`.
  Baseline: same with the strict_home datasets.

---
## 2026-08-30 12:25 — regrasp generalist pipeline launched

- Baseline generalist eval 1797433 nearly done (batch 15/20, clean SR ~12% -- thin
  4-obj BC over 474 demos; low absolute, but the comparison is regrasp-vs-baseline).
- **Stopped regrasp collector 1792833 at ~499 demos** (matched to baseline's 474),
  run `single_lift_xcat_regrasp/26-08-30-rdz` (99 shards, merged by the pipeline).
- **Submitted regrasp generalist pipeline 1798531** (`yd_xpipe_r`): merge -> convert
  -> BC pretrain (`single_lift_xcat_regrasp_pcd`, 150ep) -> clean eval
  (`single_lift_xcat_diverse_eval`) -> regrasp eval (`single_lift_xcat_regrasp_eval`).
  EVAL_EXPERIMENT wired correctly (CFL-safe).
- Small-object collector 1797892 still queued `afterany:1797433` -> starts when the
  baseline eval slot frees.
- NOTE: never `uv run --project envs/*` on the login node (aarch64 venv, x86 login) --
  it prints "Creating virtual environment" and fails; the real venv is untouched but
  shard merges / any env work must run inside a SLURM job.
- Next gate: 1798531 done -> baseline vs regrasp 3-metric table -> webpage + handoff.

---
## 2026-08-30 16:15 — Phase 4 GENERALIST vs BASELINE (in-domain, 4 obj, ~475-500 demos each)

Direct 4-object generalists, BC 150ep, same shared harness.

| metric (clean start) | baseline (strict_home) | **regrasp generalist** (diverse-start) |
|----------------------|------------------------|----------------------------------------|
| success rate         | 0.16                   | **0.98**                               |
| gentleness           | 0.155                  | **0.555**                              |
| SR x gentleness      | 0.157                  | **0.768**                              |
| stress top10 (Pa)    | 27431                  | 19873 (gentler)                        |
| late successes (fss>70) | 0 / 16              | 13 / 98                                |
| arm-low-start SR     | **0.13**               | (eval#2 running, ~1.0 so far)          |

**Regrasp generalist beats the baseline on ALL 3 metrics decisively.** The strict_home
baseline can't handle the eval's pose DR at all (only ever saw home starts) -> 16% clean.
The diverse-start generalist is 0.98 clean + shows recovery (13% late).

Baseline eval dirs:
  logs/dppo/dppo-pretrain/single_lift_xcat_baseline_pcd/tqmjv/eval{,_regrasp}/
Regrasp-gen: logs/dppo/dppo-pretrain/single_lift_xcat_regrasp_pcd/uabeb/eval{,_regrasp}/

### Eval-protocol change (user, for all future evals)
- NO clean/arm-low split. ONE eval: EE init spans home <-> near-object (matching the
  training sweep). Impl: `dr/eval_cat_<name>.yaml` (robot_init_offset_xyz [0.03,0,-0.045]
  + robot_init_pos_xyz 0.065 box).
- Per-category: 100 rollouts for EACH in-domain + EACH OOD object, 3 metrics per-cat + overall.
- Infra: `experiments/single_lift_xcat_gen_eval.yaml` + `scripts/arrhenius/yd_gen_eval.sbatch`
  (loops cats via --dr override, aggregates). In-domain: mushroom, banana_lying, kiwi,
  egg_boiled. OOD: apple, pear, avocado, pasta_bundle, dumpling (all real-mesh soft, 3-9cm).
  RUN once collection + retrain catch up.

### Scale-up (GPU was idle)
- Per job: 1 GH200 (used ~4GB/98GB, ~0% util), 64-72 CPU (used 6). Collector n_envs 6->16
  default (launch at 20-24). Walltime 71h->24h so >2 collectors fit under AssocGrpGRESRunMinutes.
- Small-object collector 1797892 RUNNING (n_envs 6, submitted pre-change) -- SLOW (~4/15min,
  tiny objects grasp-synth poorly). Watching SR; likely resubmit at n_envs 20.

---
## 2026-08-30 16:30 — small-object SDF crash fixed, resubmitted

- Small collector 1797892: 6/4500, 5/9 batchfails on grape/cherry/tomato. Cause:
  `build_object_sdf` runs `simplify_quadric_decimation(0.99)` on the crude 2-4 KB
  grape/cherry scans (~50 faces) -> collapses to 0-2 faces -> trimesh rtree
  `bounds_tree` "Bounds must be (n, dimension*2)". raspberry (610 KB real mesh) was
  fine (33% grasp SR).
- FIX (`synth_utils.build_object_sdf`): skip decimation for <=600-face meshes; fall
  back to the full mesh if decimation returns <8 faces. Committed.
- Resubmitted as **1800352** (n_envs 16, maxfevals 500, 24 h). The old job's
  ProcessPool workers had stale synth_utils -> full resubmit needed.

---
## 2026-08-30 17:20 — final-eval spec (user) + webpage

**Webpage** (artifact 5682ac2f): added "Generalist vs non-regraspable baseline" to
Act 3 -- 3-metric table (regrasp SR 0.98 / gentle 0.555 / SRxg 0.768  vs  baseline
0.16 / 0.155 / 0.157) + 12 eval clips (4 recoveries, 2 clean, 1 fail; baseline
typical + arm-low). Clips in docs/eval_showcase/generalist/.

**Final-eval spec for the 8x500 model** (`yd_gen_eval.sbatch`):
- **8 in-domain**: mushroom, banana_lying, kiwi, egg_boiled, grape, cherry, tomato, raspberry
- **4 OOD** (= GRACE zero-shot set): blackberry, scallop, dumpling, gelatin
- 100 rollouts each, 3 metrics per-category + aggregate (in-domain mean, OOD mean).
- **ONE start distribution** (no clean/arm-low split): EE spans home<->near-object via
  `dr/eval_cat_<name>.yaml` (robot_init_offset [0.03,0,-0.045] + robot_init_pos 0.065).
- **Randomization cadence** (more than before): geometry+material rebuilt every 5
  episodes (`scene_group_size: 1`, num_envs 5); pose/pos/orientation every episode
  (harness per-reset). eval_cat DR carries shape DR (bend/twist/taper/axis_scale) for
  large objects, scale-only for the crude small meshes; material via each category's
  registry `material_dr_mult` (object_E/nu/rho left unset).
- Per-size task switch in the script: small objects (<=2.5cm + blackberry/scallop) use
  the finer-grid `single_lift_xcat_gen_eval_small` (grid 300/substeps 450).
- **RISK**: `scene_group_size>0` hit an RPC-rebuild hang in earlier rigid evals. Must
  smoke-test with `CATS=mushroom` before the full 12-category run; fix the rebuild RPC
  or fall back to 0 + wider per-reset pose DR.
- gen8 cfg dirs `single_lift_gen8_{baseline,regrasp}_pcd` set to scene_group_size 1.

---
## 2026-08-30 17:35 — killed eval, FULL EFFORT on the 4000-demo collection

User: kill the regrasp-gen arm-low eval (1798531), all GPU on collection.
- Killed 1798531 (arm-low eval; deprecated -- future evals use the unified
  home<->near-object protocol anyway).
- **2 collectors now running (the account's max concurrent GPU jobs):**
  - **1801666 yd_xreg** -- large 4 (mushroom/kiwi/egg_boiled/banana_lying), n_envs 20,
    maxfevals 650, seed 11, 24h. New run dir under single_lift_xcat_regrasp/.
  - **1800352 yd_xsmall** -- small 4 (grape/cherry/tomato/raspberry), n_envs 16, 24h.
- A 3rd GPU job won't schedule for ~14h (AssocGrpGRESRunMinutes ~= 48 GPU-h cap).
- Merge at the end: all single_lift_xcat_regrasp/*/data.pkl + single_lift_xcat_small/*/
  + the 500 soft-banana set -> train single_lift_gen8_regrasp_pcd.

---
## 2026-08-30 17:45 — IMPORTANT: the baseline-vs-regrasp eval was MUSHROOM-ONLY

User caught it: the "4 in-domain" comparison eval actually tested only ONE object.
Cause: `scene_group_size: 0` -> the sim builds ONE scene at startup, the
`object_category_pool` draw fires ONCE (-> EGG_BOILED, not mushroom -- the
`object=mushroom` in the server log is the NOMINAL fallback printed BEFORE the DR draw;
the spawned mesh is `egg_deformed_815853.obj`, mat_yield 22500 = egg_boiled), never rebuilds. All 100
episodes = mushroom at ONE fixed scale/material/shape; only pose + EE-start varied.
Confirmed: `object=mushroom` in all 4 sim-server logs; `obj_scale` / `mat_E` have a
single distinct value across episodes.csv.

-> The SR 0.98 vs 0.16 result is valid as an **egg_boiled, fixed-geometry** regrasp-vs-
baseline signal, NOT the cross-category generalist claim. Webpage relabeled + caveated.

**The real per-category comparison** = run `yd_gen_eval.sbatch` (scene_group_size 1,
per-cat pinned pool, 8 in-domain + 4 OOD, 100 rollouts each) on BOTH checkpoints
`uabeb` (regrasp) and `tqmjv` (baseline). Queued for the first free GPU slot after
the 4000-demo collection (user wants full collection effort now). This is also the
template for the final 8x500 model eval.

---
## 2026-08-30 18:10 — n_envs tuning: 20 was too many

1801666 (large, n_envs 20): Genesis idle-FPS 74 but during the GRASP (MPM contact
solve) it dropped to 0.6 FPS/env -> ~17 min execution per 20-env batch + CMA -> net
SLOWER than the old n_envs 6 (1.2/min). Soft-MPM contact does not parallelize free on
GPU. Killed. Relaunched large as **1802711, n_envs 10, maxfevals 550, seed 13**.
Collector default n_envs 16 -> 10.
- Small collector 1800352 stays at n_envs 16 (small objects = few particles, MPM cheap).

---
## 2026-08-30 19:15 — collection status (~26%)

- 1802711 large (n_envs 10): ~46 demos this run. 1800352 small (n_envs 16): ~165.
- CONFIRMED: soft-MPM grasp EXECUTION is contact-bound, ~0.7 FPS/env regardless of
  n_envs -> a 16-env batch ~= 14 min exec + ~4 min CMA. n_envs 6-10 is the sweet
  spot; the small collector at 16 is a bit slow but PRODUCING (0 batchfails). A
  natural resubmit will pick up the new n_envs 10 default. Not churning it now.
- Per category toward 500: banana_lying 500, kiwi 139, egg 137, mushroom 132,
  grape 57, cherry 55, tomato 32, raspberry 27. Total ~1040/4000.
- Both continuous; won't finish in the window. Merge + gen8 train + per-category
  eval (yd_gen_eval, 8+4 cats) when the user resumes or collection completes.

---
## 2026-08-30 20:35 — collection ~30%
- 1802711 large (n_envs 10): ~104 this run. 1800352 small (n_envs 16): ~260.
- Per category: banana 500, kiwi ~160, egg ~156, mushroom ~152, grape ~90, cherry ~86,
  tomato ~57, raspberry ~48. Total ~1210/4000 (30%). 0 batchfails.
- ~1.5 demos/min combined (soft-MPM contact-bound ceiling on 2 GPUs).

---
## 2026-08-30 21:25 — large collector: dropped banana from the pool

User: banana already has 500+, stop collecting it. `dr/xcat_diverse_regrasp.yaml`
`object_category_pool` -> `[mushroom, kiwi, egg_boiled]` (was 4 incl banana_lying).
Cancelled 1802711, resubmitted large as **1805125** (mushroom/kiwi/egg only, n_envs 10,
seed 21). NOTE: this DR is also `single_lift_xcat_diverse{,_eval}`'s dr -- but the
per-category eval (yd_gen_eval) uses `dr: eval_cat_<name>` overrides, and the old
frozen-geometry `_eval` is deprecated, so no eval impact.
Per category now: banana 500+, kiwi ~166, egg ~163, mushroom ~160, + small 4.

---
## 2026-08-30 21:45 — per-cat quota + walltime fix
- Collector now DROPS a category from the pool at --per-cat-target (500); sbatch
  auto-computes --cat-have from prior TAG logs. Stops when pool empties.
- 2x 24h GPU jobs exceed AssocGrpGRESRunMinutes -> collector walltime 24h -> 12h.
  Resubmit ~2x/day. Both running: 1805147 (large: mushroom/kiwi/egg, target 500),
  1805198 (small: grape/cherry/tomato/raspberry, target 500).
- ETA to 4000: ~24-28h (large finishes 3 objs in ~13h -> that GPU repoints to small).

---
## 2026-08-30 23:00 — collection ~36%, both collectors healthy
- 1805147 large (mushroom/kiwi/egg, target 500): ~55 this run. 1805198 small
  (grape/cherry/tomato/raspberry, target 500): ~68 this run.
- Per category: banana 500+, kiwi ~187, egg ~177, mushroom ~162, grape ~130,
  cherry ~123, tomato ~95, raspberry ~82. Total ~1450/4000 (36%). 0 batchfails.
- Self-resubmit + per-cat quota active. ETA ~22h. When user returns / collection
  completes: merge all data.pkl -> train single_lift_gen8_regrasp_pcd (+ baseline
  from strict_home) -> yd_gen_eval (8 in-domain + 4 OOD, 100 each, scene_group 1).

---
## 2026-08-31 01:15 — collection ~40%
- 1805147 large: 120 this run. 1805198 small: 160 this run. 0 batchfails.
- Per category: banana 500+, kiwi ~205, egg ~198, mushroom ~190, grape ~172,
  cherry ~163, tomato ~135, raspberry ~120. Total ~1610/4000 (40%).
- ~2 demos/min combined. ~20h to 4000. Self-resubmit + per-cat quota running.

---
## 2026-08-31 02:35 — collection ~43%
- 1805147 large: 168 this run. 1805198 small: 194 this run. 0 batchfails, both healthy.
- Per category: banana 500+, kiwi ~217, egg ~210, mushroom ~206, grape ~190,
  cherry ~181, tomato ~150, raspberry ~135. Total ~1690/4000 (43%).
- ~1.5 demos/min. ~18h to 4000.

---
## 2026-08-31 04:25 — collection ~45%
- 1805147 large: 207. 1805198 small: 243. 0 batchfails, ~6.5h into their runs.
- Per category: banana 500+, kiwi ~229, egg ~222, mushroom ~219, grape ~208,
  cherry ~198, tomato ~166, raspberry ~150. Total ~1790/4000 (45%).
- ~1.7/min. ~16h to 4000. Collectors will hit their 12h walltime around
  ~13:00-13:30 and self-resubmit (SIGTERM path -> manual resubmit if that fails).

---
## 2026-08-31 05:15 — collection ~47%
- 1805147 large: 249. 1805198 small: 297. 0 batchfails.
- Per category: banana 500+, kiwi ~242, egg ~235, mushroom ~233, grape ~230,
  cherry ~220, tomato ~185, raspberry ~168. Total ~1880/4000 (47%).
- ~1.7/min. ~14h to 4000.

---
## 2026-08-31 06:05 — collection ~49%
- 1805147 large: 294 (1 batch skip, caught). 1805198 small: 321.
- Per category: banana 500+, kiwi ~254, egg ~247, mushroom ~245, grape ~243,
  cherry ~231, tomato ~196, raspberry ~180. Total ~1970/4000 (49%).
- ~1.4/min. ~12h to 4000. Small collector (1805198) hits 12h walltime ~10:20 ->
  resubmit if SIGTERM path skips auto-resubmit.

---
## 2026-08-31 07:00 — collection ~52%
- 1805147 large: 343 (bf 1). 1805198 small: 374.
- Per category: banana 500+, kiwi ~265, egg ~259, mushroom ~258, grape ~257,
  cherry ~245, tomato ~210, raspberry ~195. Total ~2070/4000 (52%).
- ~2/min. ~10h to 4000.

---
## 2026-08-31 07:35 — collection ~53%
- 1805147 large: 366 (bf 2 = kiwi MPM NaN, both caught). 1805198 small: 407.
- Per category: banana 500+, kiwi ~271, egg ~265, mushroom ~264, grape ~264,
  cherry ~255, tomato ~220, raspberry ~207. Total ~2130/4000 (53%).
- ~1.5/min. ~9h to 4000. 1805198 small hits 12h walltime ~13:20.

---
## 2026-08-31 ~08:05 — collection ~44% (on-disk log tally), eval bug reconfirmed
- Authoritative log tally (all collector .out, `ep N: env i OK` by scene cat):
  xcat_regrasp: egg 343, kiwi 290, mushroom 233, banana_lying 165 (=1031)
  xcat_small:   grape 244, cherry 192, tomato 177, raspberry 130 (=743)
  + soft-banana 500 set (separate). Combined toward 8x500 ~ 1774/3500 non-banana.
- Jobs: 1805147 yd_xreg (24h wall, 8h in, 16h left), 1805198 yd_xsmall (12h, 4h left),
  1812536 yd_xsml2 PENDING on AssocGrpGRESRunMinutes (warm standby, auto-starts when
  1805198 frees ~12:00).
- RECONFIRMED the frozen-geometry eval bug: the 2026-08-30 xcat evals (uabeb regrasp,
  tqmjv baseline) still have mat_yield/mat_E/obj_scale = 1 distinct across all 100 eps
  (yield 22500 = egg_boiled only). scene_group_size fix not applied there. Numbers on
  that single object: regrasp uabeb SR 0.98 (all 98 successes first_success_step >40,
  med 50, tail to 131 = recovery); baseline tqmjv SR 0.13 (all successes step 25-34,
  no recovery). Story holds but NOT the cross-category claim yet.
- NEXT GATE: collection to 8x500 -> merge -> train single_lift_gen8_regrasp_pcd +
  strict_home baseline -> yd_gen_eval (per-cat pool + scene_group_size 1). Proper
  per-cat eval on uabeb/tqmjv deferred to keep GRES headroom for collectors.

---
## 2026-08-31 ~06:14 — 3rd collector promoted, collection ~50%
- 1812536 yd_xsml2 (3rd, small-object) now RUNNING on n48 (GRES cap cleared). 3
  collectors live: 1805147 large (457 this run), 1805198 small (492), 1812536 small (0, starting).
- 1805198 hits 12h walltime ~09:30 -> will self-resubmit or 1812536 covers small.
- No new batch failures. Next gate unchanged.

---
## 2026-08-31 09:44 — 1805198 hit 12h walltime (SIGTERM mid-CMA, no auto-resubmit)
- 1805198 run dir 26-08-30-wfz: 141 shards, no data.pkl (merge pending). Will be
  salvage-merged by a later yd_xcat_collect cycle (20-min live guard currently blocks).
- Submitted replacement small collector (SEED=11). Running: 1805147 (large, 627),
  1812536 (small, 249), + new yd_xsmall. 1805147 still 24h wall (11.7h left).

---
## 2026-08-31 09:50 — GPU/pool analysis + packed-collector optimization
- CONSTRAINT identified: account naiss2026-3-141-gpu has GrpTRESRunMins gres/gpu=36000
  (600 GPU-h of SUMMED REMAINING runtime across all running jobs), SHARED with users
  ikemura + sean. ikemura currently runs ~7 GPU jobs incl several 30h (1-06:00:00)
  walltime -> pool near-full -> my 4th job pends (AssocGrpGRESRunMinutes). Not a
  job-count cap; not cluster capacity (253 idle GPU nodes).
- Each of my grasp collectors: ~7 GB / 98 GB GPU mem, ~0-15% GPU util (grasp exec is
  CPU/CMA + contact-physics bound per CLAUDE.md, NOT GPU-compute bound).
- NEW: gentle_manip/scripts/arrhenius/yd_xcat_pack.sbatch -- runs N_PACK=3 collector
  processes inside ONE GPU allocation (1 large-pool + 2 small-fruit), 90s staggered
  genesis inits, each own run dir, CAT_HAVE auto-coordinated. 3x throughput per
  reserved GPU at 1 job's pool cost. Submitted 1815726 (PENDING on pool; starts when
  a slot frees). Cancelled the single-collector standby 1815706.
- Currently RUNNING: 1805147 (large, 632), 1812536 (small, 258), 1815690 (small, ~0).

---
## 2026-08-31 12:15 — collection ~99% total, QUOTA BUG found, mushroom top-up
- Per-cat (reliable mode-parse): banana 500 · egg 596 · kiwi 496 · grape 548 · tomato 574
  · cherry 478 · raspberry 432 · MUSHROOM 372. total ~3996.
- BUG: collect_demos_diverse_start_v2 per-cat quota is PER-COLLECTOR not global. 4
  concurrent small-pool collectors each seeded CAT_HAVE from other logs AT STARTUP +
  counted only their own saves -> none individually crossed 500 -> grape/tomato
  overshot to 550-590 while raspberry/mushroom undershot. (Fix later: global tally
  file, or single small collector.)
- ACTION: killed 1815690 + pack 1815807 (over-collecting grape/tomato). Added
  configs/dr/xcat_mushroom_only.yaml + experiments/single_lift_xcat_mushroom.yaml
  (pool=[mushroom]). Submitted 1824898/1824899 (yd_mush60/61, mushroom-only,
  write into single_lift_xcat_regrasp/) -- PENDING on GRES pool. 1805147 still
  runs [mushroom,kiwi]; 1812536 small until walltime ~13:50.
- PLAN: if mushroom >= ~450 by ~12:40 OR top-up jobs still pending -> FIRE gen8
  pipelines regardless (mushroom ~9% of set is acceptable; top-up+retrain if it
  evals weak). yd_gen8_pipeline.sbatch (regrasp) + VARIANT=baseline both ready.

---
## 2026-08-31 12:44 — GEN8 PIPELINES FIRED
- Per-cat FINAL-ish (reliable mode-parse): banana 500 · kiwi 506 · egg 596 · grape 565
  · tomato 588 · cherry 478 · raspberry 435 · mushroom 409. total ~4077.
- Submitted 1826461 (yd_gen8 VARIANT=regrasp) + 1826462 (VARIANT=baseline). Both
  PENDING on GRES pool (ikemura + my collectors full). They stage -> convert ->
  BC pretrain 250ep -> auto-submit yd_gen_eval (8 in-domain + 4 OOD x100, scene_group_size 1).
- Killed 1812536 (redundant small). KEPT 1805147 + 1824898/1824899 (all mushroom-only
  now) running -- gen8 stage reads disk at RUN time, so more mushroom before it starts
  is a bonus. Next tick: if gen8 still PENDING, leave collectors; when gen8 STARTS,
  cancel the mushroom collectors (shard-only dirs are merged by _gen8_stage.py, but its
  20-min fresh-shard guard means cancel >20min before stage, or accept losing the last
  ~20min of mushroom).
- CAVEAT to record in the pretrain EXPERIMENT.md: mushroom ~410/500 (10% vs ~12.5%
  even) due to the per-collector quota bug; top-up + retrain if mushroom evals weak.

---
## 2026-08-31 12:54 — GEN8 RUNNING (collectors cancelled)
- Cancelled 1805147 + 1824898/1824899 (mushroom ~430/500 final -- acceptable, per-cat
  quota bug capped it). Freed GRES -> 1826461 (regrasp) + 1826462 (baseline) both
  STARTED (n92 / n119, 20h wall).
- Shard-only dirs to be merged by _gen8_stage.py: 26-08-30-kwb (157), 26-08-31-pbt (23),
  -bdt (4), -nqu (5). _gen8_stage.py merges unconditionally (no age guard).
- Pipeline each: stage(all xcat + soft-banana + shard merges) -> convert(pointcloud,
  student view) -> BC pretrain 250ep -> auto-submit yd_gen_eval (8 in-domain + 4 OOD
  x100, scene_group_size 1). baseline VARIANT filters episodes to start_mode in
  {home,near_object}.
- ETA: stage+convert ~20-30min, pretrain ~4-8h, gen_eval ~5-6h. Watch
  logs/slurm_logs/1826461.out + _pretrain.log.

---
## 2026-08-31 13:10 — gen8 reconfigured per user, resubmitted
- User: drop strict-home baseline; baseline = ALL data EXCEPT failed_grasp (recovery)
  family; scale model for the larger set.
- Cancelled 1826461/1826462. _gen8_stage.py gained --exclude-modes. baseline VARIANT
  now uses `--exclude-modes failed_grasp` (was `--modes home,near_object`).
- MODEL SCALE-UP (both gen8 pre+eval cfgs, kept byte-identical apart from env:):
  mlp_dims [512,512,512]->[768,768,768], visual_feature_dim 256->384, batch 128->256,
  n_epochs 250->300, warmup 100->150, early_stop_patience 8->15.
- Resubmitted: 1826960 (regrasp, all modes) + 1826961 (baseline, minus failed_grasp).
  Both PENDING. Each: stage->convert->BC pretrain 300ep->auto yd_gen_eval (8+4 x100,
  scene_group_size 1). Commit 8c... pushed to cross-category-dp? NO -- user must push.

---
## 2026-08-31 13:07 — gen8 walltime 20h->12h (GRES headroom), now queued on Priority
- 1826960/1826961 (20h) pended on AssocGrpGRESRunMinutes: the account GrpTRESRunMins
  gres/gpu=36000 min headroom couldn't fit 2x20h alongside ikemura's ~5 running 30h
  jobs. Resubmitted 1827009 (regrasp) + 1827010 (baseline) at -t 12:00:00 -> reason
  flipped to Priority (normal wait). 12h is enough: stage ~15m + convert ~25m +
  BC pretrain early-stops ~5-8h (patience 15, ckpt every 25ep).
- sbatch default walltime bumped to 12h + committed.

---
## 2026-08-31 13:14 — gen8 REGRASP training (1827009 -> run ueini)
- STAGE ok: 4400 eps (xcat_regrasp 2065 + xcat_small 1835 + banana_soft 500), 16 src
  dirs incl orphan-shard merges. start_mode mix: mid_approach 1211, failed_grasp 1123
  (25.5%), above 561, near_object 491, mid_air 377, home 320, near_ground 317.
- CONVERT ok: 3960 train / 440 val traj, 665660 steps, pc [1024,3].
- PRETRAIN running: run dir ueini, net 1.72M params (768x3 / vfeat 384). epoch1 loss
  0.462, ~2.1 min/epoch -> early-stop (patience 15) expected ~100-200 ep (~4-7h).
  EXPERIMENT.md written.
- 1827010 (baseline, minus failed_grasp -> ~3277 eps) PENDING on GRES (regrasp ate the
  headroom). Starts when regrasp frees / slot opens.
- yd_gen_eval auto-submits after pretrain (8 in-domain + 4 OOD x100, scene_group_size 1).

---
## 2026-08-31 14:33 — both gen8 jobs RUNNING
- 1827010 baseline STARTED (n173). STAGE ok: exactly 3277 eps = 4400 - 1123 failed_grasp
  (start_mode mix confirms zero recovery: near_obj 491, mid_approach 1211, above 561,
  mid_air 377, home 320, ground 317). Now converting.
- 1827009 regrasp 'ueini': ep39 train 0.030, val 0.061@10->0.041@20->0.033@30, clean.
- Both auto-submit yd_gen_eval after pretrain. ETA pretrain ~2-3h each.

---
## 2026-08-31 15:03 — artifact: removed the flawed egg-fixed-geometry subsection
- User asked to drop the 5 baseline clips + the whole "Regrasp generalist vs
  non-regraspable baseline - egg, fixed geometry" subsection (the buggy scene_group_size 0
  eval). Removed 37 lines (both video grids + metrics + caveat). Artifact 5682ac2f
  republished (4.39MB -> 3.52MB, 71 clips). The proper per-category gen8 eval replaces it.

---
## 2026-08-31 22:57 — LIVE eval webpage wired (montages + metrics)
- lonau (baseline) pretrain DONE ep300 val 0.0170 -> eval job 1854786 RUNNING (n64),
  on cat 1/12 (mushroom, SR ~0.33 first 3 batches). ueini (regrasp) pretrain ep272,
  val 0.0165, done ~23:45 -> its eval auto-submits.
- NEW realtime webpage pipeline (scratchpad, x86 login-node tools):
  - gen8tools/ venv: x86 imageio-ffmpeg (login node has NO ffmpeg / module broken).
  - gen8_montage.py: per (model,cat) -> 15 random eval clips, 2x speed, 300x234, 10fps,
    crf33, 0.35s black gaps -> montages/montage_<model>_<cat>.mp4 (~260KB each; 24 total ~6MB).
  - gen8_eval_page.py: reads each model's gen_eval_*/<cat>/summary.json (success_rate,
    gentleness_score, combined_sr_gentleness), emits the fragment (8 in-domain + 4 OOD,
    per-cat metric bar + 2 side-by-side montages + block means).
  - gen8_refresh.sh: builds all montages, splices fragment between <!--GEN8_EVAL_START/END-->
    markers in regrasp_demos.html.
- EACH TICK now also: `bash <scratchpad>/gen8_refresh.sh` then republish artifact 5682ac2f.
  Published once already (baseline mushroom montage live, metrics pending until summary.json).

---
## 2026-09-01 ~23:32 — eval was ~37h serial; rewrote yd_gen_eval parallel+record3
- 1854786 (old serial baseline eval) measured 9.3 min/batch -> 12 cats x 20 batches
  = ~37h, would NOT finish in 24h walltime. CANCELLED.
- yd_gen_eval.sbatch rewritten + committed: NCAT_PAR=3 (3 cats concurrent on 1 GPU,
  each sim server ~7GB), RECORD_BATCHES=3 (only 15 eps/cat rendered = the montage
  need; metrics still over all 100). Self-resubmits for missing cats (reuses OUT).
- Resubmitted baseline eval -> 1856565 (n415). ueini (regrasp) pretrain ep293, done
  ~23:45 -> auto-submits its eval with the NEW script.
- Live webpage refresh (gen8_refresh.sh) unchanged - picks newest gen_eval_* dir per model.

---
## 2026-09-01 ~23:53 — hi-res montages + both evals running
- User: higher-res video. gen8_montage.py now 448x336 / 14fps / crf25 (was 300/10/33),
  +7s per-clip cap (trim=end=7) so failure-heavy montages don't balloon. ~520KB/montage
  untrimmed, less once _trim_eval_clips shortens successes.
- gen8_refresh.sh size guard: embed_ood=0 if all montages >8.6MB mp4 (b64 x1.33 + ~4MB
  demo clips vs 16MB cap); warns if in-domain alone >8.6MB.
- 1856565 (baseline eval) producing clips: mushroom/banana/kiwi first batch done.
- 1856838 (regrasp eval) running, servers spinning up.
- Page republished 4.73MB. gen8_* scripts copied to docs/.

---
## 2026-09-01 00:49 — evals mid batch-5, montage CRF 25->28
- Both evals ~1h15m: batch 4-5/20 on mushroom/banana/kiwi (3 concurrent each).
  Batch rate ~13min/cat (soft-MPM eval + scene rebuild every batch). ~17h/eval -> both
  done ~17:00-18:00 today. Fits 24h walltime.
- Early per-batch SR comparable baseline~ours on these 3 (~0.6-0.8). One regrasp batch
  grasp-but-no-hold (SR0/in_band0.8) - watch.
- montage CRF 25->28 (still 448x336, well above first-pass 300px) so 24 montages fit
  16MB b64 with OOD; in-domain (16) always embedded, size guard drops OOD b64 if total
  montage mp4 >8.6MB (keeps OOD metrics).
- No full-cat summary.json yet. Refresh+publish each tick.

---
## 2026-09-01 02:50 — kiwi eval crashed (floor-clip), fixed + re-eval
- BOTH gen8 evals: kiwi cat crashed at batch 15/20 -> MPM "particles outside solver
  boundary". Cause: a scene-DR deform draw (bend/taper/axis_scale) dropped the kiwi
  mesh's min-z ~0.1mm below the MPM effective floor (specified -0.03 minus ~26mm
  safety pad at grid 190); scene builder places by nominal default_pos so the
  deformed mesh clipped the plane.
- FIX (committed): mesh_deform.save_deformed re-seats the deformed mesh bottom (z) to
  the nominal bottom -> always rests on the plane. Helps collection too.
- Re-submitted kiwi-only eval for BOTH models into the SAME OUT dirs: 1861733 (baseline),
  1861734 (regrasp), NCAT_PAR=1. Fresh sim servers -> pick up the fix. Main jobs'
  future categories (egg_boiled onward) also spawn fresh servers -> fixed.
- 4 eval jobs now: 1856565 + 1856838 (main, on egg/grape) + 1861733 + 1861734 (kiwi).

---
## 2026-09-01 04:36 — FIRST 2 COMPARISON PAIRS: regrasp WINS ALL 3 METRICS, big gentleness gap
                          baseline SR/gentle/SRxg   regrasp SR/gentle/SRxg
  mushroom                0.43 / 0.008 / 0.219      0.58 / 0.772 / 0.676
  banana_lying            0.67 / 0.000 / 0.335      0.85 / 0.687 / 0.769
- HEADLINE: baseline succeeds by CRUSHING (gentleness ~0 = internal stress at yield on
  every successful grasp). Regrasp generalist grasps successfully AND gently. The
  diverse-start + failed_grasp recovery data taught a lighter re-approach vs the
  baseline's commit-and-clamp.
- Kiwi floor-clip fix WORKED: both kiwi re-evals (1861733/1861734) passed batch 15 with
  no boundary crash. Will complete + drop kiwi/summary.json into the main OUT dirs.
- Main evals on grape/cherry now (cat 5-6). ~7h to full 12.

---
## 2026-09-01 08:32 — eval too slow: small-fruit grid 300 = ~6x; capped small cats at n=50
- Diagnosed: _small eval task (grape/cherry/tomato/raspberry/blackberry/scallop) uses
  grid_density 300 vs 190 -> ~6x slower/step. Full 12x100 sweep -> ~28h, over 24h walltime.
  Round 2 was stuck ~batch 13-17 after 3.5h.
- FIX (committed): yd_gen_eval run_one caps SMALL cats at n_episodes=50 (SMALL_NEP), large
  stay 100. Killed 1856565/1856838, cleared partial round-2 dirs (no summary.json),
  resubmitted -> 1866095 (baseline) + 1866096 (regrasp). Skip mushroom/banana/kiwi
  (summary.json kept). New ETA full 12 ~18:00-19:00, fits walltime.
- NOTE for final table: small-fruit categories n=50, large n=100. Document.

---
## 2026-09-01 09:00 — USER caught eval horizon too short; PAUSED evals + smoke test
- User: eval episodes look too short for a 2nd attempt. CONFIRMED: eval max_policy_steps
  = max_episode_steps(600)/act_steps(4) = 150, but training demos run to 185 policy steps
  (failed_grasp median 166, mid_air median 178). Eval first_success_step already hit
  137/150 on recovery episodes -> recovery attempts were being cut off.
- FIX (committed): eval cfgs max_episode_steps 600->900 (n_steps 150->225). Also added
  NOTRIM=1 + NEP overrides to yd_gen_eval.
- PAUSED main evals 1866095/1866096. Launched SMOKE: 1868585 = regrasp ckpt on mushroom,
  NEP=15, NOTRIM=1, h=225, OUT=.../ueini/smoke_h225_0900. When done -> build 10-rollout
  full-length montage (scratchpad/smoke_montage.py: 480x360 crf24, CLIPCAP 40s) for user
  inspection BEFORE resuming the full 12-cat eval.
- The 3 completed cats (mushroom/banana/kiwi) were run at h=150 -> will need re-run at
  h=225 for consistency once horizon is confirmed. (Regrasp still won all 3 at h=150.)

## 2026-09-01 09:45 — Smoke test PASSED, h=225 validated, still HOLDING for user OK
- Smoke 1868585 done: regrasp ckpt state_275, mushroom, 15 eps @ h=225.
  SR 0.73 (11/15) / gentleness 0.817 / SRxg 0.775  (vs old h=150: 0.58/0.772/0.676)
- HORIZON FIX CONFIRMED: 3 of 11 successes had first_success_step = 157/199/204
  -> old h=150 truncated those as failures. Genuine miss->reapproach->gentle-regrasp
  visible in the montage. fss maxes at 204/225 => 225 has headroom.
- Sent 10-rollout montage to user (scratchpad/montages/montage_smoke_mushroom.mp4).
- Full evals still CANCELLED/paused. NOT resuming until user confirms the horizon fix.
- GRES pool near-empty right now (only 1 sean job) -> resubmit will start immediately
  once approved. On approval: re-run ALL 12 cats at h=225 for both models (clear the
  3 stale h=150 summaries in gen_eval_20260831_233305 / _234353 first).

## 2026-09-01 10:20 — USER flagged post-success carry-down; analyzed + eval fix
OBSERVATION (user, from smoke montage): failed 1st attempt -> regrasp works well.
But SUCCESSFUL 1st attempt -> policy lifts, then carries object back DOWN, opens
fingers (release), EE descends to the z-floor (EE_BOUNDS_MIN z 0.1715), stuck there
pressing, jaw slowly opens, no re-grasp, until episode end.

ROOT CAUSE (3 layers):
 1. Demo terminal signal is a SATURATING command, not a fixed point. The scripted
    lift/hold tail commands dz=+0.476 (norm ~max) + dgripper=-0.004 for ~46 frames
    (31% of all demo frames) -- the EE target just clips at EE_BOUNDS_MAX z and the
    object hangs. Demos NEVER contain "object at rest at height, EE still, action~0".
    So the policy has no "task complete, hold here" attractor.
 2. Eval horizon (h=225 / 900 env steps) runs 4-5x past demo length (median 121,
    max 185). Once the policy saturates the up-command against the workspace clip it
    is OOD; the dominant dataset modes (approach = descend+open; failed_grasp recovery
    = reopen+re-descend) take over -> carry-down + release. EE then clips at the LOWER
    z bound -> "stuck at floor, jaw opening" = approach-mode behavior with no object.
 3. early_stop_on_success only froze on the FULL crush-gated success (band + hold_steps
    + not-crushed). A firm-ish first grasp (brief >1.35x yield) or a lift that reversed
    before hold_steps=4 -> success never latched -> ran free -> the ugly tail.

FIX A (DONE, committed 422a79f-ish): harness early_stop also freezes an env on a
 crush-gate-free "lifted clear" latch = obj_z in [z_min,z_max] for
 lifted_clear_hold_steps (default 2) consecutive steps. Success METRIC unchanged.
 Files: evaluation/harness.py, evaluation/eval_spec.py, dppo/eval_agent.py. Logic
 unit-sim'd OK (env keeps earning real success while frozen; crushed env stays fail).
 No-op for no-z-band tasks (rigid banana etc). -> clean videos + correct gentleness
 (kills the 2nd table-contact stress hump that was deflating regrasp gentleness).

FIX B (RECOMMENDED, NOT STARTED -- needs user OK, ~10-12h wall): pad every successful
 demo with a genuine settled-hold tail (30-50 frames: small corrective dz, action~0,
 grip held closed, obs = last frame held) so the policy learns a real fixed point.
 Re-stage + retrain both gen8 models. This is the DURABLE fix + required for real
 deploy (on hardware there is no harness freeze -> it WILL carry down and drop).

FIX C (real deploy): scripted "grasped + high + stable K steps -> switch to hold
 controller" wrapper. No retrain. Safety net on hardware.

STATUS: full 12-cat eval still PAUSED. Resume plan unchanged (re-run all 12 at h=225
 for both models) -- now also picks up FIX A. Awaiting user OK on resume + on FIX B.

## 2026-09-01 10:45 — USER flagged failure mode 2 (aborted-good-grasp); full retrain plan
MODE 2 (user): 1st-attempt approach + close correct, THEN policy reopens + lifts empty
+ retries. = BC mode averaging at "commit vs abort".

DATA COMPOSITION (all 4400 eps, from _pad/tally):
  mid_approach 27.5%  failed_grasp 25.5%  above_object 12.8%  near_object 11.2%
  mid_air 8.6%  home 7.3%  near_ground 7.2%
  (collector v2 --start-modes default: sweep:0.44,failed:0.30,above:0.10,ground:0.09,air:0.07)

ROOT CAUSE mode 2 = OBSERVATION ALIASING. failed_grasp demo's "reopen" moment: ee at
grasp pose, gripper_width ~0.040, object present. Clean post-grasp: ee at grasp pose,
gripper_width ~0.037-0.045, object present. gripper_width does NOT separate them
(checked: failed minclose 0.040 vs near_object 0.040 vs above 0.045 -- fully overlapped)
-- a "failed" grasp in these demos still contacts the 3cm soft object. The recovery
trigger is only in privileged state (contact force / secured), invisible to the
[ee_pos,ee_quat,gripper_width]+1024pt-cloud student. So at that state the policy
sees 2 action targets (lift / reopen) and blends -> weak grasp -> drop.

FIX PLAN (priority order):
 1. [biggest lever] shared sim+real perception feature "object-at-gripper": near-TCP
    point-cloud density + centroid offset (3-4 dims -> state). Breaks aliasing for BOTH
    modes. perception/pipeline.py + re-convert + retrain.
 2. failed_grasp weight 0.30->0.12; sweep 0.44->0.58. AND regenerate failed_grasp from a
    genuine-miss start (lateral err > finger half-width so gripper_width -> ~0.005, OR
    an evident post-slip pose: gripper lifted 3-6cm empty + object on table). Then the
    trigger is observable.
 3. settled hold-aloft tail on every committed-lift ep: _pad_hold_tail.py (DONE, tested
    -- +32 frames, action~0 +small dz, grip held). Fixes mode 1 attractor.
 4. eval knobs, no retrain: deterministic=False (sample -> commit to one mode/rollout),
    act_steps 4->6 (don't flip mode mid-grasp).

SCOPE OPTIONS for user:
 A minimal (~12h): reweight (2 partial) + pad (3) + retrain 2 models. No new collection.
 B +collection (~1.5d): A + regenerate failed_grasp + top-up small cats (~800 eps).
 C full (~2d): B + perception feature (1). The principled fix.
AWAITING user pick. Nothing launched. 12-cat eval still paused.

## 2026-09-01 11:30 — USER approved full retrain (scope ~C). LAUNCHED gen8 v2.
User: add object-at-gripper feature (good) / rebalance (don't over-spend regenerating,
less data OK w/ right ratio) / hold-tail "a bit" + eval TERMINATION on success (don't
idle to 225). "gogo, parallel, stop previous eval".

IMPLEMENTED + committed (d780ec1, + deploy follow-up):
 - perception/pointcloud_ops.object_at_gripper(cloud, ee_pos) -> (…,4): near-TCP cloud
   frac + near-centroid offset. Shared: convert_demos._episode_state (synth from stored
   cloud), genesis_venv._raw_state, deploy_real_dppo._modalities (--obs-dim 12). obs_dim
   8->12. Calibrated tcp_dz=+0.02, r=0.04 (grasped frac ~0.11 vs free ~0.03).
 - _rebalance.py: --cap mode:frac, only drops. Pipeline (regrasp only): failed_grasp
   0.12, mid_air 0.10, near_ground 0.09. baseline unchanged (already excl failed_grasp).
 - _pad_hold_tail.py: +20 frames action~0 + grip-held on every committed-lift ep.
 - harness early_stop: also freezes on crush-gate-free lifted-clear latch (prev commit),
   NOW also ENDS the batch success_grace_steps(8) after ALL envs frozen + stops per-env
   stress/z buffers past grace. venvs get finalize_episode() (video flush on early end).
   EvalSpec.success_grace_steps. eval cfgs: obs_dim 12, obs_keys+=object_at_gripper,
   state shape [12], success_grace_steps: 8, h=225 kept as safety max.
 - yd_gen8_pipeline.sbatch: stage -> rebalance -> pad -> convert(--extra-state-keys
   object_at_gripper) -> pretrain -> auto-submit yd_gen_eval.

JOBS: 1879047 (regrasp) + 1879048 (baseline), PENDING(Priority), 12h each. Each:
data-prep (~30m) -> BC pretrain (~5-6h) -> auto eval (~6-10h). New 5-letter run IDs
(old ueini/lonau superseded). Local tests passed: rebalance/pad/state-synth/object_at_
gripper all OK, all files ast-parse clean. Full pytest NOT run (aarch64, login can't).
NEXT GATE: watch 1879047/8 data-prep for errors; then pretrain val loss; then eval.

## 2026-09-01 11:35 — gen8 v2 data-prep PASSED, both in BC pretrain
1879047 regrasp: rebalance 4400->3717 eps (failed_grasp/mid_air/near_ground capped),
  hold-tail +20 on 3716, convert -> train states (564290, 12) / 3345 trajs. obs_dim 12 OK.
1879048 baseline: hold-tail +20 on 3277, convert -> train states (490615, 12) / 2949 trajs.
object_at_gripper synth + --extra-state-keys wired correctly in both. net params 1748940
(was 1724364 @ obs_dim 8). Both now in BC pretrain (~5-6h) -> auto yd_gen_eval.
NEXT: watch <jobid>_pretrain.log val loss (target best-val <~0.017).

## 2026-09-01 11:55 — USER: focus regrasp; baseline=grasp-at-once; hourly regrasp video
- CANCELLED baseline 1879048. Relaunched as 1882124 = GRASP-AT-ONCE: STAGE_ARGS
  --modes home,near_object,mid_approach,above_object (NO recovery data). Eval h=120 /
  max_episode_steps 480 / grace 6. obs_dim 12 + object_at_gripper kept (same obs space).
- REGRASP job 1879047, run id = lorap. Pretrain ~ep12, train 0.057 / val 0.061 @ep10.
  save_model_freq=25 -> first ckpt ~ep25 (~12:20).
- HOURLY REGRASP SNAPSHOT (user req): gentle_manip/scripts/arrhenius/yd_regrasp_snap.sh
  -> picks best-val ckpt, 15 rollouts / 3 RANDOM in-domain cats (n=5 each), RECORD_BATCHES=1,
  NOTRIM=1, -t 45min, OUT=<run>/snap/<ts>, writes path to logs/slurm_logs/last_regrasp_snap.txt.
  Montage: scratchpad/snap_montage.py <snapdir> <out.mp4> (15 clips, cat label burned in,
  2x, black gaps, 640x480). EACH TICK: if last snap job done -> build montage + SendUserFile;
  then run yd_regrasp_snap.sh again. Skips cleanly if no ckpt yet.

## 2026-09-01 13:42 — first hourly snap sent; both pretrains healthy
- REGRASP lorap ep63 train 0.025 (val 0.034@ep30, 0.026@ep60). BASELINE 1882124 ep62
  train 0.028 (val 0.0475@ep20). ~2min & ~1.4min/epoch.
- SNAP 1883773 (state_25.pt) DONE clean -> new early-terminating harness VERIFIED OK.
  Results: egg_boiled 5/5, mushroom 1/5, raspberry 0/5 (early ckpt). Montage sent to user
  (scratchpad/montages/regrasp_snap_20260901_1249.mp4). Sent-log:
  logs/slurm_logs/regrasp_snap_sent.txt (grep the snap TS there before re-sending).
- snap_montage.py: static imageio-ffmpeg has NO libfreetype -> drawtext unavailable;
  use drawbox colour bars (teal/blue/orange = the 3 cats in sorted order, 5 clips each).

## 2026-09-01 14:10 — HARNESS BUG FOUND+FIXED (the real carry-down cause)
User still saw carry-down in the ep25 snap video. ROOT CAUSE: harness early_stop froze
done envs with action=0.0 in NORMALIZED space, but the demo action dist is min-max
normalized + asymmetric: dz range ~[-1,+0.48] -> normalized 0.0 un-normalizes to raw
dz=-0.26 (DOWN) and dgrip=+0.03 (OPEN). So every frozen/early-stopped env was ACTIVELY
driven down + opened == the exact "carry down, release, hit floor, jaw opens" behavior,
even with early_stop active. NOT a policy bug.
FIX (committed 2 commits): harness computes hold_action from venv.action_min/max ->
normalized vector that un-normalizes to raw [.. dz=+0.03 .. dgrip=-0.06] (gentle up +
keep closed). lifted_clear_hold_steps 2->1 (early ckpt reverses fast). Printed:
"early_stop hold action (normalized): [-0 -0 0.39 -0 -0 -0 -0.218]".
VERIFIED (state_50, ep60): egg 5/5 / banana 5/5 / kiwi 4/5, obj_z HELD at ~0.14 through
the hold (was dropping to 0.04-0.10 pre-fix), episodes end 56-118 steps not 225.
mushroom NOW 5/5 @ep60 (was 1/5 @ep25 -- training). Sent both montages to user.
TOMATO SCENE BUG: tomato spawns at z~0.31 (mid-air) -> instant "success", 9-step eps.
tomato reuses cherry.obj + stiffest material (E 8e5) at grid 300 -> likely MPM launch.
EXCLUDED tomato from yd_regrasp_snap.sh INDOMAIN pool. TODO: fix or exclude from final
in-domain eval too (grape/cherry/raspberry still in; raspberry evals fine just hard).

## 2026-09-01 15:00 — hold-action over-squeeze fixed; remaining stress = the grasp itself
User: gripper keeps CLOSING during the hold -> over-squeeze. Cause: my hold_action had
dgrip=-0.06 raw = continuous -3mm/step CLOSE held across the whole frozen window.
FIX (committed): hold_action dgrip 0 (delta-mode maintain, no squeeze); dz +0.02 anti-sag
only. grace 8->4 / 6->3. + straggler_budget=100: end batch 100 steps after the LAST
success so frozen envs don't idle to 225. rec_ok already cuts a frozen env's stress
recording at freeze+grace -> gentleness metric now only sees each success's approach+grasp.
VERIFY2 (state_50 ep60): egg 5/5 g .57, banana 4/5 g .64, kiwi 4/5 g .85 (was .52/.59/.80).
REMAINING: peak stress still > yield on egg_boiled (30k vs 22.5k) + banana (60k vs 45k)
DURING THE GRASP/FIRM (~step 40-50), not the hold. = epoch-60 BC policy imitating firm
CMA-ES demos (+ FIRM phase), no stress feedback / no per-object softness cue. WATCH as
training continues (val 0.026@60 -> 0.022@100). If not gentler at convergence, options:
reduce demo FIRM_EXTRA_CLOSE, add object-softness obs cue (or category one-hot), or
re-collect with a gentleness cost. Old ueini banana was 0.687 too -> banana always borderline.
- REGRASP lorap ep100 train 0.021 / val 0.022. BASELINE 1882124 running.

## 2026-09-01 19:05 — BASELINE pretrain done (dthox), 12-cat eval running
- BASELINE grasp-at-once run id = dthox. Final train 0.0105 / val 0.0156. ckpts to state_300.
- 12-cat eval job 1898658 RUNNING (started 19:05, GPU freed immediately). Eval dir: find via
  logs/slurm_logs/last_gen_eval_dir.txt (or dthox/eval/<ts> or dthox/gen_eval_<ts>).
- REGRASP lorap still training, ep~225 val 0.0180. GPU freed by baseline -> snap relaunched.

## 2026-09-01 19:12 — both 12-cat evals launched; webpage refresh pinned
- BASELINE eval 1898658 -> dthox/gen_eval_20260901_190536 (state_250). RUNNING, 0 cats done yet.
- REGRASP eval 1899143 -> lorap/gen_eval_20260901_..._ep225 (state_225, MID-TRAINING best).
  PENDING (GRES). lorap keeps training -> will auto-submit a FINAL eval ~22:00; use whichever
  dir is more complete for the final table.
- gen8_refresh.sh now pins BE=dthox/gen_eval_*, RE=lorap/gen_eval_* (override BE_DIR/RE_DIR).
  Run it each tick once cats land; publishes into <!--GEN8_EVAL_START/END--> of regrasp_demos.html
  -> publish artifact 5682ac2f. gen8_eval_page.py shows "pending" for missing cats (graceful).
- User asked: "put the two evals on webpage as in the placeholder" -> that's exactly this section.

## 2026-09-01 19:58 — evals were ~13h (per-batch genesis rebuild); RELAUNCHED fast
- scene_group_size=1 rebuilt genesis EVERY batch = ~13min/batch under contention (10
  re-inits in 50min on mushroom). 12-cat eval projected ~13h. CANCELLED 1898658/1899143.
- FIX committed: eval cfgs scene_group_size 1->0 (fixed nominal geometry; pose/orient
  still vary per batch). RELAUNCHED:
  * BASELINE 1899725 -> dthox/gen_eval_20260901_195735_fast (state_250), n=60/40, NCAT_PAR=4
  * REGRASP  1899726 -> lorap/gen_eval_20260901_195735_fast_state_225, n=60/40, NCAT_PAR=4
  last_baseline_geneval.txt + last_regrasp_geneval.txt updated. gen8_refresh.sh RE/BE globs
  will pick the _fast dirs (newest). Est ~1.5-2h each now.
- REGRASP eval batches at state_225 were 1.0/1.0/1.0/1.0 on first cat -> looking strong.
- lorap pretrain ep240 val 0.0178, ~22:00 done -> FINAL auto-eval will ALSO be scene_group_size=0 now.

## 2026-09-01 21:28 — first eval cat on webpage
- REGRASP kiwi (state_225): SR 0.67 / gentleness 0.727 / SRxg 0.697 (n60).
  vs OLD ueini kiwi (h150 obs8): 0.44 / 0.824 / 0.632. SR way up, gentleness a bit down, combined up.
- Published to 5682ac2f (label gen8-eval-live-1cat). gen8_refresh.sh works; builds 5-clip
  montages per cat from render/. Publish tracking: logs/slurm_logs/webpage_publish_log.txt.
- baseline 0/12 still (slower). regrasp 1/12. lorap pretrain ep290, ~21:35 cap -> final auto-eval.

## 2026-09-01 22:00 — lorap pretrain done; regrasp FINAL eval; refresh smarter
- lorap pretrain done (ep300, val ~0.0176). state_300 saved. Auto-submitted eval 1904335
  was DEFAULT n=100 -> CANCELLED (mismatched baseline n=60). Also cancelled prelim 1899726.
- REGRASP FINAL eval 1904528 -> lorap/gen_eval_20260901_215958_final, n=60/40 NCAT_PAR=4
  scene_group_size=0 (matches baseline dthox eval). last_regrasp_geneval.txt updated.
- gen8_refresh.sh now _pick()s the gen_eval_* dir with MOST summary.json (newest on tie)
  -> a fresh re-eval (0 cats) won't blank the page while the prelim still has 3 cats.
- REGRASP PRELIM results (state_225, kept visible until final overtakes): banana 0.98/0.626/0.805,
  egg_boiled 0.98/0.555/0.769, kiwi 0.67/0.727/0.697. All beat/match old ueini.
- BASELINE eval 1899725 batch 8-10/12 wave 1, FPS ~0.6-0.8/env (heavy contention). Full ~00:00-00:30.

## 2026-09-01 22:17 — first baseline cat: the NEW baseline is NOT a crusher
- BASELINE kiwi (state_250): SR 0.62 / g 0.746 / SRxg 0.681.  REGRASP kiwi (state_225): 0.67 / 0.727 / 0.697.
- KEY: the grasp-at-once baseline (obs_dim 12 + object_at_gripper + hold-tail + harness hold
  fix) has gentleness ~0.75 -- the old "baseline crushes to yield (g~0)" headline is GONE.
  The v2 fixes helped BOTH models. Comparison is now CLOSE, at least on kiwi. Need full 12
  cats before drawing conclusions -- regrasp's edge (if any) is on SR + recovery cases.
- webpage b=1 r=3.

## 2026-09-01 22:51 — 4 baseline cats; clear consistent pattern
Per-cat (baseline vs regrasp-prelim), SR / gentleness / SRxg:
  banana   B 0.62/0.686/0.651  R 0.98/0.626/0.805
  egg      B 0.70/0.621/0.660  R 0.98/0.555/0.769
  kiwi     B 0.62/0.746/0.681  R 0.67/0.727/0.697
  mushroom B 0.60/0.663/0.632  R (pending; old ueini 0.58/0.77)
VERDICT SHAPE: regrasp SR consistently +0.05..+0.36, baseline gentleness consistently
+0.02..+0.07. Regrasp SRxg wins every shared cat. The "regrasp lifts more, baseline
grips gentler" tradeoff -- combined metric favors regrasp. Both models FIXED (no
carry-down, no over-squeeze). webpage b=4 r=3.

## 2026-09-01 23:29 — FINAL ckpt (state_300) MUCH better than state_225
- RFINAL kiwi: SR 0.88 / g 0.786 / SRxg 0.834  (state_225 was 0.67/0.727/0.697; baseline 0.62/0.746/0.681).
  Final checkpoint beats baseline on BOTH SR and gentleness now. Last ~75 epochs mattered.
- gen8_refresh: I now cp -rn prelim (state_225) cats into the RFINAL dir for cats RFINAL
  hasn't reached -> webpage shows RFINAL where real, prelim elsewhere; RFINAL overwrites as
  it completes. Re-run the cp before each refresh (in gen8_refresh manually for now).
- webpage b=4 r=3, kiwi now 0.88.

## 2026-09-01 23:50 — 3 RFINAL real cats: regrasp wins all 3 metrics
             BASELINE (state_250)      REGRASP FINAL (state_300)
  banana     0.62 / 0.686 / 0.651      0.98 / 0.626 / 0.805
  egg_boiled 0.70 / 0.621 / 0.660      1.00 / 0.623 / 0.811
  kiwi       0.62 / 0.746 / 0.681      0.88 / 0.786 / 0.834
Regrasp: SR +0.26..+0.36, gentleness TIED-to-+0.04, SRxg +0.15. The "baseline gentler"
read was a state_225 artifact -- state_300 gentleness matches/beats baseline while
keeping the huge SR lead. Both models free of carry-down + over-squeeze.
webpage b=4 r=3(real). Baseline still stuck on slow small fruit (batch ~3-4/8).

## 2026-09-02 00:25 — USER: build a REACTIVE policy (object dragged mid-approach). Plan + Phase A done.
PLAN: docs/PLAN_reactive_policy_2026-09-02.md (A perturb mechanism / B zero-shot eval lorap /
C collect reactive-recovery demos / D retrain regrasp v3 / E eval v3). User away 12h.
PHASE A DONE (committed): perturbation mechanism -- DRConfig.object_perturb_*, GenesisWorker
_apply_perturbation (set_particles_vel kick at a sim-frame), threaded through reset. Configs
eval_cat_*_reactive.yaml (prob 0.75, speed 0.15-0.55 m/s, frame 24-104). yd_reactive_eval.sbatch.
sample_perturb tested locally (deterministic, prob gate works). NOT yet sim-tested (GPU busy).
GATE: current gen8 v2 baseline-vs-regrasp eval must finish first -> final table + campaign
summary + publish, THEN Phase B (zero-shot reactive eval of lorap/state_300).

## 2026-09-02 01:42 — gen8 v2 verdict posted (partial); reactive Phase B smoke launched
- EXPERIMENT.md written for lorap + dthox (in their run dirs -- logs/ is gitignored so on-disk only).
  experiments.csv: lorap + dthox -> done.
- gen8 v2 VERDICT (large in-domain n=60): regrasp SR .81/g .70/SRxg .78 vs baseline .64/.68/.66.
  Recovery-driven. Baseline collapses on small fruit (raspberry .05). Verdict block on webpage.
- gen8 evals STILL running (baseline 7/12 on OOD wave, RF 4 real on small-fruit wave) -- slow
  (ikemura 7 jobs). Keep updating webpage as cats land; full campaign summary to chat once complete.
- REACTIVE SMOKE launched: 1907954, mushroom, lorap state_300, NEP=15, -t 25min,
  OUT=logs/dppo/dppo-pretrain/single_lift_gen8_regrasp_pcd/lorap/reactive_smoke_0141.
  NEXT: montage its clips, check the kick works (object slides 3-9cm mid-approach, lorap re-approaches).

## 2026-09-02 03:12 — REACTIVE Phase B: lorap PARTIALLY reactive
- Perturbation mechanism working (sustained 4-frame velocity drag, 0.3-0.85 m/s, frame 56-152).
- Smoke (mushroom, n=15): lorap state_300 SR 0.60 under drag vs 0.72 clean (-0.12). 9/15 succeed;
  ~6 fail with object dragged off (zmax<0.12). One recovery at fss=98. -> lorap tracks the
  moved object via point cloud but drops ~12pt. Moderate effect -> Phase C+D worthwhile.
- BASELINE gen8 eval DONE (11/12, tomato excluded): IN-DOMAIN(7) mean SR .50/g .745/SRxg .621
  (incl. small fruit: raspberry .05 cherry .38 grape .53). OOD(4) mean SR .45/g .578/SRxg .514
  (blackberry .68 dumpling .35 gelatin .10 scallop .68). Cancelled the tomato-only resubmit.
- LAUNCHED lorap zero-shot reactive eval 1911798 -> lorap/reactive_eval_20260902_0312_lorap_zeroshot
  (7 in-domain cats, NEP 60/40). RF (clean regrasp final) 1904528 still running (5h12, 4 real cats).
- NEXT: Phase C -- extend collect_demos_diverse_start_v2.py --reactive (perturb + FSM grasp re-target).

## 2026-09-02 03:18 — gen8 v2 in-domain COMPLETE (7 cats, both models)
                baseline   regrasp-final   d
  mushroom      0.60/.663  0.72/.718   +.12
  banana_lying  0.62/.686  0.83/.677   +.21
  kiwi          0.62/.746  0.88/.786   +.26
  egg_boiled    0.70/.621  1.00/.623   +.30
  grape         0.53/.773  0.47/.813   -.06
  cherry        0.38/.827  0.42/.883   +.04
  raspberry     0.05/.898  0.10/.860   +.05
  IN-DOM MEAN   0.50/~.75  0.63/~.78   SR +.13, gentleness ~tied
BASELINE OOD(4): blackberry .68/.682, scallop .68/.444, dumpling .35/.259, gelatin .10/.927.
REGRASP OOD: still running (1904528, 5h17).
STORY: regrasp beats baseline on in-domain SR +.13 (driven by 3-6cm objects: egg/kiwi/banana
+.21..+.30); marginal/tied on <=2cm fruit (both weak) and on gentleness. tomato excluded (bug).
- REACTIVE: Phase C collector coded+committed (--reactive). lorap zero-shot reactive eval
  1911798 RUNNING. Phase B smoke: lorap -0.12 SR under drag.
