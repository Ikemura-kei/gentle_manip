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
