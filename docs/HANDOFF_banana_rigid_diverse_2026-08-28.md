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
