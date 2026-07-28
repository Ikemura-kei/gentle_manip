# Training & Evaluation scripts — reference

Map of the training/eval/collection entry points, which env each runs in, and how they connect.
Analysis only — **staleness flags are at the bottom**; nothing here prescribes changes.

## Experiment tracking (all training runs)
Every TRAINING run gets a unique **5-letter ID** (`gentle_manip/utils/experiment_registry.py`)
that names its dir `logs/<algo>/<task>/<id>/` and is recorded in project-root `experiments.csv`
(gitignored; rebuild with `python -m gentle_manip.scripts.reconcile_experiments`). The **wandb**
run name == the ID. wandb is enabled by default for DPPO (disable per-run with `wandb=null` or
`WANDB_MODE=offline`) and via `--wandb` for SERL. Eval runs keep datetime naming and aren't
registered. `reconcile_experiments.py --list` prints the table.

## The env + transport picture
Genesis needs Python 3.12; the policy stacks need other versions, so they run in separate
`envs/` and talk over a localhost socket (`gentle_manip/envs/rpc.py`). The **sim server** runs
the Genesis `PolicyEnv` and answers reset/step/render/randomize_scene; the **client** (trainer /
eval / policy) drives it.

| env | py | used for |
|-----|----|----------|
| `envs/sim` | 3.12 | Genesis sim server, demo collection (incl. CMA-ES synth), dev scripts, tests |
| `envs/serl` | 3.10 (jax) | SERL SAC/RLPD trainer |
| `envs/dppo` | 3.10 | DPPO pretrain / finetune / eval (torch) |
| `envs/dppo_deploy` | 3.10 | DPPO policy + `RealBackend` on the real XArm7 (one process, genesis-free; pyrealsense cp310) |
| `envs/dp3` | 3.8 | DP3 train/eval + **real deploy** (pyrealsense cp38) |
| `envs/deploy` | 3.11 | teleop demo collection, Open3D viz |
| `envs/sim_arrhenius`, `envs/dppo_arrhenius` | 3.12 / 3.10 | aarch64 GH200 (NAISS Arrhenius) variants of `envs/sim` / `envs/dppo` — same code, cu126 torch build; x86 envs untouched |

Run everything from the repo root with `uv run --project envs/<name> …`.

---

## 1. Demo collection (sim)
- **`grasp_synthesis/collect_demos_synth.py`** (envs/sim) — **the current primary path for
  rigid-object data at scale** (e.g. the 330-episode `single_lift_mushroom_rigid` datasets).
  Per-batch: reset with pose DR → per-env CMA-ES grasp-pose synthesis (SDF-based cost against
  the object mesh + finger geometry, `grasp_synthesis/synth_utils.py`; NOT the Q_SM
  stress-minimization metric — that's a separate, not-yet-wired-in R&D track in the same dir,
  see `grasp_synthesis/CLAUDE.md`) → scripted approach→grasp→lift→hold execution. Config comes
  entirely from `--experiment` (task/obs/action/DR — the same `Experiment.load` protocol as
  every other script). Supports scene-level SIZE+SHAPE DR (`--scene-dr-every`, rebuilds the
  worker on a freshly deformed mesh) and BOTH action representations
  (`ActionConfig.mode: delta|absolute`, §1a below) — the recorded action's inversion
  (`_invert_actions` / `_invert_actions_absolute`) branches automatically on the experiment's
  action config.
  ```
  uv run --project envs/sim python grasp_synthesis/collect_demos_synth.py \
      --experiment single_lift_mushroom_rigid_abs_action \
      --n-episodes 330 --n-envs 8 --scene-dr-every 1
  ```
  → `dataset/demos/<task>/<date>-<xyz>/data.pkl` (+ `videos/`, `videos_failed/`, `config.yaml`,
  `stats.yaml`, `cmaes_logs/`).
- **`examples/collect_demos_sim.py`** (envs/sim) — config-driven collection. Drives a
  `PolicyEnv(SimBackend)` with either the **scripted expert** (`demos/scripted_policy.py`,
  `ScriptedLiftDemonstrator` — size-adaptive approach→grasp→lift→hold on privileged state) or
  **keyboard teleop** (`demos/teleop_keyboard.py`), recording SUPERSET obs + per-step reward via
  `demos/record.py::DemoRecorder`. Full-DR relaunch via `scene_dr_every` + `dr_override`.
  ```
  MUJOCO_GL=egl uv run --project envs/sim python examples/collect_demos_sim.py \
      --config gentle_manip/configs/collect/single_lift_mushroom_soft_fulldr.yaml
  ```
  → `dataset/demos/<task>/<date>-<xyz>/data.pkl` (+ `videos/`, `config_resolved.yaml`).
- **`demos/record.py`** (envs/deploy) — same `DemoRecorder` CLI for real teleop (`--input
  spacemouse|keyboard`).

### 1a. Action representation: delta vs absolute
`ActionConfig.mode` (`gentle_manip/actions/action_config.py`) — **`"delta"` is the default**
(unchanged behaviour, every existing config/script keeps working with no `mode` key):
policy output is clipped then scaled into a physical Δpose+Δgripper that the backend
accumulates onto a running target (`configs/action/delta_pose_delta_gripper*.yaml`).
`"absolute"` (`configs/action/abs_pose_abs_gripper.yaml`) maps a 10-dim raw output
(3 pos + 6D rotation + 1 gripper, Zhou et al. 2019 Gram-Schmidt orthonormalization for the
rotation) directly into an 8-dim physical target (pos + quat + gripper) that the backend SETS
each step instead of accumulating. `SimBackend.step()`/`RealBackend.step()` dispatch on the
**output width** (7 = delta, 8 = absolute) — no constructor/signature changes anywhere, so
this is fully backward compatible. `single_lift_mushroom_rigid_abs_action.yaml` is the
experiment wired to the absolute config; the `_fixedhome`/`_state` experiment variants
(fixed home-pose DR / full privileged state incl. `priv_object_rot6d`+`priv_object_dr_params`)
compose independently of action mode.

## 2. DPPO — the current point-cloud diffusion line
One launcher for all three DPPO stages: **`python -m gentle_manip.dppo.train`** (envs/dppo) — a
thin wrapper (`dppo/train.py`) that pins repo-root on `sys.path`, sets `DPPO_LOG_DIR`/`DPPO_DATA_DIR`,
and hands a hydra config to DPPO's `script/run.py`. The config's `_target_` picks the stage.

- **Convert demos → DPPO data:** `python -m gentle_manip.dppo.convert_demos <demo.pkl> --out
  dataset/dppo/<name> --experiment <name> --view student --point-cloud` → `train/val/
  normalization.npz` (proprio state + raw point cloud). **Always pass `--experiment`/`--view`**
  (not bare `--obs-keys`/`--point-cloud`) so the obs-key order is derived the same way as the
  online env — the single-source-of-truth rule (see `CLAUDE.md` Conventions). `--view student`
  → point-cloud student (`obs_dim=8`: ee_pos+ee_quat+gripper); `--view teacher` on a `_state`
  experiment (e.g. `single_lift_mushroom_rigid_state`, `obs: superset_rigid_full_state`) → the
  19-dim full-state view (`STATE_VIEW_FULL`: + `priv_object_pos/rot6d/dr_params`) for a
  **state-based** DPPO policy, still with `--point-cloud` alongside for a dual-purpose dataset
  (same demos, drop the state view later for a student re-train with no re-collection). Works
  transparently with either `ActionConfig.mode` (§1a) — `action_dim` in the output meta is 7
  (delta) or 10 (absolute), read from the recorded array shape, no flag needed.
- **BC pretrain:** `--config-name pre_diffusion_pointnet` (point-cloud) or `pre_diffusion_mlp`
  (state) → `TrainDiffusionAgent` + `PointNetDiffusionMLP`/plain MLP + the matching dataset
  loader. Offline (no sim). → `logs/dppo/dppo-pretrain/…`.
- **PPO finetune:** `--config-name ft_ppo_diffusion_pointnet base_policy_path=<bc.pt>` →
  `TrainPPOImgDiffusionAgent` + `PointNetDiffusionMLP` actor + `PointNetCritic`. Needs a **student
  server** running:
  ```
  serl_sim_server --experiment single_lift_mushroom_soft --view student --num-envs 5 \
                  --render-rgb --scene-dr-every 25    # within-run full DR
  ```
  Reward = the sim's shaped reward incl. the von-Mises stress penalty. → `logs/dppo/dppo-finetune/…`.
- **DPPO eval:** `--config-name eval_diffusion_pointnet base_policy_path=<ckpt.pt>` →
  `dppo/eval_agent.py::EvalHarnessAgent`, which routes through the **shared canonical harness**
  (`gentle_manip/evaluation/`). Needs a student server with **`--subprocess`** (harness drives
  per-group scene DR). → `<policy_run>/eval/<datetime>/{summary.json, episodes.csv, render/}`.
- **Real deploy:** **`gentle_manip/scripts/deploy_real_dppo.py`** (envs/dppo_deploy) — runs a
  BC or PPO-finetune DPPO checkpoint closed-loop on the real XArm7 (policy + `RealBackend` share
  one process; genesis-free). `--ft-denoising-steps 0` for a pure-BC checkpoint, `>0` (e.g. 10)
  for a finetuned one (shortened DDPM chain). `--normalization` must be the SAME dataset's
  `normalization.npz` the checkpoint trained on. `--record <dir>` saves the run in the demo
  shard schema for `examples/sim2real_diagnose/` (§7). See `gentle_manip/scripts/deploy_real.sh`
  for worked examples (rigid BC, real-demo BC, soft finetune).
- Model/data code: `dppo/pointnet_diffusion.py` (encoder+actor+critic), `dppo/pointcloud_dataset.py`,
  `dppo/genesis_venv.py` (the rpc bridge = DPPO VectorEnv), `dppo/hydra_snapshot.py` (env-cfg snapshot).

## 3. SERL — the state teacher (SAC/RLPD)
- **Training:** **`gentle_manip/serl/train_serl.py`** (envs/serl) — HIL-SERL SAC/RLPD, one
  actor + one learner over agentlace, driven by an experiment config (`--view teacher`). Connects
  to a **teacher server**: `serl_sim_server --experiment <name> --view teacher --num-envs N`.
  Writes `EXPERIMENT.md` + checkpoints to `logs/serl/…`.
- **Demos → RLPD transitions:** `serl/convert_demos.py` → a flat list of
  `{observations, actions, next_observations, rewards, masks, dones}` for the demo replay buffer.
- **SERL eval:** **inline in `train_serl.py`** (per-episode return/success logged to wandb/run.log).
  There is **no standalone SERL eval through the canonical harness** yet — see flag (F4).
- `serl/gym_env.py` = `SimGymEnv` (single-env gym wrapper over the rpc client for SERL).

## 4. Scripted-policy eval (baseline)
- **`gentle_manip/scripts/eval_scripted.py`** (envs/sim) — evaluates the scripted expert through
  the **same canonical harness** as the learned policies (`SimEvalVenv` + `ScriptedPolicy`, both
  size-adaptive). Needs a teacher-view server (has `priv_object_pos`); `--subprocess --scene-group-size K`
  for full-DR eval. → `logs/scripted_policy/<datetime>/`. Baseline to compare learned policies against.

## 5. Shared sim servers & the eval harness
- **`scripts/serl_sim_server.py`** (envs/sim) — the general Genesis rpc server used by **SERL AND
  DPPO** (training + eval). Flags: `--view teacher|student`, `--num-envs`, `--render-rgb`,
  `--scene-dr-every N` (auto relaunch → subprocess), `--subprocess` (harness-driven eval),
  `--dr <name>` (override DR). Despite the `serl_` name it's general — see flag (F1).
- **`scripts/sim_server.py`** (envs/sim) — an **older, DP3-specific** rpc server (in-process,
  viewer). Used by the DP3 line (`eval_sim.py`, `deploy_sim.py`). Lacks render-rgb/scene-dr — see (F2).
- **`gentle_manip/evaluation/`** — the canonical eval harness (`EvalSpec` 100 eps/5 envs/fixed
  scenario seq; `run_eval`; per-episode CSV incl. geometry + stress). Used by DPPO eval + scripted eval.

## 6. DP3 — the original point-cloud line (torch, 3.8)
- **Convert:** `scripts/convert_demo_to_dp3.py` → DP3 zarr.
- **Train:** `third_party/DP3/3D-Diffusion-Policy/train.py` (the DP3 fork).
- **Eval:** `scripts/eval_sim.py` (envs/dp3) — DP3 checkpoint eval via `SimXArm7Runner`, spawns
  `sim_server.py`. **Predates the canonical harness** — see (F3).
- **Deploy:** `scripts/deploy_sim.py` (DP3 policy on the sim, via `sim_server`) and
  `scripts/deploy_real.py` (DP3 policy + `RealBackend` on the real XArm7). `smoke_real.py` = gated
  hardware bring-up. `inspect_demo.py` = sim/real parity check on a demo pkl.

## 7. Sim2real diagnostics & demo analysis (offline, no sim server needed)
- **`examples/sim2real_diagnose/`** (envs/sim, genesis) — open-loop diagnostics that isolate
  control-vs-perception sim2real gaps:
  - `replay_demo_in_sim.py` — replay a recorded TELEOP demo's actions in sim; compares ee_pos/
    ee_quat/gripper/point-cloud, real vs sim, per episode.
  - `replay_deploy_in_sim.py` — same idea for a real **deployment** run (`--record` output of
    `deploy_real_dppo.py`): loads `Experiment.load(--experiment)` for task/obs/action (single
    source of truth, no hardcoded ranges), replays each episode's recorded actions in a fresh
    sim rollout, and reports per-episode EE/quat/gripper error + point-cloud z-offset. Found the
    persistent ~3-12mm real-table-higher-than-sim z-offset via this route.
  - `analyze_deploy_gap.py` (envs/deploy) — real-data-only companion: action-following gap
    (commanded vs actual Δpose from consecutive obs) + point-cloud stats, no sim replay needed.
  All write into `<run_dir>/sim_replay/<timestamp>/` or `<deploy_dir>/gap_analysis/`. Output
  figures under `figures/<run-name>/` are local scratch by convention — check `git status`
  before assuming a given run's output is tracked (the scripts themselves are).
- **`examples/demo_analysis/`** (envs/sim or envs/deploy depending on script) — dataset QA on
  any recorded `data.pkl` (teleop, CMA-ES synth, or real deploy):
  - `action_distribution.py` — per-axis histograms; auto-detects delta (7-dim, zero-fraction
    analysis) vs absolute (10-dim, full distribution — zero isn't special in absolute mode) from
    the recorded action width. For absolute mode, replaces the delta "zero%"/magnitude framing
    with **consecutive unchanged-action run-length** analysis (`action_held_frames.png` +
    per-episode/aggregate held% table) — the meaningful "held still" signal when the command is
    a target, not a step. Scales to large datasets: full per-episode table ≤20 episodes,
    aggregate stats + outlier flagging (deviating `n_runs`) beyond that.
  - `trajectory_smoothness.py` — from the REALIZED ee_pos/gripper trajectory (robust to action
    mode): speed/accel/jerk, log-dimensionless-jerk (Hogan & Sternad / Balasubramanian, duration-
    and amplitude-normalized so episodes are comparable), path efficiency, plus recorded-action
    step-size (command jumpiness — most informative for absolute-mode actions, which have no
    delta-accumulation to smooth over a jerky commanded sequence). Detailed per-episode
    time-series plots cap at 8 evenly-sampled episodes; beyond that the summary figure switches
    from one-bar-per-episode to histograms, and the printed table to aggregate stats + 2σ
    outlier flagging (LDJ / path efficiency) — same large-dataset pattern as above.
  - `grasp_pose_analysis.py`, `visualize_tactile_demos.py` — grasp-pose distribution and paired
    point-cloud+tactile video rendering respectively; see each script's docstring for usage.

---

## Staleness / consolidation flags (analysis only — not changing anything)

- **(F1) `serl_sim_server` is misnamed.** It's the general Genesis rpc server used by **both** SERL
  and DPPO (and the scripted/DPPO evals). The `serl_` prefix is misleading; a rename to something
  like `sim_rpc_server.py` would reflect reality. *Not urgent — purely cosmetic.*
- **(F2) Two sim servers.** `sim_server.py` (DP3 line, 2026-06-30) and `serl_sim_server.py`
  (SERL/DPPO, 2026-07-04) overlap. `serl_sim_server` is the newer, more capable one (views,
  render-rgb, scene-dr, subprocess restart). `sim_server` lacks all of those. If the DP3 line is
  retired, `sim_server` becomes dead; otherwise it's a second, feature-poor server to maintain.
- **(F3) `eval_sim.py` predates the canonical eval harness.** It's DP3-specific (`SimXArm7Runner`,
  reward-threshold "success", its own output format) and does NOT go through
  `gentle_manip/evaluation/run_eval`. So DP3 and DPPO/scripted evals are **not apples-to-apples**
  (different protocol/metrics). If DP3 stays in use, wiring it through the harness would unify it.
- **(F4) No standalone SERL eval through the harness.** SERL only evaluates inline in `train_serl`.
  A `SERL EvalVenv`+`Policy` adapter (mirroring `eval_scripted`) would let SERL report the same
  `summary.json`/`episodes.csv` as DPPO — this was noted as a fast-follow when the harness was built.
- **(F5) Three demo converters.** `serl/convert_demos.py` (RLPD transitions), `dppo/convert_demos.py`
  (train/val/normalization npz), `scripts/convert_demo_to_dp3.py` (DP3 zarr) — all consume the same
  recorded `data.pkl` for different downstream formats. Correct by design (different targets), but
  worth knowing they're parallel; a shared "load recorded demo" front-end could de-duplicate the
  read/subset logic.
- **(F6) DP3 vs DPPO line.** The DP3 line (`convert_demo_to_dp3` → DP3 fork `train.py` → `eval_sim`
  / `deploy_sim` / `sim_server`) is the **original** point-cloud policy path; DPPO (PointNet
  diffusion, this doc §2) is the **current** one. **Partially resolved:** DPPO now has its own
  real-deploy path (`deploy_real_dppo.py`, envs/dppo_deploy), so the deploy side no longer forces
  a DP3 checkpoint. Still open: `deploy_sim.py`/`eval_sim.py` remain DP3-only (no DPPO
  equivalent for sim-side DP3-style deploy/eval outside the canonical harness), and the DP3
  training/convert side (`convert_demo_to_dp3`, DP3 fork `train.py`) is unchanged. **Decide
  whether DP3 training stays in use, or is fully retired now that DPPO covers train+eval+deploy.**
- **(F7) Older `dr/` presets vs `food_shape`.** `configs/dr/{mild,aggressive}.yaml` predate the
  size/shape/material work; the tuned full-DR config is `configs/dr/food_shape.yaml`. The presets
  still load but aren't the current full-DR default.
- **(F8) Two demo-collection entry points, different scope.** `examples/collect_demos_sim.py`
  (scripted-expert or teleop, general-purpose, used historically for soft-body full-DR) and
  `grasp_synthesis/collect_demos_synth.py` (CMA-ES grasp synthesis, this doc §1 — the current
  path for RIGID-object data at scale, e.g. the 330-ep `single_lift_mushroom_rigid` sets). Not
  redundant (different grasp-generation strategy, and only the synth path optimizes each
  episode's grasp against the object's actual mesh/pose), but worth knowing they're parallel and
  a newcomer could reasonably expect one canonical collector.
- **(F9) `grasp_synthesis/` mixes two unrelated tracks.** `collect_demos_synth.py`/
  `synth_utils.py` (SDF-cost CMA-ES grasp synthesis, used for demo collection, this doc §1) and
  the Q_SM stress-minimization grasp metric (`grasp_synthesis/CLAUDE.md`, `qsm_objective.py`,
  `grasp_synthesis_qsm.py`, `demo_qsm_grasp.py`) share a directory but are **not integrated** —
  Q_SM is a separate, more physically-grounded grasp-quality R&D metric that could in principle
  replace the SDF cost `collect_demos_synth.py` currently optimizes, per that CLAUDE.md's §9,
  but this wiring has not been done.
- **(F10) `envs/*_arrhenius` are hand-maintained mirrors.** `envs/sim_arrhenius`/
  `envs/dppo_arrhenius` are copies of `envs/sim`/`envs/dppo` with only the torch build + a
  header differing (aarch64/cu126 for the NAISS Arrhenius GH200 cluster). Any dependency change
  to the x86 envs needs a matching manual update on the arrhenius side — no shared source of
  truth beyond the comment noting the divergence.
