# Training & Evaluation scripts — reference

Map of the training/eval/collection entry points, which env each runs in, and how they connect.
Analysis only — **staleness flags are at the bottom**; nothing here prescribes changes.

## The env + transport picture
Genesis needs Python 3.12; the policy stacks need other versions, so they run in separate
`envs/` and talk over a localhost socket (`gentle_manip/envs/rpc.py`). The **sim server** runs
the Genesis `PolicyEnv` and answers reset/step/render/randomize_scene; the **client** (trainer /
eval / policy) drives it.

| env | py | used for |
|-----|----|----------|
| `envs/sim` | 3.12 | Genesis sim server, demo collection, dev scripts, tests |
| `envs/serl` | 3.10 (jax) | SERL SAC/RLPD trainer |
| `envs/dppo` | 3.10 | DPPO pretrain / finetune / eval (torch) |
| `envs/dp3` | 3.8 | DP3 train/eval + **real deploy** (pyrealsense cp38) |
| `envs/deploy` | 3.11 | teleop demo collection, Open3D viz |

Run everything from the repo root with `uv run --project envs/<name> …`.

---

## 1. Demo collection (sim)
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

## 2. DPPO — the current point-cloud diffusion line
One launcher for all three DPPO stages: **`python -m gentle_manip.dppo.train`** (envs/dppo) — a
thin wrapper (`dppo/train.py`) that pins repo-root on `sys.path`, sets `DPPO_LOG_DIR`/`DPPO_DATA_DIR`,
and hands a hydra config to DPPO's `script/run.py`. The config's `_target_` picks the stage.

- **Convert demos → DPPO data:** `python -m gentle_manip.dppo.convert_demos <demo.pkl> --point-cloud
  --out dataset/dppo/<name>` → `train/val/normalization.npz` (proprio state + raw point cloud).
- **BC pretrain:** `--config-name pre_diffusion_pointnet` → `TrainDiffusionAgent` +
  `PointNetDiffusionMLP` + `pointcloud_dataset.py`. Offline (no sim). → `logs/dppo/dppo-pretrain/…`.
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
  diffusion, this doc §2) is the **current** one. `deploy_real.py`/`deploy_sim.py` still target a
  DP3 checkpoint — if the DPPO policy becomes the deployment target, those deploy paths (and the
  DP3 line generally) may need a DPPO variant or become legacy. **Decide which line is canonical.**
- **(F7) Older `dr/` presets vs `food_shape`.** `configs/dr/{mild,aggressive}.yaml` predate the
  size/shape/material work; the tuned full-DR config is `configs/dr/food_shape.yaml`. The presets
  still load but aren't the current full-DR default.
