# DPPO + DP3 (point-cloud diffusion policy) training recipe

A soup-to-nuts recipe: collected demos → converted dataset → BC pretrain → (optional) PPO
finetune → eval, for the **point-cloud diffusion policy** pipeline used throughout this repo's
sim2real work (the `cho/ahaxs` checkpoint referenced across `examples/sim2real_diagnose/` is a
BC-pretrain output of exactly this recipe). Companion to
`docs/grasp_synthesis_data_collection.md` (stage 0, producing the input demos) and the more
compact map in `docs/training_and_eval.md` (§2) — this doc goes one level deeper on the config
structure specifically, per request.

## 0. Naming: which "DP3" is this?

Two separate point-cloud lines exist in this repo (`docs/training_and_eval.md` §2 vs §6):

- **DPPO's point-cloud path** (this doc) — `agent.pretrain.train_diffusion_agent` +
  `gentle_manip/dppo/pointnet_diffusion.py::PointNetDiffusionMLP`. The network is a
  **DP3-style** (3D Diffusion Policy) PointNet-encoder + diffusion-MLP actor, but it trains and
  finetunes *inside DPPO's* pretrain/PPO-finetune machinery, sharing one launcher
  (`gentle_manip.dppo.train`) with the state-only (non-point-cloud) DPPO configs. This is the
  **current, actively used** line, and what "DPPO DP3" means in this doc.
- **The standalone legacy DP3 line** (`third_party/DP3/3D-Diffusion-Policy`, zarr datasets,
  `scripts/convert_demo_to_dp3.py` / `eval_sim.py` / `deploy_sim.py`) — the *original*
  point-cloud policy before it was folded into DPPO. Still present and functional but not the
  subject of this recipe; see `docs/training_and_eval.md` §6 if you need it.

## 1. Prerequisites

```bash
uv sync --project envs/sim && uv sync --project envs/dppo
# torch is installed manually in both (CUDA build differs per env — see each pyproject header):
#   uv pip install --python envs/sim/.venv/bin/python  "torch==2.5.1+cu121" --index-url https://download.pytorch.org/whl/cu121
#   uv pip install --python envs/dppo/.venv/bin/python "torch==<pinned>+cu121" --index-url https://download.pytorch.org/whl/cu121
bash examples/env_debug/run_all.sh sim dppo   # verify both envs before real runs
```
Two Python environments are involved: `envs/sim` (3.12, Genesis — collection only) and
`envs/dppo` (3.10, torch — conversion/pretrain/finetune/eval). Run everything from the **repo
root** with `uv run --project envs/<name> …` (never `--directory`).

## 2. Stage 1 — collect demos (envs/sim)

Full detail in `docs/grasp_synthesis_data_collection.md`. Short version:

```bash
uv run --project envs/sim python grasp_synthesis/collect_demos_synth_v2.py \
    --experiment single_lift_mushroom_rigid_state_abs_action_force \
    --n-episodes 650 --n-envs 8 --maxfevals 1145 --scene-dr-every 1 --seed 0
# -> dataset/demos/single_lift_mushroom_rigid/<date>-<xyz>/data.pkl
```

The real example behind `cho/ahaxs`: 650 episodes, 83.76% CMA-ES success rate, ~110 minutes on
8 parallel envs (`dataset/demos/single_lift_mushroom_rigid/26-07-29-cho/`).

**Nothing about this stage is DPPO-specific** — `data.pkl` is the generic superset-obs demo
schema; the same file could feed the legacy DP3 line, SERL, or a state-only DPPO run. The
experiment (`--experiment`) determines *what got recorded*; which DPPO config you point at it
next determines *which slice of that recording* actually gets used (§3).

## 3. Stage 2 — convert to DPPO format (envs/dppo): data conversion notes

`gentle_manip/dppo/convert_demos.py` (Genesis-free, pure numpy) turns the superset demo pkl(s)
into DPPO's flat-array contract. **Always drive it with `--experiment`/`--view`, not bare
`--obs-keys`/`--point-cloud`** — this is the same single-source-of-truth argument as
collection: it derives the obs-key order from the SAME `Experiment.view_obs(view)` call the
online env uses, so the offline pretrain data and a later PPO-finetune env can never disagree
on channel order.

```bash
uv run --project envs/dppo --no-sync python -m gentle_manip.dppo.convert_demos \
    dataset/demos/single_lift_mushroom_rigid/26-07-29-cho \
    --out dataset/dppo/single_lift_mushroom_rigid/cho \
    --experiment single_lift_mushroom_rigid_state_abs_action_force --view student \
    --point-cloud
```

### 3a. What actually happens, step by step

1. **Find demos.** `_find_demo_pkls(root)` recurses the given path for `data.pkl` (or any
   `*.pkl` if none found) — you can point it at a single file or a whole run directory.
2. **Pick the obs-key order.** With `--experiment`, `exp.view_obs(args.view).obs_keys()` gives
   the view's ordered key list, with point-cloud/voxel/image/tactile keys filtered out (those
   are handled separately, step 4) — leaving only the **flat, concatenable** state keys.
   - `--view teacher` on a `_state` experiment (e.g. `single_lift_mushroom_rigid_state`, whose
     `obs:` is `superset_rigid_full_state`) → `STATE_VIEW_FULL` = `[ee_pos, ee_quat,
     gripper_width, priv_object_pos, priv_object_rot6d, priv_object_dr_params]`, **19-dim** — a
     full-privileged-state policy.
   - `--view student` → `PROPRIO_VIEW` = `[ee_pos, ee_quat, gripper_width]`, **8-dim** — the
     point-cloud student's proprioception half (matches `obs_dim: 8` in the `cho/ahaxs`
     training config — see §4).
   - Without `--experiment`, `convert_demos.py` falls back to the hardcoded module constants
     `STATE_VIEW` (14-dim, includes `priv_object_pos/vel` but not rotation/DR-params) or
     `PROPRIO_VIEW` if `--point-cloud` is set — kept for quick one-offs, but **do not use this
     path for a real training dataset**: it can silently drift from the online env's obs order.
3. **Flatten each episode's state.** `_episode_state`: for every key in the resolved order,
   reshape that episode's `(T, ...)` array to `(T, -1)` and concatenate columns → one
   `(T, obs_dim)` array per episode.
4. **Point cloud (student path only), `--point-cloud [key]`** (default key `"point_cloud"`):
   stored as **raw, un-normalized** `(T, N, 3)` metric coordinates alongside the flat state —
   the PointNet encoder consumes real xyz, not a normalized flat vector. One dataset can carry
   *both* the state view and the raw cloud simultaneously (`--view teacher/student` + always
   `--point-cloud`), so **one demo set is dual-purpose**: a state policy and a point-cloud
   student can both train from it without re-collecting.
5. **Normalization stats — computed over ALL data, in raw units:**
   `obs_min/max = states.min(0)/.max(0)`, `action_min/max` likewise, matching DPPO's own
   convention exactly (`2*(x-min)/(max-min+1e-6) - 1`, the same formula
   `third_party/dppo/script/dataset/process_robomimic_dataset.py` uses). Point clouds are
   **never normalized** — copied through in metric coordinates.
6. **Train/val split — BY TRAJECTORY**, not by frame (`val_split=0.1` default): a fixed-seed
   permutation of episode indices, first `n_train` → train. States/actions inside each split
   are written **already normalized**; `traj_lengths` records each episode's length so DPPO can
   reconstruct the concatenated-trajectory boundaries (`terminals` is otherwise all-`False`).

### 3b. Output contract

```
<out>/train.npz           states, actions, rewards, terminals, traj_lengths[, point_cloud]
<out>/val.npz             same fields, held-out trajectories
<out>/normalization.npz   obs_min, obs_max, action_min, action_max   (RAW units)
```
`normalization.npz` is loaded again at PPO-finetune time (`GenesisMultiStepVecEnv`) and at real
deploy (`deploy_real_dppo.py --normalization`) — the BC-pretrained policy, the online finetune
env, and a deployed checkpoint must all share **the same** `normalization.npz`, or the raw↔[-1,1]
mapping silently disagrees between them.

**Real numbers** (`dataset/dppo/single_lift_mushroom_rigid/cho/`, the `cho/ahaxs` dataset):

| file | episodes | total steps | shape highlights |
|---|---|---|---|
| `train.npz` | 585 | 117,368 | `states (117368,8)`, `actions (117368,10)`, `point_cloud (117368,1024,3)` |
| `val.npz` | 65 | 13,056 | same shapes |

(585 + 65 = 650, matching the 650 episodes collection saved — the default `val_split=0.1` 90/10
trajectory split, confirmed by `traj_lengths.sum() == states.shape[0]` in both files.)

**Data-scaling variant:** `gentle_manip/dppo/subsample_dataset.py` builds a nested subset (e.g.
150/300 of a 1000-episode source) for data-scaling studies — it de-normalizes back to raw units
with the *source* stats, re-computes normalization over just the subset's train split, and
re-normalizes val with those new stats (so val stays comparable across sizes while train size
varies). Not part of the base recipe; use it only for that specific ablation.

## 4. Stage 3 — BC pretrain (envs/dppo): how the hydra config is used

One launcher for every DPPO stage: `python -m gentle_manip.dppo.train` (`gentle_manip/dppo/
train.py`) — a thin wrapper that pins the repo root onto `sys.path` (so `gentle_manip` imports
survive hydra's `chdir` into the run's logdir), sets `DPPO_LOG_DIR`/`DPPO_DATA_DIR` env vars if
unset, then hands off to DPPO's own hydra entry point (`third_party/dppo/script/run.py`)
unchanged. The hydra config's `_target_` field picks which of DPPO's agent classes actually
runs — pretrain vs finetune vs eval are just different configs through the same launcher.

```bash
uv run --project envs/dppo --no-sync python -m gentle_manip.dppo.train \
    --config-path "$(pwd)/gentle_manip/dppo/cfg/single_lift_mushroom_rigid_abs_pcd" \
    --config-name pre_diffusion_pointnet \
    env=single_lift_mushroom_rigid/cho
```
(`single_lift_mushroom_rigid_abs_pcd/` is the **absolute**-action config dir — `action_dim: 10`,
`experiment: single_lift_mushroom_rigid_abs_action` — and is what the `cho/ahaxs` checkpoint
actually used, confirmed against its resolved `.hydra/config.yaml`. The sibling
`single_lift_mushroom_rigid_pcd/` dir is the **delta**-action variant, `action_dim: 7`,
otherwise identical file layout — use whichever matches the `ActionConfig.mode` your demos were
collected/converted with, §3a.)

### 4a. Config anatomy (annotated, from the actual `cho/ahaxs` run's resolved config)

```yaml
_target_: agent.pretrain.train_diffusion_agent.TrainDiffusionAgent   # which DPPO agent class

# ── path templates, resolved by two OmegaConf resolvers registered in train.py ──
logdir: ${oc.env:DPPO_LOG_DIR}/dppo-pretrain/${env}/${exp_id:}   # ${exp_id:} mints ONE fresh
                                                                  # 5-letter run ID per process
                                                                  # (this run: "ahaxs") — the
                                                                  # run-dir leaf, the wandb run
                                                                  # name, AND the row written
                                                                  # into project-root experiments.csv
train_dataset_path: ${oc.env:DPPO_DATA_DIR}/${env}/train.npz     # <repo>/dataset/dppo/${env}/train.npz
val_dataset_path:   ${oc.env:DPPO_DATA_DIR}/${env}/val.npz

# ── the "env" / "experiment" split (easy to confuse — they answer different questions) ──
env: single_lift_mushroom_rigid/cho          # WHERE the data + logs live (a path fragment,
                                              # arbitrary string — "cho" here is just this
                                              # dataset's label, not a gentle_manip concept)
experiment: single_lift_mushroom_rigid_abs_action   # WHICH gentle_manip Experiment describes
                                              # the task/DR/action this policy is FOR — read by
                                              # downstream eval/finetune configs (via
                                              # ExperimentSnapshot, §4b) to snapshot the actual
                                              # env config into the run dir; NOT used by
                                              # pretrain itself (offline, no sim)

seed: 42
device: cuda:0
obs_dim: 8              # PROPRIO_VIEW width — MUST match what convert_demos.py wrote (§3a)
action_dim: 10           # 10 = absolute-mode action (3 pos + 6D rot + 1 gripper); 7 = delta mode
denoising_steps: 20      # full DDPM chain length for BC (finetune can shorten this, see §5)
horizon_steps: 4         # action-chunk length the policy predicts per inference (DPPO "ta")
cond_steps: 2            # how many past proprio observations condition the policy
pc_cond_steps: 1         # how many past point-cloud frames condition the policy (usually 1)
n_points: 1024           # points per cloud (must match the obs config's point_cloud.max_points)
visual_feature_dim: 256  # PointNet encoder output width, feeding the diffusion-MLP actor

train:
  n_epochs: 2000
  batch_size: 128
  learning_rate: 0.0001
  save_model_freq: 400   # checkpoint cadence -> logs/.../checkpoint/state_<epoch>.pt
  val_freq: 10

model:
  _target_: model.diffusion.diffusion.DiffusionModel
  network:
    _target_: gentle_manip.dppo.pointnet_diffusion.PointNetDiffusionMLP   # the DP3-style network (§0)
    cond_dim: ${eval:'${obs_dim} * ${cond_steps}'}    # hydra `${eval:...}` — 8*2=16 here
    pointnet: {in_channels: 3, use_layernorm: true, final_norm: layernorm}

train_dataset:            # and an identical val_dataset: block
  _target_: gentle_manip.dppo.pointcloud_dataset.StitchedSequencePointCloudDataset
  dataset_path: ${train_dataset_path}
  horizon_steps: ${horizon_steps}
  cond_steps: ${cond_steps}
  pc_cond_steps: ${pc_cond_steps}
```

**The pattern to internalize:** almost every top-level scalar (`obs_dim`, `action_dim`,
`horizon_steps`, `cond_steps`, `pc_cond_steps`, `device`, the dataset paths) is defined **once**
at the top and then referenced via `${...}` interpolation into `model:`/`train_dataset:`/
`val_dataset:` — so a single `env=... obs_dim=... horizon_steps=...`-style hydra override on
the command line propagates everywhere it's needed, rather than requiring edits in three
places. This is standard hydra config composition, not a gentle_manip-specific mechanism, but
it's why the override list on any launch command (`env=`, `train.n_epochs=`, etc.) is short even
though the resolved config is long.

Both `single_lift_mushroom_rigid_pcd/` and `single_lift_mushroom_rigid_abs_pcd/` (see the note
in §4 above) have identical file names (`pre_diffusion_pointnet.yaml`,
`eval_diffusion_pointnet.yaml`, `ft_ppo_diffusion_pointnet.yaml`) — only the resolved values
(`action_dim`, `experiment:`) differ, so the same `--config-name` works regardless of which
directory you point `--config-path` at.

### 4b. Every run gets a snapshot + a registry row

`gentle_manip/dppo/hydra_snapshot.py`'s `ExperimentSnapshot` hydra callback (enabled via a
`hydra.callbacks` entry + the `experiment:` field above) copies the resolved `tasks/obs/action/
dr/experiments` YAML files into `<run>/config/` — the record of *what env this policy actually
trained on*, independent of `git log` (configs can change after a run). `gentle_manip/utils/
experiment_registry.py::new_id()`/`add_entry()` mint the ID and add the `experiments.csv` row
(id, algo, task, run_dir, created, commit, status) — see the root `CLAUDE.md` Conventions
section for the full experiment-tracking contract (every run needs an ID + registry row +
`EXPERIMENT.md`; only the ID+registry half is automatic for DPPO today).

## 5. Stage 4 (optional) — PPO finetune

Needs a **live sim server** (a **student**-view server, matching the pretrain's obs config) in
addition to the trainer:

```bash
# terminal 1 (envs/sim) — leave running
uv run --project envs/sim --no-sync python -m gentle_manip.scripts.serl_sim_server \
    --experiment single_lift_mushroom_rigid_abs_action --view student \
    --num-envs 12 --render-rgb --scene-dr-every 25 --port 5570

# terminal 2 (envs/dppo)
uv run --project envs/dppo --no-sync python -m gentle_manip.dppo.train \
    --config-path "$(pwd)/gentle_manip/dppo/cfg/single_lift_mushroom_rigid_abs_pcd" \
    --config-name ft_ppo_diffusion_pointnet \
    base_policy_path=logs/dppo/dppo-pretrain/single_lift_mushroom_rigid/cho/ahaxs/checkpoint/state_800.pt
```
`ft_ppo_diffusion_pointnet.yaml`'s `_target_` is `TrainPPOImgDiffusionAgent` (a `PointNetCritic`
alongside the same `PointNetDiffusionMLP` actor); the reward is the sim's shaped reward
(+ stress penalty for soft tasks). `dppo/genesis_venv.py` is the rpc bridge acting as DPPO's
`VectorEnv` — see `docs/dppo_finetune_example.md` for a fully worked, copy-pasteable example
(server flags, expected log lines, troubleshooting).

## 6. Stage 5 — eval through the canonical harness

**Never write a second eval loop for DPPO** — every eval (BC or finetuned) goes through
`gentle_manip.evaluation.run_eval` via `dppo/eval_agent.py::EvalHarnessAgent`:

```bash
uv run --project envs/dppo --no-sync python -m gentle_manip.dppo.train \
    --config-path "$(pwd)/gentle_manip/dppo/cfg/single_lift_mushroom_rigid_abs_pcd" \
    --config-name eval_diffusion_pointnet \
    base_policy_path=logs/dppo/dppo-pretrain/single_lift_mushroom_rigid/cho/ahaxs/checkpoint/state_800.pt \
    ft_denoising_steps=0     # 0 for a pure-BC checkpoint; >0 (e.g. 10) for a PPO-finetuned one
```
Needs a student-view server with `--subprocess` (the harness drives per-scene-group DR itself).
Output: `<policy_run>/eval/<datetime>/{summary.json, episodes.csv, render/*.mp4}` — one clip per
episode, **required**, see `docs/dppo_eval.md` for the full protocol and the hard
per-trajectory-video requirement.

## 7. Real deploy

`gentle_manip/scripts/deploy_real_dppo.py` (envs/dppo_deploy — one process, policy +
`RealBackend`, genesis-free) runs a BC or finetuned checkpoint closed-loop on the real XArm7.
`--ft-denoising-steps 0` for BC, `>0` for finetuned (shortened DDPM chain, matching whatever the
finetune used); `--normalization` must point at the **same** `normalization.npz` from §3b — see
`gentle_manip/scripts/deploy_real.sh` for worked examples including the `cho/ahaxs` entry.

## 8. End-to-end summary

```
grasp_synthesis/collect_demos_synth_v2.py --experiment <exp>
    -> dataset/demos/<task>/<date>-<xyz>/data.pkl                         [docs/grasp_synthesis_data_collection.md]
gentle_manip/dppo/convert_demos.py --experiment <exp> --view student --point-cloud
    -> dataset/dppo/<env>/{train,val,normalization}.npz                   [§3 above]
gentle_manip.dppo.train --config-name pre_diffusion_pointnet env=<env>
    -> logs/dppo/dppo-pretrain/<env>/<id>/checkpoint/state_<epoch>.pt     [§4 above]
(optional) gentle_manip.dppo.train --config-name ft_ppo_diffusion_pointnet base_policy_path=<bc.pt>
    -> logs/dppo/dppo-finetune/<...>/<id>/checkpoint/state_<epoch>.pt     [§5 above]
gentle_manip.dppo.train --config-name eval_diffusion_pointnet base_policy_path=<ckpt.pt>
    -> <policy_run>/eval/<datetime>/{summary.json, episodes.csv, render/} [§6 above]
gentle_manip/scripts/deploy_real_dppo.py --ckpt <ckpt.pt> --normalization <norm.npz>
    -> closed-loop on the real XArm7                                     [§7 above]
```
