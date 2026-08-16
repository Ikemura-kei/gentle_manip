# Local runbook — train & evaluate a DPPO PointNet diffusion policy

Concise companion to `docs/cluster_data_collection.md`: once a demo `data.pkl` is back from the
cluster, this is the **convert → pretrain → eval** flow, done locally. For the deep dive (config
anatomy, normalization, finetuning) see `docs/dppo_dp3_training_recipe.md` and `docs/dppo_eval.md`.

The three stages, at a glance (example: the `highcam_rot6d` variant used for the smoke test):

```bash
EXP=single_lift_mushroom_soft_abs_action_highcam_rot6d          # the experiment the demos were collected with
ENV=single_lift_mushroom_soft_abs_pcd_highcam_rot6d             # the DPPO config dir + data/log dir name
CFG=$(pwd)/gentle_manip/dppo/cfg/$ENV
DATA=dataset/demos/single_lift_mushroom_soft_highcam/<id>/data.pkl

# 1. convert demos -> DPPO npz (envs/dppo)
env -u PYTHONPATH -u ROS_DISTRO uv run --project envs/dppo --no-sync \
  python -m gentle_manip.dppo.convert_demos "$DATA" \
    --out dataset/dppo/$ENV --experiment $EXP --view student --point-cloud

# 2. BC pretrain the PointNet diffusion policy (envs/dppo)
env -u PYTHONPATH -u ROS_DISTRO uv run --project envs/dppo --no-sync \
  python -m gentle_manip.dppo.train --config-path $CFG --config-name pre_diffusion_pointnet
#   -> logs/dppo/dppo-pretrain/$ENV/<exp_id>/checkpoint/state_<epoch>.pt

# 3. harness eval (two processes: sim server in envs/sim  <->  eval agent in envs/dppo)
env -u PYTHONPATH -u ROS_DISTRO MUJOCO_GL=egl uv run --project envs/sim --no-sync \
  python -m gentle_manip.scripts.serl_sim_server --experiment $EXP --view student \
    --num-envs 5 --render-rgb --subprocess --port 5570 &          # background; wait for "SIM_SERVER_READY"
env -u PYTHONPATH -u ROS_DISTRO uv run --project envs/dppo --no-sync \
  python -m gentle_manip.dppo.train --config-path $CFG --config-name eval_diffusion_pointnet \
    base_policy_path=logs/dppo/dppo-pretrain/$ENV/<exp_id>/checkpoint/state_<epoch>.pt
```

## Environments

- **Training + convert + eval-agent** run in **`envs/dppo`** (Python 3.8, DP3 + torch).
- **The eval sim server** runs in **`envs/sim`** (Python 3.12, Genesis) — it renders the point cloud
  the policy consumes. The two talk over a socket (`--port`), bridging the 3.8↔3.12 gap.
- torch is installed manually in each env (see the `envs/*/pyproject.toml` headers); a bare
  `uv sync` removes it.

## Stage 1 — convert (envs/dppo)

`convert_demos` turns the superset demo pkl into `train.npz` / `val.npz` / `normalization.npz`
(split BY TRAJECTORY). Always pass `--experiment` + `--view student` so the obs-key order is derived
from the same config the policy trains on (not hardcoded). `--point-cloud` stores the raw cloud
alongside the flat proprio state.

- Output dir MUST match the DPPO config's `env` (`dataset/dppo/$ENV/`), since the config resolves
  `train_dataset_path = $DPPO_DATA_DIR/$ENV/train.npz`.
- The proprio `obs_dim` is auto-derived from the view's keys: **8** for a quaternion obs
  (`ee_pos3 + ee_quat4 + gripper1`), **10** for a rot6d obs (`ee_pos3 + ee_rot6d6 + gripper1`). The
  DPPO config's `obs_dim` must match (the `_rot6d` configs are already set to 10).

## Stage 2 — pretrain (envs/dppo)

`gentle_manip.dppo.train` is the one launcher for every DPPO stage; the config's `_target_` picks
it (`TrainDiffusionAgent` for pretrain). Useful overrides (hydra, `key=value` on the CLI):

- `train.n_epochs=<N>` — default in the config (e.g. 700); lower for a quick check. **Gotcha:** if you
  drop it below the LR warmup (`train.lr_scheduler.warmup_steps`, default **100**), the cosine scheduler
  asserts (`warmup > cycle`). For a short smoke also pass `train.lr_scheduler.warmup_steps=<small>`.
- `train.save_model_freq=<N>` — checkpoint cadence (→ `state_<epoch>.pt`).
- `wandb=null` (or `WANDB_MODE=offline`) — skip wandb.

Run dir: `logs/dppo/dppo-pretrain/$ENV/<exp_id>/` (5-letter ID, registered in `experiments.csv`);
checkpoints under `checkpoint/state_<epoch>.pt`; the env config is snapshotted into `config/`.

## Stage 3 — harness eval (server in envs/sim + agent in envs/dppo)

Eval routes through the **shared canonical harness** (`run_eval`), so numbers are apples-to-apple
across policies: fixed `EvalSpec` (100 episodes, 5 sub-envs, deterministic per-batch scenario seed),
`summary.json` + `episodes.csv` (per-episode success + stress + DR params) + per-episode videos,
written into the policy's own run dir (`<run>/eval/<datetime>/`).

1. **Start the sim server** (envs/sim), `--num-envs 5 --render-rgb`. Add `--subprocess` if the eval
   uses `scene_group_size>0` (full-DR geometry rebuild); omit it for `scene_group_size=0` (fixed
   geometry). Wait for it to print `SIM_SERVER_READY`. Its `--experiment` MUST be the one the policy
   was trained on (so the camera + obs — including `ee_rot6d` — match).
2. **Run the eval agent** (envs/dppo) with `base_policy_path=<checkpoint>`. It connects to the
   server's `--port` (config default 5570; override with `env.specific.port=<P>`).

Handy eval overrides: `n_episodes=<N>`, `scene_group_size=<K>` (0 = fixed geometry), `record_batches=0`
(disable video for speed).

## Notes

- **Match the eval experiment to training.** A rot6d policy needs a rot6d server experiment
  (`..._rot6d`); a highcam policy needs the highcam camera. The `env`/experiment names in the DPPO
  config already encode this — keep them paired.
- **Reference checkpoint eval** (moxzh, 700-ep quat point-cloud) landed ~31% success on the soft task,
  so use that as the sanity floor when judging a new run — not 100%.
- Full pipeline map + all script staleness flags: `docs/training_and_eval.md`.
