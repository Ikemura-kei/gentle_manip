# Cluster experiments — isolating why `ahaxs` is still the best real policy

## Context (read first)

`ahaxs` (a BC PointNet diffusion policy) is **still the best-performing policy on the real robot**,
despite many later iterations. `ahaxs` was trained on the **`cho`** dataset
(`dataset/demos/single_lift_mushroom_rigid/26-07-29-cho`). Since `cho`, several things changed *at
once* (FEM grasp synthesis, soft MPM sim, rot6d proprio, flip DR, material DR, fewer epochs), so the
regression can't be attributed to any single change. These experiments **change ONE variable at a
time from the `cho`/`ahaxs` recipe** to find what matters.

**The `ahaxs`/`cho` baseline recipe** (hold everything here constant unless the experiment says otherwise):
- physics: **rigid** (`task: single_lift_mushroom_rigid`)
- grasp synthesis: **v1 SDF** (`grasp_synthesis/collect_demos_synth.py`) — still has cho's exact
  constants: FSM `N_HOME_TO_PRE=87, N_SETTLE=1, N_GRASP=39, N_LIFT=70, N_HOLD=12`, **no firm phase**.
- proprio: **quat** (obs_dim 8 = ee_pos3 + ee_quat4 + gripper1)
- action: **absolute** (`abs_pose_abs_gripper`, action_dim 10)
- DR: **`rigid_orientation`** — ±45° pitch/roll, full yaw, xy 0.04, scale/bend/twist/taper/axis.
  **No flip, no material randomization.**
- collection control: 650 episodes, 8 envs, maxfevals 1145, scene_dr_every 1, seed 0.
- training: **2000 epochs**, batch 128, lr 1e-4, ckpt/500, small net (visual_feature_dim 256,
  mlp_dims [512,512,512]). Reference sim success ≈ **0.76** (state ~2000).

Experiment config `single_lift_mushroom_rigid_abs_action` IS the cho recipe (rigid, quat, abs,
`rigid_orientation` DR). DPPO cfg dir: `gentle_manip/dppo/cfg/single_lift_mushroom_rigid_abs_pcd`.

## Prerequisites / conventions

- **Pull first** — this doc assumes the pushed code (new v3 knobs `--n-lift/--n-firm`, the rot6d/soft
  configs referenced below). `git pull` on master.
- Envs: **training/convert/eval-agent** in `envs/dppo`; **collection + eval sim server** in `envs/sim`.
  Prefix genesis/sim commands with `env -u PYTHONPATH -u ROS_DISTRO MUJOCO_GL=egl`.
- `DPPO_DATA_DIR` (converted npz root, `.../dataset/dppo`), `DPPO_LOG_DIR` (`.../logs`) must be set.
- Each collection lands in `dataset/demos/single_lift_mushroom_rigid/<date-id>/`; convert with
  `--out $DPPO_DATA_DIR/single_lift_mushroom_rigid/<id>` and train with `env=single_lift_mushroom_rigid/<id>`.
- **Eval gotchas:** the sim server needs `--subprocess` (all these carry shape DR, scene_group_size 4);
  give each concurrent run its **own `--port`**; pass `env.specific.port=<P>` to the eval agent.
  If the eval's `normalization_path` doesn't resolve to the training dataset, override it explicitly:
  `normalization_path=$DPPO_DATA_DIR/single_lift_mushroom_rigid/<id>/normalization.npz`.
- Register/track each run and fill its EXPERIMENT.md (motivation/hypothesis) per repo convention.

The four experiments are independent — run concurrently, one sim server + port each.

---

## (1) Reproduce `ahaxs` on the EXISTING `cho` dataset

**Goal:** confirm `ahaxs` is reproducible (rules out training nondeterminism / data drift). Also make
an 800-epoch checkpoint since we use 800 as the "best ckpt" elsewhere.

Uses the already-converted `cho` dataset — `env=single_lift_mushroom_rigid/cho`
(`$DPPO_DATA_DIR/single_lift_mushroom_rigid/cho/{train,val,normalization}.npz`). If it isn't on the
cluster, convert it from the raw demos first (see (2)'s convert step, pointing at
`dataset/demos/single_lift_mushroom_rigid/26-07-29-cho/data.pkl`).

```bash
# Run A — reproduce ahaxs exactly (2000 epochs)
env -u PYTHONPATH -u ROS_DISTRO uv run --project envs/dppo python -m gentle_manip.dppo.train \
  --config-path gentle_manip/dppo/cfg/single_lift_mushroom_rigid_abs_pcd --config-name pre_diffusion_pointnet \
  env=single_lift_mushroom_rigid/cho train.n_epochs=2000 train.save_model_freq=500

# Run B — same, 800 epochs
env -u PYTHONPATH -u ROS_DISTRO uv run --project envs/dppo python -m gentle_manip.dppo.train \
  --config-path gentle_manip/dppo/cfg/single_lift_mushroom_rigid_abs_pcd --config-name pre_diffusion_pointnet \
  env=single_lift_mushroom_rigid/cho train.n_epochs=800 train.save_model_freq=200
```
Eval the last ckpt of each (see the shared eval block at the bottom, experiment
`single_lift_mushroom_rigid_abs_action`). **Expect ≈ 0.76** if `ahaxs` reproduces.

---

## (2) Re-collect `cho` (same cfg) and retrain

**Goal:** does *re-collecting* with today's code reproduce `ahaxs`? (Rules out collector drift.) The
current v1 collector still carries cho's exact SDF synthesis + FSM constants, so this is a faithful redo.

```bash
# collect (envs/sim) — v1 SDF, cho recipe
env -u PYTHONPATH -u ROS_DISTRO MUJOCO_GL=egl uv run --project envs/sim python \
  grasp_synthesis/collect_demos_synth.py --experiment single_lift_mushroom_rigid_abs_action \
  --n-episodes 650 --n-envs 8 --maxfevals 1145 --seed 0 --scene-dr-every 1 --record-video 50
# -> dataset/demos/single_lift_mushroom_rigid/<id>/   (note the <id>)

# convert (envs/dppo) — quat student, obs_dim 8
env -u PYTHONPATH -u ROS_DISTRO uv run --project envs/dppo python -m gentle_manip.dppo.convert_demos \
  dataset/demos/single_lift_mushroom_rigid/<id>/data.pkl \
  --out $DPPO_DATA_DIR/single_lift_mushroom_rigid/<id> \
  --experiment single_lift_mushroom_rigid_abs_action --view student --point-cloud

# train — ahaxs params (2000 epochs)
env -u PYTHONPATH -u ROS_DISTRO uv run --project envs/dppo python -m gentle_manip.dppo.train \
  --config-path gentle_manip/dppo/cfg/single_lift_mushroom_rigid_abs_pcd --config-name pre_diffusion_pointnet \
  env=single_lift_mushroom_rigid/<id> train.n_epochs=2000 train.save_model_freq=500
```
Eval with experiment `single_lift_mushroom_rigid_abs_action`.

---

## (3) rot6d proprio on `cho` (conversion done LOCALLY — cluster does DPPO convert + train + eval)

**Goal:** does rot6d EE orientation help/hurt vs quat, on the otherwise-identical `cho` data?

The raw demos are already rot6d-augmented locally (`ee_rot6d` derived from the recorded `ee_quat`,
`ee_quat` kept) and rsynced to `dataset/demos/single_lift_mushroom_rigid/<ID3>/` (see the rsync
command handed over separately; `<ID3>` = the dir name it lands in). Everything else is cho.

```bash
# DPPO convert — rot6d student (obs_dim 10). ee_rot6d is already in the demo obs; select it explicitly.
env -u PYTHONPATH -u ROS_DISTRO uv run --project envs/dppo python -m gentle_manip.dppo.convert_demos \
  dataset/demos/single_lift_mushroom_rigid/<ID3>/data.pkl \
  --out $DPPO_DATA_DIR/single_lift_mushroom_rigid/<ID3> \
  --obs-keys ee_pos ee_rot6d gripper_width --point-cloud
# (verify it prints obs_dim 10)

# train — rot6d DPPO cfg, ahaxs params
env -u PYTHONPATH -u ROS_DISTRO uv run --project envs/dppo python -m gentle_manip.dppo.train \
  --config-path gentle_manip/dppo/cfg/single_lift_mushroom_rigid_abs_pcd_rot6d --config-name pre_diffusion_pointnet \
  env=single_lift_mushroom_rigid/<ID3> train.n_epochs=2000 train.save_model_freq=500
```
Eval uses the **rot6d** cfg + experiment (obs must match): server `--experiment
single_lift_mushroom_rigid_abs_action_rot6d`, eval `--config-path .../single_lift_mushroom_rigid_abs_pcd_rot6d`,
`env_name=single_lift_mushroom_rigid/<ID3>`.

---

## (4) Soft sim, otherwise `cho` (soft physics is the ONLY change)

**Goal:** does switching rigid→soft MPM (same DR, same SDF synthesis, same FSM, quat) hurt sim2real?

Uses the new experiment `single_lift_mushroom_soft_abs_action_chomatch` (soft task, `rigid_orientation`
DR = no flip/no material, quat student obs_dim 8 — identical proprio to cho, so training reuses the
rigid quat DPPO cfg). Collect with the **v1 SDF** collector at cho's FSM.

```bash
# collect (envs/sim) — soft sim, v1 SDF, cho FSM constants
env -u PYTHONPATH -u ROS_DISTRO MUJOCO_GL=egl uv run --project envs/sim python \
  grasp_synthesis/collect_demos_synth.py --experiment single_lift_mushroom_soft_abs_action_chomatch \
  --n-episodes 650 --n-envs 8 --maxfevals 1145 --seed 0 --scene-dr-every 1 --record-video 50
#   NOTE: v1 on a SOFT task is less battle-tested than on rigid — do a tiny --n-episodes 8 smoke
#   first and confirm episodes save + grasps look sane before the full run. A soft object dropped
#   from height blows up MPM, so it must spawn resting; the soft task cfg handles that.

# convert — quat student (same as cho)
env -u PYTHONPATH -u ROS_DISTRO uv run --project envs/dppo python -m gentle_manip.dppo.convert_demos \
  dataset/demos/single_lift_mushroom_soft/<id>/data.pkl \
  --out $DPPO_DATA_DIR/single_lift_mushroom_soft/<id> \
  --experiment single_lift_mushroom_soft_abs_action_chomatch --view student --point-cloud

# train — reuse the rigid quat DPPO cfg (obs_dim 8), override env + experiment snapshot
env -u PYTHONPATH -u ROS_DISTRO uv run --project envs/dppo python -m gentle_manip.dppo.train \
  --config-path gentle_manip/dppo/cfg/single_lift_mushroom_rigid_abs_pcd --config-name pre_diffusion_pointnet \
  env=single_lift_mushroom_soft/<id> experiment=single_lift_mushroom_soft_abs_action_chomatch \
  train.n_epochs=2000 train.save_model_freq=500
```
Eval: server `--experiment single_lift_mushroom_soft_abs_action_chomatch` (soft is slow — this is the
long pole); eval agent with `experiment=single_lift_mushroom_soft_abs_action_chomatch` and the matching
`env`/`normalization_path`.

---

## Shared eval block (rigid quat experiments (1),(2))

```bash
PORT=<pick a free port per run>
# sim server (envs/sim) — leave running
env -u PYTHONPATH -u ROS_DISTRO MUJOCO_GL=egl uv run --project envs/sim python \
  -m gentle_manip.scripts.serl_sim_server --experiment single_lift_mushroom_rigid_abs_action \
  --view student --num-envs 5 --render-rgb --subprocess --port $PORT   # wait for SIM_SERVER_READY
# eval agent (envs/dppo)
env -u PYTHONPATH -u ROS_DISTRO uv run --project envs/dppo python -m gentle_manip.dppo.train \
  --config-path gentle_manip/dppo/cfg/single_lift_mushroom_rigid_abs_pcd --config-name eval_diffusion_pointnet \
  env=single_lift_mushroom_rigid/<id> base_policy_path=<.../checkpoint/state_2000.pt> \
  normalization_path=$DPPO_DATA_DIR/single_lift_mushroom_rigid/<id>/normalization.npz \
  env.specific.port=$PORT
```
Results land in `<run>/eval/<datetime>/summary.json` (success_rate) + per-episode videos.

## What to report back
Per experiment: sim `success_rate` of the last (and 800/2000) ckpt vs the **0.76** `ahaxs` baseline,
plus the collection `success_rate`. The headline question each answers:
(1) reproducible? (2) collection-stable? (3) rot6d help/hurt? (4) does soft physics cost sim2real?
