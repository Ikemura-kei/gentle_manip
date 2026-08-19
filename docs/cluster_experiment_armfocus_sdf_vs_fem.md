# Cluster experiment: arm-focus point cloud — SDF vs FEM(+2mm) on the cho rigid backbone

**Goal:** does the new **arm-focus point cloud** (object-dense sampling) + gentle **FEM** grasps get us
gentle *and* reliable, vs the proven **SDF** (cho) recipe? Two datasets, identical except the grasp
synthesizer, both with the arm-focus cloud; train each like ahaxs; eval a checkpoint sweep.

## Baseline & what's held constant
- Backbone = **cho/ahaxs**: rigid sim, **quat proprio (obs_dim 8)**, abs action (action_dim 10),
  DR `rigid_orientation` (±45° pitch/roll, full yaw, xy 0.04, **no flip, no material**), FSM
  `N_HOME_TO_PRE=87 / N_SETTLE=1 / N_GRASP=39 / N_LIFT=70 / N_HOLD=12, no firm`, 650 demos, 8 envs,
  maxfevals 1145, scene_dr_every 1, seed 0. Training: **2000 epochs, save_model_freq 400**, small net.
- ahaxs sim reference ≈ **0.76** (state_800, SDF, no arm-focus).
- **NEW common element (both datasets):** obs `superset_rigid_armfocus` — arm-body points sampled at
  0.15×, object + finger-safe near-EE **sphere** at full weight (see `docs/` note / `focus_weights`).
- Experiment config for BOTH: **`single_lift_mushroom_rigid_abs_action_armfocus`** (already committed).

The ONLY difference between the two datasets is the grasp synthesis.

## Prerequisites (same as every cluster run here)
- `git pull` on master first (uses the committed arm-focus code + configs).
- Do **NOT** export `DPPO_LOG_DIR`/`DPPO_DATA_DIR` (launcher defaults to `logs/dppo`, `dataset/dppo`);
  if you must, `DPPO_LOG_DIR` MUST end in `logs/dppo`.
- **`wandb=null`** on every train (env has a `/` subdir → invalid wandb project otherwise).
- Eval needs the sim server with **`--subprocess`** (rigid_orientation carries shape DR → scene_group_size 4).
- Prefix sim/collect commands with `env -u PYTHONPATH -u ROS_DISTRO MUJOCO_GL=egl`.

---

## STEP 1 — Collect the two datasets (both use experiment `single_lift_mushroom_rigid_abs_action_armfocus`)

### 1a. SDF dataset (same as cho, + arm-focus cloud) — v1 collector
```bash
env -u PYTHONPATH -u ROS_DISTRO MUJOCO_GL=egl uv run --project envs/sim python \
  grasp_synthesis/collect_demos_synth.py --experiment single_lift_mushroom_rigid_abs_action_armfocus \
  --n-episodes 650 --n-envs 8 --maxfevals 1145 --seed 0 --scene-dr-every 1 --record-video 50
# v1 carries cho's exact SDF synthesis + FSM (87/1/39/70/12, no firm). -> dataset/demos/single_lift_mushroom_rigid/<idA>/
```

### 1b. FEM+2mm dataset (still gentle) — v3 collector, cho FSM + 2mm extra squeeze
```bash
env -u PYTHONPATH -u ROS_DISTRO MUJOCO_GL=egl uv run --project envs/sim python \
  grasp_synthesis/collect_demos_synth_v3.py --experiment single_lift_mushroom_rigid_abs_action_armfocus \
  --n-episodes 650 --n-envs 8 --maxfevals 1145 --grasp-gpu --seed 0 --scene-dr-every 1 \
  --n-home-to-pre 87 --n-grasp 39 --n-lift 70 --n-firm 0 --grasp-extra-close 0.002 \
  --record-video 50
# 2mm (not 4/8) keeps FEM gentle — 4mm/8mm made scripted FEM harsher than SDF. -> dataset/demos/single_lift_mushroom_rigid/<idB>/
```
(Note the two `<idA>`/`<idB>` dir names printed as "Data →"; they name the DPPO env dirs below.)

## STEP 2 — Convert each (quat student, obs_dim 8)
```bash
for ID in <idA> <idB>; do
  env -u PYTHONPATH -u ROS_DISTRO uv run --project envs/dppo python -m gentle_manip.dppo.convert_demos \
    dataset/demos/single_lift_mushroom_rigid/$ID/data.pkl \
    --out $DPPO_DATA_DIR/single_lift_mushroom_rigid/$ID \
    --experiment single_lift_mushroom_rigid_abs_action_armfocus --view student --point-cloud
done   # verify each prints obs_dim 8
```

## STEP 3 — Train each like ahaxs (2000 epochs, save/400)
```bash
for ID in <idA> <idB>; do
  env -u PYTHONPATH -u ROS_DISTRO uv run --project envs/dppo python -m gentle_manip.dppo.train \
    --config-path gentle_manip/dppo/cfg/single_lift_mushroom_rigid_abs_pcd --config-name pre_diffusion_pointnet \
    env=single_lift_mushroom_rigid/$ID \
    experiment=single_lift_mushroom_rigid_abs_action_armfocus \
    train.n_epochs=2000 train.save_model_freq=400 wandb=null
done   # -> logs/dppo/dppo-pretrain/single_lift_mushroom_rigid/$ID/<exp_id>/checkpoint/state_{400,800,1200,1600,2000}.pt
```

## STEP 4 — Eval sweep, EACH CHECKPOINT AS SOON AS IT APPEARS
Evaluate **state_400, state_800, state_1200, state_2000** — do NOT wait for training to finish; kick
off each eval the moment its checkpoint is written (schedule as you see fit — one shared sim server
per dataset, reuse it across the four checkpoints).

Sim server (leave running per dataset; the experiment MUST be the arm-focus one so the eval cloud is
sampled the SAME way as training):
```bash
env -u PYTHONPATH -u ROS_DISTRO MUJOCO_GL=egl uv run --project envs/sim python \
  -m gentle_manip.scripts.serl_sim_server --experiment single_lift_mushroom_rigid_abs_action_armfocus \
  --view student --num-envs 5 --render-rgb --subprocess --port <P>          # wait for SIM_SERVER_READY
```
Eval agent, per checkpoint (override the experiment + normalization to the arm-focus dataset):
```bash
env -u PYTHONPATH -u ROS_DISTRO uv run --project envs/dppo python -m gentle_manip.dppo.train \
  --config-path gentle_manip/dppo/cfg/single_lift_mushroom_rigid_abs_pcd --config-name eval_diffusion_pointnet \
  experiment=single_lift_mushroom_rigid_abs_action_armfocus \
  base_policy_path=<run>/checkpoint/state_<N>.pt \
  normalization_path=$DPPO_DATA_DIR/single_lift_mushroom_rigid/$ID/normalization.npz \
  env.specific.port=<P>
```
Results: `<run>/eval/<datetime>/summary.json` (success_rate + the now-recorded `git_commit`) + videos.

## What to report
For BOTH datasets, the sweep success_rate at **400 / 800 / 1200 / 2000**, vs ahaxs **0.76**:
- **SDF + arm-focus** — does the object-dense cloud lift SDF *above* 0.76 (arm-focus helps)?
- **FEM+2mm + arm-focus** — does gentle FEM, with the better cloud, match/beat SDF here (the goal:
  gentle AND reliable)?
- Best checkpoint is usually early (ahaxs deployed state_800; policies here have overfit past ~800–1200),
  so 400/800 matter most; 2000 mainly confirms the overfit tail.

## Notes / gotchas
- The arm-focus obs (`superset_rigid_armfocus`) MUST be used for BOTH collection AND eval (the server
  `--experiment` above) — a policy trained on the focused cloud will fail on an unfocused one.
- Datasets land in `dataset/demos/single_lift_mushroom_rigid/` — distinct dated dirs; keep the two ids straight.
- FEM run: verify a quick smoke (`--n-episodes 8`) saves demos before the full 650 (v1/v3 on rigid is fine;
  this is just prudent). SDF (v1) is the cho path, well-tested.
