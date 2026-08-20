# Cluster experiment: action-space ablation on REAL single_lift_mushroom (DPPO vs DP3 × abs vs delta)

**Goal:** which action space + which codebase works best on the real 55-demo set? Four BC runs, IDENTICAL
data + obs, only the action representation and the algorithm differ:

| # | run | algorithm | action |
|---|-----|-----------|--------|
| 1a | **DPPO-abs**   | DPPO PointNet diffusion | 7d absolute (3 pos + 3 euler + 1 gripper) |
| 1b | **DPPO-delta** | DPPO PointNet diffusion | 7d delta (dx,dy,dz,droll,dpitch,dyaw,dgrip) |
| 2a | **DP3-abs**    | DP3 (3D Diffusion Policy) | 7d absolute (same euler encoding) |
| 2b | **DP3-delta**  | DP3 | 7d delta |

All: **6000 epochs, checkpoint every 500**, wandb online. Obs is FIXED across all four: **quat proprio
(obs_dim 8)** + **arm-focus 1024-pt cloud** — collected that way, already in the demos.

## What's shared / pre-processed — nothing extra needed
The obs (arm-focus cloud + quat proprio) is already baked into the recorded demos. Both action
representations are **derived from the same recorded EE-pose trajectory at convert time** (target for
step t = next observed pose), via `--derive-action` — and BOTH converters (DPPO `convert_demos`, DP3
`convert_demo_to_dp3`) now support it, using one shared `gentle_manip.actions.derive`. So the delta and
absolute datasets come from the *identical* demos (clean apples-to-apples), and there is **no shared
pre-processing to do before rsync — just ship the raw merged pkl**.

## Prerequisites
- `git pull` on master (uses the committed 7d-euler action, `invert_delta`, `--derive-action` on both
  converters, DP3 task configs, and `train_dp3.sh` extra-override forwarding).
- Do NOT export `DPPO_LOG_DIR`/`DPPO_DATA_DIR` (launcher defaults to `logs/dppo`, `dataset/dppo`).
- Prefix conversion with `env -u PYTHONPATH -u ROS_DISTRO` (DP3 conversion needs `--project envs/dp3`).

---

## STEP 1 — rsync the RAW merged demo pkl (55 eps: cmh + smoke)
```bash
rsync -avhP dataset/demos/single_lift_mushroom_real_merged \
  ikemura@arrhenius1.hpc.arrhenius.naiss.se:/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip/dataset/demos/
```
`data.pkl` is all that's needed (the videos/analysis PNG can be skipped).

## STEP 2 — convert (4 datasets, both action reps × both algorithms), from the ONE raw pkl
```bash
PKL=dataset/demos/single_lift_mushroom_real_merged/data.pkl
DELTA=gentle_manip/configs/action/delta_pose_delta_gripper_fast_rot.yaml   # 7d delta
ABS=gentle_manip/configs/action/abs_pose_euler_abs_gripper.yaml            # 7d euler absolute

# --- DPPO (npz) ---
for R in delta abs; do
  CFG=$([ $R = delta ] && echo $DELTA || echo $ABS)
  env -u PYTHONPATH -u ROS_DISTRO uv run --project envs/dppo python -m gentle_manip.dppo.convert_demos "$PKL" \
    --out $DPPO_DATA_DIR/single_lift_mushroom_real_$R --obs-keys ee_pos ee_quat gripper_width --point-cloud \
    --derive-action $CFG                       # verify meta: obs_dim 8, action_dim 7
done

# --- DP3 (zarr; path must match the task config: <DP3_DIR>/data/single_lift_mushroom_real_<R>.zarr) ---
DP3_DIR=third_party/DP3/3D-Diffusion-Policy
for R in delta abs; do
  CFG=$([ $R = delta ] && echo $DELTA || echo $ABS)
  env -u PYTHONPATH -u ROS_DISTRO uv run --project envs/dp3 python gentle_manip/scripts/convert_demo_to_dp3.py "$PKL" \
    -o $DP3_DIR/data/single_lift_mushroom_real_$R.zarr --overwrite --derive-action $CFG   # action (T,7), state (T,8)
done
```

## STEP 3 — train the four runs (6000 ep, save/500)

### DPPO (config `single_lift_mushroom_real` is already obs_dim 8 / action_dim 7 / 6000 ep / save 500)
Same config for abs & delta — only the dataset (`env`) differs (the 7d width is identical, semantics live
in the action config used at deploy). wandb project = `gentle-manip-single_lift_mushroom_real_<R>`.
```bash
# hydra needs an ABSOLUTE --config-path (it resolves relative to the dppo run.py, not the cwd)
CFG="--config-path $PWD/gentle_manip/dppo/cfg/single_lift_mushroom_real --config-name pre_diffusion_pointnet"
for R in abs delta; do
  env -u PYTHONPATH -u ROS_DISTRO uv run --project envs/dppo python -m gentle_manip.dppo.train $CFG \
    env=single_lift_mushroom_real_$R experiment=single_lift_mushroom_real
done
# -> logs/dppo/dppo-pretrain/single_lift_mushroom_real_<R>/<exp_id>/checkpoint/state_{500,1000,...,6000}.pt
```

### DP3 (train_dp3.sh dp3 <task> <info> <seed> <gpu> [extra hydra overrides])
The task config points at the zarr; forward the 6000-ep / save-500 overrides (train_dp3.sh now passes
extra args through). `env_runner: null` in the task config = no sim rollout (real BC).
```bash
for R in abs delta; do
  bash gentle_manip/scripts/train_dp3.sh dp3 single_lift_mushroom_real_$R realabl 42 0 \
    training.num_epochs=6000 training.checkpoint_every=500 checkpoint.save_ckpt=True
done
# -> third_party/DP3/3D-Diffusion-Policy/data/outputs/single_lift_mushroom_real_<R>-dp3-realabl_seed42/checkpoints/
```

## Notes
- **Action reps:** delta = `delta_pose_delta_gripper_fast_rot.yaml` (the teleop scales the demos were
  collected with — lossless to derive). Absolute = `abs_pose_euler_abs_gripper.yaml` (7d: 3 pos + 3 euler
  + 1 grip). Both are 7-dim, so DPPO/DP3 model widths match across arms.
- **Data quality:** 55 clean grasp-and-lift demos, all lifted >5 cm; grasp width ~3 cm; grasps span the
  workspace with yaw variety (see `grasp_analysis.png`).
- **Seeds:** the table above is 1 seed each. For the 3-seed version, loop `seed`/`training.seed` over
  {42, 43, 44} — DPPO mints a fresh `exp_id` per run; DP3 encodes the seed in the run dir.
- **Deploy caveat (not blocking training):** deploying the *absolute-euler* arm needs a small fix to the
  deploy loop's absolute warmup (`_current_raw_pose` assumes rot6d/9 pose dims; euler is 6) — handle when
  you deploy the winner, not for training.
