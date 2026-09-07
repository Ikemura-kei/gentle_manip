# DPPO / DP3 training — from collected demos to an evaluated checkpoint (recipe v5, 2026-09-07)

The generic pipeline for a PointNet diffusion policy (DP3-style, trained with the DPPO fork's BC
pretrainer) on demos from the frozen collector v4.2 (`docs/final/grasp_synthesis_final.md`). Every
command below is the one used for training rounds 1–4 (2026-09-06/07, tofu); only the recipe knobs and
the epoch rule changed, and those now live in a single place, the anchor script. Cluster job wrapping
is out of scope here (the cluster agent has its own tooling): each step is a plain shell command that
runs from the repo root.

Environments: conversion/training/eval-client run in `envs/dppo` (Python 3.8, torch); the sim server for
eval runs in `envs/sim` (3.12, genesis). Always `uv run --project envs/<name>`, never `--directory`.

## 0. What goes in

| input | where | notes |
|---|---|---|
| demos | `dataset/demos/<task>/<run>/data.pkl` (+ `config/`, `config.yaml`, `stats.yaml`, `dr_params.csv`) | one run dir per collector job; `stats.yaml` has success / tiers / sub-yield |
| experiment config | `gentle_manip/configs/experiments/<name>.yaml` | the one the demos were collected with (it is in the run's `config/`) |
| action yamls | `gentle_manip/configs/action/abs_pose_abs_gripper_z15.yaml` (recorded 10-D rot6d) and `abs_pose_euler_abs_gripper_z15.yaml` (trained 7-D euler) | both `_z15`; see the warning in §1 |
| paired real–sim file | `dataset/dppo/paired/paired_red_cube_play_2026-09-05.npz` | object-agnostic regularizer input, built once (§2); copy it to the training machine as is |
| training cfg | `gentle_manip/dppo/cfg/sim2real_v1/pre_diffusion_pointnet.yaml` | object-agnostic; never fork it, override through the anchor script |
| eval cfg | `gentle_manip/dppo/cfg/sim2real_v1/eval_diffusion_pointnet.yaml` | reads dataset + experiment from the checkpoint's run |

Data-quality check before converting (per run): `stats.yaml` success rate, and `synth_tiers` (tier 2 = yield
gate off, tier 3 = seed fallback). Cherry tomato had a third of its demos at tier 2; decide whether such
runs go in before merging, because the converter takes every `data.pkl` under the dir you give it.

## 1. Convert demos to a DPPO dataset

One call per object (or per group of runs of the same object). This is the round-1..4 command verbatim
(saved by the converter itself as `<out>/launch_command.sh`):

```bash
uv run --project envs/dppo python -m gentle_manip.dppo.convert_demos \
    dataset/demos/single_lift_tofu_soft/26-09-05-jvt \
    --out dataset/dppo/single_lift_tofu_sim2real_v1 \
    --experiment single_lift_tofu_soft_abs_action_armfocus_7d_realws --view student \
    --point-cloud point_cloud \
    --derive-action        gentle_manip/configs/action/abs_pose_euler_abs_gripper_z15.yaml \
    --derive-source-action gentle_manip/configs/action/abs_pose_abs_gripper_z15.yaml \
    --val-split 0.1
```

- The positional argument is a run dir or a parent dir: the converter recurses over every `data.pkl`
  below it. Put only the runs you want there, or convert run dirs one at a time and merge (§1.1).
- `--experiment … --view student` fixes the proprio order (ee_pos, ee_quat, gripper_width = 8-D) and
  `--point-cloud` stores the raw 1024-point cloud beside it. Clouds are stored CLEAN; all noise is
  train-time (§3).
- `--derive-source-action` MUST be the action yaml the demos were ENCODED with (`abs_pose_abs_gripper_z15`
  for every v4.2 collection). Decoding with the non-`z15` yaml shifts targets 15–25 mm and the policy
  grasps beside the object (run `covel`, 0/20). Sanity check if in doubt: decode a stored action through
  the source yaml and compare with `ee_pos` four steps later; the median gap must be a few mm.
- Output: `train.npz`, `val.npz` (10 % of episodes, split by trajectory), `normalization.npz`,
  `sources.yaml` (provenance: path, episodes, experiment, task, commit per source), `launch_command.sh`.
- No hold-tail post-processing. v4.2 records a 10-step hold natively; `augment_hold_tail.py` was only
  for pre-freeze demos (rounds 2–4's `_tail60/_tail20/_tail10` datasets).

### 1.1 Mixed datasets (several objects, or sim + real)

Convert each source separately with its own experiment, then merge at the npz level. The merge
de-normalizes each source with its own stats, concatenates, and re-normalizes jointly; clouds concatenate
as is; no oversampling. Its `sources.yaml` is the union of the inputs' manifests.

```bash
uv run --project envs/dppo python -m gentle_manip.dppo.merge_npz_datasets \
    dataset/dppo/<set_a> dataset/dppo/<set_b> [...] --out dataset/dppo/<combined>
```

All sources must share the action space (7-D euler, `_z15` bounds) and proprio view. A merged set has no
single experiment: pass the main object's experiment to training (it is provenance and the default
eval target, nothing in the training loop reads it) and evaluate other objects with `experiment=…`
on the eval command line. The training run copies the dataset's `sources.yaml` into
`<run>/config/dataset_sources.yaml`.

## 2. Paired real–sim file (already built, reused as is)

Input to the paired-feature encoder term (`PAIRED_W`). Built once from real spacemouse play data on the
red cube and its step-for-step sim twin; independent of the training object.

```bash
uv run --project envs/dppo python -m gentle_manip.dppo.build_paired_npz \
    --real dataset/demos/play_red_cube_real/26-09-05-xiv --sim dataset/demos/play_red_cube_soft/26-09-05-xiv \
    --out dataset/dppo/paired/paired_red_cube_play_2026-09-05.npz --stride 2
```

`--stride 2` keeps every second step (clouds are redundant frame to frame): 1031 pairs.

**On the cluster (2026-09-07):** the raw play runs and the built npz are in
`/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip/dataset/transfer/paired_simreal/`
(`demos/play_red_cube_{real,soft}/26-09-05-xiv`, `dppo/paired/paired_red_cube_play_2026-09-05.npz`, `MANIFEST.md`).
Install repo-relative so the training cfg's `${DPPO_DATA_DIR}/paired/…` path resolves:
```bash
B=dataset/transfer/paired_simreal
rsync -a $B/demos/ dataset/demos/ && rsync -a $B/dppo/ dataset/dppo/
```
No rebuild is needed; the command above is only for regenerating the npz from the raw runs. The play data
itself came from `gentle_manip/scripts/final/collect_play_data.sh` (real) and `gen_sim_play_data.sh`
(sim twin). The training cfg points at this file through `${DPPO_DATA_DIR}/paired/…`, so on another
machine it must sit at `dataset/dppo/paired/` under the repo.

## 3. Train

**Anchor script: [`gentle_manip/scripts/final/train_dppo_dp3.sh`](../../gentle_manip/scripts/final/train_dppo_dp3.sh).**
It pins the cfg dir, holds the one recipe, and exposes only the per-run knobs as env vars. Extra hydra
overrides pass through as arguments.

```bash
DATASET=<dataset dir name under dataset/dppo> \
EXPERIMENT=<experiment name> \
bash gentle_manip/scripts/final/train_dppo_dp3.sh            # add wandb=null to disable wandb
```

Recipe v5 (the script's defaults; the user's decision of 2026-09-07 for the cluster-campaign data):

| knob | value | what it does |
|---|---|---|
| `PC_AUG` | `d435i_noise_train` | BC path, every sample: measured D435i stereo noise (axial 2e-3 m/m², lateral 5e-4) + 3 % dropout + leaked table-residue clusters at p = 0.15 |
| `CLOUD_OFFSET` | `[0.005,0.0035,0.0015]` | BC path: per-sample rigid cloud offset, per axis U(±5, ±3.5, ±1.5 mm); proprio untouched |
| `PAIRED_W` | 0.5 | real-vs-sim paired encoder term on the §2 file; sim twins noised like the BC path |
| `CONS_W` / `CONS_FRAC` | 0.3 / 0.3 | clean-vs-perturbed encoder consistency on 30 % of each batch, cosine distance, stop-grad on the clean side |
| `CONS_AUG` | `d435i_noise_strong` | the perturbed view: noise ×1.5, 10 % dropout, one ≤1 cm occlusion patch above z = 10 cm, residue at p = 0.30 |
| `CONS_OFFSET` | 0 | no offset in the consistency view (feature-level shift invariance blinds a PointNet to position) |
| `SEED` | 42 | |

Fixed in the cfg: batch 128, lr 1e-4 with one cosine cycle over the run (100 warm-up epochs, min 1e-5), EMA
0.995, checkpoints every 250 epochs, val every 10, network = PointNet(512) + MLP [1024,1024,1024], horizon 4,
cond 2, 20 denoising steps.

### 3.1 Epoch rule

Epochs are derived from a target gradient-step count so every dataset size trains for the same number of
updates:

```
TARGET_STEPS = 380,000  (batch 128)
batches_per_epoch = ceil(N_train_steps / 128)      # N_train_steps = sum of train.npz traj_lengths
EPOCHS = ceil(TARGET_STEPS / batches_per_epoch)
```

`EPOCHS=auto` (the default) computes this from the dataset's `train.npz` and prints it before launch.
Override with `EPOCHS=<n>` or `TARGET_STEPS=<n>`. Reference points:

| dataset | N_train_steps | batches/epoch | epochs at 380k |
|---|---|---|---|
| round 4, 100 tofu demos (`_tail10`) | 17,803 | 140 | 2715 (round 4 itself ran 2000 = 280k steps) |
| 50k train steps | 50,000 | 391 | 972 |
| 500k train steps | 500,000 | 3907 | 98 |

The cosine schedule's cycle length follows `n_epochs`, so the LR decays over the whole run whatever the
epoch count. Checkpoints land every 250 epochs; for short runs (< 1000 epochs) pass `train.save_model_freq=<n>`
so at least four are kept. Val loss is not predictive of on-robot success (documented on `xagzg`); keep
several checkpoints and sweep them.

### 3.2 What a run produces

`logs/dppo/dppo-pretrain/<DATASET>/<id>/` with a 5-letter `<id>` (also the wandb run name and the row in
`experiments.csv`): `checkpoint/state_<epoch>.pt`, `config/` (experiment snapshot + `dataset_sources.yaml`),
`launch_command.sh`, `run.log`, `EXPERIMENT.md` (fill motivation / hypothesis / final summary). Wandb project
`gentle-manip-<DATASET>` with per-component losses (`train/loss` = BC, `train/loss_paired`,
`train/loss_consistency`, `val/…`). Set `DPPO_WANDB_ENTITY` or append `wandb=null`. Do not export
`DPPO_LOG_DIR` / `DPPO_DATA_DIR`; the launcher sets them to the repo's `logs/dppo` and `dataset/dppo`.

Startup check: the console must show `[pc_aug] train-time cloud noise d435i_noise_train: axial 0.002 … | rigid
offset …`; without that line the run trained on clean clouds. Round-4 speed on a 4090: ~6.5 s/epoch at 140 batches (≈ 2.1 ms/step
… ~3.7 h per 2000 epochs); a shared GPU (orphaned Genesis children) doubled it.

## 4. Evaluate (sim, canonical harness)

Two processes: a sim server (`envs/sim`) and the eval client (`envs/dppo`). The server's experiment must
be the one the checkpoint was trained on (or the object you want to test, with `experiment=…` on the
client). `--augmentation d435i_noise` = sensor noise only, the fair default for every policy; a robustness
eval opts in with `--augmentation d435i_noise_train` (residue clusters injected, flagged per episode in
`signals/`). Never evaluate with train-time-only augmentation by default.

```bash
# 1) sim server (leave running; one server per eval; own --port when running several)
setsid uv run --project envs/sim python -m gentle_manip.scripts.serl_sim_server \
    --experiment <experiment> --view student --num-envs 5 --render-rgb --subprocess --port 5570 \
    --augmentation d435i_noise > sim_server.log 2>&1 &
sleep 60
# 2) eval client: canonical = 200 episodes (the cfg default); teaser = n_episodes=20
uv run --project envs/dppo python -m gentle_manip.dppo.train \
    --config-path $(pwd)/gentle_manip/dppo/cfg/sim2real_v1 --config-name eval_diffusion_pointnet \
    base_policy_path=logs/dppo/dppo-pretrain/<DATASET>/<id>/checkpoint/state_<epoch>.pt \
    [n_episodes=20] [experiment=<other object's experiment>]
# 3) stop the server by its process group (a plain kill leaves genesis spawn_main children on the GPU)
kill -- -<pgid>; pkill -f "serl_sim_server"
```

Output under the checkpoint's run: `<run>/eval/<datetime>/` with `summary.json` (success_rate,
ever_success_rate, checkpoint), `episodes.csv` (per-episode success, DR params, stress), `render/` (one
clip per episode), `signals/` (per-episode command-vs-state plots + residue flag), `config/`.
`scene_group_size: 4` rebuilds the geometry every 4 batches, which is why the server needs `--subprocess`.
A 20-episode teaser cannot rank recipes (round-4 checkpoints spread 3–6/20 within one run); use the
200-episode canonical eval on the two or three candidate checkpoints, and decide on the robot.

## 5. Deploy on the robot

Add a checkpoint entry to `gentle_manip/scripts/final/deploy_dppo.sh` (pattern of the existing entries:
run id + checkpoint + the real rig obs config with `ground_residual` on) and run through
`pre_deploy_check.sh`; visualize recordings with `viz_deploy.sh`. Details in `docs/DEVLOG.md`, 2026-09-06/07.

## 6. Where the earlier rounds and their reasoning live

`docs/training_plan_sim2real_2026-09.md` (round-by-round history table at the end), `docs/DEVLOG.md`
(2026-09-05 … 09-07 entries), each run's `EXPERIMENT.md`.
