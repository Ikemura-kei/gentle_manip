# Generalist v5, first large-dataset run (`bqvzh`, local, 2026-09-07) — setup and findings

The first training of the confirmed recipe v5 (`docs/final/dppo_dp3_training.md`) on the cluster campaign data, run
locally on the lab 4090 as a twin of the cluster's queued generalist job. This page records exactly what was trained,
how long it took, what the sim teasers showed, and what we learned about the recipe at this data scale.
Run dir: `logs/dppo/dppo-pretrain/single_lift_generalist_soft_v5/bqvzh/` (EXPERIMENT.md, `config/`, `chain_launch.sh`,
`config/dataset_manifest.txt`, `eval/`). Wandb: `gentle_manip_generalist/runs/67qce45j`.

## 1. Data

| source | runs | episodes |
|---|---|---|
| cluster list `dataset/demos/TRAINING_RUNS_2026-09-07.txt` (frozen collector v4.2; tofu, mushroom, banana_chunk, pasta_bundle, cherry_tomato, merged tomato 420 + strawberry 75, prim_{cuboid,cylinder,ellipsoid,sphere,torus}_mush, 18 `bs_*` basic shapes) | 31 | 5,870 |
| local: cherry_tomato `26-09-06-xxg` (35, consolidated from shards) + `26-09-07-kfx` (215); prim_capsule/hexprism/frustum_mush (50 each) | 5 | 400 |
| **total** | **36** | **6,270** |

Downloaded without videos (15 GB, `rsync --files-from` the list). Staged as hardlinked `data.pkl` files under
`dataset/demos_staging/generalist_v5/<task>/<run>/` so that (a) the merged runs' `parts/*.pkl` shards are not double-counted
and (b) the shard-only partial run is not silently dropped — the converter recurses on `data.pkl` and otherwise falls back to
every pkl. Per-run episode counts: `config/dataset_manifest.txt` in the run dir.

Conversion (one call; the same command as rounds 1–4, verbatim in `dataset/dppo/single_lift_generalist_soft_v5/launch_command.sh`):
tofu experiment, student view (proprio 8-D + 1024-point cloud), `--derive-action abs_pose_euler_abs_gripper_z15` with
`--derive-source-action abs_pose_abs_gripper_z15`, val split 0.1 by trajectory. Result: **1,193,991 steps, 5,643 train /
627 val trajectories**, `train.npz` 11.7 GB + `val.npz` 1.3 GB. Provenance in `sources.yaml` (36 entries).

## 2. Training setup

Anchor script `gentle_manip/scripts/final/train_dppo_dp3.sh`, cfg `gentle_manip/dppo/cfg/sim2real_v1/pre_diffusion_pointnet.yaml`,
git `aa498b5`:

| knob | value |
|---|---|
| BC cloud augmentation | `d435i_noise_train`: measured D435i stereo noise + 3 % dropout + leaked table residue p 0.15 |
| BC cloud offset | per-axis rigid U(±5, ±3.5, ±1.5 mm), proprio untouched |
| paired real–sim term | w 0.5 on the 1,031 red-cube play pairs |
| encoder consistency | w 0.3 on 30 % of each batch, view `d435i_noise_strong` (residue p 0.30), no offset |
| network | PointNet 512 + MLP [1024,1024,1024], horizon 4, cond 2, 20 denoising steps |
| batch / lr | 128 / 1e-4, one cosine cycle to 1e-5 |
| epochs | **46** = `EPOCHS=auto` from the 380,000-gradient-step target (8,393 batches/epoch) |
| run-length scaling (user-approved, hydra overrides) | warm-up 3 epochs, checkpoint every 5, EMA from epoch 1, val every 5 |
| seed | 42 |

Why the overrides: the recipe's warm-up (100 epochs), checkpoint interval (250) and EMA start (10) are counted in epochs and
were set for ~2000-epoch runs; at 46 epochs the LR would never have finished warming up and only the final checkpoint would
have been saved. Warm-up was scaled to the same ~5 % of the run as round 4.

## 3. What happened

- **Wall clock 2 h 16 min (17:12–19:28) for 386k gradient steps = 20 ms/step, 17.9 GB GPU** — identical per-step cost to
  round 4 (2.8 s/epoch × 140 batches). Training time is set by the step target, not the dataset size. (An earlier "≈5 h"
  estimate used a stale 46 ms/step from a GPU-contended round; wrong.)
- **Loss curves** (val every 5 epochs):

  | epoch | 5 | 10 | 20 | 30 | 40 | 45 |
  |---|---|---|---|---|---|---|
  | train | 0.00719 | 0.00406 | 0.00242 | 0.00185 | 0.00157 | 0.00149 |
  | val | 0.00647 | 0.00381 | 0.00239 | 0.00199 | 0.00164 | 0.00157 |

  Val tracked train within ~5 % for the whole run and was still falling at the end (−10 % per 5 epochs until the LR
  bottomed out). Round 4 on 100 demos had val bottom at 42k steps and then rise. **No overfitting signal at 6,270 episodes:
  the run is under-trained, not converged.**
- **Teasers** (20 episodes, `serl_sim_server --augmentation d435i_noise`, the same 20 scenario seeds as rounds 1–4,
  final checkpoint `state_46.pt`):

  | experiment | success | ever grasped | eval dir |
  |---|---|---|---|
  | tofu (`single_lift_tofu_soft_abs_action_armfocus_7d_realws`) | **11/20** | 14/20 | `eval/2026-09-07_19-38-41` |
  | mushroom (`single_lift_mushroom_soft_abs_action_armfocus_7d_realws`) | **9/20** | 11/20 | `eval/2026-09-07_20-27-22` |
  | banana_chunk (`single_lift_banana_chunk_soft_abs_action_armfocus_7d_realws`) | **12/20** | 15/20 | `eval/2026-09-07_20-38-17` |
  | tomato (`single_lift_tomato_soft_abs_action_armfocus_7d_realws`) | **16/20** | 19/20 | `eval/2026-09-07_20-47-45` |

  Tofu is the campaign's best sim teaser (round 2 `qzhek_750` 10/20 ever 12; round 4 `yuoqe_2000` 6/20 ever 9); 3 holds
  lost after a grasp, 6 never grasped. `eval/2026-09-07_19-29-51` is a mis-selected `state_5.pt` (5/20) — see §5.

## 4. Findings

1. **The 380k-step target is a 100-demo number.** It gives 2,000 epochs on 100 demos but only 46 on 6,270; each sample is
   seen 46 times and val is still descending. The target should scale with the data (or be replaced by a val-plateau
   rule). Suggested next run: same recipe, `TARGET_STEPS` 2–3× (100–140 epochs, 5–7 h at 20 ms/step); decide the cluster
   generalist's target the same way so the two runs agree.
2. **Per-step cost is flat in dataset size** (the whole train set sits on the GPU: 1.19 M clouds ≈ 14 GB of the 24 GB).
   Estimate wall time as steps × 20 ms; never from epochs. A dataset ~1.6× larger will not fit on the 4090 with the
   current all-on-GPU dataset class.
3. **Val loss did not separate from train**, unlike every 100-demo round. Val loss is still not a robot-performance
   predictor (xagzg lesson), but the absence of the overfit knee is new information about this data scale.
4. **Teasers remain 20-episode reads.** 11/20 vs 6–10/20 is not a ranking; the canonical 200-episode eval on `state_46`
   and the real tofu deploy are the tests that count. A `deploy_dppo.sh` entry is a one-line addition.

## 5. Tooling lessons (recorded so they are not re-learned)

- **"Latest checkpoint" selection**: `ls …/state_*.pt | sort -t_ -k2 -n` splits the *path* on every underscore and picked
  `state_5` over `state_46`; the chain's automatic teaser ran on epoch 5. Use `sort -V`. Fixed in `chain_launch.sh`.
- **Monitors**: a `tail -F | grep --line-buffered | awk` Monitor emitted nothing for eight matching lines; a 60 s polling
  loop that greps the log worked. Prefer polling waiters for multi-hour jobs.
- **`append_experiment_note` under `2>/dev/null`** hid its failure; the DPPO hydra callback does not create EXPERIMENT.md,
  so call `write_experiment_md` first, then append.
- **Sending the converted dataset to the cluster** (`dataset/dppo/single_lift_generalist_soft_v5/`, 13 GB, portable npz)
  skips conversion there; it includes the 400 local episodes and pins the val split, so cluster and local runs become
  directly comparable.
- Disk: everything pre-2026-09-01 under `dataset/` and `logs/` was deleted (66 + 11 GB) before this run; 195 GB free.
