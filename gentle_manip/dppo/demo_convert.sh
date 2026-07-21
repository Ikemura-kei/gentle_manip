#!/usr/bin/env bash
# Reproducible pipeline for the WIDE-coverage mushroom-lift dataset + BC base policy:
#   collect (sim, scripted, parallel) -> convert (DPPO format) -> pretrain (Diffusion Policy BC)
#
# Run from the repo root. Each stage is independent; set DEMO_RUN after collecting (it is a
# per-run datetime dir). The instance that produced the current base policy `gllzd` used
# DEMO_RUN=26-07-19-021443 (303 demos) and DATA_ENV=single_lift_mushroom_soft_pcd_wide.
#
# Only stage 3 auto-saves its exact command (logs/.../<id>/launch_command.sh via the DPPO
# hydra callback). Stages 1 and 2 are NOT auto-recorded anywhere else — this file is their
# source of truth.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

DATA_ENV=single_lift_mushroom_soft_pcd_wide          # DPPO dataset + logdir name (keeps old data)
N_DEMOS=300
POSE_BOX="0.15 0.15 0.10"                             # per-axis half-range = 30x30x20 cm start box

# ── 1. COLLECT: 300 successful demos, 5 parallel envs, scripted expert, success-only ──────────
# Wide coverage: enlarged robot-pose box (collection-only) + eval object/orientation/material DR
# + geometry rebuild every 5 batches, at eval sim fidelity (sim_substeps/mpm_grid_density from
# the task cfg = 210 / 250). Writes dataset/demos/mushroom_soft_batched/<datetime>/shard_*.pkl.
MUJOCO_GL=egl uv run --project envs/sim --no-sync python examples/collect_mushroom_demos_batched.py \
  --n-demos "$N_DEMOS" --n-envs 5 --pose-box $POSE_BOX \
  --scene-dr-every 5 --shard-size 20 --max-steps 320 --seed 0

# Point this at the datetime dir the collect above created (ls -t dataset/demos/mushroom_soft_batched):
DEMO_RUN=26-07-19-021443

# ── 2. CONVERT: sharded demos -> DPPO pretrain format (train/val/normalization .npz) ──────────
# Recurses the run dir for shard_*.pkl. --point-cloud selects the PROPRIO_VIEW state
# (ee_pos+ee_quat+gripper_width, 8-dim) + the raw 1024-pt cloud. Writes to
# dataset/dppo/$DATA_ENV/ so the old single_lift_mushroom_soft_pcd data is untouched.
uv run --project envs/dppo --no-sync python -m gentle_manip.dppo.convert_demos \
  "dataset/demos/mushroom_soft_batched/$DEMO_RUN" \
  --out "dataset/dppo/$DATA_ENV" --point-cloud

# ── 3. TRAIN: Diffusion Policy BC base (DPPO pre_diffusion_pointnet) ───────────────────────────
# Pure supervised (no sim server). env=$DATA_ENV routes the dataset path + logdir to the new
# name; experiment= stays the real env definition (single_lift_mushroom_soft). Mints a 5-letter
# run id under logs/dppo/dppo-pretrain/$DATA_ENV/<id>/ (this run = gllzd). 3000 epochs.
uv run --project envs/dppo --no-sync python -m gentle_manip.dppo.train \
  --config-path "$(pwd)/gentle_manip/dppo/cfg/single_lift_mushroom_soft_pcd" \
  --config-name pre_diffusion_pointnet \
  env="$DATA_ENV"

# ── 4. (next) EVAL the base policy through the canonical harness — see docs/dppo_eval.md ──────
#   BC checkpoint eval: base_policy_path=logs/.../<id>/checkpoint/state_3000.pt  ft-denoising-steps 0
