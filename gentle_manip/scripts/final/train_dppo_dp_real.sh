#! /bin/bash
# [final] DPPO BC pretrain on REAL teleop demos, RGB observations — the image twin of
# train_dppo_dp3_real.sh. Same data, same split, same diffusion and optimisation settings as the
# point-cloud run, so the ONLY difference is the observation branch and its encoder.
#   DATASET=<dataset name> bash gentle_manip/scripts/final/train_dppo_dp_real.sh [wandb=null ...]
set -euo pipefail
# ENCODER: ImageNet-pretrained ResNet-18 with BatchNorm->GroupNorm (Diffusion Policy's real-world
# recipe; BN interacts badly with the EMA this loop keeps), replacing DPPO's from-scratch depth-1
# ViT, which is a weak encoder at ~120 demonstrations. Images are 224 px to suit pretrained
# features. Normalisation (/255 + ImageNet mean/std) lives INSIDE the encoder so training and
# deploy cannot diverge; the resize is the same PIL BILINEAR call in convert_demos and deploy.
# AUGMENTATION: random shift (robomimic's most impactful) + photometric jitter for illumination
# drift across collection days. Implemented as TrainOnlyImageAug, which self-disables under
# model.eval() — DPPO's stock RandomShiftsAug does NOT, and would jitter every deploy inference.
# Real-only, so the three sim2real terms stay off (no paired term, no encoder consistency, no
# experiment). Rationale + citations: docs/final/baseline_training.md §2.
#
# Dataset must be converted WITH --images at 224:
#   uv run --project envs/dppo python -m gentle_manip.dppo.convert_demos \
#     dataset/transfer/real_demos_2026-09-09 --out dataset/dppo/single_lift_real7_bc_rgb224_v1 \
#     --obs-keys ee_pos ee_quat gripper_width --images image_cam_ext --image-size 224 --val-split 0.1
#   (--obs-keys is REQUIRED: the proprio view is only the default when --point-cloud is given,
#    otherwise the converter falls back to the privileged STATE_VIEW and fails on real data.)
DATASET=${DATASET:?set DATASET=<dataset/dppo/<name>> (train/val/normalization.npz with `images`)}
EPOCHS=${EPOCHS:-2000}
SAVE_FREQ=${SAVE_FREQ:-100}     # 20 checkpoints; overfitting is expected, sweep rather than assume
WARMUP=${WARMUP:-100}
VAL_FREQ=${VAL_FREQ:-10}
SEED=${SEED:-42}
# Photometric ranges. Defaults are modest on purpose: the measured drift across our collection days
# is ~1 % (R/G 1.4 %, B/G 0.4 %, luminance 0.5 %) because the camera auto-exposes, so heavy jitter
# would spend fit on variation that does not occur. CHANNEL_GAIN models white balance and replaces
# hue rotation, which would blur a cue the policy legitimately uses (red tomato / brown mushroom /
# white tofu).
BRIGHTNESS=${BRIGHTNESS:-0.25}
CONTRAST=${CONTRAST:-0.25}
SATURATION=${SATURATION:-0.2}
CHANNEL_GAIN=${CHANNEL_GAIN:-0.05}
SHIFT_PAD=${SHIFT_PAD:-4}
CFG_PATH=$(pwd)/gentle_manip/dppo/cfg/real_rgb_v2   # ABSOLUTE: hydra resolves relative paths against the dppo fork script dir

uv run --project envs/dppo python -m gentle_manip.dppo.train \
    --config-path ${CFG_PATH} --config-name pre_diffusion_vision \
    env=${DATASET} \
    train.n_epochs=${EPOCHS} train.save_model_freq=${SAVE_FREQ} \
    train.lr_scheduler.warmup_steps=${WARMUP} train.val_freq=${VAL_FREQ} \
    model.network.aug_pad=${SHIFT_PAD} \
    model.network.aug_brightness=${BRIGHTNESS} model.network.aug_contrast=${CONTRAST} \
    model.network.aug_saturation=${SATURATION} model.network.aug_channel_gain=${CHANNEL_GAIN} \
    seed=${SEED} "$@"
