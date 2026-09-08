#! /bin/bash
# [final] DPPO BC pretrain on REAL teleop demos, RGB observations — the image twin of
# train_dppo_dp3_real.sh (dp, not dp3: a ViT on cam_ext RGB instead of a PointNet on clouds).
#   DATASET=<dataset name> bash gentle_manip/scripts/final/train_dppo_dp_real.sh [wandb=null ...]
set -euo pipefail
# Same data, same 10 % split, same diffusion and optimisation settings as the point-cloud run, so the
# two are directly comparable; the ONLY change is the observation branch. Like that run: no cloud/image
# augmentation, no paired real-sim term, no encoder consistency, no experiment (real data has no sim
# env config). Structure follows DPPO's proven image recipe (cfg/robomimic/.../pre_diffusion_mlp_img).
#   AUGMENT=True turns on RandomShiftsAug (DPPO's image default; off here to match the cloud run).
# Dataset must be converted WITH --images:
#   uv run --project envs/dppo python -m gentle_manip.dppo.convert_demos \
#     dataset/transfer/real_demos_2026-09-08 --out dataset/dppo/single_lift_real6_bc_rgb_v1 \
#     --obs-keys ee_pos ee_quat gripper_width --images image_cam_ext --image-size 96 --val-split 0.1
#   (--obs-keys is REQUIRED here: the proprio view is only the default when --point-cloud is given,
#    otherwise the converter falls back to the privileged STATE_VIEW and fails on real data.)
DATASET=${DATASET:?set DATASET=<dataset/dppo/<name>> (train/val/normalization.npz with `images`)}
EPOCHS=${EPOCHS:-2000}
SAVE_FREQ=${SAVE_FREQ:-100}     # 20 checkpoints; overfitting is expected, take an early one
WARMUP=${WARMUP:-100}
VAL_FREQ=${VAL_FREQ:-10}
AUGMENT=${AUGMENT:-False}       # RandomShiftsAug on the RGB input
SEED=${SEED:-42}
CFG_PATH=$(pwd)/gentle_manip/dppo/cfg/real_rgb_v1   # ABSOLUTE: hydra resolves relative paths against the dppo fork script dir

uv run --project envs/dppo python -m gentle_manip.dppo.train \
    --config-path ${CFG_PATH} --config-name pre_diffusion_vision \
    env=${DATASET} \
    train.n_epochs=${EPOCHS} train.save_model_freq=${SAVE_FREQ} \
    train.lr_scheduler.warmup_steps=${WARMUP} train.val_freq=${VAL_FREQ} \
    model.network.augment=${AUGMENT} \
    seed=${SEED} "$@"
