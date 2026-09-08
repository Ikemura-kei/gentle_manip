#! /bin/bash
# [final] DPPO BC pretrain on REAL teleop demos — the real-only twin of train_dppo_dp3.sh.
#   DATASET=<dataset name> bash gentle_manip/scripts/final/train_dppo_dp3_real.sh [wandb=null ...]
set -euo pipefail
# Same cfg, network and diffusion settings as the sim2real anchor (sim2real_v1: PointNet encoder,
# 20 denoising steps, horizon 4, proprio history 2, batch 128, lr 1e-4, one cosine cycle). What
# differs is everything that exists to bridge sim -> real, because here there is no domain gap:
#   * NO cloud augmentation      — real clouds already carry real sensor noise and table residue
#   * NO paired real-sim term    — it needs a sim twin per cloud; there is none
#   * NO encoder consistency     — it bought robustness to a gap that does not exist here
#   * NO experiment              — a real run has no sim env config; provenance is the dataset's
#                                  own sources.yaml (and train_launch.sh, written by hand)
# wandb goes to the SAME project as the generalist runs (user, 2026-09-08) so the loss curves sit
# side by side — note only the BC term is comparable: the generalist's total loss also carries the
# paired and consistency terms, which are off here.
# Schedule knobs scale with the run length: 2000 epochs -> warmup 100 (5 %), ckpt every 100 (20 kept),
# val every 10. Expect the useful checkpoint EARLY: 2000 epochs over ~25k train steps means each
# sample is seen ~2000x, the ratio at which the sim rounds 1-4 overfitted (val bottomed ~1/5 in).
DATASET=${DATASET:?set DATASET=<dataset/dppo/<name>> (train/val/normalization.npz)}   # dataset dir = log-dir leaf
EPOCHS=${EPOCHS:-2000}          # 110 real eps = 24.6k train steps = 193 batches -> 386k gradient steps
SAVE_FREQ=${SAVE_FREQ:-100}     # 20 checkpoints; the val minimum is expected around epoch ~200-400
WARMUP=${WARMUP:-100}           # 5 % of the run
VAL_FREQ=${VAL_FREQ:-10}
SEED=${SEED:-42}
CFG_PATH=$(pwd)/gentle_manip/dppo/cfg/sim2real_v1   # ABSOLUTE: hydra resolves relative paths against the dppo fork script dir

uv run --project envs/dppo python -m gentle_manip.dppo.train \
    --config-path ${CFG_PATH} --config-name pre_diffusion_pointnet \
    env=${DATASET} experiment="" \
    train.n_epochs=${EPOCHS} train.save_model_freq=${SAVE_FREQ} \
    train.lr_scheduler.warmup_steps=${WARMUP} train.val_freq=${VAL_FREQ} \
    model.paired_consistency_weight=0 model.pc_aug="" model.pc_offset=0 \
    model.consistency_weight=0 \
    wandb.project=${WANDB_PROJECT:-gentle_manip_generalist} \
    seed=${SEED} "$@"
