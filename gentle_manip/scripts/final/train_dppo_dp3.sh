#! /bin/bash
# [final] DPPO BC pretrain — the anchor for every sim2real training run. The cfg (sim2real_v1) is
# OBJECT-AGNOSTIC; the per-object / per-run knobs come in as env vars, extra hydra overrides via "$@".
#   DATASET=<dataset name> EXPERIMENT=<experiment name> EPOCHS=<see below> bash gentle_manip/scripts/final/train_dppo_dp3.sh [wandb=null ...]
set -euo pipefail
# ── LARGE-DATASET RECIPE (user, 2026-09-07) — the one recipe for the cluster campaign data. Round history: docs/training_plan_sim2real_2026-09.md.
#    BC path (every sample): measured D435i stereo noise + 3 % dropout, leaked-residue clusters p=0.15, rigid cloud offset per axis
#    U(+-5, +-3.5, +-1.5 mm) with proprio untouched. Paired real-sim term w=0.5 (twins noised like the BC path). Encoder consistency
#    w=0.3 on 30 % of each batch, perturbed view = d435i_noise_strong (noise x1.5, 10 % dropout, <=1 cm occlusion patch above 10 cm,
#    residue p=0.30), NO offset in that view (feature-level shift invariance blinds a PointNet to position).
#    Data: frozen collector v4.2 (10-step hold recorded natively — no post-hoc tail). EPOCHS from the TARGET GRADIENT-STEP COUNT:
#    380,000 steps at batch 128 (user, 2026-09-07) -> EPOCHS = ceil(380000 / ceil(N_train_steps / 128)). Reference: round 4
#    (17.8k train steps -> 140 batches/epoch) ran 2000 epochs = 280,000 steps; e.g. 50k train steps -> 391 batches -> 972 epochs.
DATASET=${DATASET:?set DATASET=<dataset/dppo/<name>> (train/val/normalization.npz)}   # dataset dir = log-dir leaf
EXPERIMENT=${EXPERIMENT:?set EXPERIMENT=<configs/experiments/<name>> (snapshotted to <run>/config/; default eval target)}
TARGET_STEPS=${TARGET_STEPS:-380000}   # target gradient steps at batch 128 (user, 2026-09-07)
EPOCHS=${EPOCHS:-auto}          # auto = ceil(TARGET_STEPS / batches_per_epoch) from the dataset's train.npz; or set explicitly
PAIRED_W=${PAIRED_W:-0.5}       # real-vs-sim paired encoder term; 0 = off
PC_AUG=${PC_AUG:-d435i_noise_train}   # BC path: stereo noise (p=1) + leaked residue (p=0.15); "" = off
CLOUD_OFFSET=${CLOUD_OFFSET:-[0.005,0.0035,0.0015]}   # BC path: per-axis rigid cloud offset U(+-x,+-y,+-z) m; 0 = off
CONS_W=${CONS_W:-0.3}           # clean-vs-perturbed encoder consistency weight; 0 = off
CONS_FRAC=${CONS_FRAC:-0.3}     # fraction of each batch the consistency term uses
CONS_AUG=${CONS_AUG:-d435i_noise_strong}   # the perturbed view (residue p=0.30)
CONS_OFFSET=${CONS_OFFSET:-0}   # NO offset in the consistency view
SEED=${SEED:-42}
CFG_PATH=$(pwd)/gentle_manip/dppo/cfg/sim2real_v1   # ABSOLUTE: hydra resolves relative paths against the dppo fork script dir
if [ "${EPOCHS}" = "auto" ]; then
  NPZ=dataset/dppo/${DATASET}/train.npz
  EPOCHS=$(uv run --project envs/dppo python -c "
import numpy as np, math; T=int(np.load('${NPZ}')['traj_lengths'].sum()); b=math.ceil(T/128); print(math.ceil(${TARGET_STEPS}/b))" 2>/dev/null | tail -1)
  echo "EPOCHS=auto -> ${EPOCHS}  (train steps from ${NPZ}, batch 128, target ${TARGET_STEPS} gradient steps)"
fi

uv run --project envs/dppo python -m gentle_manip.dppo.train \
    --config-path ${CFG_PATH} --config-name pre_diffusion_pointnet \
    env=${DATASET} experiment=${EXPERIMENT} \
    train.n_epochs=${EPOCHS} model.paired_consistency_weight=${PAIRED_W} \
    model.pc_aug=${PC_AUG} model.pc_offset=${CLOUD_OFFSET} \
    model.consistency_weight=${CONS_W} model.consistency_frac=${CONS_FRAC} model.consistency_aug=${CONS_AUG} model.consistency_offset=${CONS_OFFSET} \
    seed=${SEED} "$@"
