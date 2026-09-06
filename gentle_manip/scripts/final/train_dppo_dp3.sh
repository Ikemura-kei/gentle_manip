#! /bin/bash
# [final] DPPO BC pretrain — the anchor for every sim2real training run. The cfg (sim2real_v1) is
# OBJECT-AGNOSTIC; the per-object / per-run knobs come in as env vars, extra hydra overrides via "$@".
#   DATASET=single_lift_tofu_sim2real_v1 EXPERIMENT=single_lift_tofu_soft_abs_action_armfocus_7d_realws \
#   EPOCHS=2800 PAIRED_W=0.5 bash gentle_manip/scripts/final/train_dppo_dp3.sh [wandb=null ...]
set -euo pipefail
# ── Round 1 (2026-09-06 morning, runs covel/tzdhk): 100 tofu demos, no hold tail, BC aug + paired 0.5 ──
# DATASET=${DATASET:-single_lift_tofu_sim2real_v1}
# EPOCHS=${EPOCHS:-3000}; PAIRED_W=${PAIRED_W:-0.5}; PC_AUG=${PC_AUG:-d435i_noise}; CLOUD_OFFSET=${CLOUD_OFFSET:-0.008}
# CONS_W=0   # (no encoder consistency term yet)
# ── Round 2 (2026-09-06 evening): hold tail K=60, leaked-residue aug (in d435i_noise), clean-vs-perturbed encoder
#    consistency (d435i_noise_strong, w 0.3, half batch, offset 12 mm), paired 0.5. Collector now keeps a 60-step hold. ──
DATASET=${DATASET:-single_lift_tofu_sim2real_v1_tail60}   # dataset/dppo/<DATASET>/{train,val,normalization}.npz; log-dir leaf
EXPERIMENT=${EXPERIMENT:-single_lift_tofu_soft_abs_action_armfocus_7d_realws}   # snapshotted to <run>/config/
EPOCHS=${EPOCHS:-1500}          # development test length (user, 2026-09-06); ~3000 x 29648 / N_train_steps, clamped [800, 3000]; tzdhk's val min was at ep 750
PAIRED_W=${PAIRED_W:-0.5}       # real-vs-sim paired encoder term; 0 = off
PC_AUG=${PC_AUG:-d435i_noise_train}   # train-time cloud sensor noise + leaked residue (p=0.5), configs/augmentation/<name>.yaml; "" = off
CLOUD_OFFSET=${CLOUD_OFFSET:-0.008}   # per-sample rigid cloud offset U(+-m) on the BC clouds; 0 = off
CONS_W=${CONS_W:-0.3}           # clean-vs-perturbed encoder consistency weight; 0 = off
CONS_AUG=${CONS_AUG:-d435i_noise_strong}   # the perturbed view's augmentation yaml
CONS_OFFSET=${CONS_OFFSET:-0.012}          # rigid offset U(+-m) on the perturbed view
SEED=${SEED:-42}
CFG_PATH=$(pwd)/gentle_manip/dppo/cfg/sim2real_v1   # ABSOLUTE: hydra resolves relative paths against the dppo fork script dir

uv run --project envs/dppo python -m gentle_manip.dppo.train \
    --config-path ${CFG_PATH} --config-name pre_diffusion_pointnet \
    env=${DATASET} experiment=${EXPERIMENT} \
    train.n_epochs=${EPOCHS} model.paired_consistency_weight=${PAIRED_W} \
    model.pc_aug=${PC_AUG} model.pc_offset=${CLOUD_OFFSET} \
    model.consistency_weight=${CONS_W} model.consistency_aug=${CONS_AUG} model.consistency_offset=${CONS_OFFSET} \
    seed=${SEED} "$@"
