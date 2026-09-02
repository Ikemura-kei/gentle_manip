#!/bin/bash
# [arrhenius] REACTIVE policy (v3) pipeline: _reactive_stage (v2 regrasp + reactive demos,
# ~30% reactive) -> convert (obs_dim 12) -> BC pretrain -> auto yd_reactive_eval.
#SBATCH -J yd_rpipe
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -A naiss2026-3-141-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --output=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip/logs/slurm_logs/%j.out
#SBATCH --error=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip/logs/slurm_logs/%j.err
set -uo pipefail
REPO=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip
source "$REPO/gentle_manip/scripts/arrhenius/yd_env_common.sh"
ENV_NAME=single_lift_gen8_reactive_pcd
CFG_DIR=$REPO/gentle_manip/dppo/cfg/$ENV_NAME
STAGE=$REPO/dataset/demos/_gen8_reactive

echo "========== stage =========="
uv run --project envs/dppo_arrhenius --no-sync python \
  "$REPO/gentle_manip/scripts/arrhenius/_reactive_stage.py" "$STAGE" --reactive-frac "${REACTIVE_FRAC:-0.30}"
[ -f "$STAGE/data.pkl" ] || { echo "no data.pkl"; exit 1; }

echo "========== convert -> $DPPO_DATA_DIR/$ENV_NAME =========="
uv run --project envs/dppo_arrhenius --no-sync python -m gentle_manip.dppo.convert_demos \
  "$STAGE" --out "$DPPO_DATA_DIR/$ENV_NAME" \
  --experiment single_lift_xcat_diverse --view student --point-cloud --val-split 0.1 \
  --extra-state-keys object_at_gripper
uv run --project envs/dppo_arrhenius --no-sync python -c \
  "import numpy as np;d=np.load('$DPPO_DATA_DIR/$ENV_NAME/train.npz');print('train',{k:d[k].shape for k in d.files})" || true

echo "========== BC pretrain =========="
PRE_LOG=$REPO/logs/slurm_logs/${SLURM_JOB_ID}_pretrain.log; : > "$PRE_LOG"
uv run --project envs/dppo_arrhenius --no-sync python -m gentle_manip.dppo.train \
  --config-path "$CFG_DIR" --config-name pre_diffusion_pointnet env=$ENV_NAME 2>&1 | tee "$PRE_LOG"
RUN_DIR=$(grep -oE '/nobackup/[^ ]+/dppo-pretrain/[^ ]+/config/' "$PRE_LOG" | head -1 | sed 's#/config/##')
echo "[rpipe] run dir: $RUN_DIR"
echo "$RUN_DIR" > "$REPO/logs/slurm_logs/last_reactive_run.txt"

echo "========== submit reactive eval =========="
ENV_NAME="$ENV_NAME" RUN_DIR="$RUN_DIR" \
  CATS="mushroom banana_lying kiwi egg_boiled grape cherry raspberry" NEP=60 SMALL_NEP=40 NCAT_PAR=4 \
  sbatch -t 06:00:00 "$REPO/gentle_manip/scripts/arrhenius/yd_reactive_eval.sbatch" || echo "eval submit FAILED"
