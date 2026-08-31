#!/bin/bash
# [arrhenius] 8-object GENERALIST pipeline: stage all cross-category demos -> convert ->
# BC pretrain -> submit per-category canonical eval (yd_gen_eval.sbatch).
#
# Two variants (env VARIANT):
#   regrasp  (default): ALL start modes -> single_lift_gen8_regrasp_pcd
#   baseline           : home + near_object only (non-regraspable) -> single_lift_gen8_baseline_pcd
#
# Submit both:
#   sbatch gentle_manip/scripts/arrhenius/yd_gen8_pipeline.sbatch
#   VARIANT=baseline sbatch gentle_manip/scripts/arrhenius/yd_gen8_pipeline.sbatch
#SBATCH -J yd_gen8
#SBATCH -N 1
#SBATCH -t 20:00:00
#SBATCH -A naiss2026-3-141-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --output=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip/logs/slurm_logs/%j.out
#SBATCH --error=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip/logs/slurm_logs/%j.err
set -euo pipefail
REPO=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip
source "$REPO/gentle_manip/scripts/arrhenius/yd_env_common.sh"

VARIANT=${VARIANT:-regrasp}
if [ "$VARIANT" = "baseline" ]; then
  ENV_NAME=single_lift_gen8_baseline_pcd
  MODES="home,near_object"
else
  ENV_NAME=single_lift_gen8_regrasp_pcd
  MODES=""
fi
CFG_DIR=$REPO/gentle_manip/dppo/cfg/$ENV_NAME
STAGE=$REPO/dataset/demos/_gen8_${VARIANT}
EXPERIMENT=${EXPERIMENT:-single_lift_xcat_diverse}          # obs-key order for convert (student view)
echo "[gen8] variant=$VARIANT env=$ENV_NAME modes='${MODES:-all}' stage=$STAGE"

echo; echo "========== stage demos =========="
uv run --project envs/dppo_arrhenius --no-sync python \
  "$REPO/gentle_manip/scripts/arrhenius/_gen8_stage.py" "$STAGE" ${MODES:+--modes "$MODES"}
[ -f "$STAGE/data.pkl" ] || { echo "[gen8] staging produced no data.pkl"; exit 1; }

if [ "${SKIP_TRAIN:-0}" = "1" ] && [ -n "${RUN_DIR:-}" ]; then
  echo "[gen8] SKIP_TRAIN -> RUN_DIR=$RUN_DIR"
else
  echo; echo "========== convert -> $DPPO_DATA_DIR/$ENV_NAME =========="
  uv run --project envs/dppo_arrhenius --no-sync python -m gentle_manip.dppo.convert_demos \
    "$STAGE" --out "$DPPO_DATA_DIR/$ENV_NAME" \
    --experiment "$EXPERIMENT" --view student --point-cloud --val-split 0.1
  uv run --project envs/dppo_arrhenius --no-sync python -c \
    "import numpy as np; d=np.load('$DPPO_DATA_DIR/$ENV_NAME/train.npz'); print('train', {k:d[k].shape for k in d.files})" || true

  echo; echo "========== BC pretrain ($ENV_NAME) =========="
  PRE_LOG=$REPO/logs/slurm_logs/${SLURM_JOB_ID}_pretrain.log
  : > "$PRE_LOG"
  uv run --project envs/dppo_arrhenius --no-sync python -m gentle_manip.dppo.train \
    --config-path "$CFG_DIR" --config-name pre_diffusion_pointnet env=$ENV_NAME 2>&1 | tee "$PRE_LOG"
  RUN_DIR=$(grep -oE '/nobackup/[^ ]+/dppo-pretrain/[^ ]+/config/' "$PRE_LOG" | head -1 | sed 's#/config/##')
fi
echo "[gen8] pretrain run dir: $RUN_DIR"
ls -1v "$RUN_DIR"/checkpoint/state_*.pt | tail -3
echo "$RUN_DIR" > "$REPO/logs/slurm_logs/last_gen8_${VARIANT}_run.txt"

echo; echo "========== submit per-category eval (yd_gen_eval.sbatch) =========="
# yd_gen_eval picks best-val ckpt from RUN_DIR, loops 8 in-domain + 4 OOD, scene_group_size 1.
# ENV_NAME here selects the eval cfg dir (eval_diffusion_pointnet.yaml) -- use the gen8 one.
ENV_NAME="$ENV_NAME" RUN_DIR="$RUN_DIR" \
  sbatch "$REPO/gentle_manip/scripts/arrhenius/yd_gen_eval.sbatch" \
  && echo "[gen8] eval submitted for $RUN_DIR" || echo "[gen8] eval submit FAILED -- run manually"
