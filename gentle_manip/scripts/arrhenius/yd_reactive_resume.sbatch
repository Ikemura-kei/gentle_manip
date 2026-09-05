#!/bin/bash
# [arrhenius] Resume the v4 reactive BC-pretrain (single_lift_gen8_reactive_pcd) from a
# checkpoint to completion, then auto-submit the 12-cat reactive eval -- unattended,
# self-resubmitting chain (job 1971677 hit its 6h walltime at epoch 256/300 with no
# resubmit logic, stalling the campaign with nobody watching it).
#
# Two layers of timeout resilience:
#   1. --signal sends SIGUSR1 a few minutes before the hard kill; the trap stops
#      training cleanly, checks progress, and resubmits itself (or submits eval if done)
#      BEFORE SLURM kills the job -- so a graceful chain link never gets lost.
#   2. If SLURM still kills the whole job before the trap can finish (contention, a
#      slow shutdown), the NEXT invocation just resumes from whatever periodic
#      checkpoint (save_model_freq=25 epochs) made it to disk -- at most ~24 epochs
#      of rework, never a full restart.
#
# Usage: RESUME_FROM=<run>/checkpoint/state_<N>.pt sbatch yd_reactive_resume.sbatch
#   knobs: RESUME_FROM (required first launch) CHAIN_DEPTH (0) MAX_CHAIN (10, safety cap)
#SBATCH -J yd_rresume
#SBATCH -N 1
#SBATCH -t 08:00:00
#SBATCH --signal=B:USR1@300
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
N_EPOCHS=300
CHAIN_DEPTH=${CHAIN_DEPTH:-0}
MAX_CHAIN=${MAX_CHAIN:-10}
RESUME_FROM=${RESUME_FROM:?set RESUME_FROM=<run_dir>/checkpoint/state_N.pt}

echo "[rresume] chain depth $CHAIN_DEPTH/$MAX_CHAIN resuming from $RESUME_FROM"
PRE_LOG=$REPO/logs/slurm_logs/${SLURM_JOB_ID}_pretrain.log
: > "$PRE_LOG"

RUN_DIR=""
finish_or_chain() {
  RUN_DIR=$(grep -oE '/nobackup/[^ ]+/dppo-pretrain/[^ ]+/config/' "$PRE_LOG" | head -1 | sed 's#/config/##')
  if [ -z "$RUN_DIR" ]; then
    echo "[rresume] no run dir parsed from log yet (killed before hydra even wrote its config?) -- resubmitting same RESUME_FROM"
    if [ "$CHAIN_DEPTH" -lt "$MAX_CHAIN" ]; then
      RESUME_FROM="$RESUME_FROM" CHAIN_DEPTH=$((CHAIN_DEPTH+1)) MAX_CHAIN=$MAX_CHAIN \
        sbatch --job-name="$SLURM_JOB_NAME" "$REPO/gentle_manip/scripts/arrhenius/yd_reactive_resume.sbatch" || true
    fi
    return
  fi
  echo "[rresume] run dir: $RUN_DIR"
  echo "$RUN_DIR" > "$REPO/logs/slurm_logs/last_reactive_run.txt"
  LAST_CKPT=$(ls -1v "$RUN_DIR"/checkpoint/state_*.pt 2>/dev/null | tail -1)
  LAST_EPOCH=$(basename "${LAST_CKPT:-state_0.pt}" | grep -oE '[0-9]+')
  echo "[rresume] last checkpoint: ${LAST_CKPT:-none} (epoch ${LAST_EPOCH:-0})"

  DONE=0
  if grep -qi "early stop" "$PRE_LOG"; then
    echo "[rresume] early-stopped -> training complete"; DONE=1
  elif [ -n "$LAST_EPOCH" ] && [ "$LAST_EPOCH" -ge "$N_EPOCHS" ]; then
    echo "[rresume] reached n_epochs=$N_EPOCHS -> training complete"; DONE=1
  fi

  if [ "$DONE" = "1" ]; then
    echo "========== submit v4 reactive eval (12 cats: 8 in-domain + 4 OOD) =========="
    ENV_NAME="$ENV_NAME" RUN_DIR="$RUN_DIR" \
      CATS="mushroom banana_lying kiwi egg_boiled grape cherry raspberry blackberry scallop dumpling gelatin tomato" \
      NEP=50 SMALL_NEP=40 NCAT_PAR=4 RECORD_BATCHES=1 \
      sbatch -t 12:00:00 --job-name=yd_v4drag "$REPO/gentle_manip/scripts/arrhenius/yd_reactive_eval.sbatch" \
      && echo "[rresume] eval submitted" || echo "[rresume] eval submit FAILED -- run manually"
  elif [ -z "$LAST_CKPT" ]; then
    echo "[rresume] no checkpoint on disk in $RUN_DIR -- stopping chain here, needs a manual look"
  elif [ "$CHAIN_DEPTH" -ge "$MAX_CHAIN" ]; then
    echo "[rresume] MAX_CHAIN ($MAX_CHAIN) reached without finishing -- stopping, needs a manual look"
  else
    echo "[rresume] not done (epoch ${LAST_EPOCH:-0}/$N_EPOCHS) -- resubmitting chain depth $((CHAIN_DEPTH+1))"
    RESUME_FROM="$LAST_CKPT" CHAIN_DEPTH=$((CHAIN_DEPTH+1)) MAX_CHAIN=$MAX_CHAIN \
      sbatch --job-name="$SLURM_JOB_NAME" "$REPO/gentle_manip/scripts/arrhenius/yd_reactive_resume.sbatch" || true
  fi
}

TRAIN_PID=""
term_handler() {
  echo "[rresume] pre-timeout signal caught -- stopping training pid $TRAIN_PID cleanly"
  [ -n "$TRAIN_PID" ] && kill -TERM "$TRAIN_PID" 2>/dev/null
  for _ in $(seq 1 20); do kill -0 "$TRAIN_PID" 2>/dev/null || break; sleep 5; done
  kill -0 "$TRAIN_PID" 2>/dev/null && kill -KILL "$TRAIN_PID" 2>/dev/null
  wait "$TRAIN_PID" 2>/dev/null || true
  finish_or_chain
  exit 0
}
trap term_handler SIGUSR1 SIGTERM

uv run --project envs/dppo_arrhenius --no-sync python -m gentle_manip.dppo.train \
  --config-path "$CFG_DIR" --config-name pre_diffusion_pointnet env=$ENV_NAME \
  +resume_from="$RESUME_FROM" > "$PRE_LOG" 2>&1 &
TRAIN_PID=$!
wait "$TRAIN_PID" || true
finish_or_chain
