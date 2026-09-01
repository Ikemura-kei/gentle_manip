#!/bin/bash
# Hourly training-progress snapshot of the gen8 REGRASP policy: pick the current best-val
# checkpoint, run 15 rollouts across 3 RANDOM in-domain categories (5 each), record every
# clip. Fast (early-terminating harness + n=5/cat). Prints the OUT dir; the caller builds a
# montage from OUT/*/render/*.mp4 once the job finishes.
set -uo pipefail
REPO=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip
RUN_DIR=$(ls -dt "$REPO"/logs/dppo/dppo-pretrain/single_lift_gen8_regrasp_pcd/*/ 2>/dev/null | head -1)
RUN_DIR=${RUN_DIR%/}
[ -n "$RUN_DIR" ] || { echo "no regrasp run dir yet"; exit 0; }
CKPTS=$(ls -1v "$RUN_DIR"/checkpoint/state_*.pt 2>/dev/null | wc -l)
[ "$CKPTS" -ge 1 ] || { echo "no checkpoint in $RUN_DIR yet ($CKPTS) -- skip this snap"; exit 0; }

# LARGE-only pool for the hourly snap: small fruit (grape/cherry/raspberry, grid 300) are
# ~6x slower per sim step, and 2 of them concurrent under NCAT_PAR=3 blew a 20min backfill
# walltime with zero output (1897808 TIMEOUT). Small cats still get covered by the full
# 12-cat eval; the snap is for a quick training-progress read, not full coverage.
INDOMAIN="mushroom banana_lying kiwi egg_boiled"  # tomato excluded: spawns at z~0.31 (scene bug, see handoff)
CATS=$(echo "$INDOMAIN" | tr ' ' '\n' | shuf -n2 | tr '\n' ' ')
TS=$(date +%Y%m%d_%H%M)
OUT="$RUN_DIR/snap/$TS"
echo "$OUT" > "$REPO/logs/slurm_logs/last_regrasp_snap.txt"

ENV_NAME=single_lift_gen8_regrasp_pcd RUN_DIR="$RUN_DIR" CATS="$CATS" \
  NEP=5 SMALL_NEP=5 RECORD_BATCHES=1 NCAT_PAR=2 NOTRIM=1 OUT="$OUT" \
  sbatch -t 00:20:00 --job-name=yd_rsnap \
  "$REPO/gentle_manip/scripts/arrhenius/yd_gen_eval.sbatch"
echo "snap submitted -> $OUT   cats: $CATS   (best-val ckpt auto-picked from $CKPTS)"
