#!/bin/bash
# pi0.5 VLA on the REAL 7-object teleop data: convert -> norm stats -> full fine-tune.
#   sbatch gentle_manip/scripts/arrhenius/pi05_real_vla.sbatch
# Knobs (env): REPO_ID, EXP_NAME, STEPS, BATCH, SKIP_CONVERT=1
#SBATCH -J pi05_real7
#SBATCH -N 1
#SBATCH -t 20:00:00
#SBATCH -A naiss2026-3-141-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --output=/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip/logs/slumr_logs/%j.out
#SBATCH --error=/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip/logs/slumr_logs/%j.err
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip
T=$R/dataset/transfer/real_paired_7obj_2026-09-01
REPO_ID=${REPO_ID:-gm/real7_ext}
EXP_NAME=${EXP_NAME:-pi05_real7_ext}
STEPS=${STEPS:-30000}
BATCH=${BATCH:-64}
cd $R/third_party/openpi          # openpi's uv project
# Our scripts import `gentle_manip`, which is NOT installed in openpi's uv env -- put the
# repo root on the path explicitly (the earlier ad-hoc pi05 runs did this at the shell).
export PYTHONPATH=$R${PYTHONPATH:+:$PYTHONPATH}

# The 7 PINNED real runs, in a fixed order, with their object names for the language labels.
SRCS=""; NAMES=""
for o in mushroom tofu tomato cherry_tomato strawberry grape padron_pepper; do
  d=$(ls -d $T/single_lift_${o}_real/*/ 2>/dev/null | head -1)
  [ -n "$d" ] || { echo "MISSING real run for $o under $T"; exit 1; }
  SRCS="$SRCS ${d%/}"; NAMES="$NAMES $o"
done
echo "[pi05] sources:"; for s in $SRCS; do echo "    ${s#$T/}"; done

if [ "${SKIP_CONVERT:-0}" != "1" ]; then
  echo "[pi05] === convert -> LeRobot ($REPO_ID), ext only, REAL phrasings ==="
  uv run python $R/gentle_manip/pi05/convert_to_lerobot.py $SRCS \
      --object-names $NAMES --repo-id "$REPO_ID" --cameras ext --phrasings real
fi

echo "[pi05] === norm stats ==="
uv run python $R/gentle_manip/pi05/compute_norm_stats.py --repo-id "$REPO_ID"

echo "[pi05] === full fine-tune (wrist slot MASKED OFF) ==="
uv run python $R/gentle_manip/pi05/train_real_vla.py \
    --repo-id "$REPO_ID" --exp-name "$EXP_NAME" --steps "$STEPS" --batch-size "$BATCH"
echo "[pi05] DONE"
