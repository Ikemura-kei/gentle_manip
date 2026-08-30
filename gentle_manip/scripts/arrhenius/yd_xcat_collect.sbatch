#!/bin/bash
# [arrhenius] CONTINUOUS cross-category diverse-start collection (Phase 4).
# 9 held-in soft objects drawn per scene (dr: xcat_diverse_regrasp). Runs to
# N_EPISODES (default 4500 = 500/object) or walltime, whichever first -- resubmit
# to keep going; salvage-merges leftover shards from any crashed prior run.
# Submit:
#   sbatch gentle_manip/scripts/arrhenius/yd_xcat_collect.sbatch                     # regraspable
#   START_MODES=strict_home:1.0 TAG=xcat_baseline sbatch ...yd_xcat_collect.sbatch    # non-regraspable baseline
#   knobs (env): N_EPISODES N_ENVS MAXFEVALS SEED START_MODES CRUSH_FRAC SCENE_DR_EVERY TAG EXPERIMENT
#SBATCH -J yd_xcat
#SBATCH -N 1
#SBATCH -t 71:00:00
#SBATCH -A naiss2026-3-141-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip/logs/slurm_logs/%j.out
#SBATCH --error=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip/logs/slurm_logs/%j.err
set -euo pipefail
REPO=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip
source "$REPO/gentle_manip/scripts/arrhenius/yd_env_common.sh"

N_EPISODES=${N_EPISODES:-4500}
N_ENVS=${N_ENVS:-6}
EXPERIMENT=${EXPERIMENT:-single_lift_xcat_diverse}
SCENE_DR_EVERY=${SCENE_DR_EVERY:-1}          # draw a new object every batch
MAXFEVALS=${MAXFEVALS:-700}
SEED=${SEED:-0}
CRUSH_FRAC=${CRUSH_FRAC:-1.20}   # banana proof used ~1.25; tighter grip (wcap 0.8*short) does the gentleness work
START_MODES=${START_MODES:-sweep:0.44,failed:0.30,above:0.10,ground:0.09,air:0.07}
TAG=${TAG:-xcat_regrasp}
TASK_DIR=${TASK_DIR:-single_lift_${TAG}}     # output folder under dataset/demos/

echo "[xcat] tag=$TAG exp=$EXPERIMENT n_episodes=$N_EPISODES n_envs=$N_ENVS crush=$CRUSH_FRAC"
echo "[xcat] start_modes=$START_MODES"

# Salvage leftover shards from a crashed prior run of this TAG.
for RD in "$REPO"/dataset/demos/$TASK_DIR/*/; do
  if [ -n "$(ls "$RD"shard_*.pkl 2>/dev/null)" ] && [ ! -f "$RD/data.pkl" ]; then
    echo "[xcat] merging leftover shards in $RD"
    uv run --project envs/sim_arrhenius --no-sync python - "$RD" <<'PYM'
import sys; sys.path.insert(0,"grasp_synthesis")
from pathlib import Path
import collect_demos_synth as v1
print("  merged", v1._merge_shards(Path(sys.argv[1])))
PYM
  fi
done

env -u PYTHONPATH -u ROS_DISTRO MUJOCO_GL=egl uv run --project envs/sim_arrhenius --no-sync python \
  grasp_synthesis/collect_demos_diverse_start_v2.py \
  --experiment "$EXPERIMENT" --task-name "$TASK_DIR" \
  --n-episodes "$N_EPISODES" --n-envs "$N_ENVS" --maxfevals "$MAXFEVALS" \
  --scene-dr-every "$SCENE_DR_EVERY" --seed "$SEED" --start-modes "$START_MODES" \
  --crush-frac "$CRUSH_FRAC" --record-video 60 --out-dir "$REPO/dataset/demos"

RUN=$(ls -dt "$REPO"/dataset/demos/$TASK_DIR/* | head -1)
echo "[xcat] DONE -> $RUN"
cat "$RUN/stats.yaml" || true
echo "$RUN" > "$REPO/logs/slurm_logs/last_${TAG}_run.txt"
