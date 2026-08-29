#!/bin/bash
# [arrhenius] Collect diverse-start rigid-banana demos (OmniReset-inspired regrasp
# campaign). Single-attempt always-successful top-down grasps; wide object-pose DR +
# rich EE-start diversity; one RGB video per saved episode (no frozen frames).
# Submit:  sbatch gentle_manip/scripts/arrhenius/yd_banana_collect.sbatch
#   knobs (env): N_EPISODES(500) N_ENVS(8) MAXFEVALS(700) SEED(0) START_MODES
#SBATCH -J yd_bcollect
#SBATCH -N 1
#SBATCH -t 16:00:00
#SBATCH -A naiss2026-3-141-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip/logs/slurm_logs/%j.out
#SBATCH --error=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip/logs/slurm_logs/%j.err
set -euo pipefail
REPO=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip
source "$REPO/gentle_manip/scripts/arrhenius/yd_env_common.sh"

N_EPISODES=${N_EPISODES:-500}   # full 500 fresh (6 start modes incl above_object + mid_approach)
N_ENVS=${N_ENVS:-8}
EXPERIMENT=${EXPERIMENT:-single_lift_banana_rigid_diverse}
TASK_DIR=${TASK_DIR:-single_lift_banana_rigid}   # output folder under dataset/demos/
SCENE_DR_EVERY=${SCENE_DR_EVERY:-1}
MAXFEVALS=${MAXFEVALS:-700}
SEED=${SEED:-0}
START_MODES=${START_MODES:-sweep:0.44,failed:0.30,above:0.10,ground:0.09,air:0.07}

echo "[collect] n_episodes=$N_EPISODES n_envs=$N_ENVS maxfevals=$MAXFEVALS seed=$SEED"
echo "[collect] start_modes=$START_MODES"

# Salvage: merge shard_*.pkl from any prior run dir that crashed before _merge_shards.
for RD in "$REPO"/dataset/demos/$TASK_DIR/*/; do
  if [ -n "$(ls "$RD"shard_*.pkl 2>/dev/null)" ] && [ ! -f "$RD/data.pkl" ]; then
    echo "[collect] merging leftover shards in $RD"
    uv run --project envs/sim_arrhenius --no-sync python - "$RD" <<'PYM'
import sys; sys.path.insert(0,"grasp_synthesis")
from pathlib import Path
import collect_demos_synth as v1
p=v1._merge_shards(Path(sys.argv[1]))
import pickle; d=pickle.load(open(p,"rb"))
print("  merged", p, "->", len(d) if isinstance(d,list) else "?", "episodes")
PYM
  fi
done

env -u PYTHONPATH -u ROS_DISTRO MUJOCO_GL=egl uv run --project envs/sim_arrhenius --no-sync python \
  grasp_synthesis/collect_demos_diverse_start_v2.py \
  --experiment "$EXPERIMENT" --task-name "$TASK_DIR" \
  --n-episodes "$N_EPISODES" --n-envs "$N_ENVS" --maxfevals "$MAXFEVALS" \
  --scene-dr-every "$SCENE_DR_EVERY" --seed "$SEED" --start-modes "$START_MODES" \
  --record-video --out-dir "$REPO/dataset/demos"

RUN=$(ls -dt "$REPO"/dataset/demos/$TASK_DIR/* | head -1)
echo "[collect] DONE -> $RUN"
cat "$RUN/stats.yaml" || true
echo "$RUN" > "$REPO/logs/slurm_logs/last_banana_collect_run.txt"
