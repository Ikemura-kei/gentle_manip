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

N_EPISODES=${N_EPISODES:-500}
N_ENVS=${N_ENVS:-8}
MAXFEVALS=${MAXFEVALS:-700}
SEED=${SEED:-0}
START_MODES=${START_MODES:-home:0.25,near_object:0.30,near_ground:0.25,mid_air:0.20}

echo "[collect] n_episodes=$N_EPISODES n_envs=$N_ENVS maxfevals=$MAXFEVALS seed=$SEED"
echo "[collect] start_modes=$START_MODES"

env -u PYTHONPATH -u ROS_DISTRO MUJOCO_GL=egl uv run --project envs/sim_arrhenius --no-sync python \
  grasp_synthesis/collect_demos_diverse_start_v2.py \
  --experiment single_lift_banana_rigid_diverse \
  --n-episodes "$N_EPISODES" --n-envs "$N_ENVS" --maxfevals "$MAXFEVALS" \
  --scene-dr-every 1 --seed "$SEED" --start-modes "$START_MODES" \
  --record-video --out-dir "$REPO/dataset/demos"

RUN=$(ls -dt "$REPO"/dataset/demos/single_lift_banana_rigid/* | head -1)
echo "[collect] DONE -> $RUN"
cat "$RUN/stats.yaml" || true
echo "$RUN" > "$REPO/logs/slurm_logs/last_banana_collect_run.txt"
