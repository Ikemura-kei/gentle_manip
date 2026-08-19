#!/bin/bash
# [arrhenius] Grasp-synthesis demo collection (v1, original SDF-cost CMA-ES) on the GH200
# (aarch64) nodes. One process: envs/sim_arrhenius running grasp_synthesis/collect_demos_synth.py.
# v1 is the oldest collector (predates v2/v3) -- no --grasp-gpu, no --n-home-to-pre/--n-grasp/
# --grasp-extra-close (those were added to v2/v3 only), and --record-video is a bare boolean
# (all episodes or none -- no "first N" cap like v2/v3 have).
#
# Submit:  EXPERIMENT=<name> sbatch gentle_manip/scripts/arrhenius/collect_demos_synth_v1.sbatch
#   knobs (env):  EXPERIMENT (required), N_EPISODES, N_ENVS, TASK_NAME (output-folder override),
#                 RECORD_VIDEO (0/1, default 0 -- v1 has no first-N cap so default OFF to avoid
#                 recording all N_EPISODES clips), MAXFEVALS, SCENE_DR_EVERY, SEED, OUT_DIR
#
#SBATCH -J grasp_synth_v1
#SBATCH -N 1
#SBATCH -t 17:00:00
#SBATCH -A naiss2026-3-141-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip/logs/slumr_logs/%j.out
#SBATCH --error=/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip/logs/slumr_logs/%j.err

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
REPO=/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip
cd "$REPO"

# ---- knobs -------------------------------------------------------------------------
EXPERIMENT=${EXPERIMENT:?set EXPERIMENT=<configs/experiments/*.yaml name>}
N_EPISODES=${N_EPISODES:-650}
N_ENVS=${N_ENVS:-8}
RECORD_VIDEO=${RECORD_VIDEO:-0}   # 0/1 -- bare flag in v1, no first-N cap available
MAXFEVALS=${MAXFEVALS:-}          # empty -> let the collector's own default (901) apply
SCENE_DR_EVERY=${SCENE_DR_EVERY:-}  # empty -> let the collector's own default (1) apply
SEED=${SEED:-0}
DESCRIPTION=${DESCRIPTION:-}
OUT_DIR=${OUT_DIR:-$REPO/dataset/demos}
TASK_NAME=${TASK_NAME:-}

echo "[sbatch] host=$(hostname) arch=$(uname -m) job=${SLURM_JOB_ID:-NA}"
echo "[sbatch] experiment=$EXPERIMENT n_episodes=$N_EPISODES n_envs=$N_ENVS record_video=$RECORD_VIDEO task_name=${TASK_NAME:-<default>}"

# NOTE: no --grasp-gpu (v1 has no GPU FEM grasp search -- CPU-only CMA-ES over the SDF cost).
ARGS=(--experiment "$EXPERIMENT" --n-episodes "$N_EPISODES" --n-envs "$N_ENVS"
      --seed "$SEED" --out-dir "$OUT_DIR")
[ "$RECORD_VIDEO" = "1" ] && ARGS+=(--record-video)
[ -n "$MAXFEVALS" ] && ARGS+=(--maxfevals "$MAXFEVALS")
[ -n "$SCENE_DR_EVERY" ] && ARGS+=(--scene-dr-every "$SCENE_DR_EVERY")
[ -n "$TASK_NAME" ] && ARGS+=(--task-name "$TASK_NAME")
[ -n "$DESCRIPTION" ] && ARGS+=(--description "$DESCRIPTION")

unset PYTHONPATH ROS_DISTRO 2>/dev/null || true
export MUJOCO_GL=egl

COLLECT_LOG=$REPO/logs/slumr_logs/${SLURM_JOB_ID:-local}_collect.log
: > "$COLLECT_LOG"

echo "[sbatch] launching: python grasp_synthesis/collect_demos_synth.py ${ARGS[*]}"
uv run --project envs/sim_arrhenius --no-sync python grasp_synthesis/collect_demos_synth.py \
    "${ARGS[@]}" 2>&1 | tee "$COLLECT_LOG"
RC=${PIPESTATUS[0]}
echo "[sbatch] collection finished (job ${SLURM_JOB_ID:-NA}) rc=$RC"
exit $RC
