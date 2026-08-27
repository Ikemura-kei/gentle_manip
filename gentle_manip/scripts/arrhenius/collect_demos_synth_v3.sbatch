#!/bin/bash
# [arrhenius] Grasp-synthesis demo collection (v3, FEM-based gentleness metric, optional GPU
# solver, configurable FSM phase lengths) on the GH200 (aarch64) nodes. One process:
# envs/sim_arrhenius running grasp_synthesis/collect_demos_synth_v3.py.
# See docs/cluster_data_collection.md for the v2-vs-v3 comparison.
#
# Submit:  EXPERIMENT=<name> sbatch gentle_manip/scripts/arrhenius/collect_demos_synth_v3.sbatch
#   knobs (env):  EXPERIMENT (required), N_EPISODES, N_ENVS, TASK_NAME (output-folder override),
#                 RECORD_VIDEO, MAXFEVALS, SCENE_DR_EVERY, SEED, OUT_DIR, GRASP_GPU (1=use GPU
#                 FEM solver), N_HOME_TO_PRE, N_GRASP, N_LIFT, N_FIRM (0=no firm phase),
#                 GRASP_EXTRA_CLOSE (meters, extra squeeze beyond the synthesized width)
#
#SBATCH -J grasp_synth_v3
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
RECORD_VIDEO=${RECORD_VIDEO:-50}
MAXFEVALS=${MAXFEVALS:-}            # empty -> collector's own default (1145)
SCENE_DR_EVERY=${SCENE_DR_EVERY:-}  # empty -> collector's own default (1)
SEED=${SEED:-0}
OUT_DIR=${OUT_DIR:-$REPO/dataset/demos}
TASK_NAME=${TASK_NAME:-}            # override only the OUTPUT FOLDER name (see v2 launcher note)
GRASP_GPU=${GRASP_GPU:-0}           # 1 = --grasp-gpu (GPU FEM solver; default CPU, off GPU contention)
N_HOME_TO_PRE=${N_HOME_TO_PRE:-}    # empty -> collector's own module default
N_GRASP=${N_GRASP:-}
N_LIFT=${N_LIFT:-}
N_FIRM=${N_FIRM:-}                  # 0 = no firm phase at all (matches pre-firm v1/cho)
GRASP_EXTRA_CLOSE=${GRASP_EXTRA_CLOSE:-}   # meters, extra squeeze beyond synthesized width

echo "[sbatch] host=$(hostname) arch=$(uname -m) job=${SLURM_JOB_ID:-NA}"
echo "[sbatch] experiment=$EXPERIMENT n_episodes=$N_EPISODES n_envs=$N_ENVS record_video=$RECORD_VIDEO task_name=${TASK_NAME:-<default>}"
echo "[sbatch] grasp_gpu=$GRASP_GPU n_home_to_pre=${N_HOME_TO_PRE:-<default>} n_grasp=${N_GRASP:-<default>} n_lift=${N_LIFT:-<default>} n_firm=${N_FIRM:-<default>} grasp_extra_close=${GRASP_EXTRA_CLOSE:-<default>}"

ARGS=(--experiment "$EXPERIMENT" --n-episodes "$N_EPISODES" --n-envs "$N_ENVS"
      --record-video "$RECORD_VIDEO" --seed "$SEED" --out-dir "$OUT_DIR")
[ -n "$MAXFEVALS" ] && ARGS+=(--maxfevals "$MAXFEVALS")
[ -n "$SCENE_DR_EVERY" ] && ARGS+=(--scene-dr-every "$SCENE_DR_EVERY")
[ -n "$TASK_NAME" ] && ARGS+=(--task-name "$TASK_NAME")
[ "$GRASP_GPU" = "1" ] && ARGS+=(--grasp-gpu)
[ -n "$N_HOME_TO_PRE" ] && ARGS+=(--n-home-to-pre "$N_HOME_TO_PRE")
[ -n "$N_GRASP" ] && ARGS+=(--n-grasp "$N_GRASP")
[ -n "$N_LIFT" ] && ARGS+=(--n-lift "$N_LIFT")
[ -n "$N_FIRM" ] && ARGS+=(--n-firm "$N_FIRM")
[ -n "$GRASP_EXTRA_CLOSE" ] && ARGS+=(--grasp-extra-close "$GRASP_EXTRA_CLOSE")

unset PYTHONPATH ROS_DISTRO 2>/dev/null || true
export MUJOCO_GL=egl

COLLECT_LOG=$REPO/logs/slumr_logs/${SLURM_JOB_ID:-local}_collect.log
: > "$COLLECT_LOG"

# EXTRA_ARGS passthrough (2026-08-27). Without it this sbatch SILENTLY IGNORED any flag not
# backed by a named env var — two raspberry collections ran with defaults while appearing to use
# --grasp-area-min-mm2/--grasp-width-max-mm/--grasp-yaw-max-deg. Same silent-drop class as the
# rpc info whitelist (B13) and the NORM default (B1): the run completes and looks fine.
if [ -n "${EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    ARGS+=($EXTRA_ARGS)
fi
echo "[sbatch] launching: python grasp_synthesis/collect_demos_synth_v3.py ${ARGS[*]}"
uv run --project envs/sim_arrhenius --no-sync python grasp_synthesis/collect_demos_synth_v3.py \
    "${ARGS[@]}" 2>&1 | tee "$COLLECT_LOG"
RC=${PIPESTATUS[0]}
echo "[sbatch] collection finished (job ${SLURM_JOB_ID:-NA}) rc=$RC"
exit $RC
