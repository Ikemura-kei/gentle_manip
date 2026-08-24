#!/bin/bash
# [arrhenius] Grasp-synthesis demo collection (v3, FEM gentleness metric) on the GH200 (aarch64)
# nodes. One process: envs/sim_arrhenius running grasp_synthesis/collect_demos_synth_v3.py.
# See docs/cluster_data_collection.md for the full flag reference / what to expect.
#
# Submit:  EXPERIMENT=<name> sbatch gentle_manip/scripts/arrhenius/collect_demos_synth.sbatch
#   knobs (env):  EXPERIMENT (required), N_EPISODES, N_ENVS, TASK_NAME (output-folder override),
#                 RECORD_VIDEO, MAXFEVALS, SCENE_DR_EVERY, SEED, N_HOME_TO_PRE, N_GRASP,
#                 GRASP_EXTRA_CLOSE, OUT_DIR
#
#SBATCH -J grasp_synth
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
MAXFEVALS=${MAXFEVALS:-}          # empty -> let the collector's own default (1145) apply
SCENE_DR_EVERY=${SCENE_DR_EVERY:-}  # empty -> let the collector's own default (1) apply
SEED=${SEED:-0}
N_HOME_TO_PRE=${N_HOME_TO_PRE:-}  # empty -> let the collector's own default (98) apply; overrides the
                                  # home->pre-grasp approach-phase length (--n-home-to-pre), recorded
                                  # into the run's config.yaml (control.n_home_to_pre) when set
N_GRASP=${N_GRASP:-}             # empty -> let the collector's own default (37) apply; overrides the
                                  # gripper-close phase length (--n-grasp)
GRASP_EXTRA_CLOSE=${GRASP_EXTRA_CLOSE:-}  # empty -> let the collector's own default (0.0, no change)
                                  # apply; squeeze this many meters tighter than the synthesized
                                  # width for every grasp (--grasp-extra-close)
DESCRIPTION=${DESCRIPTION:-}     # free-text note recorded into the run's config.yaml (description)
OUT_DIR=${OUT_DIR:-$REPO/dataset/demos}
# TASK_NAME overrides only the OUTPUT FOLDER name (docs/cluster_data_collection.md) -- use it to
# keep the canonical single_lift_mushroom_soft/ folder for a camera-study fork experiment whose
# own task cfg has a variant name (e.g. single_lift_mushroom_soft_highcam).
TASK_NAME=${TASK_NAME:-}

echo "[sbatch] host=$(hostname) arch=$(uname -m) job=${SLURM_JOB_ID:-NA}"
echo "[sbatch] experiment=$EXPERIMENT n_episodes=$N_EPISODES n_envs=$N_ENVS record_video=$RECORD_VIDEO task_name=${TASK_NAME:-<default>} n_home_to_pre=${N_HOME_TO_PRE:-<default>} n_grasp=${N_GRASP:-<default>} grasp_extra_close=${GRASP_EXTRA_CLOSE:-<default>}"

ARGS=(--experiment "$EXPERIMENT" --n-episodes "$N_EPISODES" --n-envs "$N_ENVS" --grasp-gpu
      --record-video "$RECORD_VIDEO" --seed "$SEED" --out-dir "$OUT_DIR")
[ -n "$MAXFEVALS" ] && ARGS+=(--maxfevals "$MAXFEVALS")
[ -n "$SCENE_DR_EVERY" ] && ARGS+=(--scene-dr-every "$SCENE_DR_EVERY")
[ -n "$TASK_NAME" ] && ARGS+=(--task-name "$TASK_NAME")
[ -n "$N_HOME_TO_PRE" ] && ARGS+=(--n-home-to-pre "$N_HOME_TO_PRE")
[ -n "$N_GRASP" ] && ARGS+=(--n-grasp "$N_GRASP")
[ -n "$GRASP_EXTRA_CLOSE" ] && ARGS+=(--grasp-extra-close "$GRASP_EXTRA_CLOSE")
[ -n "$DESCRIPTION" ] && ARGS+=(--description "$DESCRIPTION")
# EXTRA_ARGS: free-form passthrough for newer collector flags (e.g. the v3.3 recipe's
# --approach-speed / --cam-azimuth-max-deg / --grasp-area-min-mm2 ...). Space-split.
[ -n "${EXTRA_ARGS:-}" ] && ARGS+=($EXTRA_ARGS)

unset PYTHONPATH ROS_DISTRO 2>/dev/null || true
export MUJOCO_GL=egl

COLLECT_LOG=$REPO/logs/slumr_logs/${SLURM_JOB_ID:-local}_collect.log
: > "$COLLECT_LOG"

echo "[sbatch] launching: python grasp_synthesis/collect_demos_synth_v3.py ${ARGS[*]}"
uv run --project envs/sim_arrhenius --no-sync python grasp_synthesis/collect_demos_synth_v3.py \
    "${ARGS[@]}" 2>&1 | tee "$COLLECT_LOG"
RC=${PIPESTATUS[0]}
echo "[sbatch] collection finished (job ${SLURM_JOB_ID:-NA}) rc=$RC"
exit $RC
