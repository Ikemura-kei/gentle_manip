#!/bin/bash
# [arrhenius] Orchestrator: wait for a demo-collection job to finish (via --dependency=afterany
# set at SUBMIT time, not here), sanity-check the result, convert to DPPO npz, launch the BC
# pretrain job, and launch a checkpoint-watcher job that auto-evals as checkpoints appear.
# Entirely SLURM-native (job dependencies + this job's own sbatch calls) so the whole chain
# survives independent of any interactive session -- safe to fire and walk away from.
#
# Submit:
#   COLLECT_RUN_DIR=dataset/demos/single_lift_mushroom_soft/26-08-14-xnh \
#   EXPERIMENT=single_lift_mushroom_soft_abs_action \
#   DPPO_CFG_DIR=single_lift_mushroom_soft_abs_pcd \
#   sbatch --dependency=afterany:<collect_job_id> --parsable \
#     gentle_manip/scripts/arrhenius/convert_and_pretrain.sbatch
#
#SBATCH -J convert_train
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -A naiss2026-3-141-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip/logs/slumr_logs/%j.out
#SBATCH --error=/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip/logs/slumr_logs/%j.err

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
REPO=/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip
cd "$REPO"

COLLECT_JOB_ID=${COLLECT_JOB_ID:-}
COLLECT_RUN_DIR=${COLLECT_RUN_DIR:-}
if [ -z "$COLLECT_RUN_DIR" ]; then
    [ -n "$COLLECT_JOB_ID" ] || { echo "[orchestrate] ABORT: set COLLECT_RUN_DIR or COLLECT_JOB_ID"; exit 1; }
    # collect_demos_synth_v3.py prints "  Data   -> <run_dir>/data.pkl  (shards flushed ...)"
    # near the very start of its own log -- discover the run dir it picked automatically.
    for _ in $(seq 1 60); do
        COLLECT_RUN_DIR=$(grep -oE 'Data +→ [^ ]+/data\.pkl' \
            "$REPO/logs/slumr_logs/${COLLECT_JOB_ID}_collect.log" 2>/dev/null \
            | head -1 | awk '{print $3}' | sed 's#/data\.pkl$##')
        [ -n "$COLLECT_RUN_DIR" ] && break
        sleep 5
    done
    [ -n "$COLLECT_RUN_DIR" ] || { echo "[orchestrate] ABORT: could not discover run dir from collect job $COLLECT_JOB_ID's log"; exit 1; }
    echo "[orchestrate] discovered COLLECT_RUN_DIR=$COLLECT_RUN_DIR (from collect job $COLLECT_JOB_ID)"
fi
EXPERIMENT=${EXPERIMENT:?set EXPERIMENT=<configs/experiments/*.yaml name>}
DPPO_CFG_DIR=${DPPO_CFG_DIR:?set DPPO_CFG_DIR=<gentle_manip/dppo/cfg/* dir name>}
N_EPOCHS=${N_EPOCHS:-800}
SAVE_FREQ=${SAVE_FREQ:-200}
TARGETS=${TARGETS:-"200 400 600 800"}

echo "[orchestrate] host=$(hostname) job=${SLURM_JOB_ID:-NA}"
echo "[orchestrate] collect_run_dir=$COLLECT_RUN_DIR experiment=$EXPERIMENT dppo_cfg_dir=$DPPO_CFG_DIR"

# ---- sanity-check the collection actually produced data ---------------------------------
DATA_PKL="$COLLECT_RUN_DIR/data.pkl"
if [ ! -f "$DATA_PKL" ]; then
    echo "[orchestrate] ABORT: $DATA_PKL missing -- collection job likely failed or was killed"
    echo "[orchestrate] stats.yaml (if any):"; cat "$COLLECT_RUN_DIR/stats.yaml" 2>/dev/null || echo "  (none)"
    exit 1
fi
if [ -f "$COLLECT_RUN_DIR/stats.yaml" ]; then
    echo "[orchestrate] collection stats:"; cat "$COLLECT_RUN_DIR/stats.yaml"
fi

unset PYTHONPATH ROS_DISTRO 2>/dev/null || true

# ---- convert demos -> DPPO npz (pure numpy, no GPU needed, but the job already has one) --
DPPO_DATA_DIR=$REPO/dataset/dppo
echo "[orchestrate] converting -> $DPPO_DATA_DIR/$DPPO_CFG_DIR"
uv run --project envs/dppo_arrhenius --no-sync python -m gentle_manip.dppo.convert_demos \
    "$DATA_PKL" --out "$DPPO_DATA_DIR/$DPPO_CFG_DIR" \
    --experiment "$EXPERIMENT" --view student --point-cloud --val-split 0.1
[ -f "$DPPO_DATA_DIR/$DPPO_CFG_DIR/train.npz" ] || { echo "[orchestrate] ABORT: conversion did not produce train.npz"; exit 1; }

# ---- launch BC pretrain (own GPU job; online wandb is dppo_pretrain.sbatch's own default) --
CFG_ABS_DIR=$REPO/gentle_manip/dppo/cfg/$DPPO_CFG_DIR
PRETRAIN_JID=$(DATA_ENV="$DPPO_CFG_DIR" N_EPOCHS="$N_EPOCHS" SAVE_FREQ="$SAVE_FREQ" \
    CFG_DIR="$CFG_ABS_DIR" CFG_NAME=pre_diffusion_pointnet \
    GM_MOTIVATION="BC-pretrain a PointNet diffusion policy (ABS action) on the newly-collected $EXPERIMENT dataset ($COLLECT_RUN_DIR). Launched automatically by convert_and_pretrain.sbatch (job ${SLURM_JOB_ID:-NA}) once collection finished." \
    GM_HYPOTHESIS="First policy trained on this exact grasp-synthesis collection; baseline for the rand-scene/DR eval." \
    sbatch --parsable "$REPO/gentle_manip/scripts/arrhenius/dppo_pretrain.sbatch")
echo "[orchestrate] pretrain launched: job $PRETRAIN_JID"

# ---- launch the checkpoint watcher (starts once pretrain has actually begun running) -----
WATCH_JID=$(PRETRAIN_JID="$PRETRAIN_JID" DPPO_CFG_DIR="$DPPO_CFG_DIR" EXPERIMENT="$EXPERIMENT" \
    TARGETS="$TARGETS" \
    sbatch --parsable --dependency=after:"$PRETRAIN_JID" -J "watch_${DPPO_CFG_DIR}" \
    "$REPO/gentle_manip/scripts/arrhenius/pretrain_eval_watch.sbatch")
echo "[orchestrate] checkpoint watcher launched: job $WATCH_JID (depends on pretrain having started)"
echo "[orchestrate] done -- pretrain=$PRETRAIN_JID watcher=$WATCH_JID"
