#!/bin/bash
# [arrhenius] Checkpoint watcher: polls a running BC-pretrain job for state_<EP>.pt checkpoints
# and submits a canonical eval job for each as it appears (SIM_EXPERIMENT/EVAL_EXPERIMENT =
# EXPERIMENT, CFG_DIR = the dataset's own eval_diffusion_pointnet.yaml -- n_episodes=200,
# record_batches=null already baked into that config). Also hang-detects its own submitted
# eval jobs (a stalled scene-rebuild sim server never reaching SIM_SERVER_READY -> summary.json)
# and auto-resubmits, so the whole thing survives fully unattended overnight.
#
# Submit (usually launched BY convert_and_pretrain.sbatch, not by hand):
#   PRETRAIN_JID=<jid> DPPO_CFG_DIR=<dir> EXPERIMENT=<name> TARGETS="200 400 600 800" \
#     sbatch --dependency=after:<jid> gentle_manip/scripts/arrhenius/pretrain_eval_watch.sbatch
#
#SBATCH -J pretrain_watch
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -A naiss2026-3-141-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip/logs/slumr_logs/%j.out
#SBATCH --error=/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip/logs/slumr_logs/%j.err

set -uo pipefail   # no -e: this is a long poll loop, one bad iteration must not kill the watcher
export PATH="$HOME/.local/bin:$PATH"
REPO=/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip
cd "$REPO"

PRETRAIN_JID=${PRETRAIN_JID:?set PRETRAIN_JID=<pretrain sbatch job id>}
DPPO_CFG_DIR=${DPPO_CFG_DIR:?set DPPO_CFG_DIR=<gentle_manip/dppo/cfg/* dir name>}
EXPERIMENT=${EXPERIMENT:?set EXPERIMENT=<configs/experiments/*.yaml name>}
TARGETS=${TARGETS:-"200 400 600 800"}
STALL=420   # sim-log idle (s) after SIM_SERVER_READY, no summary.json => hung

NORM=$REPO/dataset/dppo/$DPPO_CFG_DIR/normalization.npz
CFG_ABS_DIR=$REPO/gentle_manip/dppo/cfg/$DPPO_CFG_DIR
N_TARGETS=$(echo "$TARGETS" | wc -w)

echo "[watch] host=$(hostname) job=${SLURM_JOB_ID:-NA} pretrain_jid=$PRETRAIN_JID cfg_dir=$DPPO_CFG_DIR experiment=$EXPERIMENT targets=[$TARGETS]"

# ---- phase 1: find the pretrain run dir (from its OWN trainer log) -----------------------
RD=""
for _ in $(seq 1 60); do
    RD=$(grep -oE "/nobackup/[^ ]+/dppo-pretrain/${DPPO_CFG_DIR}/[^ ]+/config/" \
         "$REPO/logs/slumr_logs/${PRETRAIN_JID}_pretrain.log" 2>/dev/null | head -1 | sed 's#/config/##')
    [ -n "$RD" ] && break
    st=$(squeue -j "$PRETRAIN_JID" -h -o '%T' 2>/dev/null)
    [ -z "$st" ] && { echo "[watch] pretrain job ended before run dir appeared -- aborting"; exit 1; }
    sleep 15
done
[ -n "$RD" ] || { echo "[watch] never found pretrain run dir -- aborting"; exit 1; }
echo "[watch] pretrain run dir: $RD"

# ---- phase 2: submit an eval as each target checkpoint appears ---------------------------
declare -A EVAL_JOB   # target-epoch -> current eval job id (for hang-detection resubmit)
declare -A RETRY
SUBMITTED=""
GRACE=0

submit_eval() {
    local EP="$1"
    local CK="$RD/checkpoint/state_${EP}.pt"
    local SUB="state_${EP}_eval"
    local JID
    JID=$(CKPT="$CK" NORM="$NORM" FT_DENOISING=0 \
          CFG_DIR="$CFG_ABS_DIR" CFG_NAME=eval_diffusion_pointnet \
          EVAL_SUBDIR="$SUB" SIM_EXPERIMENT="$EXPERIMENT" EVAL_EXPERIMENT="$EXPERIMENT" \
          sbatch --parsable -J "eval_${DPPO_CFG_DIR}_${EP}" -t 04:00:00 \
          "$REPO/gentle_manip/scripts/arrhenius/dppo_eval.sbatch")
    EVAL_JOB[$EP]="$JID"
    echo "[watch] state_${EP}.pt -> eval job $JID (subdir=$SUB)"
}

for tick in $(seq 1 2000); do
    train_running=0
    squeue -j "$PRETRAIN_JID" -h -o '%T' 2>/dev/null | grep -q . && train_running=1

    for EP in $TARGETS; do
        case " $SUBMITTED " in *" $EP "*) continue;; esac
        [ -f "$RD/checkpoint/state_${EP}.pt" ] || continue
        submit_eval "$EP"
        SUBMITTED="$SUBMITTED $EP"
    done

    # ---- hang-detection on already-submitted evals -----
    for EP in "${!EVAL_JOB[@]}"; do
        JID=${EVAL_JOB[$EP]}
        STATE=$(squeue -j "$JID" -h -o '%T' 2>/dev/null)
        [ "$STATE" = "RUNNING" ] || continue
        OUT="$REPO/logs/slumr_logs/${JID}.out"
        SL=$(ls "$REPO/logs/slumr_logs/${JID}"_*_simserver.log 2>/dev/null | head -1)
        [ -n "$SL" ] || continue
        grep -q SIM_SERVER_READY "$SL" 2>/dev/null || continue
        EVDIR=$(grep -oE 'unique eval logdir -> [^ ]+' "$OUT" 2>/dev/null | head -1 | awk '{print $NF}')
        [ -n "$EVDIR" ] || continue
        [ -f "$EVDIR/summary.json" ] && continue
        idle=$(( $(date +%s) - $(stat -c %Y "$SL" 2>/dev/null || echo $(date +%s)) ))
        [ "$idle" -gt "$STALL" ] || continue
        RETRY[$EP]=$(( ${RETRY[$EP]:-0} + 1 ))
        if [ "${RETRY[$EP]}" -gt 3 ]; then
            echo "[watch] state_${EP} eval stalled again but retries exhausted -- giving up on it"
            continue
        fi
        echo "[watch] HUNG eval for ckpt $EP (job $JID, idle=${idle}s) -> scancel+resubmit (retry ${RETRY[$EP]})"
        scancel "$JID" 2>/dev/null
        rm -rf "$EVDIR"
        sleep 3
        submit_eval "$EP"
    done

    [ "$(echo $SUBMITTED | wc -w)" -ge "$N_TARGETS" ] || true
    if [ "$train_running" = "0" ]; then
        if [ "$GRACE" = 0 ]; then GRACE=1; sleep 30; continue; fi   # final-checkpoint-write race
        if [ "$(echo $SUBMITTED | wc -w)" -lt "$N_TARGETS" ]; then
            echo "[watch] pretrain ended; submitted $(echo $SUBMITTED|wc -w)/$N_TARGETS checkpoints (missing ones never appeared)"
        fi
        # keep looping a while longer purely for hang-detection on the evals already submitted
    fi

    # exit once all targets submitted AND all their eval jobs have left the queue
    if [ "$(echo $SUBMITTED | wc -w)" -ge "$N_TARGETS" ]; then
        any_running=0
        for JID in "${EVAL_JOB[@]}"; do
            squeue -j "$JID" -h -o '%T' 2>/dev/null | grep -q . && any_running=1
        done
        [ "$any_running" = 0 ] && { echo "[watch] all evals submitted and finished -- done"; break; }
    fi
    sleep 90
done

echo ""
echo "=== $DPPO_CFG_DIR eval results (exp=$EXPERIMENT, n_episodes=200, group=4 shapes) ==="
for EP in $TARGETS; do
    D="$RD/eval/state_${EP}_eval"
    if [ -f "$D/summary.json" ]; then
        python3 "$REPO/gentle_manip/scripts/summary_to_tsv.py" "$D" --no-header 2>/dev/null | cut -f1,3
    else
        echo "  state_$EP: (no result)"
    fi
done
