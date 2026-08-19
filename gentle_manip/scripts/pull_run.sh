#!/bin/bash
# [local] Download a DPPO run's checkpoint(s) + .hydra config from the cluster for local eval.
# Run this ON YOUR LOCAL MACHINE (not on the cluster) -- it rsyncs FROM the remote host TO here.
#
# Usage:
#   ./gentle_manip/scripts/pull_run.sh <run_id> [--ckpt 400,800] [--dest DIR] [--host HOST] [--all]
#
# <run_id>       5-letter run ID (looked up in experiments.csv on the remote) OR a full remote
#                path to the run dir (must contain a '/').
# --ckpt N,N,... only pull these checkpoint epochs (default: only the LATEST checkpoint present).
# --all          pull every checkpoint in the run's checkpoint/ dir.
# --dest DIR     local destination root (default: ./downloaded_runs/<run_id>).
# --host HOST    SSH alias for the cluster (default: $GM_REMOTE_HOST env var, or the ~/.ssh/config
#                Host below -- override either if your alias differs).
#
# Always pulls .hydra/ (exact resolved config the checkpoint was trained/evaled with) and
# EXPERIMENT.md if present. Uses rsync (resumable, only transfers what changed on re-run).
set -euo pipefail

REMOTE_HOST="${GM_REMOTE_HOST:-arrhenius1.hpc.arrhenius.naiss.se}"
REMOTE_REPO="/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip"
CKPTS=""
DEST=""
PULL_ALL=0
RUN_ID=""

while [ $# -gt 0 ]; do
    case "$1" in
        --ckpt) CKPTS="$2"; shift 2 ;;
        --dest) DEST="$2"; shift 2 ;;
        --host) REMOTE_HOST="$2"; shift 2 ;;
        --all) PULL_ALL=1; shift ;;
        -h|--help) grep '^#' "$0" | sed 's/^#//'; exit 0 ;;
        *) RUN_ID="$1"; shift ;;
    esac
done
[ -n "$RUN_ID" ] || { echo "usage: $0 <run_id> [--ckpt N,N] [--dest DIR] [--host HOST] [--all]" >&2; exit 1; }

# Resolve run_id -> remote run dir. IDs are minted globally-unique (new_id() checks the whole
# table, not per-task), so this SHOULD be a single row -- but don't just take the first match
# and guess: if the table ever ends up with more than one (e.g. an ID reused after its original
# row was dropped by reconcile_experiments, or manual edits), fail loudly and list every
# candidate rather than silently pulling the wrong run's checkpoint.
if [[ "$RUN_ID" == */* ]]; then
    RUN_DIR="$RUN_ID"
else
    MATCHES=$(ssh "$REMOTE_HOST" "grep '^${RUN_ID},' '$REMOTE_REPO/experiments.csv'")
    N_MATCHES=$(echo -n "$MATCHES" | grep -c '^' 2>/dev/null || echo 0)
    if [ -z "$MATCHES" ]; then
        echo "run '$RUN_ID' not found in experiments.csv on $REMOTE_HOST" >&2
        exit 1
    elif [ "$N_MATCHES" -gt 1 ]; then
        echo "AMBIGUOUS: '$RUN_ID' matches $N_MATCHES rows in experiments.csv -- refusing to guess:" >&2
        echo "$MATCHES" >&2
        echo "Pass the full remote path instead of the bare ID to disambiguate." >&2
        exit 1
    fi
    RUN_DIR=$(echo "$MATCHES" | cut -d, -f5)
fi
echo "[pull_run] $RUN_ID -> $REMOTE_HOST:$RUN_DIR"

DEST="${DEST:-./downloaded_runs/$(basename "$RUN_ID")}"
mkdir -p "$DEST"

echo "[pull_run] .hydra/ ..."
rsync -avzP "$REMOTE_HOST:$RUN_DIR/.hydra/" "$DEST/.hydra/"

if ssh "$REMOTE_HOST" "[ -f '$RUN_DIR/EXPERIMENT.md' ]" 2>/dev/null; then
    echo "[pull_run] EXPERIMENT.md ..."
    rsync -avzP "$REMOTE_HOST:$RUN_DIR/EXPERIMENT.md" "$DEST/EXPERIMENT.md"
fi

mkdir -p "$DEST/checkpoint"
if [ "$PULL_ALL" = "1" ]; then
    echo "[pull_run] checkpoint/ (all) ..."
    rsync -avzP "$REMOTE_HOST:$RUN_DIR/checkpoint/" "$DEST/checkpoint/"
elif [ -n "$CKPTS" ]; then
    IFS=',' read -ra EPOCHS <<< "$CKPTS"
    for ep in "${EPOCHS[@]}"; do
        echo "[pull_run] checkpoint/state_${ep}.pt ..."
        rsync -avzP "$REMOTE_HOST:$RUN_DIR/checkpoint/state_${ep}.pt" "$DEST/checkpoint/"
    done
else
    LATEST=$(ssh "$REMOTE_HOST" "ls -v '$RUN_DIR/checkpoint/' 2>/dev/null | tail -1")
    [ -n "$LATEST" ] || { echo "no checkpoints found under $RUN_DIR/checkpoint/" >&2; exit 1; }
    echo "[pull_run] checkpoint/$LATEST (latest; pass --ckpt or --all for more) ..."
    rsync -avzP "$REMOTE_HOST:$RUN_DIR/checkpoint/$LATEST" "$DEST/checkpoint/"
fi

echo "[pull_run] done -> $DEST"
