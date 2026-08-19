#!/bin/bash
# [local] Download a DPPO run's checkpoint(s), .hydra config, and/or eval results from the
# cluster for local inspection. Run this ON YOUR LOCAL MACHINE (not on the cluster) -- it
# rsyncs FROM the remote host TO here.
#
# Usage:
#   ./gentle_manip/scripts/pull_run.sh <run_id> [options]
#
# <run_id>          5-letter run ID (looked up in experiments.csv on the remote) OR a full
#                   remote path to the run dir (must contain a '/').
#
# Checkpoint options:
#   --ckpt N,N,...  only pull these checkpoint epochs (default: latest checkpoint present).
#   --all           pull every checkpoint in the run's checkpoint/ dir.
#   --no-ckpt       skip checkpoints entirely (e.g. for an eval-only pull).
#
# Eval options (eval subdir names vary, e.g. state_200_eval, state_200_eval_sgs2 -- see
# --list-evals):
#   --list-evals    list available eval/<name> dirs on the remote for this run, then exit
#                   (no download). Use this first if you don't know the exact name.
#   --eval NAME,... pull eval/<name>/ for each given name (summary.json + episodes.csv +
#                   config/ by default; videos excluded, see --with-videos).
#   --eval-all      pull every eval/<name>/ dir for this run.
#   --with-videos   also include render/*.mp4 when pulling eval dir(s) (can be large).
#
# --dest DIR        local destination root (default: ./downloaded_runs/<run_id>).
# --host HOST       SSH alias for the cluster (default: $GM_REMOTE_HOST env var, or the
#                   ~/.ssh/config Host below -- override either if your alias differs).
#
# Always pulls .hydra/ (exact resolved config) and EXPERIMENT.md if present, unless only
# --list-evals is given. Uses rsync (resumable, only transfers what changed on re-run).
set -euo pipefail

REMOTE_HOST="${GM_REMOTE_HOST:-arrhenius1.hpc.arrhenius.naiss.se}"
REMOTE_REPO="/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip"
CKPTS=""
DEST=""
PULL_ALL_CKPT=0
NO_CKPT=0
EVALS=""
EVAL_ALL=0
LIST_EVALS=0
WITH_VIDEOS=0
RUN_ID=""

while [ $# -gt 0 ]; do
    case "$1" in
        --ckpt) CKPTS="$2"; shift 2 ;;
        --all) PULL_ALL_CKPT=1; shift ;;
        --no-ckpt) NO_CKPT=1; shift ;;
        --eval) EVALS="$2"; shift 2 ;;
        --eval-all) EVAL_ALL=1; shift ;;
        --list-evals) LIST_EVALS=1; shift ;;
        --with-videos) WITH_VIDEOS=1; shift ;;
        --dest) DEST="$2"; shift 2 ;;
        --host) REMOTE_HOST="$2"; shift 2 ;;
        -h|--help) grep '^#' "$0" | sed 's/^#//'; exit 0 ;;
        *) RUN_ID="$1"; shift ;;
    esac
done
[ -n "$RUN_ID" ] || { grep '^#' "$0" | sed 's/^#//'; exit 1; }

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

if [ "$LIST_EVALS" = "1" ]; then
    ssh "$REMOTE_HOST" "ls -1 '$RUN_DIR/eval/' 2>/dev/null" || echo "(no eval/ dir found)"
    exit 0
fi

DEST="${DEST:-./downloaded_runs/$(basename "$RUN_ID")}"
mkdir -p "$DEST"

echo "[pull_run] .hydra/ ..."
rsync -avzP "$REMOTE_HOST:$RUN_DIR/.hydra/" "$DEST/.hydra/"

if ssh "$REMOTE_HOST" "[ -f '$RUN_DIR/EXPERIMENT.md' ]" 2>/dev/null; then
    echo "[pull_run] EXPERIMENT.md ..."
    rsync -avzP "$REMOTE_HOST:$RUN_DIR/EXPERIMENT.md" "$DEST/EXPERIMENT.md"
fi

if [ "$NO_CKPT" != "1" ]; then
    mkdir -p "$DEST/checkpoint"
    if [ "$PULL_ALL_CKPT" = "1" ]; then
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
        if [ -z "$LATEST" ]; then
            echo "no checkpoints found under $RUN_DIR/checkpoint/ (use --no-ckpt to skip)" >&2
        else
            echo "[pull_run] checkpoint/$LATEST (latest; pass --ckpt or --all for more) ..."
            rsync -avzP "$REMOTE_HOST:$RUN_DIR/checkpoint/$LATEST" "$DEST/checkpoint/"
        fi
    fi
fi

# ---- eval dir(s) --------------------------------------------------------------------------
RSYNC_EVAL_EXCL=()
[ "$WITH_VIDEOS" != "1" ] && RSYNC_EVAL_EXCL=(--exclude 'render/')

pull_eval_dir() {
    local name="$1"
    echo "[pull_run] eval/${name}/ ...$([ "$WITH_VIDEOS" != "1" ] && echo ' (videos excluded, use --with-videos)')"
    mkdir -p "$DEST/eval/$name"
    rsync -avzP "${RSYNC_EVAL_EXCL[@]}" "$REMOTE_HOST:$RUN_DIR/eval/$name/" "$DEST/eval/$name/"
}

if [ "$EVAL_ALL" = "1" ]; then
    NAMES=$(ssh "$REMOTE_HOST" "ls -1 '$RUN_DIR/eval/' 2>/dev/null")
    [ -n "$NAMES" ] || echo "no eval/ dir found on remote" >&2
    while IFS= read -r n; do
        [ -n "$n" ] && pull_eval_dir "$n"
    done <<< "$NAMES"
elif [ -n "$EVALS" ]; then
    IFS=',' read -ra NAMES <<< "$EVALS"
    for n in "${NAMES[@]}"; do
        pull_eval_dir "$n"
    done
fi

echo "[pull_run] done -> $DEST"
