#!/usr/bin/env bash
# [script] Local sequential pipeline: wait for collection → convert → BC pretrain 4000 ep
#          → eval ckpt-2000 → eval ckpt-4000.
# Used by: manual launch on the lab workstation
# Status: active
#
# Run from the repo root:
#   bash gentle_manip/scripts/train_eval_rigid_bc_local.sh
#
# Env-overridable knobs:
#   DEMO_RUN   — run dir under dataset/demos/single_lift_mushroom_rigid/ (default: 26-07-25-kqs)
#   PORT       — sim server port (default 5570)
#   WANDB_MODE — offline (default; set to online to stream to wandb)

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# ── Knobs ──────────────────────────────────────────────────────────────────────
REPO="$(pwd)"
DEMO_RUN="${DEMO_RUN:-26-07-25-kqs}"
DEMO_DIR="$REPO/dataset/demos/single_lift_mushroom_rigid/$DEMO_RUN"
DATA_ENV="single_lift_mushroom_rigid_pcd"
DATA_DIR="${DPPO_DATA_DIR:-$REPO/dataset/dppo}"
LOG_DIR="${DPPO_LOG_DIR:-$REPO/logs/dppo}"
CFG_DIR="$REPO/gentle_manip/dppo/cfg/single_lift_mushroom_rigid_pcd"
EXPERIMENT="single_lift_mushroom_rigid_eval"
PORT="${PORT:-5570}"
SRV_LOG="$REPO/logs/sim_server_rigid_bc_eval.log"
export WANDB_MODE="${WANDB_MODE:-online}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export DPPO_DATA_DIR="$DATA_DIR"
export DPPO_LOG_DIR="$LOG_DIR"

echo "================================================================"
echo " Rigid BC local pipeline"
echo "   demo:     $DEMO_DIR"
echo "   data_env: $DATA_ENV"
echo "   data_dir: $DATA_DIR/$DATA_ENV"
echo "   log_dir:  $LOG_DIR/dppo-pretrain/$DATA_ENV/<id>"
echo "   port:     $PORT"
echo "================================================================"

# ── 1. Wait for data collection to finish ──────────────────────────────────────
DATA_PKL="$DEMO_DIR/data.pkl"
echo ""
echo "[1/4] Waiting for data collection to produce $DATA_PKL ..."
while [ ! -f "$DATA_PKL" ]; do
    NSHARDS=$(ls "$DEMO_DIR"/shard_*.pkl 2>/dev/null | wc -l || echo 0)
    echo "      still collecting — $NSHARDS shards so far; sleeping 60s ..."
    sleep 60
done
echo "      data.pkl found — collection done."
python3 -c "
import pickle
d = pickle.load(open('$DATA_PKL','rb'))
print(f'      episodes: {d[\"meta\"][\"n_episodes\"]}  |  created: {d[\"meta\"][\"created\"]}')
"

# ── 2. Convert demos → train.npz / val.npz / normalization.npz ────────────────
echo ""
echo "[2/4] Converting demos → DPPO format ($DATA_DIR/$DATA_ENV) ..."
uv run --project envs/dppo python -m gentle_manip.dppo.convert_demos \
    "$DATA_PKL" \
    --out "$DATA_DIR/$DATA_ENV" \
    --point-cloud --val-split 0.1
echo "      convert done."

# ── 3. BC pretrain (4000 epochs, save every 500) ──────────────────────────────
echo ""
echo "[3/4] BC pretraining — 4000 epochs, ckpt every 500 ..."
uv run --project envs/dppo python -m gentle_manip.dppo.train \
    --config-path "$CFG_DIR" \
    --config-name pre_diffusion_pointnet

# Find the run dir created by this pretrain (newest under dppo-pretrain/$DATA_ENV/)
RUN_DIR=$(ls -td "$LOG_DIR/dppo-pretrain/$DATA_ENV"/*/  2>/dev/null | head -1)
RUN_DIR="${RUN_DIR%/}"   # strip trailing slash
echo "      pretrain done — run dir: $RUN_DIR"

CKPT_DIR="$RUN_DIR/checkpoint"
CKPT_2000="$CKPT_DIR/state_2000.pt"
CKPT_4000="$CKPT_DIR/state_4000.pt"
NORM="$DATA_DIR/$DATA_ENV/normalization.npz"

[ -f "$CKPT_2000" ] || { echo "ERROR: $CKPT_2000 not found"; exit 1; }
[ -f "$CKPT_4000" ] || { echo "ERROR: $CKPT_4000 not found"; exit 1; }
[ -f "$NORM"      ] || { echo "ERROR: $NORM not found"; exit 1; }

# ── Helper: run one eval (start server, wait, eval, kill server) ───────────────
run_eval() {
    local CKPT="$1"
    local SUBDIR="$2"
    echo ""
    echo "  [eval] checkpoint: $CKPT"
    echo "  [eval] output dir: $RUN_DIR/eval/$SUBDIR"

    # Start sim server in background
    echo "  [eval] starting sim server on port $PORT ..."
    uv run --project envs/sim python -m gentle_manip.scripts.serl_sim_server \
        --experiment "$EXPERIMENT" --view student \
        --num-envs 5 --render-rgb \
        --port "$PORT" \
        > "$SRV_LOG" 2>&1 &
    SRV_PID=$!

    # Trap ensures server is killed even if eval fails
    cleanup_server() {
        echo "  [eval] stopping sim server ($SRV_PID) ..."
        kill "$SRV_PID" 2>/dev/null || true
        wait "$SRV_PID" 2>/dev/null || true
    }
    trap cleanup_server EXIT INT TERM

    # Wait for sim server to be ready (up to 5 min)
    echo "  [eval] waiting for SIM_SERVER_READY ..."
    for _ in $(seq 1 60); do
        grep -q "SIM_SERVER_READY" "$SRV_LOG" 2>/dev/null && { echo "  [eval] server ready."; break; }
        kill -0 "$SRV_PID" 2>/dev/null || { echo "  [eval] ERROR: server died — last log:"; tail -30 "$SRV_LOG"; exit 1; }
        sleep 5
    done
    grep -q "SIM_SERVER_READY" "$SRV_LOG" || { echo "  [eval] ERROR: server not ready after 5 min"; tail -30 "$SRV_LOG"; exit 1; }

    # Run canonical eval
    uv run --project envs/dppo python -m gentle_manip.dppo.train \
        --config-path "$CFG_DIR" \
        --config-name eval_diffusion_pointnet \
        base_policy_path="$CKPT" \
        normalization_path="$NORM" \
        experiment="$EXPERIMENT" \
        ft_denoising_steps=0 \
        record_batches=null \
        logdir="$RUN_DIR/eval/$SUBDIR" \
        env.specific.port="$PORT"

    echo "  [eval] done — results: $RUN_DIR/eval/$SUBDIR/summary.json"
    cleanup_server
    trap - EXIT INT TERM
    sleep 5   # let port unbind before next server
}

# ── 4. Eval ckpt-2000 then ckpt-4000 (sequentially) ──────────────────────────
echo ""
echo "[4/4] Running sequential evals ..."
run_eval "$CKPT_2000" "epoch_2000"
run_eval "$CKPT_4000" "epoch_4000"

echo ""
echo "================================================================"
echo " All done."
echo "   pretrain:   $RUN_DIR"
echo "   eval 2000:  $RUN_DIR/eval/epoch_2000/summary.json"
echo "   eval 4000:  $RUN_DIR/eval/epoch_4000/summary.json"
echo "================================================================"
