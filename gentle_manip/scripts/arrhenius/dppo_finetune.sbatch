#!/bin/bash
# [arrhenius] DPPO point-cloud finetune on the GH200 (aarch64) nodes.
# Two processes on ONE GPU: a genesis serl_sim_server (envs/sim_arrhenius) + the DPPO
# trainer (envs/dppo_arrhenius) talking over the rpc socket (127.0.0.1:PORT). This script
# starts the server, waits for SIM_SERVER_READY, runs the trainer, and kills the server on exit.
#
# Submit:  sbatch gentle_manip/scripts/arrhenius/dppo_finetune.sbatch
# Smoke test (short) is the default; set SMOKE=0 in the env / edit below for a full run.
#
#SBATCH -J dppo_ft
#SBATCH -N 1
#SBATCH -t 01:30:00
#SBATCH -A naiss2026-3-141-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip/logs/slumr_logs/%j.out
#SBATCH --error=/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip/logs/slumr_logs/%j.err

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
REPO=/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip
cd "$REPO"

# ---- run knobs -------------------------------------------------------------------------
SMOKE=${SMOKE:-1}                     # 1 = short pipeline check; 0 = full run (lqitl params)
# Sim server serves the reward, so its experiment must match the trainer's nostress intent.
# (Config header says single_lift_mushroom_soft; nostress is the reward-correct match — confirm
#  against how lqitl was launched before trusting the sweep numbers.)
SIM_EXPERIMENT=${SIM_EXPERIMENT:-single_lift_mushroom_soft_nostress}
VIEW=student
NUM_ENVS=12                            # MUST match cfg env.n_envs and divide logprob_batch_size (504)
PORT=5570                              # MUST match cfg env.specific.port
CKPT=$REPO/logs/dppo/dppo-pretrain/2026-07-04_20-03-36_42/checkpoint/state_3000.pt
CFG_DIR=$REPO/gentle_manip/dppo/cfg/single_lift_mushroom_soft_pcd
CFG_NAME=ft_ppo_diffusion_pointnet_modex_nostress
NORM=$REPO/dataset/dppo/single_lift_mushroom_soft_pcd/normalization.npz

export WANDB_MODE=offline              # GH200 compute nodes have no outbound internet
export MUJOCO_GL=egl                   # headless genesis render
unset PYTHONPATH ROS_DISTRO 2>/dev/null || true

# ---- preflight -------------------------------------------------------------------------
echo "[sbatch] host=$(hostname) arch=$(uname -m) job=${SLURM_JOB_ID:-NA} smoke=$SMOKE"
[ -f "$CKPT" ] || { echo "[sbatch] MISSING checkpoint: $CKPT"; exit 1; }
[ -f "$NORM" ] || { echo "[sbatch] MISSING normalization: $NORM (copy dataset/dppo/... first)"; exit 1; }

SRV_LOG=$REPO/logs/slumr_logs/${SLURM_JOB_ID:-local}_simserver.log

# ---- start the sim server (background) -------------------------------------------------
echo "[sbatch] launching sim server ($SIM_EXPERIMENT / $VIEW / $NUM_ENVS envs) on port $PORT"
uv run --project envs/sim_arrhenius --no-sync python -m gentle_manip.scripts.serl_sim_server \
    --experiment "$SIM_EXPERIMENT" --view "$VIEW" --num-envs "$NUM_ENVS" \
    --render-rgb --subprocess --port "$PORT" > "$SRV_LOG" 2>&1 &
SRV_PID=$!
cleanup() { echo "[sbatch] stopping sim server ($SRV_PID)"; kill "$SRV_PID" 2>/dev/null || true; wait "$SRV_PID" 2>/dev/null || true; }
trap cleanup EXIT

# ---- wait for readiness (server prints SIM_SERVER_READY once its port is bound) ---------
echo "[sbatch] waiting for SIM_SERVER_READY (server log: $SRV_LOG)"
for _ in $(seq 1 180); do                          # up to ~15 min (first genesis build compiles kernels)
    grep -q "SIM_SERVER_READY" "$SRV_LOG" 2>/dev/null && { echo "[sbatch] server ready"; break; }
    kill -0 "$SRV_PID" 2>/dev/null || { echo "[sbatch] server died early — tail:"; tail -50 "$SRV_LOG"; exit 1; }
    sleep 5
done
grep -q "SIM_SERVER_READY" "$SRV_LOG" || { echo "[sbatch] server not ready in time — tail:"; tail -50 "$SRV_LOG"; exit 1; }

# ---- run the trainer -------------------------------------------------------------------
SMOKE_OVERRIDES=()
if [ "$SMOKE" = "1" ]; then
    # short: exercise rollout(socket) + grad update + periodic eval(scene rebuild+video) + ckpt I/O
    SMOKE_OVERRIDES=(train.n_train_itr=2 train.n_steps=20 train.val_freq=2
                     train.save_model_freq=2 train.render.freq=2 train.eval_n_batches=1)
else
    # full run = lqitl's tested params
    SMOKE_OVERRIDES=(train.n_train_itr=250 train.actor_lr=2e-5 train.target_kl=0.05)
fi

echo "[sbatch] launching DPPO finetune: ${SMOKE_OVERRIDES[*]}"
uv run --project envs/dppo_arrhenius --no-sync python -m gentle_manip.dppo.train \
    --config-path "$CFG_DIR" --config-name "$CFG_NAME" \
    base_policy_path="$CKPT" \
    "${SMOKE_OVERRIDES[@]}"

echo "[sbatch] finetune finished OK"
