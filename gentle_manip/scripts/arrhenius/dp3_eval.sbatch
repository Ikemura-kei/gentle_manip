#!/bin/bash
# [arrhenius] Canonical-harness eval of a DP3 checkpoint on the GH200 (aarch64) nodes.
# Two-process pattern like dppo_eval.sbatch: serl_sim_server (envs/sim_arrhenius,
# --subprocess for the per-group scene rebuilds) + eval_dp3_harness.py (envs/dp3_arrhenius,
# cwd = 3D-Diffusion-Policy for the namespace import, GCC14 libstdc++ for pytorch3d).
#
# Submit:  CKPT=<.ckpt> EXPERIMENT=<exp> sbatch gentle_manip/scripts/arrhenius/dp3_eval.sbatch
#   knobs: CKPT (required), EXPERIMENT (required), N_EPISODES(200), NUM_ENVS(5), SEED(42),
#          PORT(5570), OUT(default <run>/sim_eval_harness/<ckpt-stem>)
#
#SBATCH -J dp3_eval
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -A naiss2026-3-141-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip/logs/slumr_logs/%j.out
#SBATCH --error=/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip/logs/slumr_logs/%j.err

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
REPO=/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip
DP3_DIR="$REPO/third_party/DP3/3D-Diffusion-Policy"
cd "$REPO"

CKPT=${CKPT:?set CKPT=<path to .ckpt>}
EXPERIMENT=${EXPERIMENT:?set EXPERIMENT=<sim experiment name (action pipeline!)>}
N_EPISODES=${N_EPISODES:-200}
NUM_ENVS=${NUM_ENVS:-5}
SEED=${SEED:-42}
# per-job port (jobs sharing a 4-GPU node MUST NOT collide on the rpc port —
# a client connected to another job's server dies with "socket closed mid-message")
PORT=${PORT:-$((5570 + ${SLURM_JOB_ID:-0} % 400))}
OUT=${OUT:-}

GCCLIB=/software/sse2/el9_gh200/manual/GCC/14.3.0/hpc1/lib64
unset PYTHONPATH ROS_DISTRO 2>/dev/null || true

SRV_LOG=$REPO/logs/slumr_logs/${SLURM_JOB_ID:-local}_dp3eval_simserver.log
echo "[sbatch] dp3 harness eval: ckpt=$CKPT experiment=$EXPERIMENT n_episodes=$N_EPISODES port=$PORT"
uv run --project envs/sim_arrhenius --no-sync python -m gentle_manip.scripts.serl_sim_server \
    --experiment "$EXPERIMENT" --view student --num-envs "$NUM_ENVS" \
    --render-rgb --subprocess --port "$PORT" > "$SRV_LOG" 2>&1 &
SRV_PID=$!
cleanup() { echo "[sbatch] stopping sim server ($SRV_PID)"; kill "$SRV_PID" 2>/dev/null || true; wait "$SRV_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "[sbatch] waiting for SIM_SERVER_READY ($SRV_LOG)"
for _ in $(seq 1 180); do
    grep -q "SIM_SERVER_READY" "$SRV_LOG" 2>/dev/null && break
    kill -0 "$SRV_PID" 2>/dev/null || { echo "[sbatch] sim server died:"; tail -20 "$SRV_LOG"; exit 1; }
    sleep 5
done
grep -q "SIM_SERVER_READY" "$SRV_LOG" || { echo "[sbatch] server never ready"; tail -20 "$SRV_LOG"; exit 1; }

cd "$DP3_DIR"
LD_LIBRARY_PATH="$GCCLIB:${LD_LIBRARY_PATH:-}" \
uv run --project "$REPO/envs/dp3_arrhenius" --no-sync python \
    "$REPO/gentle_manip/scripts/eval_dp3_harness.py" \
    --ckpt "$CKPT" --experiment "$EXPERIMENT" --n-episodes "$N_EPISODES" \
    --num-envs "$NUM_ENVS" --seed "$SEED" --port "$PORT" ${OUT:+--out "$OUT"}
echo "[sbatch] dp3 harness eval finished OK"
