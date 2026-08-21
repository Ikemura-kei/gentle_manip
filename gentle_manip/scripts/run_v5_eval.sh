#!/bin/bash
# [local] Part D step 4: canonical-harness eval of a v5_7d BC checkpoint.
#
#   bash gentle_manip/scripts/run_v5_eval.sh <checkpoint.pt> [port]
#
# Pattern from misc_scripts/eval_lfjih800.sh. EVAL experiment = the CANONICAL
# single_lift_mushroom_soft_abs_action (soft_orientation DR) — the robust-start DR is a
# COLLECTION knob; evaluating on it would break comparability with every recorded number.
set -eu
CKPT=${1:?checkpoint .pt}
PORT=${2:-5745}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
CFG=$REPO/gentle_manip/dppo/cfg/single_lift_mushroom_soft_v5_7d
EXP=single_lift_mushroom_soft_abs_action
NORM=$REPO/dataset/dppo/single_lift_mushroom_soft_v5_7d/normalization.npz
SRV_LOG=logs/grasp_bench/v5_eval_server_$PORT.log

: > "$SRV_LOG"
nohup env -u PYTHONPATH -u ROS_DISTRO MUJOCO_GL=egl uv run --project envs/sim --no-sync \
  python -m gentle_manip.scripts.serl_sim_server --experiment $EXP --view student \
  --num-envs 5 --render-rgb --subprocess --port "$PORT" > "$SRV_LOG" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null' EXIT INT TERM
for i in $(seq 1 150); do grep -q SIM_SERVER_READY "$SRV_LOG" && break; sleep 5; done
grep -q SIM_SERVER_READY "$SRV_LOG" || { echo "server never became ready"; exit 1; }

env -u PYTHONPATH -u ROS_DISTRO uv run --project envs/dppo --no-sync \
  python -m gentle_manip.dppo.train --config-path "$CFG" --config-name eval_diffusion_pointnet \
  base_policy_path="$CKPT" normalization_path="$NORM" "env.specific.port=$PORT"
