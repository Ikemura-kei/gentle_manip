#!/usr/bin/env bash
# Eval sweep for the both-aux run (re-run after the strict-load fix). Reuses one sim server.
set -u
cd /home/kei/kei/gentle_manip || exit 1
RUN_ENVP="env -u PYTHONPATH -u ROS_DISTRO MUJOCO_GL=egl"
SIM="uv run --project envs/sim --no-sync"
DPPO="uv run --project envs/dppo --no-sync"
ENV_V2=single_lift_mushroom_soft_abs_pcd_v2
EXPERIMENT=single_lift_mushroom_soft_abs_action_armfocus
DATA_V2=dataset/dppo/${ENV_V2}
CFG=/home/kei/kei/gentle_manip/gentle_manip/dppo/cfg/single_lift_mushroom_soft_abs_pcd
PORT=5588
OUT=logs/overnight_aux
RUN_DIR="${RUN_DIR:-logs/dppo/dppo-pretrain/single_lift_mushroom_soft_abs_pcd_v2/sevrq/}"
say () { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/eval_sweep.log"; }
echo "STAGE4B_EVAL running" > "$OUT/STATUS"

say "=== eval sweep run=$RUN_DIR ==="
$RUN_ENVP $SIM python -m gentle_manip.scripts.serl_sim_server --experiment $EXPERIMENT \
  --view student --num-envs 5 --render-rgb --subprocess --port $PORT > "$OUT/4b_sim_server.log" 2>&1 &
SRV=$!
for i in $(seq 1 120); do
  grep -q "serving on" "$OUT/4b_sim_server.log" 2>/dev/null && break
  kill -0 $SRV 2>/dev/null || break
  sleep 5
done
grep -q "serving on" "$OUT/4b_sim_server.log" || { say "server never ready"; kill $SRV 2>/dev/null; echo "FAILED: eval server" > "$OUT/STATUS"; exit 1; }
say "sim server ready (pid $SRV, port $PORT)"

for N in 200 300 400 500 600 800 1000; do
  CKPT="${RUN_DIR}checkpoint/state_${N}.pt"
  [ -f "$CKPT" ] || { say "skip $N (missing)"; continue; }
  echo "STAGE4B_EVAL ckpt $N" > "$OUT/STATUS"
  say "eval state_${N} ..."
  # Leaner than the canonical protocol (50 eps, 1 video/ckpt) so the 7-ckpt sweep finishes overnight;
  # run a full n_episodes=200 record_batches=null canonical eval on the best ckpt afterward.
  $RUN_ENVP $DPPO python -m gentle_manip.dppo.train --config-path "$CFG" --config-name eval_diffusion_pointnet \
    env_name=${ENV_V2} experiment=$EXPERIMENT base_policy_path="$CKPT" \
    normalization_path="$DATA_V2/normalization.npz" env.specific.port=$PORT \
    n_episodes=50 record_batches=1 > "$OUT/4b_eval_${N}.log" 2>&1
  SR=$(grep -oE "\[eval\] DONE — success [0-9.]+" "$OUT/4b_eval_${N}.log" | tail -1)
  say "  state_${N} -> ${SR:-FAILED (see 4b_eval_${N}.log)}"
done
kill $SRV 2>/dev/null
echo "DONE_EVAL run_dir=$RUN_DIR" > "$OUT/STATUS"
say "=== eval sweep COMPLETE ==="
