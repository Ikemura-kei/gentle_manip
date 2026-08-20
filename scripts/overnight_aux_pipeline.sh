#!/usr/bin/env bash
# Overnight autonomous pipeline for the coupling-force aux-objective policy:
#   1. re-collect 300 soft demos (proper MPM coupling-force contact label)
#   2. convert -> DPPO npz (env = single_lift_mushroom_soft_abs_pcd_v2)
#   3. train BOTH aux objectives (contact + object-pos), 1000 ep, save/100, wandb ONLINE
#   4. after training, sweep-eval checkpoints 200/300/400/500/600/800/1000 via the shared harness
# Self-chaining + a STATUS file so a heartbeat can report progress. Each stage guards on the prior.
set -u
cd /home/kei/kei/gentle_manip || exit 1

RUN_ENVP="env -u PYTHONPATH -u ROS_DISTRO MUJOCO_GL=egl"
SIM="uv run --project envs/sim --no-sync"
DPPO="uv run --project envs/dppo --no-sync"
ENV_V2=single_lift_mushroom_soft_abs_pcd_v2
EXPERIMENT=single_lift_mushroom_soft_abs_action_armfocus
DATA_V2=dataset/dppo/${ENV_V2}
CFG=/home/kei/kei/gentle_manip/gentle_manip/dppo/cfg/single_lift_mushroom_soft_abs_pcd  # hydra needs ABSOLUTE
PORT=5583
OUT=logs/overnight_aux
mkdir -p "$OUT"
STATUS="$OUT/STATUS"
say () { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/pipeline.log"; }
setst () { echo "$*" > "$STATUS"; say "STATUS -> $*"; }
fail () { setst "FAILED: $*"; say "ABORT: $*"; exit 1; }

say "=== overnight aux pipeline start (commit $(git rev-parse --short HEAD)) ==="

# ── Stage 1: re-collect 300 demos (coupling-force contact) ─────────────────────────
# Resume: if DATA_PKL_OVERRIDE points at an existing data.pkl, skip the collect.
if [ -n "${DATA_PKL_OVERRIDE:-}" ] && [ -f "${DATA_PKL_OVERRIDE}" ]; then
  DATA_PKL="${DATA_PKL_OVERRIDE}"
  setst "STAGE1_COLLECT skipped (resume)"
  say "using pre-collected $DATA_PKL"
else
  setst "STAGE1_COLLECT running"
  $RUN_ENVP $SIM python grasp_synthesis/collect_demos_synth_v3.py --experiment $EXPERIMENT \
    --n-episodes 300 --n-envs 8 --maxfevals 1145 --grasp-gpu --seed 0 --scene-dr-every 1 \
    --grasp-extra-close 0.0025 --record-video 3 --out-dir dataset/demos > "$OUT/1_collect.log" 2>&1
  [ $? -eq 0 ] || fail "collect exited nonzero"
  DATA_PKL=$(grep -oE "dataset/demos/single_lift_mushroom_soft/[0-9a-z-]+/data.pkl" "$OUT/1_collect.log" | tail -1)
  [ -f "$DATA_PKL" ] || fail "collect produced no data.pkl"
  say "collected -> $DATA_PKL"
fi

# ── Stage 2: convert -> DPPO npz (quat student + aux labels) ───────────────────────
setst "STAGE2_CONVERT running"
$RUN_ENVP $DPPO python -m gentle_manip.dppo.convert_demos "$DATA_PKL" \
  --out "$DATA_V2" --experiment $EXPERIMENT --view student --point-cloud > "$OUT/2_convert.log" 2>&1
[ $? -eq 0 ] || fail "convert exited nonzero"
[ -f "$DATA_V2/train.npz" ] || fail "convert produced no train.npz"
say "converted -> $DATA_V2 ($(grep -oE 'n_episodes: [0-9]+' "$OUT/2_convert.log" | tail -1))"

# ── Stage 3: train BOTH aux objectives (1000 ep, save/100, wandb online) ───────────
setst "STAGE3_TRAIN running"
$RUN_ENVP $DPPO python -m gentle_manip.dppo.train --config-path "$CFG" --config-name pre_diffusion_pointnet \
  env=${ENV_V2} experiment=$EXPERIMENT \
  model.network.aux_contact=true model.aux_contact_weight=1.0 \
  model.network.aux_object_pos=true model.aux_object_pos_weight=1.0 \
  wandb.group=dppo-pretrain-aux-local > "$OUT/3_train.log" 2>&1
[ $? -eq 0 ] || fail "train exited nonzero"
RUN_DIR=$(ls -dt logs/dppo/dppo-pretrain/${ENV_V2}/*/ 2>/dev/null | head -1)
[ -n "$RUN_DIR" ] || fail "no train run dir"
say "trained -> $RUN_DIR"

# ── Stage 4: eval sweep (one sim server reused across ckpts) ───────────────────────
setst "STAGE4_EVAL running"
$RUN_ENVP $SIM python -m gentle_manip.scripts.serl_sim_server --experiment $EXPERIMENT \
  --view student --num-envs 5 --render-rgb --subprocess --port $PORT > "$OUT/4_sim_server.log" 2>&1 &
SRV=$!
say "sim server pid $SRV (port $PORT) — waiting for 'serving on'"
for i in $(seq 1 120); do
  grep -q "serving on" "$OUT/4_sim_server.log" 2>/dev/null && break
  kill -0 $SRV 2>/dev/null || { fail "sim server died before ready"; }
  sleep 5
done
grep -q "serving on" "$OUT/4_sim_server.log" || { kill $SRV 2>/dev/null; fail "sim server never ready"; }
say "sim server ready"

for N in 200 300 400 500 600 800 1000; do
  CKPT="${RUN_DIR}checkpoint/state_${N}.pt"
  if [ ! -f "$CKPT" ]; then say "skip state_${N} (missing)"; continue; fi
  setst "STAGE4_EVAL ckpt $N"
  say "eval state_${N} ..."
  $RUN_ENVP $DPPO python -m gentle_manip.dppo.train --config-path "$CFG" --config-name eval_diffusion_pointnet \
    env_name=${ENV_V2} experiment=$EXPERIMENT \
    base_policy_path="$CKPT" normalization_path="$DATA_V2/normalization.npz" \
    env.specific.port=$PORT > "$OUT/4_eval_${N}.log" 2>&1
  SR=$(grep -oE "\[eval\] DONE — success [0-9.]+" "$OUT/4_eval_${N}.log" | tail -1)
  say "  state_${N} -> ${SR:-(no summary — see 4_eval_${N}.log)}"
done
kill $SRV 2>/dev/null
say "sim server stopped"

setst "DONE run_dir=$RUN_DIR"
say "=== overnight aux pipeline COMPLETE ==="
