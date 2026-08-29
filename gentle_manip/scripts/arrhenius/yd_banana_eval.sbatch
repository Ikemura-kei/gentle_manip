#!/bin/bash
# [arrhenius] Standalone canonical eval of a rigid/soft banana diverse-start BC checkpoint.
#   CKPT=<state_N.pt>  RUN_DIR=<pretrain run dir>  (picks best-val ckpt if CKPT unset)
#   ENV_NAME / EXPERIMENT / CFG_DIR overridable for the soft/generalist variants.
#SBATCH -J yd_beval
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -A naiss2026-3-141-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip/logs/slurm_logs/%j.out
#SBATCH --error=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip/logs/slurm_logs/%j.err
set -euo pipefail
REPO=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip
source "$REPO/gentle_manip/scripts/arrhenius/yd_env_common.sh"

ENV_NAME=${ENV_NAME:-single_lift_banana_rigid_diverse_pcd}
EXPERIMENT=${EXPERIMENT:-single_lift_banana_rigid_diverse}
CFG_DIR=${CFG_DIR:-$REPO/gentle_manip/dppo/cfg/$ENV_NAME}
RUN_DIR=${RUN_DIR:-$(ls -dt $REPO/logs/dppo/dppo-pretrain/$ENV_NAME/*/ | head -1)}
RUN_DIR=${RUN_DIR%/}

if [ -z "${CKPT:-}" ]; then
  # best-val checkpoint: parse the hydra train log for min "val loss", map to nearest saved state_N.pt
  TRAINLOG=$(ls -t "$RUN_DIR"/*/train.log "$RUN_DIR"/train.log 2>/dev/null | head -1 || true)
  [ -z "$TRAINLOG" ] && TRAINLOG=$(grep -l "$(basename "$RUN_DIR")" "$REPO"/logs/slurm_logs/*_pretrain.log 2>/dev/null | head -1 || true)
  CKPT=$("$UV_BIN_DIR/uv" run --project envs/dppo_arrhenius --no-sync python - "$RUN_DIR" "${TRAINLOG:-}" <<'PY'
import sys, re, glob, os
rd, log = sys.argv[1], sys.argv[2]
cks = {int(re.findall(r'(\d+)', os.path.basename(p))[0]): p
       for p in glob.glob(os.path.join(rd, "checkpoint", "state_*.pt"))}
best_ep, best_v = None, 1e9
if log and os.path.exists(log):
    for m in re.finditer(r'-\s*(\d+):\s*train loss\s+[\d.]+\s*\|\s*val loss\s+([\d.]+)', open(log).read()):
        ep, v = int(m.group(1)), float(m.group(2))
        if v < best_v: best_v, best_ep = v, ep
if best_ep is None:
    print(cks[max(cks)]); sys.exit()
pick = min(cks, key=lambda e: abs(e - best_ep))
sys.stderr.write(f"best val {best_v:.4f} @ ep{best_ep} -> state_{pick}.pt\n")
print(cks[pick])
PY
)
fi
echo "[eval] checkpoint: $CKPT"
[ -f "$CKPT" ] || { echo "[eval] missing $CKPT"; exit 1; }

PORT=$((5571 + ${SLURM_JOB_ID:-0} % 300))
SRV_LOG=$REPO/logs/slurm_logs/${SLURM_JOB_ID:-local}_simserver.log
uv run --project envs/sim_arrhenius --no-sync python -m gentle_manip.scripts.serl_sim_server \
  --experiment "$EXPERIMENT" --view student --num-envs 5 --render-rgb --subprocess --port "$PORT" > "$SRV_LOG" 2>&1 &
SRV=$!; trap 'kill $SRV 2>/dev/null||true' EXIT
for _ in $(seq 1 180); do grep -q SIM_SERVER_READY "$SRV_LOG" 2>/dev/null && break
  kill -0 $SRV 2>/dev/null || { tail -40 "$SRV_LOG"; exit 1; }; sleep 5; done
grep -q SIM_SERVER_READY "$SRV_LOG" || { tail -40 "$SRV_LOG"; exit 1; }

TAG=${EVAL_TAG:-$(basename "$EXPERIMENT")_${SLURM_JOB_ID:-local}}
LOGDIR="$RUN_DIR/eval/$TAG"
uv run --project envs/dppo_arrhenius --no-sync python -m gentle_manip.dppo.train \
  --config-path "$CFG_DIR" --config-name eval_diffusion_pointnet \
  base_policy_path="$CKPT" ft_denoising_steps=0 experiment="$EXPERIMENT" \
  env.specific.port="$PORT" logdir="$LOGDIR"
ED="$LOGDIR"
echo "[eval] DONE -> $ED"; cat "$ED/summary.json" 2>/dev/null || true
# trim success clips to end at the success frame (no carry-down tail)
uv run --project envs/sim_arrhenius --no-sync python \
  "$REPO/gentle_manip/scripts/arrhenius/_trim_eval_clips.py" "$ED" 4 16 || echo "[eval] clip trim skipped"
echo "$ED" > "$REPO/logs/slurm_logs/last_banana_eval_dir.txt"
