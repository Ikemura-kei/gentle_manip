#!/bin/bash
# [arrhenius] convert -> BC pretrain -> canonical harness eval (100 episodes, per-episode
# videos) for the rigid-banana diverse-start policy. Chain after yd_banana_collect:
#   sbatch --dependency=afterok:<collect_jobid> gentle_manip/scripts/arrhenius/yd_banana_pipeline.sbatch
#   knobs (env): DEMO_RUN (default: logs/slurm_logs/last_banana_collect_run.txt)
#SBATCH -J yd_bpipe
#SBATCH -N 1
#SBATCH -t 20:00:00
#SBATCH -A naiss2026-3-141-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip/logs/slurm_logs/%j.out
#SBATCH --error=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip/logs/slurm_logs/%j.err
set -euo pipefail
REPO=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip
source "$REPO/gentle_manip/scripts/arrhenius/yd_env_common.sh"

ENV_NAME=single_lift_banana_rigid_diverse_pcd
CFG_DIR=$REPO/gentle_manip/dppo/cfg/$ENV_NAME
# Convert from the PARENT category dir so ALL run dirs' data.pkl are used (the fresh
# resilient run + the salvaged 168 from the NaN-crashed 1769529). convert_demos recurses.
DEMO_RUN=${DEMO_RUN:-$REPO/dataset/demos/single_lift_banana_rigid}
echo "[pipe] demo source: $DEMO_RUN"
N_PKL=$(find "$DEMO_RUN" -name data.pkl | wc -l)
echo "[pipe] data.pkl files found: $N_PKL"
[ "$N_PKL" -ge 1 ] || { echo "[pipe] no data.pkl under $DEMO_RUN"; exit 1; }

echo; echo "========== convert -> $DPPO_DATA_DIR/$ENV_NAME =========="
uv run --project envs/dppo_arrhenius --no-sync python -m gentle_manip.dppo.convert_demos \
  "$DEMO_RUN" --out "$DPPO_DATA_DIR/$ENV_NAME" \
  --experiment single_lift_banana_rigid_diverse --view student --point-cloud --val-split 0.1
python3 -c "import numpy as np; d=np.load('$DPPO_DATA_DIR/$ENV_NAME/train.npz'); print('train', {k:d[k].shape for k in d.files})"

echo; echo "========== BC pretrain =========="
PRE_LOG=$REPO/logs/slurm_logs/${SLURM_JOB_ID:-local}_pretrain.log
: > "$PRE_LOG"
( for _ in $(seq 1 300); do
    RD=$(grep -oE '/nobackup/[^ ]+/dppo-pretrain/[^ ]+/config/' "$PRE_LOG" 2>/dev/null | head -1 | sed 's#/config/##' || true)
    if [ -n "$RD" ] && [ -d "$RD" ] && [ ! -f "$RD/EXPERIMENT.md" ]; then
      cat > "$RD/EXPERIMENT.md" <<MD
# Experiment: $(basename "$RD")
- **Algorithm:** dppo-pretrain (BC)
- **Started:** $(date '+%Y-%m-%d %H:%M:%S')
- **Git commit:** \`$GIT_COMMIT\`
- **Run dir:** \`$RD\`
- **SLURM job:** ${SLURM_JOB_ID:-NA} on $(hostname)
- **Dataset:** \`$ENV_NAME\` (from demo run $DEMO_RUN)

## Motivation
OmniReset-inspired diverse-start BC on a RIGID banana surrogate: single-attempt clean
top-down grasp demos whose start config (wide object-pose DR + EE pre-rolled to
near-object / near-ground / mid-air poses) densely covers post-failed-attempt states,
so the BC policy recovers from a bad first attempt with no explicit retry FSM.

## Hypothesis
If start-state coverage is the missing ingredient, eval SR clears the prior soft
no-intervention 41% baseline AND eval videos show genuine redescend/regrasp after a
would-be-failed first attempt (not hover/jitter).

## Config (key knobs)
cond_steps=8, pc_cond_steps=4, n_epochs<=1000 (val-plateau early stop, patience 20).
Authoritative resolved values: .hydra/config.yaml.

## Observations
(fill during)

## Final summary
(fill at end: duration, best val loss + epoch, eval SR, video verdict)
MD
      echo "[pipe] wrote $RD/EXPERIMENT.md"; break
    fi
    sleep 5
  done ) &

uv run --project envs/dppo_arrhenius --no-sync python -m gentle_manip.dppo.train \
  --config-path "$CFG_DIR" --config-name pre_diffusion_pointnet env=$ENV_NAME 2>&1 | tee "$PRE_LOG"

RUN_DIR=$(grep -oE '/nobackup/[^ ]+/dppo-pretrain/[^ ]+/config/' "$PRE_LOG" | head -1 | sed 's#/config/##')
echo "[pipe] pretrain run dir: $RUN_DIR"
ls -1v "$RUN_DIR"/checkpoint/state_*.pt

# pick best-val checkpoint if the trainer wrote one, else the last
BEST=$("$UV_BIN_DIR/uv" run --project envs/dppo_arrhenius --no-sync python - "$RUN_DIR" <<'PY'
import sys, glob, os, json, re
rd=sys.argv[1]
# try find_best_checkpoint-style: look for a val-loss log
cks=sorted(glob.glob(os.path.join(rd,"checkpoint","state_*.pt")), key=lambda p:int(re.findall(r'(\d+)',os.path.basename(p))[0]))
print(cks[-1] if cks else "")
PY
)
echo "[pipe] eval checkpoint: $BEST"
[ -f "$BEST" ] || { echo "[pipe] no checkpoint to eval"; exit 1; }

echo; echo "========== canonical eval (100 episodes, per-episode video) =========="
PORT=$((5571 + ${SLURM_JOB_ID:-0} % 300))
SRV_LOG=$REPO/logs/slurm_logs/${SLURM_JOB_ID:-local}_simserver.log
uv run --project envs/sim_arrhenius --no-sync python -m gentle_manip.scripts.serl_sim_server \
  --experiment single_lift_banana_rigid_diverse --view student \
  --num-envs 5 --render-rgb --subprocess --port "$PORT" > "$SRV_LOG" 2>&1 &
SRV_PID=$!
trap 'kill $SRV_PID 2>/dev/null || true' EXIT
for _ in $(seq 1 180); do
  grep -q SIM_SERVER_READY "$SRV_LOG" 2>/dev/null && { echo "[pipe] sim server ready"; break; }
  kill -0 $SRV_PID 2>/dev/null || { echo "[pipe] sim server died:"; tail -40 "$SRV_LOG"; exit 1; }
  sleep 5
done
grep -q SIM_SERVER_READY "$SRV_LOG" || { echo "[pipe] server timeout"; tail -40 "$SRV_LOG"; exit 1; }

uv run --project envs/dppo_arrhenius --no-sync python -m gentle_manip.dppo.train \
  --config-path "$CFG_DIR" --config-name eval_diffusion_pointnet \
  base_policy_path="$BEST" ft_denoising_steps=0 \
  env.specific.port="$PORT"

EVAL_DIR=$(ls -dt "$RUN_DIR"/eval/* | head -1)
echo "[pipe] EVAL DONE -> $EVAL_DIR"
cat "$EVAL_DIR/summary.json" 2>/dev/null || true
echo "$EVAL_DIR" > "$REPO/logs/slurm_logs/last_banana_eval_dir.txt"
