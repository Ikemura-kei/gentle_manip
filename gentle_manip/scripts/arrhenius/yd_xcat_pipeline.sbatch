#!/bin/bash
# [arrhenius] convert -> BC pretrain -> canonical harness eval (100 episodes, per-episode
# videos) for the rigid-banana diverse-start policy. Chain after yd_banana_collect:
#   sbatch --dependency=afterok:<collect_jobid> gentle_manip/scripts/arrhenius/yd_banana_pipeline.sbatch
#   knobs (env): DEMO_RUN (default: logs/slurm_logs/last_xcat_regrasp_run.txt)
#SBATCH -J yd_xpipe
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

ENV_NAME=${ENV_NAME:-single_lift_xcat_diverse_pcd}
EXPERIMENT=${EXPERIMENT:-single_lift_xcat_diverse}
CFG_DIR=$REPO/gentle_manip/dppo/cfg/$ENV_NAME
# DEMO_RUN: explicit, or last completed run, or (PRELIM) newest dir under the TAG folder.
TAG=${TAG:-xcat_regrasp}
if [ -z "${DEMO_RUN:-}" ]; then
  DEMO_RUN=$(cat "$REPO/logs/slurm_logs/last_${TAG}_run.txt" 2>/dev/null || true)
  [ -z "$DEMO_RUN" ] && DEMO_RUN=$(ls -dt "$REPO"/dataset/demos/single_lift_${TAG}/*/ 2>/dev/null | head -1)
fi
DEMO_RUN=${DEMO_RUN%/}
echo "[pipe] demo run: $DEMO_RUN"
# PRELIM mode: collection still running -> COPY its current shards to a staging dir
# and merge THERE (never touch the live run dir: _merge_shards deletes shards, which
# would make the collector's end-of-run merge drop everything collected so far).
if [ ! -f "$DEMO_RUN/data.pkl" ]; then
  STAGE="$REPO/dataset/demos/single_lift_${TAG}/_prelim_${SLURM_JOB_ID:-local}"
  echo "[pipe] no data.pkl -> PRELIM: staging $(ls "$DEMO_RUN"/shard_*.pkl 2>/dev/null | wc -l) shards -> $STAGE"
  mkdir -p "$STAGE"
  cp "$DEMO_RUN"/shard_*.pkl "$STAGE"/ 2>/dev/null || true
  cp "$DEMO_RUN"/config.yaml "$STAGE"/ 2>/dev/null || true
  uv run --project envs/dppo_arrhenius --no-sync python - "$STAGE" <<'PYM'
# inline shard merge (stdlib only -- collect_demos_synth imports scipy at module top)
import sys, pickle, datetime
from pathlib import Path
rd = Path(sys.argv[1])
shards = sorted(rd.glob("shard_*.pkl"))
eps, meta = [], None
for p in shards:
    try:
        d = pickle.load(open(p, "rb"))
    except Exception as e:
        print("  skip", p.name, e); continue
    meta = meta or dict(d["meta"]); eps.extend(d["episodes"])
meta = meta or {}
meta["n_episodes"] = len(eps)
meta["created"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
out = rd / "data.pkl"
pickle.dump({"meta": meta, "episodes": eps}, open(out.with_suffix(".tmp"), "wb"))
out.with_suffix(".tmp").replace(out)
print(f"[pipe] merged {len(shards)} shards -> {out} ({len(eps)} episodes)")
PYM
  DEMO_RUN="$STAGE"
fi
[ -f "$DEMO_RUN/data.pkl" ] || { echo "[pipe] MISSING $DEMO_RUN/data.pkl (no shards?)"; exit 1; }

echo; echo "========== convert -> $DPPO_DATA_DIR/$ENV_NAME =========="
uv run --project envs/dppo_arrhenius --no-sync python -m gentle_manip.dppo.convert_demos \
  "$DEMO_RUN" --out "$DPPO_DATA_DIR/$ENV_NAME" \
  --experiment "$EXPERIMENT" --view student --point-cloud --val-split 0.1
uv run --project envs/dppo_arrhenius --no-sync python -c "import numpy as np; d=np.load('$DPPO_DATA_DIR/$ENV_NAME/train.npz'); print('train', {k:d[k].shape for k in d.files})" || true

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
OmniReset-inspired diverse-start BC on a CROSS-CATEGORY (4 soft objects: mushroom, banana_lying, kiwi, egg_boiled): single-attempt clean
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

# pick the BEST-VAL checkpoint (parse the pretrain log); fall back to the last.
BEST=$("$UV_BIN_DIR/uv" run --project envs/dppo_arrhenius --no-sync python \
  "$REPO/gentle_manip/scripts/arrhenius/_pick_best_ckpt.py" "$RUN_DIR" "$PRE_LOG")
echo "[pipe] eval checkpoint: $BEST"
[ -f "$BEST" ] || { echo "[pipe] no checkpoint to eval"; exit 1; }

echo; echo "========== canonical eval (100 episodes, per-episode video) =========="
PORT=$((5571 + ${SLURM_JOB_ID:-0} % 300))
SRV_LOG=$REPO/logs/slurm_logs/${SLURM_JOB_ID:-local}_simserver.log
uv run --project envs/sim_arrhenius --no-sync python -m gentle_manip.scripts.serl_sim_server \
  --experiment "$EXPERIMENT" --view student \
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
uv run --project envs/sim_arrhenius --no-sync python "$REPO/gentle_manip/scripts/arrhenius/_trim_eval_clips.py" "$EVAL_DIR" 4 16 || true
echo "[pipe] EVAL DONE (clean) -> $EVAL_DIR"
cat "$EVAL_DIR/summary.json" 2>/dev/null || true
echo "$EVAL_DIR" > "$REPO/logs/slurm_logs/last_xcat_eval_dir.txt"

# ---- 2nd eval: REGRASP experiment (arm starts low over the object) ----
REGRASP_EXPERIMENT=${REGRASP_EXPERIMENT:-single_lift_xcat_regrasp_eval}
if [ -n "$REGRASP_EXPERIMENT" ] && [ -f "$REPO/gentle_manip/configs/experiments/$REGRASP_EXPERIMENT.yaml" ]; then
  echo; echo "========== eval #2: $REGRASP_EXPERIMENT (regrasp start) =========="
  cleanup 2>/dev/null || kill $SRV_PID 2>/dev/null || true; sleep 3
  PORT2=$((5871 + ${SLURM_JOB_ID:-0} % 300))
  SRV_LOG2=$REPO/logs/slurm_logs/${SLURM_JOB_ID:-local}_simserver2.log
  uv run --project envs/sim_arrhenius --no-sync python -m gentle_manip.scripts.serl_sim_server \
    --experiment "$REGRASP_EXPERIMENT" --view student --num-envs 5 --render-rgb --subprocess --port "$PORT2" > "$SRV_LOG2" 2>&1 &
  SRV2=$!; trap 'kill $SRV2 2>/dev/null||true' EXIT
  for _ in $(seq 1 180); do grep -q SIM_SERVER_READY "$SRV_LOG2" 2>/dev/null && break
    kill -0 $SRV2 2>/dev/null || { tail -30 "$SRV_LOG2"; break; }; sleep 5; done
  if grep -q SIM_SERVER_READY "$SRV_LOG2"; then
    uv run --project envs/dppo_arrhenius --no-sync python -m gentle_manip.dppo.train \
      --config-path "$CFG_DIR" --config-name eval_diffusion_pointnet \
      base_policy_path="$BEST" ft_denoising_steps=0 experiment="$REGRASP_EXPERIMENT" \
      env.specific.port="$PORT2" logdir="$RUN_DIR/eval_regrasp/$(date +%Y-%m-%d_%H-%M-%S)" || echo "[pipe] regrasp eval failed"
    ED2=$(ls -dt "$RUN_DIR"/eval_regrasp/* 2>/dev/null | head -1)
    uv run --project envs/sim_arrhenius --no-sync python "$REPO/gentle_manip/scripts/arrhenius/_trim_eval_clips.py" "$ED2" 4 16 || true
    echo "[pipe] REGRASP EVAL DONE -> $ED2"; cat "$ED2/summary.json" 2>/dev/null || true
    echo "$ED2" > "$REPO/logs/slurm_logs/last_banana_regrasp_eval_dir.txt"
  fi
fi
