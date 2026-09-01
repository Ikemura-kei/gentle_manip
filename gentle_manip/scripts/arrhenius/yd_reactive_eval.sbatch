#!/bin/bash
# [arrhenius] REACTIVE eval: per-category canonical eval WITH a random mid-approach object velocity impulse (--dr eval_cat_<name>_reactive). Same harness/metrics as yd_gen_eval; tests reactive re-grasp under perturbation.
# canonical harness eval (n_episodes=100, num_envs=5, seed=0, scene_group_size=1 -> geometry
# +material re-randomised every 5 episodes), EE init spanning home<->near-object (dr:
# eval_cat_<name>). RECORD_BATCHES batches get per-episode video (default 3 => 15 clips,
# feeds the montage webpage). Runs NCAT_PAR categories CONCURRENTLY on one GPU (each sim
# server ~7 GB / 98 GB; ~2x wall speedup) so 12 categories fit a 24h job. Aggregates the
# 3 metrics (SR / gentleness / SR*gentleness) per-category + in-domain/OOD means.
#   ENV_NAME / RUN_DIR (-> best-val CKPT) | CATS="mushroom kiwi ..." | NCAT_PAR (3) | RECORD_BATCHES (3)
#SBATCH -J yd_reacteval
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -A naiss2026-3-141-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --output=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip/logs/slurm_logs/%j.out
#SBATCH --error=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip/logs/slurm_logs/%j.err
set -uo pipefail
REPO=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip
source "$REPO/gentle_manip/scripts/arrhenius/yd_env_common.sh"

ENV_NAME=${ENV_NAME:-single_lift_xcat_regrasp_pcd}
CFG_DIR=$REPO/gentle_manip/dppo/cfg/$ENV_NAME
RUN_DIR=${RUN_DIR:-$(ls -dt $REPO/logs/dppo/dppo-pretrain/$ENV_NAME/*/ | head -1)}; RUN_DIR=${RUN_DIR%/}
NCAT_PAR=${NCAT_PAR:-3}
RECORD_BATCHES=${RECORD_BATCHES:-3}
if [ -z "${CKPT:-}" ]; then
  PRE_LOG=$(grep -l "$(basename "$RUN_DIR")" "$REPO"/logs/slurm_logs/*_pretrain.log 2>/dev/null | head -1 || true)
  CKPT=$("$UV_BIN_DIR/uv" run --project envs/dppo_arrhenius --no-sync python \
    "$REPO/gentle_manip/scripts/arrhenius/_pick_best_ckpt.py" "$RUN_DIR" "${PRE_LOG:-}")
fi
echo "[geneval] ckpt: $CKPT  ncat_par=$NCAT_PAR  record_batches=$RECORD_BATCHES"
INDOMAIN="mushroom banana_lying kiwi egg_boiled grape cherry tomato raspberry"
OOD="blackberry scallop dumpling gelatin"
SMALL="grape cherry tomato raspberry blackberry scallop"
CATS=${CATS:-$INDOMAIN $OOD}
OUT=${OUT:-$RUN_DIR/gen_eval_$(date +%Y%m%d_%H%M%S)}; mkdir -p "$OUT"
echo "[geneval] categories: $CATS -> $OUT"

run_one() {           # $1 = category, $2 = port
  local cat="$1" port="$2"
  local SRV="$REPO/logs/slurm_logs/${SLURM_JOB_ID:-local}_srv_${cat}.log"
  local LOGDIR="$OUT/$cat"
  [ -f "$LOGDIR/summary.json" ] && { echo "[$cat] already done -> skip"; return 0; }
  local EXP=single_lift_xcat_gen_eval NEP=${NEP:-100}
  # small-fruit categories use the finer-grid _small task (grid 300 vs 190 -> ~6x slower
  # per step); cap them at 50 episodes so the 12-cat sweep still fits one walltime.
  case " $SMALL " in *" $cat "*) EXP=single_lift_xcat_gen_eval_small; NEP=${SMALL_NEP:-50} ;; esac
  echo "=== [$cat] ($EXP, n=$NEP) port $port ==="
  uv run --project envs/sim_arrhenius --no-sync python -m gentle_manip.scripts.serl_sim_server \
    --experiment "$EXP" --dr eval_cat_${cat}_reactive --view student \
    --num-envs 5 --render-rgb --subprocess --port "$port" > "$SRV" 2>&1 &
  local P=$!
  for _ in $(seq 1 240); do grep -q SIM_SERVER_READY "$SRV" && break
    kill -0 $P 2>/dev/null || { echo "[$cat] server died"; tail -30 "$SRV"; return 1; }; sleep 5; done
  grep -q SIM_SERVER_READY "$SRV" || { echo "[$cat] SKIP (no server)"; kill $P 2>/dev/null||true; return 1; }
  uv run --project envs/dppo_arrhenius --no-sync python -m gentle_manip.dppo.train \
    --config-path "$CFG_DIR" --config-name eval_diffusion_pointnet \
    base_policy_path="$CKPT" ft_denoising_steps=0 experiment="$EXP" n_episodes="$NEP" \
    record_batches="$RECORD_BATCHES" env.specific.port="$port" logdir="$LOGDIR" || echo "[$cat] eval failed"
  kill $P 2>/dev/null || true; sleep 3
  if [ "${NOTRIM:-0}" != "1" ]; then
    uv run --project envs/sim_arrhenius --no-sync python \
      "$REPO/gentle_manip/scripts/arrhenius/_trim_eval_clips.py" "$LOGDIR" 4 40 || true
  fi
  echo "[$cat] DONE"
}

pids=(); k=0; port0=$((5600 + (${SLURM_JOB_ID:-0}) % 200))
for cat in $CATS; do
  run_one "$cat" $((port0 + k)) &
  pids+=($!); k=$((k+1))
  if [ ${#pids[@]} -ge $NCAT_PAR ]; then
    wait "${pids[0]}"; pids=("${pids[@]:1}")
    sleep 20   # stagger next server's genesis init
  fi
done
for p in "${pids[@]}"; do wait "$p"; done

echo; echo "========== AGGREGATE =========="
uv run --project envs/dppo_arrhenius --no-sync python - "$OUT" "$INDOMAIN" "$OOD" <<'PY'
import sys, json, glob, os
out, indom, ood = sys.argv[1], sys.argv[2].split(), sys.argv[3].split()
rows=[]
for d in sorted(glob.glob(os.path.join(out,"*"))):
    f=os.path.join(d,"summary.json")
    if not os.path.isfile(f): continue
    s=json.load(open(f)); cat=os.path.basename(d)
    rows.append((cat, s.get("success_rate",0), s.get("gentleness_score",0), s.get("combined_sr_gentleness",0)))
def blk(name, cats):
    sel=[r for r in rows if r[0] in cats]
    if not sel: return
    print(f"\n-- {name} --\n{'category':14s} {'SR':>6s} {'gentle':>7s} {'SRxg':>7s}")
    for c,sr,g,sg in sel: print(f"{c:14s} {sr:6.2f} {g:7.3f} {sg:7.3f}")
    n=len(sel)
    print(f"{'MEAN':14s} {sum(r[1] for r in sel)/n:6.2f} {sum(r[2] for r in sel)/n:7.3f} {sum(r[3] for r in sel)/n:7.3f}")
blk("IN-DOMAIN", indom); blk("OOD", ood)
json.dump({"rows":rows}, open(os.path.join(out,"aggregate.json"),"w"), indent=1)
print(f"\n-> {out}/aggregate.json")
PY
echo "$OUT" > "$REPO/logs/slurm_logs/last_gen_eval_dir.txt"

# self-resubmit if any category still missing a summary.json (walltime cut us off)
MISSING=""
for cat in $CATS; do [ -f "$OUT/$cat/summary.json" ] || MISSING="$MISSING $cat"; done
if [ -n "$MISSING" ]; then
  echo "[geneval] incomplete ->$MISSING ; resubmitting for the remainder"
  ENV_NAME=$ENV_NAME RUN_DIR="$RUN_DIR" CKPT="$CKPT" CATS="$MISSING" OUT="$OUT" \
    NCAT_PAR=$NCAT_PAR RECORD_BATCHES=$RECORD_BATCHES \
    sbatch --job-name="$SLURM_JOB_NAME" "$REPO/gentle_manip/scripts/arrhenius/yd_gen_eval.sbatch" || true
fi
