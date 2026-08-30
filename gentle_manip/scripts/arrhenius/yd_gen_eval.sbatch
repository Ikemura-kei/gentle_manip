#!/bin/bash
# [arrhenius] Generalist per-category eval. Loops over in-domain + OOD categories;
# for each: 100-episode canonical harness eval with EE init spanning home<->near-object
# (dr: eval_cat_<name>), one video per episode, trimmed. Aggregates the 3 metrics
# (SR / gentleness / SR*gentleness) per-category and overall.
#   CKPT=<state_N.pt> (or RUN_DIR -> best-val)   CATS="mushroom kiwi ..." (override list)
#SBATCH -J yd_geneval
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -A naiss2026-3-141-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --output=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip/logs/slurm_logs/%j.out
#SBATCH --error=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip/logs/slurm_logs/%j.err
set -euo pipefail
REPO=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip
source "$REPO/gentle_manip/scripts/arrhenius/yd_env_common.sh"

ENV_NAME=${ENV_NAME:-single_lift_xcat_regrasp_pcd}
CFG_DIR=$REPO/gentle_manip/dppo/cfg/$ENV_NAME
RUN_DIR=${RUN_DIR:-$(ls -dt $REPO/logs/dppo/dppo-pretrain/$ENV_NAME/*/ | head -1)}; RUN_DIR=${RUN_DIR%/}
if [ -z "${CKPT:-}" ]; then
  PRE_LOG=$(grep -l "$(basename "$RUN_DIR")" "$REPO"/logs/slurm_logs/*_pretrain.log 2>/dev/null | head -1 || true)
  CKPT=$("$UV_BIN_DIR/uv" run --project envs/dppo_arrhenius --no-sync python \
    "$REPO/gentle_manip/scripts/arrhenius/_pick_best_ckpt.py" "$RUN_DIR" "${PRE_LOG:-}")
fi
echo "[geneval] ckpt: $CKPT"
INDOMAIN="mushroom banana_lying kiwi egg_boiled grape cherry tomato raspberry"
OOD="blackberry scallop dumpling gelatin"          # matches the GRACE zero-shot set
SMALL="grape cherry tomato raspberry blackberry scallop"   # -> finer-grid _small eval task
CATS=${CATS:-$INDOMAIN $OOD}
OUT="$RUN_DIR/gen_eval_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$OUT"
echo "[geneval] categories: $CATS -> $OUT"

i=0
for cat in $CATS; do
  i=$((i+1)); PORT=$((5600 + (${SLURM_JOB_ID:-0} + i) % 250))
  SRV=$REPO/logs/slurm_logs/${SLURM_JOB_ID:-local}_srv_${cat}.log
  EXP=single_lift_xcat_gen_eval
  case " $SMALL " in *" $cat "*) EXP=single_lift_xcat_gen_eval_small ;; esac
  echo; echo "=== [$cat]  ($EXP) ==="
  uv run --project envs/sim_arrhenius --no-sync python -m gentle_manip.scripts.serl_sim_server \
    --experiment "$EXP" --dr eval_cat_${cat} --view student \
    --num-envs 5 --render-rgb --subprocess --port "$PORT" > "$SRV" 2>&1 &
  P=$!; trap 'kill $P 2>/dev/null||true' EXIT
  for _ in $(seq 1 180); do grep -q SIM_SERVER_READY "$SRV" && break
    kill -0 $P 2>/dev/null || { echo "[$cat] server died"; tail -30 "$SRV"; break; }; sleep 5; done
  grep -q SIM_SERVER_READY "$SRV" || { echo "[$cat] SKIP (no server)"; kill $P 2>/dev/null||true; continue; }
  LOGDIR="$OUT/$cat"
  uv run --project envs/dppo_arrhenius --no-sync python -m gentle_manip.dppo.train \
    --config-path "$CFG_DIR" --config-name eval_diffusion_pointnet \
    base_policy_path="$CKPT" ft_denoising_steps=0 experiment="$EXP" \
    env.specific.port="$PORT" logdir="$LOGDIR" || echo "[$cat] eval failed"
  kill $P 2>/dev/null || true; sleep 3
  uv run --project envs/sim_arrhenius --no-sync python \
    "$REPO/gentle_manip/scripts/arrhenius/_trim_eval_clips.py" "$LOGDIR" 4 16 || true
done

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
    print(f"\n-- {name} --")
    print(f"{'category':14s} {'SR':>6s} {'gentle':>7s} {'SRxg':>7s}")
    for c,sr,g,sg in sel: print(f"{c:14s} {sr:6.2f} {g:7.3f} {sg:7.3f}")
    n=len(sel)
    print(f"{'MEAN':14s} {sum(r[1] for r in sel)/n:6.2f} {sum(r[2] for r in sel)/n:7.3f} {sum(r[3] for r in sel)/n:7.3f}")
blk("IN-DOMAIN", indom); blk("OOD", ood)
json.dump({"rows":rows}, open(os.path.join(out,"aggregate.json"),"w"), indent=1)
print(f"\n-> {out}/aggregate.json")
PY
echo "$OUT" > "$REPO/logs/slurm_logs/last_gen_eval_dir.txt"
