#!/bin/bash
# [arrhenius] PACKED cross-category collection: run N collector processes inside ONE
# GPU allocation. Each genesis grasp-collector uses only ~7 GB / 98 GB and ~0-15% GPU
# (grasp execution is CPU/CMA + contact-physics bound, not GPU-compute bound), so a
# single GH200 comfortably hosts 3-4 collectors -> 3-4x throughput per reserved GPU,
# and it costs the shared project GrpTRESRunMins pool only ONE job's worth of minutes.
#
# Slots (default 3): 1x large-object pool (xcat_regrasp) + 2x small-fruit (xcat_small).
# Each slot writes its own timestamped run dir; per-cat quota + CAT_HAVE auto-compute
# from the shared log dir keep them loosely coordinated (same as separate jobs).
#
# Submit:  sbatch gentle_manip/scripts/arrhenius/yd_xcat_pack.sbatch
#   knobs: N_PACK (3)  WALL (-t)  PER_CAT_TARGET (500)  SEED0 (30)
#SBATCH -J yd_xpack
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -A naiss2026-3-141-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=192
#SBATCH --output=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip/logs/slurm_logs/%j.out
#SBATCH --error=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip/logs/slurm_logs/%j.err
set -euo pipefail
REPO=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip
source "$REPO/gentle_manip/scripts/arrhenius/yd_env_common.sh"

N_PACK=${N_PACK:-3}
PER_CAT_TARGET=${PER_CAT_TARGET:-500}
SEED0=${SEED0:-30}
MAXFEVALS=${MAXFEVALS:-700}

# slot spec: "TAG:EXPERIMENT:N_ENVS"
SLOTS=(
  "xcat_regrasp:single_lift_xcat_diverse:10"
  "xcat_small:single_lift_xcat_small_diverse:12"
  "xcat_small:single_lift_xcat_small_diverse:12"
  "xcat_regrasp:single_lift_xcat_diverse:10"
)
START_MODES=${START_MODES:-sweep:0.44,failed:0.30,above:0.10,ground:0.09,air:0.07}

echo "[xpack] N_PACK=$N_PACK per_cat=$PER_CAT_TARGET seed0=$SEED0"
nvidia-smi --query-gpu=memory.total,memory.used --format=csv || true

pids=()
for i in $(seq 0 $((N_PACK-1))); do
  spec="${SLOTS[$i]}"
  TAG="${spec%%:*}"; rest="${spec#*:}"
  EXPERIMENT="${rest%%:*}"; N_ENVS="${rest##*:}"
  TASK_DIR="single_lift_${TAG}"
  SEED=$((SEED0 + i))

  # per-category quota: count OK episodes already collected in ALL prior/live logs of this TAG
  CAT_HAVE=$(python3 - "$REPO/logs/slurm_logs" "$TAG" <<'PYH'
import sys, re, glob
logdir, tag = sys.argv[1], sys.argv[2]
have={}
for lg in glob.glob(f"{logdir}/*.out"):
    if f"tag={tag} " not in open(lg,errors="ignore").read(800): continue
    cur=None
    for ln in open(lg,errors="ignore"):
        m=re.search(r'scene object -> (\w+)', ln) or re.search(r'scene_cat=(\w+)', ln)
        if m: cur=m.group(1)
        if re.search(r'\bep \d+: env \d+\s+OK\b', ln) and cur: have[cur]=have.get(cur,0)+1
print(",".join(f"{k}:{v}" for k,v in have.items()))
PYH
)
  LOG="$REPO/logs/slurm_logs/${SLURM_JOB_ID}_slot${i}_${TAG}.out"
  echo "[xpack] slot $i: tag=$TAG exp=$EXPERIMENT n_envs=$N_ENVS seed=$SEED have='${CAT_HAVE:-none}' -> $LOG"
  ( echo "[xcat] tag=$TAG exp=$EXPERIMENT (packed slot $i)"
    env -u PYTHONPATH -u ROS_DISTRO MUJOCO_GL=egl uv run --project envs/sim_arrhenius --no-sync python \
      grasp_synthesis/collect_demos_diverse_start_v2.py \
      --experiment "$EXPERIMENT" --task-name "$TASK_DIR" \
      --n-episodes 4500 --n-envs "$N_ENVS" --maxfevals "$MAXFEVALS" \
      --scene-dr-every 1 --seed "$SEED" --start-modes "$START_MODES" \
      --per-cat-target "$PER_CAT_TARGET" --cat-have "${CAT_HAVE:-}" \
      --crush-frac 1.20 --record-video 0 --out-dir "$REPO/dataset/demos"
  ) > "$LOG" 2>&1 &
  pids+=($!)
  sleep 90   # stagger genesis inits so they don't collide on CUDA context creation
done

echo "[xpack] ${#pids[@]} collectors launched: ${pids[*]}"
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "[xpack] all slots done (rc=$rc)"

# self-resubmit unless every slot's TAG reports all-categories-at-target
if grep -q "every category reached its per-cat target" "$REPO"/logs/slurm_logs/${SLURM_JOB_ID}_slot*.out 2>/dev/null; then
  echo "[xpack] a slot finished its pool; check others before manual resubmit"
fi
echo "[xpack] resubmitting"
SEED0=$((SEED0 + N_PACK)) N_PACK=$N_PACK PER_CAT_TARGET=$PER_CAT_TARGET \
  sbatch --job-name="$SLURM_JOB_NAME" "$REPO/gentle_manip/scripts/arrhenius/yd_xcat_pack.sbatch" || true
