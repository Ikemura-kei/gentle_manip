#!/bin/bash
# [arrhenius] REACTIVE + RECOVERY demo collection v4 -- ALL 8 in-domain categories.
# Fork of yd_reactive_pack: 4 packed collector slots, 2x large-object pool
# (single_lift_xcat_diverse, now mushroom/banana/kiwi/egg) + 2x small-fruit
# (single_lift_xcat_small_diverse, grape/cherry/tomato/raspberry). Every slot runs
# collect_demos_diverse_start_v2.py --reactive: a random mid-approach lateral drag
# (0.30-0.90 m/s, MATCHED to the eval perturbation 0.40-0.95) + scripted re-target,
# AND --start-modes carries failed:0.28 so the same run also yields failed_grasp
# recovery demos. The v2 collector now has a dedicated FIRM phase (extra grip close
# before the lift) so demo grasps are firm enough to lift first-try.
# Writes dataset/demos/single_lift_xcat_reactive_v4/<ts>/.
#
# Submit:  sbatch gentle_manip/scripts/arrhenius/yd_reactive_pack_v4.sbatch
#   knobs: PER_CAT_TARGET (250)  SEED0 (400)  MAXFEVALS (400)  STOP_AT (1800)
#SBATCH -J yd_rpack4
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -A naiss2026-3-141-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=192
#SBATCH --output=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip/logs/slurm_logs/%j.out
#SBATCH --error=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip/logs/slurm_logs/%j.err
set -uo pipefail
REPO=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip
source "$REPO/gentle_manip/scripts/arrhenius/yd_env_common.sh"

PER_CAT_TARGET=${PER_CAT_TARGET:-250}
SEED0=${SEED0:-400}
MAXFEVALS=${MAXFEVALS:-320}
STOP_AT=${STOP_AT:-1100}
OUT_TASK=single_lift_xcat_reactive_v4
START_MODES=${START_MODES:-sweep:0.50,failed:0.28,above:0.10,ground:0.07,air:0.05}

# slot spec: "EXPERIMENT:N_ENVS" -- 6 packed collectors (GPU has ~97 GB, ~7 GB/slot;
# 192 CPUs / 72 envs = 2.6 cores/env). CMA is the bottleneck, so more slots ~= more throughput.
SLOTS=(
  "single_lift_xcat_diverse:12"
  "single_lift_xcat_small_diverse:12"
  "single_lift_xcat_diverse:12"
  "single_lift_xcat_small_diverse:12"
  "single_lift_xcat_diverse:12"
  "single_lift_xcat_small_diverse:12"
)
N_PACK=${N_PACK:-${#SLOTS[@]}}

echo "[rpack4] per_cat=$PER_CAT_TARGET seed0=$SEED0 stop_at=$STOP_AT modes=$START_MODES"
nvidia-smi --query-gpu=memory.total,memory.used --format=csv || true

_have() {   # per-category OK counts across all v4 dirs
  python3 - "$REPO/dataset/demos/$OUT_TASK" <<'PYH'
import sys, glob, pickle, collections, os
root = sys.argv[1]; have = collections.Counter()
for f in glob.glob(os.path.join(root, "*", "data.pkl")) + glob.glob(os.path.join(root, "*", "shard_*.pkl")):
    try:
        d = pickle.load(open(f, "rb"))
        for e in d.get("episodes", []):
            have[e.get("category") or e.get("object_name") or "?"] += 1
    except Exception:
        pass
print(",".join(f"{k}:{v}" for k, v in have.items() if k != "?"))
PYH
}

pids=()
for i in $(seq 0 $((N_PACK-1))); do
  spec="${SLOTS[$i]}"; EXP="${spec%%:*}"; N_ENVS="${spec##*:}"
  SEED=$((SEED0 + i))
  CAT_HAVE=$(_have)
  LOG="$REPO/logs/slurm_logs/${SLURM_JOB_ID}_rslot${i}.out"
  echo "[rpack4] slot $i exp=$EXP n_envs=$N_ENVS seed=$SEED have='${CAT_HAVE:-none}' -> $LOG"
  ( echo "[reactive-v4] slot $i exp=$EXP seed=$SEED"
    env -u PYTHONPATH -u ROS_DISTRO MUJOCO_GL=egl uv run --project envs/sim_arrhenius --no-sync python \
      grasp_synthesis/collect_demos_diverse_start_v2.py \
      --experiment "$EXP" --task-name "$OUT_TASK" \
      --n-episodes 5000 --n-envs "$N_ENVS" --maxfevals "$MAXFEVALS" \
      --scene-dr-every 1 --seed "$SEED" --start-modes "$START_MODES" \
      --reactive --reactive-prob 0.85 --reactive-speed 0.30 0.90 --reactive-frame 12 45 \
      --per-cat-target "$PER_CAT_TARGET" --cat-have "${CAT_HAVE:-}" \
      --crush-frac 1.20 --record-video 0 --out-dir "$REPO/dataset/demos"
  ) > "$LOG" 2>&1 &
  pids+=($!)
  sleep 90
done

echo "[rpack4] ${#pids[@]} collectors launched: ${pids[*]}"
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "[rpack4] all slots done (rc=$rc)"

TOT=$(python3 - "$REPO/dataset/demos/$OUT_TASK" <<'PYT'
import sys, glob, pickle, os
n = rr = 0
for f in glob.glob(os.path.join(sys.argv[1], "*", "data.pkl")) + glob.glob(os.path.join(sys.argv[1], "*", "shard_*.pkl")):
    try:
        for e in pickle.load(open(f, "rb")).get("episodes", []):
            n += 1
            if e.get("start_mode") == "reactive_recover": rr += 1
    except Exception: pass
print(f"{n} {rr}")
PYT
)
echo "[rpack4] total v4 episodes / reactive_recover: $TOT"
NRR=$(echo "$TOT" | awk '{print $2}')
if [ "${NRR:-0}" -lt "$STOP_AT" ]; then
  echo "[rpack4] reactive_recover $NRR < $STOP_AT -> resubmit"
  SEED0=$((SEED0 + N_PACK)) PER_CAT_TARGET=$PER_CAT_TARGET STOP_AT=$STOP_AT \
    sbatch --job-name="$SLURM_JOB_NAME" "$REPO/gentle_manip/scripts/arrhenius/yd_reactive_pack_v4.sbatch" || true
else
  echo "[rpack4] target reached ($NRR reactive_recover) -> not resubmitting"
fi
