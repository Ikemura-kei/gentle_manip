#!/bin/bash
# [arrhenius] REACTIVE-recovery demo collection (Phase C of the reactive-policy plan).
# Same packed-collector pattern as yd_xcat_pack, but every slot runs
# collect_demos_diverse_start_v2.py --reactive: a random lateral drag is applied to the
# object mid-approach and the scripted FSM re-targets the grasp to its new position and
# re-approaches. -> demos of "object dragged away -> arm redirects -> grasps + lifts".
# Writes dataset/demos/single_lift_xcat_reactive/<ts>/ (large soft pool).
#
# Submit:  sbatch gentle_manip/scripts/arrhenius/yd_reactive_pack.sbatch
#   knobs: N_PACK (3)  PER_CAT_TARGET (400)  SEED0 (200)
#SBATCH -J yd_rpack
#SBATCH -N 1
#SBATCH -t 08:00:00
#SBATCH -A naiss2026-3-141-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=192
#SBATCH --output=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip/logs/slurm_logs/%j.out
#SBATCH --error=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip/logs/slurm_logs/%j.err
set -uo pipefail
REPO=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip
source "$REPO/gentle_manip/scripts/arrhenius/yd_env_common.sh"

N_PACK=${N_PACK:-3}
PER_CAT_TARGET=${PER_CAT_TARGET:-400}
SEED0=${SEED0:-200}
MAXFEVALS=${MAXFEVALS:-350}
# reactive-heavy starts: mostly clean approaches (the DRAG is the perturbation); no failed_grasp
START_MODES=${START_MODES:-sweep:0.72,above:0.16,ground:0.08,air:0.04}
N_ENVS=${N_ENVS:-14}

echo "[rpack] N_PACK=$N_PACK per_cat=$PER_CAT_TARGET seed0=$SEED0"
nvidia-smi --query-gpu=memory.total,memory.used --format=csv || true

pids=()
for i in $(seq 0 $((N_PACK-1))); do
  SEED=$((SEED0 + i))
  LOG="$REPO/logs/slurm_logs/${SLURM_JOB_ID}_rslot${i}.out"
  CAT_HAVE=$(python3 - "$REPO/dataset/demos/single_lift_xcat_reactive" <<'PYH'
import sys, glob, pickle, collections, os
root = sys.argv[1]; have = collections.Counter()
for f in glob.glob(os.path.join(root, "*", "data.pkl")) + glob.glob(os.path.join(root, "*", "shard_*.pkl")):
    try:
        d = pickle.load(open(f, "rb"))
        for e in d.get("episodes", []):
            have[e.get("category") or e.get("task_name") or "?"] += 1
    except Exception:
        pass
print(",".join(f"{k}:{v}" for k, v in have.items() if k != "?"))
PYH
)
  echo "[rpack] slot $i seed=$SEED have='${CAT_HAVE:-none}' -> $LOG"
  ( echo "[reactive-collect] slot $i seed=$SEED"
    env -u PYTHONPATH -u ROS_DISTRO MUJOCO_GL=egl uv run --project envs/sim_arrhenius --no-sync python \
      grasp_synthesis/collect_demos_diverse_start_v2.py \
      --experiment single_lift_xcat_diverse --task-name single_lift_xcat_reactive \
      --n-episodes 4000 --n-envs "$N_ENVS" --maxfevals "$MAXFEVALS" \
      --scene-dr-every 1 --seed "$SEED" --start-modes "$START_MODES" \
      --reactive --reactive-prob 0.85 --reactive-speed 0.12 0.38 \
      --per-cat-target "$PER_CAT_TARGET" --cat-have "${CAT_HAVE:-}" \
      --crush-frac 1.20 --record-video 0 --out-dir "$REPO/dataset/demos"
  ) > "$LOG" 2>&1 &
  pids+=($!)
  sleep 90
done

echo "[rpack] ${#pids[@]} collectors launched: ${pids[*]}"
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "[rpack] all slots done (rc=$rc)"

TOT=$(python3 - "$REPO/dataset/demos/single_lift_xcat_reactive" <<'PYT'
import sys, glob, pickle, os
n = 0
for f in glob.glob(os.path.join(sys.argv[1], "*", "data.pkl")) + glob.glob(os.path.join(sys.argv[1], "*", "shard_*.pkl")):
    try: n += len(pickle.load(open(f, "rb")).get("episodes", []))
    except Exception: pass
print(n)
PYT
)
echo "[rpack] total reactive episodes on disk: $TOT"
if [ "${TOT:-0}" -lt 1400 ]; then
  echo "[rpack] < 1400 -> resubmit"
  SEED0=$((SEED0 + N_PACK)) N_PACK=$N_PACK PER_CAT_TARGET=$PER_CAT_TARGET \
    sbatch --job-name="$SLURM_JOB_NAME" "$REPO/gentle_manip/scripts/arrhenius/yd_reactive_pack.sbatch" || true
fi
