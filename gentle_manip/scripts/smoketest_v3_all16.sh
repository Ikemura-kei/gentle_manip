#!/usr/bin/env bash
# Smoke test collect_demos_synth_v3.py (FEM gentleness grasp synthesis) across all
# 16 roster categories (12 training + 4 zero-shot test) before trusting it at scale.
# Small n-episodes/n-envs/maxfevals for speed; --record-video so every synthesized
# grasp can be visually inspected. Output goes to dataset/demos_smoketest_v3_all16/
# (gitignored scratch dir, matches dataset/demos_smoketest*/ pattern).
#
# Run from repo root:
#   nohup bash gentle_manip/scripts/smoketest_v3_all16.sh >> logs/smoketest_v3_all16/driver.log 2>&1 &
set -uo pipefail

CATEGORIES=(mushroom raspberry grape kiwi egg_boiled strawberry banana tomato \
            chicken_breast shrimp pasta_bundle cherry scallop gelatin blackberry dumpling)

OUT_DIR="dataset/demos_smoketest_v3_all16"
LOG_DIR="logs/smoketest_v3_all16"
mkdir -p "$OUT_DIR" "$LOG_DIR"

echo "=== smoketest_v3_all16 starting $(date) ==="
for cat in "${CATEGORIES[@]}"; do
  echo ""
  echo "=== [$cat] $(date) ==="
  env -u PYTHONPATH -u VIRTUAL_ENV MUJOCO_GL=egl uv run --project envs/sim --no-sync python grasp_synthesis/collect_demos_synth_v3.py \
    --experiment "single_lift_${cat}_soft_easy" \
    --n-episodes 3 --n-envs 3 --maxfevals 150 \
    --out-dir "$OUT_DIR" --shard-size 3 \
    --grasp-gpu --record-video \
    > "$LOG_DIR/${cat}.log" 2>&1
  status=$?
  n_eps=$(find "$OUT_DIR/single_lift_${cat}_soft" -name "*.pkl" 2>/dev/null | xargs -I{} python3 -c "
import pickle,sys
try:
    d=pickle.load(open('{}','rb'))
    print(len(d['episodes']))
except Exception:
    print(0)
" 2>/dev/null | awk '{s+=$1} END{print s+0}')
  n_vids=$(find "$OUT_DIR/single_lift_${cat}_soft" -iname "*.mp4" 2>/dev/null | wc -l)
  echo "=== [$cat] exit=$status episodes_saved=$n_eps videos=$n_vids ==="
done
echo ""
echo "=== smoketest_v3_all16 done $(date) ==="
