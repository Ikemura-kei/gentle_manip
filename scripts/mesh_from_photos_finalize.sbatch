#!/bin/bash
#SBATCH -A naiss2026-3-141-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 02:00:00
#SBATCH -J triposg-final
#SBATCH -o logs/mesh_from_photos/final-%j.out
#SBATCH -e logs/mesh_from_photos/final-%j.out
# Cleanup + validation + turntables. CPU-only work, but the gpu partition is the only
# one this account can submit to, and the login node kills jobs this size (exit 144
# on a 1.9M-face mesh).
set -euo pipefail
OBJ="${OBJ:-mushroom1}"
source scripts/mesh_from_photos_env.sh
cd "$PROJ"
RUN="uv run --project envs/triposg_arrhenius python"
RUNS="obj_meshes/$OBJ/runs"
MEAS="obj_images/$OBJ/measurements.json"

for d in "$RUNS"/*/; do
  tag=$(basename "$d")
  [ -f "$d/raw.glb" ] || continue
  ARGS=(--object "$OBJ" --tag "$tag")
  [ -f "$MEAS" ] && ARGS+=(--measurements "$MEAS")
  echo "--- $tag ---"
  $RUN scripts/mesh_from_photos/postprocess.py "${ARGS[@]}" \
    || echo "[ERROR] postprocess failed for $tag (continuing)"
done
echo "=== POSTPROCESS OK ==="

# Turntable for each candidate named in TAGS (space separated).
for tag in ${TAGS:-}; do
  view="${tag%%_seed*}"
  [ -f "$RUNS/$tag/clean.obj" ] || { echo "[ERROR] no clean.obj for $tag, skipping turntable"; continue; }
  $RUN scripts/mesh_from_photos/turntable.py \
      --mesh "$RUNS/$tag/clean.obj" \
      --reference "obj_meshes/$OBJ/prepped/$view.png" \
      --out "$RUNS/$tag/turntable.mp4" --frames 90 --size 640 \
    || echo "[ERROR] turntable failed for $tag (continuing)"
done
echo "=== TURNTABLE OK ==="
