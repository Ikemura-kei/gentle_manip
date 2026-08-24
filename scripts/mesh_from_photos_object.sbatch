#!/bin/bash
#SBATCH -A naiss2026-3-141-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 03:00:00
#SBATCH -J triposg-obj
#SBATCH -o logs/mesh_from_photos/obj-%j.out
#SBATCH -e logs/mesh_from_photos/obj-%j.out
# End-to-end for ONE object: generate -> postprocess -> turntable -> compare sheet.
# Assumes prep_images.py has already run AND a human has reviewed prepped/_contact_sheet.png.
#   sbatch --export=ALL,OBJ=banana1,VIEWS="a b",SEEDS="0 1 2" scripts/mesh_from_photos_object.sbatch
set -uo pipefail
OBJ="${OBJ:?set OBJ}"
SEEDS="${SEEDS:-0 1 2}"
source scripts/mesh_from_photos_env.sh
cd "$PROJ"
RUN="uv run --project envs/triposg_arrhenius python"
RUNS="obj_meshes/$OBJ/runs"
MEAS="obj_images/$OBJ/measurements.json"

# Default VIEWS = every prepped view.
if [ -z "${VIEWS:-}" ]; then
  VIEWS=$(cd "obj_meshes/$OBJ/prepped" && ls *.png | grep -v '^_' | sed 's/\.png$//' | tr '\n' ' ')
fi
echo "OBJ=$OBJ VIEWS=[$VIEWS] SEEDS=[$SEEDS]"

$RUN scripts/mesh_from_photos/generate.py --object "$OBJ" --views $VIEWS --seeds $SEEDS \
  || { echo "[FATAL] generation failed"; exit 1; }
echo "=== GENERATE OK ==="

for d in "$RUNS"/*/; do
  tag=$(basename "$d"); [ -f "$d/raw.glb" ] || continue
  ARGS=(--object "$OBJ" --tag "$tag"); [ -f "$MEAS" ] && ARGS+=(--measurements "$MEAS")
  echo "--- $tag ---"
  $RUN scripts/mesh_from_photos/postprocess.py "${ARGS[@]}" \
    || echo "[ERROR] postprocess failed for $tag (continuing)"
done
echo "=== POSTPROCESS OK ==="

for d in "$RUNS"/*/; do
  tag=$(basename "$d"); view="${tag%%_seed*}"
  [ -f "$d/clean.obj" ] || { echo "[ERROR] no clean.obj for $tag"; continue; }
  $RUN scripts/mesh_from_photos/turntable.py --mesh "$d/clean.obj" \
      --reference "obj_meshes/$OBJ/prepped/$view.png" \
      --out "$d/turntable.mp4" --frames 90 --size 640 \
    || echo "[ERROR] turntable failed for $tag (continuing)"
done
echo "=== TURNTABLE OK ==="

$RUN scripts/mesh_from_photos/compare_sheet.py --object "$OBJ" || true
echo "=== DONE $OBJ ==="
