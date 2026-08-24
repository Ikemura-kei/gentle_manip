#!/bin/bash
# Postprocess every generated run of an object, then render a turntable for one of them.
# CPU only -- runs fine on the login node, no GPU queue wait.
#   bash scripts/mesh_from_photos_finalize.sh mushroom1 [best_tag]
set -euo pipefail
OBJ="${1:?usage: finalize.sh <object> [best_tag]}"
source scripts/mesh_from_photos_env.sh
cd "$PROJ"
PY=.cache/prep_venv_x86/bin/python
RUNS="obj_meshes/$OBJ/runs"

for d in "$RUNS"/*/; do
  tag=$(basename "$d")
  [ -f "$d/raw.glb" ] || { echo "[skip] $tag: no raw.glb"; continue; }
  MEAS="obj_images/$OBJ/measurements.json"
  ARGS=(--object "$OBJ" --tag "$tag")
  [ -f "$MEAS" ] && ARGS+=(--measurements "$MEAS")
  $PY scripts/mesh_from_photos/postprocess.py "${ARGS[@]}"
done

BEST="${2:-}"
if [ -n "$BEST" ]; then
  VIEW="${BEST%%_seed*}"
  $PY scripts/mesh_from_photos/turntable.py \
      --mesh "$RUNS/$BEST/clean.obj" \
      --reference "obj_meshes/$OBJ/prepped/$VIEW.png" \
      --out "obj_meshes/$OBJ/turntable.mp4"
  cp "$RUNS/$BEST/clean.obj"   "obj_meshes/$OBJ/clean.obj"
  cp "$RUNS/$BEST/report.json" "obj_meshes/$OBJ/report.json"
  echo "[finalize] selected $BEST -> obj_meshes/$OBJ/{clean.obj,report.json,turntable.mp4}"
fi
