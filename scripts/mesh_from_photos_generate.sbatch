#!/bin/bash
#SBATCH -A naiss2026-3-141-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 03:00:00
#SBATCH -J triposg-gen
#SBATCH -o logs/mesh_from_photos/gen-%j.out
#SBATCH -e logs/mesh_from_photos/gen-%j.out
set -euo pipefail
source scripts/mesh_from_photos_env.sh
cd "$PROJ"
echo "host=$(hostname) arch=$(uname -m)"
uv sync --project envs/triposg_arrhenius
echo "=== SYNC OK ==="
uv run --project envs/triposg_arrhenius python scripts/mesh_from_photos/generate.py \
  --object mushroom1 --views back front left right --seeds 0 1 2
echo "=== GENERATE OK ==="
