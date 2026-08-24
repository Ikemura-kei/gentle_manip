#!/bin/bash
#SBATCH -A naiss2026-3-141-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 01:00:00
#SBATCH -J triposg-trace
#SBATCH -o logs/mesh_from_photos/trace-%j.out
#SBATCH -e logs/mesh_from_photos/trace-%j.out
set -uo pipefail
source scripts/mesh_from_photos_env.sh
cd "$PROJ"
RUN="uv run --project envs/triposg_arrhenius python"
$RUN scripts/mesh_from_photos/genus_trace.py --object strawberry1 --tag photo_seed0
$RUN scripts/mesh_from_photos/genus_trace.py --object banana1 --tag IMG20260824145616_seed0
$RUN scripts/mesh_from_photos/genus_trace.py --object mushroom1 --tag back_seed0
echo "=== TRACE DONE ==="
