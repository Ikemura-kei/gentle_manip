#!/bin/bash
#SBATCH -A naiss2026-3-141-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 01:00:00
#SBATCH -J triposg-sync
#SBATCH -o logs/mesh_from_photos/sync-%j.out
#SBATCH -e logs/mesh_from_photos/sync-%j.out
set -euo pipefail
source scripts/mesh_from_photos_env.sh
cd "$PROJ"
echo "host=$(hostname) arch=$(uname -m)"
uv sync --project envs/triposg_arrhenius
echo "=== SYNC OK ==="
uv run --project envs/triposg_arrhenius python -c "
import torch, sys
print('python', sys.version.split()[0])
print('torch', torch.__version__, 'cuda avail', torch.cuda.is_available())
print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
import diffusers, transformers, trimesh, skimage, rembg, imageio, fast_simplification
print('diffusers', diffusers.__version__, 'transformers', transformers.__version__)
print('trimesh', trimesh.__version__, 'skimage', skimage.__version__, 'fast_simplification OK')
"
echo "=== VERIFY OK ==="
