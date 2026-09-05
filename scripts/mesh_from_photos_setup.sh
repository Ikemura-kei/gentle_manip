#!/bin/bash
# One-command setup for the photo->mesh (TripoSG) pipeline.
# Safe to re-run. See docs/triposg_setup.md for what each step does and why.
#
#   bash scripts/mesh_from_photos_setup.sh
#
# MUST be run where aarch64 wheels are correct, i.e. inside a GPU-node job on Arrhenius:
#   srun -A naiss2026-3-141-gpu -p gpu --gres=gpu:1 -n1 -c16 -t 1:00:00 \
#        bash scripts/mesh_from_photos_setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/mesh_from_photos_env.sh

echo "[1/4] submodule"
git submodule update --init third_party/TripoSG
echo "      pinned at $(git -C third_party/TripoSG rev-parse --short HEAD)"

echo "[2/4] arch check"
if [ "$(uname -m)" != "aarch64" ]; then
  echo "      WARNING: on $(uname -m), not aarch64. envs/triposg_arrhenius pins aarch64" >&2
  echo "      cu126 torch wheels; sync will resolve the wrong build. Run this in a GPU job." >&2
fi

echo "[3/4] uv sync (torch ~2 GB on first run)"
uv sync --project envs/triposg_arrhenius

echo "[4/4] verify"
uv run --project envs/triposg_arrhenius python - <<'PY'
import sys
from pathlib import Path
R = Path.cwd(); T = R/"third_party"/"TripoSG"
sys.path.insert(0, str(T)); sys.path.insert(0, str(T/"scripts"))
sys.path.append(str(R/"scripts"/"mesh_from_photos"/"shims"))
import torch
print("      torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
from triposg.pipelines.pipeline_triposg import TripoSGPipeline  # noqa: F401
print("      TripoSGPipeline import OK")
PY
echo "OK. Next: docs/triposg_setup.md -> 'Running it'"
