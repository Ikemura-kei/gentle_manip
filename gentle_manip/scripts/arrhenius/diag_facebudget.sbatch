#!/bin/bash
# One-off diagnostic: find a mesh_prep max_faces default that keeps tetgen's tet count
# under the fem_gate cap, now that mesh repair succeeds broadly (2026-09-09 fix).
#SBATCH -J diag_facebudget
#SBATCH -N 1
#SBATCH -t 0:20:00
#SBATCH -A naiss2026-3-141-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip/logs/slurm_logs/%j.out
#SBATCH --error=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip/logs/slurm_logs/%j.err

set -euo pipefail
REPO=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip
cd "$REPO"
git submodule update third_party/genesis 2>&1 | sed 's/^/[sbatch] submodule: /' || true
export PATH="$REPO/.uv_aarch64:$PATH"
unset PYTHONPATH ROS_DISTRO 2>/dev/null || true
echo "[sbatch] host=$(hostname) arch=$(uname -m) job=${SLURM_JOB_ID:-NA}"
uv run --project envs/sim_arrhenius --no-sync python gentle_manip/scripts/object_expansion/diag_facebudget.py
