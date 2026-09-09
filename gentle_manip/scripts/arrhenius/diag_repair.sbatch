#!/bin/bash
# One-off diagnostic: instrument mesh_prep.repair_watertight to find exactly where
# watertightness is lost in the production env (envs/sim_arrhenius, aarch64, GH200) --
# the object-expansion pipeline is failing repair on 92.7% of candidates.
#SBATCH -J diag_repair
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
uv run --project envs/sim_arrhenius --no-sync python gentle_manip/scripts/object_expansion/diag_repair.py
