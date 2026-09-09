#!/bin/bash
# Compress one grasp-demo clip per fully-collected category to a small base64-embeddable
# JSON payload for a standalone video-gallery Artifact.
#SBATCH -J export_videos
#SBATCH -N 1
#SBATCH -t 0:30:00
#SBATCH -A naiss2026-3-141-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip/logs/slurm_logs/%j.out
#SBATCH --error=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip/logs/slurm_logs/%j.err

set -euo pipefail
REPO=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip
cd "$REPO"
export PATH="$REPO/.uv_aarch64:$PATH"
unset PYTHONPATH ROS_DISTRO 2>/dev/null || true
echo "[sbatch] host=$(hostname) arch=$(uname -m) job=${SLURM_JOB_ID:-NA}"
uv run --project envs/sim_arrhenius --no-sync python gentle_manip/scripts/object_expansion/export_video_gallery.py \
    --out dataset/object_expansion/video_gallery.json --max-kb 260
