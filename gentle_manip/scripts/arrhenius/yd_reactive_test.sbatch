#!/bin/bash
#SBATCH -J yd_rtest
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -A naiss2026-3-141-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --output=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip/logs/slurm_logs/%j.out
#SBATCH --error=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip/logs/slurm_logs/%j.err
set -uo pipefail
REPO=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip
source "$REPO/gentle_manip/scripts/arrhenius/yd_env_common.sh"
env -u PYTHONPATH -u ROS_DISTRO MUJOCO_GL=egl uv run --project envs/sim_arrhenius --no-sync python \
  grasp_synthesis/collect_demos_diverse_start_v2.py \
  --experiment single_lift_xcat_diverse --task-name single_lift_xcat_reactive_TEST \
  --n-episodes 8 --n-envs 4 --maxfevals 500 --scene-dr-every 1 --seed 1 \
  --start-modes "sweep:0.8,above:0.2" --reactive --reactive-prob 0.9 --reactive-speed 0.15 0.45 \
  --record-video 8 --crush-frac 1.2 --out-dir "$REPO/dataset/demos"
