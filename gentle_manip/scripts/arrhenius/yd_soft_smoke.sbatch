#!/bin/bash
#SBATCH -J yd_ssmoke
#SBATCH -N 1 -t 00:40:00 -A naiss2026-3-141-gpu -p gpu --gres=gpu:1
#SBATCH --output=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip/logs/slurm_logs/%j.out
#SBATCH --error=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip/logs/slurm_logs/%j.err
set -euo pipefail
REPO=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip
source "$REPO/gentle_manip/scripts/arrhenius/yd_env_common.sh"
rm -rf "$REPO/dataset/diag/soft_smoke"
env -u PYTHONPATH -u ROS_DISTRO MUJOCO_GL=egl uv run --project envs/sim_arrhenius --no-sync python \
  grasp_synthesis/collect_demos_diverse_start_v2.py \
  --experiment single_lift_banana_soft_diverse \
  --n-episodes 3 --n-envs 3 --maxfevals 400 --scene-dr-every 1 --seed 1 \
  --start-modes "sweep:0.5,failed:0.5" --keep-failures --record-video \
  --out-dir "$REPO/dataset/diag/soft_smoke" 2>&1 | grep -vE "\[Genesis\].*FPS" | tail -60
R=$(ls -dt "$REPO/dataset/diag/soft_smoke"/single_lift_banana_soft/* 2>/dev/null | head -1 || true)
echo "run: $R"; [ -n "$R" ] && cat "$R/stats.yaml" 2>/dev/null || true
[ -f "$R/data.pkl" ] && echo "SOFT SMOKE PASS" || echo "SOFT SMOKE FAIL"
