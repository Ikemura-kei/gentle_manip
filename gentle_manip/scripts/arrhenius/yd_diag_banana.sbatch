#!/bin/bash
#SBATCH -J yd_diag
#SBATCH -N 1
#SBATCH -t 01:30:00
#SBATCH -A naiss2026-3-141-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip/logs/slurm_logs/%j.out
#SBATCH --error=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip/logs/slurm_logs/%j.err
set -euo pipefail
REPO=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip
source "$REPO/gentle_manip/scripts/arrhenius/yd_env_common.sh"

run() {
  local tag="$1"; shift
  echo; echo "########## DIAG $tag ##########"
  rm -rf "$REPO/dataset/diag/$tag"
  env -u PYTHONPATH -u ROS_DISTRO MUJOCO_GL=egl uv run --project envs/sim_arrhenius --no-sync python \
    grasp_synthesis/collect_demos_diverse_start_v2.py \
    --experiment single_lift_banana_rigid_diverse \
    --out-dir "$REPO/dataset/diag/$tag" --keep-failures --record-video \
    "$@" 2>&1 | grep -vE "\[Genesis\].*FPS|Iterat #Fevals|^ +[0-9]+ +[0-9]+ +[0-9.e+-]+ " | tail -60
  R=$(ls -dt "$REPO/dataset/diag/$tag"/single_lift_banana_rigid/* 2>/dev/null | head -1 || true)
  echo "run dir: $R"; [ -n "$R" ] && cat "$R/stats.yaml" 2>/dev/null || true
}
# A: home start, NO scene DR -- baseline: does a lying-banana top-down grasp lift?
run A_home_noDR --n-episodes 6 --n-envs 6 --maxfevals 600 --scene-dr-every 0 --start-modes "home:1.0"
# B: home start + scene DR (shape/scale deformation)
run B_home_DR   --n-episodes 6 --n-envs 6 --maxfevals 600 --scene-dr-every 1 --start-modes "home:1.0"
# C: full diverse start modes + scene DR
run C_full      --n-episodes 8 --n-envs 8 --maxfevals 600 --scene-dr-every 1 \
    --start-modes "home:0.4,near_object:0.3,near_ground:0.15,mid_air:0.15"
echo; echo "DIAG DONE"
