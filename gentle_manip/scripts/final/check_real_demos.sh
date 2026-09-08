#! /bin/bash
# [final] Inspect a real demo set collected with collect_real_world_demo.sh:
#   1. RENDER  <run>/viz/ep_NNN_cloud.mp4    paired RGB | point cloud (3 views), frame-locked
#              <run>/viz/ep_NNN_signals.png  commanded action (decoded) vs measured proprio
#   2. CHECK   episode counts/lengths, channel integrity, cloud occupancy, and the gripper
#              CLOSING SPEED vs the sim demos the generalist was trained on -> <run>/checks/checks.json
#
#   bash gentle_manip/scripts/final/check_real_demos.sh dataset/demos/single_lift_cherry_tomato_real/26-09-08-rzz
#
# Env knobs:
#   EPISODES=3      render only the first N episodes (default: all; ~1 min/episode for the video)
#   STRIDE=4        video frame stride (default 2) — bigger = faster, choppier
#   NO_VIDEO=1      skip the videos, keep the signal plots and the checks
#   CHECKS_ONLY=1   skip rendering entirely
#   SIM=<dir>       sim reference set (default dataset/dppo/single_lift_generalist_soft_v5; "" = skip)
#   ACTION_CFG=...  action yaml the demos were RECORDED with (default: the z15 yaml the generalist uses)
set -euo pipefail
cd "$(dirname "$0")/../../.."
RUN=${1:?usage: check_real_demos.sh <demo run dir> [extra args for the renderer]}; shift || true
ACTION_CFG=${ACTION_CFG:-gentle_manip/configs/action/abs_pose_euler_abs_gripper_z15.yaml}
SIM=${SIM-dataset/dppo/single_lift_generalist_soft_v5}

if [ "${CHECKS_ONLY:-0}" != "1" ]; then
  echo "== rendering $RUN -> $RUN/viz"
  CUDA_VISIBLE_DEVICES="" uv run --project envs/deploy python -m gentle_manip.visualization.deploy_episode_viz \
    "$RUN" --action-config "$ACTION_CFG" \
    ${EPISODES:+--episodes $EPISODES} ${NO_VIDEO:+--no-video} ${STRIDE:+--stride $STRIDE} "$@"
fi

echo "== checks"
CUDA_VISIBLE_DEVICES="" uv run --project envs/deploy python -m gentle_manip.scripts.final.check_real_demos \
  "$RUN" --sim-dataset "$SIM"
