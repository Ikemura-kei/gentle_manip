#! /bin/bash
# [final] Visualize a real-deploy recording (deploy_dppo.sh --record ...): per episode
#   <run>/viz/ep_NNN_cloud.mp4    RGB (if --record-rgb was on) | point cloud in 3 views, frame-locked
#   <run>/viz/ep_NNN_signals.png  commanded target (decoded through ACTION_CFG) vs measured state
# usage: bash gentle_manip/scripts/final/viz_deploy.sh <record dir> [extra args, e.g. --no-video --stride 3]
# ACTION_CFG must be the yaml the policy was deployed with (default = the sim2real z15 yaml).
set -euo pipefail
cd "$(dirname "$0")/../../.."
RUN=${1:?usage: viz_deploy.sh <record dir> [extra args]}; shift
ACTION_CFG=${ACTION_CFG:-gentle_manip/configs/action/abs_pose_euler_abs_gripper_z15.yaml}
CUDA_VISIBLE_DEVICES="" uv run --project envs/deploy python -m gentle_manip.visualization.deploy_episode_viz \
    "$RUN" --action-config "$ACTION_CFG" "$@"
