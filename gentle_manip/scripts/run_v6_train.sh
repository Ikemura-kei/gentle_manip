#!/bin/bash
# [local] Part D steps 2-3: audit -> convert to 7d euler -> bwvei-style BC train.
#
#   bash gentle_manip/scripts/run_v5_train.sh dataset/demos/single_lift_mushroom_soft/<run>
#
# Stops at the first failed step. Eval (step 4) is launched separately once checkpoints exist —
# every checkpoint goes through the canonical harness (docs/training_and_eval.md).
set -eu
RUN=${1:?demo run dir}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"

echo "=== 1/3 rate-bound audit (the dataset is what trains, so the dataset is what's audited) ==="
env -u PYTHONPATH -u ROS_DISTRO uv run --project envs/sim --no-sync python \
    -m gentle_manip.scripts.audit_demo_rate_bound "$RUN" \
    --experiment single_lift_mushroom_soft_abs_action_robust

echo "=== 2/3 convert -> 7d euler absolute (seam-fixed) + point cloud ==="
env -u PYTHONPATH -u ROS_DISTRO uv run --project envs/dppo python -m gentle_manip.dppo.convert_demos \
    "$RUN" \
    --out dataset/dppo/single_lift_mushroom_soft_v6_7d \
    --obs-keys ee_pos ee_quat gripper_width --point-cloud \
    --derive-action gentle_manip/configs/action/abs_pose_euler_abs_gripper.yaml \
    --derive-source-action gentle_manip/configs/action/abs_pose_abs_gripper.yaml

echo "=== 3/3 BC train (bwvei setup: 800 epochs, save/200, batch 128, lr 1e-4) ==="
CFG="--config-path $REPO/gentle_manip/dppo/cfg/single_lift_mushroom_soft_v6_7d --config-name pre_diffusion_pointnet"
env -u PYTHONPATH -u ROS_DISTRO uv run --project envs/dppo python -m gentle_manip.dppo.train $CFG \
    wandb=null experiment=single_lift_mushroom_soft_abs_action_robust
