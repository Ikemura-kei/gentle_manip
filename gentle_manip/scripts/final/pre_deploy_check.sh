#!/bin/bash
# Live view of the PROCESSED real cloud (the deploy twin obs config; ground_residual filter is default ON in it).
# Key M toggles the filter on/off live (starts ON); ESC quits; do not press SPACE (it would save an episode).
# --speed 0.3: slow teleop for nudging the arm while looking at the cloud.

uv run --project envs/deploy python -m gentle_manip.demos.record \
  --setup gentle_manip/configs/setup/real_lab.yaml \
  --obs-config gentle_manip/configs/obs/point_cloud_1cam_armfocus.yaml \
  --action-config gentle_manip/configs/action/delta_pose_delta_gripper_fast_rot.yaml \
  --task-name pcd_preview --input keyboard --show-pointcloud --speed 0.3