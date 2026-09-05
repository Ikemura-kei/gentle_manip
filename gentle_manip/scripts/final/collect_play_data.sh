#!/bin/bash

obj=red_cube  
uv run --project envs/deploy python -m gentle_manip.demos.record \
  --setup gentle_manip/configs/setup/real_lab.yaml \
  --obs-config gentle_manip/configs/obs/point_cloud_1cam_armfocus_rgb.yaml \
  --action-config gentle_manip/configs/action/delta_pose_delta_gripper_fast_rot.yaml \
  --record-action-config gentle_manip/configs/action/abs_pose_euler_abs_gripper.yaml \
  --task-name play_${obj}_real \
  --input spacemouse-kb \
  --description "The play data used for sim-real paired encoder regularization" \
  --show-pointcloud