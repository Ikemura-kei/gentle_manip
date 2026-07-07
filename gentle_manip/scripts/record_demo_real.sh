#!/bin/bash

uv run --project envs/deploy python -m gentle_manip.demos.record \
  --setup      gentle_manip/configs/setup/real_lab_tactile.yaml \
  --obs-config gentle_manip/configs/obs/real_tactile.yaml \
  --action-config gentle_manip/configs/action/delta_pose_delta_gripper.yaml \
  --task-name  mushroom_lift_tactile \
  --input      keyboard \
  --speed      0.385 \
  --show-pointcloud