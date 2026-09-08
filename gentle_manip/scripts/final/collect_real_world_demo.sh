#!/bin/bash

# ── REAL DEMO COLLECTION for the generalist v5 (cotrain) + pi0.5 VLA baseline ──
# 20 eps per object; ONE collection feeds BOTH consumers:
#   - point cloud view == the generalist's student/deploy view (same crop/armfocus/1024)
#   - obs["image_cam_ext"] = PAIRED RGB inside the obs (idle-trimmed with all channels;
#     the pi05 convert_to_lerobot.py path consumes this key). --record-rgb mp4 is
#     presentation-only and NOT trim-paired — don't rely on it for training data.
#   - SAVED action = 7d euler ABSOLUTE via --record-action-config, while teleop drives in smooth
#     delta mode. MUST be the _z15 yaml: the generalist v5 sim demos were converted with
#     abs_pose_euler_abs_gripper_z15 (pos_min z 0.015, pos_max x 0.55). The non-z15 file has
#     z 0.003 / x 0.59, so actions recorded with it normalize against different bounds and decode
#     15-25 mm off when merged with the sim set (this is the `covel` failure, 0/20).
#   - input: SpaceMouse pose + Z/X keyboard gripper; SPACE save / BACKSPACE discard / ESC quit.
#   - one run per object; task name = single_lift_<object>_real (naming convention);
#     --description is stored in the run's config.yaml — put the object + intent there.

obj=cube
uv run --project envs/deploy python -m gentle_manip.demos.record \
  --setup gentle_manip/configs/setup/real_lab.yaml \
  --obs-config gentle_manip/configs/obs/point_cloud_1cam_armfocus_rgb.yaml \
  --action-config gentle_manip/configs/action/delta_pose_delta_gripper_fast_rot.yaml \
  --record-action-config gentle_manip/configs/action/abs_pose_euler_abs_gripper_z15.yaml \
  --task-name single_lift_${obj}_real \
  --input spacemouse-kb \
  --description "${obj}: 20 real eps, generalist-v5 cotrain + pi0.5 RGB baseline (z15 actions)" \
  --show-pointcloud