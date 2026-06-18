#!/bin/bash

TASK=red_cube

uv run --project deploy python -m gentle_manip.demos.record \
  --obs-config gentle_manip/configs/obs/point_cloud_1cam.yaml \
  --task-name "$TASK" --input keyboard --rate 30 --speed 0.3