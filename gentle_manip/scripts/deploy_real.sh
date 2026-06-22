#!/bin/bash

uv run --project envs/dp3 python gentle_manip/scripts/deploy_real.py \
  --ckpt third_party/DP3/3D-Diffusion-Policy/data/outputs/real_xarm7_red_cube-dp3-0112_seed0/checkpoints/latest.ckpt \
  --obs-config gentle_manip/configs/obs/point_cloud_1cam.yaml \
  --max-steps 5000 --rate 30
