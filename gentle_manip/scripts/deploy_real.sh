#!/bin/bash

# uv run --project envs/dp3 python gentle_manip/scripts/deploy_real.py \
#   --ckpt third_party/DP3/3D-Diffusion-Policy/data/outputs/real_xarm7_red_cube-dp3-0112_seed0/checkpoints/latest.ckpt \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam.yaml \
#   --max-steps 5000 --rate 30

uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
  --ckpt logs/dppo/dppo-pretrain/single_lift_mushroom_soft_pcd_wide/xxiaw/checkpoint/state_8000.pt \
  --ft-denoising-steps 0 \
  --normalization dataset/dppo/single_lift_mushroom_soft_pcd_wide/normalization.npz \
  --pose-scale 0.999 \
  --record dataset/real_deploy/xxiaw8000 \
  --shard-size 10          # --record is now a DIR of shard_XXXX.pkl (10 trajectories each)