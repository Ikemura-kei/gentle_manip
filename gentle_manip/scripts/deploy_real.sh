#!/bin/bash

# uv run --project envs/dp3 python gentle_manip/scripts/deploy_real.py \
#   --ckpt third_party/DP3/3D-Diffusion-Policy/data/outputs/real_xarm7_red_cube-dp3-0112_seed0/checkpoints/latest.ckpt \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam.yaml \
#   --max-steps 5000 --rate 30

ckpt=logs/dppo/dppo-pretrain/single_lift_mushroom_soft_pcd_wide1k_n150/gpieh/checkpoint/state_3000.pt
normalization=dataset/dppo/single_lift_mushroom_soft_pcd_wide1k_n150/normalization.npz

uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
  --ckpt ${ckpt} \
  --ft-denoising-steps 0 \
  --normalization ${normalization} \
  --pose-scale 0.999 \
  --record dataset/real_deploy/gpieh3000 \
  --shard-size 10          # --record is now a DIR of shard_XXXX.pkl (10 trajectories each)