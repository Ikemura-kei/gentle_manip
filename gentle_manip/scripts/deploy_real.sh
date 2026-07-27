#!/bin/bash

# uv run --project envs/dp3 python gentle_manip/scripts/deploy_real.py \
#   --ckpt third_party/DP3/3D-Diffusion-Policy/data/outputs/real_xarm7_red_cube-dp3-0112_seed0/checkpoints/latest.ckpt \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam.yaml \
#   --max-steps 5000 --rate 30

# ── BC pretrain on REAL demos (single_lift_mushroom_real) ─────────────────────────────────────
# action config = delta_pose_delta_gripper_fast_rot.yaml (MUST match demo collection config;
# rot scales 0.008/0.008/0.03 — very different from the standard 0.001/0.001/0.001).
# ft-denoising-steps 0 = pure BC (no PPO finetune noise annealing).
#
# ckpt=logs/dppo/dppo-pretrain/single_lift_mushroom_soft_pcd_wide1k_n150/gpieh/checkpoint/state_3000.pt
# normalization=dataset/dppo/single_lift_mushroom_soft_pcd_wide1k_n150/normalization.npz
#
# ckpt=logs/dppo/dppo-pretrain/single_lift_mushroom_real/mhaoi/checkpoint/state_1500.pt
# ckpt=logs/dppo/dppo-pretrain/single_lift_mushroom_real/mhaoi/checkpoint/state_2000.pt
# normalization=dataset/dppo/single_lift_mushroom_real/normalization.npz
#
# ckpt=logs/dppo/dppo-pretrain/single_lift_mushroom_real/goyip/checkpoint/state_8000.pt
# normalization=dataset/dppo/single_lift_mushroom_real/normalization.npz
#
# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --ckpt ${ckpt} \
#   --ft-denoising-steps 0 \
#   --normalization ${normalization} \
#   --action-config gentle_manip/configs/action/delta_pose_delta_gripper_fast_rot.yaml \
#   --pose-scale 0.999 \
#   --record dataset/real_deploy/tmp \
#   --shard-size 10

# ── BC pretrain on SIM RIGID demos (single_lift_mushroom_rigid, sma dataset) ─────────────────
# action config = delta_pose_delta_gripper_fast_rot.yaml (CMA-ES collection used fast_rot scales).
# ft-denoising-steps 0 = pure BC checkpoint (no PPO finetune).
# normalization from the sim rigid sma dataset.
# obs-config = point_cloud_1cam_outlier.yaml (matches superset_rigid training crop/1024/outlier).
#
ckpt=logs/dppo/dppo-pretrain/single_lift_mushroom_rigid/sma/apioc/checkpoint/state_2000.pt
normalization=dataset/dppo/single_lift_mushroom_rigid/sma/normalization.npz

uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
  --ckpt ${ckpt} \
  --ft-denoising-steps 0 \
  --normalization ${normalization} \
  --obs-config gentle_manip/configs/obs/point_cloud_1cam_outlier.yaml \
  --action-config gentle_manip/configs/action/delta_pose_delta_gripper_fast_rot.yaml \
  --pose-scale 0.999 \
  --record dataset/real_deploy/rigid_sma_apioc2000 \
  --shard-size 10

# ── DPPO finetune (sim-trained BC + PPO finetune, single_lift_mushroom_soft_pcd) ─────────────
# action config = delta_pose_delta_gripper.yaml (standard; sim demos used standard scales).
# normalization from the SIM dataset (single_lift_mushroom_soft_pcd_wide1k_n150), not real.
# ft-denoising-steps 10 = finetuned checkpoint (enables the shortened DDPM chain).
# ckpt=logs/dppo/dppo-finetune/single_lift_mushroom_soft_pcd/luqsl/checkpoint/state_249.pt
# normalization=dataset/dppo/single_lift_mushroom_soft_pcd_wide1k_n150/normalization.npz

# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --ckpt ${ckpt} \
#   --ft-denoising-steps 10 \
#   --normalization ${normalization} \
#   --action-config gentle_manip/configs/action/delta_pose_delta_gripper.yaml \
#   --pose-scale 0.999 \
#   --record dataset/real_deploy/luqsl249 \
#   --shard-size 10