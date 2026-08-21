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
# ckpt=logs/dppo/dppo-pretrain/single_lift_mushroom_rigid/sma/apioc/checkpoint/state_2000.pt
# normalization=dataset/dppo/single_lift_mushroom_rigid/sma/normalization.npz
#
# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --ckpt ${ckpt} \
#   --ft-denoising-steps 0 \
#   --normalization ${normalization} \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_outlier.yaml \
#   --action-config gentle_manip/configs/action/delta_pose_delta_gripper_fast_rot.yaml \
#   --pose-scale 0.999 \
#   --record dataset/real_deploy/rigid_sma_apioc2000 \
#   --shard-size 10

# ── BC pretrain on SIM RIGID demos, ABSOLUTE-POSE action (single_lift_mushroom_rigid, cho) ────
# action config = abs_pose_abs_gripper.yaml (10-dim: pos3 + 6D-rotation6 + gripper1; MUST match
# CMA-ES collection + eval_diffusion_pointnet.yaml under single_lift_mushroom_rigid_abs_pcd —
# see that cfg's action_dim=10 / obs_dim=8). ft-denoising-steps 0 = pure BC checkpoint.
# obs-config = point_cloud_1cam_outlier.yaml, same as the "sma" delta-mode entry above — the
# real-only outlier denoise, no object_focus (matches superset_rigid's crop/1024, arm kept in).


# ckpt=logs/dppo/dppo-pretrain/single_lift_mushroom_rigid/rpk/fjyis/checkpoint/state_3500.pt
# normalization=dataset/dppo/single_lift_mushroom_rigid/rpk/normalization.npz

# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --ckpt ${ckpt} \
#   --ft-denoising-steps 0 \
#   --normalization ${normalization} \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_outlier.yaml \
#   --action-config gentle_manip/configs/action/abs_pose_abs_gripper.yaml \
#   --smooth-alpha 0.1 \
#   --max-pos-step-m 0.01 \
#   --record dataset/real_deploy/tmp \
#   --shard-size 10 \
#   --max-steps 5000

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

# Working very nicely:
# ckpt=logs/dppo/dppo-pretrain/single_lift_mushroom_rigid/cho/ahaxs/checkpoint/state_800.pt
# normalization=dataset/dppo/single_lift_mushroom_rigid/cho/normalization.npz
# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --ckpt ${ckpt} \
#   --ft-denoising-steps 0 \
#   --normalization ${normalization} \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_outlier.yaml \
#   --action-config gentle_manip/configs/action/abs_pose_abs_gripper.yaml \
#   --smooth-alpha 0.6 \
#   --max-pos-step-m 0.0065 \
#   --record dataset/real_deploy/tmp \
#   --shard-size 10 \
#   --max-steps 5000

# ckpt=logs/dppo/dppo-pretrain/single_lift_mushroom_soft_abs_pcd_rot6d/bwvei/checkpoint/state_400.pt
# normalization=dataset/dppo/single_lift_mushroom_soft_abs_pcd_rot6d/normalization.npz
# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --normalization ${normalization} \
#   --ckpt ${ckpt} \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_outlier_rot6d.yaml \
#   --action-config gentle_manip/configs/action/abs_pose_abs_gripper.yaml \
#   --ft-denoising-steps 0 \
#   --smooth-alpha 0.6 \
#   --max-pos-step-m 0.015 \
#   --record dataset/real_deploy/tmp \
#   --shard-size 10 \
#   --max-steps 5000

# ckpt=./logs/dppo/dppo-pretrain/single_lift_mushroom_soft_abs_pcd_rot6d_hwo/vpstw/checkpoint/state_600.pt
# normalization=./dataset/dppo/single_lift_mushroom_soft_abs_pcd_rot6d_hwo/normalization.npz
# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --normalization ${normalization} \
#   --ckpt ${ckpt} \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_outlier_rot6d.yaml \
#   --action-config gentle_manip/configs/action/abs_pose_abs_gripper.yaml \
#   --ft-denoising-steps 0 \
#   --smooth-alpha 0.6 \
#   --max-pos-step-m 0.015 \
#   --record dataset/real_deploy/tmp \
#   --shard-size 10 \
#   --max-steps 5000

# ckpt=./logs/dppo/dppo-pretrain/single_lift_mushroom_soft_abs_pcd_rot6d_v2/ndkwc/checkpoint/state_800.pt # or 200
# normalization=./dataset/dppo/single_lift_mushroom_soft_abs_pcd_rot6d_v2/normalization.npz
# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --normalization ${normalization} \
#   --ckpt ${ckpt} \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_outlier_rot6d.yaml \
#   --action-config gentle_manip/configs/action/abs_pose_abs_gripper.yaml \
#   --ft-denoising-steps 0 \
#   --smooth-alpha 0.6 \
#   --max-pos-step-m 0.0065 \
#   --record dataset/real_deploy/tmp \
#   --shard-size 10 \
#   --max-steps 5000

# ckpt=/home/kei/kei/gentle_manip/logs/dppo/dppo-pretrain/single_lift_mushroom_rigid/cho/lzhto/checkpoint/state_800.pt
# normalization=dataset/dppo/single_lift_mushroom_rigid/cho/normalization.npz
# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --normalization ${normalization} \
#   --ckpt ${ckpt} \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_outlier.yaml \
#   --action-config gentle_manip/configs/action/abs_pose_abs_gripper.yaml \
#   --ft-denoising-steps 0 \
#   --smooth-alpha 0.6 \
#   --max-pos-step-m 0.0065 \
#   --record dataset/real_deploy/tmp \
#   --shard-size 10 \
#   --max-steps 5000

# ckpt=/home/kei/kei/gentle_manip/logs/dppo/dppo-pretrain/single_lift_mushroom_rigid/cho/kydpe/checkpoint/state_800.pt
# normalization=dataset/dppo/single_lift_mushroom_rigid/cho/normalization.npz
# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --normalization ${normalization} \
#   --ckpt ${ckpt} \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_outlier.yaml \
#   --action-config gentle_manip/configs/action/abs_pose_abs_gripper.yaml \
#   --ft-denoising-steps 0 \
#   --smooth-alpha 0.6 \
#   --max-pos-step-m 0.0065 \
#   --record dataset/real_deploy/tmp \
#   --shard-size 10 \
#   --max-steps 5000

# ckpt=logs/dppo-pretrain/single_lift_mushroom_rigid/cak/lfjih/checkpoint/state_800.pt
# normalization=/home/kei/kei/gentle_manip/dataset/dppo/single_lift_mushroom_rigid/cak/normalization.npz
# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --normalization ${normalization} \
#   --ckpt ${ckpt} \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_outlier.yaml \
#   --action-config gentle_manip/configs/action/abs_pose_abs_gripper.yaml \
#   --ft-denoising-steps 0 \
#   --smooth-alpha 0.6 \
#   --max-pos-step-m 0.0065 \
#   --record dataset/real_deploy/tmp \
#   --shard-size 10 \
#   --max-steps 5000
  
# ckpt=/home/kei/kei/gentle_manip/logs/dppo/dppo-pretrain/single_lift_mushroom_rigid/vqgsn/checkpoint/state_400.pt
# normalization=/home/kei/kei/gentle_manip/dataset/dppo/single_lift_mushroom_rigid/26-08-19-ibx/normalization.npz
# # vqgsn trained with the ARM-FOCUS cloud (superset_rigid_armfocus) -> deploy with the matching
# # arm-focus obs (point_cloud_1cam_armfocus), NOT plain outlier, or the cloud distribution won't match.
# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --normalization ${normalization} \
#   --ckpt ${ckpt} \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_armfocus.yaml \
#   --action-config gentle_manip/configs/action/abs_pose_abs_gripper.yaml \
#   --ft-denoising-steps 0 \
#   --smooth-alpha 0.6 \
#   --max-pos-step-m 0.0065 \
#   --record dataset/real_deploy/tmp \
#   --shard-size 10 \
#   --max-steps 5000

# ckpt=/home/kei/kei/gentle_manip/downloaded_runs/geozl/checkpoint/state_200.pt
# # geozl trained on dataset single_lift_mushroom_soft_abs_pcd_hwo (per its EXPERIMENT.md) — use THAT
# # normalization, not hwooo (a different dataset with different obs/action min-max).
# normalization=/home/kei/kei/gentle_manip/dataset/dppo/single_lift_mushroom_soft_abs_pcd_hwo/normalization.npz
# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --normalization ${normalization} \
#   --ckpt ${ckpt} \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_outlier.yaml \
#   --action-config gentle_manip/configs/action/abs_pose_abs_gripper.yaml \
#   --ft-denoising-steps 0 \
#   --smooth-alpha 0.6 \
#   --max-pos-step-m 0.0065 \
#   --record dataset/real_deploy/tmp \
#   --shard-size 10 \
#   --max-steps 5000

# ckpt=/home/kei/kei/gentle_manip/downloaded_runs/jfhlu/checkpoint/state_400.pt
# normalization=/home/kei/kei/gentle_manip/dataset/dppo/single_lift_mushroom_soft_abs_pcd_hwo/normalization.npz
# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --normalization ${normalization} \
#   --ckpt ${ckpt} \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_outlier.yaml \
#   --action-config gentle_manip/configs/action/abs_pose_abs_gripper.yaml \
#   --ft-denoising-steps 0 \
#   --smooth-alpha 0.6 \
#   --max-pos-step-m 0.0065 \
#   --record dataset/real_deploy/tmp \
#   --shard-size 10 \
#   --max-steps 5000

ckpt=/home/kei/kei/gentle_manip/downloaded_runs/eibno/checkpoint/state_100.pt
normalization=/home/kei/kei/gentle_manip/downloaded_runs/eibno/normalization2.npz
uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
  --ckpt ${ckpt} \
  --normalization ${normalization} \
  --obs-config gentle_manip/configs/obs/point_cloud_1cam_outlier.yaml \
  --action-config gentle_manip/configs/action/abs_pose_euler_abs_gripper.yaml \
  --ft-denoising-steps 0 \
  --smooth-alpha 1.0 \
  --max-pos-step-m 0.015 \
  --record dataset/real_deploy/tmp \
  --shard-size 10 \
  --max-steps 5000
