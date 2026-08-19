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
#
# --pose-scale is a DELTA-mode-only knob and is a no-op here (absolute targets, not deltas).
# --smooth-alpha is the absolute-mode equivalent for a jittery/shaky commanded pose (EMA
# low-pass on pos+rotation, gripper dim untouched so grasp/release stays decisive) — start
# around 0.3 and tune down further if motion still looks jerky, up if it feels sluggish/laggy
# behind the policy's intent. NOTE: 0.0 freezes the commanded pose at whatever it was seeded
# to (never tracks a new target) — that's likely not what you want; use a value in (0, 1].
# --max-pos-step-m is a HARD per-tick safety cap (meters/axis) on top of/independent from
# smooth_alpha — no single tick can move the target further than this, no matter what the
# network outputs; both filters seed from the robot's ACTUAL current pose at reset/re-home,
# so the very first commanded action is bounded too, not just steady-state jitter.
# Sim eval of this checkpoint (band 0.175-0.275): success 0.76, ever_success 0.815 (see
# logs/dppo/dppo-pretrain/single_lift_mushroom_rigid/cho/ahaxs/eval/).


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
#   --max-pos-step-m 0.015 \
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
#   --max-pos-step-m 0.015 \
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
#   --max-pos-step-m 0.015 \
#   --record dataset/real_deploy/tmp \
#   --shard-size 10 \
#   --max-steps 5000

ckpt=/home/kei/kei/gentle_manip/logs/dppo/dppo-pretrain/single_lift_mushroom_rigid/cho/nrwts/checkpoint/state_800.pt
normalization=dataset/dppo/single_lift_mushroom_rigid/cho/normalization.npz
uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
  --normalization ${normalization} \
  --ckpt ${ckpt} \
  --obs-config gentle_manip/configs/obs/point_cloud_1cam_outlier.yaml \
  --action-config gentle_manip/configs/action/abs_pose_abs_gripper.yaml \
  --ft-denoising-steps 0 \
  --smooth-alpha 0.6 \
  --max-pos-step-m 0.015 \
  --record dataset/real_deploy/tmp \
  --shard-size 10 \
  --max-steps 5000