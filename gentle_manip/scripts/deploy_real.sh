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

# ckpt=/home/kei/kei/gentle_manip/downloaded_runs/eibno/checkpoint/state_100.pt
# normalization=/home/kei/kei/gentle_manip/downloaded_runs/eibno/normalization2.npz
# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --ckpt ${ckpt} \
#   --normalization ${normalization} \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_outlier.yaml \
#   --action-config gentle_manip/configs/action/abs_pose_euler_abs_gripper.yaml \
#   --ft-denoising-steps 0 \
#   --smooth-alpha 1.0 \
#   --max-pos-step-m 0.015 \
#   --record dataset/real_deploy/tmp \
#   --shard-size 10 \
#   --max-steps 5000

# ── CLUSTER SHORTLIST: afucm/state_400 — 7d EULER abs, arm-focus cloud, realws + 8% real co-train ──
# The action-space-ablation deployment candidate (docs/action_space_ablation_final_report.md:
# realws sim success 0.685 @ state_400; curve 0.58/0.63/0.66/0.685/0.635/0.47 — 400 is the peak).
# Trained: quat proprio (obs_dim 8) + ARM-FOCUS cloud + euler-7d COMMANDED actions (seam-fixed via
# euler_frame_offset [180,0,0]) on the realws workspace box, co-trained with the 50 real demos
# (plain concat, no oversampling — the winning co-train variant).
#
# Wiring notes (all verified 2026-08-23):
#   * obs-config point_cloud_1cam_armfocus.yaml — REQUIRED, and fully consistent: BOTH the sim
#     demos AND the real co-train slice (single_lift_mushroom_real_merged: 55 eps = the 51-ep
#     26-08-20-cmh session + a 4-ep top-up, all recorded THROUGH this very config; uniform cloud
#     fingerprint verified across all 55) are arm-focus clouds. (An earlier caveat here claimed the real slice
#     was unfocused — that looked at the obsolete July recordings, not cmh. Retracted.)
#   * action-config abs_pose_euler_abs_gripper.yaml — carries the euler frame offset AND
#     rate_limit: the RealBackend clamps every executed step to the delta-fast_rot bounds
#     (rotation x1.5), so a policy-emitted pose jump executes as a bounded walk, never a
#     full-speed servo jump. (The training snapshot predates rate_limit; the clamp does not
#     change the action decode, only bounds per-step motion.)
#   * net arch (visual 512, mlp [1024]^3) auto-loads from downloaded_runs/afucm/.hydra.
#   * checkout needs 76f5efa (euler offset) + 9938b40 (7d warmup/smoothing) — both in master.
#   * REAL TABLE PLACEMENT: object inside x [0.29, 0.48], y [-0.11, 0.11] (robot-base frame)
#     — the realws box this policy trained on.
#
# ckpt=downloaded_runs/afucm/checkpoint/state_400.pt
# normalization=downloaded_runs/afucm/normalization.npz

# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --ckpt ${ckpt} \
#   --ft-denoising-steps 0 \
#   --normalization ${normalization} \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_armfocus.yaml \
#   --action-config gentle_manip/configs/action/abs_pose_euler_abs_gripper.yaml \
#   --smooth-alpha 0.6 \
#   --max-pos-step-m 0.0065 \
#   --record dataset/real_deploy/afucm400 \
#   --shard-size 10 \
#   --max-steps 5000

# ── CLUSTER SHORTLIST: nmbtz/state_500 — PURE-SIM realws, 7d euler abs, arm-focus cloud ──────────
# The pure-sim deployment candidate (realws sim success 0.71 @ state_500 — the campaign's best
# no-real-data policy; curve 0.47/0.655/0.69/0.665/0.71/0.675). Identical stack to the afucm entry
# above MINUS the real co-training: quat proprio (obs_dim 8) + ARM-FOCUS cloud + euler-7d COMMANDED
# actions on the realws box. Deploying BOTH this and afucm answers "does real co-training help in
# real?" — the devlog's open question — on the same rig, same day.
# Wiring identical to afucm (armfocus obs REQUIRED; rate-limit clamp active; big net auto-loads
# from downloaded_runs/nmbtz/.hydra). REAL TABLE PLACEMENT: x [0.29, 0.48], y [-0.11, 0.11].
#
# ckpt=downloaded_runs/nmbtz/checkpoint/state_500.pt
# normalization=downloaded_runs/nmbtz/normalization.npz

# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --ckpt ${ckpt} \
#   --ft-denoising-steps 0 \
#   --normalization ${normalization} \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_armfocus.yaml \
#   --action-config gentle_manip/configs/action/abs_pose_euler_abs_gripper.yaml \
#   --smooth-alpha 0.6 \
#   --max-pos-step-m 0.0065 \
#   --record dataset/real_deploy/nmbtz500 \
#   --shard-size 10 \
#   --max-steps 5000

# ── CLUSTER SHORTLIST: qjzsf/state_1000 — REAL-ONLY (55 demos), 7d euler abs (commanded + K4) ────
# The real-data-only candidate: DPPO trained purely on the 55 real teleop demos, actions DERIVED
# as euler-7d absolute from the delta recordings (commanded accumulation + K=4 lookahead — teleop
# moves ±2.6mm/step, so K4 restores a stall-safe lead; see DEVLOG conclusion 5). Val-loss minimum
# spans state_500-1000; state_1000 is the downloaded one (realws-box real→sim eval: 0.60 @1000,
# 0.58 @500). No sim2real gap by construction — its weakness is 55 demos of coverage, not transfer.
#
# Wiring notes:
#   * obs-config point_cloud_1cam_armfocus.yaml — the training clouds are ARM-FOCUS: the real
#     demos (single_lift_mushroom_real_merged, 55 eps incl. 26-08-20-cmh) were RECORDED through
#     point_cloud_1cam_armfocus (baked into the pkl;
#     conversion reads stored clouds), regardless of the superset_real yaml in the run's config
#     snapshot (that describes the EXPERIMENT env, not the pkl's record-time processing).
#     Deploying with plain outlier would mismatch what the policy trained on.
#   * action-config abs_pose_euler_abs_gripper.yaml — euler offset + rate-limit clamp active.
#   * big net auto-loads from downloaded_runs/qjzsf/.hydra; load-smoked.
#   * Object placement: the real demos' own workspace (the realws box is a safe subset).
#
ckpt=downloaded_runs/qjzsf/checkpoint/state_1000.pt
normalization=downloaded_runs/qjzsf/normalization.npz

uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
  --ckpt ${ckpt} \
  --ft-denoising-steps 0 \
  --normalization ${normalization} \
  --obs-config gentle_manip/configs/obs/point_cloud_1cam_armfocus.yaml \
  --action-config gentle_manip/configs/action/abs_pose_euler_abs_gripper.yaml \
  --smooth-alpha 0.6 \
  --max-pos-step-m 0.0065 \
  --record dataset/real_deploy/qjzsf1000 \
  --shard-size 10 \
  --max-steps 5000
