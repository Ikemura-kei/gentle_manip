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
#   --action-config gentle_manip/configs/action/delta_pose_delta_gripper_fast_rot.yaml \
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
# ★ REAL RESULT (2026-08-23): ~75% success — BEST of the shortlist. Co-training wins in real.
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

# ── CLUSTER: alzey/state_200 — ITEM 16: paired-feature encoder regularization (w=0.5) on the
# afucm setup. Same data as afucm (realws sim + 55 real noos, 7d euler commanded, armfocus cloud,
# big net 512+[1024]^3) PLUS a cosine consistency loss pulling PointNet features of the 4,148
# paired real/sim cube3 steps together (dataset/dppo/paired_cube3_clouds.npz). Hypothesis: encoder
# domain alignment improves real transfer beyond raw co-training. Direct A/B vs afucm/state_400.
#
# Wiring identical to afucm (all reverified):
#   * obs point_cloud_1cam_armfocus.yaml; action abs_pose_euler_abs_gripper (7d euler commanded);
#     rate-limit clamp active at RealBackend.
#   * net arch auto-loads from downloaded_runs/alzey/.hydra (visual 512, mlp [1024]^3). The
#     paired-reg model class (cluster commit 20b0082, not yet on master) is TRAINING-only — the
#     EMA checkpoint is a plain PointNetDiffusionMLP state_dict (verified: no extra keys), so the
#     standard deploy loader works.
#   * normalization MUST be alzey's own (same data as afucm but stats are the dataset's).
#   * REAL TABLE PLACEMENT: object inside x [0.29, 0.48], y [-0.11, 0.11] (robot-base frame).
#
# ── CLUSTER: wclac/state_300 — ITEM 3: real-data-amount ablation, N=5 real demos (afucm recipe:
# realws sim 585 eps + FIRST 5 real demos, plain concat, union norm; big net; standard model).
# Curve context: nmbtz N=0 (sim 0.71, real WORST) · wclac N=5 · afucm N=50 (real ~75%, BEST).
# The real-robot ranking over N is the deliverable — sim success is expected flat (~0.65-0.71).
#
# Wiring identical to afucm (armfocus obs; 7d euler commanded; net arch auto-loads from
# downloaded_runs/wclac/.hydra; rate-limit clamp at RealBackend). normalization MUST be wclac's
# own (its union stats cover only 5 real demos — do NOT reuse afucm's).
# REAL TABLE PLACEMENT: object inside x [0.29, 0.48], y [-0.11, 0.11] (robot-base frame).
#
# ckpt=downloaded_runs/wclac/checkpoint/state_300.pt
# normalization=downloaded_runs/wclac/normalization.npz

# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --ckpt ${ckpt} \
#   --ft-denoising-steps 0 \
#   --normalization ${normalization} \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_armfocus.yaml \
#   --action-config gentle_manip/configs/action/abs_pose_euler_abs_gripper.yaml \
#   --smooth-alpha 0.6 \
#   --max-pos-step-m 0.0065 \
#   --record dataset/real_deploy/wclac300 \
#   --shard-size 10 \
#   --max-steps 5000

# ── CLUSTER: luewz/state_500 — ITEM 3: real-data-amount ablation, N=10 real demos (afucm recipe:
# realws sim 585 eps + FIRST 10 real demos, plain concat, union norm; big net; standard model).
# Curve: nmbtz N=0 (real worst) · wclac N=5 · luewz N=10 · afucm N=50 (real ~75%, best).
# Wiring identical to wclac/afucm; normalization MUST be luewz's own (N=10 union stats).
# REAL TABLE PLACEMENT: object inside x [0.29, 0.48], y [-0.11, 0.11] (robot-base frame).
#
# ckpt=downloaded_runs/luewz/checkpoint/state_500.pt
# normalization=downloaded_runs/luewz/normalization.npz

# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --ckpt ${ckpt} \
#   --ft-denoising-steps 0 \
#   --normalization ${normalization} \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_armfocus.yaml \
#   --action-config gentle_manip/configs/action/abs_pose_euler_abs_gripper.yaml \
#   --smooth-alpha 0.6 \
#   --max-pos-step-m 0.0065 \
#   --record dataset/real_deploy/luewz500 \
#   --shard-size 10 \
#   --max-steps 5000

# ckpt=downloaded_runs/alzey/checkpoint/state_200.pt
# normalization=downloaded_runs/alzey/normalization.npz

# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --ckpt ${ckpt} \
#   --ft-denoising-steps 0 \
#   --normalization ${normalization} \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_armfocus.yaml \
#   --action-config gentle_manip/configs/action/abs_pose_euler_abs_gripper.yaml \
#   --smooth-alpha 0.6 \
#   --max-pos-step-m 0.0065 \
#   --record dataset/real_deploy/alzey200 \
#   --shard-size 10 \
#   --max-steps 5000

# ── CLUSTER SHORTLIST: nmbtz/state_500 — PURE-SIM realws, 7d euler abs, arm-focus cloud ──────────
# REAL RESULT (2026-08-23): WORST of the shortlist despite the best sim score (0.71) — pure sim
# still carries a sim2real gap; sim rankings do not transfer across data regimes.
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
# REAL RESULT (2026-08-23): second — behind afucm (co-train), ahead of nmbtz (pure sim).
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
# ckpt=downloaded_runs/qjzsf/checkpoint/state_1000.pt
# normalization=downloaded_runs/qjzsf/normalization.npz

# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --ckpt ${ckpt} \
#   --ft-denoising-steps 0 \
#   --normalization ${normalization} \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_armfocus.yaml \
#   --action-config gentle_manip/configs/action/abs_pose_euler_abs_gripper.yaml \
#   --smooth-alpha 0.6 \
#   --max-pos-step-m 0.0065 \
#   --record dataset/real_deploy/qjzsf1000 \
#   --shard-size 10 \
#   --max-steps 5000

# ── CLUSTER: lulkx/state_600 — v33b_shift9 + PAIRED-REG (item 16, w=0.5, cube3 pairs), seed 43.
# The first entry on the FIXED dataset: v33b re-converted the real slice properly and shift9 uses
# the bias-corrected real clouds. Health-checked before recommending (do this for every co-train):
#   * normalization is CLEAN — action z max 0.072 -> 0.269 m and gripper max 80.0 mm, i.e. the
#     demos' real values. (The poisoned v33 read 0.75 -> 0.438 m and 44 mm; that is the tell.)
#   * pre-deploy probe PASSES on all four obs combinations (sim/real proprio x sim/real cloud):
#     descends and holds 80 mm open at t0. orkam/kjljs FAILED this same probe.
#       uv run --project envs/dppo python examples/sim2real_diagnose/probe_policy_real_obs.py \
#         --ckpt downloaded_runs/lulkx/checkpoint/state_600.pt \
#         --normalization downloaded_runs/lulkx/normalization.npz \
#         --real dataset/demos/single_lift_mushroom_real_merged_shift9mm \
#         --sim dataset/demos/single_lift_mushroom_soft/26-08-25-zrg --sim-episode 3
#   * checkpoint carries no paired-reg extras (28 keys, plain PointNetDiffusionMLP) — the
#     regularizer is training-only, so the standard deploy loader is correct. Big net (512 +
#     [1024]^3) auto-loads from downloaded_runs/lulkx/.hydra.
#
# ⚠ DEPLOY PAIRING — MANDATORY: this policy trained on SHIFT-CORRECTED real clouds, so
# real_lab.yaml MUST have `point_cloud_shift: [0.009, 0.0, 0.0]` ACTIVE (it currently is).
# Running it with the shift at zero silently reintroduces the full ~9 mm bias — no error.
# NOTE this also makes it NOT directly comparable to the afucm ~75% baseline, which was measured
# with the shift OFF on uncorrected data; both pairings are self-consistent, the data differs.
# REAL TABLE PLACEMENT: object inside x [0.29, 0.48], y [-0.11, 0.11] (robot-base frame).
#
# ckpt=downloaded_runs/lulkx/checkpoint/state_600.pt
# normalization=downloaded_runs/lulkx/normalization.npz
# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --ckpt ${ckpt} --ft-denoising-steps 0 --normalization ${normalization} \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_armfocus.yaml \
#   --action-config gentle_manip/configs/action/abs_pose_euler_abs_gripper.yaml \
#   --smooth-alpha 0.6 --max-pos-step-m 0.0065 \
#   --record dataset/real_deploy/lulkx600 --shard-size 10 --max-steps 5000

# ── CLUSTER: njhbz/state_300 — the PLAIN v33b_shift9 ANCHOR (no paired-reg), seed 42.
# The campaign's best sim policy (0.805/0.820 @300, sustained 28.1) and the A/B CONTROL for the
# paired-reg family: njhbz(seed 42, plain) vs mqlxj(seed 42, paired-reg w=0.5) differ ONLY by
# the regularizer, so that pair isolates item 16's real-robot effect. Same dataset, same wiring,
# same MANDATORY shift-ACTIVE pairing as the lulkx block. Verified: normalization identical to
# lulkx's and clean; 28 keys; probe PASSES (cmd z 0.1976 +-1.5 mm descending, grip 79.9 mm).
#
# ckpt=downloaded_runs/njhbz/checkpoint/state_300.pt
# normalization=downloaded_runs/njhbz/normalization.npz

# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --ckpt ${ckpt} --ft-denoising-steps 0 --normalization ${normalization} \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_armfocus.yaml \
#   --action-config gentle_manip/configs/action/abs_pose_euler_abs_gripper.yaml \
#   --smooth-alpha 0.6 --max-pos-step-m 0.0065 \
#   --record dataset/real_deploy/njhbz300 --shard-size 10 --max-steps 5000

# ── CLUSTER: mqlxj/state_400 — paired-reg seed 42, completing the family (42 mqlxj / 27 avfnp /
# 43 lulkx). Matched control for njhbz above (same seed, regularizer is the only difference).
# Verified: normalization identical to lulkx's and clean; 28 keys, no paired-reg extras; probe
# PASSES (cmd z 0.1983 +-0.8 mm descending — the tightest spread of the family, grip 80.0 mm).
#
# ckpt=downloaded_runs/mqlxj/checkpoint/state_400.pt
# normalization=downloaded_runs/mqlxj/normalization.npz

# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --ckpt ${ckpt} --ft-denoising-steps 0 --normalization ${normalization} \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_armfocus.yaml \
#   --action-config gentle_manip/configs/action/abs_pose_euler_abs_gripper.yaml \
#   --smooth-alpha 0.6 --max-pos-step-m 0.0065 \
#   --record dataset/real_deploy/mqlxj400 --shard-size 10 --max-steps 5000

# ── CLUSTER: avfnp/state_400 — SEED SIBLING of lulkx (same v33b_shift9 dataset, same paired-reg
# w=0.5, seed 27 vs lulkx's 43; different best checkpoint, 400 vs 600). Same wiring, same
# MANDATORY shift-ACTIVE pairing, same table placement as the lulkx block above.
# Verified identically: normalization is byte-identical to lulkx's (same dataset) and clean
# (z max 0.269 m, gripper 80.0 mm); no paired-reg extras in the checkpoint (28 keys); pre-deploy
# probe PASSES all four obs combinations.
# Because the two differ ONLY by seed, running both measures seed variance on the real robot —
# useful context for reading any single real number, since our real trial counts are ~30.
#
# ckpt=downloaded_runs/avfnp/checkpoint/state_400.pt
# normalization=downloaded_runs/avfnp/normalization.npz

# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --ckpt ${ckpt} --ft-denoising-steps 0 --normalization ${normalization} \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_armfocus.yaml \
#   --action-config gentle_manip/configs/action/abs_pose_euler_abs_gripper.yaml \
#   --smooth-alpha 0.6 --max-pos-step-m 0.0065 \
#   --record dataset/real_deploy/avfnp400 --shard-size 10 --max-steps 5000

# ══════════════════════════════════════════════════════════════════════════════════════════
# ⛔ DO NOT DEPLOY any v33 policy (orkam, kjljs, …) UNTIL THE REAL SLICE IS RE-CONVERTED.
# Root cause (2026-08-25, diagnosed from dataset/real_deploy/orkam200 + offline probes):
# the real 7d slice merged into `single_lift_mushroom_simreal_realws_noos_cmd_v33`
# (dataset/dppo/single_lift_mushroom_real) was NOT derived — the recorded DELTA actions were
# written through as if they were ABSOLUTE. A delta of ~0 decodes to the MIDDLE of each
# absolute range, so that slice teaches, on every real-looking cloud:
#     dz  ~ 0  ->  absolute z    = 0.252 m   (median commanded z; achieved is 0.096 m)
#     dgw ~ 0  ->  absolute grip = 44 mm     (median+max commanded; demos actually hold 80 mm)
# Deployed orkam did exactly that: climbed toward z 0.25 with the gripper at 44 mm.
# Confirmed: v33's merged normalization (downloaded_runs/orkam/normalization.npz) carries the
# broken slice's ranges (action z max 0.75 -> 0.438 m; no sim collection exceeds 0.235 m).
# afucm is UNAFFECTED (its merged z max 0.072 -> 0.239 m) — its real slice was derived properly.
# Offline probe reproduces it with no robot: real cloud -> 44 mm + z up; sim cloud -> correct;
# proprio irrelevant; fails even on the real demos in its OWN training set. ckpt 200 and 400 alike.
# FIX: re-convert the real demos WITH derivation, then re-merge + retrain:
#   convert_demos dataset/demos/single_lift_mushroom_real_merged --out <real_7d> \
#     --obs-keys ee_pos ee_quat gripper_width --point-cloud \
#     --derive-action  gentle_manip/configs/action/abs_pose_euler_abs_gripper.yaml \
#     --derive-source-action gentle_manip/configs/action/delta_pose_delta_gripper_fast_rot.yaml \
#     --derive-lookahead 4
# GATE before any co-train deploy: derived commanded gripper must match the demos' achieved
# gripper (~80 mm open at t0), and commanded z must track achieved z within ~1 cm — not 15 cm.
# ══════════════════════════════════════════════════════════════════════════════════════════

# ── CLUSTER: kjljs/state_100 — v3.3 co-train + AUX GRASP-WIDTH head (aux_grasp_width_weight 1.5,
# AuxDiffusionModel). Sibling of orkam on the SAME v33 dataset, so it inherits the broken real
# slice above — offline probe: real cloud -> grip 44-55 mm + z 0.23-0.24, sim cloud -> correct.
# ⛔ blocked on the re-convert; entry kept ready for the rerun.
# Wiring notes: the checkpoint carries 4 EXTRA `network.width_head.*` keys (the aux head lives
# inside the network). They load without error and are unused at deploy — the action path is the
# standard PointNetDiffusionMLP — so no adapter change is needed. Big net auto-loads from .hydra;
# normalization MUST be kjljs's own. REAL TABLE PLACEMENT: x [0.29, 0.48], y [-0.11, 0.11].
#
# ckpt=downloaded_runs/kjljs/checkpoint/state_100.pt
# normalization=downloaded_runs/kjljs/normalization.npz

# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --ckpt ${ckpt} \
#   --ft-denoising-steps 0 \
#   --normalization ${normalization} \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_armfocus.yaml \
#   --action-config gentle_manip/configs/action/abs_pose_euler_abs_gripper.yaml \
#   --smooth-alpha 0.6 \
#   --max-pos-step-m 0.0065 \
#   --record dataset/real_deploy/kjljs100 \
#   --shard-size 10 \
#   --max-steps 5000

# ── CLUSTER: orkam/state_200 — v3.3 RECIPE co-train (item 2+5+6+18 combined; sim 0.715 > afucm 0.685).
# ⛔ SEE THE BLOCK ABOVE — broken real slice; deployed run climbed to z 0.25 with grip 44 mm.
# Data: 599 v3.3 sim eps (4-mushroom pool, scale [0.8,1.5], real-matched kinematics, pinch-filtered)
# + the same 55 real demos (plain concat, union norm). Wiring identical to afucm: armfocus obs,
# 7d euler commanded actions, rate-limit clamp active, big net auto-loads from .hydra.
# Pull first:  bash gentle_manip/scripts/pull_run.sh logs/dppo/dppo-pretrain/single_lift_mushroom_simreal_realws_noos_cmd_v33/orkam downloaded_runs/orkam
# normalization MUST be orkam's own (v33 union stats — new collection, NOT afucm's).
# REAL TABLE PLACEMENT: x [0.29, 0.48], y [-0.11, 0.11] (robot-base frame).
#
# ckpt=downloaded_runs/orkam/checkpoint/state_400.pt
# normalization=downloaded_runs/orkam/normalization.npz
# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --ckpt ${ckpt} --ft-denoising-steps 0 --normalization ${normalization} \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_armfocus.yaml \
#   --action-config gentle_manip/configs/action/abs_pose_euler_abs_gripper.yaml \
#   --smooth-alpha 0.6 --max-pos-step-m 0.0065 \
#   --record dataset/real_deploy/orkam400 --shard-size 10 --max-steps 5000
# ─────────────────────────────────────────────────────────────────────────────
# GENERALIST 12-object policy (cluster run cvzth, ckpt 80) — added 2026-09-01.
# Trained on the 12-object v4.1 dataset (cluster: "12-object generalist" commit aafb638);
# action = 7d euler absolute (confirmed vs downloaded_runs/cvzth/config snapshot), obs =
# superset -> point_cloud_1cam_armfocus deploy view. NOTE vs the cluster's suggested
# command: envs/dppo_deploy (NOT envs/dp3 — that is the cluster's env name; on this box
# all DPPO deploys run in dppo_deploy) and the standard safety knobs added
# (--smooth-alpha / --max-pos-step-m, conservative motion).
#
# ckpt=downloaded_runs/cvzth/checkpoint/state_80.pt
# normalization=downloaded_runs/cvzth/normalization.npz
# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --ckpt ${ckpt} --ft-denoising-steps 0 --normalization ${normalization} \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_armfocus.yaml \
#   --action-config gentle_manip/configs/action/abs_pose_euler_abs_gripper.yaml \
#   --smooth-alpha 0.6 --max-pos-step-m 0.0065 \
#   --record dataset/real_deploy/cvzth80_generalist --shard-size 10 \
#   --max-steps 5000

# GENERALIST 12-object + ALL 7 REAL objects (cluster run zdwii, ckpt 91) — added 2026-09-02.
# THE LATEST GENERALIST TRAINED WITH REAL-WORLD DATA. Dataset
# single_lift_generalist_12obj_real7 = the 12 v4.1 sim collections + all 141 real teleop episodes
# from dataset/transfer/real_paired_7obj_2026-09-01 (7 objects; grape and padron_pepper have no
# sim counterpart, included deliberately). Normalization recomputed AFTER merging, so it is NOT
# interchangeable with the sim-only cvzth stats above — using the wrong one decodes wrong commands.
#
# Same arch/action/obs as cvzth: [3072]x3, 7d euler absolute, armfocus point cloud.
# ⚠ NOT the objraw/objemb variants: those need a first-frame object crop that deploy_real_dppo.py
# does not build. This entry is the plain (base) generalist.
#
# SEED CHOICE (mushroom, 200 eps / 40 geometries, state_91):
#   zdwii (321)  66.5% success  13.0% damage   <- best gentleness at near-top success: USE THIS
#   rtgob (42)   68.0% success  16.0% damage      +1.5 pt success for +3 pt damage
#   asavh (27)   59.0% success  14.0% damage
# Differences are within seed noise (success SD 4.8, damage SD 1.5) -- zdwii is the better BET,
# not a demonstrated optimum.
#
# ⚠ vs the SIM-ONLY generalist (cvzth): adding real data COST ~8 pts of mushroom success in sim
# (64.5% vs 72.8% pooled) at unchanged damage. Real-robot transfer is what the real data is FOR
# and a sim eval cannot measure it -- that is exactly what this deployment tests.
#
#   ./gentle_manip/scripts/pull_run.sh zdwii --ckpt 91
#
# ckpt=downloaded_runs/zdwii/checkpoint/state_91.pt
# normalization=downloaded_runs/zdwii/normalization.npz
# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --ckpt ${ckpt} --ft-denoising-steps 0 --normalization ${normalization} \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_armfocus.yaml \
#   --action-config gentle_manip/configs/action/abs_pose_euler_abs_gripper.yaml \
#   --smooth-alpha 0.6 --max-pos-step-m 0.0065 \
#   --record dataset/real_deploy/zdwii91_generalist_real7 --shard-size 10 \
#   --max-steps 5000
# Keep point_cloud_shift [0.009,0,0] ACTIVE in real_lab.yaml (sim clouds are unbiased; the real
# cloud must be corrected into the frame the policy trained in).

# ── PRE-DEPLOY CHECK: live view of the EXACT point cloud the policy sees ─────
# (post-perception: crop + outlier removal + ARM-FOCUS + FPS subsample, driven
# through PerceptionPipeline with a live robot connection — the armfocus filter
# needs real ee_pos, so this runs record.py in view-only mode. Robot homes on
# start; W/S etc. move it if pressed; just watch the cloud and ESC to quit,
# save nothing.)
#
# NOTE --action-config is REQUIRED: record.py's DEFAULT is now the 10-dim ABSOLUTE
# config (absolute-demo collection era), but teleop emits 7-dim deltas ->
# "IndexError: index 9 out of bounds" in _process_absolute without it (hit 2026-09-01).
# uv run --project envs/deploy python -m gentle_manip.demos.record \
#   --setup gentle_manip/configs/setup/real_lab.yaml \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_armfocus.yaml \
#   --action-config gentle_manip/configs/action/delta_pose_delta_gripper_fast_rot.yaml \
#   --task-name pcd_preview --input keyboard --show-pointcloud
#
# Camera-only quick look (NO robot, so the armfocus filter is SKIPPED — crop +
# outlier + subsample only; raw gray vs processed orange overlay + crop box):
#
# uv run --project envs/deploy python -m gentle_manip.visualization.point_cloud_viewer \
#   --setup gentle_manip/configs/setup/real_lab.yaml \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_armfocus.yaml \
#   --show-crop --show-processed

# ── REAL DEMO COLLECTION for the 12-obj generalist (cotrain) + pi0.5 VLA baseline ──
# 20 eps per object; ONE collection feeds BOTH consumers:
#   - point cloud view == the generalist's student/deploy view (same crop/armfocus/1024)
#   - obs["image_cam_ext"] = PAIRED RGB inside the obs (idle-trimmed with all channels;
#     the pi05 convert_to_lerobot.py path consumes this key). --record-rgb mp4 is
#     presentation-only and NOT trim-paired — don't rely on it for training data.
#   - SAVED action = 7d euler ABSOLUTE (matches cvzth training) via --record-action-config,
#     while teleop drives in smooth delta mode.
#   - input: SpaceMouse pose + Z/X keyboard gripper; SPACE save / BACKSPACE discard / ESC quit.
#   - one run per object; task name = single_lift_<object>_real (naming convention);
#     --description is stored in the run's config.yaml — put the object + intent there.
#
obj=tofu   # repeat per object: mushroom strawberry cherry_tomato raspberry tomato tofu ...
uv run --project envs/deploy python -m gentle_manip.demos.record \
  --setup gentle_manip/configs/setup/real_lab.yaml \
  --obs-config gentle_manip/configs/obs/point_cloud_1cam_armfocus_rgb.yaml \
  --action-config gentle_manip/configs/action/delta_pose_delta_gripper_fast_rot.yaml \
  --record-action-config gentle_manip/configs/action/abs_pose_euler_abs_gripper.yaml \
  --task-name single_lift_${obj}_real \
  --input spacemouse-kb \
  --description "${obj}: 20 real eps, generalist-cotrain + pi0.5 RGB baseline" \
  --show-pointcloud
