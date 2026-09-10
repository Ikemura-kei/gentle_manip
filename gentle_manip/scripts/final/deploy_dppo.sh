#! /bin/bash
# [final] Real-robot deploy of a DPPO BC policy (envs/dppo_deploy). One entry per policy, newest LAST and ACTIVE;
# older entries stay as comments (same convention as gentle_manip/scripts/deploy_real.sh).
#
# READING THE BLOCKS: each policy's description sits inside a +---+ box (lines starting `# |`). Those are
# DOCUMENTATION — leave them commented. Only the lines BELOW a box (ckpt= / normalization= / uv run ...)
# are the command: comment out the active one and uncomment the block you want.
# Rules (docs/training_plan_sim2real_2026-09.md §6): obs config = the real twin of the training obs
# (point_cloud_1cam_armfocus == superset_soft_armfocus_board: same crop z>=19 mm, 1024 pts, outlier + object focus;
#  plus the REAL-ONLY ground_residual filter, default ON since 2026-09-06),
# action config = the SAME yaml the demos were converted with (z15), normalization = the training dataset's.
# --ddim-steps N: DDIM sampling at inference (RGB entries only). DP3 trains 100 / infers 10, DPPO's image
# configs train 100 / infer 5; we ran full denoising, which costs 59 ms at 100 steps vs 8 ms at DDIM-10.
# --record-rgb: also saves cam_ext RGB per step to <record>/videos/ep_NNN.mp4 (presentation only, never policy input).
# Visualize a recording afterwards: bash gentle_manip/scripts/final/viz_deploy.sh <record dir>
# Before the first deploy of the day: uv run --project envs/deploy python -m gentle_manip.diagnostics.drift_check
set -euo pipefail
cd "$(dirname "$0")/../../.."


# +-----------------------------------------------------------------------------------------------------------
# | generalist v5 LOCAL, wiayg (2026-09-08), 1M steps. Sim teasers: tofu 14/20, mushroom 17/20, banana
# | 13/20, cherry 3/20.
# +-----------------------------------------------------------------------------------------------------------
# ckpt=logs/dppo/dppo-pretrain/single_lift_generalist_soft_v5/wiayg/checkpoint/state_120.pt
# normalization=dataset/dppo/single_lift_generalist_soft_v5/normalization.npz
# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --ckpt ${ckpt} --ft-denoising-steps 0 --normalization ${normalization} \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_armfocus.yaml \
#   --action-config gentle_manip/configs/action/abs_pose_euler_abs_gripper_z15.yaml \
#   --act-steps 4 --smooth-alpha 0.6 --max-pos-step-m 0.0065 \
#   --record dataset/real_deploy/generalist_v5_wiayg_120 --shard-size 10 --record-rgb \
#   --max-steps 5000 "$@"


# +-----------------------------------------------------------------------------------------------------------
# | G0 = fdcjk (cluster): recipe v5 with the PAIRED real-sim term ablated (w=1e-8, log-only). Not yet
# | evaluated.
# +-----------------------------------------------------------------------------------------------------------
# ckpt=downloaded_runs/fdcjk/checkpoint/state_120.pt
# normalization=downloaded_runs/fdcjk/normalization.npz
# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --ckpt ${ckpt} --ft-denoising-steps 0 --normalization ${normalization} \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_armfocus.yaml \
#   --action-config gentle_manip/configs/action/abs_pose_euler_abs_gripper_z15.yaml \
#   --act-steps 4 --smooth-alpha 0.6 --max-pos-step-m 0.0065 \
#   --record dataset/real_deploy/generalist_v5_fdcjk_G0_120 --shard-size 10 --record-rgb \
#   --max-steps 5000 "$@"


# +-----------------------------------------------------------------------------------------------------------
# | G2 = ttukt (cluster): recipe v5 with the ENCODER CONSISTENCY term ablated (w=1e-8, log-only). Not yet
# | evaluated.
# +-----------------------------------------------------------------------------------------------------------
# ckpt=downloaded_runs/ttukt/checkpoint/state_120.pt
# normalization=downloaded_runs/ttukt/normalization.npz
# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --ckpt ${ckpt} --ft-denoising-steps 0 --normalization ${normalization} \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_armfocus.yaml \
#   --action-config gentle_manip/configs/action/abs_pose_euler_abs_gripper_z15.yaml \
#   --act-steps 4 --smooth-alpha 0.6 --max-pos-step-m 0.0065 \
#   --record dataset/real_deploy/generalist_v5_ttukt_G2_120 --shard-size 10 --record-rgb \
#   --max-steps 5000 "$@"


# +-----------------------------------------------------------------------------------------------------------
# | G1 = bmbrv (cluster): recipe v5 EXACT, the cluster twin of wiayg. Same dataset + val split. Not yet
# | evaluated.
# +-----------------------------------------------------------------------------------------------------------
# ckpt=downloaded_runs/bmbrv/checkpoint/state_120.pt
# normalization=downloaded_runs/bmbrv/normalization.npz
# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --ckpt ${ckpt} --ft-denoising-steps 0 --normalization ${normalization} \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_armfocus.yaml \
#   --action-config gentle_manip/configs/action/abs_pose_euler_abs_gripper_z15.yaml \
#   --act-steps 4 --smooth-alpha 0.6 --max-pos-step-m 0.0065 \
#   --record dataset/real_deploy/generalist_v5_bmbrv_G1_120 --shard-size 10 --record-rgb \
#   --max-steps 5000 "$@"


# +-----------------------------------------------------------------------------------------------------------
# | REAL-only BC = jiupy (2026-09-08): trained on 110 real teleop eps (6 objects), no sim data, no aug.
# | state_500 = val minimum (0.0099; val rises after — see EXPERIMENT.md). NORMALIZATION IS THE REAL SET.
# +-----------------------------------------------------------------------------------------------------------
# ckpt=logs/dppo/dppo-pretrain/single_lift_real6_bc_v1/jiupy/checkpoint/state_1500.pt
# normalization=dataset/dppo/single_lift_real6_bc_v1/normalization.npz
# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --ckpt ${ckpt} --ft-denoising-steps 0 --normalization ${normalization} \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_armfocus.yaml \
#   --action-config gentle_manip/configs/action/abs_pose_euler_abs_gripper_z15.yaml \
#   --act-steps 4 --smooth-alpha 0.6 --max-pos-step-m 0.0065 \
#   --record dataset/real_deploy/real6_bc_jiupy_1500 --shard-size 10 --record-rgb \
#   --max-steps 5000 "$@"


# +-----------------------------------------------------------------------------------------------------------
# | REAL-only BC, RGB = tawuv (2026-09-08): same 110 real eps, ViT on cam_ext RGB instead of the cloud.
# | state_400 = 220 epochs past the val bottom (ep 180); val loss is not predictive here, so this tests it.
# +-----------------------------------------------------------------------------------------------------------
# ckpt=logs/dppo/dppo-pretrain/single_lift_real6_bc_rgb_v1/tawuv/checkpoint/state_200.pt
# normalization=dataset/dppo/single_lift_real6_bc_rgb_v1/normalization.npz
# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --ckpt ${ckpt} --ft-denoising-steps 0 --normalization ${normalization} \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_armfocus_rgb.yaml \
#   --action-config gentle_manip/configs/action/abs_pose_euler_abs_gripper_z15.yaml \
#   --act-steps 4 --ddim-steps 10 --smooth-alpha 0.6 --max-pos-step-m 0.0065 \
#   --record dataset/real_deploy/real6_bc_rgb_tawuv_200 --shard-size 10 --record-rgb \
#   --max-steps 5000 "$@"


# +-----------------------------------------------------------------------------------------------------------
# | REAL-only BC, RGB + AUGMENTATION = xkhrc (2026-09-08): RandomShiftsAug(pad=4) on the RGB input. Val
# | 0.0104 @ep820 vs 0.0176 without aug (cloud run: 0.0099). state_1200 = last (stopped there).
# +-----------------------------------------------------------------------------------------------------------
# ckpt=logs/dppo/dppo-pretrain/single_lift_real6_bc_rgb_v1/xkhrc/checkpoint/state_1200.pt
# normalization=dataset/dppo/single_lift_real6_bc_rgb_v1/normalization.npz
# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --ckpt ${ckpt} --ft-denoising-steps 0 --normalization ${normalization} \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_armfocus_rgb.yaml \
#   --action-config gentle_manip/configs/action/abs_pose_euler_abs_gripper_z15.yaml \
#   --act-steps 4 --ddim-steps 10 --smooth-alpha 0.6 --max-pos-step-m 0.0065 \
#   --record dataset/real_deploy/real6_bc_rgb_aug_xkhrc_1200 --shard-size 10 --record-rgb \
#   --max-steps 5000 "$@"


# +-----------------------------------------------------------------------------------------------------------
# | REAL-only BC, RGB + DPPO IMAGE RECIPE = fuuoy (2026-09-08): the aug run with DPPO's own image settings
# | (denoising 100, batch 256) instead of ours (20/128). Overfits less (1.5x at ep1600 vs 2.5x at ep1200)
# | and peaks later; its val is NOT comparable to the 20-step runs (different noise-level mixture).
# | state_1350 = val minimum (0.0160).
# +-----------------------------------------------------------------------------------------------------------
# ckpt=logs/dppo/dppo-pretrain/single_lift_real6_bc_rgb_v1/fuuoy/checkpoint/state_1000.pt
# normalization=dataset/dppo/single_lift_real6_bc_rgb_v1/normalization.npz
# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --ckpt ${ckpt} --ft-denoising-steps 0 --normalization ${normalization} \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_armfocus_rgb.yaml \
#   --action-config gentle_manip/configs/action/abs_pose_euler_abs_gripper_z15.yaml \
#   --act-steps 4 --ddim-steps 10 --smooth-alpha 0.6 --max-pos-step-m 0.0065 \
#   --record dataset/real_deploy/real6_bc_rgb_dppo_fuuoy_1000 --shard-size 10 --record-rgb \
#   --max-steps 5000 "$@"


# +-----------------------------------------------------------------------------------------------------------
# | GENERALIST G5 + ENSEMBLING = fsynt: G3 (horizon 16, exec 4, tail 22) + mean(+)max pooling. Best
# | cherry result, replicated 10 and 9 /20 (baseline 3) at the lowest stress; mushroom 18/20 hold 0.
# | Pooling and ensembling INTERACT -- alone 6/20 and 4.5/20. smooth-alpha damps orientation dither.
# +-----------------------------------------------------------------------------------------------------------
# ckpt=logs/dppo/dppo-pretrain/single_lift_generalist_soft_v5_tail22/fsynt/checkpoint/state_113.pt
# normalization=dataset/dppo/single_lift_generalist_soft_v5_tail22/normalization.npz
# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --ckpt ${ckpt} --ft-denoising-steps 0 --normalization ${normalization} \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_armfocus.yaml \
#   --action-config gentle_manip/configs/action/abs_pose_euler_abs_gripper_z15.yaml \
#   --act-steps 1 --temporal-ensemble --ensemble-m 0.01 --smooth-alpha 0.6 --max-pos-step-m 0.0065 \
#   --record dataset/real_deploy/generalist_g5ens_fsynt_113 --shard-size 10 --record-rgb \
#   --max-steps 5000 "$@"


# +-----------------------------------------------------------------------------------------------------------
# | REAL-only BC, G5 RECIPE = jfwqs: meanmax + horizon 16, otherwise identical to jiupy (same data,
# | 2000 epochs) so the two changes are attributable. First real policy that can ensemble at all.
# | state_400 = val min (ep 390); val loss has misled before, so sweep 800/1500/2000 if it is weak.
# +-----------------------------------------------------------------------------------------------------------
# ckpt=logs/dppo/dppo-pretrain/single_lift_real6_bc_v1/jfwqs/checkpoint/state_1500.pt
# normalization=dataset/dppo/single_lift_real6_bc_v1/normalization.npz
# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --ckpt ${ckpt} --ft-denoising-steps 0 --normalization ${normalization} \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_armfocus.yaml \
#   --action-config gentle_manip/configs/action/abs_pose_euler_abs_gripper_z15.yaml \
#   --act-steps 4 --temporal-ensemble --ensemble-m 0.33 --smooth-alpha 0.6 --max-pos-step-m 0.0065 \
#   --record dataset/real_deploy/real6_bc_g5recipe_jfwqs_1500 --shard-size 10 --record-rgb \
#   --max-steps 5000 "$@"


# +-----------------------------------------------------------------------------------------------------------
# | REAL-only BC, RGB = lyslr: ImageNet-pretrained ResNet-18 (GroupNorm) at 224px, horizon 16, shift +
# | photometric aug. Matched to the point-cloud arm otherwise. Stopped at 500 ep; val bottomed at 100,
# | so try EARLY ckpts first (100/200/400), not the last. Normalization is inside the encoder.
# +-----------------------------------------------------------------------------------------------------------
# ckpt=logs/dppo/dppo-pretrain/single_lift_real7_bc_rgb224_v1/lyslr/checkpoint/state_200.pt
# normalization=dataset/dppo/single_lift_real7_bc_rgb224_v1/normalization.npz
# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --ckpt ${ckpt} --ft-denoising-steps 0 --normalization ${normalization} \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_armfocus_rgb.yaml \
#   --action-config gentle_manip/configs/action/abs_pose_euler_abs_gripper_z15.yaml \
#   --act-steps 4 --temporal-ensemble --ensemble-m 0.33 --smooth-alpha 0.6 --max-pos-step-m 0.0065 \
#   --record dataset/real_deploy/real7_bc_rgb_lyslr_200 --shard-size 10 --record-rgb \
#   --max-steps 5000 "$@"


# +-----------------------------------------------------------------------------------------------------------
# | REAL-only BC, RGB 600ep = pwifv: lyslr with EPOCHS=600 only. NOT a truncation -- the cosine anneals to
# | min_lr by 600, where lyslr was still near peak LR, so state_600 is a converged short run (DP uses 600).
# | Compare state_600 against lyslr/state_200. Falls back through 500/400 if it is worse.
# +-----------------------------------------------------------------------------------------------------------
# ckpt=logs/dppo/dppo-pretrain/single_lift_real7_bc_rgb224_v1/pwifv/checkpoint/state_200.pt
# normalization=dataset/dppo/single_lift_real7_bc_rgb224_v1/normalization.npz
# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --ckpt ${ckpt} --ft-denoising-steps 0 --normalization ${normalization} \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_armfocus_rgb.yaml \
#   --action-config gentle_manip/configs/action/abs_pose_euler_abs_gripper_z15.yaml \
#   --act-steps 4 --temporal-ensemble --ensemble-m 0.33 --smooth-alpha 0.6 --max-pos-step-m 0.0065 \
#   --record dataset/real_deploy/real7_bc_rgb_pwifv_200 --shard-size 10 --record-rgb \
#   --max-steps 5000 "$@"


# +-----------------------------------------------------------------------------------------------------------
# | REAL-only BC, RGB 600ep, ResNet FROM SCRATCH = bhzdw: pwifv with pretrained=False, ONE config line apart.
# | Tests whether ImageNet features are what generalize -- so test on objects NOT in the 117 demos; both look
# | fine on demonstrated ones. Verified random at the weights (0/64 conv1 filters aligned to ImageNet).
# +-----------------------------------------------------------------------------------------------------------
ckpt=logs/dppo/dppo-pretrain/single_lift_real7_bc_rgb224_v1/bhzdw/checkpoint/state_200.pt
normalization=dataset/dppo/single_lift_real7_bc_rgb224_v1/normalization.npz
uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
  --ckpt ${ckpt} --ft-denoising-steps 0 --normalization ${normalization} \
  --obs-config gentle_manip/configs/obs/point_cloud_1cam_armfocus_rgb.yaml \
  --action-config gentle_manip/configs/action/abs_pose_euler_abs_gripper_z15.yaml \
  --act-steps 4 --temporal-ensemble --ensemble-m 0.33 --smooth-alpha 0.6 --max-pos-step-m 0.0065 \
  --record dataset/real_deploy/real7_bc_rgb_bhzdw_200 --shard-size 10 --record-rgb \
  --max-steps 5000 "$@"
