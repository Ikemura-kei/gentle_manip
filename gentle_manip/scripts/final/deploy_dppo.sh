#! /bin/bash
# [final] Real-robot deploy of a DPPO BC policy (envs/dppo_deploy). One entry per policy, newest LAST and ACTIVE;
# older entries stay as comments (same convention as gentle_manip/scripts/deploy_real.sh).
# Rules (docs/training_plan_sim2real_2026-09.md §6): obs config = the real twin of the training obs
# (point_cloud_1cam_armfocus == superset_soft_armfocus_board: same crop z>=19 mm, 1024 pts, outlier + object focus;
#  plus the REAL-ONLY ground_residual filter, default ON since 2026-09-06),
# action config = the SAME yaml the demos were converted with (z15), normalization = the training dataset's.
# --record-rgb: also saves cam_ext RGB per step to <record>/videos/ep_NNN.mp4 (presentation only, never policy input).
# Visualize a recording afterwards: bash gentle_manip/scripts/final/viz_deploy.sh <record dir>
# Before the first deploy of the day: uv run --project envs/deploy python -m gentle_manip.diagnostics.drift_check
set -euo pipefail
cd "$(dirname "$0")/../../.."

# ── generalist v5 LOCAL, wiayg (2026-09-08), 1M steps. Sim teasers: tofu 14/20, mushroom 17/20, banana 13/20, cherry 3/20.
# ckpt=logs/dppo/dppo-pretrain/single_lift_generalist_soft_v5/wiayg/checkpoint/state_120.pt
# normalization=dataset/dppo/single_lift_generalist_soft_v5/normalization.npz
# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --ckpt ${ckpt} --ft-denoising-steps 0 --normalization ${normalization} \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_armfocus.yaml \
#   --action-config gentle_manip/configs/action/abs_pose_euler_abs_gripper_z15.yaml \
#   --act-steps 4 --smooth-alpha 0.6 --max-pos-step-m 0.0065 \
#   --record dataset/real_deploy/generalist_v5_wiayg_120 --shard-size 10 --record-rgb \
#   --max-steps 5000 "$@"

# ── G0 = fdcjk (cluster): recipe v5 with the PAIRED real-sim term ablated (w=1e-8, log-only). Not yet evaluated.
# ckpt=downloaded_runs/fdcjk/checkpoint/state_120.pt
# normalization=downloaded_runs/fdcjk/normalization.npz
# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --ckpt ${ckpt} --ft-denoising-steps 0 --normalization ${normalization} \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_armfocus.yaml \
#   --action-config gentle_manip/configs/action/abs_pose_euler_abs_gripper_z15.yaml \
#   --act-steps 4 --smooth-alpha 0.6 --max-pos-step-m 0.0065 \
#   --record dataset/real_deploy/generalist_v5_fdcjk_G0_120 --shard-size 10 --record-rgb \
#   --max-steps 5000 "$@"

# ── G2 = ttukt (cluster): recipe v5 with the ENCODER CONSISTENCY term ablated (w=1e-8, log-only). Not yet evaluated.
ckpt=downloaded_runs/ttukt/checkpoint/state_120.pt
normalization=downloaded_runs/ttukt/normalization.npz
uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
  --ckpt ${ckpt} --ft-denoising-steps 0 --normalization ${normalization} \
  --obs-config gentle_manip/configs/obs/point_cloud_1cam_armfocus.yaml \
  --action-config gentle_manip/configs/action/abs_pose_euler_abs_gripper_z15.yaml \
  --act-steps 4 --smooth-alpha 0.6 --max-pos-step-m 0.0065 \
  --record dataset/real_deploy/generalist_v5_ttukt_G2_120 --shard-size 10 --record-rgb \
  --max-steps 5000 "$@"

# ── G1 = bmbrv (cluster): recipe v5 EXACT, the cluster twin of wiayg. Same dataset + val split. Not yet evaluated.
# ckpt=downloaded_runs/bmbrv/checkpoint/state_120.pt
# normalization=downloaded_runs/bmbrv/normalization.npz
# uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#   --ckpt ${ckpt} --ft-denoising-steps 0 --normalization ${normalization} \
#   --obs-config gentle_manip/configs/obs/point_cloud_1cam_armfocus.yaml \
#   --action-config gentle_manip/configs/action/abs_pose_euler_abs_gripper_z15.yaml \
#   --act-steps 4 --smooth-alpha 0.6 --max-pos-step-m 0.0065 \
#   --record dataset/real_deploy/generalist_v5_bmbrv_G1_120 --shard-size 10 --record-rgb \
#   --max-steps 5000 "$@"
