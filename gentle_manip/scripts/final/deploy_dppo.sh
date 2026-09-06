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

# ── tofu sim2real v1, run tzdhk (2026-09-06): sim-only demos (100, run 26-09-05-jvt) + paired reg w=0.5 +
#    train-time cloud aug (d435i_noise + offset 8 mm); stopped at ep 2000, val min @750; sim teaser state_1500 = 0.15/0.50.
#    Alternatives on disk: state_750 (val min) / state_1000 / state_2000.
ckpt=logs/dppo/dppo-pretrain/single_lift_tofu_sim2real_v1/tzdhk/checkpoint/state_1000.pt
normalization=dataset/dppo/single_lift_tofu_sim2real_v1/normalization.npz
uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
  --ckpt ${ckpt} --ft-denoising-steps 0 --normalization ${normalization} \
  --obs-config gentle_manip/configs/obs/point_cloud_1cam_armfocus.yaml \
  --action-config gentle_manip/configs/action/abs_pose_euler_abs_gripper_z15.yaml \
  --act-steps 4 --smooth-alpha 0.6 --max-pos-step-m 0.0065 \
  --record dataset/real_deploy/tofu_sim2real_v1_tzdhk_1000 --shard-size 10 --record-rgb \
  --max-steps 5000 "$@"
