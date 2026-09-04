#!/bin/bash
set -u
EXP=$1; N=$2; BASE=$3; BW=${4:-own}; EXTRA=${5:-}
cd "$(dirname "$0")/.."
env -u PYTHONPATH -u ROS_DISTRO uv run --project envs/sim --no-sync python \
  grasp_synthesis/collect_demos_baseline.py \
  --baseline $BASE --baseline-width $BW $EXTRA \
  --experiment "$EXP" \
  --n-episodes $N --n-envs 8 --scene-dr-every 1 --maxfevals 1145 --seed 0 \
  --n-home-to-pre 77 --n-grasp 20 --n-settle 1 \
  --cam-azimuth-max-deg 60 \
  --grasp-diversity-tol 0 --grasp-jitter-deg 0 --grasp-jitter-pos 0 --grasp-pitch-seed-deg 0 \
  --grasp-w-peak 0.3 --approach-xy-finish 0.45 0.75 --approach-speed 0.0024 \
  --held-run-max 12 --held-run-keep 10 \
  --grasp-area-min-mm2 auto --grasp-width-max-mm auto --grasp-yaw-max-deg auto \
  --grasp-w-press 0.05 --record-video 100000 \
  --description "E1 BASELINE $BASE/$BW: gentleness-blind comparison, same executor"
