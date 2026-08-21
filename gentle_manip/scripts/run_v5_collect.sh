#!/bin/bash
# [local] Part D collection: 500 v5 episodes, rate-bounded, azimuth-bounded, retry on, robust DR.
#
# GATED: refuses to start unless the v5 n=100 success confirmation passed (success >= 0.95).
# The n=25 sweep showed success 1.000, but n=25 draws only the easy first batches — the shelf
# taught that lesson at a cost of a day (0.960 at n=25 was 0.820 at n=100).
#
#   bash gentle_manip/scripts/run_v5_collect.sh [n_episodes]
set -u
NEPS=${1:-500}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
UV="env -u PYTHONPATH -u ROS_DISTRO MUJOCO_GL=egl uv run --project envs/sim --no-sync python"

# ── gate ─────────────────────────────────────────────────────────────────────
EVAL_LOG=logs/grasp_bench/v5_n100_eval.log
grep -aq "DONE" "$EVAL_LOG" || { echo "GATE: v5 n=100 not finished"; exit 2; }
SUCC=$(grep -a "DONE" "$EVAL_LOG" | tail -1 | sed -n 's/.*success \([0-9.]*\).*/\1/p')
echo "GATE: v5 n=100 success = $SUCC (need >= 0.95)"
awk "BEGIN{exit !($SUCC >= 0.95)}" || { echo "GATE FAILED — do not collect; reassess the azimuth bound"; exit 1; }

# ── collect ──────────────────────────────────────────────────────────────────
# --grasp-profile v5: the SAME objective resolution the benchmark uses (grasp_profiles.py).
# Rate bound comes from the experiment's action config (abs_pose_abs_gripper.rate_limit);
# robust-start DR from the experiment (soft_orientation_robust); retry + width-range = the
# v4.1 robustness knobs. First ~100 episodes double as the unforced retry verification (Part C):
# watch the [retry] lines — natural slip rate should be a few %, recoveries should lift.
$UV grasp_synthesis/collect_demos_synth_v4.py \
    --experiment single_lift_mushroom_soft_abs_action_robust \
    --n-episodes "$NEPS" --n-envs 8 --scene-dr-every 1 --record-video 20 --grasp-gpu \
    --grasp-profile v5 --preshape-factor 1.35 \
    --retry-max 2 --init-width-range 0.05 0.08
