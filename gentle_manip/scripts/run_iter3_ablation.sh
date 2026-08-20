#!/bin/bash
# [local] Iteration-3 ablation: which v4 objective terms actually fix the defects?
#
# Each arm changes ONE thing from the honest collector_v3 baseline, so an effect is attributable.
# Run sequentially (each needs its own fresh server — the server owns the geometry RNG) and compare
# against the Iteration-0 collector_v3 gate run, which used identical scenarios.
#
# What to look for in each summary.json:
#   pinch_grasp_rate / stem_grasp_rate   the defects the terms target
#   grasp_tilt_deg_mean / grasp_occ_pred_mean
#   success_rate                          MUST NOT regress — a term that fixes a defect by
#                                         refusing to grasp is not a fix
#   stress_*_mean                         the gentleness the whole objective exists for
#
#   NEPS=25 bash gentle_manip/scripts/run_iter3_ablation.sh
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"

NEPS=${NEPS:-25}
PORT=${PORT:-5584}
BASE="--grasp-profile collector_v3 --scene-group-size 2 --maxfevals 1000 --record-batches 0"

# name                 extra flags
run () {
    echo "=================================================================="
    echo "[iter3] $1   (n=$NEPS)"
    echo "=================================================================="
    shift
    bash gentle_manip/scripts/run_grasp_bench.sh "iter3_$1" "$PORT" "$NEPS" $BASE "${@:2}"
    sleep 5
}

# baseline repeat at this episode count, so every arm is compared like-for-like
run "baseline (collector_v3)"            baseline
# w_peak: the anti-pinch term that has been inert in every run to date
run "peak (anti-pinch, w_peak=0.3)"      peak        --grasp-peak 0.3
# hard floor on the worst pad's contact area — 20 mm^2, mid-distribution per Iteration 1
run "areamin (anti-pinch, 20mm2)"        areamin     --grasp-area-min 2e-5
# COM lever arm — the anti-stem term
run "com (anti-stem, w_com=2e5)"         com         --grasp-com 2e5
# verticality prior + structurally tightened roll band (the two anti-side-grasp levers)
run "tilt (w_tilt=2e4)"                  tilt        --grasp-tilt 2e4
run "roll30 (roll_max=30deg)"            roll30      --grasp-roll-max-deg 30
# occlusion — note it is driven by the closing-axis YAW, not by tilt, so this is NOT
# redundant with the two above
run "occ (w_occ=2e4)"                    occ         --grasp-occ 2e4
# everything that helped, together (weights refined after reading the singles)
run "combined"                           combined    --grasp-peak 0.3 --grasp-area-min 2e-5 \
                                                     --grasp-com 2e5 --grasp-occ 2e4 \
                                                     --grasp-roll-max-deg 30

echo "[iter3] all arms complete:"
ls -1dt "$REPO"/logs/scripted_policy/*grasp_synth_fem | head -9
echo
echo "compare with: uv run --project envs/sim python -m gentle_manip.scripts.compare_evals <base> <arm>"
