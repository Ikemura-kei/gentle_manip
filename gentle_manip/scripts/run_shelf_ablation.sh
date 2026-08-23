#!/bin/bash
# [local] v4.1 shelf-lift ablation — the 2x2, then the theta sweep.
#
# WHY A 2x2 AND NOT A SWEEP. Rotating the closing axis toward vertical puts one finger beneath the
# other, so the object's weight is carried by a NORMAL force instead of by friction. But at a FIXED
# width the rotation ADDS normal load (mg*sin(theta)/2, first order in von Mises) while only removing
# shear (second order). So the rotation alone is a plausible REGRESSION; the gain is supposed to come
# from spending the freed grip margin on a width release. The 2x2 separates those two effects. A
# sweep run first would confound them.
#
#     theta=0,  open=0     -> baseline (identical to v4)
#     theta=0,  open=2.5mm -> the release alone (expected: drops the object, or at least less grip)
#     theta=55, open=0     -> the rotation alone (the regression test)
#     theta=55, open=2.5mm -> the proposal
#
# 55 deg = arctan(1/mu) for mu=0.7, where P_min(theta) = (mg/2)*max(cos/mu, sin) is minimized
# (0.57x baseline). 90 deg is WORSE (0.70x) -- the upper pad's contact becomes the binding
# constraint. The sweep in stage 2 measures the sim's EFFECTIVE mu by locating the true minimum.
#
#   bash gentle_manip/scripts/run_shelf_ablation.sh [2x2|sweep]
#   NEPS=25 OPEN=0.0025 bash gentle_manip/scripts/run_shelf_ablation.sh 2x2
set -u
STAGE=${1:-2x2}
NEPS=${NEPS:-25}
OPEN=${OPEN:-0.0025}
PORT0=${PORT0:-5610}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BENCH="$REPO/gentle_manip/scripts/run_grasp_bench.sh"

# v4fix = the resolved v4 objective (execute_offset 4.5mm + area_min 4e-5 + collector diversity).
# The shelf changes only the LIFT, so the grasp objective must be held fixed or the comparison is
# against a different grasp distribution rather than a different lift.
COMMON="--grasp-profile v4fix --traj v4 --preshape-factor 1.35"

run() {  # run <tag> <port> <shelf_deg> <shelf_open>
    local tag=$1 port=$2 deg=$3 open=$4
    echo "=== [$(date +%H:%M:%S)] $tag  theta=${deg}deg open=$(python3 -c "print($open*1e3)")mm ==="
    bash "$BENCH" "$tag" "$port" "$NEPS" $COMMON --shelf-deg "$deg" --shelf-open "$open"
}

case "$STAGE" in
  2x2)
    run shelf_t0_o0    $((PORT0+0)) 0   0
    run shelf_t0_oX    $((PORT0+1)) 0   "$OPEN"
    run shelf_t55_o0   $((PORT0+2)) 55  0
    run shelf_t55_oX   $((PORT0+3)) 55  "$OPEN"
    ;;
  sweep)
    # theta sweep at the open that won the 2x2 (pass it as OPEN=). theta=0 is already measured by
    # the 2x2 at the same open, so it is not repeated here.
    i=0
    for deg in 30 45 55 70 90; do
        run "shelf_t${deg}" $((PORT0+10+i)) "$deg" "$OPEN"
        i=$((i+1))
    done
    ;;
  occ)
    # GROUND-TRUTH occlusion, baseline vs the winning shelf, on the point-cloud experiment.
    # `grasp_occ_pred` is computed at the GRASP pose and therefore goes stale the moment the wrist
    # starts rotating, so a shelf run's occlusion can only be read from the rendered cloud
    # (occ_pcd_grasp / occ_pcd_lift). DEG must be passed in.
    export GM_BENCH_EXPERIMENT=single_lift_mushroom_soft_grasp_eval_pcd
    run occ_t0            $((PORT0+20)) 0              0
    run "occ_t${DEG:?DEG}" $((PORT0+21)) "${DEG}" "$OPEN"
    ;;
  *) echo "unknown stage: $STAGE (want 2x2 | sweep | occ)"; exit 2 ;;
esac
echo "=== [$(date +%H:%M:%S)] $STAGE done ==="
