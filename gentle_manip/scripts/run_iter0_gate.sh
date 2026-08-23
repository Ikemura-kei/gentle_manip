#!/bin/bash
# [local] Iteration-0 GATE: establish the HONEST v3 baseline.
#
# Runs the same benchmark twice, changing only the grasp OBJECTIVE:
#   strict        the historical benchmark config (metric defaults, no diversity) -- should
#                 reproduce the ~0.98-1.00 already on record, confirming nothing regressed
#   collector_v3  what collect_demos_synth_v3.py ACTUALLY runs (w_align 2000 + tilt seeding +
#                 diversity) -- the configuration that generated every dataset, and the number
#                 the rest of the v4 work must be measured against
#
# Sequential, not parallel: each run needs its own fresh server (the server owns the geometry RNG),
# and two Genesis processes would contend for the GPU and skew the timing comparison.
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"

NEPS=${NEPS:-100}
PORT=${PORT:-5583}
COMMON=${COMMON:---scene-group-size 2 --maxfevals 1000}

for PROFILE in strict collector_v3; do
    echo "=================================================================="
    echo "[gate] profile=$PROFILE  n_episodes=$NEPS"
    echo "=================================================================="
    bash gentle_manip/scripts/run_grasp_bench.sh "gate_$PROFILE" "$PORT" "$NEPS" \
        --grasp-profile "$PROFILE" $COMMON
    echo "[gate] $PROFILE finished rc=$?"
    sleep 5
done

echo "[gate] both runs complete. Newest two scripted_policy runs:"
ls -1dt "$REPO"/logs/scripted_policy/*grasp_synth_fem | head -2
