#!/bin/bash
# [local] Benchmark the OPERATING-POINT fix against the honest v3 baseline.
#
# Compares `v4fix` (execute_offset + w_peak + area_min) with `collector_v3` on the identical
# canonical scenarios. The offline result was a 2.6x reduction in executed stress (54.8 -> 21.0 kPa,
# i.e. below the 40 kPa yield instead of 37% above it) on ONE grasp; this is the confirmation.
#
# The thing that could sink it: a wider grip may SLIP. Success rate is the gate, not stress.
#   pass  = stress down materially AND success_rate not materially below the baseline
#   fail  = stress bought with success
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); cd "$REPO"
NEPS=${NEPS:-100}; PORT=${PORT:-5585}
bash gentle_manip/scripts/run_grasp_bench.sh v4fix "$PORT" "$NEPS" \
     --grasp-profile v4fix --scene-group-size 2 --maxfevals 1000
echo "[v4fix] done. Compare against the collector_v3 gate run:"
echo "  uv run --project envs/sim python -m gentle_manip.scripts.compare_evals <collector_v3_dir> <v4fix_dir>"
