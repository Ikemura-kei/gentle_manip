#!/bin/bash
# Per-axis real->sim point-cloud shift of the paired PLAY data, arm vs object (plane cut), first frames.
# Pair = the runs used by gen_sim_play_data.sh. CPU only, a few seconds.
set -euo pipefail
cd "$(dirname "$0")/../../.."
real=dataset/demos/play_red_cube_real/26-09-05-xiv
sim=dataset/demos/play_red_cube_soft/26-09-05-xiv
uv run --project envs/sim python -m gentle_manip.scripts.paired_cloud_shift \
    --real "$real" --sim "$sim" --frames "${FRAMES:-10}" --z-cut "${ZCUT:-0.06}" --plot "$@"   # plots: <sim run>/shift_ep*_frame5.png
