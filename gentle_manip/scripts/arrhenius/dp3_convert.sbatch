#!/bin/bash
# [arrhenius] Convert a demo pkl to a DP3 zarr dataset on the GH200 (aarch64) nodes, via
# envs/dp3_arrhenius. Needs the GCC 14.3 buildenv module's libstdc++ on LD_LIBRARY_PATH at
# RUNTIME (not just build time) for PyTorch3D's compiled extension to load -- the system
# /lib64/libstdc++.so.6 lacks the CXXABI symbol version PyTorch3D was compiled against.
#
# Submit:  PKL=<path> OUT=<path> [DERIVE_ACTION=<cfg.yaml>] sbatch gentle_manip/scripts/arrhenius/dp3_convert.sbatch
#   knobs (env): PKL (required), OUT (required), OBS_KEYS (default "ee_pos ee_quat gripper_width"),
#                POINT_CLOUD_KEY (default point_cloud), DERIVE_ACTION (optional action-derive cfg),
#                OVERWRITE (default 1)
#
#SBATCH -J dp3_convert
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -A naiss2026-3-141-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip/logs/slumr_logs/%j.out
#SBATCH --error=/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip/logs/slumr_logs/%j.err

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
REPO=/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip
cd "$REPO"

GCCLIB=/software/sse2/el9_gh200/manual/GCC/14.3.0/hpc1/lib64
export LD_LIBRARY_PATH="$GCCLIB:${LD_LIBRARY_PATH:-}"

PKL=${PKL:?set PKL=<demo pkl path>}
OUT=${OUT:?set OUT=<output .zarr path>}
AGENT_POS_KEYS=${AGENT_POS_KEYS:-"ee_pos ee_quat gripper_width"}
POINT_CLOUD_KEY=${POINT_CLOUD_KEY:-point_cloud}
DERIVE_ACTION=${DERIVE_ACTION:-}
DERIVE_LOOKAHEAD=${DERIVE_LOOKAHEAD:-1}
DERIVE_SOURCE_ACTION=${DERIVE_SOURCE_ACTION:-}
OVERWRITE=${OVERWRITE:-1}

echo "[sbatch] host=$(hostname) arch=$(uname -m) job=${SLURM_JOB_ID:-NA}"
echo "[sbatch] pkl=$PKL out=$OUT derive_action=${DERIVE_ACTION:-<none>}"
[ -f "$PKL" ] || { echo "[sbatch] MISSING pkl: $PKL"; exit 1; }

ARGS=("$PKL" -o "$OUT" --agent-pos-keys $AGENT_POS_KEYS --point-cloud-key "$POINT_CLOUD_KEY")
[ "$OVERWRITE" = "1" ] && ARGS+=(--overwrite)
[ -n "$DERIVE_ACTION" ] && ARGS+=(--derive-action "$DERIVE_ACTION" --derive-lookahead "$DERIVE_LOOKAHEAD")
[ -n "$DERIVE_SOURCE_ACTION" ] && ARGS+=(--derive-source-action "$DERIVE_SOURCE_ACTION")

unset PYTHONPATH ROS_DISTRO 2>/dev/null || true
uv run --project envs/dp3_arrhenius --no-sync python gentle_manip/scripts/convert_demo_to_dp3.py "${ARGS[@]}"
echo "[sbatch] conversion done (job ${SLURM_JOB_ID:-NA})"
