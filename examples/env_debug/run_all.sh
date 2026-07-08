#!/usr/bin/env bash
# Run every per-env smoke test in its OWN uv environment and print a summary.
# Each env must already be synced (`uv sync --project envs/<name>`); torch is installed
# manually in sim/dp3/dppo (see each pyproject header). Usage:
#   bash examples/env_debug/run_all.sh              # all envs
#   bash examples/env_debug/run_all.sh sim dppo     # only these
set -u
cd "$(dirname "$0")/../.." || exit 1              # repo root

if [ "$#" -gt 0 ]; then ENVS=("$@"); else ENVS=(sim deploy dp3 dppo dppo_deploy serl); fi

# Clean, cluster-like environment: drop any inherited PYTHONPATH/ROS pollution (uv envs
# are self-contained) and force headless GL for the genesis (sim) import.
RUN=(env -u PYTHONPATH -u ROS_DISTRO MUJOCO_GL=egl)

declare -A RESULT
for e in "${ENVS[@]}"; do
  echo
  echo "############################################################"
  echo "# envs/$e"
  echo "############################################################"
  if [ ! -d "envs/$e/.venv" ]; then
    echo "  SKIP: envs/$e/.venv missing — run 'uv sync --project envs/$e' first"
    RESULT[$e]="SKIP (not synced)"
    continue
  fi
  # --no-sync so a missing manual-torch install surfaces as a FAIL, not a silent re-sync.
  if "${RUN[@]}" uv run --project "envs/$e" --no-sync python "examples/env_debug/check_$e.py"; then
    RESULT[$e]="PASS"
  else
    RESULT[$e]="FAIL (see output above)"
  fi
done

echo
echo "============================================================"
echo "SUMMARY"
echo "============================================================"
rc=0
for e in "${ENVS[@]}"; do
  printf "  %-8s %s\n" "$e" "${RESULT[$e]}"
  case "${RESULT[$e]}" in FAIL*) rc=1 ;; esac
done
exit $rc
