#!/bin/bash
# [local] Run ONE grasp-synthesis benchmark end-to-end: launch a fresh sim server, wait for it to
# be ready, run eval_grasp_synth against it, then tear the server down.
#
# The server must be FRESH per run: it owns the scene-geometry RNG, so reusing one across two
# profiles would give the second run a different geometry sequence and break the apples-to-apples
# comparison the whole Iteration-0 gate depends on.
#
#   bash gentle_manip/scripts/run_grasp_bench.sh <tag> <port> <n_episodes> [extra eval args...]
#
# e.g. bash gentle_manip/scripts/run_grasp_bench.sh strict 5583 100 --grasp-profile strict
set -u

TAG=${1:?tag}; PORT=${2:?port}; NEPS=${3:?n_episodes}; shift 3
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
LOGDIR=${GM_BENCH_LOGDIR:-$REPO/logs/grasp_bench}
mkdir -p "$LOGDIR"
SRV_LOG="$LOGDIR/${TAG}_server.log"
EVAL_LOG="$LOGDIR/${TAG}_eval.log"

cd "$REPO"
UV="env -u PYTHONPATH -u ROS_DISTRO MUJOCO_GL=egl uv run --project envs/sim --no-sync python"

cleanup() { [ -n "${SRV_PID:-}" ] && kill "$SRV_PID" 2>/dev/null; wait "$SRV_PID" 2>/dev/null; }
trap cleanup EXIT INT TERM

# --- server -----------------------------------------------------------------
echo "[$TAG] launching sim server on :$PORT" | tee "$EVAL_LOG"
$UV -m gentle_manip.scripts.serl_sim_server \
    --experiment "${GM_BENCH_EXPERIMENT:-single_lift_mushroom_soft_grasp_eval}" --view teacher \
    --num-envs "${GM_BENCH_ENVS:-5}" --render-rgb --subprocess --port "$PORT" > "$SRV_LOG" 2>&1 &
SRV_PID=$!

for i in $(seq 1 120); do
    grep -q SIM_SERVER_READY "$SRV_LOG" 2>/dev/null && break
    kill -0 "$SRV_PID" 2>/dev/null || { echo "[$TAG] server DIED during startup:" | tee -a "$EVAL_LOG"; tail -20 "$SRV_LOG" | tee -a "$EVAL_LOG"; exit 1; }
    sleep 2
done
grep -q SIM_SERVER_READY "$SRV_LOG" || { echo "[$TAG] server never became ready" | tee -a "$EVAL_LOG"; exit 1; }
echo "[$TAG] server ready (pid $SRV_PID)" | tee -a "$EVAL_LOG"

# --- eval -------------------------------------------------------------------
$UV -m gentle_manip.scripts.eval_grasp_synth \
    --synth "${GM_BENCH_SYNTH:-fem}" --port "$PORT" --n-episodes "$NEPS" \
    --num-envs "${GM_BENCH_ENVS:-5}" --seed "${GM_BENCH_SEED:-0}" \
    "$@" >> "$EVAL_LOG" 2>&1
RC=$?
echo "[$TAG] eval exited rc=$RC" | tee -a "$EVAL_LOG"
tail -3 "$EVAL_LOG"
exit $RC
