#!/bin/bash
# [arrhenius] Object-expansion pilot: mesh_prep -> fem_gate_check -> register_object ->
# small collect_demos_synth_v4 run per candidate (gentle_manip/scripts/object_expansion/
# batch_expand.py). GPU only needed for the genesis/MPM collection step; FEM gate is CPU.
#
# Submit from the NOBACKUP checkout (envs/sim_arrhenius is aarch64-native, matches the GH200
# gpu-partition nodes -- the x86_64 login-node env cannot run this):
#   cd /nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip
#   CANDIDATES_CSV=dataset/object_expansion/candidates.csv N_OBJECTS=5 \
#     sbatch gentle_manip/scripts/arrhenius/object_expansion_pilot.sbatch
#
#SBATCH -J objexp_pilot
#SBATCH -N 1
#SBATCH -t 4:00:00
#SBATCH -A naiss2026-3-141-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip/logs/slurm_logs/%j.out
#SBATCH --error=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip/logs/slurm_logs/%j.err

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
REPO=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip
cd "$REPO"

N_OBJECTS=${N_OBJECTS:-5}
PER_CATEGORY=${PER_CATEGORY:-1}
N_EPISODES=${N_EPISODES:-5}
N_ENVS=${N_ENVS:-5}
MIN_SUCCESS=${MIN_SUCCESS:-0.5}
SKIP_COLLECT=${SKIP_COLLECT:-0}
CANDIDATES_CSV=${CANDIDATES_CSV:-dataset/object_expansion/candidates.csv}
ONLY_CATEGORIES=${ONLY_CATEGORIES:-}

echo "[sbatch] host=$(hostname) arch=$(uname -m) job=${SLURM_JOB_ID:-NA}"
echo "[sbatch] n_objects=$N_OBJECTS per_category=$PER_CATEGORY n_episodes=$N_EPISODES n_envs=$N_ENVS candidates_csv=$CANDIDATES_CSV only_categories=${ONLY_CATEGORIES:-<all>}"

ARGS=(--n-objects "$N_OBJECTS" --per-category "$PER_CATEGORY" --n-episodes "$N_EPISODES"
      --n-envs "$N_ENVS" --min-success "$MIN_SUCCESS" --candidates-csv "$CANDIDATES_CSV")
[ "$SKIP_COLLECT" = "1" ] && ARGS+=(--skip-collect)
if [ -n "$ONLY_CATEGORIES" ]; then
    # shellcheck disable=SC2206
    ARGS+=(--only-categories $ONLY_CATEGORIES)
fi

unset PYTHONPATH ROS_DISTRO 2>/dev/null || true
export MUJOCO_GL=egl

echo "[sbatch] launching: python -m gentle_manip.scripts.object_expansion.batch_expand ${ARGS[*]}"
uv run --project envs/sim_arrhenius --no-sync python -m gentle_manip.scripts.object_expansion.batch_expand \
    "${ARGS[@]}"
RC=$?
echo "[sbatch] batch_expand finished (job ${SLURM_JOB_ID:-NA}) rc=$RC"
exit $RC
