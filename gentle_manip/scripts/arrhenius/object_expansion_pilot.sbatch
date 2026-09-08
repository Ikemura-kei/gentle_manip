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
REPO=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip
cd "$REPO"
# Guard against submodule pointer drift (2026-09-08: third_party/genesis was checked out at
# an OLDER commit than this branch's recorded pointer -- missing get_links_net_coupling_force,
# which crashed every single soft-body collection attempt with an AttributeError. `git
# submodule update` is a fast, safe sync (no --init needed, submodule already exists) so do
# it every run rather than trust the working tree state.
git submodule update third_party/genesis 2>&1 | sed 's/^/[sbatch] submodule: /' || true
# NOT `export PATH="$HOME/.local/bin:$PATH"` (the pattern other .sbatch files in this repo
# use): $HOME/.local/bin/uv on THIS account is an x86_64 binary (installed from the login
# node), and $HOME is the SAME path on the aarch64 GH200 compute nodes, so it would shadow
# any usable uv with one that fails "Exec format error" (hit exactly this in job 2163478).
# Use a dedicated aarch64 uv binary instead (downloaded from astral-sh/uv releases,
# uv-aarch64-unknown-linux-gnu.tar.gz) so this never depends on what's in $HOME/.local/bin.
export PATH="$REPO/.uv_aarch64:$PATH"

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

# Self-heal pymeshlab if it's missing/broken (idempotent -- job 2177412 hit "No module named
# 'pymeshlab'": envs/sim_arrhenius's pyproject.toml declares it but it was never actually
# installed there, and it needs a GH200-specific post-install lib swap besides -- see
# gentle_manip/scripts/arrhenius/fix_pymeshlab_gh200.sh's header for the full story. Both
# steps must run ON an aarch64 node, so bundle them into the job itself rather than costing
# another queue round-trip).
if ! uv run --project envs/sim_arrhenius --no-sync python -c "import pymeshlab" 2>/dev/null; then
    echo "[sbatch] pymeshlab missing/broken -- installing + applying the GH200 lib-swap fix"
    uv pip install --python envs/sim_arrhenius/.venv/bin/python         --python-platform aarch64-manylinux_2_35 'pymeshlab>=2025.7' ||         echo "[sbatch] WARN: pymeshlab install failed, continuing without it (repair_watertight doesn't need it)"
    bash gentle_manip/scripts/arrhenius/fix_pymeshlab_gh200.sh ||         echo "[sbatch] WARN: pymeshlab GH200 fix script failed, continuing without it"
fi

echo "[sbatch] launching: python -m gentle_manip.scripts.object_expansion.batch_expand ${ARGS[*]}"
uv run --project envs/sim_arrhenius --no-sync python -m gentle_manip.scripts.object_expansion.batch_expand \
    "${ARGS[@]}"
RC=$?
echo "[sbatch] batch_expand finished (job ${SLURM_JOB_ID:-NA}) rc=$RC"
exit $RC
