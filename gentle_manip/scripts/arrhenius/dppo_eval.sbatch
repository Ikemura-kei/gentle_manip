#!/bin/bash
# [arrhenius] Canonical DPPO eval on the GH200 (aarch64) nodes — the shared eval harness
# (gentle_manip.dppo.eval_agent.EvalHarnessAgent via eval_diffusion_pointnet). Same two-process
# pattern as the finetune: a genesis serl_sim_server (envs/sim_arrhenius, --subprocess for the
# per-group scene rebuilds) + the eval launcher (envs/dppo_arrhenius) over 127.0.0.1:PORT.
#
# Submit:  sbatch gentle_manip/scripts/arrhenius/dppo_eval.sbatch
# Eval a specific checkpoint:  CKPT=/path/to/state_X.pt sbatch .../dppo_eval.sbatch
# Writes to <base_policy_run>/eval/<datetime>/ (summary.json + episodes.csv + per-episode videos).
#
#SBATCH -J dppo_eval
#SBATCH -N 1
#SBATCH -t 01:30:00
#SBATCH -A naiss2026-3-141-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip/logs/slumr_logs/%j.out
#SBATCH --error=/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip/logs/slumr_logs/%j.err

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
REPO=/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip
cd "$REPO"

# ---- knobs (eval = canonical protocol; do NOT change num_envs/seed/n_episodes) ---------
# Env-overridable so you can eval a finetuned ckpt with its TRAIN-TIME config:
#   SIM_EXPERIMENT (server task/reward+DR), EVAL_EXPERIMENT (eval-cfg experiment snapshot),
#   FT_DENOISING (ft_denoising_steps: 0 for a BC base ckpt, 10 for a finetuned ckpt),
#   GM_SIM_SUBSTEPS / GM_MPM_SAMPLER (sim substeps + MPM sampler, read by scene_builder).
# CANONICAL eval sim is pinned in a CONFIG FILE, not reconstructed via env-vars: the default
# experiment single_lift_mushroom_soft_eval fixes sim_substeps=235 / mpm_grid_density=250 in its
# task config, so EVERY policy is evaluated on the same reference sim regardless of how it trained.
# Do NOT pass GM_SIM_SUBSTEPS for a canonical eval (the eval task pins the fidelity); only override
# for a deliberate off-reference probe. Only ft_denoising (policy: 0=BC, 10=finetuned) + CKPT vary.
SIM_EXPERIMENT=${SIM_EXPERIMENT:-single_lift_mushroom_soft_eval}   # server sim: pinned 235/250
EVAL_EXPERIMENT=${EVAL_EXPERIMENT:-single_lift_mushroom_soft_eval} # eval-cfg experiment snapshot (same)
export GM_MPM_SAMPLER=${GM_MPM_SAMPLER:-regular}             # pinned sampler (aarch64 reference)
FT_DENOISING=${FT_DENOISING:-0}                              # 0 = eval BC base; 10 = eval finetuned denoising
VIEW=student
NUM_ENVS=5                                      # canonical EvalSpec; MUST match cfg n_envs
PORT=5570
CKPT=${CKPT:-$REPO/logs/dppo/dppo-pretrain/2026-07-04_20-03-36_42/checkpoint/state_3000.pt}
CFG_DIR=$REPO/gentle_manip/dppo/cfg/single_lift_mushroom_soft_pcd
CFG_NAME=eval_diffusion_pointnet
NORM=$REPO/dataset/dppo/single_lift_mushroom_soft_pcd/normalization.npz

export WANDB_MODE=offline
export MUJOCO_GL=egl
unset PYTHONPATH ROS_DISTRO 2>/dev/null || true

echo "[sbatch] host=$(hostname) arch=$(uname -m) job=${SLURM_JOB_ID:-NA} eval ckpt=$CKPT"
[ -f "$CKPT" ] || { echo "[sbatch] MISSING checkpoint: $CKPT"; exit 1; }
[ -f "$NORM" ] || { echo "[sbatch] MISSING normalization: $NORM"; exit 1; }

SRV_LOG=$REPO/logs/slumr_logs/${SLURM_JOB_ID:-local}_simserver.log

# ---- sim server (subprocess backend: eval drives deterministic per-group scene rebuilds) ----
echo "[sbatch] launching sim server ($SIM_EXPERIMENT / $VIEW / $NUM_ENVS envs, --subprocess) on $PORT"
uv run --project envs/sim_arrhenius --no-sync python -m gentle_manip.scripts.serl_sim_server \
    --experiment "$SIM_EXPERIMENT" --view "$VIEW" --num-envs "$NUM_ENVS" \
    --render-rgb --subprocess --port "$PORT" > "$SRV_LOG" 2>&1 &
SRV_PID=$!
cleanup() { echo "[sbatch] stopping sim server ($SRV_PID)"; kill "$SRV_PID" 2>/dev/null || true; wait "$SRV_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "[sbatch] waiting for SIM_SERVER_READY ($SRV_LOG)"
for _ in $(seq 1 180); do
    grep -q "SIM_SERVER_READY" "$SRV_LOG" 2>/dev/null && { echo "[sbatch] server ready"; break; }
    kill -0 "$SRV_PID" 2>/dev/null || { echo "[sbatch] server died early — tail:"; tail -50 "$SRV_LOG"; exit 1; }
    sleep 5
done
grep -q "SIM_SERVER_READY" "$SRV_LOG" || { echo "[sbatch] server not ready — tail:"; tail -50 "$SRV_LOG"; exit 1; }

# ---- run the canonical eval (BC base policy; record_batches=null = ONE clip PER EPISODE) ----
# Hard requirement #2: per-trajectory video, no exceptions — record_batches=null records ALL
# batches / ALL envs (100 clips), NOT a subset. (The x86 ref run used record_batches=2; that
# is a video subset only and does not change the success number.)
# N_EPISODES: full canonical eval = 100 (20 batches); set smaller (e.g. 25) for a quick diagnostic.
N_EPISODES=${N_EPISODES:-100}
EVAL_EXP_OVR=(); [ -n "$EVAL_EXPERIMENT" ] && EVAL_EXP_OVR=(experiment="$EVAL_EXPERIMENT")
# Unique output subdir (optional). By default the eval writes to <run>/eval/<datetime> where the
# datetime has 1-SECOND resolution and is stamped when the eval PROCESS starts — so evaluating
# several checkpoints of the SAME run in parallel makes them collide on that dir and clobber each
# other's summary.json / episodes.csv / videos. Set EVAL_SUBDIR to a unique name (e.g. the
# checkpoint stem "state_150") and the eval writes to <run>/eval/$EVAL_SUBDIR instead — collision
# free. Unset -> unchanged datetime behaviour. (RUN_DIR = the ckpt's run dir = <run>/checkpoint/..).
LOGDIR_OVR=()
if [ -n "${EVAL_SUBDIR:-}" ]; then
    RUN_DIR=$(dirname "$(dirname "$CKPT")")
    LOGDIR_OVR=(logdir="$RUN_DIR/eval/$EVAL_SUBDIR")
    echo "[sbatch] unique eval logdir -> $RUN_DIR/eval/$EVAL_SUBDIR (EVAL_SUBDIR set)"
fi
echo "[sbatch] running eval of $CKPT (n_episodes=$N_EPISODES, ft_denoising=$FT_DENOISING, ${EVAL_EXP_OVR[*]:-exp=cfg-default})"
uv run --project envs/dppo_arrhenius --no-sync python -m gentle_manip.dppo.train \
    --config-path "$CFG_DIR" --config-name "$CFG_NAME" \
    base_policy_path="$CKPT" ft_denoising_steps="$FT_DENOISING" record_batches=null n_episodes="$N_EPISODES" \
    "${EVAL_EXP_OVR[@]}" "${LOGDIR_OVR[@]}"

echo "[sbatch] eval finished OK"
