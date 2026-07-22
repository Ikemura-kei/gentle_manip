#!/bin/bash
# [arrhenius] DPPO point-cloud BC PRETRAIN on the GH200 (aarch64) nodes.
# UNLIKE finetune, this is PURE SUPERVISED: no genesis, no sim server, no substeps/sampler —
# it trains a Diffusion Policy on the converted .npz only. (The 235+regular eval fidelity is a
# SEPARATE concern, handled later by dppo_eval.sbatch.) One process on one GPU: the trainer in
# envs/dppo_arrhenius reading dataset/dppo/$DATA_ENV/train.npz.
#
# Submit:  sbatch gentle_manip/scripts/arrhenius/dppo_pretrain.sbatch
#   knobs (env):  DATA_ENV, N_EPOCHS, SAVE_FREQ, SMOKE=1 (short check), GM_EXTRA_OVERRIDES
# Prereq: dataset/dppo/$DATA_ENV/{train,val,normalization}.npz uploaded to the cluster $REPO.
#
#SBATCH -J dppo_pre
#SBATCH -N 1
#SBATCH -t 22:00:00
#SBATCH -A naiss2026-3-141-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip/logs/slumr_logs/%j.out
#SBATCH --error=/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip/logs/slumr_logs/%j.err

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
REPO=/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip
cd "$REPO"

# ---- run knobs -------------------------------------------------------------------------
SMOKE=${SMOKE:-0}                       # 1 = short pipeline check (4 epochs); 0 = full run
DATA_ENV=${DATA_ENV:-single_lift_mushroom_soft_pcd_wide1k}   # dataset + logdir name (1000-demo set)
N_EPOCHS=${N_EPOCHS:-8000}              # matches xxiaw's epoch budget (NOT matched-compute: the 1000-set
                                        # does ~3.3x more grad updates/epoch). Eval checkpoints (every
                                        # SAVE_FREQ) later + pick the best; final-vs-final can mislead.
SAVE_FREQ=${SAVE_FREQ:-1000}            # checkpoint every N epochs (eval each -> compare best-vs-best)
CFG_DIR=$REPO/gentle_manip/dppo/cfg/single_lift_mushroom_soft_pcd
CFG_NAME=pre_diffusion_pointnet

# The pretrain config resolves the dataset path + logdir from these (${oc.env:...}); export them
# so the job is self-contained regardless of the login-shell profile.
export DPPO_DATA_DIR=${DPPO_DATA_DIR:-$REPO/dataset/dppo}
export DPPO_LOG_DIR=${DPPO_LOG_DIR:-$REPO/logs/dppo}
export WANDB_MODE=${WANDB_MODE:-online}  # GH200 compute nodes have outbound internet (auth ~/.netrc)
unset PYTHONPATH ROS_DISTRO 2>/dev/null || true

# ---- preflight -------------------------------------------------------------------------
echo "[sbatch] host=$(hostname) arch=$(uname -m) job=${SLURM_JOB_ID:-NA} smoke=$SMOKE data_env=$DATA_ENV"
TRAIN_NPZ=$DPPO_DATA_DIR/$DATA_ENV/train.npz
[ -f "$TRAIN_NPZ" ] || { echo "[sbatch] MISSING dataset: $TRAIN_NPZ (upload dataset/dppo/$DATA_ENV/ first)"; exit 1; }

# ---- overrides -------------------------------------------------------------------------
if [ "$SMOKE" = "1" ]; then
    OVERRIDES=(train.n_epochs=4 train.save_model_freq=2)
else
    OVERRIDES=(train.n_epochs="$N_EPOCHS" train.save_model_freq="$SAVE_FREQ")
fi
# first_cycle_steps auto-tracks train.n_epochs in the config (single cosine cycle, no warm restart).

# Per-run extra hydra overrides (space-separated key=val), appended so they WIN (last wins).
if [ -n "${GM_EXTRA_OVERRIDES:-}" ]; then
    # shellcheck disable=SC2206  (intentional word-split into array elements)
    OVERRIDES+=($GM_EXTRA_OVERRIDES)
fi

TRAINER_LOG=$REPO/logs/slumr_logs/${SLURM_JOB_ID:-local}_pretrain.log
: > "$TRAINER_LOG"
GIT_COMMIT=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)

# Per-run Motivation/Hypothesis for EXPERIMENT.md (HARD REQ #6). Override via GM_MOTIVATION / GM_HYPOTHESIS.
: "${GM_MOTIVATION:=Coverage test: BC-pretrain the PointNet diffusion policy on the 1000-demo WIDE set ($DATA_ENV) to check whether more demonstrations lift the grasp-width generalization ceiling seen at 300 demos (xxiaw 8000ep = 0.77 success). Pure supervised, no sim.}"
: "${GM_HYPOTHESIS:=If failures were a data-coverage limit, success at the best checkpoint beats the 300-demo 0.77; if it plateaus near 0.77 the bottleneck is perception/tactile, not data.}"

# ---- EXPERIMENT.md writer (background): DPPO doesn't write one; do it once the run dir appears
# (the ExperimentSnapshot callback prints "... -> <run_dir>/config/" at launch).
(
  for _ in $(seq 1 240); do
    RD=$(grep -oE '/nobackup/[^ ]+/dppo-pretrain/[^ ]+/config/' "$TRAINER_LOG" 2>/dev/null | head -1 | sed 's#/config/##' || true)
    if [ -n "$RD" ] && [ -d "$RD" ]; then
      if [ ! -f "$RD/EXPERIMENT.md" ]; then
        cat > "$RD/EXPERIMENT.md" <<MD
# Experiment: $(basename "$RD")

- **Algorithm:** dppo-pretrain (BC)
- **Started:** $(date '+%Y-%m-%d %H:%M:%S')
- **Git commit:** \`$GIT_COMMIT\`
- **Run dir:** \`$RD\`
- **SLURM job:** ${SLURM_JOB_ID:-NA} on $(hostname)
- **Dataset:** \`$DATA_ENV\` (train.npz = $TRAIN_NPZ)
- **wandb:** project gentle-manip-$DATA_ENV, run $(basename "$RD") (mode $WANDB_MODE)
- **Extra hydra overrides:** \`${GM_EXTRA_OVERRIDES:-<none>}\`

## Motivation
$GM_MOTIVATION

## Hypothesis
$GM_HYPOTHESIS

## Config (key knobs)
n_epochs=$N_EPOCHS, save_model_freq=$SAVE_FREQ (single cosine cycle: first_cycle_steps=n_epochs,
warmup 100, min_lr 1e-5, base lr 1e-4). Authoritative resolved values: \`.hydra/config.yaml\`.

## Observations
(fill during the run)

## Final summary
(fill at end: duration, final loss, best-checkpoint eval success vs the 300-demo 0.77 baseline)
MD
        echo "[sbatch] wrote $RD/EXPERIMENT.md"
      fi
      break
    fi
    sleep 5
  done
) &

# ---- run the trainer -------------------------------------------------------------------
echo "[sbatch] pretrain: env=$DATA_ENV overrides=${OVERRIDES[*]}"
uv run --project envs/dppo_arrhenius --no-sync python -m gentle_manip.dppo.train \
    --config-path "$CFG_DIR" --config-name "$CFG_NAME" \
    env="$DATA_ENV" "${OVERRIDES[@]}" 2>&1 | tee "$TRAINER_LOG"

echo "[sbatch] pretrain done (job ${SLURM_JOB_ID:-NA})"
