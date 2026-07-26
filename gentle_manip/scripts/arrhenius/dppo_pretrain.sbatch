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
# Config dir/name are overridable so this launcher works for other datasets/tasks (e.g. the
# real-world config under .../single_lift_mushroom_real). DATA_ENV must still match the config's
# `env:` (it resolves the dataset path + logdir); pass both together.
CFG_DIR=${CFG_DIR:-$REPO/gentle_manip/dppo/cfg/single_lift_mushroom_soft_pcd}
CFG_NAME=${CFG_NAME:-pre_diffusion_pointnet}

# The pretrain config resolves the dataset path + logdir from these (${oc.env:...}); export them
# so the job is self-contained regardless of the login-shell profile.
export DPPO_DATA_DIR=${DPPO_DATA_DIR:-$REPO/dataset/dppo}
export DPPO_LOG_DIR=${DPPO_LOG_DIR:-$REPO/logs/dppo}
export WANDB_MODE=${WANDB_MODE:-online}  # GH200 compute nodes have outbound internet (auth ~/.netrc)
unset PYTHONPATH ROS_DISTRO 2>/dev/null || true

# ---- preflight -------------------------------------------------------------------------
echo "[sbatch] host=$(hostname) arch=$(uname -m) job=${SLURM_JOB_ID:-NA} smoke=$SMOKE data_env=$DATA_ENV"
TRAIN_NPZ=$DPPO_DATA_DIR/$DATA_ENV/train.npz
VAL_NPZ=$DPPO_DATA_DIR/$DATA_ENV/val.npz
[ -f "$TRAIN_NPZ" ] || { echo "[sbatch] MISSING dataset: $TRAIN_NPZ (upload dataset/dppo/$DATA_ENV/ first)"; exit 1; }
# The config now wires a held-out val_dataset (-> "loss - val"); catch a missing split HERE rather
# than 2 minutes in on a hydra instantiate error. Pass GM_EXTRA_OVERRIDES="val_dataset=null" to
# train without validation on a dataset that has no val.npz.
if [ ! -f "$VAL_NPZ" ] && [[ "${GM_EXTRA_OVERRIDES:-}" != *"val_dataset=null"* ]]; then
    echo "[sbatch] MISSING val split: $VAL_NPZ"
    echo "[sbatch]   -> add val.npz, or pass GM_EXTRA_OVERRIDES=\"val_dataset=null\" to skip validation"
    exit 1
fi

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
set +e
uv run --project envs/dppo_arrhenius --no-sync python -m gentle_manip.dppo.train \
    --config-path "$CFG_DIR" --config-name "$CFG_NAME" \
    env="$DATA_ENV" "${OVERRIDES[@]}" 2>&1 | tee "$TRAINER_LOG"
TRAIN_RC=${PIPESTATUS[0]}          # rc of the TRAINER, not of tee
set -e
echo "[sbatch] pretrain done (job ${SLURM_JOB_ID:-NA}) rc=$TRAIN_RC"

# ---- optional: auto-eval EVERY checkpoint (GM_AUTO_EVAL=1) ------------------------------
# Runs the evals INLINE (bash the eval script) rather than sbatch-ing one job per checkpoint:
# this job already holds a GPU node, and on a busy partition re-queueing costs hours per
# checkpoint. Cost: this job's walltime must cover training + ~1.5h x n_checkpoints.
# Each eval uses THIS dataset's normalization (the checkpoint was trained on it; the wrong one
# silently mis-scales the obs) and the eval EXPERIMENT below — which MUST match the policy's task:
#   AUTO_EVAL_EXP (default single_lift_mushroom_soft_eval, the soft 235/250 reference). Set it to
#   the matching task's eval experiment for a non-soft policy, e.g. single_lift_mushroom_rigid_eval
#   for a rigid-trained policy (rigid object, substeps 5, rigid_orientation DR, superset_rigid obs).
#   Evaluating a rigid policy on the SOFT experiment is a task mismatch (deformable object / wrong
#   obs+DR) and produces meaningless numbers. AUTO_EVAL_TAG names the eval subdir (default eval235).
AUTO_EVAL_EXP=${AUTO_EVAL_EXP:-single_lift_mushroom_soft_eval}
AUTO_EVAL_TAG=${AUTO_EVAL_TAG:-eval235}
if [ "${GM_AUTO_EVAL:-0}" = "1" ] && [ "$TRAIN_RC" = "0" ] && [ "$SMOKE" != "1" ]; then
    RD=$(grep -oE '/nobackup/[^ ]+/dppo-pretrain/[^ ]+/config/' "$TRAINER_LOG" | head -1 | sed 's#/config/##')
    NORM_AE=$DPPO_DATA_DIR/$DATA_ENV/normalization.npz
    if [ -z "$RD" ] || [ ! -d "$RD" ]; then
        echo "[sbatch] AUTO-EVAL SKIPPED: could not locate the run dir in $TRAINER_LOG"
    else
        CKPTS=$(ls -1v "$RD"/checkpoint/state_*.pt 2>/dev/null)
        echo "[sbatch] auto-eval: $(echo "$CKPTS" | grep -c .) checkpoint(s) from $RD"
        AE_I=0
        for CK in $CKPTS; do
            CKN=$(basename "$CK" .pt)
            # Distinct port per eval: these run back-to-back in ONE job, so they would all
            # derive the same port from SLURM_JOB_ID and could collide on a socket still in
            # TIME_WAIT from the previous checkpoint's server.
            AE_I=$((AE_I + 1))
            export PORT=$((5570 + (${SLURM_JOB_ID:-0} + AE_I * 7) % 400))
            echo "[sbatch] === auto-eval $CKN ==="
            # subshell + `|| true`: one failed checkpoint eval must not abort the rest
            ( CKPT="$CK" NORM="$NORM_AE" FT_DENOISING=0 N_EPISODES="${AUTO_EVAL_EPISODES:-100}" \
              EVAL_SUBDIR="${CKN}_${AUTO_EVAL_TAG}" \
              SIM_EXPERIMENT="$AUTO_EVAL_EXP" \
              EVAL_EXPERIMENT="$AUTO_EVAL_EXP" \
              bash "$REPO/gentle_manip/scripts/arrhenius/dppo_eval.sbatch" ) || \
                echo "[sbatch] auto-eval FAILED for $CKN (continuing)"
        done
        echo "[sbatch] auto-eval finished; summaries under $RD/eval/state_*_eval235/"
    fi
fi
exit $TRAIN_RC
