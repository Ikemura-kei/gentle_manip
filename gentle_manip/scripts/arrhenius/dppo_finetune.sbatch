#!/bin/bash
# [arrhenius] DPPO point-cloud finetune on the GH200 (aarch64) nodes.
# Two processes on ONE GPU: a genesis serl_sim_server (envs/sim_arrhenius) + the DPPO
# trainer (envs/dppo_arrhenius) talking over the rpc socket (127.0.0.1:PORT). This script
# starts the server, waits for SIM_SERVER_READY, runs the trainer, and kills the server on exit.
#
# Submit:  sbatch gentle_manip/scripts/arrhenius/dppo_finetune.sbatch
# Smoke test (short) is the default; set SMOKE=0 in the env / edit below for a full run.
#
#SBATCH -J dppo_ft
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
SMOKE=${SMOKE:-1}                     # 1 = short pipeline check; 0 = full run (lqitl params)
# Sim server serves the reward, so its experiment MUST be the nostress one (reward flows from the
# server: genesis_venv sums the server's r; the trainer does not recompute it). lqitl's trainer
# config experiment == single_lift_mushroom_soft_nostress, so the server serves the same.
SIM_EXPERIMENT=${SIM_EXPERIMENT:-single_lift_mushroom_soft_nostress}
VIEW=student
NUM_ENVS=${NUM_ENVS:-12}               # server --num-envs; MUST match cfg env.n_envs (pass
                                       # train.n_envs=$NUM_ENVS in GM_EXTRA_OVERRIDES) and
                                       # logprob_batch_size must be divisible by it
PORT=5570                              # MUST match cfg env.specific.port
CKPT=$REPO/logs/dppo/dppo-pretrain/2026-07-04_20-03-36_42/checkpoint/state_3000.pt
CFG_DIR=$REPO/gentle_manip/dppo/cfg/single_lift_mushroom_soft_pcd
CFG_NAME=ft_ppo_diffusion_pointnet_modex_nostress
NORM=$REPO/dataset/dppo/single_lift_mushroom_soft_pcd/normalization.npz

export WANDB_MODE=${WANDB_MODE:-online}  # GH200 compute nodes DO have outbound internet (verified);
                                         # auth via ~/.netrc (api.wandb.ai). Override =offline to buffer.
export MUJOCO_GL=egl                    # headless genesis render
unset PYTHONPATH ROS_DISTRO 2>/dev/null || true

# ---- preflight -------------------------------------------------------------------------
echo "[sbatch] host=$(hostname) arch=$(uname -m) job=${SLURM_JOB_ID:-NA} smoke=$SMOKE"
[ -f "$CKPT" ] || { echo "[sbatch] MISSING checkpoint: $CKPT"; exit 1; }
[ -f "$NORM" ] || { echo "[sbatch] MISSING normalization: $NORM (copy dataset/dppo/... first)"; exit 1; }

SRV_LOG=$REPO/logs/slumr_logs/${SLURM_JOB_ID:-local}_simserver.log

# ---- start the sim server (background) -------------------------------------------------
echo "[sbatch] launching sim server ($SIM_EXPERIMENT / $VIEW / $NUM_ENVS envs) on port $PORT"
uv run --project envs/sim_arrhenius --no-sync python -m gentle_manip.scripts.serl_sim_server \
    --experiment "$SIM_EXPERIMENT" --view "$VIEW" --num-envs "$NUM_ENVS" \
    --render-rgb --subprocess --port "$PORT" > "$SRV_LOG" 2>&1 &
SRV_PID=$!
cleanup() { echo "[sbatch] stopping sim server ($SRV_PID)"; kill "$SRV_PID" 2>/dev/null || true; wait "$SRV_PID" 2>/dev/null || true; }
trap cleanup EXIT

# ---- wait for readiness (server prints SIM_SERVER_READY once its port is bound) ---------
echo "[sbatch] waiting for SIM_SERVER_READY (server log: $SRV_LOG)"
for _ in $(seq 1 180); do                          # up to ~15 min (first genesis build compiles kernels)
    grep -q "SIM_SERVER_READY" "$SRV_LOG" 2>/dev/null && { echo "[sbatch] server ready"; break; }
    kill -0 "$SRV_PID" 2>/dev/null || { echo "[sbatch] server died early — tail:"; tail -50 "$SRV_LOG"; exit 1; }
    sleep 5
done
grep -q "SIM_SERVER_READY" "$SRV_LOG" || { echo "[sbatch] server not ready in time — tail:"; tail -50 "$SRV_LOG"; exit 1; }

# ---- run the trainer -------------------------------------------------------------------
if [ "$SMOKE" = "1" ]; then
    # short: exercise rollout(socket) + grad update + periodic eval(scene rebuild+video) + ckpt I/O
    OVERRIDES=(train.n_train_itr=2 train.n_steps=20 train.val_freq=2
               train.save_model_freq=2 train.render.freq=2 train.eval_n_batches=1)
else
    # full run == lqitl's EXACT resolved params (from its .hydra/config.yaml): only actor_lr +
    # min_lr differ from the config defaults; target_kl(0.03)/update_epochs(1)/n_train_itr(250)
    # already match. (The EXPERIMENT.md's 0.05/2 were aspirational notes, NOT what lqitl ran.)
    OVERRIDES=(train.n_train_itr=250 train.actor_lr=2e-5 train.actor_lr_scheduler.min_lr=2e-6)
fi

# In-training eval: OFF by default (force_train) — the 12-env/5-batch periodic eval correlates
# poorly with the harness eval and just burns wall time; ckpt saving is INDEPENDENT and kept.
# Set GM_INTRAIN_EVAL=1 to re-enable the periodic eval. NOTE: force_train is read via
# cfg.train.get("force_train", False), so it is NOT a declared config key -> it must be ADDED
# with a '+' prefix (a plain `train.force_train=true` errors "not in struct" under hydra's
# struct mode). Do NOT also pass +train.force_train in GM_EXTRA_OVERRIDES (double-add errors).
if [ "${GM_INTRAIN_EVAL:-0}" != "1" ]; then
    OVERRIDES+=(+train.force_train=true train.render.freq=99999)
fi

# Per-run extra hydra overrides (space-separated key=val), appended so they WIN over the base
# (hydra: last value wins). e.g. GM_EXTRA_OVERRIDES="train.target_kl=1 train.batch_size=256".
if [ -n "${GM_EXTRA_OVERRIDES:-}" ]; then
    # shellcheck disable=SC2206  (intentional word-split into array elements)
    OVERRIDES+=($GM_EXTRA_OVERRIDES)
fi

TRAINER_LOG=$REPO/logs/slumr_logs/${SLURM_JOB_ID:-local}_trainer.log
: > "$TRAINER_LOG"
GIT_COMMIT=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)

# Per-run Motivation/Hypothesis for EXPERIMENT.md — set GM_MOTIVATION / GM_HYPOTHESIS to describe
# what THIS run actually tests (avoids a stale hardcoded narrative). Generic accurate defaults:
: "${GM_MOTIVATION:=DPPO nostress point-cloud finetune on the aarch64 GH200 cluster. sim_substeps=${GM_SIM_SUBSTEPS:-220}, sampler=${GM_MPM_SAMPLER:-default}, base = lqitl config + the Extra hydra overrides above (if any). Exact resolved params: the Config section below + .hydra/config.yaml.}"
: "${GM_HYPOTHESIS:=Runs to 250 iters without a rigid-solver NaN; success climbs from the BC baseline (~0.69). (Set GM_HYPOTHESIS to state the specific thing this run probes.)}"

# ---- EXPERIMENT.md writer (background): DPPO doesn't write one; do it once the run dir appears
# (the ExperimentSnapshot callback prints "... -> <run_dir>/config/" at launch). HARD REQ #6.
(
  for _ in $(seq 1 240); do
    RD=$(grep -oE '/nobackup/[^ ]+/dppo-finetune/[^ ]+/config/' "$TRAINER_LOG" 2>/dev/null | head -1 | sed 's#/config/##' || true)  # || true: no-match must not trip set -e in this subshell
    if [ -n "$RD" ] && [ -d "$RD" ]; then
      if [ ! -f "$RD/EXPERIMENT.md" ]; then
        cat > "$RD/EXPERIMENT.md" <<MD
# Experiment: $(basename "$RD")

- **Algorithm:** dppo-finetune
- **Started:** $(date '+%Y-%m-%d %H:%M:%S')
- **Git commit:** \`$GIT_COMMIT\`
- **Run dir:** \`$RD\`
- **SLURM job:** ${SLURM_JOB_ID:-NA} on $(hostname)
- **wandb:** project gentle-manip-single_lift_mushroom_soft_pcd, run $(basename "$RD") (mode $WANDB_MODE)
- **Extra hydra overrides (on top of the lqitl base below):** \`${GM_EXTRA_OVERRIDES:-<none>}\`

## Motivation
$GM_MOTIVATION

## Hypothesis
$GM_HYPOTHESIS

## Config (key knobs)
Trainer hyperparameters (actor_lr, target_kl, clip, batch_size, update_epochs, n_steps, ...):
the AUTHORITATIVE resolved values are in \`.hydra/config.yaml\`; per-run deviations from the
lqitl base are the **Extra hydra overrides** line above. Sim/env knobs (NOT in .hydra):
\`\`\`yaml
base_policy: state_3000.pt (dppo-pretrain/2026-07-04_20-03-36_42)
experiment: single_lift_mushroom_soft_nostress
n_envs: 12
sim_substeps: ${GM_SIM_SUBSTEPS:-220}   # GM_SIM_SUBSTEPS env override (GH200 grasp-stability)
sampler: ${GM_MPM_SAMPLER:-default}     # GM_MPM_SAMPLER env override; default pbs(x86)/random(aarch64)
# lqitl base (UNLESS changed by the Extra overrides above — .hydra/config.yaml is truth):
#   actor_lr 2e-5 (min 2e-6), target_kl 0.03, update_epochs 1, clip 0.005/0.0005, n_steps 75, n_train_itr 250
\`\`\`

## Observations (append during/after — agent + user)
-

## Final summary (fill when the run ends)
_Record: duration, learner steps, final success/stress, verdict._
-
MD
        echo "[sbatch] wrote $RD/EXPERIMENT.md"
      fi
      # Make the config snapshot self-accurate: GM_* overrides are applied at genesis
      # build time and NOT written back by ExperimentSnapshot, so the snapshotted task
      # YAML shows the SOURCE sim_substeps (misleading). Patch it + drop gm_overrides.yaml.
      if [ -n "${GM_SIM_SUBSTEPS:-}${GM_MPM_SAMPLER:-}" ] && [ -d "$RD/config" ]; then
        if [ -n "${GM_SIM_SUBSTEPS:-}" ]; then
          sed -i -E "s|^sim_substeps:.*|sim_substeps: ${GM_SIM_SUBSTEPS}   # EFFECTIVE via GM_SIM_SUBSTEPS override|" "$RD"/config/tasks__*.yaml 2>/dev/null || true
        fi
        cat > "$RD/config/gm_overrides.yaml" <<OV
# Runtime overrides applied in scene_builder via env vars — what the sim ACTUALLY used.
GM_SIM_SUBSTEPS: ${GM_SIM_SUBSTEPS:-<unset, used task default>}
GM_MPM_SAMPLER: ${GM_MPM_SAMPLER:-<unset, genesis default: pbs on x86 / random on aarch64>}
OV
        echo "[sbatch] recorded GM overrides in $RD/config/"
      fi
      break
    fi
    sleep 5
  done
) &

echo "[sbatch] launching DPPO finetune (SMOKE=$SMOKE): ${OVERRIDES[*]}"
uv run --project envs/dppo_arrhenius --no-sync python -m gentle_manip.dppo.train \
    --config-path "$CFG_DIR" --config-name "$CFG_NAME" \
    base_policy_path="$CKPT" \
    "${OVERRIDES[@]}" 2>&1 | tee "$TRAINER_LOG"
TRAIN_RC=${PIPESTATUS[0]}
echo "[sbatch] finetune finished (rc=$TRAIN_RC)"

# ---- optional appended CANONICAL eval (GM_AUTO_EVAL=1) — no external monitor needed ---------
# After a SUCCESSFUL full run, evaluate the final checkpoint IN THIS JOB by running the tested
# dppo_eval.sbatch as a child. It DEFAULTS to the reference eval sim (single_lift_mushroom_soft_eval
# = 235/250), NOT the training sim, so a coarse-trained run is still measured at full fidelity.
# GM_SIM_SUBSTEPS is UNSET for the eval (the eval task pins the fidelity); GM_MPM_SAMPLER stays regular.
if [ "${GM_AUTO_EVAL:-0}" = "1" ] && [ "$TRAIN_RC" = "0" ] && [ "$SMOKE" != "1" ]; then
    RD=$(grep -oE '/nobackup/[^ ]+/dppo-finetune/[^ ]+/config/' "$TRAINER_LOG" 2>/dev/null | head -1 | sed 's#/config/##')
    LAST=$(ls -1v "$RD"/checkpoint/state_*.pt 2>/dev/null | tail -1)
    if [ -n "$RD" ] && [ -f "$LAST" ]; then
        CKN=$(basename "$LAST" .pt)
        echo "[sbatch] AUTO-EVAL: canonical eval of $LAST on the reference sim (235/250)"
        cleanup                       # stop the training sim server: free port $PORT + GPU
        trap - EXIT                   # the child dppo_eval.sbatch installs its own cleanup trap
        # FORCE the reference eval config: the training SIM_EXPERIMENT is exported into this job's
        # env (sbatch --export=ALL), so the child eval's `${SIM_EXPERIMENT:-...eval}` default would
        # NOT fire -> it would (wrongly) run the eval on the TRAINING sim. Set both explicitly so a
        # coarse-trained run is still evaluated on 235/250. (GM_SIM_SUBSTEPS unset -> eval task pins 235.)
        env -u GM_SIM_SUBSTEPS \
            SIM_EXPERIMENT=single_lift_mushroom_soft_eval \
            EVAL_EXPERIMENT=single_lift_mushroom_soft_eval \
            FT_DENOISING=10 CKPT="$LAST" N_EPISODES=100 EVAL_SUBDIR="${CKN}_eval235" \
            SLURM_JOB_ID="${SLURM_JOB_ID:-local}e" \
            bash "$REPO/gentle_manip/scripts/arrhenius/dppo_eval.sbatch" \
          && echo "[sbatch] AUTO-EVAL done -> $RD/eval/${CKN}_eval235" \
          || echo "[sbatch] AUTO-EVAL failed (training + checkpoints are intact)"
    else
        echo "[sbatch] AUTO-EVAL: no run dir / final checkpoint found; skipped"
    fi
fi
exit $TRAIN_RC
