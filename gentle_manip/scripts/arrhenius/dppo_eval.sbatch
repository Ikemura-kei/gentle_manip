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
#SBATCH -t 08:00:00
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
VIEW=${VIEW:-student}                           # student = point-cloud policy; teacher = privileged
                                                 # full-state policy (no cameras) -- set VIEW=teacher
                                                 # for a state-only (no point-cloud) eval config.
# Canonical EvalSpec is 5; overridable ONLY for controlled probes (e.g. the size-sweep, which
# wants one sub-env per fixed pose context). MUST be passed to the cfg's n_envs too — the
# override below does that, so the server and the client cannot disagree.
NUM_ENVS=${NUM_ENVS:-5}
# Unique per job: several checkpoint evals often run in parallel, and if two land on the SAME
# node a fixed port makes the second server fail to bind (or worse, the client attaches to the
# other job's sim). The value is arbitrary, but it MUST be given to BOTH ends: the server takes
# --port, and the eval client reads cfg env.specific.port (default 5570) — so the override
# env.specific.port="$PORT" below is REQUIRED, not optional. Without it the two only agree by
# the coincidence that both defaulted to 5570, and the client dies with "Connection refused".
PORT=${PORT:-$((5570 + ${SLURM_JOB_ID:-0} % 400))}
CKPT=${CKPT:-$REPO/logs/dppo/dppo-pretrain/2026-07-04_20-03-36_42/checkpoint/state_3000.pt}
# Config DIR is overridable: a rigid policy needs the rigid eval config (its task params differ
# from soft — e.g. max_episode_steps 400 vs 300 [rigid demos ~354 steps; the soft 300 truncates
# rollouts and undercounts success], n_episodes 200, scene_group_size 0). The network arch +
# shape_meta match across configs, so a checkpoint loads under either; the TASK params do not.
CFG_DIR=${CFG_DIR:-$REPO/gentle_manip/dppo/cfg/single_lift_mushroom_soft_pcd}
# Which eval PROTOCOL to run. The protocol (n_episodes, scene_group_size, seed) lives in a
# CONFIG FILE, not in shell flags, so an eval can't silently drift:
#   eval_diffusion_pointnet        canonical: 100 episodes, geometry rebuilt every 4 batches
#   eval_diffusion_pointnet_200ep  variance check: 200 episodes, rebuilt every 2 batches
CFG_NAME=${CFG_NAME:-eval_diffusion_pointnet}
# The policy's TRAINING normalization — MUST match the dataset the checkpoint was trained on.
# The eval cfg's default points at single_lift_mushroom_soft_pcd; override NORM for a policy
# trained on a different dataset (e.g. ..._wide1k) or the obs are mis-scaled and results are junk.
NORM=${NORM:-$REPO/dataset/dppo/single_lift_mushroom_soft_pcd/normalization.npz}

# ---- RECORD COMMANDS + OBSERVATIONS BY DEFAULT (user mandate 2026-08-28) -------------------
# Every eval dumps its commanded actions and observations unless explicitly disabled. Rationale:
# aggregate metrics (success/stress) cannot diagnose BEHAVIOUR. The arm-F approach-offset was
# only visible because a human watched the renders, and answering it needed a SECOND eval purely
# to capture obs -- wasted GPU time that a default-on dump makes unnecessary. Dumps are small
# (state+action only); the point CLOUD stays opt-in via GM_OBS_DUMP_CLOUD=1 because it is large.
# Set GM_DUMP_OFF=1 to disable.
if [ "${GM_DUMP_OFF:-0}" != "1" ]; then
    _DTAG=${EVAL_SUBDIR:-eval}_${SLURM_JOB_ID:-local}
    export GM_WIDTH_DUMP=${GM_WIDTH_DUMP:-$_DTAG}
    export GM_OBS_DUMP=${GM_OBS_DUMP:-$_DTAG}
    export GM_WIDTH_NORM=${GM_WIDTH_NORM:-$NORM}   # dumps convert with THIS eval's normalization
    echo "[sbatch] dumps ON (default): width=$GM_WIDTH_DUMP obs=$GM_OBS_DUMP cloud=${GM_OBS_DUMP_CLOUD:-0}"
fi

export WANDB_MODE=offline
export MUJOCO_GL=egl
unset PYTHONPATH ROS_DISTRO 2>/dev/null || true

echo "[sbatch] host=$(hostname) arch=$(uname -m) job=${SLURM_JOB_ID:-NA} eval ckpt=$CKPT"
[ -f "$CKPT" ] || { echo "[sbatch] MISSING checkpoint: $CKPT"; exit 1; }
[ -f "$NORM" ] || { echo "[sbatch] MISSING normalization: $NORM"; exit 1; }
# PRE-FLIGHT: NORM must be the dataset the CHECKPOINT was trained on. The default above is
# soft_pcd, which is WRONG for every other lineage, and the mismatch is SILENT — action
# denormalization is corrupted for the whole eval while the run still completes and reports a
# plausible-looking success rate. Cost us a full margin sweep on 2026-08-26. Derive the expected
# dataset from the run's own hydra config and refuse to start if it disagrees.
CKPT_RUN_CFG="$(dirname "$(dirname "$CKPT")")/.hydra/config.yaml"
if [ -f "$CKPT_RUN_CFG" ]; then
    WANT_ENV=$(grep -m1 "^env: " "$CKPT_RUN_CFG" | awk "{print \$2}")
    if [ -n "$WANT_ENV" ] && [ "$(basename "$(dirname "$NORM")")" != "$WANT_ENV" ]; then
        echo "[sbatch] NORMALIZATION MISMATCH — refusing to run."
        echo "[sbatch]   checkpoint trained on env : $WANT_ENV"
        echo "[sbatch]   NORM points at dataset    : $(basename "$(dirname "$NORM")")"
        echo "[sbatch]   fix: NORM=\$REPO/dataset/dppo/$WANT_ENV/normalization.npz"
        exit 1
    fi
    echo "[sbatch] normalization OK: $WANT_ENV"
fi

# Include EVAL_SUBDIR when set: several evals may run SEQUENTIALLY inside one job (pretrain
# auto-eval), and a single shared filename would leave only the last one's server log for
# debugging. Unique name = one server log per checkpoint.
SRV_LOG=$REPO/logs/slumr_logs/${SLURM_JOB_ID:-local}${EVAL_SUBDIR:+_${EVAL_SUBDIR}}_simserver.log

# ---- sim server (subprocess backend: eval drives deterministic per-group scene rebuilds) ----
# cfg_dir printed on its OWN line (not just cfg=$CFG_NAME below) so a resubmit (e.g. the eval
# hang-detection watchdog) can grep the EXACT config dir the dying job used, instead of assuming
# the rigid_pcd default -- CFG_DIR varies now that state-only configs exist (single_lift_mushroom_
# rigid_state), and guessing wrong makes the resubmit fail with "Cannot find primary config".
echo "[sbatch] cfg_dir=$CFG_DIR"
echo "[sbatch] view=$VIEW"
echo "[sbatch] launching sim server ($SIM_EXPERIMENT / $VIEW / $NUM_ENVS envs, --subprocess) on $PORT"
uv run --project envs/sim_arrhenius --no-sync python -m gentle_manip.scripts.serl_sim_server \
    --experiment "$SIM_EXPERIMENT" --view "$VIEW" --num-envs "$NUM_ENVS" \
    --render-rgb --subprocess --port "$PORT" > "$SRV_LOG" 2>&1 &
SRV_PID=$!
cleanup() { echo "[sbatch] stopping sim server ($SRV_PID)"; kill "$SRV_PID" 2>/dev/null || true; wait "$SRV_PID" 2>/dev/null || true; kill "${WATCHDOG_PID:-}" 2>/dev/null || true; }
trap cleanup EXIT

echo "[sbatch] waiting for SIM_SERVER_READY ($SRV_LOG)"
for _ in $(seq 1 180); do
    grep -q "SIM_SERVER_READY" "$SRV_LOG" 2>/dev/null && { echo "[sbatch] server ready"; break; }
    kill -0 "$SRV_PID" 2>/dev/null || { echo "[sbatch] server died early — tail:"; tail -50 "$SRV_LOG"; exit 1; }
    sleep 5
done
grep -q "SIM_SERVER_READY" "$SRV_LOG" || { echo "[sbatch] server not ready — tail:"; tail -50 "$SRV_LOG"; exit 1; }

# ---- sim-server WATCHDOG (2026-08-26) --------------------------------------------------
# Observed 7x: the sim server's FPS decays (e.g. 17->12, 12.6->11.3) and then the log goes
# SILENT while the job sits idle until the 8h wall clock — burning a GPU for hours and
# delaying the retry. Suspected resource growth in the long-lived genesis child under
# multi-eval contention (root cause open; sim server is the local agent's module).
# This watchdog turns that silent stall into a FAST FAILURE: if $SRV_LOG has not been
# written for GM_EVAL_WATCHDOG_MIN minutes (default 25; 0 disables), kill the job so the
# monitors see FAILED and resubmit immediately. A healthy eval writes an FPS line every
# ~40 s, so 25 min is ~35x the normal gap — no false positives expected.
GM_EVAL_WATCHDOG_MIN=${GM_EVAL_WATCHDOG_MIN:-25}
if [ "$GM_EVAL_WATCHDOG_MIN" -gt 0 ] 2>/dev/null; then
  (
    while sleep 120; do
      [ -f "$SRV_LOG" ] || continue
      if [ -n "$(find "$SRV_LOG" -mmin +$GM_EVAL_WATCHDOG_MIN 2>/dev/null)" ]; then
        echo "[sbatch] WATCHDOG: sim server silent >${GM_EVAL_WATCHDOG_MIN}min — killing job ${SLURM_JOB_ID}"
        tail -5 "$SRV_LOG"
        scancel "${SLURM_JOB_ID}" 2>/dev/null || kill -9 $$ 2>/dev/null
        break
      fi
    done
  ) &
  WATCHDOG_PID=$!
  echo "[sbatch] sim-server watchdog armed (${GM_EVAL_WATCHDOG_MIN} min silence -> job killed), pid $WATCHDOG_PID"
fi

# ---- DEGENERATE-EVAL watchdog (2026-08-27) ------------------------------------------------
# A broken width mechanism collapses to ~0 success and then burns the full 2h for a number we
# already know. Kill once at least GM_EVAL_MIN_EPISODES episodes have run and cumulative success
# is below GM_EVAL_MIN_SUCCESS. Set GM_EVAL_MIN_SUCCESS=0 to disable.
# FALSE-POSITIVE MATH: batches are fixed-seed and ~iid, so a policy whose TRUE rate is 0.3 shows
# 0/20 with probability 0.7^20 = 0.08%. A true-0.10 policy is killed ~12% of the time — acceptable,
# since a 0.10 policy is not a candidate. Never set the threshold near a rate you care about.
GM_EVAL_MIN_SUCCESS=${GM_EVAL_MIN_SUCCESS:-0.05}
GM_EVAL_MIN_EPISODES=${GM_EVAL_MIN_EPISODES:-20}
if [ "$(awk -v v="$GM_EVAL_MIN_SUCCESS" 'BEGIN{print (v>0)?1:0}')" = "1" ]; then
  JOB_OUT="${REPO}/logs/slumr_logs/${SLURM_JOB_ID}.out"
  (
    while sleep 120; do
      [ -f "$JOB_OUT" ] || continue
      read -r NB MEAN <<< "$(grep -o 'success=[0-9.]*' "$JOB_OUT" 2>/dev/null \
        | sed 's/success=//' | awk '{n++; s+=$1} END {print n+0, (n?s/n:0)}')"
      [ "${NB:-0}" -eq 0 ] && continue
      EPS=$(( NB * NUM_ENVS ))
      if [ "$EPS" -ge "$GM_EVAL_MIN_EPISODES" ] && \
         [ "$(awk -v m="$MEAN" -v t="$GM_EVAL_MIN_SUCCESS" 'BEGIN{print (m<t)?1:0}')" = "1" ]; then
        echo "[sbatch] DEGENERATE: success ${MEAN} over ${EPS} episodes < ${GM_EVAL_MIN_SUCCESS} — killing job ${SLURM_JOB_ID} early"
        scancel "${SLURM_JOB_ID}" 2>/dev/null || kill -9 $$ 2>/dev/null
        break
      fi
    done
  ) &
  echo "[sbatch] degenerate-eval watchdog armed (<${GM_EVAL_MIN_SUCCESS} after ${GM_EVAL_MIN_EPISODES} eps -> early kill), pid $!"
fi

# ---- run the canonical eval (BC base policy; record_batches=null = ONE clip PER EPISODE) ----
# Hard requirement #2: per-trajectory video, no exceptions — record_batches=null records ALL
# batches / ALL envs (100 clips), NOT a subset. (The x86 ref run used record_batches=2; that
# is a video subset only and does not change the success number.)
# N_EPISODES: full canonical eval = 100 (20 batches); set smaller (e.g. 25) for a quick diagnostic.
# UNSET = let the chosen config decide (100 for the canonical protocol, 200 for _200ep). Only
# set N_EPISODES for a deliberate short diagnostic — forcing it here would silently override
# whatever protocol CFG_NAME selected, which is exactly the drift the config files prevent.
N_EPISODES=${N_EPISODES:-}
NEP_OVR=(); [ -n "$N_EPISODES" ] && NEP_OVR=(n_episodes="$N_EPISODES")
EVAL_EXP_OVR=(); [ -n "$EVAL_EXPERIMENT" ] && EVAL_EXP_OVR=(experiment="$EVAL_EXPERIMENT")
# SCENE_GROUP_SIZE: override how often (every K batches) the harness rebuilds object geometry
# (size/shape/material) -- default per the chosen CFG_NAME (canonical = 4, i.e. every 20
# episodes at num_envs=5). Unset = use the config's own value; only override for a deliberate
# DR-density diagnostic (denser rebuild = harder/more varied geometry per episode budget).
SCENE_GROUP_SIZE=${SCENE_GROUP_SIZE:-}
SGS_OVR=(); [ -n "$SCENE_GROUP_SIZE" ] && SGS_OVR=(scene_group_size="$SCENE_GROUP_SIZE")
# ACT_STEPS: execution horizon override -- how many of the policy's predicted horizon_steps
# actions actually get executed before replanning (the model itself is unchanged; this only
# affects the eval-time receding-horizon loop). Unset = use the config's own default (4).
ACT_STEPS=${ACT_STEPS:-}
ACT_OVR=(); [ -n "$ACT_STEPS" ] && ACT_OVR=(act_steps="$ACT_STEPS")
# N_STEPS: policy-level episode length (chunk-executions per episode) -- the config's own default
# is calibrated for its default act_steps (n_steps * act_steps = max_episode_steps sim steps).
# Overriding ACT_STEPS WITHOUT also correcting N_STEPS silently shortens/lengthens the episode's
# physical time budget -- a confound, not a fair comparison. Set N_STEPS explicitly whenever
# ACT_STEPS is overridden (e.g. max_episode_steps=400, act_steps=1 -> N_STEPS=400).
N_STEPS=${N_STEPS:-}
NSTEPS_OVR=(); [ -n "$N_STEPS" ] && NSTEPS_OVR=(n_steps="$N_STEPS")
# USE_DDIM / DDIM_STEPS: switch the (already-trained, DDPM) checkpoint to DDIM sampling at eval
# time -- fewer reverse-diffusion steps than the model's own trained denoising_steps, no
# retraining needed. USE_DDIM=1 enables it; DDIM_STEPS sets the step count (config default: 0,
# i.e. ft_denoising_steps -- set DDIM_STEPS explicitly for a BC-base DDIM eval).
USE_DDIM=${USE_DDIM:-0}
DDIM_STEPS=${DDIM_STEPS:-}
DDIM_OVR=()
if [ "$USE_DDIM" = "1" ]; then
    DDIM_OVR=(use_ddim=true)
    [ -n "$DDIM_STEPS" ] && DDIM_OVR+=(ddim_steps="$DDIM_STEPS")
fi
# HORIZON_STEPS / DENOISING_STEPS / COND_STEPS: architecture-level overrides needed to load a
# checkpoint trained with a NON-DEFAULT value of these (horizon_steps/cond_steps change the
# network's input/output dims -- a mismatch fails state_dict loading loudly; denoising_steps
# changes the noise schedule length). Unset = use the config's own default (must match the
# checkpoint's own training config or loading fails/produces garbage).
HORIZON_STEPS=${HORIZON_STEPS:-}
HORIZON_OVR=(); [ -n "$HORIZON_STEPS" ] && HORIZON_OVR=(horizon_steps="$HORIZON_STEPS")
DENOISING_STEPS_OVR_VAL=${DENOISING_STEPS:-}
DENOISING_OVR=(); [ -n "$DENOISING_STEPS_OVR_VAL" ] && DENOISING_OVR=(denoising_steps="$DENOISING_STEPS_OVR_VAL")
COND_STEPS=${COND_STEPS:-}
COND_OVR=(); [ -n "$COND_STEPS" ] && COND_OVR=(cond_steps="$COND_STEPS")
# SMOOTH_ALPHA: EMA low-pass on the commanded pos+rot6d (jerk suppression), same mechanism as
# deploy_real.py's --smooth-alpha (GenesisMultiStepVecEnv._smooth). Unset (default) = off; the
# target eval config must declare env.specific.smooth_alpha (currently: the *_bignet/*_bignet2
# cfg dirs) for a plain override to apply -- otherwise use SMOOTH_ALPHA_NEW=1 to add the key.
SMOOTH_ALPHA=${SMOOTH_ALPHA:-}
SMOOTH_ALPHA_NEW=${SMOOTH_ALPHA_NEW:-0}
SMOOTH_OVR=()
if [ -n "$SMOOTH_ALPHA" ]; then
    if [ "$SMOOTH_ALPHA_NEW" = "1" ]; then
        SMOOTH_OVR=(+env.specific.smooth_alpha="$SMOOTH_ALPHA")
    else
        SMOOTH_OVR=(env.specific.smooth_alpha="$SMOOTH_ALPHA")
    fi
fi
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
# GM_EXTRA_OVERRIDES: arbitrary extra hydra overrides (space-separated key=val), for anything not
# covered by a named knob above (e.g. action_dim=7, env_name=..., for an action-space-ablation
# arm whose eval config needs values the default cfg doesn't have). Appended LAST so it wins.
EXTRA_OVR=()
if [ -n "${GM_EXTRA_OVERRIDES:-}" ]; then
    # shellcheck disable=SC2206  (intentional word-split into array elements)
    EXTRA_OVR=($GM_EXTRA_OVERRIDES)
fi
echo "[sbatch] running eval of $CKPT (cfg=$CFG_NAME, n_episodes=${N_EPISODES:-<from cfg>}, ft_denoising=$FT_DENOISING, smooth_alpha=${SMOOTH_ALPHA:-off}, scene_group_size=${SCENE_GROUP_SIZE:-<from cfg>}, act_steps=${ACT_STEPS:-<from cfg>} n_steps=${N_STEPS:-<from cfg>}, use_ddim=$USE_DDIM ddim_steps=${DDIM_STEPS:-<from cfg>}, horizon_steps=${HORIZON_STEPS:-<from cfg>} denoising_steps=${DENOISING_STEPS_OVR_VAL:-<from cfg>} cond_steps=${COND_STEPS:-<from cfg>}, ${EVAL_EXP_OVR[*]:-exp=cfg-default}, extra=${GM_EXTRA_OVERRIDES:-<none>})"
uv run --project envs/dppo_arrhenius --no-sync python -m gentle_manip.dppo.train \
    --config-path "$CFG_DIR" --config-name "$CFG_NAME" \
    base_policy_path="$CKPT" normalization_path="$NORM" \
    ft_denoising_steps="$FT_DENOISING" record_batches=null "${NEP_OVR[@]}" \
    env.specific.port="$PORT" \
    "${EVAL_EXP_OVR[@]}" "${LOGDIR_OVR[@]}" "${SMOOTH_OVR[@]}" "${SGS_OVR[@]}" "${ACT_OVR[@]}" "${DDIM_OVR[@]}" "${NSTEPS_OVR[@]}" \
    "${HORIZON_OVR[@]}" "${DENOISING_OVR[@]}" "${COND_OVR[@]}" "${EXTRA_OVR[@]}"

echo "[sbatch] eval finished OK"
