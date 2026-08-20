#!/bin/bash
# [arrhenius] DP3 training on the GH200 (aarch64) nodes, via envs/dp3_arrhenius. Mirrors
# gentle_manip/scripts/train_dp3.sh's positional-arg convention but runs on this cluster's GPU
# partition. Needs the GCC 14.3 buildenv module's libstdc++ on LD_LIBRARY_PATH at RUNTIME (not
# just build time) for PyTorch3D's compiled extension to load, AND must cd into
# 3D-Diffusion-Policy before running (diffusion_policy_3d is a namespace package with no
# __init__.py -- find_packages() registers nothing, so it only resolves via cwd-on-sys.path).
#
# Submit:  ALG=dp3 TASK=<task_cfg> INFO=<tag> SEED=42 sbatch gentle_manip/scripts/arrhenius/dp3_train.sbatch \
#            training.num_epochs=6000 training.checkpoint_every=500 logging.project=<wandb-project>
#   (trailing args after the sbatch script path are forwarded as extra hydra overrides)
#
#SBATCH -J dp3_train
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
DP3_DIR="$REPO/third_party/DP3/3D-Diffusion-Policy"

GCCLIB=/software/sse2/el9_gh200/manual/GCC/14.3.0/hpc1/lib64
export LD_LIBRARY_PATH="$GCCLIB:${LD_LIBRARY_PATH:-}"

ALG=${ALG:?set ALG=<alg name, e.g. dp3>}
TASK=${TASK:?set TASK=<task config name>}
INFO=${INFO:?set INFO=<addition_info tag>}
SEED=${SEED:-42}
GPU_ID=${GPU_ID:-0}

EXP_NAME=${TASK}-${ALG}-${INFO}
RUN_DIR="data/outputs/${EXP_NAME}_seed${SEED}"

echo "[sbatch] host=$(hostname) arch=$(uname -m) job=${SLURM_JOB_ID:-NA}"
echo "[sbatch] alg=$ALG task=$TASK info=$INFO seed=$SEED exp_name=$EXP_NAME extra_overrides=$*"

cd "$DP3_DIR"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=$GPU_ID
unset PYTHONPATH ROS_DISTRO 2>/dev/null || true

uv run --project "$REPO/envs/dp3_arrhenius" --no-sync python train.py \
    --config-name="${ALG}.yaml" \
    task="$TASK" \
    hydra.run.dir="$RUN_DIR" \
    training.debug=False \
    training.seed="$SEED" \
    training.device="cuda:0" \
    exp_name="$EXP_NAME" \
    logging.mode=online \
    checkpoint.save_ckpt=True \
    "$@"
echo "[sbatch] DP3 training finished (job ${SLURM_JOB_ID:-NA})"
