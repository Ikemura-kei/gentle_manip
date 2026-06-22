#!/bin/bash
# uv wrapper for DP3 training — mirrors third_party/DP3/scripts/train_policy.sh,
# but runs train.py in the envs/dp3 (Python 3.8) uv env instead of an activated
# conda env. Same positional args as the original; runnable from anywhere.
#
#   bash gentle_manip/scripts/train_dp3.sh <alg> <task> <addition_info> <seed> <gpu_id>
#   e.g. bash gentle_manip/scripts/train_dp3.sh dp3 real_xarm7_red_cube 0112 0 0
set -e

# Resolve repo root from this script's location: <repo>/gentle_manip/scripts/.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../.." && pwd)
DP3_DIR="$REPO/third_party/DP3/3D-Diffusion-Policy"

DEBUG=False
save_ckpt=True

alg_name=${1}
task_name=${2}
config_name=${alg_name}
addition_info=${3}
seed=${4}
gpu_id=${5}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

if [ "$DEBUG" = True ]; then wandb_mode=offline; else wandb_mode=online; fi
echo -e "\033[33mgpu id: ${gpu_id}   exp: ${exp_name}\033[0m"

# train.py uses hydra config paths + data/ + outputs relative to the package dir,
# so cwd must be 3D-Diffusion-Policy. --project keeps cwd here while using envs/dp3.
cd "$DP3_DIR"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

uv run --project "$REPO/envs/dp3" python train.py \
    --config-name=${config_name}.yaml \
    task=${task_name} \
    hydra.run.dir=${run_dir} \
    training.debug=$DEBUG \
    training.seed=${seed} \
    training.device="cuda:0" \
    exp_name=${exp_name} \
    logging.mode=${wandb_mode} \
    checkpoint.save_ckpt=${save_ckpt}
