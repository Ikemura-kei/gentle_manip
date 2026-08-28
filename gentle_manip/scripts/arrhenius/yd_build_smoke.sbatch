#!/bin/bash
# [arrhenius] Build envs/sim_arrhenius + envs/dppo_arrhenius on a GH200 node, then
# smoke-test the WHOLE rigid-banana diverse-start pipeline end to end (4-episode
# collect w/ video -> convert -> 2-epoch BC train). PASS/FAIL printed at the end.
# Submit:  sbatch gentle_manip/scripts/arrhenius/yd_build_smoke.sbatch
#SBATCH -J yd_build
#SBATCH -N 1
#SBATCH -t 05:00:00
#SBATCH -A naiss2026-3-141-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip/logs/slurm_logs/%j.out
#SBATCH --error=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip/logs/slurm_logs/%j.err
set -euo pipefail
REPO=/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip
source "$REPO/gentle_manip/scripts/arrhenius/yd_env_common.sh"

FAIL=0
step() { echo; echo "========== $* =========="; }

step "uv sync envs/sim_arrhenius"
uv sync --project envs/sim_arrhenius

step "check gstaichi / genesis / torch-cuda (sim_arrhenius)"
if ! uv run --project envs/sim_arrhenius --no-sync python -c "import gstaichi" 2>/dev/null; then
  echo "[build] gstaichi missing -> uv pip install gstaichi"
  uv pip install --python envs/sim_arrhenius/.venv/bin/python gstaichi
fi
uv run --project envs/sim_arrhenius --no-sync python - <<'PY' || FAIL=1
import torch, gstaichi
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available(),
      "dev", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None))
import genesis  # noqa
print("genesis import OK")
PY

step "uv sync envs/dppo_arrhenius"
uv sync --project envs/dppo_arrhenius
uv run --project envs/dppo_arrhenius --no-sync python -c "import torch, agent, model, util; print('dppo torch', torch.__version__, torch.cuda.is_available(), '| dppo pkgs OK')" || FAIL=1

if [ "$FAIL" != 0 ]; then echo; echo "SMOKE RESULT: FAIL (env build)"; exit 1; fi

step "smoke: 4-episode diverse-start collection (rigid banana) + video"
SMOKE_DEMOS=$REPO/dataset/demos_smoke
rm -rf "$SMOKE_DEMOS"
env -u PYTHONPATH -u ROS_DISTRO MUJOCO_GL=egl uv run --project envs/sim_arrhenius --no-sync python \
  grasp_synthesis/collect_demos_diverse_start_v2.py \
  --experiment single_lift_banana_rigid_diverse \
  --n-episodes 4 --n-envs 2 --maxfevals 150 --scene-dr-every 1 --seed 1 \
  --record-video --out-dir "$SMOKE_DEMOS" || FAIL=1
RUN=$(ls -dt "$SMOKE_DEMOS"/single_lift_banana_rigid/* 2>/dev/null | head -1 || true)
echo "[smoke] collection run dir: $RUN"
ls -la "$RUN" "$RUN/videos" 2>/dev/null || true
[ -f "$RUN/data.pkl" ] || { echo "[smoke] no data.pkl"; FAIL=1; }
NVID=$(ls "$RUN/videos"/*.mp4 2>/dev/null | wc -l || echo 0); echo "[smoke] videos: $NVID"
[ "$NVID" -ge 1 ] || FAIL=1

step "smoke: convert -> dppo npz"
rm -rf "$DPPO_DATA_DIR/smoke_banana_rigid"
uv run --project envs/dppo_arrhenius --no-sync python -m gentle_manip.dppo.convert_demos \
  "$RUN" --out "$DPPO_DATA_DIR/smoke_banana_rigid" \
  --experiment single_lift_banana_rigid_diverse --view student --point-cloud --val-split 0.25 || FAIL=1
ls -la "$DPPO_DATA_DIR/smoke_banana_rigid" 2>/dev/null || true

step "smoke: 2-epoch BC pretrain"
uv run --project envs/dppo_arrhenius --no-sync python -m gentle_manip.dppo.train \
  --config-path "$REPO/gentle_manip/dppo/cfg/single_lift_banana_rigid_diverse_pcd" \
  --config-name pre_diffusion_pointnet \
  env=smoke_banana_rigid train.n_epochs=2 train.save_model_freq=1 train.val_freq=1 \
  train.early_stop_patience=999 wandb=null || FAIL=1

echo
if [ "$FAIL" = 0 ]; then echo "SMOKE RESULT: PASS"; else echo "SMOKE RESULT: FAIL"; exit 1; fi
