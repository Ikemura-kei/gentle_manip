# [arrhenius] shared env bootstrap for yd_* banana-rigid-diverse jobs (source this).
# Installs an arch-local uv (the login-node uv is x86; GH200 compute nodes are aarch64)
# and exports the standard knobs. REPO is the caller's checkout on /nobackup.
set -euo pipefail
export REPO=${REPO:-/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip}
ARCH=$(uname -m)
export UV_BIN_DIR="$HOME/.local/uv-$ARCH"
if [ ! -x "$UV_BIN_DIR/uv" ]; then
  echo "[env] installing uv for $ARCH -> $UV_BIN_DIR"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$UV_BIN_DIR" UV_NO_MODIFY_PATH=1 sh
fi
export PATH="$UV_BIN_DIR:$PATH"
uv --version
cd "$REPO"
unset PYTHONPATH ROS_DISTRO 2>/dev/null || true
export MUJOCO_GL=egl
export WANDB_MODE=${WANDB_MODE:-offline}
export DPPO_DATA_DIR=${DPPO_DATA_DIR:-$REPO/dataset/dppo}
export DPPO_LOG_DIR=${DPPO_LOG_DIR:-$REPO/logs/dppo}
export GIT_COMMIT=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)
echo "[env] host=$(hostname) arch=$ARCH job=${SLURM_JOB_ID:-NA} commit=$GIT_COMMIT repo=$REPO"
