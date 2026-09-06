# Shared env for the photo->mesh pipeline. Source this in every job/srun.
# Keeps big caches off $HOME (only ~24 GB free there); project storage has ~2 TB.
export PROJ=/nobackup/proj/disk/softenable-codesign26/personal/${USER}/gentle_manip
export UV_CACHE_DIR=$PROJ/.cache/uv
export HF_HOME=$PROJ/.cache/huggingface
export TORCH_HOME=$PROJ/.cache/torch
export U2NET_HOME=$PROJ/.cache/u2net          # rembg model cache
export TOKENIZERS_PARALLELISM=false
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$U2NET_HOME"
export PYTHONUNBUFFERED=1

# GPU nodes are aarch64; a per-user ~/.local/bin/uv is often an x86_64 build picked up
# on the login node. If an aarch64 uv is staged at ~/.local/uv-aarch64, prefer it here
# so `uv` resolves correctly inside GPU jobs without needing a login-node reinstall.
if [ "$(uname -m)" = "aarch64" ] && [ -x "$HOME/.local/uv-aarch64/uv" ]; then
  export PATH="$HOME/.local/uv-aarch64:$PATH"
fi
