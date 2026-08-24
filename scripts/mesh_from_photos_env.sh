# Shared env for the photo->mesh pipeline. Source this in every job/srun.
# Keeps big caches off $HOME (only ~24 GB free there); project storage has ~2 TB.
export PROJ=/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip
export UV_CACHE_DIR=$PROJ/.cache/uv
export HF_HOME=$PROJ/.cache/huggingface
export TORCH_HOME=$PROJ/.cache/torch
export U2NET_HOME=$PROJ/.cache/u2net          # rembg model cache
export TOKENIZERS_PARALLELISM=false
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$U2NET_HOME"
export PYTHONUNBUFFERED=1
