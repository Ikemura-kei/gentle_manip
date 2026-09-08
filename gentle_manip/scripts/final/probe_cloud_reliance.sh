#! /bin/bash
# Is the point cloud actually used, or is the encoder decoration?
# Fixed-noise val-loss ablation on a trained checkpoint: real cloud vs a fixed cloud from another
# episode vs zeros vs within-batch shuffled. See probe_cloud_reliance.py for the reading.
#
#   RUN=logs/dppo/dppo-pretrain/single_lift_generalist_soft_v5/wiayg EPOCH=120 \
#     bash gentle_manip/scripts/final/probe_cloud_reliance.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
RUN=${RUN:?set RUN=<training run dir with .hydra/ and checkpoint/>}
EPOCH=${EPOCH:-}            # empty = the last checkpoint
BATCHES=${BATCHES:-200}; SEED=${SEED:-0}
OUT=${OUT:-${RUN}/cloud_reliance$([ -n "$EPOCH" ] && echo "_ep${EPOCH}").json}
uv run --project envs/dppo python -m gentle_manip.scripts.final.probe_cloud_reliance \
  --run "$RUN" ${EPOCH:+--epoch $EPOCH} --batches "$BATCHES" --seed "$SEED" --out "$OUT" "$@"
