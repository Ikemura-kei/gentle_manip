#!/bin/bash
# 12 sim slices + the 7 REAL objects from real_paired_7obj_2026-09-01 -> generalist_12obj_real7
#
# REAL SOURCE IS PINNED (user, 2026-09-01): ONLY dataset/transfer/real_paired_7obj_2026-09-01.
# No other real data may enter -- notably NOT single_lift_mushroom_real_merged_shift9mm, which the
# separate `znmbh` run uses. The two must not be confused.
# These demos are ALREADY bias-corrected: collected with point_cloud_shift [0.009,0,0] ACTIVE
# (per the bundle README), so their clouds are in the same frame as the unbiased sim clouds.
# Their actions are ALREADY 7-dim euler-absolute -> NO --derive-action (unlike the sim slices,
# whose collector records 10-dim rot6d).
#SBATCH -J g12r7_build
#SBATCH -N 1
#SBATCH -t 10:00:00
#SBATCH -A naiss2026-3-141-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --output=/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip/logs/slumr_logs/%j.out
#SBATCH --error=/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip/logs/slumr_logs/%j.err
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip
cd $R
OUT=$R/dataset/dppo/single_lift_generalist_12obj_real7
PIN=$R/dataset/transfer/real_paired_7obj_2026-09-01
mkdir -p $OUT

# --- guard: every real slice must live under the PINNED dir -------------------------------
while read -r NAME DIR; do
  case "$DIR" in "$PIN"/*) ;; *) echo "REFUSING: real slice $NAME outside the pin: $DIR"; exit 1;; esac
  [ -f "$DIR/data.pkl" ] || { echo "MISSING $DIR/data.pkl"; exit 1; }
done < $R/.agent_tmp/real7_slices.txt
echo "[build] real-source pin OK: all 7 under $PIN"
# --- sim slices: the same provenance guard as the sim-only generalist ----------------------
uv run --project envs/sim_arrhenius --no-sync python .agent_tmp/verify_round_provenance.py --merge-inputs || exit 1

SLICES=""
while read -r NAME DIR; do SLICES="$SLICES $R/dataset/dppo/g12_$NAME"; done < $R/.agent_tmp/g12_slices.txt
while read -r NAME DIR; do
  S=$R/dataset/dppo/g12r_$NAME
  if [ ! -f "$S/train.npz" ]; then
    echo "[build] converting REAL $NAME <- $DIR"
    uv run --project envs/dppo_arrhenius --no-sync python -m gentle_manip.dppo.convert_demos \
      "$DIR/data.pkl" --out "$S" --obs-keys ee_pos ee_quat gripper_width --point-cloud
  fi
  SLICES="$SLICES $S"
done < $R/.agent_tmp/real7_slices.txt

echo "[build] merging 12 sim + 7 real -> $OUT"
uv run --project envs/dppo_arrhenius --no-sync python -m gentle_manip.dppo.merge_npz_datasets $SLICES --out $OUT

uv run --project envs/sim_arrhenius --no-sync python - <<'PY'
import numpy as np
o="dataset/dppo/single_lift_generalist_12obj_real7"; s="dataset/dppo/single_lift_generalist_12obj"
m=np.load(f"{o}/normalization.npz")
for sp in ("train","val"):
    d=np.load(f"{o}/{sp}.npz"); a=d["actions"]
    print(f"[verify] {sp}: {len(d['traj_lengths'])} trajs, {a.shape[0]} steps, dim={a.shape[1]}, "
          f"range [{a.min():+.4f},{a.max():+.4f}]")
    assert a.shape[1]==7 and a.min()>=-1.001 and a.max()<=1.001
t=np.load(f"{o}/train.npz")["actions"]
assert t.min()<-0.98 and t.max()>0.98, "not jointly renormalized"
old=np.load(f"{s}/normalization.npz")
moved=(not np.allclose(old["action_min"],m["action_min"])) or (not np.allclose(old["action_max"],m["action_max"]))
print(f"[verify] normalization DIFFERS from sim-only: {moved}")
assert moved, "normalization identical to sim-only -- real data did not reach the joint stats"
print("[verify] PASS: 7-dim, JOINT normalization recomputed AFTER merging sim+real")
PY
echo "[build] DONE"
