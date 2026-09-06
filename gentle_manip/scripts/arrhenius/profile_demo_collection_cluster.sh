#!/bin/bash
# Cluster adaptation of gentle_manip/scripts/final/profile_demo_collection.sh: identical collector
# flags, but ONE SLURM JOB PER OBJECT (parallel) instead of a sequential local loop, aggregated
# afterwards. First used 2026-09-05/06 (stamp 260905-2055) — results + caveats in DEVLOG.
#   bash gentle_manip/scripts/arrhenius/profile_demo_collection_cluster.sh [objects...]
# Aggregate when all jobs finish (login node is fine):
#   python3 gentle_manip/scripts/arrhenius/aggregate_profile.py logs/profile_demo_collection/<stamp>
# Prereqs on this cluster: envs/sim_arrhenius synced + fix_pymeshlab_gh200.sh applied (coarse-CAD
# meshes crash at the isotropic remesh without it).
set -euo pipefail
R=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$R"
STAMP=$(date +%y%m%d-%H%M)
OBJS=${@:-"tofu strawberry banana_chunk mushroom raspberry prim_cylinder_mush tomato cherry_tomato"}
echo "profile run -> logs/profile_demo_collection/$STAMP"
for obj in $OBJS; do
  JID=$(sbatch --parsable -J prof_$obj -N 1 -t 3:00:00 -A naiss2026-3-141-gpu -p gpu --gres=gpu:1 --mem=0 \
    --output=$R/logs/slumr_logs/%j.out --error=$R/logs/slumr_logs/%j.err \
    --wrap "export PATH=\$HOME/.local/bin:\$PATH; cd $R; mkdir -p logs/profile_demo_collection/$STAMP; \
      OMP_NUM_THREADS=8 MUJOCO_GL=egl uv run --project envs/sim_arrhenius --no-sync python \
      grasp_synthesis/collect_demos_synth_v4.py \
      --experiment single_lift_${obj}_soft_abs_action_armfocus_7d_realws \
      --task-name profile_$obj --out-dir logs/profile_demo_collection/$STAMP \
      --table-z 0.0138 --n-episodes 20 --n-envs 10 --seed 0 --scene-dr-every 1 \
      --record-video 100000 > logs/profile_demo_collection/$STAMP/$obj.log 2>&1")
  echo "  $obj -> $JID"
done
