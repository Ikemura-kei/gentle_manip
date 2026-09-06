#! /bin/bash
# [final] Synthesis-only diagnosis for one object: runs the frozen collector with --skip-execution, so every batch
# builds the FEM, plans a grasp per env, records WHY seeds were rejected / how many survived each stage / whether it
# fell back to the default grasp, then resets without executing. Output: <run>/synth_stats.csv + a printed aggregate
# (per batch = per scene-DR draw, since geometry changes every batch with --scene-dr-every 1).
# usage: bash gentle_manip/scripts/final/synth_diagnose.sh [obj=banana_chunk] [n_attempts=20] [n_envs=1] [seed=0]
obj=${1:-banana_chunk}; n=${2:-20}; envs=${3:-1}; seed=${4:-0}
cd "$(dirname "$0")/../../.."
OMP_NUM_THREADS=8 uv run --project envs/sim python grasp_synthesis/collect_demos_synth_v4.py \
  --experiment single_lift_${obj}_soft_abs_action_armfocus_7d_realws \
  --task-name  synth_diag_${obj} \
  --table-z 0.0138 \
  --n-episodes $n --n-envs $envs --seed $seed --scene-dr-every 1 --skip-execution "${@:5}"
