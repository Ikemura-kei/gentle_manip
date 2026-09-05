#! /bin/bash
obj=tofu

OMP_NUM_THREADS=8 MUJOCO_GL=glfw uv run --project envs/sim python grasp_synthesis/collect_demos_synth_v4.py \
  --experiment single_lift_${obj}_soft_abs_action_armfocus_7d_realws \
  --task-name  synthe_dev \
  --table-z 0.0138 \
  --n-episodes 10 --n-envs 1 --seed 0 --scene-dr-every 1 --dev-viewer --record-video 100000