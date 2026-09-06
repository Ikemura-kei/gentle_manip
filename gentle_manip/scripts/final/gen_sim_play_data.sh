#!/bin/bash

play_data_real=dataset/demos/play_red_cube_real/26-09-05-xiv

uv run --project envs/sim python -m gentle_manip.scripts.replay_real_to_sim_paired \
    --real-run ${play_data_real} --object-xy 0.3779 -0.0003 \
    --task-config gentle_manip/configs/tasks/single_lift_cube3_soft_board.yaml --task-name play_red_cube_soft \
    --render-only