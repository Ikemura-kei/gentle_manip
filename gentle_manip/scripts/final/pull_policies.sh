#! /bin/bash

for R in bmbrv fdcjk ttukt; do
  ./gentle_manip/scripts/pull_run.sh "$R" --ckpt 120
done