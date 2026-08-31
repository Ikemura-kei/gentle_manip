# Final v4.1 generalist dataset (13 objects)

Generated from the DATA (stats.yaml + data.pkl + dr_params.csv), not from launch commands.
Recipe, identical for every object: v4.1 `--scan-metric p98`, `--regrasp-prob 0.2`,
`--grasp-extra-close auto`, all `auto` grasp params, no manual `--closure-gain`,
`scene_dr_every 1`, `seed 0`, 500-episode target.

| object | episodes | frames | demo success | saved sub-yield | nan stress | peak top10/yield (med/max) | distinct mat_E | re-grasp | run dir |
|---|---|---|---|---|---|---|---|---|---|
| tomato | 500 | 97,017 | 63.8% (784 att) | 99.8% of 494 | 6 | 0.38 / 1.07 | 98 | 98/500 | `dataset/demos/single_lift_tomato_soft/26-08-30-hgx` |

**Totals: 1 objects, 500 episodes, 97,017 transitions.**

Notes:
- `saved sub-yield` = fraction of SAVED demos whose peak top10 von-Mises stays below the
  object's yield. Failed attempts are never saved, so this — not attempt success — is the
  gentleness figure for the dataset.
- `re-grasp` = episodes started with the gripper hovering above the object at a random
  part-closed width (`--regrasp-prob 0.2`), labelled `episode_type=re-grasp-demo`. Keep the
  label through conversion so training can weight them.
- `distinct mat_E` > 1 confirms material DR was actually applied (silently-inert material DR
  was a real past bug).
- `nan stress` = episodes whose `priv_stress` is NaN while clouds/proprio/actions stay
  finite (an MPM stress-readout hiccup, ~1%). They are TRAINABLE but not measurable for
  gentleness, so sub-yield is quoted over MEASURABLE episodes only — counting them as
  failures would understate quality and hide the true over-yield count.
- Every row carries `dataset_idx` in dr_params.csv, so DR params join to data.pkl episodes.
