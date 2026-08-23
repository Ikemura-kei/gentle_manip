# Item 2 — real-vs-scripted demo kinematics + the v3.1 synthesis update (2026-08-23)

**Datasets.** Real: `dataset/demos/single_lift_mushroom_real_merged` (55 teleop episodes —
the foundation real set). Scripted: `dataset/demos/single_lift_mushroom_soft/26-08-17-hwo`
(650 eps, the hwo-recipe foundation collection the adopted stack reproduces). Both store
obs at 30 Hz; comparison is in POSE space (positions/quats/gripper widths), so the recorded
action space (delta vs 10d-absolute) doesn't matter. Analysis script:
`examples/demo_analysis/item2_kinematics_compare.py`; raw distributions in
`examples/demo_analysis/figures/item2_kinematics/` (summary.yaml + histograms).

## Result: the recipes are already kinematically close — five real differences

Medians (55 real closed episodes vs 650 scripted):

| metric | real | scripted | verdict |
|---|---|---|---|
| translation speed (mm/step, mean / p95) | 2.20 / 3.67 | 2.22 / 3.03 | **matched** |
| rotation speed (deg/step, mean / p95) | 1.04 / 1.75 | 0.87 / 1.53 | **matched** |
| episode length (steps) | 225 | 185 | close |
| close onset (fraction of episode) | 0.52 | 0.44 | close |
| width before close (mm) | 79.6 | 79.7 | **matched** (both close from full open) |
| z at close / min z (mm) | 3.0 / 3.0 | 2.8 / 2.4 | **matched** |
| lift speed (mm/step) | 2.68 | 2.85 | **matched** |
| longest pause (steps) | 31 | 36 | close (both pause; lookahead/commanded derivation carries the lead) |
| **hover at grasp pose before close (steps)** | **6** | **2** | differ — humans settle ~0.2 s before closing |
| **close duration (steps)** | **21** | **34** | differ — humans close ~40 % faster |
| **total rotation from home at close (deg)** | **30** | **50** | differ — scripted yaws more (free CMA yaw) |
| **tool tilt from vertical at close (deg)** | **2.0** | **7.4** | differ — humans grasp vertically |
| **gripper width at settle (mm)** | **30.9** | **35.3** | differ — humans squeeze ~4 mm deeper |

## Interpretation

1. The hwo recipe already matches human speed almost exactly — speed matching is NOT the
   remaining sim2real lever on the data side.
2. The differences cluster around the GRASP EVENT: humans hover briefly, close fast,
   stay vertical, rotate less, and grip deeper.
3. The rotation difference dovetails with **item 5**: bounding the grasp yaw to ±45° of
   the camera-perpendicular direction (the validated occlusion fix — fully-hidden episodes
   24 % → 4 % in the v5c profile) ALSO pulls total rotation toward the human 30°. One knob
   serves both goals.
4. The **grip-depth difference (30.9 vs 35.3 mm) is deliberately NOT copied** this
   iteration: squeezing 4 mm deeper fights the gentleness mission (stress), interacts with
   item 10 (gentler-grasp test, which goes the OTHER direction), and changing firmness
   would confound the comparison against afucm. Recorded as a candidate lever for the
   remaining real-robot failure modes (slip).
5. Tilt (2° vs 7.4°) is left to the FEM planner this iteration — its tilt is
   stress-optimal, and capping it risks demonstrator success; revisit if v3.1 doesn't move
   real performance.

## The v3.1 update (implemented)

`grasp_synthesis/collect_demos_synth_v3.py` gains two knobs (defaults inert — baseline
recipe unchanged):
- `--n-settle N` — hold at the grasp pose before closing (was hard-coded 1). Set 6 to
  match the human hover.
- `--cam-azimuth-max-deg D` — item-5 occlusion bound: shaped penalty on grasp yaw beyond
  D° azimuth from camera-perpendicular, using the task camera's calibrated position; also
  centres the CMA seed fan (mechanism validated in the v5c profile work).

**v3.1 recipe** = foundation hwo recipe with the three human-matched/occlusion deltas:
```
--n-home-to-pre 77 --n-settle 6 --n-grasp 20 --grasp-extra-close 0.005 \
--cam-azimuth-max-deg 45
```
(close 30 → 20 steps ≈ human 21; hover 1 → 6; azimuth 45°.)

**Test protocol** (overnight run): 500-episode collection on the realws experiment →
7d-euler commanded conversion → concat the 55 real demos (no oversample, exactly the
afucm setup: big net, 600 epochs, save/100) → canonical-harness sweep of every
checkpoint → compare to afucm under the SAME local eval protocol. Results land in the
training table in DEVLOG.
