# 12-object generalist — real-robot deployment

## 1. Pull the run (on your LOCAL machine)

Use the standing tool — it resolves the run ID from `experiments.csv` and brings the checkpoint,
`.hydra/config.yaml`, the `config/` env snapshot, `EXPERIMENT.md` and the run's OWN
`normalization.npz` into one place:

    ./gentle_manip/scripts/pull_run.sh cvzth --ckpt 80
    # -> ./downloaded_runs/cvzth/{checkpoint/state_80.pt, .hydra/, config/, normalization.npz}

Verified for these runs: all three IDs (`ydvlr` seed 42, `cvzth` seed 27, `fyetc` seed 321) are
registered with exactly one row each, and the normalization auto-resolve
(`env: single_lift_generalist_12obj` -> `dataset/dppo/<env>/normalization.npz`) hits the right file.

`--ckpt 80` for the gentler checkpoint (see the checkpoint note below); `--all` for every epoch;
add `--eval canon_mushroom_200geo40_last_e94 --with-videos` to also pull the eval + clips.

**WHICH CHECKPOINT / SEED — `cvzth` (seed 27) `state_80`, chosen for GENTLENESS (user, 2026-09-01).**

Mushroom, 200 eps / 40 geometries, per arm:

| run | seed | ckpt | succ% | dmg% | sust/Y |
|---|---|---|---|---|---|
| **cvzth** | **27** | **ep80** | 72.0 | **9.0 ± 4.0** | 0.45 |
| fyetc | 321 | ep80 | 71.5 | 9.5 ± 4.1 | 0.45 |
| ydvlr | 42 | ep94 | 74.5 | 10.5 ± 4.2 | 0.48 |
| fyetc | 321 | ep94 | 72.0 | 13.0 ± 4.7 | 0.50 |
| ydvlr | 42 | ep80 | 73.5 | 15.0 ± 4.9 | 0.52 |
| cvzth | 27 | ep94 | 72.0 | 16.5 ± 5.1 | 0.51 |

⚠ **The SEED matters as much as the checkpoint, and not in a stable way.** `ydvlr`@ep80 is the
WORST arm (15.0%) while `cvzth`@ep80 is the best (9.0%) — the same epoch, opposite ends. So
"use ep80 for gentleness" is only true on the POOLED estimate (ep80 11.2% vs ep94 13.3%), and
even those CIs overlap. Picking the single lowest of six noisy measurements (each ±4-5) is partly
selection on noise.

**Honest status: `cvzth`@ep80 is the best available BET, not a demonstrated optimum.** It costs
~1 pt of success vs the best-success arm and is the gentlest thing measured. If the real-robot
result matters, evaluate 2-3 arms rather than trusting this ranking — and note the real robot is
the only gentleness test that counts.

## Run

    uv run --project envs/dp3 python -m gentle_manip.scripts.deploy_real_dppo \
      --ckpt          downloaded_runs/cvzth/checkpoint/state_80.pt \
      --normalization downloaded_runs/cvzth/normalization.npz \
      --obs-config    gentle_manip/configs/obs/point_cloud_1cam_armfocus.yaml \
      --action-config gentle_manip/configs/action/abs_pose_euler_abs_gripper.yaml \
      --ft-denoising-steps 0 \
      --record dataset/demos/single_lift_generalist_12obj_real

## THE THREE DEFAULTS YOU MUST OVERRIDE (all wrong for this checkpoint)

| flag | script default | REQUIRED | why |
|---|---|---|---|
| `--normalization` | `single_lift_mushroom_soft_abs_pcd_rot6d/normalization.npz` | the 12-obj one | 10-dim rot6d stats vs our 7-dim; decoding with the wrong min/max gives wrong commands. (There IS an action_dim guard that would catch this one.) |
| `--obs-config` | `point_cloud_1cam_outlier_rot6d.yaml` | `point_cloud_1cam_armfocus.yaml` | rot6d proprio vs our QUAT; and the non-armfocus variants lack `object_focus`, so ~80% of every cloud would be arm body — a different input distribution than training. **This one fails SILENTLY.** |
| `--action-config` | `abs_pose_abs_gripper.yaml` (10-dim) | `abs_pose_euler_abs_gripper.yaml` (7-dim) | our policy emits 7-dim euler-absolute. A 10-dim pipeline indexes `[:,9]` -> IndexError. |

`--ft-denoising-steps 0` because this is a BC checkpoint, not PPO-finetuned.

**obs match verified by PARSED config**, not by reading YAML: `point_cloud_1cam_armfocus` is an
EXACT match to training's `superset_soft_armfocus` point-cloud block (cameras, crop, max_points
1024, outlier_removal 0.01/23, object_focus z_lo 0.15 / r_ee 0.13 / arm_weight 0.15) and
`quat_noise_std 0.003`. `point_cloud_1cam_outlier` differs on `focus_z_lo` and `focus_arm_weight`.

## point_cloud_shift — LEAVE IT ACTIVE

`configs/setup/real_lab.yaml` has `point_cloud_shift: [0.009, 0.0, 0.0]` ACTIVE. **Correct for
this policy — do not comment it out.** Real clouds carry a measured -9 mm x perception bias; sim
clouds are unbiased by construction. This policy is trained on SIM clouds only (no real
co-training), so the real cloud must be corrected into the frame it was trained in. Confirmed
consumed by `RealBackend` (it prints `[RealBackend] point_cloud_shift active: ...` at startup —
check for that line).

## What this policy was trained on

12 objects, v4.1 collection, 4,931 train trajectories / 950,741 transitions, pinch+NaN filtered,
joint normalization applied after merging. `PairedRegDiffusionModel` w=0.5 with the 9mm-corrected
`paired_cube3_clouds_shift9.npz`. 94 epochs = 698,200 gradient steps (matched to ddgrl).

Objects: tofu, mushroom, strawberry, raspberry, tomato, cherry_tomato, banana_chunk,
prim_{cylinder,sphere,lamp,ellipsoid,cuboid}. **pasta_bundle is NOT in training** (OOD).

## Sim numbers, so you know what to expect

Mushroom, 200 eps / 40 geometries, pooled over 3 seeds: success 72.8 +/- 3.6%, sustained/yield
0.50, damage rate 13.3 +/- 2.7%. Comparable to the 3-object generalist ddgrl (70.5%, 11.0%) —
i.e. more categories at no measurable cost on mushroom, NOT an improvement.

⚠ **Untested on the real robot.** Every number above is sim. The real-robot gap is the thing this
deployment is meant to measure.
