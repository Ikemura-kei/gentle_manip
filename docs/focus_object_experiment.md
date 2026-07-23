# `focus_object` experiment — reallocating the point-cloud budget from the arm to the object

## Why

The DPPO point-cloud BC policy for soft mushroom grasping plateaus at **~0.75 grasp success**, and it
does **not** improve with more data (300 demos ≈ 1000 demos ≈ 0.75). The failure mode is consistent:
the gripper **approaches correctly but grasps at the wrong width** — the object slips out or the grasp
misses. Since coverage doesn't help, the ceiling is an information/precision problem, not sampling.

**Quantified root cause (all 1k demos, points within 2.5 cm of the object centre):**

| trajectory phase | object-region points (of 1024) |
|---|---|
| approach (first 10%) | mean ~76 |
| **grasp / descent minimum** (t≈0.4) | **~25–30** (median 49 at most-closed; min 23) |
| after lift | ~110 |
| **arm-body share of the budget at grasp** | **~92%** |

The point cloud is cropped to the whole workspace and farthest-point-subsampled to a **fixed 1024
points**. As the arm descends it occupies a growing share of the crop, so ~92% of the budget lands on
the arm and the **object is at its sparsest (~25–30 pts) exactly when grasp width must be decided.**
On a 3.3 cm object that is a handful of points across the grasp (y) axis — plausibly too coarse to pin
the width across the shape/size/pose distribution.

**Hypothesis:** the arm points are largely *redundant with proprioception* (the policy already knows
its own pose from `ee_pos/quat/gripper`), so dropping them and reallocating the budget to the object
should sharpen the width signal and break past 0.75.

`focus_object` (`perception/pointcloud_ops.py`, gated by `PointCloudConfig`) does exactly this: after
crop, before subsample, keep points that are **low (`z < z_lo`, table+object) OR near the EE (`< r_ee`,
gripper + grasped/lifted object)** and drop the arm body — so the 1024-point budget concentrates on the
scene. This is a **disposable probe**: an upper-bound test of "is the budget the bottleneck at all?".
The task-general version (FK+CAD arm-aware *down-sampling* that preserves scene context and protects the
fingertip/contact region) is the production follow-up if the probe pays off.

## Config architecture — why the change must live in the superset (and touch eval)

Obs processing has a **single source of truth**: an experiment's `obs:` names a *superset* config
(`superset_soft`), and every stage inherits from it —

- **collection** uses `collection_obs()` = the superset (records every modality);
- **eval / online** uses `view_obs("student")` = the superset **subviewed** to `[point_cloud]`.

So the point-cloud processing params (crop, `max_points`, `outlier_removal`, and now `object_focus`)
are defined **once** and both collect and eval read the same block — they cannot drift.

Two consequences:
1. To apply `object_focus` you change the **superset** — which changes both collection AND the eval
   obs. So **eval must change too**: a policy trained on focus clouds served *non-focus* clouds at eval
   would see a different distribution and fail. This is the legitimate exception to the "all evals use
   the same config" rule — the **sim/scenario stays identical** (task 235/250, `food_shape`, fixed
   seeds → still apples-to-apple on the physics); only the **obs processing** matches the policy.
2. Do **not** edit `superset_soft.yaml` in place (every existing policy shares it). Use **parallel**
   configs.

## The two new configs (created)

- **`gentle_manip/configs/obs/superset_soft_focus.yaml`** — `superset_soft` + `object_focus:
  {z_lo: 0.12, r_ee: 0.13}` under `point_cloud`. Used at **collection** (`--obs`) and, transitively,
  at **eval** (via the experiment below).
- **`gentle_manip/configs/experiments/single_lift_mushroom_soft_eval_focus.yaml`** — identical to
  `single_lift_mushroom_soft_eval` except `obs: superset_soft_focus`. Same physics/seeds; focus obs.

No code changes — `object_focus` is already implemented.

## Which config controls the point cloud at each stage

| stage | point-cloud obs from | live/baked |
|---|---|---|
| collect | `--obs superset_soft_focus` → `configs/obs/superset_soft_focus.yaml` | **baked** into demos |
| convert | recorded cloud | baked |
| pretrain | the `.npz` | baked (`n_points=1024`) |
| eval | server `--experiment single_lift_mushroom_soft_eval_focus` → `obs: superset_soft_focus` → `student` view | **live** |

## Run it (the wide1k pipeline with the `_focus` swaps)

Environment overrides for the eval sim fidelity: `GM_SIM_SUBSTEPS=235 GM_MPM_SAMPLER=regular`.

```bash
# 1. COLLECT (envs/sim) — new obs config; produces a new datetime run dir
MUJOCO_GL=egl GM_SIM_SUBSTEPS=235 GM_MPM_SAMPLER=regular \
uv run --project envs/sim --no-sync python examples/collect_mushroom_demos_batched.py \
  --experiment single_lift_mushroom_soft \
  --obs superset_soft_focus \
  --collect-config gentle_manip/configs/collect/single_lift_mushroom_soft_scripted.yaml \
  --n-demos 1000 --n-envs 5 --pose-box 0.15 0.15 0.10 \
  --scene-dr-every 5 --shard-size 20 --max-steps 320 --seed 1
#    -> dataset/demos/mushroom_soft_batched/<DATETIME>/   (note the run dir it prints)

# 2. CONVERT (envs/dppo) — new DATA_ENV so it doesn't clobber the non-focus data
uv run --project envs/dppo --no-sync python -m gentle_manip.dppo.convert_demos \
  dataset/demos/mushroom_soft_batched/<DATETIME> \
  --out dataset/dppo/single_lift_mushroom_soft_pcd_focus1k --point-cloud

# 3. PRETRAIN (envs/dppo)  [cluster: gentle_manip/scripts/arrhenius/dppo_pretrain.sbatch with DATA_ENV=…_focus1k]
uv run --project envs/dppo --no-sync python -m gentle_manip.dppo.train \
  --config-path "$(pwd)/gentle_manip/dppo/cfg/single_lift_mushroom_soft_pcd" \
  --config-name pre_diffusion_pointnet \
  env=single_lift_mushroom_soft_pcd_focus1k \
  train.n_epochs=4000 train.save_model_freq=1000

# 4. EVAL — server (envs/sim) with the FOCUS eval experiment
GM_MPM_SAMPLER=regular MUJOCO_GL=egl \
uv run --project envs/sim --no-sync python -m gentle_manip.scripts.serl_sim_server \
  --experiment single_lift_mushroom_soft_eval_focus --view student --num-envs 5 \
  --render-rgb --subprocess --port 5570
#    evaluator (envs/dppo)
uv run --project envs/dppo --no-sync python -m gentle_manip.dppo.train \
  --config-path "$(pwd)/gentle_manip/dppo/cfg/single_lift_mushroom_soft_pcd" \
  --config-name eval_diffusion_pointnet \
  base_policy_path="$(pwd)/logs/dppo/dppo-pretrain/single_lift_mushroom_soft_pcd_focus1k/<id>/checkpoint/state_4000.pt" \
  ft_denoising_steps=0 \
  experiment=single_lift_mushroom_soft_eval_focus \
  normalization_path="$(pwd)/dataset/dppo/single_lift_mushroom_soft_pcd_focus1k/normalization.npz"
```

### The `_focus` arg swaps vs the non-focus (wide1k) run, at a glance

| stage | non-focus | focus |
|---|---|---|
| collect | `--obs superset_soft` | `--obs superset_soft_focus` |
| convert | `--out …/single_lift_mushroom_soft_pcd_wide1k` | `--out …/single_lift_mushroom_soft_pcd_focus1k` |
| pretrain | `env=…_wide1k` | `env=…_focus1k` |
| eval server | `--experiment single_lift_mushroom_soft_eval` | `--experiment single_lift_mushroom_soft_eval_focus` |
| eval evaluator | `experiment=…_eval`, `normalization_path=…/wide1k/…` | `experiment=…_eval_focus`, `normalization_path=…/focus1k/…` |

## Reading the result

- **Success clearly > 0.75** → the point budget *was* the bottleneck. Promote `focus_object` from probe
  to the **task-general FK+CAD arm-aware down-sample** (mask arm via forward kinematics, down-sample
  rather than hard-drop so scene context survives, protect the fingertip/contact region), in the shared
  `PerceptionPipeline`, and retrain on that.
- **Success ≈ 0.75** → the object is now well-resolved but width is *still* wrong → it's the labels /
  task precision, not perception. Escalate to a **contact/stress-gated scripted expert** (close-to-force,
  not close-to-privileged-width) so the demonstrated width is correct per object, or a state-teacher →
  distillation.

## Caveats

- `object_focus` is a grasp-shaped heuristic (drop high-and-far points). Fine for this single-object
  tabletop probe; **not** general — objects on platforms/shelves (terrain_place) or wide multi-object
  scenes would lose content. The production fix is the FK arm-mask above.
- `z_lo` / `r_ee` are the tunables; validated previously to preserve the object across home/grasp/lift.
- Real deployment of a focus-trained policy would likewise need a `focus_object` obs config
  (`point_cloud_1cam_filtered.yaml` is the real-side analog).
