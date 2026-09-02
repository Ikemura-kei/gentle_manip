# Generalist: 12 SIM objects + 7 REAL objects — setups and commands (2026-09-01)

Three experiments x 3 seeds = **9 runs**, all on one merged dataset. Everything except the
object-conditioning matches `logs/dppo/dppo-pretrain/single_lift_generalist_12obj`
(runs `ydvlr`/`cvzth`/`fyetc`).

## Dataset — `dataset/dppo/single_lift_generalist_12obj_real7`

| part | source | slices |
|---|---|---|
| SIM | the 12 v4.1 collections, provenance-guarded (`.agent_tmp/g12_slices.txt`) | `dataset/dppo/g12_*` |
| REAL | **PINNED** `dataset/transfer/real_paired_7obj_2026-09-01` | `dataset/dppo/g12r_*` |

**REAL SOURCE IS PINNED (user):** ONLY `real_paired_7obj_2026-09-01` — 7 objects, 141 episodes
(mushroom/grape/tomato/padron_pepper/cherry_tomato/strawberry 20 each, tofu 21). **grape and
padron_pepper have NO sim counterpart; included deliberately — "we just want to train a mix".**
No other real data may enter — in particular NOT `single_lift_mushroom_real_merged_shift9mm`,
which the separate `znmbh` run uses. The build asserts every real slice lives under the pin.

**Already bias-corrected:** collected with `point_cloud_shift [0.009,0,0]` ACTIVE (bundle README),
so the clouds are in the same frame as the unbiased sim clouds. **No shift step.**
**Already 7-dim euler-absolute actions** — so real slices convert with **NO** `--derive-action`
(the SIM collector records 10-dim rot6d and does need it). Verified from the pkl: `actions (T,7)`.

**Normalization is recomputed AFTER merging** (asserted: joint stats must differ from the
sim-only dataset, else the real data never reached them). `aux_contact`/`aux_object_pos` are
dropped by the merge (real rows cannot carry them) — harmless, the recipe uses no aux loss.

    sbatch gentle_manip/scripts/experiments/generalist_12obj_real7/build_g12real7.sbatch          # convert 7 real + merge 19 slices + verify

## The three experiments

| tag | conditioning | params | note |
|---|---|---|---|
| `base` | none | 20,902,988 | identical arch to the 12obj generalist |
| `objraw` | first-frame object points APPENDED to the observation cloud (one PointNet pass) | 20,902,988 | **zero extra params** |
| `objemb` | first-frame object cloud ENCODED by the same backbone, feature concatenated | 22,475,852 | +1.57 M (wider first MLP layer) |

Smoke-verified: `none` reproduces the baseline param count EXACTLY and is bit-identical even when
`obj_points` is present, so the default path cannot regress.

    bash gentle_manip/scripts/experiments/generalist_12obj_real7/launch_g12r7.sh base      # 3 seeds: 42, 27, 321
    bash gentle_manip/scripts/experiments/generalist_12obj_real7/launch_g12r7.sh objraw
    bash gentle_manip/scripts/experiments/generalist_12obj_real7/launch_g12r7.sh objemb

## The object crop (`pointcloud_dataset.obj_crop`)

**ADAPTIVE ceiling = `min(0.15, z_ee(t=0) - 0.01)`.** A FIXED 6 cm ceiling truncated tomato in
57.5% of episodes and prim_cylinder in 16.9%. The TCP sits just below the finger ends (user), so
everything gripper-related is ABOVE `z_ee` -> `z_ee` is the principled ceiling, not a constant.
EE height at t=0 is bimodal with an empty band — regrasp starts 6.6-13.6 cm, home starts
17.9-21.8 cm — so **79.2% of episodes get the full 15 cm cap and truncate nothing**.

Computed from the CLOUD + PROPRIO inside the dataset, **not** from a precomputed label file, so
the identical rule runs at eval/deploy where only those two exist (user requirement).
Output is sampled/padded to a fixed 128 points.

**No outlier filter.** The first version kept only the largest voxel component; that was wrong and
is removed — the EE at t=0 sits at a median 19.8 cm against a 6 cm ceiling, so no gripper point
can be in the crop, and the filter was severing one object at a thin waist. User confirmed from
the rotating 3D renders: "what you rejected are part of the lamp."

## Fixed, inherited from the 12obj generalist

`PairedRegDiffusionModel` w=0.5 on **`paired_cube3_clouds_shift9.npz`** (same file as the 12obj
generalist, user-specified), `mlp_dims [3072,3072,3072]`, `action_dim 7`, `save_model_freq 8`,
seeds {42,27,321}, epochs solved to hold ddgrl's **695,461 gradient steps**, `warmup` and
`val_freq` scaled by ddgrl's 100/350 and 10/350 fractions.

## Code touched (mirrored to the `gm_generalist` worktree, per CHECKLISTS §5.2)

* `gentle_manip/dppo/pointcloud_dataset.py` — `obj_crop`, `obj_crop_zmax`, `obj_crop_margin`,
  `obj_crop_points`; emits `cond["obj_points"]`.
* `gentle_manip/dppo/pointnet_diffusion.py` — `obj_cond_mode: none|raw|embed`. All edits confined
  to `PointNetDiffusionMLP` (the symbol appears in 3 classes).
* `gentle_manip/scripts/label_first_frame_object.py`, `plot_first_frame_object.py`,
  `video_first_frame_object.py` — offline labelling + inspection (not used by training).

## Evaluating objraw / objemb — the eval MUST recreate the crop

`eval_agent.py` now builds `cond["obj_points"]` with the IDENTICAL rule
(`ceiling = min(zmax, z_ee - margin)`, sampled/padded to K), **latched on the first `act()` of each
episode**. Latching matters: the object is unoccluded only at t=0, so recomputing later would read
the in-gripper view — the same mistake the width floor made before it was latched.

Enable by pointing it at the dataset's normalization (needed to de-normalize `z_ee`):

    CKPT=<...>/checkpoint/state_<N>.pt \
    GM_OBJ_CROP_NORM=$REPO/dataset/dppo/single_lift_generalist_12obj_real7/normalization.npz \
    NORM=$REPO/dataset/dppo/single_lift_generalist_12obj_real7/normalization.npz \
    GM_EXTRA_OVERRIDES="model.network.mlp_dims=[3072,3072,3072] model.network.obj_cond_mode=raw" \
    SIM_EXPERIMENT=<obj>_7d_realws EVAL_EXPERIMENT=<obj>_7d_realws \
    N_EPISODES=200 NUM_ENVS=5 SCENE_GROUP_SIZE=1 \
    sbatch gentle_manip/scripts/arrhenius/dppo_eval.sbatch

Optional: `GM_OBJ_CROP_ZMAX` (0.15), `GM_OBJ_CROP_MARGIN` (0.01), `GM_OBJ_CROP_POINTS` (128) —
must match training. `base` needs NEITHER `GM_OBJ_CROP_NORM` nor the `obj_cond_mode` override.

⚠ The eval config hardcodes `mlp_dims: [1024,1024,1024]`, so **every** eval of these runs needs the
`[3072,3072,3072]` override (the comment in that file claiming it matches training is false), and
`obj_cond_mode` must be passed too — the eval cfg does not inherit training architecture.
Verify `Number of network parameters` in the eval log: base/objraw **20,902,988**,
objemb **22,475,852**. A mismatch means an override was dropped.

## STILL TO DO

* Evaluate all 9 once trained (per-object; protocol `canon_<obj>_200geo40`, eval seed 42).
* The `_ss<N>` substeps caveat still applies to tomato; cherry_tomato / banana_chunk /
  pasta_bundle evals remain unresolved (user deferred).

## Verified at launch (2026-09-01)

**Adaptive crop, measured on the real merged dataset:**
```
train: ceiling median 15.0cm (min 5.6, max 15.0); points/episode mean 79.5 min 1 (0 EMPTY of 5058)
val:   ceiling median 15.0cm (min 6.2, max 15.0); points/episode mean 80.7 min 7 (0 EMPTY of  565)
```
**Median ceiling is the full 15 cm cap**, so most episodes keep the WHOLE object — the fix for the
fixed-6 cm truncation (tomato 57.5%, prim_cylinder 16.9%). Min 5.6 cm is a regrasp episode whose
gripper is genuinely low. **Zero empty crops.**

⚠ **Min 1 point** in some episode (raspberry — mean 15.8 pts, min 1 at the 6 cm crop). Padded to
128 by resampling that single point, so the conditioning is near-empty there. Not an error; a
known information limit on the smallest object.

⚠ **WHERE THE OBJ-CROP LINE LIVES:** `[dataset] OBJ CROP: ...` is a `print` -> stdout ->
`logs/slumr_logs/<jobid>_pretrain.log`. It is **NOT** in `<run>/run.log` (hydra logger only).
Grepping run.log for it reports a false "conditioning not active". Verify obj_crop from
`<run>/.hydra/config.yaml` (`train_dataset.obj_crop: true`) AND the `_pretrain.log` line.

## RESULTS — `base` (mushroom, 200 eps / 40 geometries, eval seed 42, `state_91`)

| run | seed | success % | sust/Y | damage % |
|---|---|---|---|---|
| asavh | 27 | 59.0 | 0.42 | 14.0 |
| rtgob | 42 | 68.0 | 0.51 | 16.0 |
| zdwii | 321 | 66.5 | 0.45 | 13.0 |
| **pooled (n=600)** | | **64.5 ± 4.8** | **0.46** | **14.3 ± 1.5** |
| sim-only 12obj (no real) | | 72.8 ± 3.6 | 0.50 | 13.3 ± 2.7 |

**Adding the 7-object real mix COST ~8 points of mushroom success; damage unchanged.** Seed SDs
overlap so it is not decisive on the pooled numbers — but **every sim+real seed (59.0/68.0/66.5)
lands below every sim-only seed (72.0/72.0/74.5)**, which is stronger than the overlap suggests.

Plausible mechanism: **dilution**. The 127 real trajectories are 2.5% of the mix and come from real
TELEOP, whose action distribution differs from the CMA-ES synthesiser that produced all 4,931 sim
trajectories. **The thing real co-training is FOR — real-robot transfer — is exactly what a sim
eval cannot measure**, so this is not evidence the real data is useless; it is evidence it does not
help IN SIM. If the sim number matters, consider oversampling the real slice rather than a plain
concat, or treat `base` as the real-transfer candidate and the sim-only generalist as the sim one.

## TWO EVAL-LAUNCH BUGS (both silent-ish; fixed in `gentle_manip/scripts/experiments/generalist_12obj_real7/launch_g12r7_evals.sh`)

1. **`set -o pipefail` + a missing config key.** `obj_cond_mode` is ABSENT from a base run's
   config (never overridden), so `grep` exited 1, the pipeline inherited it under `pipefail`, and
   `set -e` killed the launcher **silently at the first base run**. The chain printed
   "launching ..." and submitted NOTHING. Fixed with `{ grep ... || true; }`.
2. **Bash assignment-prefix parsing.** `${OC:+GM_OBJ_CROP_NORM=$OC}` is not a literal
   `NAME=value` at PARSE time, so bash ended the assignment prefix there; with `OC` empty the next
   word became the command -> `GM_EXTRA_OVERRIDES=...: command not found`. Fixed by exporting
   inside a subshell instead of using a prefix list.
3. **`+` is REQUIRED for `obj_cond_mode` at eval.** It is not in the eval cfg struct:
   `Could not override 'model.network.obj_cond_mode' ... not in struct`. `mlp_dims` IS in the
   struct, so it takes a plain override — which is why `base` evals worked and all 6 obj-variant
   evals died at 1m50s. Use `+model.network.obj_cond_mode=<raw|embed>`.

⚠ **A disconnect kills chained launchers.** The eval chain died with the session and nothing
auto-dispatched; the 6 obj evals had to be launched by hand. After any disconnect, re-check
`squeue` against what was supposed to be running rather than assuming a chain survived.
