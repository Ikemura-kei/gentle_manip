# Recipe — soft-banana **regraspable** BC/DP3 policy (diverse-start, no retry FSM)

Status: **validated** on NAISS Arrhenius (GH200), 2026-08-30. Branch `cross-category-dp`.
This is the pipeline to lift into `main` for the regrasp-policy feature.

---

## 0. What this is

A point-cloud diffusion policy (DP3, via the `dppo` submodule) that **recovers from a
bad first grasp on its own** — it reopens, backs off, repositions and grasps again —
**without any retry state machine in the training data**. Every demonstration is a
single, clean, always-successful top-down grasp+lift. The recovery behaviour is bought
entirely by the **start-state distribution**: ~30 % of demos begin in a pose that looks
like where a *failed* first attempt leaves you (gripper closed on a miss, then the
recorded trajectory opens and re-approaches). OmniReset-inspired (arXiv:2603.15789).

Two knobs make it work:
1. **`failed` start family** (~30 % of demos) — the reopen-and-retry demonstration.
2. **Continuous EE-start sweep** (`sweep` family) — start pose is a uniform random
   fraction of the way from home to the grasp point, so the policy sees the whole
   approach corridor, not a few fixed poses.

---

## 1. Files

### Collector
| file | role |
|---|---|
| `grasp_synthesis/collect_demos_diverse_start_v2.py` | **the collector.** Fork of `collect_demos_synth.py` with a per-env phase FSM, continuous EE-start sampling, the `failed`/`sweep`/`above`/`ground`/`air`/`strict_home` start families, a soft crush gate, progressive lift-firming, and a CMA-straddle width cap. Modify **this file only** for regrasp work; the originals stay as the stable baseline. |
| `grasp_synthesis/collect_demos_synth.py` | reused for constants + `_synth_worker` (CMA-ES) + `_merge_shards` + `_x_to_targets`. Do not edit. |
| `grasp_synthesis/synth_utils.py` | `build_object_sdf` / `grasp_cost` / `run_cmaes`. Do not edit. |

### Configs (the single source of truth is the **experiment**)
| file | role |
|---|---|
| `gentle_manip/configs/experiments/single_lift_banana_soft_diverse.yaml` | **collection + training** experiment. `task: single_lift_banana_soft_diverse`, `dr: food_shape_banana_soft_diverse`, `obs: superset_soft`, `action: delta_pose_delta_gripper`, `augmentation: l515_noise`. |
| `gentle_manip/configs/tasks/single_lift_banana_soft_diverse.yaml` | collection task — object `banana_lying` (soft MPM), `sim_substeps 220`, `mpm_grid_density 250`, `success_z 0.175–0.275`, `hold_steps 30`, stress+dist+lift reward. |
| `gentle_manip/configs/dr/food_shape_banana_soft_diverse.yaml` | wide object pose + size/shape DR + MPM material bands (`object_E [2.79e5,3.21e5]`, `coup_friction [4.5,6.0]`), `robot_init_pos_xyz 0.03`. |
| `gentle_manip/configs/experiments/single_lift_banana_soft_diverse_eval.yaml` | **clean-start eval.** Uses the `_eval` task. |
| `gentle_manip/configs/tasks/single_lift_banana_soft_diverse_eval.yaml` | eval task — **low lifted-clear success band** (`success_z 0.13–0.45`, `hold_steps 4`) so the harness `early_stop_on_success` freezes the env the instant the object is lifted near home (no carry-down). **CFL-safe physics** `sim_substeps 340 / mpm_grid_density 200` (collection's 250/220 is above the CFL limit — fine for the collector, which skips a NaN batch, but the shared eval harness has no per-batch skip). |
| `gentle_manip/configs/experiments/single_lift_banana_soft_regrasp_eval.yaml` | **regrasp-start eval** (the honest recovery test). Same `_eval` task, `dr: food_shape_banana_soft_regrasp_eval`. |
| `gentle_manip/configs/dr/food_shape_banana_soft_regrasp_eval.yaml` | forces the arm home **low and offset over the object** (`robot_init_offset_xyz [0.04, 0.0, -0.11]`) — i.e. the state a failed first attempt leaves you in. |
| `gentle_manip/assets/registry.py` → `"banana_lying"` | the graspable-lying banana mesh (`assets/objects/banana_piece_lying.obj`). The stock `"banana"` is a 5.7 cm vertical baton — topples, ungraspable top-down; **use `banana_lying`**. |

### DP3 (dppo submodule) config dir
| file | role |
|---|---|
| `gentle_manip/dppo/cfg/single_lift_banana_soft_diverse_pcd/pre_diffusion_pointnet.yaml` | BC pretrain. `obs_dim 8` (ee_pos 3 + ee_quat 4 + grip 1), `action_dim 7` (delta pose+grip), `cond_steps 8`, `pc_cond_steps 4`, `denoising_steps 20`, `horizon_steps 4`, `n_epochs 200`, `batch_size 128`. PointNet + residual-MLP diffusion (`gentle_manip.dppo.pointnet_diffusion.PointNetDiffusionMLP`). |
| `gentle_manip/dppo/cfg/single_lift_banana_soft_diverse_pcd/eval_diffusion_pointnet.yaml` | canonical harness eval. `n_episodes 100`, `num_envs 5`, `seed 0` (all fixed/canonical), `n_steps 150`, `max_episode_steps 600`, `early_stop_on_success true`, `scene_group_size 0`, `record_batches null` (one clip per episode). `_target_: gentle_manip.dppo.eval_agent.EvalHarnessAgent` — routes through the shared `gentle_manip.evaluation.run_eval`. **Do not fork the harness.** |

### Cluster driver scripts (Arrhenius; adapt paths for your box)
| file | role |
|---|---|
| `gentle_manip/scripts/arrhenius/yd_banana_collect.sbatch` | collection job. Env knobs: `EXPERIMENT`, `START_MODES`, `N_EPISODES`, `N_ENVS`, `MAXFEVALS`, `SCENE_DR_EVERY`, `SEED`, `TASK_DIR`. Salvage-merges leftover shards from a crashed run. |
| `gentle_manip/scripts/arrhenius/yd_banana_pipeline.sbatch` | convert → BC pretrain (writes `EXPERIMENT.md`) → best-val ckpt → clean eval → regrasp eval → clip-trim. Chain with `--dependency=afterok:<collect_jobid>`. |
| `gentle_manip/scripts/arrhenius/yd_banana_eval.sbatch` | standalone eval of an existing checkpoint (picks best-val, unique `eval/<tag>` dir). |
| `gentle_manip/scripts/arrhenius/_pick_best_ckpt.py` | parse the pretrain log, map min val-loss epoch → nearest `state_N.pt`. |
| `gentle_manip/scripts/arrhenius/_trim_eval_clips.py` | truncate each success clip to `first_success_step*act_steps + tail` frames (drops the frozen post-success tail). |

---

## 2. Collect

**Local (`docs/cluster_data_collection.md` has the full runbook):**

```bash
env -u PYTHONPATH -u ROS_DISTRO MUJOCO_GL=egl \
  uv run --project envs/sim --no-sync python \
  grasp_synthesis/collect_demos_diverse_start_v2.py \
    --experiment single_lift_banana_soft_diverse \
    --task-name single_lift_banana_soft_diverse \
    --n-episodes 500 --n-envs 6 --maxfevals 700 \
    --scene-dr-every 2 --seed 0 \
    --start-modes "sweep:0.42,failed:0.30,above:0.16,ground:0.06,air:0.06" \
    --record-video --out-dir dataset/demos
```

**On Arrhenius:** `sbatch gentle_manip/scripts/arrhenius/yd_banana_collect.sbatch`
(defaults already match the above for the rigid variant; pass
`EXPERIMENT=single_lift_banana_soft_diverse TASK_DIR=single_lift_banana_soft_diverse`
for soft).

### The recipe that worked
- **500 successful demos**, ~7.2 h wall (soft MPM is the cost), 6 envs/batch.
- **Start-family mix** `sweep 0.42 / failed 0.30 / above 0.16 / ground 0.06 / air 0.06`
  → **~70 % direct grasp, ~30 % record-after-failure**. Keep this ratio.
- `--scene-dr-every 2` → new mesh scale+shape every 2 batches (banana E ±7 %, scale
  0.80–1.25, bend/twist/taper).
- **Grasp synthesis SR ~55 %** (CMA-ES SDF cost is a geometric proxy; failures are
  retried to reach 500). **~5 % of episodes rejected by the crush gate** (top-10 %
  von Mises > `1.25 × yield`; `yield = 45 kPa` for banana).
- Collector internals that matter for gentleness/robustness (already in v2):
  - **width cap** — CMA often returns a straddle wider than the object (SDF cost ~0,
    no contact); the close width is clamped to `object_short_axis + 2 mm`.
  - **progressive lift-firming** — grip tightens gradually during the lift (like a
    human), not a hard squeeze on contact.
  - **per-env phase FSM** — an env that finishes early is frozen and *stops being
    recorded* → **no frozen/padded frames** in the dataset (those corrupt BC).

Output: `dataset/demos/single_lift_banana_soft_diverse/<YY-MM-DD-xyz>/data.pkl`
(+ `config.yaml`, `stats.yaml`, `videos/`).

---

## 3. Convert + train

```bash
# convert (dppo env): superset demos -> student point-cloud view + 8-dim proprio state
uv run --project envs/dp3 --no-sync python -m gentle_manip.dppo.convert_demos \
  dataset/demos/single_lift_banana_soft_diverse/<run> \
  --out <DPPO_DATA_DIR>/single_lift_banana_soft_diverse_pcd \
  --experiment single_lift_banana_soft_diverse --view student --point-cloud --val-split 0.1

# BC pretrain
uv run --project envs/dp3 --no-sync python -m gentle_manip.dppo.train \
  --config-path gentle_manip/dppo/cfg/single_lift_banana_soft_diverse_pcd \
  --config-name pre_diffusion_pointnet env=single_lift_banana_soft_diverse_pcd
```

On Arrhenius the `yd_banana_pipeline.sbatch` does convert → train → dual eval in one job.

### What worked
- 200 epochs, `batch_size 128`, ~16 s/epoch (~55 min total on GH200).
- 450 train / 50 val episodes (from 500, `val_split 0.1`).
- Best val loss **0.033** at epoch ~200 (val plateaus ~150–200; `n_epochs` is capped
  because the dppo fork's early-stop counts val-checks differently — just cap it).
- `_pick_best_ckpt.py` selects the checkpoint nearest the min-val epoch.

---

## 4. Eval

**Two evals, both through the shared harness** (`gentle_manip.evaluation.run_eval`):

```bash
# clean start
uv run --project envs/sim --no-sync python -m gentle_manip.scripts.serl_sim_server \
  --experiment single_lift_banana_soft_diverse_eval --view student \
  --num-envs 5 --render-rgb --subprocess --port 5571 &
uv run --project envs/dp3 --no-sync python -m gentle_manip.dppo.train \
  --config-path gentle_manip/dppo/cfg/single_lift_banana_soft_diverse_pcd \
  --config-name eval_diffusion_pointnet \
  base_policy_path=<...>/checkpoint/state_200.pt ft_denoising_steps=0 \
  experiment=single_lift_banana_soft_diverse_eval env.specific.port=5571

# regrasp start: same, with experiment=single_lift_banana_soft_regrasp_eval on both
# the sim server AND the eval agent, on a different port.
```

On Arrhenius: `sbatch gentle_manip/scripts/arrhenius/yd_banana_eval.sbatch` (runs both).

**Rules (hard requirements — see CLAUDE.md "Canonical Evaluation"):**
- The sim server for eval **must** use the `_eval` / `_regrasp_eval` experiment
  (CFL-safe `substeps 340 / grid 200`, low success band), **not** the collection
  experiment. Wiring the eval agent to a different experiment than its sim server is
  a silent bug — they must match.
- `n_episodes=100`, `num_envs=5`, `seed=0` are **fixed/canonical**. Only
  `n_steps`/`max_episode_steps` are task-dependent.
- One video per episode (`record_batches: null`). `_trim_eval_clips.py` cuts the
  post-success tail afterwards.

### The regrasp signal
Read `episodes.csv` → **`first_success_step`**. From the regrasp start it is
**bimodal**: an early cluster (the start pose happened to work) and a **late cluster
(steps ~60–120) = genuine miss → re-approach → grasp within the episode**. A
unimodal early distribution = no regrasp. Eyeball 2–3 late clips to confirm it is a
real redescend, not hover/jitter.

---

## 5. Current results (checkpoint `hytxr/state_200`)

| metric | clean start | regrasp start (arm low/offset over object) |
|---|---|---|
| **success rate** | **1.00** (100/100) | **0.98** (98/100) |
| **gentleness** (1 − norm. peak stress vs yield) | 0.51 | 0.57 |
| **SR × gentleness** | 0.755 | 0.774 |
| `first_success_step` | 92/100 @ step 40–60 (one clean grasp) | **74 quick (<40) + 23 late (60–120)** |

- **Regrasp confirmed.** The 23 late successes are genuine within-episode recovery —
  frame-by-frame of one (fss 88): approach → fingers around banana but not secured →
  re-widen → reposition → grasp + lift-clear. No retry logic anywhere in the demos.
- **Gentleness caveat:** peak *single-particle* von Mises briefly exceeds the 45 kPa
  yield in ~all episodes (contact-local plasticity at the finger tips); **sustained
  bulk stress (top-20 over the worst 20 steps) is ~31 kPa, sub-yield.** The grip is
  firm but not crushing. Tighten `--crush-frac` (default 1.25 → 1.1) and/or lower the
  lift-firming rate if you want it gentler.
- Eval artifacts:
  `logs/dppo/dppo-pretrain/single_lift_banana_soft_diverse_pcd/hytxr/eval/{clean_v2,regrasp_v2}/`
  (`summary.json`, `episodes.csv`, `render/*.mp4` trimmed).
- Showcase clips: `docs/eval_showcase/soft_banana/`. Web writeup:
  https://claude.ai/code/artifact/5682ac2f-0b24-446f-88f1-b556778a9bbc

---

## 6. Known gaps / follow-ups for `main`

- **CFL vs speed:** collection runs above the CFL limit (250/220) and tolerates the
  occasional NaN via the collector's per-batch try/except; eval must use the CFL-safe
  `_eval` task. If you want one physics config for both, use `grid 200 / substeps 340`
  everywhere (slower collection).
- **Gentleness is marginal** (0.51–0.57). If `main` needs a gentler policy: lower
  `--crush-frac`, reduce `FIRM_*` in the collector, or add a stress term to the BC
  loss.
- **Real-robot transfer:** `docs/RUNBOOK_real_banana_regrasp.md` has the teleop
  collection + finetune + real-eval plan. Not runnable from the cluster (needs the
  physical XArm7) — prep only.
- **Cross-category generalist** (`single_lift_xcat_*`): same recipe over 4 soft objects
  (mushroom, banana_lying, kiwi, egg_boiled) via `dr.object_category_pool`. The other
  5 shortlist objects (grape/cherry/tomato/raspberry too small, strawberry placeholder
  mesh) need per-object grid density + size-scaled CMA bounds first. In progress on
  branch `cross-category-dp`.

---

## 7. Push

Everything above is committed on branch **`cross-category-dp`** in
`/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip`
(67 commits ahead of `origin/cross-category-dp`, clean working tree).

**Not yet pushed** — the Arrhenius login node has no cached GitHub credentials, so
`git push` cannot run non-interactively from here. To publish for the partner:

```bash
cd /nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip
git push origin cross-category-dp
```

(or from the local mirror `/home/yifeid/git/gentle_manip` after `git fetch` + fast-forward.)

**Do NOT merge to `master`** — the cross-category generalist work (`single_lift_xcat_*`)
on the same branch is still experimental. The partner should branch from
`cross-category-dp` or cherry-pick the soft-banana files listed in §1 into `main`.
