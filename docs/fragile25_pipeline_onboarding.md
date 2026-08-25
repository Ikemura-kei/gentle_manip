# Fragile25 specialist → RLDG → generalist pipeline — onboarding reference

Everything needed to run, debug, and extend the cross-category gentle-manipulation
pipeline: object registry → SAGE grasp synthesis → per-category specialist BC
training → canonical eval → RLDG rollout self-distillation → generalist merge +
training → zero-shot eval. Written for a teammate (and their Claude) picking up
this branch cold. Complements (does not replace) the top-level `CLAUDE.md`, which
covers the shared sim/real framework (`RawObs`, `PolicyEnv`, obs/action pipelines,
XArm7 config) — this doc is scoped to the specialist/generalist campaign built on
top of that framework.

For the full chronological history of decisions (why chicken_breast was dropped,
why v2→v3 grasp synthesis migration happened, etc.), see
`docs/cross_category_generalist_log.md` and `docs/cross_category_specialist_log.md`.
This doc is the *current-state* reference, not a history.

---

## 1. Pipeline overview

```
registry.py          domain          SAGE grasp          scripted MPM/rigid
(object + material)  randomization → synthesis (v3)  →   rollout + record
       │                   │                                    │
       │                   │                                    ▼
       │                   │                          dataset/demos/<task>/<run>/
       │                   │                          (data.pkl, per-episode obs+action+reward)
       │                   │                                    │
       │                   │                    merge (retry / cross-category)
       │                   │                                    │
       │                   │                          convert_demos.py → train.npz/val.npz
       │                   │                                    │
       │                   │                    per-category specialist BC (DPPO diffusion policy)
       │                   │                                    │
       │                   │                    canonical eval (gentle_manip.evaluation.run_eval)
       │                   │                                    │
       │                   │                    quality gate (success_rate >= 0.25)
       │                   │                                    │
       │                   │                    RLDG rollout self-distillation (successful-only)
       │                   │                                    │
       │                   │                    merge qualified categories + VLM/category embedding
       │                   │                                    │
       │                   │                    ONE generalist policy (category-conditioned)
       │                   │                                    │
       └───────────────────┴────────────────────────────────────▼
                                                    zero-shot eval on held-out categories
```

Two parallel generalist arms exist for ablation:
- **RLDG generalist** (`run_fragile25_merge_and_train.py`) — trains on self-distilled
  successful rollouts of quality-gated specialists.
- **Direct generalist** (`run_direct_generalist.py`) — trains directly on raw
  synthesis demos, skipping specialist training/eval/rollout entirely. Tests
  whether RLDG distillation earns its cost over just using more raw demos.

---

## 2. Object registry & materials

`gentle_manip/assets/registry.py` — `OBJECT_MAP: dict[str, ObjectDef]`, 36 registered
categories (food + a few non-food calibration/dev objects: `cal_cube_*`, `red_cube`,
`gelatin`, `sponge`).

```python
@dataclass(frozen=True)
class ObjectDef:
    name: str
    material: Material
    object_type: str = "soft"              # "soft" (MPM) | "rigid"
    size: Tuple[float, float, float] = (0.04, 0.04, 0.04)   # primitive box extents (m)
    default_pos: Tuple[float, float, float] = (0.50, 0.0, 0.03)
    mesh_path: Optional[str] = None         # None = primitive box; else gs.morphs.Mesh
    shape_dr_ranges: Optional[Dict[str, Tuple[float, float]]] = None
        # keys: bend_deg, twist_deg, taper, rbf, axis_scale, scale
    material_dr_mult: Optional[Dict[str, Tuple[float, float]]] = None
        # keys: E, nu, rho, yield — MULTIPLIERS on nominal, not absolute Pa
    sim_substeps_override: Optional[int] = None       # MPM CFL stability per-shape
    mpm_grid_density_override: Optional[float] = None
```

`get_object_def(name)` raises `KeyError` (with the sorted known-object list) if
unregistered — never silently falls back.

**Current 9-category `HELD_IN` training roster** (both generalist scripts agree on
this list — `run_direct_generalist.HELD_IN`, `run_fragile25_merge_and_train`'s
`TRAIN` differs slightly, see §8): `banana, cherry, grape, kiwi, mushroom,
pasta_bundle, raspberry, shrimp, tomato`.

**4-category zero-shot test roster**: `blackberry, scallop, dumpling, gelatin`
(`run_direct_generalist.ZERO_SHOT`) — never appear in any training/merge step.

**Categories explicitly excluded/retired** (do not re-add without new evidence):
apple/pear (rigid, not the point of this project), blueberry/avocado/fish_raw/
fish_cooked/beef_raw/beef_cooked/cheese/watermelon/tofu/sponge (documented 0%
success or physics crashes in the original 25-category campaign — see
`project_fragile25_campaign.md`), chicken_breast (retired after ~15hrs of
search-convergence failure — see `project_generalist_12plus4_campaign.md`
"chicken_breast" sections for the full diagnostic trail before ever reconsidering it).

Per-category MPM stability overrides exist for `pear`(400), `kiwi`(280),
`strawberry`(300), `peach`(400), `tomato`(380), `watermelon`(300) substeps (vs the
220-substep shared default) and `blueberry` (grid_density 500 vs 250 default, tiny
volume undersamples MPM particles otherwise) — **if you add a new category and see
MPM instability (NaN, "substep_dt > suggested_dt" warnings), tune these two fields
first**, mirroring the mushroom Config-C precedent in `CLAUDE.md`.

`gentle_manip/assets/materials.py` — `MATERIALS: dict[str, Material]`, point values
(E, nu, rho, yield in Pa/kg-m³). Convention: `yield ≈ E * 0.15` unless a cited
failure stress/strain exists (documented per-material in comments). Examples:
mushroom E=3e5/yield=4e4, egg E=2e6/yield=8e3 (brittle shell — low yield despite
high E), beef_raw E=2e3/yield=200 (very soft, tears easily).

---

## 3. Domain randomization (per-episode + per-scene)

Applied by `SimBackend`/`GenesisWorker` (see `CLAUDE.md` "Domain Randomization" for
the full mechanism). What matters for data collection:
- **Per-reset** (cheap): object xy position, yaw (±180°)/pitch/roll, robot home offset.
- **Per-scene** (`--scene-dr-every N` in the synthesis CLI, expensive — full genesis
  relaunch): material E/nu/rho/friction, object scale, and procedural mesh
  deformation (bend/twist/taper/axis_scale/rbf) via `assets/mesh_deform.py`, sourced
  from each category's `configs/dr/food_shape_<cat>.yaml`.
- A mesh-less primitive category (e.g. `gelatin`, `beef_raw`) with shape-DR ranges
  configured anyway will crash `_apply_scene_dr` (`trimesh.load(None)`) — either
  give it a real mesh asset or drop the shape-DR keys from its DR config.

---

## 4. Grasp synthesis — SAGE (v3)

**Script**: `grasp_synthesis/collect_demos_synth_v3.py` (run in `envs/sim`, needs
`env -u PYTHONPATH -u VIRTUAL_ENV` if invoked directly from an interactive shell —
a stray conda `PYTHONPATH` can shadow the repo's own genesis submodule).

**What it optimizes**: a **width-controlled** (not force-controlled) FEM contact
model — two pads close to a commanded width, indentation is prescribed, grip force
is the *output* reaction (`smgrasp/width_grasp.py`). Multi-start CMA-ES over
`[tx,ty,tz,roll,pitch,yaw,width]` maximizes
`score = -stress_top10 - W_ALIGN*(1-align) - W_PEAK*(p98 stress)`,
gated by a cheap no-FEM pre-filter and a holdability check
`2*mu*grip >= mass*(g+accel)`. This is why SAGE grasps are systematically gentler
than a purely geometric SDF-cost search — it directly minimizes FEM-predicted
indentation stress instead of proxying gentleness through grasp geometry.

**Key CLI flags** (defaults in parens):
`--experiment` (required), `--n-episodes`(50), `--n-envs`(5), `--maxfevals`(1145),
`--grasp-n-starts`(6) — bump this first if a category shows near-0% success but
reasonable-looking stress predictions (search-convergence issue, see
chicken_breast's diagnostic history), `--scene-dr-every`(1), `--grasp-gpu`
(GPU FEM solver), `--crush-frac-threshold`(1.35) — stress/yield ratio above this
anywhere in the episode marks it crushed regardless of final height,
`--record-video`, `--resume-dir`, `--seed`(0, **not actually used to seed CMA-ES**
— repeated runs with the same `--seed` are NOT guaranteed to reproduce identical
grasps).

**Success gate** (`execute_and_collect`):
```python
height_ok = obj_z > (grasp_pos_z + LIFT_HEIGHT * 0.5)   # LIFT_HEIGHT = 0.2
success = height_ok & ~crushed_mask   # crushed_mask is sticky once set
```

**Regrasp / retry flags** (idea #1–#5, added for the banana regrasp-behavior
debugging campaign — see `project_HANDOFF_regrasp_debug_2026-08-25.md` for the full
story):
- `--retry-on-slip` — rewind+regrasp on a genuine unprompted first-attempt slip,
  bounded to `MAX_REGRASP_RETRIES=3` (4 attempts total). Never induces an
  artificial failure.
- `--only-recovered` — keep ONLY episodes that recovered from a genuine slip
  (requires `--retry-on-slip`).
- `--fast-reattempt` — **the important one for regrasp-behavior training data**:
  judges each attempt pass/fail at a fixed LOW height
  (`FAST_RETRY_CHECK_HEIGHT=0.05m`, `FAST_RETRY_PASS_FRAC=0.5`) instead of a late
  checkpoint, so a failed attempt barely leaves the table before regrasping — this
  produces demonstrations where the failed-vs-successful trajectories diverge
  EARLY and OBVIOUSLY, which is what a BC policy needs to learn to disambiguate a
  real regrasp from "still holding at height" (the earlier `--retry-on-slip`-only
  method judged failure too late/high, producing near-identical failed/successful
  trajectories for a long stretch → BC multimodal averaging → policy hovers/jitters
  instead of committing to redescend). Also implements a real early-terminating
  SUCCESS check (`FAST_SUCCESS_HEIGHT=0.15m` for `FAST_SUCCESS_HOLD_STEPS=10`
  consecutive steps → episode ends immediately, no padding).
- `--early-abort` — checkpoint-based early termination; validated NOT to help
  wall-clock and NOT what fixes hover behavior — don't reach for this as "the fix"
  for regrasp problems, use `--fast-reattempt` instead.
- **Known caveat**: the `--retry-on-slip` hold-end slip check stays active
  alongside `--fast-reattempt` (independent code blocks) — an attempt that passes
  the low check can still slip out LATER at height, triggering the OLD high-height
  regrasp pattern for that one attempt. Rare (only affects attempts that looked
  good at 5cm but failed later), not yet suppressed.
- **"Firm" phase** (idea #1, always active when `--retry-on-slip` is set): once,
  at the grasp→firm phase boundary, if measured grip is weak
  (rigid: contact force `< FIRM_FORCE_THRESH_N=1.0N`; soft: von-Mises top10 rise
  `< FIRM_STRESS_THRESH_PA=2000`), closes an extra `FIRM_EXTRA_CLOSE_M=2mm`
  (soft: +2.5mm more) before proceeding to lift.

**Output schema** (per run dir `<out_dir>/<task_name>/<date>-<3 random letters>/`):
- `shard_{idx:04d}.pkl` → merged into `data.pkl`: `{"meta": {task, obs_keys, action_dim,
  rate_hz, n_episodes, created}, "episodes": [{"observations": {k: (T,...)},
  "actions": (T,7), "rewards": (T,), "recovered_from_slip": bool}, ...]}`.
- `config.yaml` — collection recipe (source, git_commit, experiment, control
  params, DR ranges used).
- `dr_params.csv` — one row per env per batch, the DR params ACTUALLY applied
  (reproducibility audit trail).
- `stats.yaml` — `{episodes_saved, episodes_failed, total_attempts, success_rate,
  elapsed_min}`.
- `videos/ep{n:04d}_env{i}.mp4` (+ `_grasp.png`), `videos_failed/fail{n:04d}_b{batch}_env{i}.mp4`.

**Orchestrator for multi-category collection**: `gentle_manip/scripts/
collect_rigid_cross_category.py` — loops `--categories`, per-category timeout
(`--timeout-s`, default 2700s), up to 4 retry attempts via `_run_with_group_kill`
(process-group SIGKILL on timeout — `uv` orphans its real child on a plain kill),
auto-resumes from the run dir with the most saved episodes
(`_find_resumable_run`, explicitly excludes any dir with `"rollout"` in the name).
Aborts launching the next category if system `MemAvailable < 4GB`. Writes
`<log_dir>/summary.json` after every category.

---

## 5. Dataset merging

Two merge scripts, different purposes — **don't confuse them**:

**`gentle_manip/scripts/merge_retry_datasets.py`** — SAME category, mixes original
first-50 raw-synthesis episodes with genuine retry/regrasp episodes into one
~100-episode training set. `merged = orig["episodes"][:50] + retry["episodes"]`.
Writes into a NEW dated run dir under the SAME task path (never invents a new task
name), with a `config.yaml` documenting exactly which source runs + how many
episodes from each. `HELD_IN` categories hardcoded as the loop target
(`main()` iterates all 9). **This is a generic starting point** — for a specific
composition ratio (e.g. the banana debugging campaign's "150 direct + 15 regrasp"
recipe, heavier-weighted than the default 50/50), write a small one-off script
following the same pattern rather than editing the shared `N_ORIGINAL=50` constant.

**`gentle_manip/scripts/merge_cross_category_demos.py`** — ACROSS categories,
symlinks each category's latest raw `data.pkl` into a temp dir, then shells out to
`convert_demos.py` once over all of them. This is what actually builds a
generalist's training set (used by both `run_fragile25_merge_and_train.py` and
`run_direct_generalist.py` internally, via each script's own `build_merge()`).
`find_latest_data_pkl` sorts by `st_mtime`, NOT lexicographically (run-dir names
aren't chronologically sortable).

**Gotcha (hit repeatedly this campaign)**: `run_fragile25_specialist.convert()`
skips re-conversion if `<DPPO_DATA_DIR>/<task>/train.npz` already exists. When
retraining on a NEWLY MERGED dataset, either `rm -rf` the stale converted dir first
or point `DPPO_DATA_DIR` somewhere fresh — otherwise it silently trains on stale
data.

---

## 6. Demo → DPPO conversion

**Script**: `gentle_manip/dppo/convert_demos.py`.

Obs-key "views" (module constants):
- `STATE_VIEW = [ee_pos, ee_quat, gripper_width, priv_object_pos, priv_object_vel]`
  — privileged teacher state.
- `STATE_VIEW_FULL` — adds `priv_object_rot6d`, `priv_object_dr_params`.
- `PROPRIO_VIEW = [ee_pos, ee_quat, gripper_width]` — deployable student view, no
  privileged obs (PointNet consumes the point cloud instead).

Key flags: `--point-cloud` (adds raw xyz cloud, NOT normalized), `--experiment
<name>` (derives obs-key order from `Experiment.load(name).view_obs(--view)
.obs_keys()` — **preferred over `--obs-keys`**, keeps every conversion in sync
with the experiment config), `--view {teacher,student}`, `--val-split`(0.1, split
by TRAJECTORY not timestep), `--category-embed --embed-source {registry,vlm}`
(requires demo files literally named `<category>.pkl`, i.e. after the
merge-scripts' symlink step).

Outputs: `train.npz`/`val.npz` (obs/actions normalized to `[-1,1]` via
`2*(x-min)/(max-min+1e-6) - 1`) + `normalization.npz` (raw-unit min/max, needed by
the sim bridge to un-normalize actions back to physical units at eval/deploy time).

---

## 7. Per-category specialist training

**Orchestrator**: `gentle_manip/scripts/run_fragile25_specialist.py`,
`run_one(category, port=5570) -> dict`. Fully idempotent/resumable via
`logs/fragile25_specialist/<category>.json` — every stage checks whether its
output field already exists before redoing work. Stages, in order:

1. `find_latest_demo_dir(category)` → `demo_dir` ("most episodes wins" heuristic —
   **pre-seed the json's `demo_dir` field explicitly** if you want a SPECIFIC
   dataset used, e.g. a newly-merged one that isn't yet the largest on disk).
2. `convert(category, demo_dir)` → `dppo_data_dir` (skip-if-`train.npz`-exists,
   see the gotcha in §5).
3. `write_configs(category, port)` → writes `pre_diffusion_pointnet.yaml` /
   `eval_diffusion_pointnet.yaml` / (later) `collect_rollouts.yaml` under
   `gentle_manip/dppo/cfg/single_lift_<cat>_soft_easy_pcd/` from the module-level
   `PRE_TEMPLATE`/`EVAL_TEMPLATE`/`ROLLOUT_TEMPLATE` Python format-strings — **edit
   these templates in the script, not the generated yaml directly**, if you want a
   change to persist across re-runs/re-categories.
4. `train(category, cfg_dir)` → `train_with_resume(...)` → `run_dir`, `train_ok`.
5. `best_checkpoint(run_dir, category)` → `find_best_checkpoint` (see the ⚠️ gotcha
   below).
6. `eval_specialist(category, cfg_dir, checkpoint, port)` → launches
   `serl_sim_server --num-envs 5 --render-rgb --subprocess`, runs the DPPO eval
   entrypoint, parses `eval_success_rate` from `summary.json`.
7. `collect_rollouts(category, checkpoint, port)` — **gated on
   `eval_success_rate >= QUALITY_GATE (0.25)`**; else recorded as
   `"skipped: eval_success_rate ... < QUALITY_GATE 0.25"` and the category is
   excluded from the generalist merge.

**Key hyperparameters** (`PRE_TEMPLATE`): `obs_dim=8, action_dim=7,
denoising_steps=20, horizon_steps=4, cond_steps=8 (proprio history),
pc_cond_steps=4 (point-cloud history), n_points=1024, visual_feature_dim=256,
n_epochs=1000, batch_size=128, lr=1e-4, save_model_freq=25, val_freq=10,
early_stop_patience=20 (in VAL-CHECKS, i.e. 200 epochs), early_stop_min_delta=1e-4`.
`cond_steps`/`pc_cond_steps` were raised from the DPPO-default 2/1 specifically to
give the policy proprio/point-cloud TREND information (needed to tell "gripper
settled on the object" from "gripper closing on nothing" as a trend, not an
instant) — relevant if you're debugging similar hover/hesitation behavior on a
new category.

**⚠️ Known bug — `best_checkpoint()` / `find_best_checkpoint()`
(`gentle_manip/scripts/train_with_resume.py`)**: it reads
`RESULTS_DIR/train_logs/<category>.log`, which is a **shared log file appended
across every historical training attempt for that category** (every retrain, every
run ID, forever). It scans the WHOLE file for the global-minimum val loss and picks
the nearest checkpoint at-or-after that epoch — but `ckpt_dir` is scoped to only
the CURRENT run's checkpoint files. If an OLDER run's best epoch number happens to
exceed the current run's max epoch so far, the fallback branch
(`max(ckpt_epochs)`) silently picks the LATEST checkpoint instead of the true best
— NOT what "best" is supposed to mean, and easy to miss since it doesn't error.
**Workaround until fixed**: when you need the genuinely-correct best checkpoint for
a specific run, scope your own val-loss scan to only that run's log lines (find
the line where the run's own snapshot/run-ID first appears, e.g. via `grep -n
"env config for '<experiment>' -> .*/<run_id>/config/"`, then only parse
`INFO] - <epoch>: train loss ... val loss ...` lines after that marker) before
picking `checkpoint/state_<epoch>.pt`. This bug will bite every category's retrain,
not just banana's — worth actually fixing `find_best_checkpoint()` to accept a
run-scoped log slice (or write a per-run log file instead of appending to a shared
one) before doing a big multi-category re-run.

**Log/result layout**: `RESULTS_DIR = logs/fragile25_specialist/` (or
`logs/fragile25_specialist_retry/` for the retry-recipe variant), per-category
`<cat>.json` (schema: category, demo_dir, dppo_data_dir, run_dir, train_ok,
checkpoint, eval_success_rate, eval_ok, rollout_ok, rollout_n_episodes,
rollout_data_path, rollout_status, timing_s), `train_logs/`, `eval_logs/`,
`rollout_logs/` (each with `<cat>.log` + `<cat>_server.log`).

**Other gotchas** (hit repeatedly this campaign, see
`project_HANDOFF_regrasp_debug_2026-08-25.md` for the fullest account):
- Any new code inside `collect_demos_synth_v3.py`'s `execute_and_collect()` must
  take config via an explicit function parameter, never reach for `args` directly
  — that function is a SIBLING of `main()`, not nested, so `args` isn't in scope;
  `ast.parse` syntax-checking does NOT catch this (runtime-only failure).
- GPU OOM risk with `--grasp-gpu` at high `n_envs` during scene-DR-heavy
  collection — the orchestrator's retry absorbs it eventually but may need a
  manual `n_envs` reduction.
- Use distinct `--port` per concurrently-running sim server (5570 is the shared
  default across most scripts) — always `ps aux | grep serl_sim_server` before
  launching a new one.

---

## 8. Canonical evaluation harness

**Every algorithm's sim eval MUST go through the shared harness** — see
`CLAUDE.md`'s "Canonical Evaluation" section for the full hard-requirement list
(one video per episode, fixed seed sequence, freeze scene-DR during eval, etc.).
Concrete API:

`gentle_manip/evaluation/eval_spec.py` — `EvalSpec` (frozen dataclass):
```python
n_episodes: int = 100        # FIXED canonical
num_envs: int = 5            # FIXED canonical (must divide n_episodes)
seed: int = 0                # FIXED canonical
max_policy_steps: int = 75   # the only task-dependent field
scene_group_size: int = 0    # 0 = fixed nominal geometry; K>0 = rebuild every K batches
early_stop_on_success: bool = False   # opt-in: freeze action once success first fires
```
`seed_for_batch(i) = seed*100003 + i`, `scene_seed_for_group(g) = (seed+991)*100003 + g`.

`gentle_manip/evaluation/harness.py` — `run_eval(venv, policy, spec, out_dir, *,
experiment_name=None, checkpoint=None, record_batches=None, extra_meta=None)`:
loops `spec.n_batches` batches, each: optional scene rebuild, `venv.seed(...)`,
`venv.reset_arg(...)`, `spec.max_policy_steps` policy-act/env-step iterations,
tracks success/ever_success/ever_in_band + 6 stress reduction signals + obj_z.
Writes `episodes.csv` + `summary.json` + an experiment config snapshot into
`<out_dir>/config/`.

**Per-algorithm adapter contract** (`eval_venv.py`, duck-typed `Protocol`s, no
inheritance needed): an `EvalVenv` (`seed`, `reset_arg`, `step` → info carries
`success`/`obj_z`/`stress_max`/`stress_mean`) and a `Policy` (`act(obs) ->
(num_envs, horizon_steps, act_dim)` normalized action chunk). DPPO's own adapter:
`gentle_manip/dppo/eval_agent.py::EvalHarnessAgent` — builds `GenesisMultiStepVecEnv`
via `genesis_venv.build_genesis_venv(...)`, wraps the diffusion model in a thin
`_DiffusionPolicy`, calls `run_eval(...)`.

**Output schema**:
- `summary.json`: `success_rate, ever_success_rate, mean_episode_reward,
  is_soft_task`, and (soft tasks only) per-stress-column `_mean/_std/_p90/_p95`,
  `gentleness_score = 1 - clip(median stress_top5mean_tmean / median yield, 0, 1)`,
  `combined_sr_gentleness = 0.5*success_rate + 0.5*gentleness_score`. Stress
  aggregates are **success-gated** (computed only over successful episodes) —
  `success_rate` itself is over all episodes.
- `episodes.csv`: one row per episode×env — success flags, 9 stress
  reduction-combo columns, DR params ACTUALLY applied that episode (reproducibility
  audit, same fields as `dr_params.csv` in the raw collection stage).
- `render/batch{i:02d}_env{j}.mp4` — one clip PER ENV PER BATCH via
  `evaluation/video.py::MultiClipRecorder` (this is what makes "num videos ==
  num episodes"). **Always sample actual video clips, not just the success-rate
  number** — an eval SR-vs-video mismatch bug appeared earlier in this campaign
  and was never fully root-caused (see `project_generalist_12plus4_campaign.md`);
  don't trust SR alone when judging a policy's real behavior (e.g. regrasp
  quality, which SR doesn't capture — a policy that eventually reaches height
  after a hover/jitter still counts as "success").

---

## 9. RLDG rollout self-distillation

**Script**: `gentle_manip/dppo/rollout_collector.py`,
`RolloutCollectorAgent` (subclasses DPPO's `EvalAgent`).

Concretely: runs the trained specialist checkpoint's OWN policy in sim
(`record_raw=True` venv mode), keeps only trajectories that ever reached
`success`, **truncates each kept trajectory right after the first success step**
(drops the idle post-success tail), writes them out in the exact demo-pickle
schema `convert_demos.py` expects. This is self-distillation/relabeling of the
specialist's own successful rollouts — NOT the original synthesis demos.

- Uses seed base `1000` (vs the canonical eval's `0`) — deliberately non-overlapping
  scenarios from the eval set.
- Target `150` successful episodes, up to `80` batches; `SHARD_SIZE=10` incremental
  flush.
- Output: `dataset/demos/<category_dir>/<date>-rollout-<3 letters>/data.pkl`, meta
  includes `source: "rldg_rollout"` and the source `checkpoint` path.
- Quality gating happens at the CALLER level (`run_one`'s `QUALITY_GATE=0.25`
  check before ever invoking this), not inside the collector — inside, the only
  filter is "did it ever succeed."
- Raises if 0 successful rollouts collected — signals a broken checkpoint or a
  `target_episodes`/`max_batches` budget that's too low, not a silent empty output.

---

## 10. VLM / category embedding

Two independent embedding tracks, both feed the SAME `category_embed` obs slot on
the generalist model — pick one via `--embed-source` at conversion/eval time.

**Track A — registry-derived** (`gentle_manip/dppo/category_embedding.py`):
one-hot over a FIXED, append-only `CATEGORY_NAMES` list (15 entries) + `[log10(E),
log10(yield), size_x, size_y, size_z, aspect_ratio]` → `EMBEDDING_DIM=21`. Cheap,
deterministic, no external model — but a NEW category outside `CATEGORY_NAMES`
raises `KeyError` rather than silently zero-filling (extend the list, don't work
around it).

**Track B — VLM** (`gentle_manip/dppo/vlm_embedding.py`, what the current
generalist configs actually use): frozen CLIP (`openai/clip-vit-base-patch32`)
embeds a representative **sim-rendered reference frame**
(`gentle_manip/assets/category_reference_frames/<category>.png` — frame 0 of a
canonical eval rollout, NOT a real photo, NOT text) → CLIP pooled features →
fixed-seed Gaussian random projection down to `VLM_EMBED_DIM=24`. Cache:
`vlm_embed_cache.npz` (precomputed via `scripts/precompute_vlm_embeddings.py`, so
downstream consumers don't need torch/transformers). Raises `FileNotFoundError` if
no reference frame exists for a category — generate one before using a new
category with `--embed-source vlm`.

Consumed via `convert_demos.py --category-embed --embed-source {registry,vlm}`
(baked into `train.npz`/`val.npz`) and live-env side via
`genesis_venv.build_genesis_venv(category=..., category_embed_source=...)`. Model
side: `PointNetDiffusionMLP(category_embed_dim=D)` reads `cond["category_embed"]`;
`category_embed_dim=0` (default) exactly reproduces the unconditioned single-
category specialist architecture — safe to leave at 0 for any non-generalist run.

---

## 11. Generalist merge + training

**RLDG generalist** — `gentle_manip/scripts/run_fragile25_merge_and_train.py`:
- `qualified_categories()` — reads `logs/fragile25_specialist/<cat>.json` for
  every category in `TRAIN` (imported from `run_fragile25_all_specialists.py`,
  currently 11: mushroom, raspberry, grape, kiwi, egg_boiled, strawberry, banana,
  tomato, shrimp, pasta_bundle, cherry), includes it iff `rollout_data_path` is
  set (i.e. it passed the quality gate AND rollout collection succeeded). Requires
  ≥2 qualified categories.
- `build_merge()` symlinks each qualified category's RLDG rollout `data.pkl` into
  `dataset/demos_merged_fragile25_TEMP/`, converts with
  `--category-embed --embed-source vlm`.
- Training hyperparameters DIFFER from the specialist recipe:
  `category_embed_dim=24, cond_steps=2, pc_cond_steps=1` (much shorter history
  than the specialist's 8/4 — the generalist doesn't need the same regrasp-history
  fix, since RLDG rollouts are already success-truncated), `n_epochs=3000,
  save_model_freq=15`. Uses `StitchedSequencePointCloudCategoryDataset` (not the
  plain point-cloud dataset — required for `category_embed` to populate).

**Direct generalist (ablation)** — `gentle_manip/scripts/run_direct_generalist.py`:
- `HELD_IN` (9 categories, see §2), `ZERO_SHOT` (4 categories, see §2).
- `_find_raw_demo_dir()` explicitly EXCLUDES any dir with `"rollout"` in the name
  — the whole point of this arm is testing "no RLDG," so it must never
  accidentally pull in RLDG rollout data.
- Reuses the SAME `TRAIN_TEMPLATE` hyperparameters as the RLDG generalist —
  only the data source differs (raw synthesis demos vs. self-distilled rollouts).
  No quality gate applied (raw demos are already gentleness-gated at collection
  time via the crush-detection success gate in §4).
- `eval_one(category, checkpoint)` evaluates on BOTH `HELD_IN` and `ZERO_SHOT`.

Both scripts wrap training in the same `train_with_resume(task=..., max_retries=5,
timeout_s=21600)` retry/resume mechanism as the specialist pipeline.

---

## 12. Model architecture

`gentle_manip/dppo/pointnet_diffusion.py`:
- `PointNetEncoderXYZ(in_channels=3, out_channels=256)` — per-point MLP
  `[3→64→128→256]` (LayerNorm+ReLU) → symmetric max-pool over points → final
  projection. `(B,N,3) -> (B,256)`.
- `PointNetDiffusionMLP(action_dim=7, horizon_steps=4, cond_dim, pc_cond_steps,
  visual_feature_dim=256, mlp_dims=(512,512,512), residual_style=True,
  category_embed_dim=0)` — the specialist/generalist policy network. `input_dim =
  time_dim + action_dim*horizon_steps + visual_feature_dim + cond_dim +
  category_embed_dim`. `cond` = `[pointnet_feat, proprio_state, category_embed?]`
  concatenated. `ResidualMLP` if `residual_style` else plain `MLP`.
- `PointNetDiffusionUNet` — alternative temporal-conv/FiLM head (DPPO's `Unet1D`),
  requires even `horizon_steps`; not the default used by this campaign's configs.
- `PointNetCritic` — PPO value head variant (for future RL finetuning, not used by
  the current pure-BC specialist/generalist pipeline).

`gentle_manip/dppo/genesis_venv.py::GenesisMultiStepVecEnv` — the sim bridge:
normalizes obs / un-normalizes actions via `normalization.npz`, executes an action
CHUNK sequentially over `n_action_steps` physical sim steps per policy step
(summing reward), auto-resets all envs synchronously on truncation (robomimic
convention), stashes terminal obs in `info["final_obs"]`.

---

## 13. Quick command reference

```bash
# Single-category grasp synthesis (SAGE v3), 50 episodes, GPU FEM
uv run --project envs/sim python -m grasp_synthesis.collect_demos_synth_v3 \
  --experiment single_lift_<cat>_soft_easy --n-episodes 50 --n-envs 5 \
  --maxfevals 800 --scene-dr-every 4 --grasp-gpu

# Multi-category orchestrated collection (resumable, retried)
uv run --project envs/sim python -m gentle_manip.scripts.collect_rigid_cross_category \
  --categories banana cherry grape kiwi --n-episodes 50 --scene-dr-every 4

# Merge original + retry demos for one category
uv run --project envs/sim python -m gentle_manip.scripts.merge_retry_datasets

# Full per-category specialist pipeline (convert -> train -> eval -> RLDG rollout)
uv run --project envs/dppo python -c "
from gentle_manip.scripts import run_fragile25_specialist as spec
print(spec.run_one('<category>', port=5570))
"

# RLDG generalist (after enough categories are qualified)
uv run --project envs/dppo python -m gentle_manip.scripts.run_fragile25_merge_and_train

# Direct generalist ablation (no RLDG)
uv run --project envs/dppo python -m gentle_manip.scripts.run_direct_generalist

# Test suite
uv run --project envs/sim python -m pytest gentle_manip/tests/ -q
```

---

## 14. Adapting to a new category — checklist

1. Register it in `gentle_manip/assets/registry.py` (`ObjectDef`) and
   `gentle_manip/assets/materials.py` (`Material`) if not already present.
2. Add `configs/dr/food_shape_<cat>_soft_easy.yaml`,
   `configs/experiments/single_lift_<cat>_soft_easy.yaml`,
   `configs/tasks/single_lift_<cat>_soft.yaml` — mirror an existing similar-shape
   category's configs rather than inventing new sim_substeps/mpm_grid_density from
   scratch (see the shared-shape-DR-profile precedent for grape/cherry).
3. **Smoke-test before bulk collection**: 3 episodes, 3 envs, low `--maxfevals`
   (~150), check for the two most common category-specific failure modes: MPM NaN
   crashes (tune `sim_substeps_override`/`mpm_grid_density_override`) and grasp
   search convergence failure (bump `--grasp-n-starts`).
4. Run full collection (50+ episodes) via the orchestrator.
5. If joining the generalist roster: add to `HELD_IN` (in both generalist scripts
   if training both arms) or `ZERO_SHOT` if it's meant to stay held-out.
6. If using VLM category embedding: generate a reference frame
   (`gentle_manip/assets/category_reference_frames/<category>.png`) and
   re-run `scripts/precompute_vlm_embeddings.py`.
7. Run the per-category specialist pipeline, confirm it clears the quality gate
   (`eval_success_rate >= 0.25`) before it can contribute to an RLDG generalist merge.
