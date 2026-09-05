# Cross-category diverse-start pipeline + gen8 generalist — partner briefing

Written 2026-09-05, branch `cross-category-dp`. Scope: what has been built on top
of the specialist→RLDG→generalist framework already described in
`docs/fragile25_pipeline_onboarding.md` (read that first for the base
architecture — object registry, SAGE grasp synthesis, canonical eval harness,
RLDG rollout distillation; this doc does not repeat it). This doc covers the
newer work: cross-category ("xcat") demo collection with baked-in recovery
behavior, the "gen8" 8-object generalist built on it, and the full retry/
disturbance lineage that led here.

**Where things actually run.** This git checkout is a *mirror* — every commit
on this branch is tagged `(mirror)` because the real work happens on the
`arrhenius` SLURM cluster (`/nobackup/proj/disk/softenable-codesign26/personal/
yifeid/gentle_manip`). Datasets, checkpoints, wandb runs, and eval outputs live
there, not in this local repo. This doc describes the *mechanism*, not a set of
result numbers — for narrative history and any numbers already recorded, see
`docs/cross_category_specialist_log.md` and `docs/cross_category_generalist_log.md`
(chronological logs, most detail near the end of each file) and
`docs/HANDOFF_banana_rigid_diverse_2026-08-28.md` / `docs/HANDOFF_cluster_migration_2026-08-28.md`.

---

## 1. The question this whole branch answers

Every scripted demo (grasp synthesis, §4 of the onboarding doc) is a clean,
open-loop trajectory: approach → settle → grasp → lift → hold, executed
perfectly. A policy trained purely on that has never seen what a *bad* first
attempt looks like, so when its own small errors compound at deployment (a few
cm off target, gripper closing on nothing) it has no data telling it what to do
next — it hovers or jitters instead of recovering. Three different fixes were
tried, in this order, before the current approach won:

1. **Bake a recovery FSM into the *collector*** (idea #1–#5, `collect_demos_synth_v2/v3.py`) — §2.
2. **Detect + correct it at the *policy* level, post-hoc** (TIDE/FAR monitor,
   ReTVL value-weighted BC) — §3. Both negative results.
3. **Never build an explicit retry mechanism at all — make the training
   distribution dense enough that recovery falls out of ordinary BC**
   (OmniReset-inspired diverse-start collection) — §4. This is what `xcat` and
   `gen8` are built on, and is the current default.

---

## 2. Retry mechanism #1: FSM-based recovery in the collector (superseded but still live code)

Three independent mechanisms, all inside the CMA-ES scripted-demo collectors.
None of these require a learned policy — they are hand-authored recovery
behaviors spliced into the otherwise-scripted trajectory.

**Grasp firming** (`collect_demos_synth_v2/v3.py`, idea #1, always on when
retry is enabled). Once per env, right at the grasp→lift boundary: if the
measured grip is weak (rigid: contact force `< 1.0N`; soft: von-Mises top10
rise `< 2000 Pa`), close an extra 2–2.5mm before lifting. Bounded to fire once.

**Lift-phase slip detection + regrasp** (idea #2, `collect_demos_synth_v2.py`
`execute_and_collect`, flag `--enable-regrasp-retry` /
`enable_regrasp_retry=`). Checked once per env exactly when it finishes the
lift phase: if the object's measured height didn't reach a fraction of the
target lift height, this is a genuine physical slip (never an induced one).
The env's phase pointer is rewound to "settle" and it re-runs
settle→grasp→firm→lift for real, up to `max_regrasp_retries` times (default
`MAX_REGRASP_RETRIES`).

**DART-style disturbance injection** (idea #3, same file,
`--disturbance-prob` / `--disturbance-max-m` / `--disturbance-phase`,
Laskey et al. 2017). For each named phase (`approach`/`grasp`/`lift`), with
independent probability `disturbance_prob` an env gets a **one-step random
positional kick** (uniform direction, magnitude up to `disturbance_max_m`)
injected into the *commanded* position at a random step during that phase.
Every subsequent step still targets the original, unperturbed plan — so the
recorded action at the next step is a genuine corrective delta back onto the
intended trajectory. This is the mechanism actually used for the cherry rigid
`_disturbed` / `_disturbed_v2` / `_graspdist` datasets
(`gentle_manip/dppo/cfg/single_lift_cherry_rigid_pcd_250_disturbed*`) —
`disturbance_prob=0.3, disturbance_max_m=0.02` (2cm), injected during `lift`
in the `_disturbed` runs and during `grasp` in `_graspdist`. A closed-loop
diagnostic found the real failure mode is compounding *position* error during
approach/grasp (not lift), which is why `_graspdist` targets `grasp` instead.

**v3's separate, more refined retry variant** (`collect_demos_synth_v3.py`,
built for the banana regrasp-debugging campaign, documented in full in
`docs/fragile25_pipeline_onboarding.md` §4): `--retry-on-slip` (bounded to 3
retries, never induces an artificial failure) + `--fast-reattempt` (judges
pass/fail at a fixed low height so a failed attempt diverges from a successful
one early and obviously — needed because judging failure too late produced
near-identical failed/successful trajectories that BC just averaged over,
producing hover/jitter instead of committing to redescend).

**Why this family was abandoned as the primary approach for regrasp behavior**:
even with `--fast-reattempt`, retraining on a 150-direct + 15-regrasp mix
reached 41% success but *still* showed hover/jitter on genuine second attempts
— the retry data existed but the policy wasn't reliably using it. This launched
the investigation in §3.

---

## 3. Retry mechanism #2: post-hoc detection/correction at the policy level (negative results)

Documented in full in `docs/cross_category_specialist_log.md` under "Banana
regrasp-hover fix: TIDE and ReTVL experiments". Summarized because the
lineage matters for anyone tempted to revisit either:

**TIDE + FAR-lite perturbation** (`gentle_manip/dppo/eval_agent_tide.py`).
Switches eval to receding-horizon (1-step) execution and measures the
disagreement between consecutive action-chunk predictions for the same step
(Temporal Inter-chunk Discrepancy Estimate). On a trip, injects Gaussian noise
for a sustained window to knock the policy off a hovering fixed point. Result:
**27% SR vs a 41% no-intervention baseline** — the perturbation trigger fired
on ~36% of all steps, disrupting good attempts as often as it rescued stuck
ones. Detector-only logging worked fine; it's specifically trigger *tuning*
that would need more work if revisited.

**ReTVL** (arXiv 2606.24633, retry-supervised value-weighted BC). Full
pipeline built: `retvl_retry_labeling.py` (algorithmic keypoint detection from
the `gripper_width` open→close(fail)→reopen→close(success) signature),
`retvl_value.py` + `train_retvl_value.py` (small PointNet-encoder value net,
the paper's Eq. 2/8/9/10), `build_retvl_weighted_dataset.py`. First attempt
(hard-pruning low-value chunks) looked good on aggregate SR (57–80%) but user
video review caught that successes were all clean first attempts — pruning
broke `cond_steps=8` history continuity exactly at the decision boundary,
inflating SR via survivorship rather than fixing regrasp. **Lesson (applies to
any future retry work): a headline SR number is not evidence that regrasp
itself works — always sample eval videos from episodes where the first
attempt is known/likely to have failed.** A weighted-*sampling* variant
(`WeightedRandomSampler` instead of hard deletion, keeps every episode
contiguous) was designed to fix this but was not carried to completion — the
user redirected to §4 before it was retrained/re-evaluated. `train_diffusion_agent_retvl.py`
and `build_retvl_alpha_weights.py` exist in the tree but their last-run state
is whatever `cross_category_specialist_log.md` recorded; nothing newer since
the pivot.

---

## 4. Retry mechanism #3 (current default): OmniReset-inspired diverse-start collection

**Core idea** (arXiv:2603.15789, translated from RL-reset-diversity to BC
demo collection): stop building an explicit multi-attempt retry trajectory or
detector. Instead, collect **single-attempt, always-successful** demos whose
*starting* configuration densely covers the states a policy would find itself
in after a bad first attempt — near the object with the wrong grip, hovering
low over the wrong spot, having wandered off entirely, or having just missed a
grasp with the gripper closed. Every one of these demos still ends in a single
continuous, successful trajectory (redirect → real approach → grasp → lift) —
no branching FSM, no failure detector, nothing for the policy to key off of at
inference time except its own point cloud + proprioception. If start-state
coverage really is the missing ingredient, ordinary BC on this data should
produce recovery behavior for free.

**Collector**: `grasp_synthesis/collect_demos_diverse_start_v2.py` (v1 is the
untouched original prototype; v2 adds RGB video capture, continuous
start-pose sampling, and a top-down grasp clamp — see its module docstring for
the full list).

**Start-pose families** (`_sample_start`, `collect_demos_diverse_start_v2.py:176`),
selected per-env by `--start-modes` weights (default
`sweep:0.44,failed:0.30,above:0.10,ground:0.09,air:0.07`):

| family | what it represents | mechanism |
|---|---|---|
| `sweep` | dense coverage of the whole home→object corridor | `t ~ U(0,1)` fraction of the way from home to grasp target; lateral + orientation jitter *grows* with `t` (folds into labels `home`/`mid_approach`/`near_object` by `t` threshold) |
| `above` | aimed at the object, wrong height/rotation | hover 3–20cm over the object, random tilt |
| `ground` | descended to the wrong spot | low near table height, laterally offset from the object |
| `air` | wandered off entirely | random point in a workspace box around home |
| `failed` | **a genuinely just-missed grasp** | see below — this is the disturbance-equivalent family |
| `strict_home` | non-regraspable baseline | exact fixed home start, full approach recorded, no pre-roll — used for the `_gen8_baseline` control (§6) |

**The `failed` family is the disturbance mechanism in this design.** `recover_from`
is set to a pose *at the object* (small random xy/z error, gripper **closed** —
a just-missed grasp), and `start_pos` is a small hover above/beside it with the
gripper **open**. The recorded trajectory begins with an explicit **recover
phase**: reopen the gripper and back away from `recover_from` toward
`start_pos` (`_env_target` phase 0, `collect_demos_diverse_start_v2.py:368`),
*then* the normal redirect → approach → grasp → lift proceeds as usual. So
every `failed_grasp`-labeled episode literally contains "I just missed, I'm
backing off, now I'm trying again for real" as continuous, successful,
recordable data — this is what teaches the policy the recovery motion, without
ever needing a runtime detector.

**Unrecorded pre-roll**: all non-`home` families are preceded by 60 unrecorded
steps moving from the true home pose to the sampled start pose (closing the
gripper partway through, for `failed`), so the *recorded* episode always
begins already at the "bad" state — the policy never sees the pre-roll, only
the recovery-then-success that follows it.

**Per-env phase FSM**: same decoupled-per-env-advance pattern as
`collect_demos_synth_v3.py` (`execute_and_collect_diverse_v2`,
`collect_demos_diverse_start_v2.py:242`) — every env advances through
`recover → approach → settle → grasp → lift → hold` independently; an env that
finishes stops being recorded (no padded/frozen frames in either the demo or
the video) while others keep going.

**Verdict so far** (from `cross_category_specialist_log.md`'s 2026-08-28
entry): 450 diverse-start episodes (55.8% success) merged with the original
150 direct-grasp banana demos → 600-episode set, BC-pretrain in progress at
the point the campaign paused for the cluster migration. Eval-against-baseline
+ genuine video-verified recovery check was the next step and, as far as this
mirror shows, is what the `gen8` A/B in §6 below is now running at
cross-category scale.

---

## 5. Cross-category (`xcat`) collection infrastructure

Scaling §4's collector from one category (banana, rigid surrogate) to many
categories concurrently, on the cluster.

**Two DR pools**, each drawing one category per scene rebuild
(`object_category_pool` in the DR config — the collector's `--scene-dr-every`
mechanism, unchanged from the base framework):
- **"Large" pool** — `gentle_manip/configs/dr/xcat_diverse_regrasp.yaml`:
  `mushroom, kiwi, egg_boiled`.
- **"Small" pool** — `gentle_manip/configs/dr/xcat_small_diverse_regrasp.yaml`:
  `grape, cherry, tomato, raspberry` (1.5–2cm fruit; scale-only shape DR —
  bend/twist/taper on a mesh this small produces degenerate geometry that
  crashes the CMA-ES SDF BVH build).
- `banana_lying` is **not** collected by either xcat pool — it's carried
  forward from the earlier single-category OmniReset campaign
  (`dataset/demos/single_lift_banana_soft_diverse/26-08-29-wyk/`, 500
  episodes) and merged in at the staging step (§6).

**Packed collection** (`gentle_manip/scripts/arrhenius/yd_xcat_pack.sbatch`,
untracked, new): runs several `collect_demos_diverse_start_v2.py` processes
**inside one GPU allocation** — a grasp collector is CPU/contact-physics bound,
not GPU-compute bound (~7GB/98GB, 0–15% GPU utilization), so a single GH200
comfortably hosts 3–4 collectors for 3–4x throughput per reserved GPU and only
charges the shared SLURM minute-pool once. Default slot layout: 1x large-pool
+ 2x small-pool (+ a 4th large-pool slot in the current script). Each slot
writes its own timestamped run dir and self-resubmits when its walltime runs
out (staggered 90s apart on launch so CUDA context creation doesn't collide).

**Per-category quota, and the bug in it.** Each collector process only sees
*its own* running tally — with 3–4 concurrent collectors drawing from
overlapping pools, nothing stopped one collector from over-collecting a
category another had already satisfied. Concretely: `mushroom` undershot to
~370/500 while `grape`/`tomato` overshot to ~550–590 (per-collector quota, not
a global one). **Current fix** (`--per-cat-target` / `--cat-have`,
`collect_demos_diverse_start_v2.py:507-577`): before launching, each slot
scrapes every SLURM `.out` log matching its tag for `ep N: env M OK` lines
tagged with the scene category, sums them into a `cat-have` string, and passes
it in; the collector excludes any category already at target from its local
pool for the whole run, and re-excludes a category mid-run the moment it
crosses target. **This is a workaround, not a real fix** — it's a regex scrape
of log text at launch time, not a shared atomic counter, so two collectors
started close together still race on the same snapshot. `xcat_mushroom_only.yaml`
+ `single_lift_xcat_mushroom.yaml` (untracked, repo root) are a manual top-up
recipe pinning the pool to `["mushroom"]` alone to bring it back to parity —
i.e. the real fix was "notice the gap after the fact and run a targeted
top-up," not "prevent the gap." See §7 for the actual fix this deserves.

**Orchestration** (`gentle_manip/scripts/arrhenius/yd_xcat_pipeline.sbatch`,
committed): convert → BC pretrain → canonical eval, chained after collection.
Supports resuming from a `--resume-dir`, a "PRELIM" mode that stages a
still-running collection's shards into a scratch dir without touching the live
run (merging in place would race with the live collector's own end-of-run
merge and could drop episodes), and a second "regrasp-start" eval variant
(arm starts low over the object) alongside the clean canonical eval.

---

## 6. The gen8 generalist: does the recovery data actually help?

**Staging** (`gentle_manip/scripts/arrhenius/_gen8_stage.py`, untracked —
**two copies exist, see §8**): merges every xcat collector run
(`single_lift_xcat_regrasp/*/`, `single_lift_xcat_small/*/`) plus the
`banana_lying` 500-episode run into one `data.pkl`. `--exclude-modes
failed_grasp` drops only the post-failure-recovery family, keeping every other
start-mode's diversity intact — this produces the **fair control** dataset:
same 8 objects, same direct-start diversity, only the explicit-recovery demos
removed.

**Two training variants, byte-identical except their data source**
(`gentle_manip/scripts/arrhenius/yd_gen8_pipeline.sbatch`, untracked):
- `VARIANT=regrasp` (default) → `single_lift_gen8_regrasp_pcd`: **all** start
  families, including `failed_grasp`.
- `VARIANT=baseline` → `single_lift_gen8_baseline_pcd`: everything except
  `failed_grasp`.

Both configs (`gentle_manip/dppo/cfg/single_lift_gen8_{regrasp,baseline}_pcd/`)
were just scaled up for the ~8x larger (8-category) dataset relative to the
single-banana configs they were forked from: `mlp_dims` 512×3→768×3,
`visual_feature_dim` 256→384, batch 128→256, `n_epochs` 250→300,
`early_stop_patience` 8→15 — kept in lock-step between the two variants
deliberately, so the only real difference is the recovery data.

**Eval sweep** (`gentle_manip/scripts/arrhenius/yd_gen_eval.sbatch`, modified
this branch): loops 8 in-domain categories (`mushroom, banana_lying, kiwi,
egg_boiled, grape, cherry, tomato, raspberry`) + 4 held-out OOD categories
(`blackberry, scallop, dumpling, gelatin`, matching the earlier GRACE
zero-shot set) through the canonical 100-episode harness
(`scene_group_size=1` — geometry+material re-randomized every 5 episodes),
computing success rate / gentleness / SR×gentleness per category plus
in-domain and OOD means. Recent changes on this branch:
- **Concurrency** (`NCAT_PAR`, default 3): runs that many categories'
  sim-server+eval pairs at once on one GPU (~7GB each), roughly a 2x wall-clock
  speedup, staggered 20s apart so Genesis inits don't collide.
- **Per-category episode budget** ("per-cat quota" in the commit log — a
  *different* quota from §5's collection-time one): the four small-fruit
  categories use a 6x-slower finer-grid eval task and are capped at 50
  episodes (`SMALL_NEP`) instead of 100, so the full 12-category sweep still
  fits one walltime window.
- **Self-resubmit**: after the sweep, any category still missing a
  `summary.json` (walltime cutoff) gets resubmitted as a fresh job scoped to
  just the missing categories, reusing the same checkpoint/output dir — same
  pattern as `yd_xcat_pack.sbatch`'s collector self-resubmit.
- **Walltime**: `yd_gen8_pipeline.sbatch` was tightened 20h→12h (matches its
  actual observed cost now that staging + convert + BC-pretrain don't need
  20h); `yd_gen_eval.sbatch` stayed at 24h given the larger 12-category ×
  100/50-episode sweep with video.

**`mesh_deform.py` fix** (`gentle_manip/assets/mesh_deform.py`, modified this
branch): bend/taper/axis_scale shape DR could drop a deformed mesh's lowest
vertex a few mm below the *nominal* mesh's bottom; since the scene builder
places objects at the nominal `default_pos`, a deformed mesh could clip
through the floor and MPM would raise "particles outside solver boundary."
Fix re-seats the deformed mesh's bottom (z only) to match the nominal bottom
after every deformation draw, before writing it out. This is a general
scene-DR fix (affects any category using shape DR), not gen8-specific, but it
surfaced during this campaign's mushroom/kiwi collection.

**Status**: as of this mirror, the eval sweep is what's queued/running — no
result numbers are readable from this local checkout (see the top-of-doc note
on where data actually lives).

---

## 7. What's missing / known issues

- **No results in hand yet, from this side.** This checkout can't see
  `aggregate.json` / wandb / `summary.json` for the gen8 regrasp-vs-baseline
  comparison. Before drawing any conclusion from an SR number, apply the §3
  lesson: pull eval videos specifically from OOD/small-fruit categories and
  from episodes where the first approach visibly missed, and confirm genuine
  recovery is visible — not just a higher aggregate SR.
- **Per-category collection quota is still a log-scrape workaround, not a real
  fix** (§5). The manual mushroom top-up patches one instance; the next
  multi-category packed run will hit the same undershoot/overshoot pattern
  for whichever category happens to be slow. A real fix needs either a shared
  file-based counter each collector locks/increments, or a single dispatcher
  process handing out categories to worker collectors on demand.
- **Duplicate, diverging `_gen8_stage.py` and `yd_gen8_pipeline.sbatch`.**
  Both exist at `gentle_manip/<name>` (repo root) *and*
  `gentle_manip/scripts/arrhenius/<name>` (untracked, both). The root copies
  are **stale drafts**, not just accidental duplicates:
  `gentle_manip/yd_gen8_pipeline.sbatch`'s baseline variant filters to
  `--modes home,near_object` (throws away the `sweep`/`above`/`ground`/`air`
  diversity too — not a fair control) and still has the old 20h walltime; the
  `scripts/arrhenius/` copies have the corrected `--exclude-modes
  failed_grasp` control and 12h walltime. Anyone running the root copy would
  silently get the wrong (unfair) baseline. Needs a decision: delete the root
  copies, or fold their differences in and delete once reviewed.
- **`gentle_manip/xcat_mushroom_only.yaml` and
  `gentle_manip/single_lift_xcat_mushroom.yaml`** sit at the `gentle_manip/`
  package root instead of `configs/dr/` and `configs/experiments/`
  respectively, where every other DR/experiment config lives. Low risk (they
  still work if referenced by full relative path) but breaks the "configs are
  organized by role" convention from the top-level `CLAUDE.md` and would
  confuse anyone grepping `configs/dr/` for the mushroom-only recipe.
- **ReTVL's weighted-sampling fix was designed but never retrained/evaluated**
  (§3) — abandoned mid-flight when the OmniReset pivot took priority, not
  because it was shown not to work. If diverse-start ever plateaus below
  target, this is a half-finished alternative worth resuming rather than
  reinventing.
- **`docs/fragile25_pipeline_onboarding.md` (the base architecture doc) does
  not mention `xcat` or `gen8` at all** — this doc fills that gap for now, but
  the onboarding doc is the one a new teammate is told to read first, and it
  currently describes only the specialist→RLDG→generalist arm, not the
  diverse-start arm this doc covers. Worth a follow-up pass to merge the two
  once gen8 results land.
- **No committed report/webpage builder for the gen8 sweep.** The specialist/
  generalist campaign had `build_report_v3.py`, referenced in
  `cross_category_specialist_log.md` but living only in a scratch directory,
  never checked into the repo. The `yd_gen_eval.sbatch` header mentions
  clips "feed the montage webpage," but that webpage-building script isn't in
  this tree either — it will need to be rebuilt or located before the gen8
  results can be presented the same way.

---

## 8. How to reuse this

**Run one category's diverse-start collection standalone** (smoke-test before
committing to a multi-hour packed run):
```bash
env -u PYTHONPATH -u ROS_DISTRO MUJOCO_GL=egl \
  uv run --project envs/sim python grasp_synthesis/collect_demos_diverse_start_v2.py \
    --experiment single_lift_<cat>_rigid_diverse --n-episodes 20 --n-envs 5 \
    --maxfevals 400 --start-modes "sweep:0.44,failed:0.30,above:0.10,ground:0.09,air:0.07" \
    --record-video 5 --seed 0
```

**Packed cross-category collection on the cluster** (large + small pools,
per-cat quota, self-resubmitting):
```bash
sbatch gentle_manip/scripts/arrhenius/yd_xcat_pack.sbatch
# knobs: N_PACK (3) WALL PER_CAT_TARGET (500) SEED0 (30) START_MODES
```

**Single-category convert→train→eval after collection**:
```bash
sbatch gentle_manip/scripts/arrhenius/yd_xcat_pipeline.sbatch
# DEMO_RUN=<run dir>  SKIP_TRAIN=1 RUN_DIR=<pretrain dir>  # to jump straight to eval
```

**Full 8-object generalist, both variants**:
```bash
sbatch gentle_manip/scripts/arrhenius/yd_gen8_pipeline.sbatch            # regrasp (default)
VARIANT=baseline sbatch gentle_manip/scripts/arrhenius/yd_gen8_pipeline.sbatch
```

**Re-run or extend the per-category eval sweep on an existing checkpoint**:
```bash
ENV_NAME=single_lift_gen8_regrasp_pcd RUN_DIR=<pretrain run dir> \
  NCAT_PAR=3 RECORD_BATCHES=3 \
  sbatch gentle_manip/scripts/arrhenius/yd_gen_eval.sbatch
# CATS="mushroom kiwi" to re-run a subset; OUT=<existing gen_eval dir> to resume into it
```

**To add a 9th category to the gen8 roster**: register it (onboarding doc §2
checklist), add it to whichever DR pool config fits its size
(`xcat_diverse_regrasp.yaml` or `xcat_small_diverse_regrasp.yaml`, or a new
pool if it needs its own shape-DR treatment), collect it via
`collect_demos_diverse_start_v2.py`, then add it to `_gen8_stage.py`'s
`SRC_GLOBS`/`SOFT_BANANA`-style source list and to both the `INDOMAIN`/`OOD`
lists in `yd_gen_eval.sbatch`.

**Key files, all in one place**:

| File | Role |
|---|---|
| `grasp_synthesis/collect_demos_diverse_start_v2.py` | diverse-start collector; `_sample_start` (start families), `execute_and_collect_diverse_v2` (per-env FSM) |
| `grasp_synthesis/collect_demos_synth_v2.py` | DART disturbance injection (`execute_and_collect`, idea #3) + lift-slip regrasp (idea #2) |
| `grasp_synthesis/collect_demos_synth_v3.py` | `--retry-on-slip` / `--fast-reattempt` single-category regrasp collector |
| `gentle_manip/dppo/eval_agent_tide.py` | TIDE/FAR monitor+perturbation (negative result, not in active use) |
| `gentle_manip/dppo/retvl_*.py`, `train_retvl_value.py`, `build_retvl_*.py` | ReTVL value-weighted BC (paused mid-fix) |
| `gentle_manip/configs/dr/xcat_diverse_regrasp*.yaml`, `xcat_small_diverse_regrasp*.yaml` | xcat DR category pools |
| `gentle_manip/scripts/arrhenius/yd_xcat_pack.sbatch` | packed multi-collector cross-category collection |
| `gentle_manip/scripts/arrhenius/yd_xcat_pipeline.sbatch` | single-run convert→train→eval |
| `gentle_manip/scripts/arrhenius/_gen8_stage.py` | merge all xcat + banana_lying demos into one gen8 dataset |
| `gentle_manip/scripts/arrhenius/yd_gen8_pipeline.sbatch` | stage→convert→train→submit-eval for the 8-object generalist, both variants |
| `gentle_manip/scripts/arrhenius/yd_gen_eval.sbatch` | 12-category (8 in-domain + 4 OOD) canonical eval sweep |
| `gentle_manip/dppo/cfg/single_lift_gen8_{regrasp,baseline}_pcd/` | the A/B training configs |

---

## 9. Suggested next steps

1. **Let the gen8 eval sweep finish, then do the video review before trusting
   any SR number** — the §3 lesson applies directly here: pull clips for
   `failed_grasp`-adjacent scenarios per category (both in-domain and the 4
   OOD categories) and confirm the regrasp variant shows genuine
   redescend/recover, not just a higher aggregate number.
2. **Fix the per-category collection quota properly** (§7) before the next
   large packed collection run — a shared counter file or a single dispatcher
   process, not per-collector log-scraping, so mushroom-style undershoot
   doesn't recur on the next new category.
3. **Resolve the duplicate `_gen8_stage.py` / `yd_gen8_pipeline.sbatch`
   files** — delete the stale root-level copies (or confirm nobody is
   pointing at them) so a future run can't accidentally use the unfair
   `home,near_object`-only baseline.
4. **Move the two misplaced xcat configs** into `configs/dr/` and
   `configs/experiments/` to match the rest of the config tree.
5. **If gen8's regrasp variant beats baseline with video-confirmed recovery**:
   fold this into `docs/fragile25_pipeline_onboarding.md` as a second
   generalist arm (alongside RLDG and direct-generalist), since it's now a
   third, competing way to build a generalist policy and a new teammate needs
   to know it exists.
6. **If it doesn't** (regrasp ≈ baseline, or videos show no real recovery):
   the ReTVL weighted-sampling variant (§3, §7) is the most promising
   half-finished alternative to resume, since its core fix (avoid breaking
   `cond_steps` continuity) directly addresses the failure mode diagnosed
   last time.
7. **Recover or rebuild a committed report/webpage builder** for the gen8
   12-category × 3-metric comparison — don't let this repeat the earlier
   pattern of a report script that only ever lived in `/tmp`.
