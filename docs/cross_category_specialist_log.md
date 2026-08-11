# Cross-Category Specialist/Generalist — Working Log

Branch: `cross-category-dp`. Running log of this work stream — key decisions, results,
and literature insights, kept human-readable and updated as things happen. Not a
replacement for `EXPERIMENT.md` per training run or the full research plan at
`~/.claude/plans/how-it-makes-trajectory-distributed-sun.md`.

**Structure**: a high-level **Summary** (read this first — current status, verdicts,
next steps), then the full **Chronological Log** below it (detailed blow-by-blow,
preserved for anyone who needs the "why" behind a specific number or decision).

---

# Summary

## Goal — two-phase plan, priority order confirmed by user

1. **Specialist-first**: get a single-category DP3/DPPO policy to **60-80%** success
   on the canonical eval harness (n=100, num_envs=5, seed=0).
2. **Once a specialist clears 60%**, move to the cross-category generalist policy.
3. Generalist conditioning direction (decided ahead of time): a **low-dimensional,
   VLM-based** semantic conditioning vector — explicitly **not** a one-hot category ID,
   explicitly kept low-dimensional.
4. Demo recovery behavior (grasp-drop-retry) is **deprioritized** — built and working,
   but not worth further polish unless revisited explicitly.

## Verdict 1 — Phase 1 (specialist reaches 60-80%): CONCLUSIVELY VALIDATED

Four independent "easy" (large object + narrow DR) specialists, identical
architecture/recipe, canonical eval n=100:

| Object | Size (mm) | Grasp-margin flag | Collection SR | **Solo eval SR** |
|---|---|---|---|---|
| apple | 65x60x65 | severe (45°→20°) | 43.5% | **65.0%** |
| avocado | 62x95x61 | severe (45°→20°) | 18.1% | 52.0% |
| kiwi | 43x46x60 | none (kept 45°) | 52.6% | 52.0% |
| pear | 62x52x65 | moderate (45°→30°) | 44.7% | 44.0% |
| cherry (tiny, wide DR) | 20x17x20 | n/a | ~80% | 25.7% (n=3 seed mean, 15-33% range) |

**Task difficulty (object size + DR range + grasp margin), not the architecture or
demo-collection pipeline, was the ceiling on cherry all along.** Every "easy" object
beats cherry by 18-40 points despite identical training. Cherry recipe-tuning
(disturbance injection, dataset size, disturbance-phase choice) is a **closed thread**
— see the timeline below for the full journey (13-33% single-seed spread → n=3 mean
25.7% once seed variance was accounted for).

**Nuance** (doesn't overturn the verdict): the fine ordering among the four "easy"
objects (65% > 52% ≈ 52% > 44%) doesn't reduce to one simple rule — grasp-margin
severity alone doesn't predict it (pear scored lowest despite a milder flag than
avocado; kiwi tied avocado despite no flag at all). Likely a mix of absolute size,
grasp-margin comfort, and per-mesh CMA-ES search difficulty. Apple is the standout
best performer; empirical per-category eval remains necessary, not a size lookup.

## Verdict 2 — Phase 2 (naive unconditioned merge): PROVEN INSUFFICIENT

Merged apple-easy + avocado-easy (160 demos, both independently confirmed 52-65%
solo) into one **unconditioned** BC policy:

| | Solo | Merged (unconditioned) |
|---|---|---|
| apple (**held-in**) | 65.0% | 39.0% (−26 pts) |
| avocado (**held-in**) | 52.0% | 31.0% (−21 pts) |
| kiwi (**zero-shot**, never in the merge) | 52.0% (as solo) | **4.0%** (collapse) |

Both held-in categories regress substantially, and zero-shot generalization
collapses to the same magnitude as this project's original (pre-this-session)
11-category 2.0% baseline. **Not a data-quality confound this time** — both source
categories are independently confirmed good. This is real evidence that the
category-conditioning branch (Stage 5 of the original research plan) is
**necessary, not optional** — a bare merge doesn't work at this model capacity.

## Current work in progress — conditioning experiments (not yet evaluated)

Two parallel training runs on the same apple+avocado merge, testing whether
conditioning fixes Verdict 2:

- **Track A** (`single_lift_cross_category_easy_conditioned_pcd`, run `cfvpj`) — the
  pre-existing registry-derived embedding (`category_embedding.py`: 15-dim one-hot +
  6 continuous material/size/shape features = 21-dim). Cheap calibration check: this
  exact embedding was tried once before (pre-this-session), scored 0-4.5%, written off
  as confounded by then-weak specialist data. That confound is now resolved (52-65%
  solo specialists), so re-running it is informative. **Caveat: 15/21 dims are a
  literal one-hot — this is NOT what the user's standing directive specifies**, it's
  a fast sanity check using infra that already existed.
- **Track B** (`single_lift_cross_category_easy_vlm_pcd`, run `atwfy`) — the user's
  actual specified direction: a frozen CLIP (ViT-B/32) embedding of a canonical scene
  frame, reduced via a fixed random projection to 24-dim (`vlm_embedding.py`, built
  fresh this session). No discrete category notion at all — generalization depends on
  continuity in CLIP embedding space, a genuinely more open-world zero-shot test than
  Track A's reserved-but-untrained one-hot slot.

Both will be evaluated the same way as the baseline (held-in apple, held-in avocado,
zero-shot kiwi) once they plateau, for a direct 3-way comparison against the
unconditioned 39%/31%/4.0% numbers above.

**Infrastructure notes**: `envs/dppo` manages torch manually outside `pyproject.toml`
and was actively training when Track B was built, so `transformers`/CLIP was
installed only in `envs/sim` and Track B's embeddings are precomputed once there
(`gentle_manip/scripts/precompute_vlm_embeddings.py`) into a disk cache
(`gentle_manip/assets/category_reference_frames/vlm_embed_cache.npz`) that
`envs/dppo` reads with plain numpy — `envs/dppo` was never touched/synced. An
`--embed-source {registry,vlm}` flag threads through `convert_demos.py`,
`merge_cross_category_demos.py`, and `genesis_venv.py` (+ the `third_party/dppo`
submodule) so either source works through the identical `category_embed_dim`
mechanism in `PointNetDiffusionMLP`.

## Key literature grounding

| Finding | Source | Relevance |
|---|---|---|
| DP3 gets 85% real-robot success with only 40 demos | [3D Diffusion Policy, RSS 2024](https://3d-diffusion-policy.github.io/) | Demo *count* was never the likely bottleneck here — confirmed: task difficulty was. |
| DART: inject disturbances *during* collection for genuine recovery behavior | [Laskey et al. 2017](http://proceedings.mlr.press/v78/laskey17a/laskey17a.pdf) | Tried (lift-phase, grasp-phase); real but modest effect (13-33%), mostly swamped by seed variance. |
| RLDG: specialists → rollout → distill into ONE generalist via BC; beats distilling raw demos | [arXiv 2412.09858](https://arxiv.org/abs/2412.09858) | Not yet tried — the natural next step once a conditioning approach shows signal: rebuild the merge from specialist ROLLOUTS, not raw CMA-ES demos. |
| PA3FF/PADP: part-aware 3D feature fields beat CLIP/DINOv2 for cross-category/unseen-object generalization | [arXiv 2602.14193](https://arxiv.org/abs/2602.14193) | Later-tier upgrade if Track A/B's simpler conditioning proves insufficient. Heavier than what's appropriate for a first attempt. |

## Key operational notes (things that cost real time this session)

- **DPPO checkpoint-resume now works**: `+resume_from=<run>/checkpoint/state_<N>.pt`
  on the hydra CLI (`third_party/dppo` commit `ed7bb3b`, parent bump `acd67ce`).
  Optimizer/LR-scheduler state isn't restored (fine for BC). Resuming mints a NEW
  hydra run dir — checkpoints land there, not in the original run's directory.
- **Always prefix sim-touching commands with `env -u PYTHONPATH`** in this shell — a
  stray `PYTHONPATH` entry (`.../Genesis_fork:...`, an old sibling clone) shadows the
  correct editable genesis install and causes `ModuleNotFoundError: No module named
  'gstaichi'` otherwise.
- **`scene_group_size=0`** is the standing eval workaround for an unresolved
  `scene_group_size>0` geometry-rebuild RPC hang — fine for internal apples-to-apples
  comparison, not using the intended shape/scale DR coverage. Fix before reporting any
  number externally.
- **`merge_cross_category_demos.py --demos-root` must be an absolute path** (or
  omitted — the default already is) — a relative path breaks its internal symlinks.
- **`envs/dppo` manages torch manually, outside `pyproject.toml`** — avoid `uv sync
  --project envs/dppo` while a training run is active in it (risks silently swapping
  CUDA torch for CPU torch mid-run); if a new dependency is needed there, precompute
  into a cache from a different env instead, as done for Track B.

## Next steps

1. Evaluate Track A and Track B (6 evals: held-in apple/avocado + zero-shot kiwi,
   each track) — in progress, see Chronological Log for live updates.
2. Compare both against the unconditioned 39%/31%/4.0% baseline and against each
   other (registry one-hot+features vs. true VLM embedding).
3. If either conditioning approach shows real signal: consider RLDG-style
   specialist-rollout distillation for the next merge, and/or add a 3rd/4th "easy"
   category (mushroom, a second pear-class object) to strengthen the n=2 held-in /
   n=1 zero-shot evidence base.
4. If neither helps: escalate to PA3FF/PADP-style part-aware conditioning, or
   reconsider whether BC-from-merged-demos is the right training paradigm at all
   for this model capacity.

---

# Chronological Log

## Key literature insights (as first compiled)

| Finding | Source | Why it matters here |
|---|---|---|
| DP3 gets **85% real-robot success with only 40 demos** | [3D Diffusion Policy, RSS 2024](https://3d-diffusion-policy.github.io/) | We have ~45-250 demos/category at 0-4.5% success — raw demo *count* is very unlikely to be the primary bottleneck; quality/diversity and task difficulty are the more likely levers. |
| **DART**: inject disturbances *during* demo collection (not post-hoc) so demos contain genuine recovery behavior | [Laskey et al. 2017, PMLR v78](http://proceedings.mlr.press/v78/laskey17a/laskey17a.pdf) | Named precedent for the compounding-error failure mode our eval videos showed. **Caveat**: 9-year-old paper — per user direction, prefer newer treatments of the same idea when available; kept here as the clearest statement of the mechanism. |
| Demo **diversity across environments/objects** matters more than raw count past a per-object threshold | [ICLR 2025 data-scaling-laws paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/88b7b2c896506daabc8d3fd587055167-Paper-Conference.pdf) | Supports trying disturbance-injected demos over just proportionally scaling the existing (undiversified) collection recipe. |
| **RLDG**: train specialists → roll them out to generate expert trajectories → distill into ONE generalist via BC; up to 40% higher success than distilling from raw demos directly | [arXiv 2412.09858](https://arxiv.org/abs/2412.09858), [project page](https://generalist-distillation.github.io/) (Dec 2024) | Named version of the user's own two-phase plan. Once specialists work, the cross-category generalist dataset should be built from **specialist rollouts**, not raw CMA-ES demo pickles. Backbone-agnostic (OpenVLA and Octo both validated) — our DP3/DPPO stack is a fine substitute. |
| **PA3FF / PADP**: part-aware dense 3D feature field (contrastive-pretrained on 3D part proposals), fed into a diffusion policy instead of raw point clouds; beats CLIP/DINOv2/Grounded-SAM; 28.8% vs GenDP's 19.4% on PartInstruct, only 6.25% drop on unseen objects | [arXiv 2602.14193](https://arxiv.org/abs/2602.14193) (Feb 2026) | Current frontier for cross-category/unseen-object generalization — exactly our zero-shot failure mode. Swap-in point is the `DP3Encoder` insertion (`pointnet_extractor.py:276`). Full architecture internals weren't extractable from the abstract alone — get the full paper text before implementing. Note: heavier/higher-dim than what the user now wants for the *first* generalist attempt — treat as a later upgrade, not the starting design. |

**Net implication for build order**: don't reintroduce cross-category conditioning
complexity until the underlying specialist data quality is fixed — Stage 5(A)'s early
negative result may just reflect garbage-in/garbage-out from 2-4.5%-success
specialists, not that conditioning doesn't help.

---

## Results so far (early session)

| Config | Data | Collection success | Eval (canonical, n=100) | Notes |
|---|---|---|---|---|
| Cross-category baseline (11 cat. mixed) | mixed | — | 2.0% (2/100) | Stage 5-7 baseline |
| + category-embedding (one-hot), epoch 300 | mixed | — | 1.0% | Within noise of baseline |
| + category-embedding (one-hot), epoch 500 | mixed | — | 0.0% | Within noise of baseline |
| Mushroom-solo (50 ep) | mushroom only | — | 4.5% (9/200) | >2x the mixed baseline — first data point suggesting specialize-per-category beats one shared policy |
| Cherry zero-shot (from mixed baseline) | — | — | 1.0% (1/100) | The number cherry-solo results below should beat |
| **Cherry-plain-250** | 250 ep, no disturbance | 80.1% (250/312 attempts) | **17.0% (17/100)** — run `nxyyn`, checkpoint 700 | Plateaued best val loss 0.0092 @ epoch 740 |
| **Cherry-disturbed-180** | 180 ep (recovered from a 250-ep target killed at timeout), `disturbance_prob=0.3`, `max=2cm` during "lift" | 80.0% (180/225 attempts) — **identical to plain**, confirms disturbance injection doesn't hurt the demonstrator | **33.0% (33/100)** — run `myspw`, checkpoint 600 | Plateaued best val loss 0.0098 @ epoch 560. **Best result this session by a wide margin** at the time — see below for how this evolved once seed variance was measured. |

### Demo videos (for visual inspection)

- Cherry-plain-250 successes: `dataset/demos/single_lift_cherry_rigid/26-08-09-wbi/videos/` (250 clips)
- Cherry-plain-250 failures: `dataset/demos/single_lift_cherry_rigid/26-08-09-wbi/videos_failed/` (62 clips)
- Cherry-disturbed-180 successes: `dataset/demos/single_lift_cherry_rigid/26-08-09-cts/videos/` (180 clips)
- Cherry-disturbed-180 failures: `dataset/demos/single_lift_cherry_rigid/26-08-09-cts/videos_failed/` (45 clips)
- Regrasp-retry smoke test (tofu, `dataset/demos_smoketest_regrasp3/single_lift_tofu_rigid/26-08-09-qmp/`):
  the three failure clips below are confirmed (via log correlation) to contain an
  actual mid-episode drop + regrasp attempt:
  - `videos_failed/fail0001_b1_env0.mp4`
  - `videos_failed/fail0002_b1_env1.mp4`
  - `videos_failed/fail0003_b2_env0.mp4`

  **Known rough edge, seen in these clips**: the rewind-to-"settle" target is a
  one-step jump with no interpolation from the env's current (still-elevated)
  position, so the arm's PD controller drives fast toward it — visually a fast,
  uncontrolled-looking motion that can clip the table. Documented as a TODO in
  the code; **deprioritized** per user direction rather than fixed.

---

## Timeline / decision log (early session)

- **Baseline + category-embedding eval**: all three land in 0-2% success,
  statistically indistinguishable at n=100 (Clopper-Pearson CIs overlap heavily).
- **Mushroom-solo control**: training on ONLY mushroom demos more than doubled the
  mixed-policy success rate on mushroom (4.5% vs 2.0%) — first signal that
  specialization matters.
- **User redirect #1**: *"success rate too low, expect 60-80%, train specialist
  first via more/better demos, then generalist via literature-grounded
  architecture; explain category conditioning; don't stop."* Superseded a planned
  cherry-solo-at-50-episodes confirmation run in favor of a bigger push: 250-episode
  cherry collection + DART-style disturbance injection.
- **Implemented DART-style disturbance injection** (idea #3: one-step positional
  kick during "lift", off by default). Smoke-test false alarm traced to a reduced
  `maxfevals=400` test budget, not the new code; re-tested at standard
  `maxfevals=800`, disturbance ON/OFF matched (47%/47%) — neutral to collection
  success as designed.
- **Collected cherry-plain-250** (80.1%) and **cherry-disturbed-250-target** (killed
  by the orchestrator's 5400s timeout at 180/250 episodes).
- **Bug found + fixed**: orchestrator's data.pkl glob had no time lower-bound, so it
  silently reported the disturbed attempt's result using a STALE data.pkl from the
  earlier plain run. Recovered the real 180 episodes from un-merged shards via
  `_merge_shards()`; patched the glob to filter by call-start time. Committed
  `c646cfb`.
- **Launched both cherry-250 trainings** (`nxyyn` plain, `myspw` disturbed) + their
  canonical eval configs. Committed `081f970`.
- **User redirect #2**: replace one-hot category conditioning with **low-dimensional**
  VLM-based (softness/size/geometry) conditioning; add genuine slip-and-retry
  behavior to demos; prioritize recent (1-3yr) literature going forward; move to
  generalist work once specialist SR clears 60%.
- **Implemented lift-phase failure detection + regrasp** (idea #2, previously a
  `TODO(retry)` stub): on a detected under-height drop at the end of "lift", rewind
  to "settle" and re-run settle→grasp→firm→lift, bounded to 1 retry/env by default.
  Smoke-tested on tofu — **confirmed triggering correctly** (4 real triggers, log +
  video-length correlation checked).
- **User caught two real issues on video review**: (1) a video from the
  `--keep-failures` smoke test was mislabeled `..._success.mp4` despite being a
  genuine failure — traced to the save loop unconditionally hardcoding the
  `_success` suffix on the `--keep-failures` fallthrough path; **fixed** (now
  `success`/`keptfail`). (2) the regrasp rewind causes a visibly fast,
  table-clipping motion — root-caused to the missing re-approach interpolation
  (documented as TODO, not fixed). Committed `1e484c4`.
- **User redirect #3**: retry/recovery work is **lower priority** — don't keep
  polishing it if it's not easy; refocus on (1) specialist SR, (2) generalist.
  Confirmed VLM conditioning should be **low-dimensional**.
- **Early-stop plateau watchers armed** for both trainings (`nxyyn`, `myspw`): each
  fires once val loss hasn't improved by >0.0005 for 300 epochs (30 val
  checkpoints), so neither run rides out its full 2000-epoch budget unnecessarily.

---

## Rollout sanity-check finding (2026-08-09, mid-training)

Per user suggestion: don't rely on BC loss alone — periodically roll out a checkpoint
and watch the actual behavior, since low loss doesn't guarantee correct closed-loop
success (this is exactly how the 2-4% mixed-baseline numbers could hide a real bug).

**Checked `nxyyn` (cherry-plain-250) at checkpoint `state_800` (val loss ~0.009-0.010,
near its best):**
- Quick 15-episode rollout (fixed nominal geometry, no video): **6.7% success (1/15)**
  — better than the 2.0% mixed baseline, but far from 60-80%, despite quite low BC loss.
  This loss-vs-rollout-SR gap is exactly the red flag the user was watching for.
- Re-ran 5 episodes WITH video and inspected frames directly (not just the number).
  **The pipeline is NOT obviously broken**: the one success episode shows a fully
  correct approach → close-near-object → lift-with-object-pinched → hold sequence —
  obs/action wiring, scaling, and signs all look right when it works.
- **But the dominant failure mode (4/5 episodes, and matches the earlier 15-episode
  run's video-free numbers) is a specific, repeatable pattern, not random noise:**
  the arm reaches to the correct vicinity of the object and then **hovers directly
  above it, gripper open, motionless, for ~100 steps** without ever closing — then
  gives up, **closes the gripper empty while already retracting away from the
  object**, and returns toward home. The object never moves in any failed episode
  (no sign the gripper ever made contact).
- **Interpretation**: this looks like a grasp-timing/trigger **reliability** problem
  rather than a wiring bug — the model has clearly learned the right behavior (proven
  by the success case) but the gripper-close decision isn't reliably triggered by
  actually being at the object; the close eventually fires on some other basis
  (time-since-episode-start?) and lands in the wrong place. Plausible contributors:
  cherries are small (~1.5-2cm) so required grasp precision is demanding; diffusion
  chunked-action policies can learn coarse temporal patterns that aren't tightly
  contingent on fine spatial alignment without more demos or an explicit
  contact/proximity signal.
- **This is a different diagnosis than "compounding drift after a good attempt"**
  (the failure mode the disturbance-injection dataset targets) — it's closer to
  "never commits to the grasp in the first place." Worth keeping in mind when
  interpreting the disturbed-180 run's eventual eval number: disturbance injection
  targets recovery-from-drop, not this hover-and-miss pattern, so it may not help
  much with what's actually the dominant failure mode here.

**Tooling note**: extracted frames from eval clips via `ffmpeg -vf select=...` and
inspected them directly (Read tool, image mode) rather than just reading the
success/fail number — this is now the recommended quick-diagnosis method when a
success rate looks suspiciously low relative to loss. Sim server for ad-hoc checks
needs `--subprocess` if `scene_group_size>0` is used; found that a `scene_group_size>0`
rebuild RPC can hang the eval client for an unrelated reason (not yet root-caused,
worked around with `scene_group_size=0` for quick checks — full canonical eval still
uses the real `scene_group_size=4`, this workaround is quick-check-only).

---

## Canonical eval results (2026-08-09)

Both trainings hit their plateau watchers and were stopped cleanly (process-group-safe):
- `nxyyn` (cherry-plain-250): plateaued at best val loss **0.0092** (epoch 740), stopped
  at epoch 1040. Checkpoint used: `state_700.pt` (closest saved checkpoint to true best).
- `myspw` (cherry-disturbed-180): plateaued at best val loss **0.0098** (epoch 560),
  stopped at epoch 860. Checkpoint used: `state_600.pt`.

**Deviation from strict canonical protocol, noted explicitly**: both evals ran with
`scene_group_size=0` (fixed nominal geometry) instead of the spec'd `4`, because a
`scene_group_size>0` geometry-rebuild RPC hung the eval client indefinitely during an
earlier quick check (root cause not yet found). `n_episodes=100`, `num_envs=5`,
`seed=0` (all canonical/unchanged); both variants evaluated identically, so the
plain-vs-disturbed comparison is still apples-to-apples, just without shape/scale DR
coverage in this particular pair of numbers.

### Result: DART-style disturbance injection is a large, real win

| Checkpoint | Success (n=100, canonical) | vs. mixed baseline (2.0%) | vs. 60-80% target |
|---|---|---|---|
| **myspw** (cherry-disturbed-180) | **33.0%** (100/100 episodes) | **16.5x** | Still short, but by far the best result at this point |
| **nxyyn** (cherry-plain-250) | *running* | — | — |

### Final head-to-head: plain-250 vs disturbed-180 (both canonical, n=100, matched protocol)

| Checkpoint | Success (n=100) | Notes |
|---|---|---|
| `nxyyn` (cherry-plain-250, 250 ep, no disturbance) | **17.0%** (17/100) | |
| `myspw` (cherry-disturbed-180, 180 ep, disturbance_prob=0.3) | **33.0%** (33/100) | Fewer episodes, ~2x the success rate |

**Conclusion at the time: DART-style disturbance injection is a large, isolated,
real win** — the disturbed dataset has FEWER episodes (180 vs 250) yet nearly
doubles success rate. This cleanly separates the disturbance-injection effect from
a raw-count effect. Both still fall short of the 60-80% target. (This conclusion
was later revised once seed variance was measured — see below.)

**Recommended next step (executed)**: collect a FULL, clean 250-episode
disturbance-injected cherry dataset with a longer collection timeout, train a
specialist on it the same way, and re-eval.

---

## Disturbed-250-v2: collection done, training launched (2026-08-09 ~20:15)

Full, uninterrupted collection finished: **250/250 saved, 79.87% success (313
attempts)** — consistent with plain-250 (80.1%) and disturbed-180 (80.0%), confirming
disturbance injection doesn't hurt collection success even at full scale. Data:
`dataset/demos/single_lift_cherry_rigid/26-08-09-zyv/data.pkl`.

Converted (225 train / 25 val episodes, 48713 train steps) and training launched:
run `vrkjr`, configs at `gentle_manip/dppo/cfg/single_lift_cherry_rigid_pcd_250_disturbed_v2/`
(committed `ddb9c1d`). Plateau watcher armed.

---

## Disturbed-250-v2 canonical eval — DONE (2026-08-09 ~23:25)

**Result: 24.0% (100/100 episodes)** — checkpoint `state_500.pt` (best val loss 0.0091,
notably lower/better than disturbed-180's 0.0098).

### Full three-way comparison (canonical, n=100, matched protocol)

| Checkpoint | Data | Best val loss | Success (n=100) |
|---|---|---|---|
| `nxyyn` (cherry-plain-250) | 250 ep, no disturbance | 0.0092 | **17.0%** |
| `myspw` (cherry-disturbed-180) | 180 ep, disturbance_prob=0.3 | 0.0098 | **33.0%** |
| `vrkjr` (cherry-disturbed-250-v2) | 250 ep, disturbance_prob=0.3 | 0.0091 (best of the three) | **24.0%** |

**Key finding: scaling the disturbance-injected dataset from 180→250 episodes did NOT
improve success rate further — it went DOWN (33.0%→24.0%), despite the 250-episode
checkpoint having the LOWEST (best) BC validation loss of all three runs.** This is a
loss-vs-rollout-SR disconnect, now a second independent data point. At n=100 the
binomial CI on both numbers is roughly ±9pp, so 24% and 33% are not dramatically
outside each other's noise band — **disturbance injection lifts success into the
~24-33% range**, a real win over plain-250's 17%, but simply collecting MORE
disturbance data does not keep buying further gains.

**Confirmed via direct video inspection**: the disturbed-250-v2 failure episode shows
the EXACT SAME failure pattern found in the very first diagnostic (nxyyn): arm
reaches the object, then retracts fully to home with the gripper never having closed
— object left completely untouched. This pattern is present across all three trained
checkpoints regardless of dataset composition — strong evidence this is a
**structural precision/reliability limitation of the current setup**, not something
more of the same kind of data collection would fix.

---

## Diagnostic deep-dive + grasp-phase disturbance (2026-08-09/10, overnight)

Per user direction ("investigate the grasp-close-timing precision issue... maybe
architecture isn't best... try different policy network or other IL policies"):

### Teacher-forced diagnostic — root cause found

Wrote a standalone script (loads the trained model + a held-out val set via the same
`StitchedSequencePointCloudDataset` class used in training, feeds real demo
observations through `model(cond=..., deterministic=True)`, compares predicted vs
ground-truth actions) to test: **does the model correctly predict gripper-closing
actions when SHOWN a real training-distribution grasp-phase state (open-loop), or
does it fail even then?**

Result on the disturbed-180 checkpoint (33.0% SR, best at the time): **the model
predicts gripper-closing direction/magnitude correctly ~90% of the time** (54/60
sampled active-motion windows), mean error 0.19 (normalized action units) vs 1.42 for
a naive "always predict no-motion" baseline. **This rules out an architecture/capacity
failure** — the model has genuinely learned the grasp-closing behavior.

**Conclusion: the ~33-67% failure rate is closed-loop compounding position error**,
not a model-competence problem. Small drift during autonomous rollout carries the
robot to a state slightly outside the training distribution right at the critical
closing moment — matches the video evidence exactly (hover near, don't close, then
close empty while retreating).

### Literature check (prioritizing recent work per standing instruction)

- **Diff-DAgger** (2025) — uncertainty-aware DAgger specifically for diffusion
  policies; directly targets this exact compounding-error mechanism. Candidate
  follow-up if the cheaper fix below doesn't close the gap.
- **Haptic-ACT** (Eljuri et al., June 2025) — force-feedback-based grasp failure
  detection **doubled in-domain success rate (80% vs 50%)**. Architecturally
  compelling, but giving the deployable **student** policy a contact-force
  observation breaks the sim/real parity the whole `RawObs`/`PerceptionPipeline`
  design is built around. **Flagged, not implemented** — a real architecture decision
  with project-wide implications.

### Fix: disturbance injection now targets the actual failure window

`execute_and_collect`'s `disturbance_phases` param (previously hardcoded to "lift"
only) now accepts ANY phase name(s), each getting an independent
`bernoulli(disturbance_prob)` draw. Since the diagnosed failure is a
position-precision problem during approach/grasp (not a post-grasp drop), a
`disturbance_phase=grasp` dataset directly manufactures the missing skill.

**Smoke-tested carefully**: an initial read at prob=0.5/max=2cm showed 50% collection
success, which looked like a real penalty — isolated via a same-seed no-disturbance
control (baseline ALSO only 41.7% at that seed) to a DR-sampled-batch-difficulty
confound, not the disturbance. Confirmed clean at actual settings
(prob=0.3/max=1-2cm): 41.7% vs 41.7%/40.5% control — no measurable penalty. **Recurring
lesson: always run a same-seed no-disturbance control before concluding a new
disturbance setting hurts collection.**

Full 250-episode `disturbance_phase=grasp` cherry collection launched. Committed
`aaab0a7`.

---

## Grasp-phase disturbance result — a surprising reversal (2026-08-10)

**Result: 13.0% (13/100)** — WORSE than lift-phase disturbance (33.0%, 24.0%) and even
the plain-250 baseline (17.0%), despite this checkpoint having the **lowest BC
validation loss of all four runs** (0.0085, vs 0.0091-0.0098 for the others).

### Updated four-way comparison (canonical, n=100, matched protocol)

| Checkpoint | Data | Best val loss | Success (n=100) |
|---|---|---|---|
| `nxyyn` — plain-250 | no disturbance | 0.0092 | 17.0% |
| `myspw` — disturbed-180 | disturbance during "lift" (180 ep, partial collection) | 0.0098 | **33.0%** (best so far) |
| `vrkjr` — disturbed-250-v2 | disturbance during "lift" (250 ep, full) | 0.0091 | 24.0% |
| `lnscz` — graspdist-250 | disturbance during "grasp" (250 ep, full) | **0.0085** (best loss) | 13.0% (worst SR) |

### Honest interpretation — this is a genuine reversal, not a clean confirmation

Two honest, non-exclusive readings:
1. **The specific mechanism may have backfired**: perturbing the EE position WHILE
   the gripper is actively closing forces the recorded recovery to be a position
   correction entangled with an in-progress grasp action — a noisier training signal
   right at the moment precision matters most.
2. **Single-seed run-to-run variance may be large enough to swamp the recipe
   effect.** Four single-seed training runs span 13.0%→33.0%, each compared as a
   clean apples-to-apples read on its dataset, but diffusion BC training is
   stochastic and no recipe had been repeated with a second seed before comparison.

**Action taken**: launched a same-data, different-seed replication of disturbed-180
(seed 123 vs the original 42) to directly measure how much of the observed spread is
training-run noise vs. real recipe signal. Run `ovgnm`, committed `790da2f`.

---

## CRITICAL FINDING: training-run variance dominates the recipe comparisons (2026-08-10)

**Seed-variance replication result: 15.0% (15/100)** — for a checkpoint trained on
the IDENTICAL disturbed-180 dataset as the 33.0%-scoring `myspw`, changing only the
training seed (42 → 123).

### This changes the interpretation of every result so far

| Run | Data | Seed | Success (n=100) |
|---|---|---|---|
| `nxyyn` — plain-250 | no disturbance, 250 ep | 42 | 17.0% |
| `myspw` — disturbed-180 | lift-phase disturbance, 180 ep | 42 | 33.0% |
| `ovgnm` — disturbed-180 (replicate) | **identical data to myspw** | **123** | **15.0%** |
| `vrkjr` — disturbed-250-v2 | lift-phase disturbance, 250 ep | 42 | 24.0% |
| `lnscz` — graspdist-250 | grasp-phase disturbance, 250 ep | 42 | 13.0% |

**Training-seed variance alone (33.0% → 15.0%, an 18-point swing on identical data) is
comparable to or larger than every "recipe" difference tested**. None of the recipe
comparisons made so far can be trusted individually. We cannot say with confidence
that "disturbance injection helps" — the two seeds of the SAME disturbed recipe
landed on both sides of the no-disturbance baseline.

**Action taken**: launched a 3rd seed (7) on the identical disturbed-180 dataset to
get a real n=3 estimate before drawing further conclusions. Committed `3e8aeab`.

**Methodological lesson for the whole session**: this same single-seed-per-condition
pattern was used for every comparison up to this point. Given the magnitude of seed
variance discovered, none of those earlier comparisons should be treated as
established conclusions — any real conclusion about which demo recipe/lever is
better requires multiple seeds per condition.

---

## PAUSED (2026-08-10 ~08:41) — everything stopped cleanly, resumed later

All training, sim server, and background monitor processes stopped on request
(process-group-safe SIGTERM). GPU fully clear. Nothing lost — checkpoints saved
every 100 epochs.

**3rd seed replicate (`hvzmv`, seed=7, disturbed-180 data)** was mid-flight at epoch
~410, not yet plateaued.

**Limitation found and later fixed**: DPPO's `TrainDiffusionAgent.run()` did not
support checkpoint-resume — `run()` unconditionally set `self.epoch = 1`. Real
resume support was built the next time this mattered (see the "Interruption + fix"
entry below).

### Where things stood — full results table

| Run | Data | Seed | Success (n=100) |
|---|---|---|---|
| Cross-category baseline (mixed) | 11 categories | — | 2.0% |
| Mushroom-solo | mushroom only | — | 4.5% |
| `nxyyn` — plain-250 | no disturbance | 42 | 17.0% |
| `myspw` — disturbed-180 | lift-phase disturbance | 42 | **33.0%** |
| `ovgnm` — disturbed-180 (replicate) | identical to myspw | 123 | 15.0% |
| `vrkjr` — disturbed-250-v2 | lift-phase disturbance, full 250 | 42 | 24.0% |
| `lnscz` — graspdist-250 | grasp-phase disturbance, full 250 | 42 | 13.0% |
| `hvzmv` — disturbed-180 (3rd replicate) | identical to myspw | 7 | PAUSED at epoch 400, not evaluated |

---

## RESUMED + strategic pivot: toy-task isolation experiment (2026-08-10 ~11:30)

Per user direction: still aiming for 60% SR, but time-pressured (want a working
cross-category policy within days) and asked to "think carefully" rather than
keep iterating on cherry demo recipes.

**Resumed the paused seed=7 replicate** (fresh restart at the time — no resume
support yet).

**New parallel direction, user-suggested**: before spending more effort on
cherry-specific demo recipes, test whether the *same pipeline* can reach 60-80%
under favorable conditions. Cherry is a uniquely hard target: ~20mm object with WIDE
domain randomization. If a bigger, easier object with narrow DR *also* can't clear
60%, that points to something more fundamental than cherry-specific difficulty.

**Toy task built**: `single_lift_apple_rigid_easy` — apple (~65mm, 3x cherry's size)
with a deliberately narrow DR (`gentle_manip/configs/dr/rigid_orientation_apple_easy.yaml`):
position half-range 0.04→0.02m, pitch/roll 20°→8°, scale/shape ranges tightened to
near-nominal. Committed `59fba37`.

---

## Toy-task collection done, training launched (2026-08-10 ~13:53)

**Apple-easy collection: 80/80 saved, 43.5% success (184 attempts)** — notably LOWER
than cherry's ~80% collection success, despite the object being 3x bigger and DR
much narrower. Not disqualifying — 80 clean episodes is within DP3's own reported
40-demo/85%-success benchmark range.

Converted (72 train / 8 val, episode lengths 209-217 steps). Training launched: run
`smcaf`, configs at `gentle_manip/dppo/cfg/single_lift_apple_rigid_easy_pcd/`
(committed `6f76db3`). Both cherry seed=7 (n=3 variance check) and apple-easy now
training in parallel.

---

## Interruption + fix: real checkpoint-resume support built (2026-08-10 ~21:00)

Both parallel trainings (cherry seed=7 at epoch ~494, apple-easy at epoch ~376) were
silently killed by what looks like a host-level restart (GPU went fully empty, both
processes vanished, both logs stop within seconds of each other with no error).
Several hours of unattended compute lost with no way to recover under the previous
"resume = restart from scratch" limitation.

**Fixed properly**: added real checkpoint-resume support to `TrainDiffusionAgent.run()`
(`third_party/dppo/agent/pretrain/train_diffusion_agent.py`) — pass
`+resume_from=<run>/checkpoint/state_<N>.pt` on the hydra CLI, loads model+EMA+epoch
and continues from `N+1`. Committed in the dppo submodule (`ed7bb3b`) and bumped in
the parent repo (`acd67ce`). Known limitation: optimizer/LR-scheduler state isn't
restored — acceptable for BC pretraining.

**Both runs resumed successfully**: cherry seed=7 at epoch 401 (lost ~94 epochs, not
hours), apple-easy at epoch 301 (lost ~76 epochs).

---

## Toy-task result: 65% — task difficulty confirmed as the bottleneck, not architecture

**Apple-easy plateaued at epoch 630** (best val loss 0.0318 at epoch 330). Evaluated
the nearest saved checkpoint by actual val loss, `state_500.pt` (val loss 0.0420).

**Canonical eval (n=100, num_envs=5, seed=0, scene_group_size=0 workaround,
per-episode video) result: 65.0% success (65/100)** — `ever_success_rate` 70%,
`hold_failure_gap` 0.01 (almost every episode reaching the target band also
completes the hold — no drop-after-lift failure mode, unlike cherry). Approx 95% CI
[55.7%, 74.3%] — squarely inside the 60-80% target band.

**The key diagnostic result for the whole cross-category effort.** Identical
architecture/recipe/harness as every cherry specialist (13-33% across n=3 seeds) —
only object size and DR range changed, closing a 40+ point gap. **Task difficulty,
not the architecture or demo-collection pipeline, was the bottleneck capping
cherry's specialists.**

**Environment note (fixed)**: the apple-easy eval initially crashed the sim server
with `ModuleNotFoundError: No module named 'gstaichi'` — traced to a stray
`PYTHONPATH` env var (`.../Genesis_fork:...`, an old sibling clone) that shadowed the
correctly-configured editable genesis install. Fixed by prefixing sim server
launches with `env -u PYTHONPATH`.

---

## Cherry seed=7 replicate: n=3 seed-variance estimate finalized

Cherry seed=7 plateaued at epoch 890 (best val loss 0.0100 at epoch 590). Evaluated
`state_600.pt` (val 0.0112). **Canonical eval result: 29.0% success (29/100)**,
`ever_success_rate` 36%.

**n=3 seed-variance estimate for the disturbed-180/lift-phase recipe, closed out:
33.0% (seed 42), 15.0% (seed 123), 29.0% (seed 7) → mean 25.7%, range 15-33 points.**
Confirms the earlier warning that 33.0% overstated the recipe's true mean. Cherry's
specialists cap out around 25-30% regardless of seed or demo-recipe tweaks — the
object/DR combination itself is the ceiling. **This thread is closed; no more cherry
recipe tuning is worth doing.**

---

## Second confirmation: avocado-easy

To rule out apple's 65% being a fluke of that specific object, launched the
identical narrow-DR recipe on **avocado** (~95mm long axis, elongated/asymmetric).
New configs: `rigid_orientation_avocado_easy.yaml`, `single_lift_avocado_rigid_easy.yaml`.

**Collection: 80/80 saved, only 18.1% collection success rate** (vs. apple's 43.5%)
— consistent with avocado's own DR config already flagging its min-extent (~61mm) as
close to the gripper's practical stroke limit (~70mm). Not necessarily predictive of
trained-policy quality (apple's own collection success was already much lower than
cherry's, yet apple's policy still hit 65%).

Converted (72/8 split). Training launched: run `wqlxl`.

**Result: plateaued at epoch 660 (best val 0.0315@360 — nearly identical to apple's
0.0318@330). Canonical eval on `state_400.pt` (val 0.0350): 52.0% success (52/100)**,
`ever_success_rate` 54%, `hold_failure_gap` 0.0. Approx 95% CI [42.2%, 61.8%].

**Nuanced finding: partial replication.** Avocado (52%) sits well above cherry's
25.7% mean but below apple's 65% and short of the target band.
`ever_success_rate`≈`success_rate` shows the policy isn't dropping objects after a
good grasp — it's failing to commit to a good grasp as often as apple's does.
**Revised takeaway: task difficulty is a SPECTRUM (size, DR range, grasp margin), not
binary** — cherry hardest, avocado (large but grasp-marginal) intermediate, apple
(large, comfortably graspable) easiest.

---

## Third confirmation + first real cross-category training run launched

Launched **kiwi-easy** (43x46x60mm, egg-shaped, no gripper-margin warning in its DR
config — a cleaner "comfortable grasp" test than avocado) as a third confirmation.
Early signal promising: 9/80 saved after only 4 batches.

**In parallel, built the first genuine cross-category (Phase 2) training run.**
`gentle_manip/scripts/merge_cross_category_demos.py` (pre-existing from an earlier
session, unused until now) merges per-category `data.pkl` files by symlinking into a
temp dir and calling `convert_demos.py` once — confirmed it picks the LATEST run per
category by mtime. **Bug found and worked around**: passing `--demos-root` as a
RELATIVE path breaks the script's symlinks — use the script's absolute default.

Merged apple-easy + avocado-easy → 160 episodes (144 train / 16 val) at
`$DPPO_DATA_DIR/single_lift_cross_category_easy_pcd`. Unconditioned baseline merge
(no category embedding), per the original research plan's Stage 6 recommendation
("do the free merge first"). New configs at
`gentle_manip/dppo/cfg/single_lift_cross_category_easy_pcd/`: training config + three
eval configs pointing the same checkpoint at different single-category sim tasks
(`_apple`/`_avocado` held-in, `_kiwi` zero-shot). Training launched: run `ioqec`.

---

## Kiwi-easy result: surprising tie with avocado

Kiwi-easy plateaued at epoch 620 (best val loss 0.0291@320 — the LOWEST val loss of
any object trained this session). Canonical eval on `state_300.pt` (val 0.0312):
**52.0% success (52/100)**, `ever_success_rate` 54%, `hold_failure_gap` 0.0.

**Striking, unexpected result: kiwi's numbers are numerically IDENTICAL to
avocado's** — despite kiwi having the best collection success rate (52.6%), lowest BC
val loss, and no gripper-margin warning. The "comfortable grasp margin" hypothesis
predicted kiwi should beat avocado, not tie it.

**Complicates the picture — likely explanation: object SIZE itself matters
independently of grasp comfort.** Kiwi (43x46x60mm) is meaningfully smaller than both
apple and avocado. Revised spectrum: cherry (25.7%) < {avocado 52%, kiwi 52%} <
apple (65%, uniquely favorable on all axes at once). Not fully resolved with n=3
objects.

---

## Cross-category generalist eval, part 1: apple held-in shows real interference

`ioqec` (apple+avocado merged, unconditioned) plateaued at epoch 760 (best val loss
0.0198, tied at epoch 460/700 — used `state_700.pt`).

**Held-in eval on APPLE: 39.0% success (39/100)**, `ever_success_rate` 40%. A
substantial drop from apple-SOLO's 65.0% — roughly a 26-point regression from
merging just ONE other category with no conditioning signal. Confirms the
"catastrophic interference from a naive merge" risk empirically.

## Cross-category generalist eval, part 2: avocado held-in also regresses

**Held-in eval on AVOCADO: 31.0% success (31/100)**, down from avocado-SOLO's 52.0%
(−21 points). **Both held-in categories regressed** — rules out an asymmetric
"one category dominates" story. Real evidence the category-conditioning branch
(Stage 5) is necessary, not a nice-to-have.

## Cross-category generalist eval, part 3: kiwi zero-shot COLLAPSES

**Zero-shot eval on KIWI (never in the training mix): 4.0% success (4/100)**,
`ever_success_rate` 5%. Essentially a collapse to near-random/failure level — the
SAME magnitude as this project's original 11-category mixed baseline (2.0%). This
REPLICATES that early negative result, now with a cleaner, better-controlled setup
(2 known-good categories at 52-65% solo, not 11 categories of uneven quality).

**Session summary at this point** — see the **Summary** section at the top of this
document for the consolidated Verdict 1 / Verdict 2 writeup that superseded the
in-line version originally written here.

---

## Fourth data point: pear-easy — complicates the grasp-margin story further

Pear-easy (44.7% collection SR) plateaued at epoch 780 (best val loss 0.0314@480).
Canonical eval on `state_500.pt` (val 0.0364): **44.0% success (44/100)**,
`ever_success_rate` 46%, `hold_failure_gap` 0.0. Approx 95% CI [34.3%, 53.7%].

**Full four-object task-difficulty spectrum** — see the table in the Summary section
at the top (Verdict 1).

**Pear breaks the clean "grasp-margin severity" story**: pear's flag (moderate) is
explicitly LESS severe than avocado's (severe), yet pear scored LOWER (44.0% vs
52.0%). Collection success rates don't explain it either (pear 44.7% is close to
apple's 43.5%, nothing like avocado's 18.1% outlier). **The fine ordering among the
four "easy" objects doesn't reduce to any single simple rule identified so far** —
likely reflects exact mesh geometry / per-shape CMA-ES search difficulty as much as
any size/margin heuristic. Empirical per-category eval remains necessary.

This does not change the two headline verdicts (Phase 1 validated; naive Phase 2
merge proven insufficient) — it only refines the "why some objects are easier than
others" sub-question. GPU/processes clean at end of this run.

---

## Conditioning experiments launched: Track A (registry) vs Track B (VLM) (2026-08-11)

User asked for a detailed plan to build the VLM-based category-conditioning branch.
Investigation found **most of the plumbing already existed from an earlier session,
unused**: `category_embedding.py` (registry-derived one-hot+features embedding),
`convert_demos.py --category-embed`, `pointcloud_dataset.py`'s
`StitchedSequencePointCloudCategoryDataset`, `PointNetDiffusionMLP(category_embed_dim=...)`,
and `genesis_venv.py`'s `category=` live-eval pinning — all built, all wired, all
committed-but-uncommitted-in-the-submodule (the `category=` kwarg pass-through in
`third_party/dppo/env/gym_utils/__init__.py` was sitting uncommitted in the working
tree since before this session; committed now alongside this work).

**The catch**: this existing embedding is 15/21 dims of literal one-hot — exactly
what the user's standing directive says NOT to build. Flagged this explicitly and
asked the user how to sequence the work. **User chose: calibration check first
(Track A, using the existing infra), then build the real VLM version (Track B)
regardless of Track A's result.**

### Track A — registry embedding calibration check

Regenerated the apple+avocado merge with `--category-embed` (existing, unused flag)
→ `single_lift_cross_category_easy_conditioned_pcd`, `category_embed_dim: 21`
confirmed. New configs mirroring the unconditioned baseline's structure, with
`category_embed_dim=21` in the network block and `env.specific.category: <name>` in
each eval config (apple/avocado held-in, kiwi zero-shot — kiwi IS in the registry's
fixed one-hot vocabulary but its slot was never activated during training, a fair
"oracle-category-name" zero-shot test for this embedding style). Training launched:
run `cfvpj`. Committed `dca1a72`.

### Track B — true VLM embedding, built fresh

The user's actual specified direction: a frozen vision-language model embedding,
low-dimensional, not a one-hot. New module `gentle_manip/dppo/vlm_embedding.py`:
frozen CLIP ViT-B/32 (`transformers`) embeds a canonical reference frame per
category, reduced via a FIXED (seeded, non-learned) random Gaussian projection to
24 dims — a standard Johnson-Lindenstrauss-style reduction, chosen over a learned
projection head to keep the embedding genuinely parameter-free outside the policy,
and over PCA (which would need many reference images per category to be well-posed;
only one is available per category here).

**Reference frames**: extracted frame 0 from tonight's own canonical eval render
clips for apple/avocado/kiwi (same camera/rendering as live deployment) — saved to
`gentle_manip/assets/category_reference_frames/<category>.png`. First attempt used
demo-collection videos, which turned out to not consistently exist across categories
(apple's collection had them, avocado/kiwi's didn't) — switched to eval render clips
uniformly, which always exist per the canonical eval harness's per-episode video
requirement.

**`transformers` API bug found and fixed**: `transformers==5.15.0`'s
`CLIPModel.get_image_features()` returns a `BaseModelOutputWithPooling` object, not a
bare tensor as in older versions — needed `.pooler_output[0]` instead of `[0]`
directly. Caught via a smoke test before wiring into the real pipeline.

**Infrastructure decision — avoid touching the actively-training `envs/dppo`**:
`envs/dppo` manages torch manually outside `pyproject.toml` (its own header warns `uv
sync` can silently pull CPU torch) and Track A was actively training in it. Installed
`transformers`+`pillow` in `envs/sim` only, and built
`gentle_manip/scripts/precompute_vlm_embeddings.py` to compute+cache embeddings there
once, to a disk cache (`gentle_manip/assets/category_reference_frames/vlm_embed_cache.npz`)
that `vlm_embedding.embed()` reads with plain numpy — verified `envs/dppo` can call
`embed()` successfully with zero `transformers` import needed. `envs/dppo` was never
synced/modified.

**Plumbing**: added an `--embed-source {registry,vlm}` flag to `convert_demos.py` and
`merge_cross_category_demos.py`, and a `category_embed_source` kwarg to
`genesis_venv.py::build_genesis_venv` (threaded through
`third_party/dppo/env/gym_utils/__init__.py`, committed in the submodule + parent
pointer bump) — both sources produce a `(D,)` vector through the identical
`embed(category) -> np.ndarray` interface, so everything downstream is agnostic to
which one produced it.

Regenerated the merge with `--embed-source vlm` → `single_lift_cross_category_easy_vlm_pcd`,
`category_embed_dim: 24` confirmed, `envs/dppo` never needed `transformers`. New
configs mirroring Track A's structure (`category_embed_dim=24`,
`category_embed_source: vlm` in each eval config's `env.specific`). Training
launched: run `atwfy`. Committed `6ff453f`.

**Both trainings running in parallel** (GPU has ample headroom — each uses ~1.7GB of
7.66GB). Will evaluate both the same way as the unconditioned baseline (held-in
apple, held-in avocado, zero-shot kiwi) once they plateau — 6 evals total, for a
direct 3-way comparison against the 39%/31%/4.0% baseline. See the **Summary**
section at the top of this document for the up-to-date status once results land.

## Bug found + fixed: eval configs missing category_embed in shape_meta

First Track A eval attempt (apple held-in) crashed immediately with
`KeyError: 'category_embed'` inside `PointNetDiffusionMLP.forward()`. Root cause:
`EvalHarnessAgent.__init__` builds `self.obs_keys = list(cfg.shape_meta.obs.keys())`
and the policy adapter only pulls THOSE keys from the live venv obs into `cond` --
the conditioned eval configs' `shape_meta.obs` block (copied from the unconditioned
template) only listed `state`/`point_cloud`, so `category_embed` was silently never
fetched even though `genesis_venv.py` was correctly producing it. Fixed by adding a
`category_embed: {shape: [21]}` (Track A) / `[24]` (Track B) entry to `shape_meta.obs`
in all 6 conditioned eval configs. Committed.

## Track A eval results

**Held-in APPLE: 42.0% success (42/100)**, `ever_success_rate` 45%,
`hold_failure_gap` 0.0. Small improvement over the unconditioned baseline's 39.0%
(+3 points), but still far below apple-solo's 65.0% -- the registry one-hot+features
embedding recovers only a fraction of the interference loss.

**Held-in AVOCADO: 32.0% success (32/100)**, `ever_success_rate` 34%. Essentially
flat vs. the unconditioned baseline's 31.0% (+1 point, within noise), still far
below avocado-solo's 52.0%.

**Zero-shot KIWI: 2.0% success (2/100)**, `ever_success_rate` 2%. Essentially
identical to the unconditioned baseline's 4.0% collapse -- the registry
one-hot+features embedding gives NO real zero-shot benefit. Kiwi's reserved-but-
never-trained one-hot slot plus its continuous material/size/shape features are
not enough to rescue generalization.

### Track A verdict

| | Unconditioned | Track A (registry) | Delta |
|---|---|---|---|
| Held-in apple | 39.0% | 42.0% | +3 |
| Held-in avocado | 31.0% | 32.0% | +1 |
| Zero-shot kiwi | 4.0% | 2.0% | -2 (noise) |

**The cheap calibration check confirms what the user's standing directive already
predicted: a one-hot-containing embedding does not meaningfully help, even with
now-good specialist data (ruling out the original data-quality confound
hypothesis).** Held-in numbers move by only a few points (within eval noise at
n=100), and zero-shot stays at collapse level. This is a genuine negative result,
not a data-quality artifact -- strengthens the case that Track B (the true VLM
embedding) is doing something structurally different, not just "the same idea with
better data." Track B evals next.

## Track B eval results (in progress)

**Held-in APPLE: 38.0% success (38/100)**, `ever_success_rate` 41%,
`hold_failure_gap` 0.0. Essentially flat vs. the unconditioned baseline's 39.0%
(within noise), slightly below Track A's 42.0%. Avocado and kiwi zero-shot evals
next.
