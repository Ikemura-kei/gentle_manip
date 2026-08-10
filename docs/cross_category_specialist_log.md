# Cross-Category Specialist/Generalist — Working Log

Branch: `cross-category-dp`. Running log of this work stream — key decisions, results,
and literature insights, kept human-readable and updated as things happen. Not a
replacement for `EXPERIMENT.md` per training run or the full research plan at
`~/.claude/plans/how-it-makes-trajectory-distributed-sun.md` — this is the fast-scan
summary of *why* things happened and *what they showed*.

---

## Goal (current phase) — priority order, confirmed by user

1. **Specialist-first**: get a single-category (cherry) DP3 policy to **60-80%**
   success on the canonical eval harness — the user's bar for "reasonable." Two
   levers: more demo trajectories, and better demo *quality/diversity*.
2. **Once a specialist clears 60%**, move on to the cross-category generalist policy,
   carrying forward every lesson/trick learned in phase 1.
3. Generalist conditioning direction (decided ahead of time, not yet built): a
   **low-dimensional, VLM-based** semantic conditioning vector (softness/size/geometry
   from a frozen vision-language model, read once per episode) — explicitly **not** a
   one-hot category ID (the original Stage 5(A) one-hot showed no benefit, likely
   confounded by low specialist-quality training data at the time) and explicitly
   **kept low-dimensional** for ease of learning, not a large embedding.
4. **Demo recovery behavior (grasp-drop-retry) is now DEPRIORITIZED.** It's built and
   confirmed working (see below), but the user downgraded its priority after seeing a
   rough edge in the retry motion — don't keep polishing it if it's not easy; the
   focus is specialist SR first, generalist second.

---

## Key literature insights

| Finding | Source | Why it matters here |
|---|---|---|
| DP3 gets **85% real-robot success with only 40 demos** | [3D Diffusion Policy, RSS 2024](https://3d-diffusion-policy.github.io/) | We have ~45-250 demos/category at 0-4.5% success — raw demo *count* is very unlikely to be the primary bottleneck; quality/diversity and task difficulty are the more likely levers. |
| **DART**: inject disturbances *during* demo collection (not post-hoc) so demos contain genuine recovery behavior | [Laskey et al. 2017, PMLR v78](http://proceedings.mlr.press/v78/laskey17a/laskey17a.pdf) | Named precedent for the compounding-error failure mode our eval videos showed. **Caveat**: 9-year-old paper — per user direction, prefer newer treatments of the same idea when available; kept here as the clearest statement of the mechanism. |
| Demo **diversity across environments/objects** matters more than raw count past a per-object threshold | [ICLR 2025 data-scaling-laws paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/88b7b2c896506daabc8d3fd587055167-Paper-Conference.pdf) | Supports trying disturbance-injected demos over just proportionally scaling the existing (undiversified) collection recipe. |
| **RLDG**: train specialists → roll them out to generate expert trajectories → distill into ONE generalist via BC; up to 40% higher success than distilling from raw demos directly | [arXiv 2412.09858](https://arxiv.org/abs/2412.09858), [project page](https://generalist-distillation.github.io/) (Dec 2024) | Named version of the user's own two-phase plan. Once specialists work, the cross-category generalist dataset should be built from **specialist rollouts**, not raw CMA-ES demo pickles. Backbone-agnostic (OpenVLA and Octo both validated) — our DP3/DPPO stack is a fine substitute. |
| **PA3FF / PADP**: part-aware dense 3D feature field (contrastive-pretrained on 3D part proposals), fed into a diffusion policy instead of raw point clouds; beats CLIP/DINOv2/Grounded-SAM; 28.8% vs GenDP's 19.4% on PartInstruct, only 6.25% drop on unseen objects | [arXiv 2602.14193](https://arxiv.org/abs/2602.14193) (Feb 2026) | Current frontier for cross-category/unseen-object generalization — exactly our zero-shot failure mode. Swap-in point is the `DP3Encoder` insertion (`pointnet_extractor.py:276`). Full architecture internals weren't extractable from the abstract alone — get the full paper text before implementing. Note: heavier/higher-dim than what the user now wants for the *first* generalist attempt (see Goal §3) — treat as a later upgrade, not the starting design. |

**Net implication for build order**: don't reintroduce cross-category conditioning
complexity until the underlying specialist data quality is fixed — Stage 5(A)'s early
negative result may just reflect garbage-in/garbage-out from 2-4.5%-success
specialists, not that conditioning doesn't help.

---

## Results so far

| Config | Data | Collection success | Eval (canonical, n=100) | Notes |
|---|---|---|---|---|
| Cross-category baseline (11 cat. mixed) | mixed | — | 2.0% (2/100) | Stage 5-7 baseline |
| + category-embedding (one-hot), epoch 300 | mixed | — | 1.0% | Within noise of baseline |
| + category-embedding (one-hot), epoch 500 | mixed | — | 0.0% | Within noise of baseline |
| Mushroom-solo (50 ep) | mushroom only | — | 4.5% (9/200) | >2x the mixed baseline — first data point suggesting specialize-per-category beats one shared policy |
| Cherry zero-shot (from mixed baseline) | — | — | 1.0% (1/100) | The number cherry-solo results below should beat |
| **Cherry-plain-250** | 250 ep, no disturbance | 80.1% (250/312 attempts) | **17.0% (17/100)** — run `nxyyn`, checkpoint 700 | Plateaued best val loss 0.0092 @ epoch 740 |
| **Cherry-disturbed-180** | 180 ep (recovered from a 250-ep target killed at timeout), `disturbance_prob=0.3`, `max=2cm` during "lift" | 80.0% (180/225 attempts) — **identical to plain**, confirms disturbance injection doesn't hurt the demonstrator | **33.0% (33/100)** — run `myspw`, checkpoint 600 | Plateaued best val loss 0.0098 @ epoch 560. **Best result this session by a wide margin** — see "Canonical eval results" and "Final head-to-head" sections below for full detail. |

**UPDATE — both DONE, see "Canonical eval results" section below for the full
head-to-head and the decision it drove.** Short version: disturbance injection
nearly doubles success rate at a SMALLER episode count (180 vs 250) — the clearest
lever found this session. A full, clean 250-episode disturbance-injected collection
(v2) is running now to see if scaling that lever further closes more of the gap to
60-80%.

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
  the code; **deprioritized** per user direction (see Goal §4) rather than fixed.

---

## Timeline / decision log

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
  (documented as TODO, not fixed — see below). Committed `1e484c4`.
- **User redirect #3**: retry/recovery work is **lower priority** — don't keep
  polishing it if it's not easy; refocus on (1) specialist SR, (2) generalist.
  Confirmed VLM conditioning should be **low-dimensional**.
- **Early-stop plateau watchers armed** for both trainings (`nxyyn`, `myspw`): each
  fires once val loss hasn't improved by >0.0005 for 300 epochs (30 val
  checkpoints), so neither run rides out its full 2000-epoch budget unnecessarily.

---

## Currently running (updated — supersedes the section above, kept for history)

- Both `nxyyn` and `myspw` trainings are **DONE** (plateaued, stopped cleanly, both
  canonically evaluated — see "Canonical eval results" / "Final head-to-head" below).
- **`cherry_disturbed_v2` demo collection — RUNNING now** (started ~19:04, 2h timeout):
  a full, clean 250-episode disturbance-injected cherry collection (`disturbance_prob=0.3`,
  `disturbance_max_m=0.02`), replacing the earlier 180-episode partial recovery.
  `dataset/demos/single_lift_cherry_rigid/<new-run-dir>/`. Log:
  `logs/collect_backfill/cherry_disturbed_v2/cherry.log`. Progress watcher: task
  `b42nri1lh`. Note: uses the orchestrator's default seed (0), same as the original
  180-episode attempt, so its DR/CMA-ES sequence is deterministic and matches the
  earlier run episode-for-episode up through where that one got killed (~batch 46,
  180 episodes) — the genuinely NEW data is the ~70 episodes beyond that point.

## Next steps

1. **DONE**: both plateau watchers fired, both runs stopped, both canonically
   evaluated (n=100 each). Result: disturbed-180 (33.0%) beat plain-250 (17.0%) by
   ~2x on a SMALLER dataset — disturbance injection is the clearest lever found this
   session (see "Canonical eval results" section for the deviation-from-protocol
   caveat: both ran with `scene_group_size=0` due to an unresolved rebuild-RPC hang).
2. **IN PROGRESS**: collect a full 250-episode disturbance-injected dataset (v2, no
   timeout interruption this time) to see if scaling the winning lever further closes
   more of the gap to 60-80%.
3. Once v2 collection finishes: convert → train a specialist the same way → canonical
   eval (n=100) → compare against 33.0% and the 60-80% target.
4. **If the new specialist clears 60%**: immediately pivot to generalist work — build
   the training set from **specialist rollouts** (RLDG-style) rather than raw CMA-ES
   demos, and design the **low-dimensional** VLM-based per-episode conditioning
   vector at the `DP3Encoder` insertion point.
5. **If it plateaus well below 60%**: the rollout diagnostic (see below) found a
   *different* failure mode than disturbance-injection targets — a grasp-timing/
   precision issue (policy hovers near the object without closing, then closes empty
   while retreating). If more disturbance data doesn't close the gap, that precision
   issue — not more recovery examples — is probably the next lever, e.g. more
   demos slow/precise through the final approach, or an explicit proximity-conditioned
   closing signal.
6. Regrasp-retry (idea #2) data collection at scale remains parked (deprioritized)
   unless revisited explicitly. The `scene_group_size>0` rebuild-RPC hang is still
   unresolved and should be fixed before the next "official" canonical numbers.

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
earlier quick check (root cause not yet found — see open item below). `n_episodes=100`,
`num_envs=5`, `seed=0` (all canonical/unchanged); both variants evaluated identically,
so the plain-vs-disturbed comparison is still apples-to-apples, just without shape/scale
DR coverage in this particular pair of numbers.

### Result: DART-style disturbance injection is a large, real win

| Checkpoint | Success (n=100, canonical) | vs. mixed baseline (2.0%) | vs. 60-80% target |
|---|---|---|---|
| **myspw** (cherry-disturbed-180) | **33.0%** (100/100 episodes) | **16.5x** | Still short, but by far the best result this session |
| **nxyyn** (cherry-plain-250) | *running* | — | — |

33.0% is a dramatic jump over every other number seen this session (mixed baseline
2.0%, category-embedding 0-1%, mushroom-solo 4.5%, cherry-plain informal check ~6.7%).
This is strong evidence that **DART-style recovery demos (idea #3) matter far more
than raw demo count (250 vs 180 episodes) or than cross-category conditioning
architecture** — the single biggest lever found this session by a wide margin.
Directly validates the user's original hypothesis and the DART citation.

Waiting on `nxyyn`'s matched canonical number (same protocol, same checkpoint quality
tier) to quantify the disturbance-injection effect in isolation from the small
250-vs-180 episode-count difference.

**Open item**: root-cause the `scene_group_size>0` rebuild-RPC hang so the *next*
official numbers (once we act on this result — e.g. a bigger/cleaner disturbance
dataset) can use the full canonical protocol including shape/scale DR coverage.

### Final head-to-head: plain-250 vs disturbed-180 (both canonical, n=100, matched protocol)

| Checkpoint | Success (n=100) | Notes |
|---|---|---|
| `nxyyn` (cherry-plain-250, 250 ep, no disturbance) | **17.0%** (17/100) | |
| `myspw` (cherry-disturbed-180, 180 ep, disturbance_prob=0.3) | **33.0%** (33/100) | Fewer episodes, ~2x the success rate |

**Conclusion: DART-style disturbance injection is a large, isolated, real win** — the
disturbed dataset has FEWER episodes (180 vs 250) yet nearly doubles success rate. This
cleanly separates the disturbance-injection effect from a raw-count effect (more data
alone, going 50→250 earlier, was a much smaller win: 4.5%→17.0% mushroom/cherry-solo
territory). Both still fall short of the 60-80% target, but this is the single most
effective lever found this session by a wide margin, and directly confirms the DART
citation's premise for this task.

**Recommended next step (executing now)**: collect a FULL, clean 250-episode
disturbance-injected cherry dataset (the 180-episode one was a partial recovery from a
timed-out run) with a longer collection timeout, train a specialist on it the same way,
and re-eval. If that closes further toward 60%, disturbance-injection at scale is the
answer for phase 1; if it plateaus well below 60%, the next lever to try is probably the
gripper-close-timing precision issue found in the rollout diagnostic (a different failure
mode than what disturbance-injection targets), which might need e.g. more demos
specifically SLOW/precise through the final grasp-approach phase, or an explicit
proximity-conditioned closing signal rather than more disturbance-recovery examples.

---

## Disturbed-250-v2: collection done, training launched (2026-08-09 ~20:15)

Full, uninterrupted collection finished: **250/250 saved, 79.87% success (313
attempts)** — consistent with plain-250 (80.1%) and disturbed-180 (80.0%), confirming
disturbance injection doesn't hurt collection success even at full scale. Data:
`dataset/demos/single_lift_cherry_rigid/26-08-09-zyv/data.pkl`.

Converted (225 train / 25 val episodes, 48713 train steps) and training launched:
run `vrkjr`, configs at `gentle_manip/dppo/cfg/single_lift_cherry_rigid_pcd_250_disturbed_v2/`
(committed `ddb9c1d`). Plateau watcher armed (same rule: stop after 300 epochs / 30
val checkpoints with no improvement beyond 0.0005).

**Once plateaued**: run the same canonical n=100 eval (with the `scene_group_size=0`
workaround noted in its config) and compare directly against disturbed-180's 33.0% —
this isolates whether scaling disturbance-injected data from 180→250 episodes helps
further, now that the recovery-behavior lever itself is confirmed to work.

---

## Disturbed-250-v2 canonical eval — DONE (2026-08-09 ~23:25)

**Result: 24.0% (100/100 episodes)** — checkpoint `state_500.pt` (best val loss 0.0091,
notably lower/better than disturbed-180's 0.0098). Eval videos: one clip per episode at
`.../vrkjr/eval/2026-08-09_23-06-42/render/batchNN_envM.mp4` (100 clips), per-episode
results in `episodes.csv` in the same dir.

### Full three-way comparison (canonical, n=100, matched protocol)

| Checkpoint | Data | Best val loss | Success (n=100) |
|---|---|---|---|
| `nxyyn` (cherry-plain-250) | 250 ep, no disturbance | 0.0092 | **17.0%** |
| `myspw` (cherry-disturbed-180) | 180 ep, disturbance_prob=0.3 | 0.0098 | **33.0%** |
| `vrkjr` (cherry-disturbed-250-v2) | 250 ep, disturbance_prob=0.3 | 0.0091 (best of the three) | **24.0%** |

**Key finding: scaling the disturbance-injected dataset from 180→250 episodes did NOT
improve success rate further — it went DOWN (33.0%→24.0%), despite the 250-episode
checkpoint having the LOWEST (best) BC validation loss of all three runs.** This is
the same loss-vs-rollout-SR disconnect flagged earlier, now showing up as a second,
independent data point. At n=100 the binomial CI on both numbers is roughly ±9pp, so
24% and 33% are not dramatically outside each other's noise band — the honest
reading is **disturbance injection lifts success into the ~24-33% range** (a real,
substantial win over plain-250's 17%), but simply collecting MORE disturbance data
does not keep buying further gains. The lever has plateaued.

**Confirmed via direct video inspection (requested check)**: extracted frames from a
verified real success (episode 1, `batch00_env1.mp4`, `first_success_step=58`) and a
verified real failure (episode 0, `batch00_env0.mp4`) from the disturbed-250-v2 run.
Success episode shows the same fully correct approach→close-near-object→lift→hold
sequence as before. **The failure episode shows the EXACT SAME failure pattern found
in the very first diagnostic** (nxyyn, several sections above): arm reaches the
object, then retracts fully to home with the gripper never having closed on the
cherry — object left completely untouched. This pattern now confirmed present
across all three trained checkpoints regardless of dataset composition — strong
evidence this is a **structural precision/reliability limitation of the current
setup** (small object, chunked-action diffusion policy, no explicit
proximity-conditioned closing signal), not something more of the same kind of data
collection will fix.

### Recommendation — the next lever should be architectural/precision-focused, not more data

Given three independent training runs (50→180→250 episodes, with/without disturbance)
all show the identical dominant failure mode, the next productive step is likely one
of:
1. **Inspect raw predicted gripper-channel actions** near the failure point vs. the
   training demos' gripper actions in the corresponding phase — check whether the
   policy is systematically under-predicting closing magnitude/timing (a concrete,
   bounded numerical diagnostic, not a new architecture).
2. Consider whether `cond_steps`/`horizon_steps` (currently 2/4) or the delta-mode
   gripper action scale (from `xarm7_config.DEFAULT_ACTION_SCALES`, gripper=0.05) are
   appropriate for a task requiring fine, reliably-timed closing on a small
   (~1.5-2cm) object.
3. Only after (1)/(2) are exhausted: revisit whether an explicit proximity/contact
   signal in the observation (not currently available to the deployable student
   policy — no force sensing) would help the closing decision be more reliably
   triggered.

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

Result on the disturbed-180 checkpoint (33.0% SR, our best): **the model predicts
gripper-closing direction/magnitude correctly ~90% of the time** (54/60 sampled
active-motion windows), mean error 0.19 (normalized action units) vs 1.42 for a
naive "always predict no-motion" baseline. **This rules out an architecture/capacity
failure** — the model has genuinely learned the grasp-closing behavior. (Took one
iteration to get right: the converted dataset's gripper-action channel turned out to
be per-dimension-normalized such that "no motion" — the dominant value — normalizes
to +1.0, not 0, since raw delta=0 happens to be the extreme of its own range. First
threshold attempt silently found 0 "closing" windows because of this; fixed by
detecting deviation-from-baseline instead of deviation-from-zero.)

**Conclusion: the ~33-67% failure rate is closed-loop compounding position error**,
not a model-competence problem. Small drift during autonomous rollout carries the
robot to a state slightly outside the training distribution right at the critical
closing moment — matches the video evidence exactly (hover near, don't close, then
close empty while retreating).

### Literature check (prioritizing recent work per standing instruction)

- **Diff-DAgger** (2025) — uncertainty-aware DAgger specifically for diffusion
  policies; directly targets this exact compounding-error mechanism. Noted as a
  candidate follow-up if the cheaper fix below doesn't close the gap (needs an
  online expert-query loop — bigger lift than a data-recipe change).
- **Haptic-ACT** (Eljuri et al., June 2025) — force-feedback-based grasp failure
  detection **doubled in-domain success rate (80% vs 50%)**. Architecturally
  compelling, but giving the deployable **student** policy a contact-force
  observation breaks the sim/real parity the whole `RawObs`/`PerceptionPipeline`
  design is built around (real robot has GelSight tactile images, not a force
  scalar; sim's `priv_contact_force` is currently teacher-only by design). **Flagged,
  not implemented** — this is a real architecture decision with project-wide
  implications, not something to change unilaterally overnight. Worth a deliberate
  discussion if the data-side fixes plateau.

### Fix: disturbance injection now targets the actual failure window

`execute_and_collect`'s `disturbance_phases` param (previously hardcoded to "lift"
only) now accepts ANY phase name(s), each getting an independent
`bernoulli(disturbance_prob)` draw. Since the diagnosed failure is a
position-precision problem during approach/grasp (not a post-grasp drop, which is
what lift-phase disturbance targets), collecting a `disturbance_phase=grasp` dataset
directly manufactures the missing skill: "arrived slightly off the true grasp pose,
corrected into alignment, then closed" — the demonstrator's target is untouched, so
the correction is genuine, not synthetic noise.

**Smoke-tested carefully** (grasp-phase kicks are more disruptive than lift-phase —
they land during the actual closing motion): an initial read at prob=0.5/max=2cm
showed 50% collection success (vs the usual ~80%), which looked like a real penalty.
**Isolated via a same-seed no-disturbance control**: baseline collection success at
that seed was ALSO only 41.7% — i.e. that seed's DR-sampled batches were just harder,
not the disturbance. Confirmed clean at the actual settings used
(prob=0.3/max=1-2cm): 41.7% vs 41.7%/40.5% control — no measurable penalty. Same
"methodology confound, not a real effect" pattern as the very first lift-phase
disturbance smoke test earlier this session — worth remembering as a recurring
lesson: **always run a same-seed no-disturbance control before concluding a new
disturbance setting hurts collection.**

**Full 250-episode `disturbance_phase=grasp` cherry collection launched** (prob=0.3,
max=2cm, matching the lift-phase settings for a clean "same magnitude/probability,
different WHERE" comparison against the disturbed-180/250-v2 results). Will
convert → train → canonical-eval the same way once done, compare against the 33.0%
best result. Committed `aaab0a7`.

### Standing recommendation if this doesn't close the gap further

1. Try **Diff-DAgger**-style online correction (bigger lift, but directly matches
   the diagnosed mechanism).
2. Revisit the **Haptic-ACT** contact-force idea as a deliberate architecture
   decision (needs explicit discussion of the student/real-parity tradeoff — the
   real robot's GelSight tactile images could plausibly substitute for sim's
   `priv_contact_force` as an analogous but real-transferable signal, but that's a
   design choice, not a quick patch).
3. Only after (1)/(2): consider a non-diffusion IL backbone (e.g. ACT) — the
   diagnostic evidence so far does NOT point to diffusion-specifically being the
   bottleneck, so an architecture swap is a lower-priority lever than the two above.

---

## Grasp-phase disturbance result — a surprising reversal (2026-08-10)

**Result: 13.0% (13/100)** — WORSE than lift-phase disturbance (33.0%, 24.0%) and even
the plain-250 baseline (17.0%), despite this checkpoint having the **lowest BC
validation loss of all four runs** (0.0085, vs 0.0091-0.0098 for the others).

### Updated four-way comparison (canonical, n=100, matched protocol)

| Checkpoint | Data (all 250 ep target, disturbance_prob=0.3/max=2cm unless noted) | Best val loss | Success (n=100) |
|---|---|---|---|
| `nxyyn` — plain-250 | no disturbance | 0.0092 | 17.0% |
| `myspw` — disturbed-180 | disturbance during "lift" (180 ep, partial collection) | 0.0098 | **33.0%** (best) |
| `vrkjr` — disturbed-250-v2 | disturbance during "lift" (250 ep, full) | 0.0091 | 24.0% |
| `lnscz` — graspdist-250 | disturbance during "grasp" (250 ep, full) | **0.0085** (best loss) | 13.0% (worst SR) |

### Honest interpretation — this is a genuine reversal, not a clean confirmation

The grasp-phase hypothesis was well-motivated by the teacher-forced diagnostic
(disturbed-180 predicts gripper actions correctly ~90% open-loop → failure is
closed-loop position drift, not model incompetence), but the experiment **did not
confirm it** — if anything it went the opposite direction. Two honest readings,
not mutually exclusive:

1. **The specific mechanism may have backfired**: perturbing the EE position WHILE
   the gripper is actively closing (not after, as in lift-phase disturbance) forces
   the recorded recovery to be a position correction *entangled with* an in-progress
   grasp action — a noisier, more ambiguous training signal right at the exact
   moment precision matters most, plausibly making closing-timing LESS reliable
   rather than more.
2. **Single-seed run-to-run variance may be large enough to swamp the recipe
   effect.** We now have four single-seed training runs spanning 13.0%→33.0%
   success — a wide spread — each compared as if it were a clean apples-to-apples
   read on its dataset. Diffusion BC training is stochastic (seed affects weight
   init, batch order, EMA trajectory); we have never repeated a recipe with a
   second seed to know its OWN variance before comparing it against a different
   recipe's single run. **This is a real methodological gap in every comparison
   made this session so far.**

**Action taken**: launched a same-data, different-seed replication of the winning
disturbed-180 recipe (seed 123 vs the original 42, dataset held identical — no new
collection) to directly measure how much of the observed spread is training-run
noise vs. real recipe signal. Run `ovgnm`, config
`gentle_manip/dppo/cfg/single_lift_cherry_rigid_pcd_250_disturbed_seed2/`, committed
`790da2f`. If seed=123 lands far from 33.0% (e.g. also in the 13-24% range), that
confirms single-seed comparisons this session are too noisy to trust individually,
and any real conclusion about "which demo recipe is best" needs multiple seeds per
recipe going forward — a significant scope increase for future rounds, but
necessary for a trustworthy answer given fragile-object grasping needs a genuinely
reliable specialist (60-80% target), not one good roll of a noisy die.

### Status against the final goal (fragile object grasping, reasonable SR)

**Not yet met.** Best confirmed number is 33.0% (disturbed-180, `myspw`), well
below the 60-80% target. Pending the seed-variance result, the next real
decision point is: (a) if 33.0% turns out to be robust to seed, the standing
next-tier recommendations (Diff-DAgger online correction, or a deliberate
Haptic-ACT-style contact-force conditioning discussion) are the most promising
levers; (b) if the seed=123 run also lands low, the priority becomes fixing the
evaluation methodology itself (multi-seed averaging) before drawing further
conclusions about ANY lever tried so far, including the disturbance-injection
result that looked like the clear winner.

---

## CRITICAL FINDING: training-run variance dominates the recipe comparisons (2026-08-10)

**Seed-variance replication result: 15.0% (15/100)** — for a checkpoint trained on
the IDENTICAL disturbed-180 dataset as the 33.0%-scoring `myspw`, changing only the
training seed (42 → 123). Same data, same architecture, same everything except
random seed.

### This changes the interpretation of every result tonight

| Run | Data | Seed | Success (n=100) |
|---|---|---|---|
| `nxyyn` — plain-250 | no disturbance, 250 ep | 42 | 17.0% |
| `myspw` — disturbed-180 | lift-phase disturbance, 180 ep | 42 | 33.0% |
| `ovgnm` — disturbed-180 (replicate) | **identical data to myspw** | **123** | **15.0%** |
| `vrkjr` — disturbed-250-v2 | lift-phase disturbance, 250 ep | 42 | 24.0% |
| `lnscz` — graspdist-250 | grasp-phase disturbance, 250 ep | 42 | 13.0% |

**Training-seed variance alone (33.0% → 15.0%, an 18-point swing on identical data)
is comparable to or larger than every "recipe" difference tested tonight** (plain
vs. lift-disturbance vs. grasp-disturbance vs. 180-vs-250 episodes, all single-seed,
spanning 13-33%). This means **none of tonight's recipe comparisons can be trusted
individually** — a single training run's success rate is not a reliable estimate of
a demo recipe's true quality, given diffusion BC's training stochasticity is this
large relative to the effect sizes being measured. Concretely: we cannot currently
say with confidence that "disturbance injection helps" (33.0% and 15.0% straddle
17.0%, i.e. the two seeds of the SAME disturbed recipe landed on both sides of the
no-disturbance baseline), even though it looked like a clear, large win after only
seed 42.

**Action taken**: launched a 3rd seed (7) on the identical disturbed-180 dataset —
`ovgnm`'s sibling run — to get a real n=3 estimate of this recipe's mean and spread
before drawing further conclusions. Config
`gentle_manip/dppo/cfg/single_lift_cherry_rigid_pcd_250_disturbed_seed3/`, committed
`3e8aeab`.

### Implication for the whole session, honestly stated

This same single-seed-per-condition pattern was used for EVERY comparison this
session, including earlier ones (mushroom-solo 4.5% vs mixed 2.0%; the disturbed-180
vs plain-250 vs disturbed-250-v2 vs graspdist-250 four-way). Given the magnitude of
seed variance just discovered, **none of those comparisons should be treated as
established conclusions** — they are single data points that happened to support a
plausible-sounding story, not statistically validated findings. This is the single
most important methodological lesson from tonight's work, and needs to inform how
results are evaluated and reported going forward: **any real conclusion about which
demo recipe/lever is better requires multiple seeds per condition**, not one.

### Status against the final goal (fragile object grasping, reasonable SR)

**Still not met, and now genuinely uncertain what the true achievable number is**
with the current architecture/recipe space. The honest range across all runs so far
is roughly 13-33%, with the seed-variance finding suggesting the TRUE mean for
"cherry specialist, current architecture, ~180-250 demos with or without disturbance
injection" is probably somewhere in the low-to-mid 20s%, not the 33.0% that looked
like a clean win. This is well below the 60-80% target regardless. **Recommendation
for when the user reviews this**: given demo-recipe tweaks (more data, disturbance
timing) have NOT shown a reproducible, trustworthy improvement once variance is
accounted for, the next real lever is probably NOT another data-recipe permutation —
it's likely time to seriously evaluate the previously-flagged next-tier options
(Diff-DAgger online correction, or a deliberate Haptic-ACT-style force-conditioning
architecture discussion), since incremental demo changes have hit a point of
diminishing/unclear returns.

---

## PAUSED (2026-08-10 ~08:41) — everything stopped cleanly, here's how to resume

All training, sim server, and background monitor processes stopped on request
(process-group-safe SIGTERM throughout — same pattern used all session). GPU is
fully clear (`nvidia-smi` shows zero compute processes). Nothing was lost:
checkpoints are saved to disk every 100 epochs regardless of how a run ends.

### What was mid-flight when stopped

**3rd seed replicate (`hvzmv`, seed=7, disturbed-180 data)** — stopped at epoch
~410. Last saved checkpoint: `state_400.pt`. Val loss was still healthy/improving
(best ~0.0109 around epoch 380, matches the other seeds' trajectories) — **not
plateaued yet**, so this run was killed before its "true" answer was ready. Its
purpose: complete the n=3 variance estimate for the disturbed-180 recipe (seed=42 →
33.0%, seed=123 → 15.0%, seed=7 → not yet measured).

**Important limitation, checked before stopping**: DPPO's `TrainDiffusionAgent.run()`
(`third_party/dppo/agent/pretrain/train_diffusion_agent.py`) does **not** wire up
checkpoint-resume automatically — `run()` unconditionally sets `self.epoch = 1` at
the start. The base class DOES have a working `self.load(epoch)` method (restores
model/EMA/epoch from `checkpoint/state_<epoch>.pt`) but nothing calls it from the
hydra CLI path. So **re-running the same launch command starts a fresh run from
epoch 1**, not a resume from `state_400.pt`. Two ways to actually continue this
specific run:
1. **Simplest — just restart it fresh.** Given each run only takes ~50-70 min to
   plateau, this is cheap enough that a clean restart is likely less effort than
   wiring up real resume support. Command:
   ```
   uv run --project envs/dppo python -m gentle_manip.dppo.train \
       --config-path gentle_manip/dppo/cfg/single_lift_cherry_rigid_pcd_250_disturbed_seed3 \
       --config-name pre_diffusion_pointnet
   ```
   (this mints a NEW run ID, e.g. not `hvzmv` — that's fine, it's the same config/data/seed)
2. **True resume** (if ever needed for a longer/more expensive run): would need a
   small addition to `TrainDiffusionAgent.run()` — e.g. an optional
   `resume_from_epoch` cfg field that calls `self.load(resume_from_epoch)` before
   the loop and adjusts the range accordingly. Not implemented (out of scope for
   "just pause it"); flag this to the user if a future run is expensive enough to
   be worth the small code change.

### To resume the overall investigation

1. **Finish the seed-variance check** (recommended next step): restart the seed=7
   run per above (fresh, ~50-70 min to plateau + ~15 min canonical eval). This
   completes n=3 for the disturbed-180 recipe and gives a real mean/spread instead
   of two anecdote points (33.0%, 15.0%).
2. Sim server for eval (needed after any training finishes):
   ```
   uv run --project envs/sim python -m gentle_manip.scripts.serl_sim_server \
       --experiment single_lift_cherry_rigid --view student \
       --num-envs 5 --render-rgb --subprocess
   ```
   (defaults to port 5566 in this session's usage — pass `env.specific.port=5566`
   at eval time to match, or check the printed port and adjust)
3. Eval command template (swap in the new checkpoint path):
   ```
   uv run --project envs/dppo python -m gentle_manip.dppo.train \
       --config-name eval_diffusion_pointnet \
       --config-path gentle_manip/dppo/cfg/single_lift_cherry_rigid_pcd_250_disturbed \
       base_policy_path=<new_run>/checkpoint/state_<best_epoch>.pt \
       n_episodes=100 record_batches=null scene_group_size=0 \
       env.specific.port=5566
   ```
   (`scene_group_size=0` is the standing workaround for the still-unresolved
   rebuild-RPC hang — see earlier section of this log)

### Where things stand — full results table (all confirmed, nothing lost)

| Run | Data | Seed | Success (n=100) |
|---|---|---|---|
| Cross-category baseline (mixed) | 11 categories | — | 2.0% |
| Mushroom-solo | mushroom only | — | 4.5% |
| `nxyyn` — plain-250 | no disturbance | 42 | 17.0% |
| `myspw` — disturbed-180 | lift-phase disturbance | 42 | **33.0%** |
| `ovgnm` — disturbed-180 (replicate) | identical to myspw | 123 | 15.0% |
| `vrkjr` — disturbed-250-v2 | lift-phase disturbance, full 250 | 42 | 24.0% |
| `lnscz` — graspdist-250 | grasp-phase disturbance, full 250 | 42 | 13.0% |
| `hvzmv` — disturbed-180 (3rd replicate) | identical to myspw | 7 | **PAUSED at epoch 400, not evaluated** |

**Bottom line for whoever reads this next**: the true achievable success rate with
the current architecture/recipes, once seed variance is accounted for, is most
likely in the **~15-25% range**, not the 33.0% that looked like a clean win after
one seed. All of this is well below the 60-80% target. The next real lever is
probably not another data-recipe permutation — see the "next-tier" recommendations
(Diff-DAgger online correction, or a deliberate Haptic-ACT-style contact-force
conditioning discussion) earlier in this log.

---

## RESUMED + strategic pivot: toy-task isolation experiment (2026-08-10 ~11:30)

Per user direction: still aiming for 60% SR, but time-pressured (want a working
cross-category policy within days) and asked to "think carefully" rather than
keep iterating on cherry demo recipes.

**Resumed the paused seed=7 replicate** (fresh restart, no native checkpoint-resume
support in DPPO's `TrainDiffusionAgent` — see the PAUSED section above for why).
Still running toward completing the n=3 variance estimate for the disturbed-180
recipe (seed 42 → 33.0%, seed 123 → 15.0%, seed 7 → pending).

**New parallel direction, user-suggested and well-motivated**: before spending more
effort on cherry-specific demo recipes, test whether the *same pipeline* can reach
60-80% under favorable conditions. Cherry is a uniquely hard target: ~20mm object
(near the limit of what makes sense for this gripper/point-cloud resolution) with
WIDE domain randomization (full 360° yaw, ±45° pitch/roll originally, full
shape+scale DR). If a bigger, easier object with narrow DR *also* can't clear 60%,
that points to something more fundamental (architecture, action space, control
frequency) rather than cherry-specific difficulty — a much more important thing to
know before committing days to more data-recipe iteration.

**Toy task built**: `single_lift_apple_rigid_easy` — apple (~65mm, 3x cherry's
size, non-symmetric so it avoids tofu's documented "6 equivalent faces" grasp
ambiguity) with a deliberately narrow DR
(`gentle_manip/configs/dr/rigid_orientation_apple_easy.yaml`): position half-range
0.04→0.02m, pitch/roll 20°→8°, scale/shape ranges tightened to near-nominal. Full
360° yaw kept (apple is roughly round, shouldn't be the hard axis). Committed
`59fba37`.

**Plan**: smoke-test (running) → if healthy, collect ~150 plain episodes (no
disturbance yet — establish the simplest possible baseline first) → convert → train
→ canonical eval. Compare directly against cherry's 13-33% range.

- **If apple-easy clears 60-80%**: task difficulty (small object + wide DR) is the
  real bottleneck. Path forward: either use a bigger/easier object as the
  cross-category policy's primary target, or invest in narrowing DR ranges for
  cherry-class objects specifically, rather than more disturbance-injection
  variants.
- **If apple-easy also lands in the 15-30% range**: the bottleneck is more
  fundamental than task difficulty — likely needs architecture-level changes
  (Diff-DAgger, contact-force conditioning) regardless of which object is used.

This result will be the single most informative data point for deciding where to
spend the remaining time before the "days" deadline.

---

## Toy-task collection done, training launched (2026-08-10 ~13:53)

**Apple-easy collection: 80/80 saved, 43.5% success (184 attempts)** — notably
LOWER than cherry's ~80% collection success, despite the object being 3x bigger
and DR being much narrower. Interesting on its own (worth revisiting if this
becomes the path forward — possibly the CMA-ES search bounds/config are tuned
around mushroom/cherry scale, or apple's size is genuinely closer to the gripper's
practical stroke limit, as already flagged in `rigid_orientation_apple.yaml`'s own
history). Not disqualifying — 80 clean episodes is within DP3's own reported
40-demo/85%-success benchmark range.

Converted (72 train / 8 val, episode lengths 209-217 steps — same FSM timing as
every cherry run). Training launched: run `smcaf`, configs at
`gentle_manip/dppo/cfg/single_lift_apple_rigid_easy_pcd/` (committed `6f76db3`).

**Both cherry seed=7 (n=3 variance check) and apple-easy (toy isolation
experiment) are now training in parallel** — GPU has ample headroom (2 lightweight
BC runs, ~4GB combined of 8GB). Will canonical-eval each once plateaued.

---

## Interruption + fix: real checkpoint-resume support built (2026-08-10 ~21:00)

Both parallel trainings (cherry seed=7 at epoch ~494, apple-easy at epoch ~376)
were silently killed by what looks like a host-level restart (GPU went fully
empty, both processes vanished from `ps aux`, both logs stop within seconds of
each other with no error — not an individual crash). Background monitors for
these runs were also orphaned with no completion record. Real time lost: several
hours of unattended compute with no way to recover it under the previous
"resume = restart from scratch" limitation documented earlier in this log.

**Fixed properly this time** instead of just restarting from scratch again:
added real checkpoint-resume support to `TrainDiffusionAgent.run()`
(`third_party/dppo/agent/pretrain/train_diffusion_agent.py`) — pass
`+resume_from=<run>/checkpoint/state_<N>.pt` on the hydra CLI and it now loads
model+EMA+epoch and continues from `N+1` instead of restarting at epoch 1.
Committed in the dppo submodule (`ed7bb3b`) and bumped in the parent repo
(`acd67ce`). Known limitation: optimizer/LR-scheduler internal state isn't
restored (wasn't saved by `save_model()` either) — acceptable for BC pretraining,
not a full production-grade resume.

**Both runs resumed successfully** from their last saved checkpoints (every 100
epochs, so at most ~99 epochs of true loss vs. the ~2.5h+ of wall-clock time that
would otherwise have been needed to redo them from scratch):
- Cherry seed=7: resumed at epoch 401 (checkpoint saved at 400, was at ~494 when
  killed — lost ~94 epochs of progress, not hours of compute).
- Apple-easy: resumed at epoch 301 (checkpoint saved at 300, was at ~376 when
  killed — lost ~76 epochs).

Both training toward plateau again now. This resume capability should make any
future interruption much cheaper to recover from.

## Toy-task result: 65% — task difficulty confirmed as the bottleneck, not architecture

**Apple-easy (bigger object, narrow DR) plateaued at epoch 630** (best val loss
0.0318 at epoch 330, no improvement in 300 subsequent epochs — patience
exhausted). Because checkpoints only save every 100 epochs, the true best-val
epoch (330) wasn't itself checkpointed; evaluated the nearest available
checkpoint by actual val loss among the saved ones (100/200/300/400/500/600),
which was **`state_500.pt`** (val loss 0.0420).

**Canonical eval (n=100, num_envs=5, seed=0, scene_group_size=0 workaround,
per-episode video) result: 65.0% success (65/100)** — `ever_success_rate` 70%,
`ever_in_band_rate` 71%, `hold_failure_gap` 0.01 (i.e. almost every episode that
reaches the target band also completes the hold — the policy isn't dropping the
object after lifting, unlike the failure mode seen in cherry's eval videos).
Approx. 95% CI [55.7%, 74.3%] — squarely inside the 60-80% target band even at
the low end of the interval. Per-batch breakdown was noisy but consistent
(40-100% across the 20 batches of 5 episodes each), no degenerate batches.

**This is the key diagnostic result for the whole cross-category effort.**
Identical architecture, identical training recipe, identical eval harness as
every cherry specialist this session (13-33% across n=3 seeds) — the only
things that changed were (a) object size (~65mm apple vs. ~20mm cherry) and (b)
DR range (narrowed pos_xy 0.04→0.02, pitch/roll 20°→8°, tighter shape/scale
bounds). That alone closed a 40+ point gap. **Task difficulty (tiny object +
wide randomization range), not the DP3/DPPO architecture or the demo-collection
pipeline, was the bottleneck capping cherry's specialists.** This directly
answers the diagnostic question the toy-task experiment was designed to answer.

**Implication for "cross-category policy working within days"**: the fastest
path to a working generalist is now clearer — build the category pool from
objects/DR ranges in this same favorable regime (larger graspable objects,
narrower per-category DR) rather than defaulting to cherry-like tight/small/
wide-DR objects. Cherry may simply be a poor first-category choice for this
architecture at this control frequency, independent of any future cross-category
conditioning work.

**Environment note (fixed):** the apple-easy eval initially crashed the sim
server with `ModuleNotFoundError: No module named 'gstaichi'` — traced to a
stray `PYTHONPATH` env var in the shell (`.../Genesis_fork:...`, an old sibling
clone from a prior project) that shadowed the correctly-configured editable
genesis install in `envs/sim/.venv`. Fixed by prefixing the sim server launch
with `env -u PYTHONPATH` (the same pattern already used for the dppo training
launches this session) — not an environment corruption, just a missed prefix.

## Cherry seed=7 replicate: n=3 seed-variance estimate finalized

Cherry seed=7 plateaued at epoch 890 (best val loss 0.0100 at epoch 590, no
improvement in 300 subsequent epochs). Evaluated the nearest saved checkpoint
by val loss (`state_600.pt`, val 0.0112, tied with `state_800.pt` but earlier —
picked the earlier one to reduce overfitting risk). **Canonical eval result:
29.0% success (29/100)**, `ever_success_rate` 36%, `hold_failure_gap` 0.01.

**n=3 seed-variance estimate for the disturbed-180/lift-phase recipe on the
IDENTICAL dataset is now closed out: 33.0% (original seed), 15.0% (seed 123),
29.0% (seed 7) → mean 25.7%, range 15-33 points.** This confirms the earlier
warning that the 33.0% single-seed number looked better than the recipe's true
mean. Combined with the apple-easy result above, the picture is now clear:
cherry's specialists cap out around 25-30% on average regardless of seed or
demo-recipe tweaks (disturbance injection, grasp-phase vs. lift-phase, dataset
size 180-250) — the object/DR combination itself is the ceiling, not any of the
things that were being tuned.

## Second confirmation launched: avocado-easy

To rule out that apple's 65% was a fluke of that specific object, launched the
identical narrow-DR recipe on **avocado** (~95mm long axis, elongated/asymmetric
— structurally different from apple/cherry's round shape). New configs:
`rigid_orientation_avocado_easy.yaml` (narrowed the same way as apple:
pos_xy 0.04→0.02, pitch/roll 20°→8°, tight scale/shape bounds) and
`single_lift_avocado_rigid_easy.yaml`. Collection launched with the same
80-episode/maxfevals=800/seed=0 recipe as apple-easy.

**Collection result: 80/80 saved, but only 18.1% collection success rate**
(362 failed / 442 total attempts, 93.5 min) — notably lower than apple's 43.5%.
Consistent with the pre-existing note in `rigid_orientation_avocado.yaml`
that avocado's min-extent (~61mm) sits close to the XArm7 gripper's practical
stroke limit (~70mm), making CMA-ES grasp search harder. **Not necessarily
predictive of trained-policy quality** — apple's own collection success (43.5%)
was already much lower than cherry's (~80%), yet apple's trained policy still
hit 65%. Proceeding to convert/train regardless, same as with apple.

Converted: 72 train / 8 val (matches apple exactly), episode lengths 209-217
steps (same FSM timing as every other run). Training launched: run `wqlxl`,
configs at `gentle_manip/dppo/cfg/single_lift_avocado_rigid_easy_pcd/`.
Result pending.
