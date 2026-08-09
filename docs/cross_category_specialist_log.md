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
| **Cherry-plain-250** | 250 ep, no disturbance | 80.1% (250/312 attempts) | *training, run `nxyyn`* | epoch 480: train loss 0.0089, val loss 0.0116 (best-so-far 0.0109 @ epoch 440); still edging down, plateau watcher armed |
| **Cherry-disturbed-180** | 180 ep (recovered from a 250-ep target killed at timeout), `disturbance_prob=0.3`, `max=2cm` during "lift" | 80.0% (180/225 attempts) — **identical to plain**, confirms disturbance injection doesn't hurt the demonstrator | *training, run `myspw`* | epoch 110: train loss 0.0223, val loss 0.0242 — still dropping fast, far from plateau |

Eval numbers for the two cherry-250 runs are the next milestone once training loss
plateaus.

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

## Currently running

- `nxyyn` — cherry-plain-250 BC pretraining. `logs/dppo_train/cherry_250_specialist.log`. Plateau watcher armed (task `bdymjczqs`).
- `myspw` — cherry-disturbed-180 BC pretraining. `logs/dppo-pretrain-launch/cherry_250_disturbed.log`. Plateau watcher armed (task `bsnm97qft`).

## Next steps

1. Let both plateau watchers fire naturally; stop each run (process-group-safe kill)
   the moment its watcher triggers rather than waiting for the full epoch budget.
2. Run canonical eval (n=100) on both resulting checkpoints, compare against cherry
   zero-shot-from-mixed (1.0%) and the 60-80% target.
3. **If a specialist clears 60%**: immediately pivot to generalist work — build the
   training set from **specialist rollouts** (RLDG-style) rather than raw CMA-ES
   demos, and design the **low-dimensional** VLM-based per-episode conditioning
   vector at the `DP3Encoder` insertion point.
4. **If neither clears 60%**: check whether the task's DR ranges (±45° pitch/roll)
   or success criterion are simply harder than typical single-object BC benchmarks,
   before reaching for more demo-collection tricks. Regrasp-retry data collection at
   scale is parked (deprioritized) unless revisited explicitly.
