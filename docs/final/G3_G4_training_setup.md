# G3 / G4 — the next two generalist runs (planned 2026-09-08 evening)

Two training slots before a 2026-09-09 15:00 meeting. This page records **what we decided to run,
what we decided NOT to run, and the evidence behind each call** — so the choices can be audited
later and so a negative result is not re-tried a third time.

Baseline for both: **G2 = `ttukt`** (cluster, recipe v5 with the encoder-consistency term ablated),
and its local twin `wiayg`. Setup reference: `docs/final/G1_G2_training_setup.md`.

| run | what it changes vs G2 | why |
|---|---|---|
| **G3** | action horizon 4 → **16**, executed steps stay **4**; hold tail 10 → **22** | the only untested change with two independent reference implementations behind it |
| **G4** | G3 **+ sample prediction** (`predict_epsilon: False`); FiLM head only if G3's teaser says the conditioning path is the problem | one further change, attributable given G3 |

## 1. Where the baseline actually stands

User's assessment of G2 on the robot, which the sim numbers support: **it approaches mostly
correctly, picks an acceptable grasp except on small objects, and retries moderately.** It is not
broken; it is imprecise in specific ways.

Measured (20-episode teasers, same 20 scenarios):

| object | wiayg (v5 local) | G2 `ttukt` |
|---|---|---|
| mushroom | 17/20 | 15/20 |
| tofu | 14/20 | 14/20 |
| banana_chunk | 13/20 | — |
| cherry_tomato | 3/20 | — |
| tomato | crashed (object-class sim instability) | crashed |

Cherry tomato is the failure worth attacking: the policy closes to a median **27 mm on a 25 mm
object** where the demonstrations close to 21 mm, and it does not respond to more training (2.6x the
gradient steps moved it ~1 mm).

## 2. Candidates considered, and what we dropped

### Dropped: width-adaptation mechanisms — 0 for 5 in this project

The obvious target is width selection. Every policy-side mechanism tried for it has failed:

| mechanism | result |
|---|---|
| aux width head, weights 0.5–2.5 | no success gain at any weight |
| 18b planned-width feed-forward | keeps success, width still flat |
| FiLM head (as a width fix) | collapse, 0.350 vs 0.575 |
| grasp-window loss weighting | no gain |
| residual width actions (v1, and units-fixed v2) | **decisive negative**, 0.170 decaying to 0.03 |

The DEVLOG's own summary: *"Residual width actions are ABANDONED; with 18b, aux weights, FiLM and
window weighting, that closes every mechanism tried for width adaptation."*

Alongside it, two measured conclusions that redirect the effort:

- **width IS recoverable from the cloud** — heads reach corr ≈ 0.8 against a data ceiling of 0.85,
  so the encoder is not losing size; the failure is downstream in policy learning;
- **small-object failures are approach/centering precision, not width** (item 17).

**Synthesis: width and approach are coupled.** A policy that arrives a few millimetres off-centre on
a 25 mm object *cannot* safely close to 21 mm — a wider close is the correct response to a bad
approach. That makes approach precision the upstream variable, and it is what G3 targets.

### Dropped: pooling changes

Motivated as a fix for the width bias. Ruled out by the same measurement: width survives max-pooling
at corr 0.8 vs a 0.85 ceiling. It targets a bottleneck shown not to exist, and it would need new
encoder code written the same evening.

### Dropped: a transformer / ViT point encoder

On clouds this means a point transformer; no such code exists here and it would have to be written
tonight and run unattended. Highest implementation risk of anything on the list, no local evidence.

### Dropped from G3, kept as G4's change: sample prediction

`predict_epsilon: False`. One config line, no new code, and a real argument: epsilon prediction is
worst-conditioned at the LOW-noise end of the schedule, which is exactly where final width and
approach precision are decided. DP3 uses sample prediction for its headline results; DPPO uses
epsilon everywhere. The two references disagree, so it is an empirical question.

⚠ **Two hardcoded lines must be fixed before any sample-prediction checkpoint is evaluated or
deployed**: `predict_epsilon: True` in `cfg/sim2real_v1/eval_diffusion_pointnet.yaml:78` and in
`scripts/deploy_real_dppo.py:178`. A mismatch does not raise — it silently decodes wrong actions.

### Conditional only: FiLM / AdaLN conditioning

Already tried here and **negative** (`tysvo`, 0.350 vs 0.575 baseline, "collapses after 100"). But
that test ran at horizon 4 with `dim_mults: [1, 2]` — one downsample, 4 → 2 — so it conflated FiLM
conditioning with a temporal U-Net on a horizon too short to have temporal structure. At horizon 16
(`dim_mults: [1, 2, 4]`, 16 → 8 → 4) it becomes a fair test for the first time. Reserved for G4 ONLY
if G3's teaser shows the conditioning path is the limiter; otherwise G4 is sample prediction.

## 3. G3 — action horizon 16, execute 4, tail 22

**Why this first.** It is the one change the cluster agent ranked first, both reference
implementations use long horizons (DP3: predict 16 / execute 8; DP: 10–16 / 8), and **the proposal
has never actually been tested here**:

- horizon 8 / execute 4 (`jjjjy`) is recorded as a **FAILED config**, not as evidence;
- the three September-2 real baselines (`tiatg`, `uirro`, `xdxvc`) used horizon 16 but **executed all
  16 steps** — 533 ms open-loop, the regime the DEVLOG then flagged as dangerous
  ("a single bad first inference is committed for 16 steps");
- both predate the 2026-09-03 camera change (D435i, sim FOV 46 → 43.15, TCP floor raised).

Predicting 16 while executing 4 keeps reactivity exactly where it is today (133 ms) and uses the
longer prediction purely as a consistency regularizer. Absolute action targets mean no compounding
over the longer chunk.

**The hold tail MUST move with the horizon.** Chunks never run off the end, so the number of chunks
lying wholly inside the hold tail is `K − horizon + 1`:

| setting | training chunks | all-hold | share |
|---|---|---|---|
| today: horizon 4, tail 10 | 1,057,346 | 39,501 | 3.7 % |
| horizon 16, tail 10 (unchanged) | 989,630 | **0** | **0.0 %** |
| horizon 16, tail 20 | 1,046,060 | 28,215 | 2.7 % |
| **horizon 16, tail 22** | **1,057,346** | **39,501** | **3.7 %** |
| round-2 style: horizon 4, tail 60 | 1,339,496 | 321,651 | 24.0 % |

At tail 10 the policy would see **zero** all-hold chunks and never learn to commit to stillness.
Tail 22 reproduces today's numbers *exactly* (both counts depend only on `K − horizon` and
`length − horizon`, and adding 12 to each leaves them unchanged), so stop supervision stays at the
3.7 % the current policies were trained on. The setting that destroyed retry was round 2's tail 60 at
24 %, six times today's dose — tail 22 is nowhere near it. **Rule: keep `tail − horizon = 6`.**

No re-collection: `gentle_manip/dppo/augment_hold_tail.py` appends the 12 extra frames to the
converted npz (+8 % data, ~14 GB, minutes) and records `hold_tail_k` for provenance.

## 4. G4 — sample prediction on top of G3

One further change, so a regression is attributable. Fix the two hardcoded `predict_epsilon` lines
first. Note DDIM is unavailable with sample prediction, which costs the generalist nothing: it
denoises in 20 steps = 13 ms against a 133 ms budget (DDIM mattered only for the 100-step RGB runs).

Fallback if G3 is a clear catastrophe (e.g. 3/20 where the baseline gets 14): G4 reverts to the G2
recipe with only the tail correction.

## 5. Not a training change: temporal ensembling

Predicting 16 and executing 4 means four overlapping predictions cover every timestep; averaging
them with exponential weights (the ACT recipe) is a direct smoothness win at contact and needs **no
training run**. Implemented deploy-side and mirrored in the eval adapter so sim and robot agree.

It likely **replaces** the `--smooth-alpha 0.6` command smoothing rather than stacking with it —
both smooth the same signal, and running both would double-smooth and could blunt the fast close.
Test with smoothing disabled in that entry.

## 6. Open observation to explain: orientation drift across retries

User: on retry the orientation changes each time, accumulates, and eventually goes out of
distribution. Actions are ABSOLUTE, so this cannot be integration error — the mechanism must be
closed-loop feedback: the policy conditions on its own (already rotated) proprio and predicts a
slightly further-rotated target each cycle.

The likeliest root cause is in the data, not the model: **the demonstrations contain no recovery
behaviour at all.** The frozen collector plans one CMA-ES grasp and executes it; both retry ideas —
lift-failure detection with regrasp, and deliberately induced failures for retry coverage — are
recorded in CLAUDE.md as NOT YET IMPLEMENTED. CLAUDE.md predicted exactly this: *"a policy trained
purely on clean successes has never seen what to do after a slip, and won't know how to recover at
deployment"*. Retry is therefore extrapolation, and orientation is the least-constrained dimension
during it.

Ruled out: the euler ±π seam. That was found and fixed (`euler_frame_offset_deg: [180, 0, 0]`,
`docs/debug_partC_euler_action_anomaly.md`); before the fix it cost ~0 % success (run `oppsu`).

What may help without new data: horizon 16 makes each attempt one coherent plan instead of a
re-decision every 4 steps, and temporal ensembling damps per-cycle wander. Neither is a real fix —
that needs recovery demonstrations, which is a collection change and out of scope before the meeting.

## 7. Schedule and evaluation

| | | |
|---|---|---|
| G3 | 21:15 → 03:00 | 1M gradient steps, ~5.7 h measured |
| teasers G3 | 03:00 → 03:30 | cherry, mushroom, tofu — read as DIAGNOSIS |
| G4 | 03:30 → 09:15 | |
| teasers G4 | 09:15 → 09:45 | |
| canonical evals | 09:45 → 12:45 | 100 episodes, cherry + mushroom, baseline and winner |
| buffer / write-up | 12:45 → 15:00 | |

**Evaluation protocol: 100 episodes, not 200.** `EvalSpec` declares `n_episodes = 100` as the FIXED
canonical value; the 200 in `eval_diffusion_pointnet.yaml` is the deviation. 100 halves each eval to
~45 min, which is what makes both a G4 run and a proper A/B fit in the window.

**Teasers are diagnosis, not ranking.** At n = 20 the spread (5) is inside 1σ (2.8), so they cannot
rank recipes — the 2026-09-08 checkpoint sweep established that. What they DO show, from the
per-episode videos and `signals/` plots, is mechanism: did it approach the right place, did it pick a
sensible width, did it retry, did the orientation wander. That is what steers the next run.

**Objects: cherry_tomato and mushroom.** Cherry is the known hard case; mushroom is the strongest
baseline result, so a regression there is the clearest warning. Tomato is excluded — 4 crashes across
2 checkpoints from a known object-class sim instability, unmeasurable without a substeps change.
