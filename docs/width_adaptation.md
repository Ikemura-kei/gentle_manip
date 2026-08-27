# Width adaptation — LIVE log & plan

**Single source of truth for the width-adaptation campaign.** Updated promptly as experiments
launch, results land, or analysis changes a conclusion. Deep analysis lives in
`docs/width_predictability.md`; literature in `docs/size_adaptation_literature.md`; project-wide
history in `docs/DEVLOG.md`.

**Last updated:** 2026-08-26 ~22:45 CEST
**Deadline:** 2 days from 2026-08-26 (venue). **Goal: adaptive width AND success >= 0.7.**

---

## 0. ⚠ HARD DEPLOYMENT CONSTRAINT (user, 2026-08-27): NO CONTACT SENSING ON THE REAL RIG

**The real XArm gripper has NO force/current feedback**, and **a soft delicate object cannot block
a position-controlled finger**, so there is no stall to detect either. MEASURED (job 1733479, demo
data) — the stall signal is not merely weak, it is INVERTED:

| phase | median lag (achieved - commanded) |
|---|---|
| early closing, PRE-contact | **4.85 mm** (p90 6.11) |
| at/after grasp, CONTACT | **0.51 mm** |

Lag is ~10x LARGER before contact than at it (servo lag during fast closing; by contact the command
has stopped changing and the gripper has caught up). A threshold rule would fire in free space and
never at contact.

**CONSEQUENCE: the contact-triggered stop (§4c) and therefore the 48%/0.833 headline in §1 are
SIM-ONLY.** They rely on `SimFeedback.extra["contact_force"]`, which has no real-robot counterpart.
Do NOT report that combination as a deployable result.

**Deployable substitute under test:** CFG (adaptation) + the VISION level-head floor (caps
over-closing) — same complementary structure, using only the point cloud, which the real rig has.
**Open option for the user:** the rig HAS 2x GelSight Mini finger sensors (CLAUDE.md), currently
excluded by the tactile-free premise. If contact sensing is the key enabler, that premise is worth
revisiting — it is a project-level decision, not a technical blocker.

## 1. [SIM-ONLY — see §0] Working mechanism in simulation (2026-08-27)

**CFG (width-only, scale 3.0) + CONTACT-TRIGGERED STOP (F* = 1.0 N)** delivers **48% of the
demonstrator's aperture range at 0.833 success** — 5.3x the baseline's 9% adaptation, at success
within noise of the luckiest seed (0.883) and ABOVE the 3-seed mean (0.756).

| arm | success | corrAT | range | %demo | gapMIN |
|---|---|---|---|---|---|
| baseline (lulkx@600) | 0.883 | +0.336 | 1.0 mm | 9% | 3.7 |
| contact stop alone (1.5 N) | 0.883 | +0.406 | 1.7 mm | 15% | 1.2 |
| CFG alone (3.0) | 0.433 | +0.386 | 5.6 mm | 49% | 4.3 |
| **CFG 3.0 + contact stop 1.0 N** | **0.833** | **+0.532** | **5.5 mm** | **48%** | **1.0** |
| (alzey — best REAL policy, for scale) | ~0.70 real | +0.229 | 1.4 mm | 12% | — |

**NEITHER MECHANISM WORKS ALONE** — CFG loses half its success; the contact stop only reaches 15%.
Together they keep essentially ALL the adaptation and nearly double the success.

**WHY IT WORKS (the diagnosis, confirmed by construction):**
1. The size dependence was ALWAYS in the learned conditional — the policy encodes size and
   UNDER-WEIGHTS it at sampling. Guidance turns that weight up: 2% -> 23% -> 49%, monotonic.
   This is why every attempt to SUPPLY more information failed; nothing was missing.
2. Guidance's failure mode is OVER-CLOSING (success 0.433, peak stress up).
3. The contact stop caps exactly that: closure halts where the object actually is, so PHYSICS sets
   the final width and there is no prediction to be wrong.

**EVIDENCE QUALITY: all WITHIN-RUN** (one checkpoint `bxopf@600`, only inference-time flags differ),
so it is immune to the seed confound that invalidated the training-side comparisons (§2g, seed band
-9%..+19%). CANONICAL 200-ep + video launched on the best point AND on CFG-alone (1732732/3) so the
"neither alone" claim is made at eval grade.

**STILL TO CONFIRM:** canonical n=200; per-category behaviour on the generalist policy; and the real
robot, which is the only test of gentleness that counts.

## 2. Current state of the numbers

### 2a. THE SCOREBOARD — report mm of aperture, not just correlation

**PROTOCOL WARNING — never mix these.** All success rates in the table below are the 60-episode
WIDTH PROBE (`n_episodes=60`, `scene_group_size=1`, video off). The CANONICAL eval is 200 episodes
/ 5 envs WITH per-episode video (hard requirements #1/#2). They differ measurably on the SAME
checkpoint: lulkx@600 = **0.820 canonical** vs **0.883 probe**. Probes SCREEN (40 min); canonical
evals CONFIRM (2 h). **No arm is claimed on probe evidence alone.**

Correlation says width moves in the RIGHT DIRECTION; it says nothing about moving ENOUGH. A
policy can score r=0.9 with a 1 mm range and be useless. **Always report both.**

| policy | r (at-grasp) | small->big width | % of demonstrator range | success |
|---|---|---|---|---|
| demonstrator (data) | +0.841 | **11.3 mm** | 100% | 0.94 (collection) |
| **alzey — best REAL policy** | **+0.229** | **1.4 mm** | **12%** | ~0.70 real |
| afucm | -0.040 | -0.3 mm | -3% | — |
| lulkx base — **re-measured, identical code** | **+0.336** | **1.0 mm** | **9%** | **0.883** probe / 0.820 canon |
| lulkx + latched floor (margin 0) | +0.474 | 3.4 mm | **30%** | **0.250** |
| lulkx + floor, margin 2 mm | +0.469 | 3.2 mm | 28% | 0.450 |
| lulkx + floor, **margin 4 mm** | **+0.511** | 3.2 mm | **29%** | **0.517** |
| **TARGET** | — | **~8 mm** | **~70%** | **>=0.70** |

**Where the target comes from (not an arbitrary threshold):** the demonstrator spans 11.3 mm
between size halves (23.8 mm across the full 0.82-1.49 scale range, slope 35.7 mm/unit scale). A
policy with ZERO adaptation therefore mis-sizes by up to +-5.7 mm at the extremes — which is
exactly the >5 mm over-tight condition that bruises mushrooms, and matches the measured ~30%
over-tight rate. Keeping worst-case width error under ~3 mm requires ~70% of the demonstrator's
range.

**Verdict on alzey: NOT size-adaptive.** 12% of the range. Its real-world success (~70%) and
gentleness are genuine but come from LEVEL (~2 mm wider) and RATE (39% slower closing), which help
every object equally. The paper cannot claim size adaptation on alzey's evidence.

**CORRECTION (01:50):** the baseline correlation quoted all night as "~0.1" (from an older probe,
0.138) is **0.336** when measured with correct normalization and the at-grasp definition. Claims of
the form "floor lifts corr from ~0.1 to 0.474" OVERSTATED the gain; it is 0.336 -> 0.511. The mm
range is unaffected and is the number to lead with: 1.0 mm -> 3.2 mm, 9% -> 29%.

**`gapMIN` validates the user's warning about episode-min:** baseline gapMIN = **3.7 mm** (its min
width sits far below its at-grasp width -> it really does close in mid-air on some episodes), vs
0.2-0.4 mm for the floor arms. Scoring adaptation by MIN width would have read partly noise on the
very arm everything is compared against.

### 2a-bis. TOFU IS NOT A WIDTH PROBLEM (job 1728683, 2026-08-26) — premise overturned

| | corrAT | range | %demo | success | gapMIN |
|---|---|---|---|---|---|
| mushroom baseline | +0.336 | 1.0 mm | **9%** | 0.883 | 3.7 |
| mushroom + floor (best) | +0.511 | 3.2 mm | 29% | 0.517 | 0.4 |
| **tofu baseline (mntlf@500)** | +0.281 | 2.8 mm | **28%** | 0.517 | **9.2** |

Tofu's demonstrator spans 9.8 mm; the tofu BASELINE policy already covers 2.8 mm = **28%**, i.e.
as much as our best mushroom MECHANISM delivers. **Tofu adapts ~3x better than mushroom out of the
box.** The premise that drove the tofu urgency — "tofu fails because width is constant and its
tolerance is narrow" — is FALSE. The width mechanism will not fix tofu's 0.585 ceiling.

What IS wrong with tofu: `gapMIN` = **9.2 mm** (vs 0.2-0.4 for the mushroom floor arms) — the
policy keeps closing far past the grasp point. Peak stress is LOW (20324 Pa vs mushroom 53498,
min_pad 129.7 vs 45.2 mm2), so it is not crushing; the likely story is closures on AIR in the ~48%
of episodes that fail, i.e. a grasp POSITIONING/TIMING problem. Needs its own investigation
(start with the eval videos), NOT more width work.

LESSON: run the baseline diagnostic BEFORE building machinery on an assumed failure mode. Two tofu
level heads were fitted before this probe was run.

### 2a-ter. THE FLOOR IS A TRADE DIAL, NOT A SOLUTION (margin sweep, 2026-08-26)

| arm | success | corrAT | range | %demo |
|---|---|---|---|---|
| baseline | 0.883 | +0.336 | 1.0 mm | 9% |
| floor margin 4 mm | 0.517 | +0.511 | 3.2 mm | 29% |
| floor margin 6 mm | 0.617 | +0.420 | 2.4 mm | 21% |
| floor margin 8 mm | **0.750** | +0.346 | 1.3 mm | **11%** |

Margin 8 mm CLEARS the 0.70 gate — but adaptation has collapsed to baseline (11% vs 9%, corr
0.346 vs 0.336): the floor has stopped binding, so 0.750 is just the baseline leaking through.
**No margin setting escapes the trade**; the exchange rate is ~linear at **+10% range per -0.18
success**. Extrapolated to the 70% target the success cost is unacceptable. This curve IS the
mechanism's Pareto frontier with the OLD head (corr 0.624 at latch, P(over>2mm) 0.51).

**FINAL VERDICT (2026-08-26): the floor is a ONE-PARAMETER family indexed by EFFECTIVE BIAS.**
Ten configurations — 5 margins x 2 head qualities x 2 latch points — all on one curve:

| arm | success | corrAT | range | %demo |
|---|---|---|---|---|
| baseline | 0.883 | +0.336 | 1.0 mm | 9% |
| **old head, margin 4** | 0.517 | +0.511 | **3.2 mm** | **29%** (best) |
| refit head, margin 0 | 0.517 | +0.478 | 2.6 mm | 23% |
| refit head, margin 2 | 0.650 | +0.456 | 2.3 mm | 21% |
| refit + latch@15%, m0 | 0.500 | +0.304 | 2.0 mm | 17% |
| refit + latch@15%, m2 | 0.667 | +0.417 | 2.2 mm | 20% |

**Both offline "improvements" made DELIVERED adaptation WORSE** — the best arm is still the
ORIGINAL head. MECHANISM: the old head's **+3.0 mm over-prediction was acting as a NEGATIVE
MARGIN**, making the floor bind more often -> more adaptation AND more drops. Fixing the bias to
-1.6 mm made it bind less -> fewer drops, less adaptation. **Bias and margin are the same knob with
opposite signs**, so head accuracy and latch timing are nearly irrelevant: only how OFTEN the floor
binds matters. This explains WHY it is one curve, not merely that it is.

CONSEQUENCE: improving the level head is pointless for this mechanism. Anything that converts a
per-episode scalar into a commanded width by CLAMPING rides this curve.

(superseded) ~~the refit head does NOT beat the curve — the floor family is EXHAUSTED.~~

| arm | success | range | %demo |
|---|---|---|---|
| old head, margin 4 mm | 0.517 | 3.2 mm | **29%** |
| REFIT head, margin 0 | 0.517 | 2.6 mm | **23%** |
| old head, margin 6 mm | 0.617 | 2.4 mm | 21% |
| REFIT head, margin 2 mm | 0.650 | 2.3 mm | 21% |

At equal success the refit gives LESS range; at equal range slightly more success. With n=60
(success SE ~0.065) both are 1-2 SE — it sits ON the same frontier. A much better head (corr
0.667->0.763, P(over>2mm) 0.58->0.19) bought essentially NOTHING in the delivered trade.

**Therefore the bottleneck is the MECHANISM'S STRUCTURE, not its inputs.** A per-episode scalar
floor can only turn a prediction into a commanded width by PREVENTING CLOSURE, and preventing
closure is what drops the object. Any mechanism of this shape rides the same curve.

**Requirement for the next mechanism: influence width WITHOUT GATING CLOSURE.** That rules out
floor/max, quantile-floor, and margin variants. It does NOT rule out (a) changing what the policy
LEARNS (align-filtered or width-capped demos, CFG on the width dim), or (b) a mechanism that
reshapes the whole width TRAJECTORY rather than clamping its endpoint.

ORIGINAL expectation, for the record:
~~What the refit arms must do: BEAT this curve, not sit on it.~~ The refit head (corr 0.741 at
latch, P(over>2mm) 0.29) should need LESS margin for the same success -> more binding -> more range
at equal success. If 1728724/1728725/1728949/1728950 land ON the trade line, the floor family is
exhausted and we move to the untried candidates (§5).

### 2d. THE REAL BAR (user, 2026-08-26): beat a naive DP on ~50 real human demos

The requirement is a width-adaptive, MULTI-CATEGORY policy with success >= a naive diffusion policy
trained on real human demos in a LOW-DATA regime — NOT an absolute 70% of demonstrator range.
That is a RELATIVE bar and it changes triage.

**The real demos ARE width-adaptive (user collected them deliberately; measured, job 1729285):**

| | n | grasp width | p10-p90 range |
|---|---|---|---|
| real human demos | 50 | 29.3 +- 4.3 mm | **9.0 mm** |
| sim CMA-ES demos | 539 | 31.5 +- 7.1 mm | 19.3 mm |

So we CANNOT claim "sim provides adaptation the human demos lack" — 4.3 mm sd is comparable to the
sim demonstrator's own size-driven variation (the sim spread is wider mainly because sim DR spans
scale 0.82-1.49, likely broader than the real mushrooms available).

**The resulting story is STRONGER, not weaker:** both data sources are width-adaptive and the
policy collapses on BOTH. Sim demos correlate 0.84 with size -> lulkx commands 1.0 mm (9%). So
"diffusion BC under-uses conditioning on a low-variance action dimension" is demonstrable on TWO
independent datasets. Our mechanism must EXTRACT adaptation that BC fails to capture, from either
source — we cannot lean on a data deficiency.

**Real object size range (user, from MEMORY — records lost; treat as an ESTIMATE, not a
measurement): ~3-4.5 cm cap, i.e. 30-45 mm.** That bounds the real adaptation SLOPE:

| | object size range | demo width range | slope |
|---|---|---|---|
| real (user session) | ~30-45 mm (**15 mm**) | 9.0 mm (p10-p90) | **~0.60 mm/mm** |
| sim CMA-ES | ~27-49 mm (22 mm) | 19.3 mm (p10-p90) | ~0.88 mm/mm |
| sim, fitted slope | — | — | 1.08 mm/mm |

So human teleop adapted at ~60% of the CMA-ES rate — less aggressive but unmistakably adaptive,
over a comparable size range (sim slightly wider). **LIMITATION:** without per-episode pairing this
is an AGGREGATE spread, so size-driven variation cannot be separated from incidental variation the
way it can in sim (corr 0.84). It is CONSISTENT WITH size-driven adaptation, which is weaker than
a correlation. If the paper needs the real corr, a few real mushrooms must be measured against
recorded episodes — the records for this session are lost.

**No current arm dominates the baseline.** lulkx 0.883/9%; floor margin 4 0.517/29%; margin 8
0.750/11%. Each trades. A dominating arm needs CFG (1729257), the align retrain, or the latch arms.

### 2e. [PARTIALLY RETRACTED 2026-08-27 — see 2g] aux-width supervision (job 1730947)

**RETRACTION:** the claim below that this is "DECISIVE" evidence the collapse is in the learning
objective is NOT supported. Its policy-side evidence (aux-width delivering 3% vs baseline 9%) is a
BETWEEN-RUN comparison, and the sibling-seed control (§2g) measures a seed-noise band of **-9% to
+19%** on that metric. 3% sits inside it. What SURVIVES is the OFFLINE half: the encoder perceives
size at 0.927 while policies deliver 3-19%, so the information is present and under-used. What does
NOT survive: "making the information more available made it WORSE."


Aux-width supervision (`aux_grasp_width_weight=1.0`, run `ccpvb`) raises the encoder's size
perception from 0.739 to **0.927** — and makes the POLICY WORSE on both axes:

| arm | success | corrAT | range | %demo | gapMIN |
|---|---|---|---|---|---|
| baseline (lulkx) | 0.883 | +0.336 | 1.0 mm | **9%** | 3.7 |
| aux-width (`ccpvb`) | 0.517 | +0.183 | 0.3 mm | **3%** | 10.4 |

**An encoder that SEES size at 0.927 drives a policy with 3% adaptation — LESS than baseline.**

This completes a clean diagnostic chain:
1. the information IS in the cloud — 0.927 with supervision;
2. policies do not use it — 3-9% of the demonstrator's range;
3. making it MORE available makes things WORSE.

=> **The collapse is in the LEARNING OBJECTIVE, not in perception or representation.** This rules
out the whole "better features / better encoder / better head" family, which is most of what was
tried in items 17-18 and most of tonight.

SURVIVING CANDIDATES (only those that touch the objective or bypass it):
- **generalist multi-object** (§4b) — makes IGNORING the conditioning costly rather than merely
  possible. Now the top learning-side hope.
- **contact-triggered stop** (§4c) — removes prediction from the loop entirely; physics sets width.
- CFG (§5) — intermediate: changes how the trained conditional is SAMPLED, not what it learned.

### 2f. CONTACT-TRIGGERED STOP BEATS THE FLOOR FRONTIER (jobs 1731308/9, 2026-08-27)

**First mechanism all night to improve adaptation WITHOUT paying success.**

| arm | success | corrAT | range | %demo | gapMIN |
|---|---|---|---|---|---|
| baseline | 0.883 | +0.336 | 1.0 mm | 9% | 3.7 |
| floor margin 4 | 0.517 | +0.511 | 3.2 mm | 29% | 0.4 |
| **contact stop F*=0.5 N** | 0.800 | +0.482 | 2.2 mm | **19%** | 1.1 |
| **contact stop F*=1.5 N** | **0.883** | +0.406 | 1.7 mm | **15%** | 1.2 |

At IDENTICAL baseline success (0.883) it delivers 15% vs baseline's 9%; at 0.800 it gives 19%
where the floor frontier would give <11%. Both points sit ABOVE that frontier. `gapMIN` also
IMPROVED (1.2 vs baseline 3.7) — FEWER mid-air closures, the opposite of the feared artifact.

**STATISTICAL HONESTY: not established at n=60.** corr SE ~0.13, so 0.406 vs 0.336 is ~0.5 SE;
the range difference is ~0.7 +- 1.4 mm. NEITHER is individually significant. What persuades is the
PATTERN (two F* values, monotonic in the predicted direction: lower threshold -> more adaptation,
less success) plus a physical mechanism that predicted exactly this shape in advance.
CANONICAL 200-ep evals with video launched (1732424/5/6 at F* = 0.5 / 1.0 / 1.5) — a claim needs n=200.

### 2g. THE SEED NOISE FLOOR — measured, and it kills the training-side comparisons

**Three seeds, SAME data, SAME recipe (jobs 1732445/6):**

| run | success | corrAT | range | %demo |
|---|---|---|---|---|
| lulkx (the "baseline" used all night) | 0.883 | +0.336 | 1.0 mm | **9%** |
| oqzdm | 0.667 | +0.038 | -1.0 mm | **-9%** |
| dntaz | 0.717 | +0.281 | 2.1 mm | **19%** |

**Seed alone spans -9% to +19% adaptation (28 points) and 0.667-0.883 success (0.22).**
`lulkx` is the LUCKIEST of three seeds — and every training-side comparison tonight was made
against it.

**CONSEQUENCES**
- DEAD (inside the noise band, n=1 run each): aux-width 3%, align-filtered 1%, CFG-control 0.700.
  None is an established regression.
- ALIVE (WITHIN-run: same checkpoint, only inference changed, so seed variance cancels):
  the floor's 10-point trade curve; the contact stop (15% at baseline success); and **CFG's
  2% -> 23% -> 49% dose-response — 2.6x the TOP of the seed band, and no seed noise produces a
  monotonic response to a guidance scale.**
- RULE going forward: **no between-run claim without >=3 seeds.** Prefer within-run (inference-time)
  comparisons wherever a question allows it.

(superseded) ~~METHODOLOGICAL GAP: training-side comparisons are CONFOUNDED with run-to-run variance~~

Three training-side interventions all came in BELOW baseline on delivered adaptation:

| run | intervention | success | range | %demo |
|---|---|---|---|---|
| lulkx | (baseline) | 0.883 | 1.0 mm | **9%** |
| ccpvb | aux width supervision | 0.517 | 0.3 mm | **3%** |
| gaqcf | align-filtered data | 0.833 | 0.1 mm | **1%** |

Tempting conclusion: "every training-side fix makes it worse." **But these are DIFFERENT TRAINING
RUNS**, so the comparison is confounded with seed/run variance, and I have NO estimate of that
variance. The baseline's 9% may itself be a lucky draw.

A narrowing explanation was tested and does NOT hold: the align filter cut width sd by only 11.6%
and the p10-p90 span by 3% — far too little to explain 9% -> 1%.

**CONTROL LAUNCHED (1732445/6): probe sibling runs `dntaz` and `oqzdm` — SAME data, SAME recipe,
different seeds.** Their spread is the noise floor against which every training-side claim must be
judged. Until then, treat the aux-width and align "regressions" as UNRESOLVED, not established.

**NOTE the contact-stop result is NOT affected**: it compares the SAME policy (lulkx@600) with only
the inference-time controller changed — a within-run comparison, and the only clean one tonight.

### 2h. CFG WORKS — 49% of demonstrator range (jobs 1732352/3/4, 2026-08-27)

| arm | success | corrAT | range | %demo | peak stress |
|---|---|---|---|---|---|
| baseline | 0.883 | +0.336 | 1.0 mm | 9% | 53.5k |
| contact stop 1.5 N | 0.883 | +0.406 | 1.7 mm | 15% | 51.6k |
| CFG control (scale 0) | 0.700 | +0.299 | 0.3 mm | 2% | 52.2k |
| CFG scale 1.5 | 0.650 | +0.323 | 2.6 mm | **23%** | 53.2k |
| **CFG scale 3.0** | 0.433 | +0.386 | **5.6 mm** | **49%** | 55.9k |

**Guidance amplifies width adaptation MONOTONICALLY and HUGELY: 2% -> 23% -> 49%** — 5x any other
mechanism and over half the demonstrator's full range.

**This settles the conditioning question.** The size information IS in the learned conditional; the
policy encodes it and UNDER-USES it. Guidance extracts it. That is the conditioning-collapse
diagnosis confirmed by a mechanism built from it — and it means the objective's problem is
under-WEIGHTING the conditioning, not failing to represent size.

COST: success 0.433 at scale 3.0, and peak stress RISES (55.9k vs 53.5k) — it OVER-CLOSES.

**SYNTHESIS LAUNCHED (1732514-7): CFG x contact stop.** Their failure modes are complementary —
guidance supplies width VARIATION but over-squeezes; the contact stop's entire function is halting
closure when the object is reached. Grid: CFG scale {1.5, 3.0} x F* {1.0, 1.5}.

### 2b. Ceilings and links

| quantity | mushroom | tofu |
|---|---|---|
| corr(cloud@t=0 -> object size) — perception ceiling | 0.739 | 0.842 |
| corr(size -> demonstrator width), success-only | 0.841 | 0.791 |
| corr(size -> width), ALL episodes (selection effect) | 0.708 | 0.791 |
| R2(width ~ size + align) | 0.818 | 0.818 |
| base policy success | 0.820 (lulkx@600) | 0.470 (gadkf@300, climbing) |

## 3. Experiment ledger

### Completed — mechanisms
| arm | result | verdict |
|---|---|---|
| aux width head (w=0.5/1.5/2.0/2.5), FiLM, 18b feed-forward, loss reweighting, residual actions | no adaptation | dead |
| per-step head, SIGHTED | copies proprio width (80->79.4, 28->28.6), 0.000 success | dead (wrong target, see §5.1) |
| per-step head, BLIND | ramp ~15 mm early, 0.000 success (end-to-end CONFIRMED vs lulkx 0.820), lift onset 2% | dead |
| discretised head (K=64 CE, "mode averaging" fix) | ramp FLAT, MAE 4.5 vs 3.9 mm, middle-band 23 vs true 14 | dead — premise refuted |
| **latched floor** `max(w_policy, w_level)` | **corr 0.474** (first real adaptation) but success 0.250 | promising, mis-centred |
| quantile level heads tau=0.10 / 0.25 | P(over>2mm) 0.58 -> 0.04 / 0.10; bias -6.7 / -3.8 mm | ready, unevaluated |

### Running (as of last update)
| job | arm | expect |
|---|---|---|
| 1728497 | baseline probe (no floor/shift), corrected NORM | anchors the "~0.1" baseline with identical code |
| ~~1728498/1728499~~ | shift alpha=0.5/1.0 — **VOID, my unit bug** (absolute width converted with the delta scale factor -> uniform -8mm squeeze, 0.000 both arms) | relaunched |
| ~~1728625/1728626~~ | shrinkage shift alpha=0.5/1.0 — **DEAD, genuinely** (conversion verified -0.351 norm, still 0.00 over 20 eps; killed by the degenerate watchdog). Inherits the same eval-time +3mm over-prediction but applies it as a UNIFORM widening, whereas the floor only loosens when it binds. | dead |
| 1728724 / 1728725 | **floor + REFIT head, margin 0 / 2 mm** | BEST CANDIDATE: head bias now -1.6mm, so little/no margin needed |
| ~~1728500~~ margin 2mm | **DONE: succ 0.450 (from 0.250), corr 0.469, 28% range, lift% 1.00, gapMIN 0.2** — the +3mm debias is nearly FREE (success +0.20 at ~0 adaptation cost) | confirms bias diagnosis |
| 1728501 | floor margin 4 mm | does more debias help, or start costing adaptation? |
| 1728675 / 1728684 | level-head refits, 60 epochs + WD (mushroom, tofu@mntlf) | 0.667 -> ~0.77 expected; lifts EVERY arm |
| 1728683 | **tofu baseline width probe** (mntlf@500) | UNTESTED ASSUMPTION: is tofu's 0.585 ceiling even a width problem? |
| 1728356 | aux-width retrain (size supervision in ENCODER) | tests whether 0.739 perception is a lower bound |
| 1728066 | latched floor, margin 0 (full eval) | success number for the 0.474 arm |
| tofu650 ckpts 500/600 | tofu curve | is tofu's failure width-related at all? |

### 2c. Level heads — REFIT (60 epochs + weight decay) fixes both accuracy AND bias

| head | policy succ | corr | bias | P(over>2mm) |
|---|---|---|---|---|
| mushroom lulkx@600 — OLD (8 ep) | 0.883 | 0.743 | **+3.0 mm** | **0.58** |
| mushroom lulkx@600 — REFIT | 0.883 | **0.763** | **-1.6 mm** | **0.19** |
| tofu gadkf@300 — REFIT | 0.470 | **0.802** | -1.9 mm | 0.15 |
| tofu mntlf@500 — REFIT | 0.585 | 0.762 | -1.5 mm | 0.18 |

The +3.0 mm over-prediction that caused the floor's ENTIRE success collapse (0.867 -> 0.250) was a
TRAINING ARTIFACT, not a property of the features. Refitting removes it and raises corr. Effect is
consistent across objects and checkpoints.

**Level-head quality does NOT track policy quality:** gadkf@300 (policy 0.470) gives a BETTER head
(0.802) than mntlf@500 (policy 0.585, head 0.762). Do not assume the best policy is the best base
for a retrofitted head.

## 4. Decisions taken (with reasons)

- ~~Stop building heads (prediction at ceiling ~0.62)~~ **RETRACTED 2026-08-26 22:20.** That
  ceiling assumed SIZE MEDIATES EVERYTHING and rested on an UNDER-TRAINED head. Job 1728668 got
  corr **0.771** cloud->width on the SAME data/features/split where Step 0 got 0.597 — the only
  difference being 300 epochs + weight decay vs 40 epochs. So (a) the cloud carries width info
  beyond scale, and (b) our level head (0.667 at t=0) is under-trained, not at ceiling. Refitting
  properly (1728675) lifts the ceiling for EVERY mechanism currently running.
- **Target METRIC EXTENT (mm), never `scene_scale`.** Scale is category-relative, undefined on
  real objects, and the goal is a multi-category generalist. `width_mm` already IS metric extent
  plus a constant (verified: join corr 0.9992-0.9998, constant ~9.5 mm offset).
- **Do not filter demos on the width residual** — circular, biases the reported statistic. Filter
  on contact QUALITY and take predictability as a by-product.
- **Latch the floor early (t=0), not at closure onset.** Vision corr is 0.667 at t=0 and collapses
  to 0.097 at contact (occlusion). The "latch later" fix would have made it worse.

## 4b. GENERALIST TEASER — multi-object training (user hypothesis, 2026-08-27, job 1730005)

**Hypothesis (user):** training on mushroom + tofu together may fix the collapse. MECHANISM: within
one category the policy can sit at ~30 mm and be roughly right for every object — the mean-width
shortcut is cheap. With two categories under a SHARED normalization (merge_npz_datasets does joint
renorm) the marginal width distribution is much broader and likely bimodal, so that shortcut
becomes expensive and the conditioning may finally get used.

Setup: mushroom (align-filtered, 481 incl. real) + tofu (587) = **1068 episodes**, epochs scaled
600 -> **350** so gradient steps stay comparable to lulkx (600 x 589) — data volume is not
confounded with training length.

**Read-out must be PER CATEGORY.** The informative failure mode is a policy that adapts BETWEEN
categories (two clusters) but stays flat WITHIN each — that is a category classifier, not size
perception, and it would look adaptive in aggregate. Report mushroom range and tofu range
separately, not just the pooled number.

Supporting evidence this could work: cross-category size perception TRANSFERS (~0.55 both
directions, job 1729369), so the encoder can in principle read size for both.

## 4c. LEADING CANDIDATE: CONTACT-TRIGGERED STOP (2026-08-27, analysis done, plumbing pending)

The floor family failed STRUCTURALLY: it converts a prediction into a width by clamping, so
prediction error -> gripper stops at the wrong place -> drop. Requirement for any successor:
**influence width WITHOUT gating closure at a PREDICTED value.**

`size_adaptation_literature.md` §2c supplies it — "a grasp is itself a measurement". Close until
CONTACT FORCE reaches F*, then stop. The stopping point is set by PHYSICS, not by a prediction,
so there is no prediction error to cause drops, and NO size estimate is needed at all.

**Why the width adaptation comes for free:** at a force-triggered stop,
`width ~= object_size - indentation(F*)`, and indentation at a given force is roughly a material
property -> width tracks size with SLOPE ~1.0, at or above the demonstrator's 0.88-1.08.

**Supporting data (dr_params `grip_N`, success-only):**

| | grip_N | CV | corr(grip, scale) | corr(width, scale) |
|---|---|---|---|---|
| mushroom (n=653) | 2.10 +- 1.50 N | 0.72 | +0.376 | +0.841 |
| tofu (n=614) | 2.22 +- 1.57 N | 0.71 | +0.475 | +0.791 |

The demonstrator is NOT constant-force — grip rises with size (heavier objects need more hold).
So the design question is the THRESHOLD, not the mechanism: constant F* adapts width well but
under-grips heavy objects; a high F* over-squeezes small ones. Resolution: scale F* by estimated
mass, where a COARSE size estimate suffices — force feedback ABSORBS the error instead of turning
it into a drop. That is the qualitative difference from the floor.

**It also earns the "reactive" answer to the reviewer** (see §6): an open-loop pose+width regressor
CANNOT stop on contact; this can. Testable by perturbing mid-grasp or varying stiffness at fixed
geometry.

**IMPLEMENTED 2026-08-27 (jobs 1731308/1731309, F* = 0.5 / 1.5 N bracketing the demonstrator's
2.1 +- 1.5 N).** Four minimal edits: `policy_env` emits `contact_force` in the step info (reusing
the sim_feedback already fetched — no extra physics round-trip); `genesis_venv` aggregates it as
the chunk MAX (contact ONSET is what matters); `harness` gains an OPT-IN `observe_info` hook that
is protocol-neutral (no change to metrics, and the policy NETWORK still sees only its declared obs
keys); `eval_agent` latches the commanded width the first step force exceeds F*. Plus a NO-OP GUARD
that raises if no contact arrives within 8 steps — which immediately caught B13 (rpc.py drops
un-whitelisted info keys) in 3 minutes instead of two 40-minute false negatives.

ORIGINAL note:
~~IMPLEMENTATION (not done — multi-file, deliberately not attempted unattended at 00:30):~~
contact force exists in sim (`sf.extra["contact_force"]`, `CONTACT_FORCE_THRESH_N`) but only
surfaces as a PRIVILEGED OBS. It must be threaded into the step `info` (like `stress_max`,
policy_env.py ~L254) -> serve_env -> SimEnvClient -> venv info -> the policy adapter. At REAL
deploy the gripper's own force/current feedback provides it, so this is NOT privileged information
at deployment — only the sim plumbing makes it look that way.

## 4d. THIRD CATEGORY: raspberry is TOO SLOW to collect in time (2026-08-27)

Smoke (job 1730025, 60 eps) runs at **7.8 FPS across 8 envs (~1 FPS/env)** — far slower than the
mushroom collections. A full 650-episode raspberry set would take **20+ hours**, which does not fit
the deadline, and a partial set is worse than none for a generalist claim. Likely cause: a 15 mm
object at the task's MPM grid density costs many substeps per unit of useful motion.

**SMOKE PASSED (26-08-27-yxu): 95.2% success, 0 fallbacks, 60 eps in 133 min (2.2 min/ep).**
Raspberry grasping is VIABLE and clean — the only obstacle is rate. Width distribution is the
point: **16.9 +- 4.0 mm** vs mushroom 31.5 +- 7.1 and tofu ~42, i.e. THREE cleanly separated width
regimes, which is exactly the pressure the generalist hypothesis needs (no single mean serves 17,
32 and 42 mm). Its corr(width, scale) is only +0.346 (vs 0.84 / 0.79) — a near-spherical small
object whose width is less determined by scale.
DECISION: 650 eps would take 24h and is out. A 200-episode set (~7h) IS affordable and gives a
genuine 3-category teaser — LAUNCHED. Do NOT launch the full 650. The generalist teaser stands on
**mushroom + tofu** (1068 eps, job 1730005). If a third category is wanted later, **strawberry**
(30.5 mm local cross-section) is the pragmatic pick — less size contrast than raspberry's 13.2 mm,
but compact and much cheaper to simulate. The raspberry smoke is still being finished for its
viability + collection-rate numbers, which are cheap now and useful for future planning.

## 4e. GENERALIST (mushroom+tofu) — hypothesis NOT supported by this run (job 1730005/xktwc)

| | success | corrAT | range | %demo |
|---|---|---|---|---|
| mushroom baseline | 0.883 | +0.336 | 1.0 mm | 9% |
| **generalist on mushroom** | 0.783 | +0.042 | -0.6 mm | **-6%** |
| tofu baseline (mntlf) | 0.517 | +0.281 | 2.8 mm | 28% |
| **generalist on tofu** | 0.617 | +0.125 | 0.7 mm | **7%** |

Within-category adaptation got WORSE on both. The telling detail: at-grasp width is **~33 mm for
BOTH categories** (33.4/32.7 mushroom, 32.6/33.3 tofu) — it did not even learn the CATEGORY-level
split that would have masqueraded as adaptation in a pooled metric. It collapsed to a single GLOBAL
mean across both objects.

POSITIVE: success on tofu IMPROVED (0.617 vs 0.517) — multi-object training helps the harder task's
competence, just not the width collapse.

CAVEAT (applied consistently): n=1 run; mushroom's -6% is INSIDE the -9%..+19% seed band (§2g).
This does not REFUTE the hypothesis, it fails to support it. A fair test needs >=3 seeds.

**FOLLOW-UPS LAUNCHED:** (a) the CONTACT STOP applied to the generalist (1732744/5) — it needs no
special training and works on any policy; (b) the generalist RETRAINED WITH conditioning dropout
(1732746) so CFG becomes available on it — i.e. the working mechanism (§1) applied to the
multi-category setting, which is the actual deliverable.

## 5. Queued ideas, highest value first

1. ~~Condition the width head on the grasp pose~~ **REFUTED 2026-08-27 (job 1728668).**
   V (vision@t=0) 0.771 | P (proprio pose@closure) 0.574 | **V+P 0.748 — pose HURTS slightly.**
   Why the inference was wrong: R2=0.818 used the MEASURED `align`, which comes from contact
   geometry (object surface normals at the patch), not from EE pose. Pose becomes align only when
   combined with local object geometry — exactly what is occluded at closure. Dead end.
2. **Align-filtered demos — DEPRIORITIZED 2026-08-26 (job 1729033), my "+0.09" was measured
   against the WRONG BASELINE.** The +0.841->0.933 figure came from RAW clq, but the real pipeline
   already applies the PINCH filter, which alone lifts corr to 0.916. On the actual pipeline input
   the align filter adds only **+0.022** (0.916 -> 0.938) and -7.7% stress, while discarding 20% of
   the data (599 -> 479 episodes). Scale coverage stays clean (74-82% per bin, all mesh variants),
   so it is not harmful — just marginal. Dataset is built and join-proven (corr 0.9999) at
   `dataset/demos/single_lift_mushroom_soft/26-08-25-clq-alignfilt` if we want it later.
   ORIGINAL (over-stated) claim below for the record:
   ~~drop bottom 20% align~~ — **NOTE: the local agent's `--grasp-width-max-mm`
   (df3f0b7) attacks the SAME phenomenon structurally, and better.** They measured CMA grasping a
   banana along its LONG axis (42-79 mm widths vs a ~17 mm cross-section; 4 of 5 spanned the
   crescent end-to-end, none lifted) because those grasps present more pad contact and so WIN on
   `area_min` + the pressure term. Bounding width removes them from the SEARCH SPACE; my filter
   removes them POST HOC and throws away collected episodes. Prefer the structural bound for new
   collections; keep filtering for data we already have. Both target the align-driven width
   variance that makes width unpredictable from the cloud.: corr 0.841 -> **0.933** AND stress -10%, scale
   coverage preserved. Sharpens the conditional the diffusion policy fits, attacking mean-seeking
   at the source. Needs a dataset rebuild + re-merge -> MUST pass `verify_derived_dataset.py`
   (this path is where the v33 poisoning happened).
3. **Proprioception after contact (RMA-style history module).** Vision dies at contact exactly
   when gripper width + contact force become a DIRECT size measurement. The two channels are
   complementary IN PHASE; a single-frame head cannot exploit that.
4. **Leave-one-mesh-variant-out** rerun of Step 0 — current 0.739/0.842 measure interpolation, not
   generalisation. Cheap; needed before any novelty claim.
5. **Classifier-free guidance on the width dim** — the one untested item from the original list;
   targets the diffusion path underusing its conditioning. Full retrain.

## 6. Reviewer questions to answer in the paper

- *"Why not regress grasp pose from the cloud and execute open-loop?"* — Reactivity. But our
  policy is not currently reactive either, so this must be EARNED (see §5.3) and demonstrated
  (perturb mid-grasp, or vary stiffness at fixed geometry).
- *"Why not regress width and do a top-down grasp?"* — NOT refuted by yaw geometry on our data
  (CMA-ES picks flush grasps, so the sqrt(2) diagonal effect never appears; measured corr -0.08).
  Either find a different argument or just run the baseline.
- Novelty claim ("no one has studied parallel-jaw aperture adaptation in IL") rests on a 6-query
  scan — needs systematic screening before print.

## 6b. Evaluation funnel (do not skip a stage)

1. **Width probe** — 60 eps, `scene_group_size=1`, no video, ~40 min. Screens mechanisms and gives
   corr + mm range + %demo. Cheap enough to sweep. NOT an eval.
2. **Canonical eval** — 200 eps / 5 envs, per-episode video, ~2 h. The only number that goes in a
   table or a claim. Required for any arm we intend to keep.
3. **Real robot** — the only test that settles gentleness.

**Canonical results so far (200 eps, video):**

| arm | probe (60 ep) | canonical succ | ever | **sustained stress** |
|---|---|---|---|---|
| lulkx@600 baseline | 0.883 | **0.820** | 0.865 | **28.1 kPa** |
| floor margin 0 | 0.250 | **0.260** | 0.310 | **14.0 kPa** |

**THE FLOOR HALVES SUSTAINED GRIP STRESS (28.1 -> 14.0 kPa).** Mushroom yield is ~40 kPa, so the
baseline holds at ~70% of yield (bruising territory) and the floor at ~35% (safe). Peak stress
agrees directionally (53.5 -> 47.7 kPa on the probe arms) but SUSTAINED is the bruising-relevant
number — prolonged compression during the hold. So the floor is NOT "adaptive but worse": on the
grasps it completes it is dramatically GENTLER, and its only defect is dropping too many. That
defect is exactly what the refit head (bias +3.0 -> -1.6mm) and latch@15% (P(over>2mm) 0.51 ->
0.29) target. Target outcome: baseline-level success AND half the grip stress AND 3x the aperture
range — a much stronger claim than "width is adaptive".

Probe-vs-canonical agreement is close on the WEAK arm (0.250/0.260) and optimistic on the STRONG
one (0.883/0.820, +0.063). So probes are fine for screening and ranking, but any arm landing near
the 0.70 gate MUST be confirmed canonically before it is believed. Videos for the floor arm:
`lulkx/eval/state_600_floor_eval/render/` (200 clips).

Everything else is probe-only and must NOT be reported as an eval result.

## 7. Update protocol

Append to §3 the moment a job launches or lands; move conclusions into §2/§4 and revise §5's
ordering when a result changes the ranking. Bugs go to the DEVLOG bug ledger, not here.
