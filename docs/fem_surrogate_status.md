# FEM gentleness surrogate — scientific status & paths forward (2026-08-30)

Companion to `grasp_synthesis_model.md` (the *what*, verified against code). This document is the
*how good is it and what next*: every claim below is measured in this repo, with the run/date noted.
Written to answer three questions:

1. Is the FEM synthesis scientifically sound enough to present as a pipeline component?
2. Why do small objects (raspberry) come out mostly ABOVE yield — surrogate failure, or skewed
   evaluation?
3. Can we get closer to full FEM while keeping the computation fast?

---

## 1. Verdict up front

**The surrogate is scientifically presentable as a *grasp-pose selector*, with one honest
sentence of scope.** It is NOT presentable as a stress *predictor*, and the raspberry problem is
**not the surrogate's fault** — it is the **executor's** (see §3). Concretely:

| claim | status |
|---|---|
| "FEM-based gentleness-aware grasp synthesis" | **SAFE** (objective contains stress terms; hard yield guard; measured sub-yield outcomes on 4/7 objects) |
| "the surrogate ranks candidate grasps by simulated stress" | **PENDING** — sub-yield controlled test ρ = +0.52 (p = 0.085, n = 12); n = 40 run in progress |
| "the surrogate predicts the stress the object experiences" | **FALSE** — planner ~3× low vs MPM (partly definitional: contact-masked top-10% vs unmasked), and zero correlation past yield (MPM saturates) |
| "demonstrations are sub-yield" | **PER-OBJECT** — mushroom 99.6 % (n=250) / tomato / tofu 100 %, strawberry 94 %; cherry_tomato 56 %, raspberry 19 % — the latter two are executor bugs, §3 |

The pipeline sentence I would defend in the paper: *"grasp poses are selected by a fast
linear-elastic FEM surrogate (quasi-static, position-controlled rigid-pad contact, solved once
per object at E = 1 and rescaled analytically); the executed squeeze is calibrated per object so
that the resulting demonstrations remain below the material's yield stress, verified per episode
in the simulator."* Every clause of that is (or will be, after §3's fix) measured.

## 2. Validation ledger (all measured, this repo)

| test | result | date |
|---|---|---|
| observational corr., scene varied, n=10 | ρ = +0.842 — **artefact** (scene size drives both; planner vs scale ρ = −0.67, MPM vs scale −0.89) | 08-28 |
| controlled, fixed scene, width swept, PAST yield | **ρ = 0.000** — MPM `ElastoPlastic` saturates at the yield surface; surrogate (no yield model) keeps rising | 08-28 |
| controlled, fixed scene, SUB-yield | ρ = +0.517 (p = 0.085, n = 12) — trend, not significant; **n = 40 running** | 08-28/30 |
| absolute calibration | planner 6.8–18.8 kPa vs MPM 20.7–46.4 kPa on the same grasps (~3× low; contact-masked vs unmasked is part of it; unmasked `hi_1` never compared — open) | 08-27 |
| per-object outcome (16-ep, own material) | mushroom 100 %/100 %, tomato 100 %/100 %, tofu 76 %/100 %, strawberry 100 %/94 %, cherry 89 %/56 %, raspberry 100 %/**19 %**, banana_chunk **53 %**/100 % (success/sub-yield) | 08-29 |

## 3. The raspberry diagnosis: the evaluation is NOT skewed — the EXECUTOR is

Strain accounting on the 16-episode raspberry verification run (`26-08-29-zlb`):

- Executed closure beyond the planned contact width: **4.4–5.5 mm on a 13.7 mm object = 32–40 %
  strain**, against a yield strain of `σ_y/E = 15 %`. The object is *commanded* to 2.1–2.7× its
  yield strain; the measured top-10 stress pinned at 1.0–1.2× yield is exactly MPM plastic
  saturation, i.e. the metric is reporting the truth.
- Where the closure comes from: **2.5 mm hardcoded baseline** (`width_cls = plan − 0.0025 − …`,
  the **third** instance of the unscaled-constant bug class after the squeeze and the firm phase)
  + 0.94 mm material-aware squeeze + 0.94–2.1 mm firm. The unscaled baseline alone is **18 % of
  the raspberry** but only 8 % of the mushroom — which is why the mushroom certifies nothing.
- Metric-skew check: top10/mean ratio is a stable ~2.3 across episodes — no small-object
  pathology in the reduction itself.
- **Confirmation probe** (running): raspberry with `--grasp-extra-close 0` → closure collapses to
  the 2.5 mm baseline alone (≈1.2× yield strain commanded). Prediction: sub-yield fraction rises
  sharply at unchanged success (the berry weighs ~1.5 g; holdability needs ~0.02 N).

**Also found while auditing: `execute_offset` is never passed by the v3 collector.** The scoring
hook exists precisely to evaluate candidates at the width the robot will actually reach
(documented in `finger_grasp.py`, with the pre-fix ρ = +0.10 note), but no call site wires it. The
planner therefore scores a grasp 4–7 mm wider than the one executed — stress is steeply nonlinear
in indentation, so this both weakens the correlation study and hides the executor's over-closure
from the planner's own yield guard.

**Consequence for §1's verdict:** the surrogate's *selection* was never contradicted by the
raspberry data. The chain "surrogate selects sensible pose → executor over-squeezes it with
unscaled constants → metric honestly reports past-yield" is fully accounted for.

## 4. Why not Genesis's full FEM + coupling, quantified

Genesis ships an implicit FEM solver (Newton + PCG per timestep) and SAP/IPC couplers. The search
evaluates **~7k–35k candidates per grasp, per env** (1145 fevals × 6 starts, × escalation). Our
per-candidate cost is one multi-RHS back-substitution on a cached factorization (+ a small dense
Schur solve) — sub-millisecond at ~2–6 k tets. A dynamic FEM+IPC evaluation needs settling
(hundreds of implicit steps, each Newton×PCG); even at an optimistic 1 s per candidate the search
would take hours **per grasp**. The E = 1 linearity is also lost (material DR would need re-solves
instead of scalar rescaling). Full-FEM-in-the-loop is not a tuning change; it is a different
system.

## 5. Paths toward full-FEM fidelity at surrogate cost (ranked)

1. **Fix the executor, not the model** — per-object measured calibration of the TOTAL closure
   (baseline + squeeze + firm from ONE budget), chosen at collection start by sweeping ~4 closure
   values on throwaway episodes and keeping the largest whose measured `priv_stress` median stays
   < 0.8× yield. Already recommended in the DEVLOG (08-29); the analytic `K·(σ_y/E)·L` rule
   mispredicts in both directions (raspberry too large, banana_chunk too small) because contact
   geometry enters the real stress. **Cost: minutes per category. This unblocks the 3 REVIEW
   objects and is the only item required before the cluster collection.**
2. **Wire `execute_offset`** (exists, unwired) so every candidate is scored at the width that
   will actually be executed under the calibrated budget. Zero added cost; makes the planner's
   yield guard operate on the real operating point.
3. **Plasticity-aware objective from the SAME solve** — the MPM material is J2 elasto-plastic, so
   past yield the physically meaningful quantity is plastic work, not peak stress. A cheap
   surrogate: per-element excess `max(σ_vm − σ_y, 0)` summed over volume (or the volume fraction
   above yield) from the existing E = 1 solution. Post-processing only; keeps the factorization
   and (per-E thresholding aside) the DR sweep. This is what would make the objective
   *discriminating in the saturated regime* where today it is provably flat (ρ = 0.00).
4. **Deformed-configuration contact, top-K only** — one Picard pass (solve → move boundary by u →
   re-derive contact set → re-solve) applied not during search but to the final few distinct
   candidates before selection. ~2× cost on ~5 candidates ≈ +ε on the search. This is the item
   that addresses the parked banana's failure mode (contact fixed on the nominal mesh) without
   paying for it 35 k times.
5. **Nonlinear rescoring of the argmax only** — neo-Hookean + a few Newton load steps on the one
   chosen grasp as a validity check (accept/reject + report), seconds per grasp. Optional.
6. **Full Genesis FEM+IPC spot-checks offline** — not in the loop; use it to validate items 3–5 on
   a handful of grasps per object for the paper's appendix.

Items 1–2 are executor plumbing and cheap. Item 3 is the one *model* change with a measured
justification. Items 4–5 are refinements with clear but unproven value — do them only if the
n = 40 sub-yield correlation comes back weak after 1–3.

## 6. Open items gating the paper text

- n = 40 controlled sub-yield correlation (running) → decides whether ρ is quotable.
- `hi_1` (unmasked planner percentile) vs MPM — separates the definitional part of the 3× gap.
- Raspberry probe (running) → experimental confirmation of §3.
- Items 1–2 above, then re-run the 7-object verification.
