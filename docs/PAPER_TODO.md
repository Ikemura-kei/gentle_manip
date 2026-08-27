# Paper TODO — Gentle Manipulation Sim2Real

Companion to [DEVLOG.md](DEVLOG.md) (engineering) and [paper/](paper/) (method + outlines).
This file tracks **only** what is needed to submit, in priority order. DEVLOG tracks what is
needed to make the system work; the two overlap but are not the same list — several DEVLOG
items are irrelevant to the paper, and several items here are pure writing/measurement tasks
with no engineering content.

**Status markers:** **[HAVE]** measured · **[RUN]** in flight · **[NEED]** not started ·
**[CUT]** deliberately excluded from the paper.

---

## Resubmission context (2026-08-26)

This is a **resubmission after rejection** of [arXiv:2510.25405](https://arxiv.org/abs/2510.25405)
(stress-guided RL; our own prior iteration, preprint public). Stated rejection reasons:

1. **Limited object variety / single object.**
2. **Lack of experiments.**
3. *(Discovered while adding those experiments)* the RL policy **did not adapt to object
   orientation or size changes**. RL must explore the variation rather than be shown it, and is
   too sample-inefficient to cover it. Direct imitation learning is faster.

**This is the narrative spine, and it is a strong one:** the method change was *caused by* a
measured limitation, not chosen in the abstract. Reviewers respect a diagnosed pivot. Reason 3
is also the second independent piece of evidence for the coverage thesis — whether you
*explore* the variation (RL) or *demonstrate* it (real-demo IL), covering it is the bottleneck;
an autonomous planner in simulation sidesteps both.

**Prioritisation consequence:** reasons 1 and 2 are *explicit reviewer demands*. The generalist
(multi-object) run and the experiment count are therefore **blocking**, not strengthening.

---

## Story (as of 2026-08-26)

**Working title:** *Coverage Without Demonstrations: Stress-Minimising Grasp Synthesis for
Fragile Objects* (alternatives in `paper/outlines.tex`).

**Thesis.** For gentle grasping the binding constraint is *coverage of physical variation*,
not demonstration count. Real teleoperation cannot buy that coverage at any affordable price;
simulation can, but only if an autonomous demonstrator knows what "gentle" means without a
human. Internal stress — the quantity that actually causes damage, unobservable on hardware —
is *exact* inside a deformable simulator, so gentleness becomes an optimisation objective.

**Three contributions.** (1) the stress-aware autonomous demonstrator; (2) coverage-over-count
with real-hardware evidence; (3) real validation from a single depth view, no tactile/force in
the loop.

**Deliberately cut** (2026-08-26, user): the lead/dwell BC failure modes; the "simulated
ranking inverts on hardware" claim (not validated, and unlikely to survive a CI at n≈20).

---

## A. Blocking — cannot submit without these

| # | item | owner | status |
|---|---|---|---|
| A1 | **Real gentleness proxy.** Post-lift deformation (before/after height or volume) or contact-force logging on real produce. Without it, "gentle" is validated only in simulation, *by the same FEM the planner minimises* — circular. Cheapest fix for the single largest reviewer attack. Days, not weeks. | | **[NEED]** |
| A2 | **Precision tolerance, numerically.** Mushroom ≈33 mm; measure the width band separating drop from crush. Needed in Intro ¶1 to earn the log N ∝ 1/(P−c) argument — without it we are borrowing that law's authority. Partly derivable from `figures/width_at_grasp_all_2026-08-26.png` / the width probe. | | **[NEED]** |
| A3 | **n and CI on every real number.** The ~75% co-train figure and the three-way real comparison. At n=20, 75% has a 95% CI of roughly [51%, 91%] — report it rather than let a reviewer compute it. | | **[NEED]** |
| A7 | **Generalist across object categories.** Rejection reason 1 was *limited object variety / single object*. A resubmission on mushroom alone does not answer it. Was B2 (strengthening); promoted to blocking. | cluster | **[RUN]** |
| A8 | **Orientation- and size-robustness experiment.** Rejection reason 3: the prior RL policy failed exactly here. The resubmission must show the new method *does* adapt across orientation and size, on hardware — this is the experiment that converts the rejection into the paper's argument. Overlaps B1 (coverage ablations); run them as one campaign. | cluster | **[RUN]** |
| A9 | **The Q\_SM degeneracy result, written up.** We implemented \citet{pan2020stressmin} and measured Q\_SM ≈ 0 for every two-pad grasp on organic meshes (mushroom, bunny, bunny head) while a cube clears it — verified across search budget, hull resolution and μ up to 1.5 (`grasp_synthesis/CLAUDE.md` §10.2). This is a **measured limitation of prior work**, not a design preference, and it is what makes the distributed-pad model a contribution rather than a tweak. Currently lives only in an internal doc. Needs a figure or table. | | **[NEED]** (data **[HAVE]**) |
| A6 | **Audit every real-robot number for the ~9 mm perception-bias confound.** The bias was found and corrected mid-campaign ([item1](item1_cube3_simreal_gap.md)); any real result collected before the correction is suspect, and at least one (the N-sweep, B3) is known invalid. Establish which side of the correction each headline real number falls on — including the ~75% co-train result and the three-way comparison — before any of them enters the paper. | | **[NEED]** |
| A4 | **Figure 1 — the coverage failure map.** The 50-real-demo policy's hardware failures plotted against (location × object size), with misgrasp and over/under-squeeze as distinct markers. This replaces the removed saturation claim and motivates the whole paper. Data already collected. | | **[NEED]** (data **[HAVE]**) |
| A5 | **Method section cut to ~1.5–2 pages.** `paper/method.tex` is 424 lines / 9 subsections. Keep: problem setting (compressed), stress-aware synthesis (full detail — it is the contribution), demo generation (recipe as a table), obs/action collapsed to one table. Appendix: evaluation protocol, real-robot deployment, residual perception calibration, implementation details, specialist→generalist mechanics. | | **[NEED]** |

## B. Strengthening — materially improves acceptance odds

| # | item | owner | status |
|---|---|---|---|
| B1 | **Coverage ablations** (size band, shape/mesh pool, OOD size). Under the corrected framing these went from "nice ablation" to **the load-bearing experiment** for the coverage thesis — it is now argued a-priori (¶2) plus one real failure map (Fig. 1), and these ablations are the controlled evidence. Also the only route to lifting the "Stronger" dimension to 8+. | cluster | **[RUN]** |
| B2 | ~~Generalist across categories~~ — **promoted to A7** (explicit rejection reason). | cluster | see A7 |
| B3 | **Real-demo-count curve.** DEVLOG roadmap item 3. ⚠️ **The existing sim sweep across N∈{1,5,10,20,30} is CONFOUNDED and must not be cited**: it predates the ~9 mm real point-cloud perception bias correction ([item1](item1_cube3_simreal_gap.md)), and was never repeated afterwards. Its flatness is therefore uninterpretable — the real slice it co-trained on was mis-registered. Re-run post-correction before any claim about how policy success scales with real-demo count, in sim or real. | user / cluster | **[NEED]** (prior result **invalid**) |
| B4 | **Tactile signal note.** Decided 2026-08-26: keep **small**. Lean on the [LightTact](https://arxiv.org/abs/2512.20591) citation (RSS 2026 — deformation-based sensing needs measurable indentation; recognised *within* the tactile community) as a one-clause parenthetical in ¶2. Our own GelSight Mini pilot observation goes in a footnote *only* if marked preliminary and explicitly not relied upon — an unquantified "the signal looked weak" is a liability out of proportion to its size. **Do not** expand into a claim that tactile fundamentally cannot work: LightTact is a *solution* paper, so that claim has a shelf life and invites "just use better hardware". | | **[NEED]** (small) |

## C. Verification — cheap, but each is a potential correction

| # | item | owner | status |
|---|---|---|---|
| C1 | **Verify the Curse of Precision citation** — exact claim, venue, and that our task is in the regime it describes (ties to A2). Currently taken on trust. | | **[NEED]** |
| C2 | **Novelty literature check** on FEM-stress-as-demonstration-supervision. `idea-evaluator` flagged this as unverified; no search has been run. Do before committing to the framing. | | **[NEED]** |
| C3 | **Is the stress-vs-geometric comparison affected by the scale bug?** DEVLOG conclusion 11: `eval_grasp_synth` planned on nominal-size meshes for scaled scenes (fixed 43b388a); absolute numbers before the fix are tainted, *paired comparisons survive*. Contribution 1 rests on this comparison — confirm it is the paired form, or re-run. | | **[NEED]** |
| C4 | **Decide the title.** Current pick: *Coverage Without Demonstrations…* (names the contribution, invokes no comparison we do not run). Avoid any "without touch" phrasing — comparative, and we have no tactile baseline. | user | **[NEED]** |

## D. Explicitly cut

| item | why |
|---|---|
| Lead/dwell absolute-action BC failure modes | User decision 2026-08-26 — not worth the space. Stays in `debug_partC_euler_action_anomaly.md` for engineering reference. |
| "Simulated ranking inverts on hardware" | Not validated; agent-added. At n≈20 the pure-sim and co-train CIs likely overlap, so the inversion may be noise. Soften to "sim score did not predict real ranking" if mentioned at all. |
| Tactile baseline experiment | No longer a gate. The argument is that tactile changes *observation*, not *supervision* — so a baseline is not needed to make the point. Also: no commercial force-gripper available (TF-Gripper is a research prototype). |
| Perception-offset methodology | Real result ([item1](item1_cube3_simreal_gap.md)) but a fourth contribution the paper cannot carry. Appendix or follow-up. |

---

## Related-work positioning (settled 2026-08-26)

Three arguments, layered by durability — **do not** reorder them:

1. **Primary (immune to hardware progress).** Tactile changes what the policy *observes*, not
   where *supervision* comes from. [TF-Gripper/RETAF](https://force-gripper.github.io/) builds
   a bespoke teleoperation rig to record human-applied grasp forces;
   [VTAM](https://arxiv.org/abs/2603.23481) finetunes on real tactile streams. Both are
   imitation learners fed by human demonstrations on real hardware, so they inherit the
   coverage cost in full — and compound it, since tactile is not reliably simulatable, closing
   the escape route to simulation. Sharp supporting detail: RETAF *decouples* force control
   from arm-pose prediction, so its base pose policy is still demo-learned — tactile rescues
   force regulation but not the where/how-to-approach sub-problem, which is where our measured
   failures live.
2. **Secondary (dated by LightTact).** Deformation-based commodity sensors are ill-conditioned
   in the gentle regime. One parenthetical, cited to LightTact. See B4.
3. **Scoping.** No comparison against force-controlled grippers; no commercial device exists.
   State plainly. **Do not** speculate that force sensing would also fail — a load cell
   measures force without needing indentation and may well work; we have no data.

**Must concede, in one sentence:** tactile provides *runtime* adaptation — a generalisation
mechanism, not merely more data. Frame runtime sensing and offline supervision as
complementary axes. Omitting this concession is the fastest way to lose a tactile reviewer.

**Reconciliation, not contradiction:** prior tactile work on fragile food uses potato chips
(stiff, brittle); our target is soft produce. If the compliance-partition story is ever
needed, it *explains* their results rather than disputing them — but it requires the
printed-vs-real control to assert, so it stays out unless B4 is expanded.
