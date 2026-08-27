# Related work map + fact check (2026-08-26)

Literature sweep for the ICRA draft. Every entry below was checked by web search this
session; arXiv IDs and venues are as returned by the source. **Nothing here has been read in
full** — abstracts and summaries only, so verify any claim you lean on hard before submission.

Companion to [PAPER_TODO.md](PAPER_TODO.md). Bib entries ready to paste are at the bottom.

---

## TL;DR — four findings that change the paper

| # | finding | consequence |
|---|---|---|
| **1** | **[Lin et al., ICLR 2025 Oral](https://arxiv.org/abs/2410.18647)** already established empirically that *diversity of environments/objects matters far more than demonstration count*, and that demos-per-object saturate past a threshold. | **Our "coverage over count" is no longer novel as a claim.** But it is now *citable* rather than something we must prove — which rescues the saturation observation we dropped. Contribution 2 must be reframed from "we show coverage matters" to "we supply a source of coverage for dimensions that cannot be varied on real objects at all." |
| **2** | **[MimicGen, CoRL 2023](https://arxiv.org/abs/2310.17596)** is the canonical automatic-demonstration-generation system (50K demos from ~200 human demos). | **Uncited direct neighbour of our contribution 3.** Must cite and differentiate: MimicGen *adapts human demos* by object-centric geometric transforms; we *plan from scratch* with a physics objective and need no human demos, and MimicGen has no notion of material stiffness. |
| **3** | **[DefGraspSim](https://arxiv.org/abs/2107.05778)** does FEM-based grasp evaluation for parallel-jaw grasps on 3D deformables, measuring stress and deformation across 34 objects / 1.1M measurements. | **Closest published work to our contribution 1, currently uncited.** Differentiate: it *evaluates* grasps by dynamic FEM in Isaac Gym (5–10 fps); we *plan* grasps with a static solve + stiffness-linearity, and generate demonstrations. |
| **4** | **[Sim-and-Real Co-Training](https://arxiv.org/abs/2503.24361)** is a recipe paper for exactly our training setup (sim data + small real slice). | Uncited; our co-training design should reference it rather than appear novel. |

---

## Axis 1 — Fragile / delicate food handling (application framing)

| work | id | note |
|---|---|---|
| Wang et al., *Towards Damage-Less Robotic Fragile Fruit Grasping* (systematic review) | J. Field Robotics 2025, doi:10.1002/rob.70021 | Establishes damage-free fragile grasping as an open problem. Good opening citation. |
| Wang et al., *Closed-Loop Force Control for Pneumatic-Driven Soft Gripper* | Soft Robotics 2024 | Representative of the compliant-hardware strand. |

Domain is active 2024–2026 (soft/pneumatic grippers, elephant-trunk enveloping designs,
sensorised compliant pads, blackberry-harvesting grippers). Framing the problem as open is
safe; claiming nobody works on it is not.

## Axis 2 — Tactile / force sensing for delicate manipulation

| work | id | note |
|---|---|---|
| Kang et al., *Learning Force-Regulated Manipulation… (TF-Gripper / RETAF)* | arXiv:2602.10013 | ~\$150 force-controlled jaw, 0.45–45 N. **Dedicated teleoperation device to record human-applied grasp forces** — this is the fact our supervision argument rests on, and it is theirs, stated. Reports chips / tofu / cherry tomatoes. Decouples force regulation from arm-pose prediction. |
| VTAM | arXiv:2603.23481 | Video-tactile-action model, modality-transfer finetuning on real tactile streams; 90% avg success; chip pick-and-place. |
| LightTact | RSS 2026, arXiv:2512.20591 | Deformation-**independent** optical contact sensing; states that deformation-based sensors need measurable indentation. Cite as the tactile community acknowledging the limitation. **It is a solution paper — do not build a "tactile fundamentally can't" argument on it.** |

## Axis 3 — Stress / damage-aware grasp quality for deformables

| work | id | note |
|---|---|---|
| Pan, Gao & Manocha, *Grasping Fragile Objects Using a Stress-Minimization Metric* | IROS 2020; ext. arXiv:1907.08749 | **General multi-contact** force-closure formulation. Our two-pad degeneracy is *outside its intended regime*, not a defect. |
| **DefGraspSim** | arXiv:2107.05778 | ⚠️ **MUST CITE.** Isaac Gym FEM, Franka parallel jaw, 34 objects, 6800 grasp evaluations, 1.1M measurements, 7 performance metrics incl. stress and deformation; reports sim↔real correspondence. |
| *A Novel Simulation-Based Quality Metric for Grasps on 3D Deformable Objects* | arXiv:2203.12420 | Same lineage as DefGraspSim; check for overlap with our score design. |
| *Minimal Work: A Grasp Quality Metric for Deformable Hollow Objects* | arXiv:1909.11226 | Alternative deformable grasp metric; one-line citation. |

## Axis 4 — Data scaling / coverage in imitation learning

| work | id | note |
|---|---|---|
| **Lin et al., *Data Scaling Laws in Imitation Learning for Robotic Manipulation*** | **ICLR 2025 Oral**, arXiv:2410.18647 | ⚠️ **MUST CITE — this is our thesis, already published.** >40k demos, >15k real rollouts. Power-law in #environments and #objects; **diversity ≫ demonstration count**; demos-per-object saturate past a threshold. |
| *The Curse of Precision* | arXiv:2607.23108 (Tsinghua / OpenMind, Jul 2026) | ✅ **VERIFIED** — `log N ∝ 1/(P−c)`, super-exponential as precision approaches a limit. **Caveats below.** |

### Fact check on *The Curse of Precision* (PAPER_TODO C1 — now resolved, with caveats)

**Confirmed:** the law form, the super-exponential claim, arXiv:2607.23108.

**Two caveats that affect how we may cite it:**

1. **Its tasks are rigid assembly** — peg insertion, cuboid stacking, roll ball. Not deformable,
   not food. Applying it to soft-produce grasping is an *extrapolation by analogy*. Either
   phrase it as such, or ground it with our own tolerance measurement (PAPER_TODO A2).
2. **The limit precision `c` is system-dependent, not a task constant** — the paper states it is
   "an emergent property of the entire agent system, including its sensors and expert policy",
   and that improving components (e.g. adding a wrist camera, using a better expert) measurably
   lowers it. This **helps us**: our stress-aware planner is precisely a better expert, so the
   law predicts a lower limit precision for our system. Worth one sentence — it converts the
   citation from a pure obstacle-argument into support for our method. But it also means we
   cannot present `c` as a fixed wall.

## Axis 5 — Automatic demonstration generation

| work | id | note |
|---|---|---|
| **MimicGen** (Mandlekar et al.) | **CoRL 2023**, arXiv:2310.17596 | ⚠️ **MUST CITE.** 50K demos / 18 tasks / broad initial-state distributions from ~200 human demos, by object-centric segmentation + context-aware trajectory transformation. Policies match or exceed those trained on equivalent additional *human* demos. |
| DexMimicGen / DynaMimicGen | — | Extensions (dexterous bimanual; dynamic scenes). One-line mention. |

**Differentiation we must state:** MimicGen *transforms existing human demonstrations* into new
contexts, so it still needs a human seed set and its adaptation is geometric. Our demonstrator
*plans from scratch* against a physics objective, needs no human demonstrations, and covers
**material stiffness**, a dimension a trajectory transform cannot express at all.

## Axis 6 — Sim-to-real transfer and sim/real co-training

| work | id | note |
|---|---|---|
| **Sim-and-Real Co-Training: A Simple Recipe for Vision-Based Robotic Manipulation** | arXiv:2503.24361 | ⚠️ Uncited; this is our training recipe. Cite it so the ~8% real slice reads as an informed choice. |
| Matas et al., *Sim-to-Real RL for Deformable Object Manipulation* | CoRL 2018 | Early deformable sim2real. |
| SimWeaver, *Zero-Shot RGB Sim-to-Real for Deformable Manipulation* | arXiv:2606.15338 | Recent; check before claiming novelty in deformable sim2real. |
| Our prior iteration | arXiv:2510.25405 | Rejected; unpublished. Per user decision, not dwelt on in the intro. |

---

## Recommended changes to the draft

1. **Reframe contribution 2.** Currently "coverage, not demonstration count, is the binding
   constraint" — Lin et al. own that. Rewrite as: *given* that diversity dominates count
   [Lin et al.], the open question is where diversity comes from when the relevant dimensions
   include material properties that cannot be varied on a physical specimen. That is the part
   nobody has taken.
2. **Add a "demonstration generation" sentence to ¶2** covering MimicGen, with the
   from-scratch-vs-transform and stiffness distinctions.
3. **Cite DefGraspSim where the FEM contact model is introduced**, with the plan-vs-evaluate and
   static-vs-dynamic distinctions.
4. **Cite Sim-and-Real Co-Training** at the co-training sentence.
5. **Soften the Curse of Precision usage** — it is a rigid-assembly result; and mention the
   system-dependent limit, which supports the "better expert" reading of our planner.
6. **Our 50-demo failure map keeps its role** — it is our own measurement, in our own setting,
   and remains the concrete evidence. Lin et al. makes it corroboration of a known law rather
   than a novel discovery, which is a safer position anyway.

## Still unsearched (worth doing before submission)

- Food-specific manipulation datasets/benchmarks (e.g. DexFruit, which appears in the draft's
  commented-out first line as `swann2025dexfruit` — unverified).
- Teacher–student distillation for grasping (DextrAH-G/RGB) — the user has these; not searched.
- 3D/point-cloud diffusion policies (DP3 lineage) — method-adjacent, needed in Method not Intro.
- Whether anyone has combined FEM-planned grasps with policy distillation, which would be the
  closest possible scoop. Not found in this sweep, but the sweep was not exhaustive.
