# Research story / introduction — three candidate framings (2026-08-25)

Grounded in `DEVLOG.md` (conclusions 1–13), the notes PDFs (`docs/pdf/`), and the report
subpages. Each version gives: **thesis → intro prose → evidence map → what it would take to
make it safe → reviewer attack surface**. Numbers are quoted only where we actually measured
them; anything unmeasured is marked **[GAP]** rather than smoothed over.

**House rules used throughout** (to avoid overclaiming):
- Sim success numbers are canonical-harness numbers (200 eps, fixed seeds); they are NOT
  comparable across machines/protocols, so the text never mixes cluster and local numbers.
- "Gentle" claims are **sim-measured stress** unless explicitly stated; we have no real
  bruising measurement yet.
- The real-robot evidence is one object category (mushroom), ~30-trial-scale sessions.

---

## Shared factual base (what we can assert today)

| Claim | Evidence |
|---|---|
| Scaling teleop-style demo COUNT does not fix precise grasping | sim BC at 50 / 300 / 1k demos all land ~0.70–0.75; failure mode is consistently *wrong grasp width*, not approach (notes 26.07.23; eval of gllzd ckpt 3000 confirms approaches correct) |
| An autonomous stress-aware demonstrator can be built and is reliable | FEM gentleness grasp synthesis (CMA-ES over 7-DOF TCP pose+width), 94% collection success on the adopted recipe; 3 collections reproduce within 0.6% (DEVLOG C3) |
| The FEM (stress-minimizing) demonstrator is gentler than a geometric SDF one | paired collections: lower mean/top10/peak von Mises at equal-or-better success (notes 26.08.19 figure) |
| Demonstrator *firmness* is a first-order lever on the learned policy | +5 mm squeeze: policy 0.62–0.70 → 0.76+, demonstrator 85% → 94% (C3) |
| Sim rankings do NOT predict real rankings across data regimes | real: co-train ~75% > real-only > pure-sim, while pure-sim was the sim-best (0.71) (C12) |
| A small real slice co-trained with broad sim coverage is the best real performer so far | afucm = realws sim + 55 real demos (~8% of data), ~75% real success (C12) |
| Sim2real gap has a measurable, correctable perception component | paired real/sim cube probe: ~9 mm systematic x-bias on the proprio-pinned arm segment; one rigid shift cuts full-cloud chamfer 14.8 → 8.4 mm (`item1_cube3_simreal_gap.md`) |
| Imitating absolute-pose actions has two non-obvious failure modes | commanded-lead requirement (C5) and trajectory **dwell** (C8): a dataset passing both older gates still trained to 0% success; fixing dwell took the same recipe 0 → 0.66 |
| Demonstrator kinematics can be measured against human teleop and matched | speed already matched (2.20 vs 2.22 mm/step); hover/close/rotation differed and were closed in v3.1–v3.3 (`item2_demo_kinematics.md`) |

**Not yet established (do not claim):** real-world gentleness/bruising measurement; any
generalist (multi-category) result; that gentleness survives transfer; that the pipeline
beats a tactile-equipped baseline.

---

## Version A — "Coverage, not more demonstrations"

**Thesis.** For *precise* gentle grasping, the bottleneck is not the number of
demonstrations but coverage of the physical variation that determines the correct action.
Real teleoperation cannot cheaply cover pose × size × shape × stiffness; an autonomous
stress-aware expert in simulation can. We show what it takes to make that transfer work,
and that a small real slice co-trained with broad sim coverage outperforms both pure-sim and
real-only policies on hardware.

### Intro draft

> Handling fragile food — a mushroom, a piece of tofu — is a *precise* manipulation problem
> disguised as a simple one. The gripper must close to a width that is large enough not to
> crush the item and small enough not to drop it, and that width is a function of the
> object's pose, size, shape, and stiffness, none of which are directly observable from a
> single depth view. Imitation learning is the natural tool, and the natural instinct when a
> policy fails is to collect more demonstrations.
>
> Our experiments do not support that instinct. Training the same point-cloud diffusion
> policy on 50, 300, and 1000 demonstrations of a soft-mushroom lift produced statistically
> indistinguishable success (≈0.70–0.75 in simulation), and the failure mode never changed:
> the approach was correct and the *grasp width* was wrong. Adding trajectories adds more
> examples of the situations the demonstrator already visits; it does not add the situations
> the policy has never seen. What is scarce is coverage, not data.
>
> This reframes the problem as one of where coverage can come from. Human teleoperation is
> expensive per trajectory and, worse, biased: an operator without force feedback judges a
> grasp from a camera feed, so the demonstrations concentrate on the poses and sizes the
> operator finds easy. Simulation can enumerate the variation directly — object pose,
> orientation, scale, shape, material — but only if an *autonomous* demonstrator exists that
> is both reliable and gentle across all of it.
>
> We build that demonstrator by turning gentleness into an optimization objective rather
> than a human skill. A static finite-element model of the object predicts the internal
> stress induced by a width-controlled parallel-jaw squeeze; a CMA-ES search over the 7-DOF
> grasp (pose + commanded width) minimizes that stress subject to holding the object against
> gravity. The result is a scripted expert that collects hundreds of gentle, successful
> demonstrations per session (94% success) and reproduces to within 0.6% across runs — and
> that is measurably gentler than the geometric SDF-based grasp heuristic it replaces.
>
> Making a policy trained on this data work on hardware turned out to require a specific
> set of choices, most of which are invisible in the loss curve: absolute actions supervised
> with *commanded* targets rather than achieved poses; demonstration trajectories without
> velocity-zero dwell; an arm-focused point cloud matching the real rig's processing; a
> workspace randomization box matching the physical table; and per-checkpoint evaluation,
> because every stable run peaks early. Two of these were discovered only through datasets
> that trained to perfect loss and evaluated at 0% success.
>
> On the real XArm7, the resulting picture inverts the simulation ranking. The policy with
> the best simulation score — trained purely in simulation — was the *worst* on hardware;
> a policy trained on 55 real demonstrations alone did better; and co-training the broad
> simulation set with those same 55 real demonstrations (8% of the training data) did best,
> at roughly 75% success. Coverage from simulation and calibration from a small real slice
> are complementary, and simulation score alone is not a model-selection signal across data
> regimes.

### Evidence map
Paragraph 2 → notes 26.07.23 + gllzd eval. P4 → C3 + FEM/SDF figure. P5 → C5, C8, C4, realws
box, C7. P6 → C12.

### To make it safe
- The demo-count saturation is a **sim** result on one task; either say so explicitly (as
  above) or add a real-demo-count curve. **The item-3 real ablation (N = 0/5/10/50) is
  exactly this experiment and is already staged** — it would upgrade P2 from "sim evidence"
  to "sim + real evidence" and is the single highest-value addition to this story.
- ~75% is one session on one object; report n and CI.

### Reviewer attack surface
"Your demo-count result just says your architecture/hyperparameters saturate." → Mitigate
with the failure-mode analysis (approach correct, width wrong) and the fact that the same
architecture reaches higher success when the *demonstrations* change (firmness lever, C3).

---

## Version B — "Gentleness without touch: moving the damage signal into simulation"

**Thesis.** Gentle manipulation is usually approached through tactile sensing, which gives a
direct contact signal at deployment. We ask whether the damage signal can instead live
entirely in *simulation* — as the internal stress field of a deformable model — and be
transferred into a policy that at deployment sees only a point cloud and proprioception. The
gentleness is embedded in the demonstrations by a stress-minimizing grasp planner rather
than learned from touch.

### Intro draft

> Robots that handle fragile objects are usually given a sense of touch. Tactile skins and
> optical tactile sensors provide a direct, local measurement of contact, and most systems
> for delicate manipulation are built around them. Touch, however, is not free: sensors add
> cost and fragility, occupy the fingertip geometry, drift and wear with use, and constrain
> the gripper design. It is worth asking how far one can get without them.
>
> The obstacle is that "gentle" is defined by a quantity the robot cannot see. What damages
> a mushroom is the internal stress induced by the squeeze, and no external camera measures
> it. Tactile sensing is popular precisely because it is the closest observable proxy. But
> there is a second place where that quantity is not merely observable but *exact*: inside a
> simulator. In a deformable-body model, the von Mises stress field is available at every
> particle, and a static finite-element model can predict the stress a commanded gripper
> width will induce before the grasp is executed.
>
> We use that access at data-generation time rather than at deployment time. A CMA-ES search
> over the 7-DOF grasp (pose and commanded width) minimizes FEM-predicted internal stress
> subject to holding the object against gravity, producing a scripted demonstrator whose
> gentleness is an optimization result rather than an operator's intuition. Compared with a
> geometric (SDF-based) grasp heuristic on the same object and the same execution pipeline,
> it induces measurably lower stress at equal or better lift success. A point-cloud
> diffusion policy is then trained on these demonstrations and deployed on a real XArm7 with
> a single external depth camera and no tactile sensing of any kind.
>
> This framing also makes a specific prediction that we can test: because the demonstrator's
> gentleness lives in *which grasp it chooses*, the properties of the demonstrations —
> not just their quantity — should dominate downstream performance. That is what we observe.
> Increasing demonstrations from 50 to 1000 leaves success at ≈0.70–0.75 with an unchanged
> failure mode (incorrect grasp width), while changing a single demonstrator property — how
> firmly the scripted expert squeezes — moves the trained policy from 0.62–0.70 to above
> 0.76. Demonstration *quality*, defined by the planner, is the lever.
>
> Transferring the result required treating the sim-to-real gap as measurable rather than
> nominal. Replaying real teleoperation actions in simulation with the object placed at the
> recorded fingertip position gives step-aligned real/sim observation pairs; on that data the
> real point cloud is displaced from its simulated twin by a systematic ~9 mm along the
> camera ray, and a single rigid correction removes 43% of the total cloud discrepancy. On
> hardware, the best configuration — broad simulation coverage co-trained with 55 real
> demonstrations — reaches ~75% lift success without touch sensing.
>
> We do not claim that touch is unnecessary in general, nor that our policy is as gentle in
> reality as it is in simulation: our gentleness metrics are simulation-measured, and a
> real bruising study remains future work **[GAP]**. What we show is that the stress signal
> is usable where it is exact, that it can be compiled into demonstrations, and that a
> tactile-free policy trained this way transfers.

### Evidence map
P3 → grasp synthesis (`qsm_metric.md`, `grasp_synthesis_*`), FEM/SDF figure. P4 → notes
26.07.23 + C3. P5 → `item1_cube3_simreal_gap.md`, C12. P6 → explicit limitation.

### To make it safe
- **The honest weak point is that we never demonstrate the tactile-free claim *against* a
  tactile baseline.** Either (a) soften to "we study how far a tactile-free pipeline gets",
  or (b) run the real tactile-DP3 baseline the rig already supports (2× GelSight Mini) on
  the same task/protocol. (b) is a genuinely strong paper move and the hardware exists.
- Needs at least a *proxy* real gentleness measure (e.g. post-lift deformation/bruise
  scoring, or contact-force logging) to keep the word "gentle" in the real half of the story.
- The two reviewer objections you already wrote down in the notes should be answered in
  the paper: *if tactile is unavailable, is a depth camera also unavailable?* and *why not
  simulate tactile instead of stress?* — the second has a good answer (stress is the causal
  quantity, tactile is a proxy for it); the first needs the cost/wear/geometry argument.

### Reviewer attack surface
"You never compare to tactile." → strongest fix is (b). Second-strongest: reposition the
contribution as *stress-as-supervision*, where tactile is context, not the baseline.

---

## Version C — "An empirical account of transferring a precise, gentle skill"

**Thesis.** A systems-and-findings paper. We build an end-to-end sim2real stack for gentle
food manipulation and report a sequence of controlled experiments about what actually
determines transfer — several of which are negative results that cost weeks and are invisible
in training loss. The deliverable is the recipe plus the evidence for each component.

### Intro draft

> Sim-to-real pipelines are usually reported as a working system: an environment, a policy
> class, a randomization scheme, and a success rate. What such reports rarely convey is how
> narrow the path is — how many individually plausible choices silently produce a policy that
> trains perfectly and does nothing useful. This paper is an attempt to report that path
> honestly for one concrete problem: lifting a soft, fragile food item (a mushroom) with a
> parallel-jaw gripper, gently, from a single external depth camera.
>
> The task is deceptively hard because it is precise: success depends on a commanded gripper
> width that must be inferred from an occluded point cloud, and the margin between dropping
> the object and crushing it is a few millimetres. We first establish that the obvious lever
> does not work — scaling demonstrations from 50 to 1000 leaves simulated success at
> ≈0.70–0.75 with an unchanged failure mode — and then work through the components that do.
>
> On the *data* side, demonstrations come from an autonomous demonstrator rather than
> teleoperation: a finite-element gentleness metric scored by CMA-ES over the 7-DOF grasp,
> which collects at 94% success, reproduces within 0.6% across runs, and is measurably
> gentler than a geometric baseline. Demonstrator properties turn out to matter more than
> demonstration count: a 5 mm change in scripted squeeze firmness moves trained-policy
> success from 0.62–0.70 to above 0.76.
>
> On the *supervision* side, we document two failure modes of behaviour-cloned absolute-pose
> actions that both produce perfect training loss and 0% success. Supervising with achieved
> next poses creates a closed-loop fixed point (the policy commands where it already is);
> and, independently, demonstrations whose velocity goes to zero mid-trajectory — the natural
> consequence of min-jerk time scaling — put 32% of consecutive actions within a
> normalization-unit of each other and stall the policy in the same way, at 0% success, even
> when the first problem is fixed. Reverting to linear time scaling took the identical recipe
> from 0 to 0.66. Both now have automated dataset gates.
>
> On the *observation* side, we measure the sim-to-real gap instead of assuming it. Replaying
> recorded real actions in simulation yields step-aligned observation pairs whose proprioceptive
> channels agree to ~2 mm, so the residual point-cloud discrepancy is attributable: the real
> cloud sits ~9 mm from its simulated twin along the camera ray, and correcting that single
> rigid offset removes 43% of the total discrepancy.
>
> Finally, on hardware, the ranking inverts: the simulation-best policy (pure simulation,
> 0.71) performs worst in reality, a real-only policy trained on 55 demonstrations does
> better, and co-training simulation coverage with those 55 demonstrations does best at ~75%.
> Simulation score is a within-family model-selection signal, not a cross-regime one — a
> caution that applies to any pipeline that tunes on simulated evaluation.
>
> We report the resulting recipe, the measurements behind each choice, and the failures we
> could not yet close: gentleness is verified in simulation but not on real produce, and the
> system remains a single-category specialist **[GAP]**.

### Evidence map
P2 → notes 26.07.23. P3 → C3, FEM/SDF. P4 → C5, C8 (numbers exact). P5 →
`item1_cube3_simreal_gap.md`. P6 → C12.

### To make it safe
Nothing structural — this version claims exactly what we measured. Its risk is *venue fit*,
not honesty: it reads as an experience report unless the two negative results (fixed-point
stall, dwell) are elevated to the headline contribution, since both are general to BC with
absolute action chunks and neither is, to our knowledge, documented elsewhere.

### Reviewer attack surface
"Incremental / engineering." → Elevate dwell + commanded-lead into a named, generalizable
finding with a minimal reproduction (they are the most transferable things we know).

---

## Recommendation

**A and B are the two publishable framings; C is the honest backbone that should supply the
experiments section of either.**

- Pick **B** if you are willing to run the tactile baseline the rig already supports. It is
  the most distinctive claim (stress-as-supervision, tactile-free deployment), it matches
  the original motivation in your notes, and the mushroom/food setting is a natural fit.
- Pick **A** if you want the safest defensible story from what exists *today*: it needs no
  new capability, and the already-running real-data-amount ablation (N = 0/5/10/50) directly
  strengthens its central claim.
- In either case take from **C**: the dwell and commanded-lead findings, the measured
  perception bias, and the sim-ranking-inverts-in-real caution. These are the parts of the
  work that are both true and not obvious.

**One framing to avoid:** "we achieve gentle manipulation of real food." We have gentle
grasp *selection* validated in simulation and lift success validated in reality; the link
between the two on real produce is exactly the missing measurement.
