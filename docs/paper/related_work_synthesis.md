# Related work — grasp synthesis (parallel-jaw scope) (2026-08-31)

Related-work map for the grasp-synthesis component of the paper. **Scope filter: parallel-jaw
(two-finger) grippers only** — multi-fingered/dexterous-hand work (GraspIt!/eigengrasps,
classic multi-finger grasp-force optimization, anthropomorphic tactile hands) is excluded, to
be cited at most in passing. Companion docs: `synthesis_experiments.md` (E1/B2 evidence),
`method_v4.md` (our method). Verify quotes against the PDFs before camera-ready; entries
marked (*) are known from abstracts only.

## 1. Rigid-body parallel-jaw grasp planning: pose is the plan, closing is assumed

The defining property of this family: a grasp IS a hand pose (± an aperture pre-shape); what
happens after the fingers touch is delegated to rigid contact and the gripper's force/stall
limit. None of these methods scores the closing depth against the object's mechanical
response.

- **GPD** — ten Pas, Gualtieri, Saenko, Platt, *Grasp Pose Detection in Point Clouds*, IJRR
  2017. Candidate sweep on a point cloud with a fixed fully-open hand model + CNN scoring.
  The formal problem statement (Problem 1) asks for "6-DOF hand poses … such that a force
  closure grasp will be formed … **when the hand closes**" — closing is an assumed event,
  not a decision. Our E1 external baseline; its reported `grasp width` is a diagnostic (the
  object slice inside the closing region), not a planned quantity.
- **Dex-Net 2.0** — Mahler et al., RSS 2017. Antipodal sampling + GQ-CNN robustness scoring
  on depth images; pose-only output.
- **Rectangle-representation detectors** — Jiang, Moseson, Saxena, ICRA 2011; Lenz, Lee,
  Saxena, IJRR 2015. The oriented rectangle's short side is a gripper opening — width enters
  the *representation* as local object extent, learned for localization, not selected
  against object response.
- **GG-CNN** — Morrison, Corke, Leitner, RSS 2018. Per-pixel quality, angle, **and width**;
  width supervised from object extent, used as pre-shape.
- **GraspNet-1Billion / AnyGrasp** — Fang et al., CVPR 2020 / T-RO 2023. 7-DOF grasp =
  6-DOF pose + width, learned end-to-end at scale; width again a clearance/pre-shape channel.
- **Contact-GraspNet** — Sundermeyer et al., ICRA 2021. Contact-point parameterization with
  a predicted width bin; same semantics.
- **VGN** — Breyer et al., CoRL 2021; **GIGA** — Jiang et al., CoRL 2021. Volumetric grasp
  detectors regressing pose + width per voxel.

**Positioning sentence:** in all of the above, a predicted "width" is a pre-shape/clearance
parameter; execution closes to a force or stall limit, which rigid contact makes safe. On
deformables the unmodeled closing decision becomes the dominant success/damage factor — our
E1 grid measures exactly this (GPD 1.5–25 % success across six soft objects; +2.4× avg with
our closure bolted on).

## 2. Grip-force control on parallel jaws: the online, force-controlled sibling

These DO decide "how hard" — but online, from tactile/slip feedback, decoupled from pose
planning, and as a force setpoint (unavailable on many position-controlled grippers,
including ours):

- **Tremblay & Cutkosky**, ICRA 1993 — incipient-slip detection to set grip force.
- **Romano et al.**, T-RO 2011 — human-inspired grip-force modulation for pick-and-place on
  a parallel-gripper PR2 (close-lift-hold with slip-triggered force increase).
- **Soft tactile parallel grippers for fragile produce** — e.g. slip-detecting bionic
  gripper for damage-free fruit/vegetable grasping, Comput. Electron. Agric. 2024 (*).
  Recipe: close until contact, increase force on slip.

**Positioning:** complementary to synthesis-time closure selection — a model-based width
plan (ours) and reactive slip control address the same variable at different timescales; on
position-controlled hardware only the former is directly commandable.

## 3. Deformable-object grasping with two jaws: squeeze as a first-class variable

The direct precedents. Split by what the squeeze decision is made WITH:

**Analytic / model-based (planar or simplified):**
- **Wakamatsu, Hirai, Iwata**, ICRA 1996 — *bounded force closure*: force closure re-derived
  under bounded contact force for deformables; earliest formalization that closing effort is
  part of the grasp.
- **Gopalakrishnan & Goldberg**, IJRR 2005 — *deform closure* in deformation space
  (D-space): computes the **minimal squeeze** guaranteeing a two-jaw grasp of a planar
  deformable part. The cleanest precedent for "width as an optimization target"; planar,
  linear-elastic, two point jaws.
- **Lin, Guo, Jia**, IROS 2013; **Jia, Guo, Lin**, IJRR 2014 — squeeze grasping of planar
  deformables under **specified finger displacements** (displacement-controlled, like our
  width command), FEM equilibrium + stick/slip transitions + energy-based optimality;
  extended toward 3D two-finger pickup in follow-ups.

**FEM-simulation-based (3D, Franka parallel jaw):**
- **DefGraspSim** — Huang et al., RA-L 2022. Exhaustive FEM *evaluation* of antipodal
  candidates on 3D deformables under controlled squeeze-force sweeps; reports stress,
  deformation, strain energy, contact metrics (34 objects, 6.8 k evals). An evaluator
  (minutes per grasp), not an in-loop planner; squeeze is force-parameterized.
- **DefGraspNets** — Huang et al., ICRA 2023. GNN surrogate of DefGraspSim's FEM (~1500×
  faster) predicting stress/deformation fields; gradient-based pose refinement on the
  predicted fields. The learned counterpart of our analytic surrogate; width still not the
  commanded variable.
- **GRIP dataset** — 2025 (*). Large-scale IPC-simulated deformable-rigid coupled grasping
  with per-grasp stress distributions, for training stress predictors.

**Learned, deformation-aware synthesis:**
- **Le et al.**, RA-L 2022 (*Deformation-Aware Data-Driven Grasp Synthesis*) — grasp
  synthesis conditioned on stiffness, trading deformation against quality (parallel jaw).

**Quality metrics for deformables (two-jaw):**
- **Pan, Gao, Manocha**, 2020 — stress-minimization metric Q_SM: contact forces optimized
  under an internal-stress bound (BEM in the paper; our `smgrasp` reimplements with FEM).
  Force-controlled formulation; our repo's measured finding: for two pads on curved organic
  surfaces the point-contact force-closure hull is degenerate (Q_SM ≈ 0), which motivated
  our pivot to the task-wrench (hold-gravity) + width-controlled model.
- **Xu, Danielczuk, Ichnowski, Goldberg**, ICRA 2020 — *Minimal Work* metric for deformable
  hollow objects: quality = deformation work + wrench resistance; squeeze effort enters the
  objective directly.

## 4. Where v4.1 sits (the claim this map supports)

1. Rigid parallel-jaw planners (§1) never decide closing depth — rigid contact resolves it
   for free; on deformables that missing decision dominates (E1).
2. Grip-force control (§2) decides it online via tactile feedback and force control —
   unavailable at plan time and on position-controlled grippers.
3. Deformable-grasping precedents (§3) make squeeze first-class but are planar analyses
   (Gopalakrishnan; Jia), offline FEM evaluators (DefGraspSim), learned surrogates needing
   large training sets (DefGraspNets, Le, GRIP), or force-parameterized metrics (Pan, Xu,
   DefGraspSim).
4. **Ours:** an analytic, material-calibrated linear-FEM surrogate that selects the
   **commanded width** (position-control semantics) jointly with an executable 7-DOF TCP
   pose, fast enough for a CMA-ES synthesis loop (factorize-once + per-candidate Schur
   solve), validated end-to-end in MPM sim (E1: only method holding success AND sub-yield
   stress simultaneously across six objects; width-swap shows neither pose quality nor
   bolt-on width selection alone suffices).
