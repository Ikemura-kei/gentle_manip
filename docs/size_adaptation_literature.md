# Policy adaptation to object size — literature scan

**Status:** advisory scan, 2026-08-26. Written for the agent working items 17/18
(policy fails to adapt grasp width to object scale). Deep-dive subpage; linked from
`docs/DEVLOG.md`.

**Method + honesty note.** This is a `three-way-scan` (WHY/HOW/WHAT per paper), NOT a
systematic review — no PRISMA screening, so *absence of a paper here is weak evidence of
absence*. Every paper below was found by search and its abstract page fetched and read;
nothing is cited from model memory. Assistant knowledge cutoff is May 2026, so mid-2026
work is under-covered by construction. Anything I could not confirm is marked UNVERIFIED
and must not be cited without checking.

---

## 0. The problem being researched

From the item 17 width probe (`docs/DEVLOG.md`):

| | corr(min commanded width, obj_scale) | mean min commanded width, small→large scale |
|---|---|---|
| training data (demos) | **0.85** | 35.5 → 50.4 mm |
| afucm/state_400 | 0.27 | 20.3 → 23.4 mm |
| prmaw/state_200 | 0.44 | 16.7 → 21.6 mm |

The demonstrator moves ~15 mm of aperture across the size range; the policies move 3–5 mm.
The policy has largely **collapsed to a mean grasp width**.

**Direct literature answer: nobody has published this controlled study.** No paper was
found that trains a single parallel-jaw imitation/diffusion policy and measures whether it
adapts *gripper aperture* to object size. The width-vs-scale correlation used above appears
to be a novel diagnostic. Related work either achieves scale generalisation architecturally
without isolating the gripper DoF, or studies size adaptation on dexterous hands.

---

## 1. Verified sources

| # | Paper | Venue | Relevance |
|---|---|---|---|
| 1 | [EquiBot: SIM(3)-Equivariant Diffusion Policy](https://arxiv.org/abs/2407.01479) — Yang, Cao, Deng, Antonova, Song, Bohg | CoRL 2024 ([PMLR 270:1048–1068](https://proceedings.mlr.press/v270/yang25a.html)) | Scale generalisation **by construction**, diffusion + point clouds |
| 2 | [EquivAct: SIM(3)-Equivariant Visuomotor Policies](https://arxiv.org/abs/2310.16050) — Yang, Deng, Wu, Antonova, Guibas, Bohg | ICRA 2024 | Predecessor; 20 demos, objects "substantially differing in scale" |
| 3 | [Rapid Motor Adaptation for Robotic Manipulator Arms](https://arxiv.org/abs/2312.04670) — Liang, Ellis, Henriques | CVPR 2024 | **Infers object mass and SHAPE from action+proprio history**, ManiSkill2 |
| 4 | [RMA: Rapid Motor Adaptation for Legged Robots](https://arxiv.org/abs/2107.04034) — Kumar, Fu, Pathak, Malik | RSS 2021 | The original privileged-vector → adaptation-module recipe |
| 5 | [Contextual RL of Visuo-tactile Multi-fingered Grasping](https://arxiv.org/abs/1911.09233) — Kumar, Hermans, Fox, Birchfield, Tremblay | 2019 | **Bounding-cuboid dims as explicit context variable** |
| 6 | [D2PPO: Diffusion Policy Policy Optimization with Dispersive Loss](https://arxiv.org/abs/2508.02644) — Zou et al. | AAAI 2026, 40(22):18891–18899 | Representation collapse → can't detect subtle variations |
| 7 | [NOCS: Category-Level 6D Pose and Size Estimation](https://arxiv.org/abs/1901.02970) — Wang, Sridhar, Huang, Valentin, Song, Guibas | CVPR 2019 | Pose **and dimensions** of unseen instances from ONE RGB-D view |
| 8 | [ShapeGrasp: Visuo-Haptic Shape Completion and Grasping](https://arxiv.org/abs/2605.02347) — Rustler, Hoffmann | 2026 | Iteratively corrects completed shape; 84–91% success |
| 9 | [Data Scaling Laws in Imitation Learning](https://arxiv.org/abs/2410.18647) — Lin, Hu, Sheng, Wen, You, Gao | 2024 | Object diversity ≫ demo count; power law in #objects |
| 10 | [Understanding Multimodal Failure in Action-Chunking BC](https://arxiv.org/abs/2605.22493) — Mazza et al. | 2026 | Small Lipschitz transport can't cover well-separated modes |
| 11 | [Diffusion for Multi-Embodiment Grasping](https://arxiv.org/abs/2410.18835) — Freiberg, Qualmann, Vien, Neumann | 2024 | Parallel-jaw → humanoid; variable-sized object heaps |
| 12 | [GraspLDP: Generalizable Grasping via Latent Diffusion](https://arxiv.org/abs/2602.22862) — Xiang, Ma, Ma, Liu, Huang | CVPR 2026 | Grasp priors into IL diffusion policy |
| 13 | [DefGraspSim](https://arxiv.org/abs/2107.05778) | 2021 | Franka parallel-jaw grasping of deformables, E 1e4–1e9 Pa |

---

## 2. THE SINGLE-VIEW BBOX QUESTION (the thing to read)

Concern raised: bbox conditioning looks attractive, but our real rig is **one fixed
external camera** (`point_cloud_1cam`, L515). Does a size conditioner survive partial
observation? Three findings, in increasing order of usefulness.

### 2a. Metric size from one RGB-D view is a solved-ish problem — with caveats

**NOCS** (#7) estimates 6D pose *and dimensions* of **unseen instances within a category**
from a single RGB-D image, by regressing pixels into a shared normalised canonical space
and combining with depth. So "get a metric bbox from one view" is not blocked in principle.

Caveat: NOCS is category-level and needs category training data. Our objects (mushroom,
banana, strawberry, shrimp) are exactly the deformable/irregular cases its rigid-category
assumption fits worst. Treat as existence proof, not a drop-in.

### 2b. Full shape completion from one view is genuinely unreliable

The single-view completion literature is explicit that a single RGB-D view yields a partial
cloud with large occluded regions, that completion is **uncertain**, and that **grasp
planning fails when hallucinated geometry is wrong**. **ShapeGrasp** (#8) exists precisely
because of this: its contribution is *correcting* a completed shape using real grasp
feedback — tactile contacts plus the negative-space constraint of "the gripper body is
here, so the object is not."

⚠️ Note for our project: ShapeGrasp's tactile channel conflicts with our tactile-free
premise (`paper_writing.txt`). The **gripper-body free-space** channel does not — that is
pure proprioception and is available to us.

**Conclusion: do NOT build this on full amodal shape completion.** It imports exactly the
hallucination failure mode we cannot detect at deployment.

### 2c. Reframe — we do not need a bbox, we need ONE scalar, and it may be observable

Two observations that make this much easier than the general problem:

1. **Geometry.** For a parallel-jaw grasp the only quantity that matters is the object
   extent **along the gripper closing axis at the grasp pose**. With a world-fixed external
   camera and a top-down/side approach, that axis is typically *perpendicular to the camera
   ray*, i.e. it lies in the visible silhouette. The dimension that is occluded is the one
   **along** the ray — which is largely the one we do NOT need. The general amodal-bbox
   problem is much harder than our specific instance of it.

2. **A grasp is itself a measurement.** Once the fingers contact, gripper width and
   `contact_force` directly encode object size. Size is *proprioceptively observable* from
   contact onward — which is exactly when the firming/lift decision is made. Only the
   pre-contact estimate must come from vision.

### 2d. Therefore: prefer RMA-style latent estimation over an external estimator

**Liang, Ellis & Henriques (#3, CVPR 2024)** is the closest precedent: RMA applied to
manipulator arms, explicitly inferring **object mass and shape** from the agent's action and
proprioceptive history, over ManiSkill2 with YCB/EGAD objects and shape/density/friction
variation. The original **RMA** (#4) trains a base policy on a privileged extrinsics vector,
then trains an adaptation module by supervised learning to regress that vector from recent
history — no privileged access at deployment.

**We are already set up for this.** `PrivilegedConfig.object_dr_params` gives
`priv_object_dr_params = [scale, bend_deg]` — an episode-constant privileged size vector,
already consumed by the state teacher via `superset_rigid_full_state.yaml` /
`STATE_VIEW_FULL`. The missing half is the adaptation module that regresses it for the
point-cloud student. This is the RMA recipe almost exactly, and it sidesteps NOCS,
category assumptions, and shape-completion hallucination entirely.

---

## 3. Recommended experiment order (cheapest decisive test first)

**Step 0 — Is the information even present? (offline, hours, no new data.)**
Train a small MLP/PointNet head to regress `priv_object_dr_params[0]` (scale) from the
**already-recorded** point clouds in the existing demo datasets. Hold out geometries.
- Regresses well → the size signal is in the observation; conditioning is viable; go to Step 1.
- Regresses poorly → the student's obs genuinely lacks the information. Check whether
  `object_focus`/`outlier_removal` cropping is discarding the silhouette extent before
  reaching for shape completion.

This one probe separates "policy can't see size" from "policy sees size but ignores it",
which the current evidence does not distinguish. Do it before any architecture change.

**Step 1 — Close the teacher/student asymmetry (RMA-style).**
Add an adaptation module regressing the privileged size vector from observation history;
feed its output to the student. Precedent #3/#4. Also covers item 18's aux-head idea, since
the auxiliary target is the privileged scale rather than an invented one.

**Step 2 — If the policy sees size but still won't act on it, it is a representation
problem, not an information problem.** #6 (D2PPO, dispersive loss) targets exactly
"semantically similar observations map to indistinguishable features"; #10 gives the
transport-smoothness account of why the action range compresses. Both are loss-side changes,
not rewrites.

**Step 3 — Architectural equivariance is the expensive option.** #1/#2 give scale
generalisation by construction.
⚠️ **Check before adopting:** SIM(3)-equivariance scales *translations* with the scene. The
gripper DoF in our `ActionConfig` is a **separate normalised scalar** — and that scalar is
precisely what must scale with object size. Whether equivariance propagates into it is an
implementation detail of the action parameterisation, not a guarantee of the symmetry.
Read EquiBot's action head before committing.

**Not recommended:** more demos per scale. #9 finds object/environment diversity dominates
demonstration count once a per-object threshold is reached — consistent with our own finding
that the least-represented scale bin (1.2–1.3, 11.5%) performs *best*.

---

## 4. Unverified / do not cite without checking

- One search summary asserted that "final closed gripper width" is used as an auxiliary
  prediction target with a dedicated MSE output. **I could not trace this to a specific
  paper.** Item 18's aux grasp-width head is a reasonable idea with BC-Z-style auxiliary-loss
  precedent generally, but this specific claim is UNVERIFIED.
- **D2PPO (#6)**: the name implies it extends Ren et al.'s DPPO — our framework — but the
  arXiv abstract page does **not** state that relationship. Verify in the PDF before
  assuming drop-in compatibility.
- NOCS dimension-accuracy numbers and stated failure modes were not on the abstract page;
  read the CVPR PDF before relying on quantitative size accuracy.
- Gripper types for #3, #6, #9, #12 were not confirmed from abstract pages.

## 5. If a gap claim goes in the paper

The "no one has studied parallel-jaw aperture adaptation across object scale in IL" claim is
based on a 6-query scan, not systematic screening. Escalate to `lit-review` or
`systematic-review` mode for defensible coverage before asserting novelty in print.
