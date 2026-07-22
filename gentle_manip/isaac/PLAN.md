# IsaacSim / IsaacLab adoption — de-risking plan (Path A)

**Goal.** Decide whether to adopt IsaacSim/IsaacLab (FEM deformables) as a **third backend** behind
our existing sim2real interface (`PolicyEnv` + `RawObs` + `PerceptionPipeline`/`ActionPipeline`),
alongside Genesis (`SimBackend`) and the real robot (`RealBackend`). See the architecture decision in
the chat: **Path A = IsaacLab-as-a-library → `IsaacBackend` → `RawObs`**, NOT IsaacLab's
`ManagerBasedRLEnv` as our env. RL-ecosystem access comes later via a **`DirectRLEnv`** extension that
*wraps* the same components (single obs/action path — see "Later" below).

**Method.** Each phase below is a **standalone spike** in `gentle_manip/isaac/` (kept local/experimental,
IsaacLab submodule stays pristine, capture→host-viz pattern from `play_deformables.py`). We clear ALL
technical obstacles as throwaway spikes FIRST; only when every phase is green do we build the real
`IsaacBackend` under the shared interface. Order is chosen so blockers surface early and each phase
feeds the next. Phases 5 and 7 are the make-or-break gates.

---

## Phase 0 — Environment & deformable basics ✅ DONE
Container up (Docker + nvidia runtime, headless + `--enable_cameras`), `play_deformables.py` validated:
- Physics ~220 steps/s (4 cubes, 5000 tets), **stable** (no CFL blow-up, unlike Genesis MPM at low substeps).
- **Per-element von-Mises stress** exposed (`sim_element_stress_w`, 3×3 Cauchy) + responds to a squeeze load.
- Point-cloud render path (camera depth, ~4× render overhead) + capture→host-viz (`viz_capture.py`).

## Phase 1 — Arm spawn + basic joint control
**Risk:** URDF→USD conversion; articulation/actuator config; holding home pose without drift.
- Convert `assets/xarm/xarm7_with_gripper.urdf` → USD (`scripts/tools/convert_urdf.py`); resolve the
  `package://xarm_description/...` mesh paths (meshes are in `assets/xarm/xarm_description/meshes/`).
- `ArticulationCfg`: init = `DEFAULT_JOINT_ANGLES`, actuators from `KP`/`KV` (implicit PD), fixed base.
- Read joint pos/vel + EE pose; command joint-position targets and confirm tracking.
- **Success:** arm stands at home, holds without drift, tracks joint targets; state readable in the
  fields `RawObs` needs (`ee_pos`, `ee_quat`, `joint_pos`, `joint_vel`).

## Phase 2 — Cartesian / EE control (the mode we actually use)
**Risk:** real robot uses **delta-pose cartesian servo**; need an Isaac equivalent for `ActionPipeline` parity.
- Use IsaacLab `DifferentialIKController` (or equivalent) to map EE delta-pose → joint targets.
- Verify EE tracks a commanded cartesian trajectory; reconcile the EE frame with `EE_LINK` +
  `TCP_API_TO_TCP_OURS_OFFSET` (our TCP is 0.13 m off the API TCP).
- **Success:** commanded delta-EE-pose moves the EE correctly; read-back EE pose matches within tol;
  TCP convention reconciled with the real side.

## Phase 3 — Gripper (parallel-jaw, mimic joints)
**Risk:** PhysX does NOT honor URDF **mimic joints** — the 1 `drive_joint` + 5 mimics won't couple.
- Handle the linkage: drive `drive_joint` and mirror the 5 mimic targets each step (or a tendon/gear).
- Map gripper action → width using our `GRIPPER_JOINT_OPEN/CLOSED` and re-characterize the
  joint↔finger-separation↔width calibration (Isaac's version of `GRIPPER_CALIB_*`).
- **Success:** gripper opens/closes to a commanded width, fingers symmetric, width↔joint mapping known.

## Phase 4 — Mesh-based deformable (the mushroom)
**Risk:** our objects are scanned meshes (`assets/objects/mushroom.obj`, in meters); Isaac must
tet-mesh an arbitrary mesh well, honor our material, and stay stable.
- Spawn a deformable from `mushroom.obj`; inspect tet count/quality vs the default cube.
- Material = our tuned mushroom preset (E=3e5, ν=0.35). Verify stable at rest and under a load.
- Sanity-compare stress magnitudes to the Genesis mushroom (order-of-magnitude, not exact).
- **Success:** mushroom imports, tet-meshes, settles stably, deforms under load, stress readable.

## Phase 5 — Grasp a soft object (integration gate) ⚠️ make-or-break
**Risk:** the full loop — rigid articulation fingers in contact with an FEM soft body. Contact
stability (interpenetration / explosion), and whether the gripper actually holds the object.
- Scripted approach → grasp → lift the mushroom (mirror our Genesis scripted expert).
- Check contact stability (no blow-up, no tunneling), gripper holds through the lift.
- Read stress during grasp; confirm gentle vs firm grasp changes stress sensibly.
- **Success:** scripted grasp lifts the mushroom, contact stable, stress responds to grasp force.
  *If this fails or is unstable, Isaac is not viable for this task — stop here.*

## Phase 6 — Stability & performance characterization
**Risk:** stays stable across our DR ranges? fast enough with arm + soft + render + PARALLEL envs?
- Sweep E / size / grasp force → a stability envelope (analog of our Genesis grid/substep sweep).
- Parallel envs via `InteractiveScene` cloning: steps/s vs `num_envs` with arm+soft+camera (the number
  that actually matters for training — the current 4-cubes-in-one-scene is NOT this).
- Determinism/repeatability at fixed seed (FEM-on-GPU may be non-bit-deterministic like MPM).
- **Success:** a stability map + a throughput-vs-`num_envs` curve → verdict on training viability.

## Phase 7 — Genesis→Isaac policy transfer smoke (sim2sim gate) ⚠️ validates the thesis
**Risk:** the end goal — a Genesis-trained policy runs in Isaac. Needs **obs parity** (Isaac camera
depth → the SAME `PerceptionPipeline` math: crop/1024-FPS/canonical-quat) and **action parity**.
- Build the point cloud in Isaac through our shared perception math; match `point_cloud_1cam_*` obs cfg.
- Feed a Genesis-trained DPPO checkpoint (`xxiaw`/`gllzd`) obs FROM Isaac; run open- then closed-loop.
- Compare to the Genesis rollout — does it approach/grasp? Measure the qualitative sim2sim gap.
- **Success:** the Genesis policy emits sensible actions in Isaac and a sim2sim gap is quantified. This
  validates "shared obs → backend swap" at the sim level BEFORE building the full backend.

---

## Then (only if 1–7 are green): the real integration
Build `IsaacBackend` under the shared interface (Path A):
- Produces batched `RawObs` (leading `num_envs` dim — already our convention) and consumes
  `ActionPipeline` output; plug into the EXISTING `PolicyEnv`; expose via an `isaac_sim_server.py`
  over the `envs/rpc.py` socket (same seam as `serl_sim_server.py`, so DPPO/SERL connect unchanged).
- **Factor for reuse (so the later extension is cheap):** keep the Isaac **scene/asset cfgs**
  (arm/deformable/camera + scene builder) and the **reward/success** (already pure functions of state)
  in separate modules — NOT buried in `IsaacBackend`.

## Later (RL ecosystem, without breaking parity)
A **`DirectRLEnv`** extension (NOT `ManagerBasedRLEnv`) that reuses the scene/reward modules and routes
`_get_observations`/`_apply_action` through the SAME `IsaacBackend` + `PerceptionPipeline`/`ActionPipeline`.
IsaacLab's RL libs (rsl_rl/skrl) wrap that. One obs/action path → deployable policies stay sim2real-faithful.
Label training heads clearly: "deployable" (through the pipelines) vs "sim-only-fast" (raw managers).

## Cross-cutting unknowns to keep on the radar
- **Mimic joints** (Phase 3) — no native PhysX support; the biggest arm-side unknown.
- **Rigid↔FEM contact stability** (Phase 5) — the biggest task-side unknown.
- **Tet-meshing quality** of scanned meshes (Phase 4).
- **`package://` mesh resolution** for the converter (Phase 1).
- **FEM GPU determinism** (Phase 6) — affects eval reproducibility.
- **Python/version seam** — Isaac Kit python ≠ our envs; the rpc socket bridge is the intended glue.
- **Image/disk footprint** — Isaac Sim image + caches are large on a near-full disk.
- **Perception/action parity tax** — re-doing the quat-sign/outlier/crop parity work for Isaac's camera
  (the same tax Genesis needed); unavoidable, lives in `IsaacBackend`, doesn't ripple outward.
