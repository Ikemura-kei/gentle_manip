# IsaacLab / Isaac Sim exploration (deformable bodies)

Exploratory comparison of PhysX-FEM deformable sim (Isaac Sim) vs our Genesis MPM, for the
gentle-manip mushroom task. **All code here lives OUTSIDE `third_party/IsaacLab`** so the
submodule stays pristine for upstream updates — we mount this dir into the container instead of
adding scripts to the submodule.

## Central questions
1. How fast is the sim? (physics-only FPS)
2. How fast is point-cloud render? (physics+render+camera FPS, point count)
3. Does it give stress? — **yes**: `DeformableObject.data.sim_element_stress_w` is the full
   `(num_instances, num_elements, 3, 3)` Cauchy stress tensor per FEM element (`play_deformables.py`
   reduces it to von Mises). Richer than Genesis MPM's per-particle von Mises.
4. How big is the sim2real gap? (later)

## One-time: build/pull the Isaac Sim image
```bash
cd third_party/IsaacLab
./docker/container.py start base      # first time pulls the Isaac Sim base image (~20-30 GB)
```
Prereqs (already satisfied on this box): Docker + the `nvidia` container runtime, a host GPU
driver meeting Isaac Sim's minimum (>= 535). **Disk**: the image + kit/shader caches are large;
keep an eye on `df -h` (we run near-full).

## Run the deformable benchmark (mounting THIS dir, no submodule edits)
```bash
cd third_party/IsaacLab
GM=/home/kei/kei/gentle_manip/gentle_manip/isaac/docker-compose.gm.yaml
./docker/container.py start base --files "$GM"     # adds /workspace/gm_isaac mount
./docker/container.py enter base --files "$GM"
# --- inside the container ---
./isaaclab.sh -p /workspace/gm_isaac/play_deformables.py --headless --enable_cameras \
    --num-cubes 4 --youngs 1e5 --num-steps 300
```
`--enable_cameras` is REQUIRED for the point-cloud phase (Isaac renders sensors offscreen even
headless). Drop `--enable_cameras` and add `--no-camera` for a pure physics-throughput number.

Stop when done: `./docker/container.py stop base`

## Phase 1 — arm bring-up (spawn + basic control)
One-time URDF->USD conversion, then spawn. The compose override now also mounts the XArm assets at
`/workspace/gm_assets/xarm`. Start/enter the container with `--files "$GM"` (as above), then IN the
container:
```bash
# 1) rewrite BOTH package:// prefixes (xarm_description = arm, xarm_gripper = gripper) to the
#    in-container mesh dirs — writes a COPY into the writable mount, never edits the original asset:
mkdir -p /workspace/gm_isaac/assets
sed -e 's#package://xarm_description/#/workspace/gm_assets/xarm/xarm_description/#g' \
    -e 's#package://xarm_gripper/#/workspace/gm_assets/xarm/xarm_gripper/#g' \
  /workspace/gm_assets/xarm/xarm7_with_gripper.urdf > /workspace/gm_isaac/assets/xarm7_gm.urdf

# 2) convert URDF -> USD (fixed base = table-mounted arm):
./isaaclab.sh -p scripts/tools/convert_urdf.py \
  /workspace/gm_isaac/assets/xarm7_gm.urdf /workspace/gm_isaac/assets/xarm7.usd --fix-base
#    (if it errors on the gripper's mimic joints, add the converter's mimic flag — see
#     `./isaaclab.sh -p scripts/tools/convert_urdf.py -h`; Phase 3 handles the gripper properly)

# 3) spawn + hold home (GUI — OMIT --headless to see the viewport):
./isaaclab.sh -p /workspace/gm_isaac/spawn_arm.py                  # holds home, prints EE/joint state
./isaaclab.sh -p /workspace/gm_isaac/spawn_arm.py --joint-test     # sweeps a joint, reports tracking
```
Success: arm stands at home, holds without drift, joint targets track (err ~0), EE pose reads back.

## Phase 2 — cartesian / EE control (delta-pose via DiffIK)
Needs the converted USD from Phase 1. Uses IsaacLab's `DifferentialIKController` in RELATIVE mode:
the command is a 6-DOF `(dx,dy,dz,droll,dpitch,dyaw)` delta — the SAME thing our `ActionPipeline`
emits (dims 0-5; dim 6 = gripper). Runs a scripted ±x/±y/±z/±yaw pattern so the EE visibly moves and
returns, printing both the controlled frame (`xarm_gripper_base_link`) and "our TCP" (fingertip =
base + `SIM_TCP_OFFSET` [0,0,0.171]).
```bash
./isaaclab.sh -p /workspace/gm_isaac/control_ee.py                 # GUI, scripted delta motion
./isaaclab.sh -p /workspace/gm_isaac/control_ee.py --headless --steps 600
```
Success: EE follows each commanded delta and returns to start after the ± pattern; TCP(world) matches
the fingertip you expect at home (reconcile against real EE_BOUNDS / DEFAULT_EE_POSE). If IK is
jittery, try `--ik-method pinv` or bump `--arm-stiffness`.

## Phase 3 — gripper (parallel-jaw drive + width calibration)
PhysX doesn't honor URDF mimic joints, so we command all 6 gripper joints (drive_joint + 5 mimics,
all multiplier=1) to the SAME angle — mirroring Genesis `XArm7Sim.apply_target`. `--sweep` ramps
open→close and prints Isaac's own joint-angle→finger-separation calibration (its `GRIPPER_CALIB`).
```bash
./isaaclab.sh -p /workspace/gm_isaac/grip_test.py --sweep         # open->close, print calibration
./isaaclab.sh -p /workspace/gm_isaac/grip_test.py --angle 0.5     # hold a fixed joint angle
```
Success: `actual` tracks `cmd` with `spread ~0` (all 6 move together), `finger_sep` decreases
smoothly, both fingers move symmetrically. Compare the printed SEP range to Genesis
(0.1409 open .. 0.0536 closed). A functional soft grasp is Phase 5.

## Phase 4 — scanned mushroom as an FEM deformable
The compose override now mounts ALL of `gentle_manip/assets` (was just `xarm/`), so recreate the
container once to pick it up: `./docker/container.py start base --files "$GM"` then re-enter (and
re-`pip install -e source/isaaclab` — fresh overlay). No convert step — the spike loads mushroom.obj
directly, welds it (trimesh), builds the deformable mesh prim, and binds the mushroom material:
```bash
./isaaclab.sh -p /workspace/gm_isaac/deform_mushroom.py
```
(`convert_mesh.py -> UsdFileCfg` was tried but its rigid USD structure won't take the deformable
schema; the direct trimesh route mirrors IsaacLab's working MeshCfg deformable spawner.)
Success: `tet-cook OK` prints elements/nodes > 0, `root_z` settles (bounded, no NaN/explosion, no
sink through floor), von-Mises finite at the REAL softness (E=3e5). If it's unstable, drop `--youngs`
or raise solver iterations; if the cook fails, the mesh may need decimation before welding.

## Files
- `play_deformables.py` — deformable FEM benchmark: physics-only vs +render+pointcloud FPS,
  per-element von-Mises stress; `--dump` captures rgb+geometry+stress, `--squeeze` compresses.
- `viz_capture.py` — host-side (envs/deploy): capture.npz -> mesh video + stress-node video + PNG.
- `spawn_arm.py` — Phase 1: spawn XArm7, hold home, read joint/EE state, `--joint-test` tracking.
- `control_ee.py` — Phase 2: cartesian delta-pose control via DiffIK (relative mode = ActionPipeline
  6-DOF command); prints controlled frame + TCP for convention reconciliation.
- `grip_test.py` — Phase 3: drive the parallel-jaw (all 6 joints equal), `--sweep` calibrates
  joint-angle → finger-separation.
- `deform_mushroom.py` — Phase 4: scanned mushroom.obj → FEM deformable (convert_mesh → UsdFileCfg
  deformable); checks tet-cook, stability, stress.
- `docker-compose.gm.yaml` — compose override: mounts this dir (`/workspace/gm_isaac`) + the XArm
  assets (`/workspace/gm_assets/xarm`) into the container, no submodule edits.
- `PLAN.md` — the Path-A adoption roadmap (this is Phase 1 of 7).
