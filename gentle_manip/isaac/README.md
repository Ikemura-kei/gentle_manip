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

## Files
- `play_deformables.py` — deformable FEM benchmark: physics-only vs +render+pointcloud FPS,
  per-element von-Mises stress; `--dump` captures rgb+geometry+stress, `--squeeze` compresses.
- `viz_capture.py` — host-side (envs/deploy): capture.npz -> mesh video + stress-node video + PNG.
- `spawn_arm.py` — Phase 1: spawn XArm7, hold home, read joint/EE state, `--joint-test` tracking.
- `docker-compose.gm.yaml` — compose override: mounts this dir (`/workspace/gm_isaac`) + the XArm
  assets (`/workspace/gm_assets/xarm`) into the container, no submodule edits.
- `PLAN.md` — the Path-A adoption roadmap (this is Phase 1 of 7).
