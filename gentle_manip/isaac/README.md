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

## Files
- `play_deformables.py` — spawns N deformable cubes + a pinhole camera, times physics-only vs
  physics+render+pointcloud FPS, and prints von-Mises peak/mean from the element stress tensor.
- `docker-compose.gm.yaml` — compose override that bind-mounts this dir to `/workspace/gm_isaac`.
