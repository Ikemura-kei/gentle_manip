# Sim2Real Diagnosis — DP3 `red_cube` policy on the Genesis sim

Why a real-trained DP3 policy approaches the cube in sim but then stalls. The
method isolates **control** gaps from **perception** gaps.

## TL;DR

Replaying a real demo's *actions* open-loop on the sim shows the **robot state is
near-identical** (ee_pos within 2–3 mm, gripper within 1.4 mm over 280 steps), so
the stall is a **point-cloud (perception) gap**, not control or `agent_pos`.

## Method — `replay_demo_in_sim.py`

Feed one demo episode's recorded **actions** through the sim (same `ActionPipeline`),
record the sim observations, and compare to the recorded **real** observations.
- robot state diverges → gap is control (IK / bounds / scaling / dynamics)
- robot state matches but cloud differs → gap is perception (the policy's input)

## Findings

**Control: matches** (`figures/replay_state.png`). Same actions → same arm motion:
ee_pos x/y/z, gripper (0.08→0.028→lift), and quat all overlay. This validates the
TCP offset (0.171 m), gripper width lookup, IK, EE bounds, and home pose. The demo's
z reaching 0.003 (= `EE_BOUNDS_MIN.z`) independently confirms the fingertip TCP.

**Perception: the gap** (`figures/{before_fov60,after_fov55}/replay_pointcloud.png`).
Gross structure is similar (gripper-dominated cluster, ~950/1024 high points), but the
sim cloud sits **higher** than real at every step *despite the arm matching to 3 mm* —
a camera/gripper-appearance difference, not a control error. It is also **clean** (no
L515 sensor noise, dropout, or density texture).

**Sim FOV was too large** (fixed). Genesis `fov` is vertical: `f = 0.5·height/tan(fov/2)`.
At `fov=60`, res 640×480 → VFOV 60°, HFOV ≈75°; the L515 depth is nominally ≈55°×70° (V×H),
so sim saw a wider cone. We swept fov and measured the per-step cloud zmean vs real
(offset in parens):

| step | real zmean | fov=60 | fov=55 | **fov=50** |
|------|-----------|--------|--------|--------|
| t=0  | 0.258 | 0.298 (+0.040) | 0.281 (+0.023) | 0.262 (**+0.004**) |
| t=140| 0.166 | 0.200 (+0.034) | 0.199 (+0.033) | 0.183 (**+0.017**) |
| t=279| 0.272 | 0.292 (+0.020) | 0.278 (+0.006) | 0.260 (**−0.012**) |
| mean &#124;offset&#124; | | 0.031 | 0.021 | **0.011** |

Narrowing the fov shrinks the offset monotonically; **`fov=49`** (VFOV 49°, HFOV ≈63°)
is the value now set in `SingleLiftTask.scene_spec`. Caveat: that is *narrower* than the
L515's nominal 55°, so it is partly **compensating** for the next gap rather than being
physically exact. The residual is attributed to the **camera extrinsic**: sim places
cam_ext by pos/lookat (approximate) rather than the exact calibrated `WORLD_T_CAM_EXT` —
a ~2° orientation error is ~3 cm at 1 m. The principled fix is to set the sim camera to
the real L515's measured **intrinsics K + extrinsic**; that is the next step.

### Multi-trajectory validation (`figures/eval_fov49/`, fov=49)

To confirm this isn't cherry-picked from one episode, 5 random trajectories were replayed
(cube placed at each demo's grasp location so the cloud compare is fair):

| ep | ee_err x,y,z (mm) | cloud zmean offset (mm) |
|----|-------------------|-------------------------|
| 13 | 2.4, 0.7, 2.5 | 12.1 |
| 15 | 2.2, 1.0, 2.2 | 10.3 |
| 24 | 2.5, 0.7, 2.2 | 10.3 |
| 29 | 2.2, 0.4, 2.3 | 10.9 |
| 39 | 2.3, 1.0, 2.2 | 10.6 |

Control holds at **~2–3 mm across all five** diverse grasps (cube x∈[0.38,0.53]); the
perception gap is a consistent **~10–12 mm** cloud-height offset — the extrinsic +
sensor-noise residual. Each `traj_NN.png` shows ee_pos x/y/z + gripper + cloud zmean(t)
and a real-vs-sim cloud overlay at the grasp.

## Suspected causes → fixes (impact order)

1. **Sensor-noise gap** (clean sim vs noisy L515) — add point-cloud jitter + random
   dropout + small per-cloud offset. Best applied at DP3 **training** time (retrain on
   augmented real-like clouds); a sim-side `--pc-noise` knob lets you test now.
2. **Camera FOV** — set sim cam_ext `fov` 60 → 55 to match the L515 (confirmed above).
   For exactness, set the sim camera to the real L515's measured intrinsics K.
3. **Camera extrinsic** — sim places cam_ext via pos/lookat (approximate); use the
   exact calibrated `WORLD_T_CAM_EXT` 4×4 so the cloud frame matches real (a few
   degrees of orientation error → ~3 cm offset at ~1 m, consistent with what we see).
4. **Gripper mesh** — the URDF gripper silhouette differs from the real gripper as the
   L515 sees it (the residual ~cm offset after 2–3 are fixed).

## Reproduce

```bash
# demo obs/action ranges (gripper-open, workspace, quat sign, crop, action):
uv run --project envs/sim python -m gentle_manip.scripts.inspect_demo \
    --demo dataset/demos/red_cube/26-06-18-jcd.pkl

# open-loop replay of N random trajectories (one figure each in --out-dir):
uv run --project envs/sim python examples/sim2real_diagnose/replay_demo_in_sim.py \
    --demo dataset/demos/red_cube/26-06-18-jcd.pkl --n-episodes 5 \
    --out-dir examples/sim2real_diagnose/figures/eval_fov49
```

## Files

- `replay_demo_in_sim.py` — the open-loop replay diagnostic.
- `figures/{before_fov60,after_fov55,after_fov50}/` — single-episode (ep 0) fov sweep;
  state is identical across all (control is fov-independent), only the cloud changes.
- `figures/eval_fov49/traj_NN.png` — 5 random trajectories at the chosen `fov=49`
  (combined per-trajectory figures; cube placed at each demo's grasp).
- `figures/gripper_curve.png` — gripper joint↔width calibration sweep (near-linear, 2.4%).
- `figures/sim_pointcloud_sanity.png` — standalone sim cloud sanity check.
- Related tool: `gentle_manip/scripts/inspect_demo.py` (run via `-m`).

`replay_state.png` is the robot-state comparison (real vs sim → **matches**);
`replay_pointcloud.png` is the point-cloud comparison (**the gap**).
