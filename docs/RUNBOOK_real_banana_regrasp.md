# RUNBOOK — real-robot diverse-start regrasp (XArm7), PREP ONLY

**This cannot be run from the Arrhenius cluster** — it needs the physical XArm7 +
D405/L515 cameras in the lab. This doc is the ready-to-execute plan for when the
user is at the robot. Everything sim-side (configs, policy, eval harness) is done by
the cluster campaign; only the physical demonstration collection and the real eval
are gated on hardware.

## 0. Why real demos at all
The sim2real transfer for the diverse-start regrasp policy is the whole point. The
sim policy (soft banana, `single_lift_banana_soft_diverse`) is the initialization;
real demos adapt it to the true camera/dynamics. Same OmniReset idea: single clean
successful grasps whose START configuration densely covers post-failed-attempt
states — collected by TELEOP, since there is no CMA-ES grasp synthesis on hardware.

## 1. Environment
```
uv run --project envs/deploy python -m gentle_manip.demos.record \
  --setup   gentle_manip/configs/setup/real_lab.yaml \
  --obs-config gentle_manip/configs/obs/point_cloud_1cam.yaml \
  --task-name single_lift_banana_real --input keyboard
```
`envs/deploy` (3.11) — pygame/pyspacemouse + XArm SDK + RealSense, genesis-free.
Writes `dataset/demos/single_lift_banana_real/<stem>.pkl` + `<stem>_config.yaml`.

## 2. Diverse-start protocol for TELEOP (mirror the sim `--start-modes` sweep)
Before each episode, MANUALLY move the EE to a start pose drawn from the same
distribution the sim collector uses (`_sample_start` in
`grasp_synthesis/collect_demos_diverse_start_v2.py`):
- **~60%** "sweep": somewhere on the line from home to just above the banana —
  vary the fraction each episode (near home / a third of the way / hovering close),
  with a few cm of lateral + a few degrees of rotational offset that GROWS the
  closer you start to the object.
- **~15%** "above": directly over the banana at a random height 3–20 cm, random yaw,
  slight tilt.
- **~12%** "ground": gripper low near the table, 5–16 cm to one side of the banana.
- **~12%** "air": a random point in the reachable workspace.
Then teleop ONE clean top-down grasp + lift + 3 s hold. No retries in the demo.
Also vary the banana: position (±8–10 cm), yaw (full), a little tilt, and swap in
2–3 physical banana pieces of different size/curvature across the session.

Target: **200–400 real episodes** (teleop is slow; the sim policy carries the rest).
Discard any episode with a frozen/dropped camera frame (BKSP) — same "no frozen
frames" rule as sim.

## 3. Convert + finetune
```
uv run --project envs/dppo python -m gentle_manip.dppo.convert_demos \
  dataset/demos/single_lift_banana_real \
  --out dataset/dppo/single_lift_banana_real_diverse_pcd \
  --experiment single_lift_banana_soft_diverse --view student --point-cloud --val-split 0.1
```
Then BC-finetune from the sim checkpoint (not from scratch):
```
uv run --project envs/dppo python -m gentle_manip.dppo.train \
  --config-path gentle_manip/dppo/cfg/single_lift_banana_soft_diverse_pcd \
  --config-name pre_diffusion_pointnet env=single_lift_banana_real_diverse_pcd \
  +resume_from=<sim_bc_checkpoint>.pt train.n_epochs=300 train.learning_rate=3e-5
```
(`+resume_from` restores model+ema; lower LR for finetune.)

## 4. Real eval
```
uv run --project envs/dp3 python gentle_manip/scripts/deploy_real.py \
  --ckpt <finetuned>.pt --normalization dataset/dppo/single_lift_banana_real_diverse_pcd/normalization.npz \
  --record   # saves runs in demo schema for the sim2real diagnose tools
```
Run ~20–30 rollouts, deliberately including start poses that look like a failed
first attempt (gripper hovering over/beside the banana). **Watch for genuine
redescend + reclose, not hover/jitter** — same bar as the sim eval.

## 5. Generalist (real)
After the single-object real loop works, repeat §2–4 with the 9-object pool
(one teleop session per object, ~150 episodes each) and finetune the cross-category
sim generalist. Same non-regraspable baseline (home-only teleop starts) for the
3-metric comparison.
