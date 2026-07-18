# configs/ — what each config is for

Configs are organized **by role**, not by method, so the reusable building blocks (task,
obs, action, dr, augmentation) are shared across every pipeline (SERL, DP3/DPPO, demo
collection, deploy) instead of being duplicated per method.

**Every config file starts with a 3-line header:**
```yaml
# [<type>] <one-line purpose>
# Used by: <script(s)> (<method/notes>)
# Status: active | legacy | experimental
```
So you can always tell, at a glance, what a config is and who consumes it.

## Directories

| dir | holds | leaf or composed | consumed by |
|-----|-------|------------------|-------------|
| `tasks/` | object + reward + success + sim dynamics for one task | leaf (referenced by an experiment's `task:`) | `SingleLiftTask(exp.task_cfg)` |
| `obs/` | observation modality sets (state / privileged / point cloud / tactile) | leaf (referenced by `obs:` / `views`) | perception pipeline via experiments |
| `action/` | action space + scales | leaf (`action:`) | action pipeline |
| `dr/` | domain-randomization ranges (sim-only) | leaf (`dr:`) | `SimBackend` |
| `augmentation/` | sim-only obs augmentation (point-cloud path) | leaf (`augmentation:`) | `PolicyEnv(augmentation=…)` |
| `experiments/` | **single-source-of-truth composition** (task+action+dr+aug+obs+views+rl) | composed | see below |
| `collect/` | demo-collection recipes (scripted / teleop) | recipe | `examples/collect_demos_sim.py` |
| `setup/` | robot/backend params (sim + real) | leaf | `XArm7Sim` / `XArm7Real` |

## The experiment is shared across BOTH training lines

One `experiments/<task>.yaml` + one demo set serves **SERL and DP3/DPPO**. The `views:` are
named subsets of the recorded superset obs — that's the mechanism that shares them:
- **`teacher`** view (state + privileged, incl. stress) → the **SERL** state teacher.
- **`student`** view (point cloud) → **DP3 / DPPO** and real **deploy**.

Method-specific parts of an experiment:
- **`rl:`** = RL-lines-only training hyperparams (SERL now, DPPO later). **Offline DP3 BC ignores it.**
- **`views:`** = obs subsets (as above) — NOT SERL-only.

## script → configs it reads

| script | reads |
|--------|-------|
| `scripts/serl_sim_server.py`, `serl/train_serl.py` (SERL) | an `experiments/*` (task + `views.teacher` + action + dr + aug + `rl`) |
| `examples/collect_demos_sim.py` (demo collection) | a `collect/*` recipe → its `experiment:` (records the SUPERSET obs + reward) |
| DP3 train / DPPO finetune (soft line) | an `experiments/*` (task + `views.student` point cloud + action + dr; NOT `rl`) |
| `scripts/deploy_real.py` (real) | `setup/real_lab.yaml` + a `point_cloud_1cam*` obs |
| `SimBackend` / `XArm7Sim` | `setup/sim_default.yaml` |

## Status legend
- **active** — in current use.
- **experimental** — one-off / staged / hypothesis-test (e.g. `augmentation/quat_snap*`, `obs/point_cloud_1cam_filtered`).
- **legacy** — superseded by the current rig/task (e.g. `obs/point_cloud_2cam` = 2-cam rig; `collect/single_lift_cube_rigid_*`).
