# Cluster runbook — grasp-synthesis demo collection (v2 & v3)

Short, practical guide for collecting sim demonstrations with the grasp-synthesis collectors
on a cluster GPU node. For the deep dive (config structure, FSM, internals) see
`docs/grasp_synthesis_data_collection.md`.

## TL;DR

```bash
# v3 (RECOMMENDED — FEM gentleness metric + v2-matched grasp-pose DIVERSITY, on by default)
env -u PYTHONPATH -u ROS_DISTRO MUJOCO_GL=egl uv run --project envs/sim --no-sync \
  python grasp_synthesis/collect_demos_synth_v3.py \
    --experiment single_lift_mushroom_soft_abs_action \
    --n-episodes 650 --n-envs 8 --grasp-gpu
```

Output → `dataset/demos/single_lift_mushroom_soft/<YY-MM-DD-xyz>/data.pkl` (+ `config.yaml`,
`stats.yaml`). That `data.pkl` is the input to DPPO training (`docs/dppo_dp3_training_recipe.md`).

## Environment

- Runs in **`envs/sim`** (Python 3.12, Genesis + torch). Always `uv run --project envs/sim`
  (keeps cwd at repo root so `dataset/...` output lands correctly).
- **torch is installed manually** (not in any pyproject) — a bare `uv sync --project envs/sim`
  removes it; reinstall after syncing:
  `uv pip install --python envs/sim/.venv/bin/python "torch==2.5.1+cu121" --index-url https://download.pytorch.org/whl/cu121`
- Needs a **CUDA GPU**. One collection = one Genesis child (soft MPM) + (v3) the FEM solver.
  GPU memory is the limiter — an 8-env soft run uses ~6–9 GB; check `nvidia-smi` before stacking
  two collections on one GPU.
- Prefix the command with `env -u PYTHONPATH -u ROS_DISTRO MUJOCO_GL=egl` (clears a stray
  PYTHONPATH/ROS env; forces headless EGL rendering).

## v3 vs v2 — which to use

| | v3 (`collect_demos_synth_v3.py`) | v2 (`collect_demos_synth_v2.py`) |
|---|---|---|
| grasp metric | **FEM gentleness** (min indentation stress, real finger geometry) | geometric **SDF** cost (nearness + normal align + penetration) |
| grasp diversity | **v2-matched, on by default** (see below) | inherently diverse (single-start CMA) |
| soft-sim success | ~85% (diverse) / ~93% (argmax) | ~92% |
| compute | GPU FEM (`--grasp-gpu`); one CMA/env in-process | CPU CMA in a subprocess pool (no GPU for the search) |
| use it for | **the current default** — gentle + diverse demos | the SDF baseline / comparison |

Use **v3** unless you specifically want the SDF baseline. Both share the same CLI skeleton and
write the identical demo schema.

## Key flags (both collectors)

| flag | typical | meaning |
|---|---|---|
| `--experiment` | `single_lift_mushroom_soft_abs_action` | **the single source of truth** — task physics, DR ranges, action mode, obs all come from `configs/experiments/<name>.yaml`. Nothing else needs to be passed to keep collection ↔ training consistent. |
| `--n-episodes` | `650` | number of **successful** demos to collect (failures are retried to reach this) |
| `--n-envs` | `8` | parallel Genesis envs per batch (GPU-memory bound) |
| `--maxfevals` | `1145` (default) | CMA evals per env — grasp-search quality/speed knob |
| `--scene-dr-every` | `1` (default) | rebuild the object mesh (fresh scale+shape DR) every N batches; needs shape/scale fields in the experiment `dr:` |
| `--task-name` | *(experiment's task)* | override the **output folder name** only. Use it to keep the canonical `single_lift_mushroom_soft/` folder when the experiment's task cfg has a variant name (e.g. a camera-study fork). |
| `--out-dir` | `dataset/demos` | output root |
| `--record-video` | off | also write RGB execution clips + grasp-pose PNGs to `<run>/videos/` (slower) |
| `--seed` | `0` | RNG seed (pose DR). NOTE: CMA itself is not fully seeded — same-seed reruns are not bit-identical. |

### v3-only: grasp params + diversity

`--grasp-gpu` (use the GPU FEM solver — recommended). The **diversity knobs are ON by default**
(they reproduce v2's grasp-pose spread: pitch σ≈14°, continuous yaw):

| flag | default | note |
|---|---|---|
| `--grasp-diversity-tol` | `0.3` | sample among feasible grasps within 30% of the best gentleness score |
| `--grasp-jitter-deg` | `20` | ± pose jitter on the selected grasp (re-verified to still hold) |
| `--grasp-jitter-pos` | `0.003` | ± position jitter (m) |
| `--grasp-align` | `2000` | alignment weight (metric default 3e4; lowered so tilted grasps are allowed → pitch diversity) |
| `--grasp-pitch-seed-deg` | `25` | jitter the CMA pitch seed so the search explores tilt |

To collect the **concentrated / argmax** grasps instead (old v3 behaviour), pass
`--grasp-diversity-tol 0 --grasp-jitter-deg 0 --grasp-pitch-seed-deg 0 --grasp-align 30000`.

## Experiments (soft mushroom lift)

- `single_lift_mushroom_soft_abs_action` — **the one to use.** Soft MPM mushroom, absolute action,
  `soft_orientation` DR (±45° pitch/roll, full yaw, 0.25 flip) + per-scene size/shape/material DR.
- Rigid variants exist (`single_lift_mushroom_rigid_abs_action`) but rigid physics drops lift
  success (point contact) — prefer soft.

## What to expect

- **Runtime**: ~5 h for 650 demos at 8 envs (dominated by the per-env CMA grasp search +
  per-batch mesh rebuild). Fewer episodes / envs scale roughly linearly.
- Monitor `stats.yaml` (`episodes_saved / total_attempts`) mid-run; success ~85% (v3 diverse) is
  healthy. If it drops far below that, check `nvidia-smi` (OOM) and the log for `SYNTH FAILED`.
- The collector is **robust to a single bad env** (retries synthesis without diversity, then a
  default top-down grasp) — one unlucky mesh won't crash a run.

## After collection (done locally, not on the cluster)

Copy the run dir back, then: `gentle_manip.dppo.convert_demos … --view student --point-cloud`
→ DPPO pretrain → harness eval. See `docs/dppo_dp3_training_recipe.md` and
`docs/training_and_eval.md`.
