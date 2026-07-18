# env_debug — standalone environment smoke tests

Quick checks that each of the project's uv environments is set up correctly: the key
third-party packages import, CUDA/JAX see the GPU, and the `gentle_manip` code (plus the
relevant policy stack) loads and runs on synthetic data. Use these after setting up the
environments on a new machine / cluster.

## The five environments

| Env | Python | Purpose | check script |
|-----|--------|---------|--------------|
| `envs/sim`    | 3.12 | sim / training / tests — genesis + torch | `check_sim.py` |
| `envs/deploy` | 3.11 | teleop demo collection + viz (hardware SDKs, **genesis-free**) | `check_deploy.py` |
| `envs/dp3`    | 3.8  | DP3 training/eval **and** real deploy (DP3 + torch + pytorch3d + RealSense/xArm) | `check_dp3.py` |
| `envs/dppo`   | 3.10 | DPPO diffusion-policy PPO finetune (torch + dppo) | `check_dppo.py` |
| `envs/serl`   | 3.10 | SERL SAC teacher — JAX/flax + serl_launcher | `check_serl.py` |

## Setup (per env), then run

```bash
# 1. sync each env you need (installs deps from envs/<name>/pyproject.toml)
uv sync --project envs/sim        # (and deploy / dp3 / dppo / serl)
# 2. torch is installed MANUALLY in sim/dp3/dppo, and pytorch3d in dp3 — see each
#    envs/<name>/pyproject.toml header for the exact `uv pip install ... torch==...` line.
#    A bare `uv sync` can drop the manual torch; reinstall it after syncing.

# run ALL env checks (each in its own env):
bash examples/env_debug/run_all.sh
# or a subset:
bash examples/env_debug/run_all.sh sim dppo
# or one env directly:
uv run --project envs/dp3 python examples/env_debug/check_dp3.py
```

Each script prints `[PASS]/[FAIL]` per check and a `RESULT: k/n passed` line, and exits
non-zero if anything failed. `run_all.sh` prints a per-env `PASS/FAIL/SKIP` summary and
exits non-zero if any env failed.

## Notes / gotchas

- **Clean env:** `run_all.sh` strips `PYTHONPATH`/`ROS_DISTRO` (uv envs are self-contained)
  and sets `MUJOCO_GL=egl` for the headless genesis import. Run it this way on a box whose
  shell sources ROS or exports unrelated `PYTHONPATH`s, so the test reflects the real env.
- **`gentle_manip` path:** it's editable-installed only in `sim`/`deploy`; in `dp3`/`dppo`/
  `serl` the real launchers inject the repo root onto `sys.path`. `_common.py` mirrors that
  (adds the repo root), and `check_dp3.py` also adds the DP3 source dir — DP3 is a
  namespace-style package meant to run with its own dir on the path. The env's actual
  third-party deps are still validated by the deep functional imports.
- **serl / jax:** `import jax` and `serl_launcher` are checked first (flushed), then a CUDA
  kernel runs LAST. `jax[cuda12]` can **segfault (exit 139) on a driver mismatch** — if that
  happens, rebuild jax per the find-links note in `envs/serl/pyproject.toml`. The earlier
  PASS lines still print, so you can tell the crash is CUDA-init, not the stack itself.
- **genesis-free assertion:** `deploy`/`dp3`/`serl` checks assert `import genesis` FAILS —
  the RawObs-boundary rule that keeps the real/policy side genesis-free.
