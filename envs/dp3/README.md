# envs/dp3 — DP3 training / inference environment

Python 3.8 uv env for DP3 (3D Diffusion Policy). Depends on the shared
`gentle-manip` library + the `diffusion_policy_3d` package (editable, from the
`third_party/DP3` submodule). Run everything with `uv run --project envs/dp3 …`.

## 1. Sync the env

```bash
uv sync --project envs/dp3
```

Installs (from `pyproject.toml` + lock): torch 2.4.1+cu121 (pinned via the cu121
index — change versions/index in `pyproject.toml` for different hardware), the
DP3 package editable, and the DP3 stack (zarr 2.12, numba 0.56.4 → numpy 1.23.5,
hydra 1.2, diffusers 0.11.1, …). No separate `pip install -e 3D-Diffusion-Policy`
or `pip install torch` needed — both are handled here.

## 2. Build pytorch3d (CUDA extension — manual)

`pytorch3d` compiles against torch, so it can't go through `uv add`/the lock —
build it with isolation off, using the **system CUDA 12.1 + system gcc** (NOT the
conda toolchain, which has its own sysroot and breaks the build):

```bash
conda deactivate                                  # drop conda's CC/CXX/sysroot (the usual culprit)

export CUDA_HOME=/usr/local/cuda-12.1
export PATH=/usr/local/cuda-12.1/bin:$PATH        # 12.1 nvcc must beat /usr/bin/nvcc (10.1)
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
export CC=/usr/bin/gcc CXX=/usr/bin/g++           # system gcc 9.4 — CUDA-12.1-compatible
export TORCH_CUDA_ARCH_LIST="8.9"                 # RTX 4090 — fewer kernels, faster build

nvcc --version        # MUST show 12.1, not 10.1
$CXX --version        # MUST show gcc 9.4.0, NOT x86_64-conda-linux-gnu

uv pip install --python envs/dp3/.venv/bin/python --no-build-isolation \
  -e third_party/DP3/third_party/pytorch3d_simplified
```

Note: `uv pip` targets the env via `--python <venv>/bin/python`, **not** `--project`
(that flag only steers `uv sync`/`uv run`).

## 3. Deploy a trained policy on the real robot

This env also carries the hardware SDKs (`pyrealsense2==2.54.2.5684` has a cp38
wheel + L515 support; `xArm-Python-SDK`), so a trained DP3 policy runs the real
XArm7 **in one process** — no IPC. `gentle_manip.scripts.deploy_real` builds
`PolicyEnv(RealBackend, task=None)` and drives it with the policy (receding
horizon): it loads the policy with the **exact training config embedded in the
checkpoint**, then loops `predict_action → execute n_action_steps → re-plan`.

```bash
# workspace clear, e-stop in hand:
uv run --project envs/dp3 python gentle_manip/scripts/deploy_real.py \
  --ckpt third_party/DP3/3D-Diffusion-Policy/data/outputs/real_xarm7_red_cube-dp3-0112_seed0/checkpoints/latest.ckpt \
  --obs-config gentle_manip/configs/obs/point_cloud_1cam.yaml \
  --max-steps 50 --rate 10
```

Start with a small `--max-steps` and low `--rate`; the first `reset()` homes the
arm, then it runs the policy. Obs/action parity is automatic — same
`point_cloud_1cam` pipeline + `ActionPipeline` as data collection, and the policy
outputs raw `[-1,1]` deltas through the same `EE_BOUNDS`-clipped path. (`gentle_manip`
and DP3 are imported from source via `sys.path`; neither is installed here.)

## Notes

- **Other hardware:** change the torch versions in `pyproject.toml` AND the cu121
  index URL to match your GPU/driver; set `CUDA_HOME` to your CUDA toolkit and
  `TORCH_CUDA_ARCH_LIST` to your GPU's compute capability.
- **`uv sync` removes manually-installed packages** not in the lock — so re-run
  the pytorch3d build (step 2) after any future `uv sync --project envs/dp3`.
- **Sim-benchmark deps not needed for real-robot training.** mujoco-py, dexart,
  Metaworld, gym-0.21, mj_envs/mjrl (and dm_control) are only reached via
  `diffusion_policy_3d.env` / a sim `env_runner`. The real task uses
  `env_runner: null`, and `train.py`'s import chain never touches them.

The other dependencies the DP3 REAME asked to install (like mujoco, gym, etc.) are not necessary because we only concern training with a pre-collected data, no native environments in this repo is required.