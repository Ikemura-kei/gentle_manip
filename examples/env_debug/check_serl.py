"""Smoke test for envs/serl (Python 3.10: JAX/flax + serl_launcher — the SAC teacher).

    uv run --project envs/serl python examples/env_debug/check_serl.py

Checks the JAX RL stack imports, JAX sees a device and runs a kernel, serl_launcher (the
SAC/BC/RLPD agents + replay buffer) loads, and the genesis bridge imports. This stack is
JAX-only and genesis-free; the genesis sim is reached over gentle_manip.envs.rpc.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _common as C  # noqa: E402

C.header("envs/serl (3.10: JAX/flax + serl_launcher)")

# Safe module imports first (no CUDA init), each flushed — so if the CUDA kernel check
# below segfaults (jax[cuda12] can crash on a driver mismatch; see envs/serl/pyproject.toml
# header for the find-links fallback), these results are already on screen.
C.check("import jax (module, no device init)", C.imp("jax"))
C.check("import flax", C.imp("flax"))
C.check("import optax", C.imp("optax"))
C.check("import serl_launcher", C.imp("serl_launcher"))
C.check("import serl_launcher.agents", lambda: __import__("serl_launcher.agents", fromlist=["x"]) and "ok")
C.check("import gentle_manip.envs.rpc (sim bridge)", lambda: C.imp("gentle_manip.envs.rpc")())
C.check("genesis is NOT importable (genesis-free env)", C.expect_absent("genesis"))


def _jax_cuda():
    # RISKY LAST: first backend op triggers CUDA init. A segfault here (exit 139) means
    # jax's CUDA wheels don't match the box's driver — rebuild jax per the pyproject header.
    import jax
    import jax.numpy as jnp
    devs = [d.platform for d in jax.devices()]
    y = float(jnp.dot(jnp.ones((64, 64)), jnp.ones((64, 64))).sum())
    return f"devices={devs}, kernel_ok={y == 64 ** 3}"


C.check("jax device + kernel (CUDA init — may segfault on driver mismatch)", _jax_cuda)
C.summary()
