"""Tiny check harness shared by the per-env smoke tests (pure stdlib, py3.8-3.12).

Each check_<env>.py runs under its OWN uv environment:
    uv run --project envs/<env> python examples/env_debug/check_<env>.py
and prints a PASS/FAIL line per check, then a summary; exits non-zero if any failed.
"""
import importlib
import sys
from pathlib import Path

# Put the repo root on sys.path so `import gentle_manip` works in EVERY env, mirroring the
# real launchers (e.g. gentle_manip.dppo.train injects the repo root after hydra's chdir).
# gentle_manip is editable-installed only in envs/sim & envs/deploy; dp3/dppo/serl reach it
# via this path injection. Its actual third-party deps are still validated by the deep
# functional imports below (they'd fail if numpy/torch/etc. were missing from the env).
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_results = []


def header(name: str) -> None:
    print(f"\n=== {name} | python {sys.version.split()[0]} ===", flush=True)
    print(f"    {sys.executable}", flush=True)


def check(label, fn) -> None:
    """Run fn(); PASS unless it raises. fn may return a short info string.
    flush=True so partial results survive if a LATER check hard-crashes the
    interpreter (e.g. a jax CUDA-init segfault)."""
    try:
        info = fn()
        print(f"  [PASS] {label}" + (f"  ({info})" if info else ""), flush=True)
        _results.append((label, True))
    except Exception as e:  # noqa: BLE001 - report any failure
        print(f"  [FAIL] {label}  -> {type(e).__name__}: {e}", flush=True)
        _results.append((label, False))


def imp(module: str):
    """Return a check fn that imports `module` and reports its version."""
    def f():
        m = importlib.import_module(module)
        return f"v{getattr(m, '__version__', '?')}"
    return f


def expect_absent(module: str):
    """Return a check fn that PASSES only if `module` is NOT importable
    (used to assert the genesis-free envs stay genesis-free)."""
    def f():
        try:
            importlib.import_module(module)
        except ImportError:
            return "absent (as required)"
        raise AssertionError(f"{module} IS importable but must not be in this env")
    return f


def torch_cuda():
    """Import torch and report CUDA availability + device."""
    import torch
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU-only"
    return f"torch v{torch.__version__}, cuda={torch.cuda.is_available()} [{dev}]"


def summary() -> None:
    n = len(_results)
    ok = sum(1 for _, p in _results if p)
    print("\n" + "-" * 56)
    if ok < n:
        failed = [l for l, p in _results if not p]
        print(f"RESULT: {ok}/{n} passed  |  FAILED: {failed}")
        sys.exit(1)
    print(f"RESULT: {ok}/{n} passed  |  ALL OK")
    sys.exit(0)
