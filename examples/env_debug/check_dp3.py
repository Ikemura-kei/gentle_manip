"""Smoke test for envs/dp3 (Python 3.8: DP3 + torch + pytorch3d + hardware SDKs).

    uv run --project envs/dp3 python examples/env_debug/check_dp3.py

This env runs BOTH DP3 training/eval AND real policy deployment, so it checks the DP3
stack, pytorch3d (with a real farthest-point-sampling call), and the hardware SDKs. It
is genesis-free (real side of the RawObs boundary).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _common as C  # noqa: E402

# DP3 is a namespace-style package meant to run with its own dir on sys.path
# (`cd 3D-Diffusion-Policy && python train.py`) — its editable install doesn't expose it
# as a top-level import. Put that dir on the path so the smoke test mirrors real DP3 usage
# and validates that DP3's own deps are installed (they'd fail below if missing).
_DP3 = Path(__file__).resolve().parents[2] / "third_party/DP3/3D-Diffusion-Policy"
if _DP3.is_dir() and str(_DP3) not in sys.path:
    sys.path.insert(0, str(_DP3))

C.header("envs/dp3 (3.8: DP3 + torch + pytorch3d + real SDKs)")
C.check("import torch (+CUDA)", C.torch_cuda)
C.check("import torchvision", C.imp("torchvision"))
C.check("import pytorch3d", C.imp("pytorch3d"))
# DP3 is a namespace package (no top-level __init__); test a real submodule it uses.
C.check("import diffusion_policy_3d.common.pytorch_util (DP3)",
        C.imp("diffusion_policy_3d.common.pytorch_util"))
# Real deployment shares this env.
C.check("import pyrealsense2", C.imp("pyrealsense2"))
C.check("import xarm.wrapper.XArmAPI", lambda: __import__("xarm.wrapper", fromlist=["XArmAPI"]) and "ok")
C.check("import gentle_manip", C.imp("gentle_manip"))
C.check("genesis is NOT importable (genesis-free env)", C.expect_absent("genesis"))


def _pytorch3d_fps():
    import torch
    from pytorch3d.ops import sample_farthest_points
    pts = torch.rand(1, 2000, 3)
    _, idx = sample_farthest_points(pts, K=512)
    assert idx.shape == (1, 512), idx.shape
    return f"FPS 2000->512 idx {tuple(idx.shape)}"


C.check("pytorch3d sample_farthest_points (the real/sim parity op)", _pytorch3d_fps)
C.summary()
