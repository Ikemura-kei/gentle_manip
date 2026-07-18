"""Smoke test for envs/dppo_deploy (Python 3.10: DPPO policy + real hardware SDKs).

    uv run --project envs/dppo_deploy python examples/env_debug/check_dppo_deploy.py

The env that runs a trained DPPO diffusion policy on the real XArm7 (policy + RealBackend in
one process, no IPC). Checks the DPPO policy stack, the hardware SDKs (import-only; no
device needed), that the genesis-free RealBackend/_DiffusionPolicy load, and that genesis is
NOT importable (real side of the RawObs boundary). Builds the PointNet policy end to end.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _common as C  # noqa: E402

C.header("envs/dppo_deploy (3.10: DPPO policy + real hardware)")
C.check("import torch (+CUDA)", C.torch_cuda)
# Hardware SDKs (import-only; no camera/robot required).
C.check("import pyrealsense2", C.imp("pyrealsense2"))
C.check("import xarm.wrapper.XArmAPI", lambda: __import__("xarm.wrapper", fromlist=["XArmAPI"]) and "ok")
# DPPO policy stack.
C.check("import dppo model.diffusion.diffusion_eval", lambda: C.imp("model.diffusion.diffusion_eval")())
C.check("import gentle_manip.dppo.eval_agent._DiffusionPolicy",
        lambda: __import__("gentle_manip.dppo.eval_agent", fromlist=["_DiffusionPolicy"]) and "ok")
# Genesis-free real side.
C.check("import RealBackend", lambda: __import__("gentle_manip.envs.real_backend", fromlist=["RealBackend"]) and "ok")
C.check("genesis is NOT importable (genesis-free env)", C.expect_absent("genesis"))


def _build_policy():
    import torch  # noqa: F401
    from gentle_manip.dppo.pointnet_diffusion import PointNetDiffusionMLP
    net = PointNetDiffusionMLP(
        action_dim=7, horizon_steps=4, cond_dim=16, pc_cond_steps=1, visual_feature_dim=256,
        mlp_dims=[512, 512, 512], residual_style=True,
        pointnet={"in_channels": 3, "use_layernorm": True, "final_norm": "layernorm"})
    return f"PointNetDiffusionMLP built, {sum(p.numel() for p in net.parameters()):,} params"


C.check("build PointNetDiffusionMLP (deploy policy stack)", _build_policy)
C.summary()
