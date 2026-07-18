"""Smoke test for envs/dppo (Python 3.10: DPPO diffusion-policy PPO + torch).

    uv run --project envs/dppo python examples/env_debug/check_dppo.py

Checks the DPPO stack (hydra/gym/torch + the dppo package's model/agent modules) and our
PointNet diffusion policy, and builds a PointNetDiffusionMLP end to end (exercises
model.common.*, model.diffusion.*, and the PointNet encoder). Also imports the genesis
bridge (gentle_manip.envs.rpc) the trainer uses to reach the sim over a socket.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _common as C  # noqa: E402

C.header("envs/dppo (3.10: DPPO + torch)")
C.check("import torch (+CUDA)", C.torch_cuda)
C.check("import hydra", C.imp("hydra"))
C.check("import omegaconf", C.imp("omegaconf"))
C.check("import gym", C.imp("gym"))
# dppo package internals (installed editable from third_party/dppo).
C.check("import dppo model.diffusion.diffusion_eval", lambda: C.imp("model.diffusion.diffusion_eval")())
C.check("import dppo agent.finetune", lambda: __import__("agent.finetune", fromlist=["x"]) and "ok")
# Our bridge + policy.
C.check("import gentle_manip.envs.rpc (sim bridge)", lambda: C.imp("gentle_manip.envs.rpc")())
C.check("import PointNetDiffusionMLP", lambda: __import__(
    "gentle_manip.dppo.pointnet_diffusion", fromlist=["PointNetDiffusionMLP"]) and "ok")


def _build_pointnet_policy():
    import torch
    from gentle_manip.dppo.pointnet_diffusion import PointNetDiffusionMLP
    # Matches the eval config: obs_dim 8 x cond_steps 2 = cond_dim 16, 1024-pt clouds.
    net = PointNetDiffusionMLP(
        action_dim=7, horizon_steps=4, cond_dim=16, pc_cond_steps=1, visual_feature_dim=256,
        mlp_dims=[512, 512, 512], residual_style=True,   # matches the trained eval config
        pointnet={"in_channels": 3, "use_layernorm": True, "final_norm": "layernorm"})
    n = sum(p.numel() for p in net.parameters())
    return f"PointNetDiffusionMLP built, {n:,} params"


C.check("build PointNetDiffusionMLP (full model stack)", _build_pointnet_policy)
C.summary()
