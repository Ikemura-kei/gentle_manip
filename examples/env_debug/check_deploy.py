"""Smoke test for envs/deploy (Python 3.11: hardware SDKs + viz, GENESIS-FREE).

    uv run --project envs/deploy python examples/env_debug/check_deploy.py

Verifies the teleop/deploy hardware SDKs import (no device needed to import them), the
viz stack loads, the gentle_manip genesis-free core runs, and — importantly — that
genesis is NOT importable here (the RawObs boundary rule).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _common as C  # noqa: E402

C.header("envs/deploy (3.11: hardware SDKs + viz, genesis-free)")
# Hardware SDKs (import-only; no camera/robot required to import).
C.check("import pyrealsense2", C.imp("pyrealsense2"))
C.check("import xarm.wrapper.XArmAPI", lambda: __import__("xarm.wrapper", fromlist=["XArmAPI"]) and "ok")
C.check("import pygame", C.imp("pygame"))
C.check("import pyspacemouse", C.imp("pyspacemouse"))
# Viz stack.
C.check("import open3d", C.imp("open3d"))
C.check("import cv2", C.imp("cv2"))
C.check("import imageio", C.imp("imageio"))
# gentle_manip genesis-free core.
C.check("import gentle_manip", C.imp("gentle_manip"))
C.check("import RealBackend", lambda: C.imp("gentle_manip.envs.real_backend")())
C.check("import DemoRecorder (record.py)", lambda: C.imp("gentle_manip.demos.record")())
# The boundary rule: this env must stay genesis-free.
C.check("genesis is NOT importable (genesis-free env)", C.expect_absent("genesis"))


def _pipelines():
    import numpy as np
    from gentle_manip.actions.action_config import ActionConfig
    from gentle_manip.actions.pipeline import ActionPipeline
    ap = ActionPipeline(ActionConfig.from_dict({"scales": [0.005] * 6 + [0.05]}))
    scaled = ap.process(np.zeros((1, 7), np.float32))
    return f"action scaled shape {np.asarray(scaled).shape}"


C.check("ActionPipeline.process", _pipelines)
C.summary()
