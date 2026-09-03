#!/usr/bin/env python3
"""LIVE view of the FINAL policy point cloud — the exact array a policy receives.

Unlike `raw_cloud_viewer.py` (camera only, no processing) and `point_cloud_viewer.py`
(camera + a standalone crop), this runs the REAL DEPLOYMENT PATH end to end:

    RealBackend (XArm7 + RealSense) -> RawObs -> PerceptionPipeline(obs_config) -> obs["point_cloud"]

so everything the policy's cloud goes through is applied and visible: the calibrated
extrinsic, the crop box, the voxel outlier filter, object_focus (which needs the LIVE
ee_pos, hence the robot connection), and the FPS/subsample down to `max_points`. Use it to
confirm what the policy actually sees before a deploy, and to sanity-check a new calibration
or crop.

    uv run --project envs/dp3 python -m gentle_manip.visualization.live_obs_cloud
    uv run --project envs/dp3 python -m gentle_manip.visualization.live_obs_cloud \
        --obs-config gentle_manip/configs/obs/point_cloud_1cam_armfocus.yaml --no-home

MOTION WARNING: `XArm7Real.connect()` HOMES the arm (position mode -> home pose -> servo
mode). That is what a real deploy does, so it is the default here. `--no-home` reads the arm
read-only instead (no motion_enable, no set_mode, no homing) — use it when the arm is already
posed where you want it, or when you simply do not want it to move.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

import gentle_manip
from gentle_manip.envs.real_backend import RealBackend
from gentle_manip.perception.obs_config import ObsConfig
from gentle_manip.perception.pipeline import PerceptionPipeline

_PKG = Path(gentle_manip.__file__).parent


def _resolve(p: Path) -> Path:
    if p.is_file():
        return p
    alt = _PKG.parent / p
    if alt.is_file():
        return alt
    raise FileNotFoundError(f"config not found: {p} (also tried {alt})")


class _ReadOnlyArm:
    """XArm7Real wrapper whose connect()/disconnect() never command the arm.

    Everything the backend READS is delegated untouched; the two motion entry points the
    backend could call (set_ee_pose / set_gripper_width) raise, so a viewer can never move
    the robot by accident.
    """

    def __init__(self, inner):
        self._inner = inner

    def connect(self) -> None:
        # XArm7Real.connect() homes; open only the SDK session and read state.
        from xarm.wrapper import XArmAPI
        self._inner._api = XArmAPI(self._inner.ip, is_radian=True)
        print("[live-obs] read-only arm session (no motion_enable / no homing)", flush=True)

    def disconnect(self) -> None:
        api = getattr(self._inner, "_api", None)
        if api is not None:
            api.disconnect()

    def __getattr__(self, name):
        if name in ("set_ee_pose", "set_gripper_width"):
            raise RuntimeError(f"{name} blocked: live_obs_cloud is a read-only viewer")
        return getattr(self._inner, name)


def main() -> None:
    p = argparse.ArgumentParser(description="Live view of the final policy point cloud")
    p.add_argument("--setup", type=Path,
                   default=_PKG / "configs" / "setup" / "real_lab.yaml")
    p.add_argument("--obs-config", type=Path,
                   default=_PKG / "configs" / "obs" / "point_cloud_1cam_armfocus.yaml")
    p.add_argument("--no-home", action="store_true",
                   help="do NOT home the arm; read its pose read-only (no motion at all)")
    p.add_argument("--hz", type=float, default=10.0, help="refresh rate cap")
    args = p.parse_args()

    setup = yaml.safe_load(open(_resolve(args.setup)))
    obs_cfg = ObsConfig.from_dict(yaml.safe_load(open(_resolve(args.obs_config))))
    pc = obs_cfg.point_cloud
    if pc is None:
        raise SystemExit(f"{args.obs_config} has no point_cloud block — nothing to show")

    print(f"[live-obs] setup      {args.setup}")
    print(f"[live-obs] obs config {args.obs_config}")
    print(f"[live-obs] cameras {pc.cameras}  max_points {pc.max_points}")
    print(f"[live-obs] crop  min {pc.crop_min}  max {pc.crop_max}")
    print(f"[live-obs] outlier_removal {getattr(pc, 'outlier_removal', None)}")
    print(f"[live-obs] object_focus    {getattr(pc, 'object_focus', None)}")
    shift = setup.get("point_cloud_shift", [0, 0, 0])
    if np.any(np.asarray(shift, float)):
        print(f"[live-obs] NOTE point_cloud_shift is ACTIVE: {shift}")

    backend = RealBackend(setup)
    if args.no_home:
        backend.robot = _ReadOnlyArm(backend.robot)

    from gentle_manip.visualization.live_cloud_viewer import LiveCloudViewer
    pipeline = PerceptionPipeline(obs_cfg)
    viewer = LiveCloudViewer(crop_min=pc.crop_min, crop_max=pc.crop_max,
                             title=f"policy cloud — {Path(args.obs_config).name}",
                             z_range=(float(pc.crop_min[2]), float(pc.crop_max[2])))
    backend.robot.connect()
    for cam in backend.cameras.values():
        cam.start()

    period = 1.0 / max(args.hz, 0.1)
    n = 0
    try:
        while viewer.alive:
            t0 = time.time()
            raw = backend._read_raw_obs()
            obs = pipeline.process(raw)
            cloud = obs["point_cloud"][0]                 # (max_points, 3), batch of 1
            n += 1
            if n % 20 == 1:
                nz = int(np.any(cloud != 0.0, axis=1).sum())
                z = cloud[np.any(cloud != 0.0, axis=1)][:, 2]
                print(f"[live-obs] frame {n}: {nz}/{len(cloud)} real points"
                      + (f"  z {z.min()*1000:.0f}..{z.max()*1000:.0f} mm" if len(z) else "")
                      + f"  ee {np.round(raw.ee_pos[0], 3)}", flush=True)
            if not viewer.update(cloud):
                break
            time.sleep(max(0.0, period - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        try:
            viewer.close()
        except Exception:
            pass
        backend.close()
        print("[live-obs] stopped")


if __name__ == "__main__":
    main()
