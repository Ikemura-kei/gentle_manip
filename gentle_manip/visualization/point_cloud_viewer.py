from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

import gentle_manip
from gentle_manip.perception.depth_to_pointcloud import depth_to_pointcloud
from gentle_manip.robot import xarm7_config as cfg

# Live point-cloud viewer for crop tuning. Camera-only — it does NOT connect to
# or move the robot. Shows the cam_ext cloud in the robot base frame (via the
# static WORLD_T_CAM_EXT), the workspace crop box (red wireframe), and the base
# frame axes, so you can position the camera/objects and read off good crop
# bounds. Runs in the 3.11 deploy env:
#   uv run --project envs/deploy python -m gentle_manip.visualization.point_cloud_viewer
#
# open3d / pyrealsense2 are imported lazily (the `real` extra).

_PKG = Path(gentle_manip.__file__).parent


def _resolve(path: Path) -> Path:
    if path.is_file():
        return path
    alt = _PKG.parent / path
    if alt.is_file():
        return alt
    raise FileNotFoundError(f"config not found: {path} (also tried {alt})")


def _crop_box(crop_min, crop_max, o3d):
    box = o3d.geometry.AxisAlignedBoundingBox(np.asarray(crop_min, float),
                                              np.asarray(crop_max, float))
    box.color = (1.0, 0.0, 0.0)
    return box


def main() -> None:
    p = argparse.ArgumentParser(description="Live cam_ext point-cloud viewer (crop tuning)")
    p.add_argument("--setup", type=Path,
                   default=_PKG / "configs" / "setup" / "real_lab.yaml")
    p.add_argument("--obs-config", type=Path,
                   default=_PKG / "configs" / "obs" / "point_cloud_1cam.yaml")
    p.add_argument("--camera", default="cam_ext", help="camera name in the setup config")
    p.add_argument("--show-crop", action="store_true",
                   help="also print how many points fall inside the crop box each second")
    args = p.parse_args()

    import open3d as o3d
    from gentle_manip.envs.realsense_camera import RealSenseCamera

    setup = yaml.safe_load(open(_resolve(args.setup)))
    obs = yaml.safe_load(open(_resolve(args.obs_config)))
    pc_cfg = obs["point_cloud"]
    crop_min = np.asarray(pc_cfg["crop_min"], dtype=np.float32)
    crop_max = np.asarray(pc_cfg["crop_max"], dtype=np.float32)

    cam_cfg = setup["cameras"][args.camera]
    extrinsic = np.asarray(
        cam_cfg.get("world_T_cam", cfg.WORLD_T_CAM_EXT), dtype=np.float32
    )

    cam = RealSenseCamera(
        name=args.camera, serial=cam_cfg["serial"],
        width=cam_cfg.get("width", 640), height=cam_cfg.get("height", 480),
        depth_min=cam_cfg.get("depth_min", 0.1), depth_max=cam_cfg.get("depth_max", 0.85),
    )
    cam.start()

    vis = o3d.visualization.Visualizer()
    vis.create_window(f"{args.camera}: cloud + crop box (red).  Close window to quit.")
    pcd = o3d.geometry.PointCloud()
    added = False
    vis.add_geometry(_crop_box(crop_min, crop_max, o3d))
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))

    print(f"viewer: {args.camera}  crop_min={crop_min.tolist()}  crop_max={crop_max.tolist()}")
    print("position the camera/objects; tighten crop bounds in the obs config, re-run.")

    try:
        while True:
            depth, rgb, K = cam.get_frame()
            pts, valid = depth_to_pointcloud(depth[None], K, extrinsic)
            pts, valid = pts[0], valid[0]
            colors = rgb.reshape(-1, 3)[valid].astype(np.float32) / 255.0  # already RGB

            pcd.points = o3d.utility.Vector3dVector(pts[valid])
            pcd.colors = o3d.utility.Vector3dVector(colors)
            if not added:
                vis.add_geometry(pcd)          # add once geometry has points (sets the view)
                added = True
            else:
                vis.update_geometry(pcd)

            if args.show_crop:
                p = pts[valid]
                inside = np.all((p >= crop_min) & (p <= crop_max), axis=1).sum()
                print(f"\rpoints: {valid.sum():6d} total   {inside:6d} in-crop", end="")

            if not vis.poll_events():           # window closed
                break
            vis.update_renderer()
    finally:
        print()
        vis.destroy_window()
        cam.stop()


if __name__ == "__main__":
    main()
