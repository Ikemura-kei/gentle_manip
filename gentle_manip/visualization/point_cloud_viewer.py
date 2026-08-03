from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

import gentle_manip
from gentle_manip.perception.depth_to_pointcloud import depth_to_pointcloud
from gentle_manip.perception.pointcloud_ops import (
    crop_pointcloud, remove_outliers_voxel, subsample_pointcloud)
from gentle_manip.perception.obs_config import ObsConfig
from gentle_manip.robot import xarm7_config as cfg

# Live point-cloud viewer for crop tuning. Camera-only — it does NOT connect to
# or move the robot. Shows the cam_ext cloud in the robot base frame (via the
# static WORLD_T_CAM_EXT), the workspace crop box (red wireframe), and the base
# frame axes, so you can position the camera/objects and read off good crop
# bounds. Runs in the 3.11 deploy env:
#   uv run --project envs/deploy python -m gentle_manip.visualization.point_cloud_viewer
#
# --show-processed additionally overlays the FINAL processed cloud (crop + outlier
# removal + subsample to max_points — the exact gentle_manip.perception.pointcloud_ops
# functions PerceptionPipeline uses, reused here so it can never silently drift from
# the real pipeline) in a second color, so you can see raw vs. what the policy
# actually receives in the SAME view. NOTE: `object_focus` (focus_z_lo) needs a live
# ee_pos, which this camera-only tool doesn't have (no robot connection) — it's
# skipped with a one-time warning if the obs config has it enabled; use
# `demos.record --show-pointcloud` instead for a fully faithful view when that
# filter matters (it drives PerceptionPipeline through a real robot connection).
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
    p.add_argument("--show-processed", action="store_true",
                   help="also overlay the FINAL processed cloud (crop + outlier removal + "
                        "subsample to max_points, via the same pointcloud_ops functions "
                        "PerceptionPipeline uses) in orange, alongside the raw gray cloud")
    args = p.parse_args()

    import open3d as o3d
    from gentle_manip.envs.realsense_camera import RealSenseCamera

    setup = yaml.safe_load(open(_resolve(args.setup)))
    obs_dict = yaml.safe_load(open(_resolve(args.obs_config)))
    pc_full_cfg = ObsConfig.from_dict(obs_dict).point_cloud   # same parsing PerceptionPipeline uses
    pc_cfg = obs_dict["point_cloud"]
    crop_min = np.asarray(pc_cfg["crop_min"], dtype=np.float32)
    crop_max = np.asarray(pc_cfg["crop_max"], dtype=np.float32)

    if args.show_processed and pc_full_cfg.focus_z_lo is not None:
        print("warning: obs config has object_focus (focus_z_lo) enabled, but this "
              "camera-only tool has no live ee_pos to drive it -- that filter is SKIPPED "
              "here (the real pipeline would additionally drop the arm body). Use "
              "`demos.record --show-pointcloud` for a fully faithful view.")

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

    title = f"{args.camera}: raw cloud + crop box (red)"
    if args.show_processed:
        title += "  |  orange = processed ({} pts)".format(pc_full_cfg.max_points)
    vis = o3d.visualization.Visualizer()
    vis.create_window(title + ".  Close window to quit.")
    pcd = o3d.geometry.PointCloud()
    added = False
    vis.add_geometry(_crop_box(crop_min, crop_max, o3d))
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))

    pcd_proc = None
    proc_added = False
    if args.show_processed:
        pcd_proc = o3d.geometry.PointCloud()

    print(f"viewer: {args.camera}  crop_min={crop_min.tolist()}  crop_max={crop_max.tolist()}")
    print("position the camera/objects; tighten crop bounds in the obs config, re-run.")
    if args.show_processed:
        print(f"orange overlay: processed cloud (crop + outlier removal -> "
              f"max_points={pc_full_cfg.max_points}) — exactly what PerceptionPipeline "
              f"would hand the policy for this obs config.")

    try:
        while True:
            depth, rgb, K = cam.get_frame()
            pts_b, valid_b = depth_to_pointcloud(depth[None], K, extrinsic)   # (1, H*W, 3)/(1, H*W)
            pts, valid = pts_b[0], valid_b[0]
            colors = rgb.reshape(-1, 3)[valid].astype(np.float32) / 255.0  # already RGB

            pcd.points = o3d.utility.Vector3dVector(pts[valid])
            pcd.colors = o3d.utility.Vector3dVector(colors)
            if not added:
                vis.add_geometry(pcd)          # add once geometry has points (sets the view)
                added = True
            else:
                vis.update_geometry(pcd)

            if args.show_processed:
                # Same steps + same shared functions PerceptionPipeline.process() calls for
                # the point_cloud modality (crop -> outlier removal -> [focus, skipped here,
                # see warning above] -> subsample) — reused, not reimplemented, so this can't
                # silently drift from what the policy actually receives.
                p_pts, p_valid = depth_to_pointcloud(
                    depth[None], K, extrinsic,
                    depth_min=pc_full_cfg.depth_min, depth_max=pc_full_cfg.depth_max,
                    pixel_sample_n=pc_full_cfg.pixel_sample_n)
                p_pts, p_valid = crop_pointcloud(p_pts, p_valid, crop_min, crop_max)
                if pc_full_cfg.outlier_voxel_size is not None:
                    p_pts, p_valid = remove_outliers_voxel(
                        p_pts, p_valid, pc_full_cfg.outlier_voxel_size,
                        pc_full_cfg.outlier_min_neighbors)
                # focus_object skipped: needs a live ee_pos this camera-only tool doesn't have.
                processed = subsample_pointcloud(p_pts, p_valid, pc_full_cfg.max_points)[0]
                n_proc = int(np.any(processed != 0.0, axis=1).sum())   # drop zero-padding rows

                pcd_proc.points = o3d.utility.Vector3dVector(processed[:n_proc])
                pcd_proc.paint_uniform_color([1.0, 0.55, 0.0])         # orange
                if not proc_added:
                    vis.add_geometry(pcd_proc)
                    proc_added = True
                else:
                    vis.update_geometry(pcd_proc)

            if args.show_crop:
                p = pts[valid]
                inside = np.all((p >= crop_min) & (p <= crop_max), axis=1).sum()
                msg = f"\rpoints: {valid.sum():6d} total   {inside:6d} in-crop"
                if args.show_processed:
                    msg += f"   {n_proc:6d} processed"
                print(msg, end="")

            if not vis.poll_events():           # window closed
                break
            vis.update_renderer()
    finally:
        print()
        vis.destroy_window()
        cam.stop()


if __name__ == "__main__":
    main()
