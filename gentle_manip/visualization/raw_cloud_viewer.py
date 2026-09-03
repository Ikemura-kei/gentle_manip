"""Standalone LIVE raw point-cloud viewer — camera only, the robot is never touched.

No arm connection, no PolicyEnv, no obs config: this opens the RealSense directly, backprojects
depth with the SAME pinhole math the perception pipeline uses (`depth_to_pointcloud`), and shows
the cloud in Open3D. Use it for camera placement, checking coverage/noise, and for judging what
the depth sensor actually sees before any cropping or filtering happens.

Two frames are available:
  --frame camera   (default) raw camera frame, +z = away from the lens. Nothing but the sensor.
  --frame world    apply the setup's world_T_cam so the cloud sits in robot-base coordinates
                   (useful to sanity-check the extrinsic; still no robot connection).

L515 short-range preset (user request 2026-09-03): `--short-range` sets the depth sensor's
rs2_l500_visual_preset to SHORT_RANGE, which is the right profile for <1 m tabletop work —
noticeably fewer flying pixels / less edge noise on close objects than the default preset.
Other presets via `--visual-preset <name>` (see the error message for the device's list).

    uv run --project envs/deploy python -m gentle_manip.visualization.raw_cloud_viewer --short-range
    uv run --project envs/deploy python -m gentle_manip.visualization.raw_cloud_viewer \
        --frame world --show-axes --max-depth 0.9

Keys: Open3D's own mouse controls; q / ESC closes the window.
"""
import argparse
from pathlib import Path

import numpy as np
import yaml

import gentle_manip
from gentle_manip.perception.depth_to_pointcloud import depth_to_pointcloud
from gentle_manip.robot import xarm7_config as cfg

_PKG = Path(gentle_manip.__file__).parent


def _resolve(path: Path) -> Path:
    if path.is_file():
        return path
    alt = _PKG.parent / path
    if alt.is_file():
        return alt
    raise FileNotFoundError(f"config not found: {path} (also tried {alt})")


def main() -> None:
    p = argparse.ArgumentParser(description="Live RAW point cloud from the camera (no robot)")
    p.add_argument("--setup", type=Path, default=_PKG / "configs" / "setup" / "real_lab.yaml",
                   help="only used for the camera serial/resolution (+ extrinsic with --frame world)")
    p.add_argument("--camera", default="cam_ext")
    p.add_argument("--short-range", action="store_true",
                   help="L515: set the SHORT_RANGE visual preset (best for <1 m tabletop work)")
    p.add_argument("--visual-preset", default=None,
                   help="explicit rs2_l500_visual_preset name (overrides --short-range), e.g. "
                        "default / no_ambient_light / low_ambient_light / max_range / short_range")
    p.add_argument("--frame", choices=("camera", "world"), default="camera",
                   help="camera = raw sensor frame (default); world = apply the setup extrinsic")
    p.add_argument("--min-depth", type=float, default=0.1, help="metres; drop nearer points")
    p.add_argument("--max-depth", type=float, default=None,
                   help="metres; drop farther points (default: the setup's depth_max)")
    p.add_argument("--show-axes", action="store_true", help="draw a 10 cm coordinate frame")
    p.add_argument("--point-size", type=float, default=1.5)
    args = p.parse_args()

    import open3d as o3d
    from gentle_manip.envs.realsense_camera import RealSenseCamera

    setup = yaml.safe_load(open(_resolve(args.setup)))
    cam_cfg = setup["cameras"][args.camera]
    preset = args.visual_preset or ("short_range" if args.short_range else None)
    max_depth = args.max_depth if args.max_depth is not None else float(
        cam_cfg.get("depth_max", 0.85))

    cam = RealSenseCamera(
        name=args.camera, serial=cam_cfg["serial"],
        width=cam_cfg.get("width", 640), height=cam_cfg.get("height", 480),
        depth_min=args.min_depth, depth_max=max_depth,
        fps=cam_cfg.get("fps", 30), visual_preset=preset)
    cam.start()
    print(f"[raw-cloud] {args.camera} serial={cam_cfg['serial']} frame={args.frame} "
          f"depth {args.min_depth:.2f}-{max_depth:.2f} m"
          + (f" preset={preset}" if preset else " preset=device default"))

    # depth_to_pointcloud applies the extrinsic itself; identity => raw camera frame.
    if args.frame == "world":
        extrinsics = np.asarray(cam_cfg.get("world_T_cam", cfg.WORLD_T_CAM_EXT), dtype=np.float32)
    else:
        extrinsics = np.eye(4, dtype=np.float32)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"raw cloud — {args.camera} ({args.frame} frame)", width=1280,
                      height=800)
    vis.get_render_option().point_size = float(args.point_size)
    pcd = o3d.geometry.PointCloud()
    added = False
    n = 0
    if args.show_axes:
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))

    try:
        while True:
            depth, rgb_img, K = cam.get_frame()             # (H,W) m, (H,W,3) uint8 RGB, (3,3)
            pts, valid = depth_to_pointcloud(depth[None], K, extrinsics,
                                             depth_min=args.min_depth, depth_max=max_depth)
            m = valid[0]                                    # (H*W,) same order as the pixels
            pts = pts[0][m]
            # colour every point with its own pixel — the mask indexes both identically
            rgb = rgb_img.reshape(-1, 3)[m].astype(np.float32) / 255.0

            n += 1
            if n % 60 == 1:
                print(f"[raw-cloud] frame {n}: {len(pts)} valid points "
                      f"({100.0 * m.mean():.0f}% of pixels)", flush=True)

            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(rgb)
            if not added:
                # only seed once there ARE points — adding an empty cloud gives Open3D a
                # degenerate bounding box and the camera never frames the scene afterwards
                if len(pts):
                    vis.add_geometry(pcd)
                    added = True
            else:
                vis.update_geometry(pcd)
            if not vis.poll_events():
                break
            vis.update_renderer()
    except KeyboardInterrupt:
        pass
    finally:
        vis.destroy_window()
        cam.stop()
        print("[raw-cloud] stopped")


if __name__ == "__main__":
    main()
