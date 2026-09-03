"""Standalone LIVE raw point-cloud viewer — camera only, the robot is never touched.

No arm connection, no PolicyEnv, no obs config: this opens the RealSense directly, backprojects
depth with the SAME pinhole math the perception pipeline uses (`depth_to_pointcloud`), and shows
the cloud in Open3D. Use it for camera placement, checking coverage/noise, and for judging what
the depth sensor actually sees before any cropping happens.

VOXEL OUTLIER FILTER (2026-09-03, for the new ~45-degrees-downward camera placement): the
production denoiser `pointcloud_ops.remove_outliers_voxel` — the SAME function the shared
perception pipeline runs — is applied live to the raw cloud so its behaviour can be judged
directly. Surviving points draw grey, removed points draw RED (so you see what it culls, not
just what it keeps). Both parameters are tunable WITHOUT restarting, which is the point of this
tool: the filter's aggressiveness depends on point DENSITY, and tilting the camera changes
density across the scene (near tabletop vs far), so the old horizontal-placement values are not
guaranteed to carry over.

  F     filter on/off (A/B against the unfiltered cloud)
  R     show/hide the removed points
  [ ]   min_neighbors -/+ 1
  - =   voxel size -/+ 2 mm

Defaults are the production values (`voxel_size=0.01`, `min_neighbors=23`, from
`configs/obs/point_cloud_1cam_armfocus.yaml`). NOTE those were tuned on the full-resolution
cropped cloud; this viewer is also full-resolution but UNCROPPED, so densities are comparable
and the numbers transfer — but any value you settle on here should be confirmed in the real
pipeline before training on it. Camera RGB is deliberately not used: colour encodes filter
decisions instead.

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
from gentle_manip.perception.pointcloud_ops import remove_outliers_voxel
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
    # ── voxel outlier filter (the production denoiser, run here on the RAW cloud) ──────────
    p.add_argument("--voxel-size", type=float, default=0.01,
                   help="voxel edge (m) for remove_outliers_voxel; production value 0.01")
    p.add_argument("--min-neighbors", type=int, default=23,
                   help="min valid points per voxel to survive; production value 23")
    p.add_argument("--no-filter", action="store_true",
                   help="start with the filter OFF (toggle live with F)")
    p.add_argument("--hide-removed", action="store_true",
                   help="start with removed points hidden instead of drawn red (toggle with R)")
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

    # Live-tunable filter state (mutated by the key callbacks below).
    st = {"on": not args.no_filter, "voxel": float(args.voxel_size),
          "minn": int(args.min_neighbors), "show_removed": not args.hide_removed}

    def _report():
        print(f"[filter] {'ON ' if st['on'] else 'OFF'} voxel={st['voxel'] * 1000:.0f} mm "
              f"min_neighbors={st['minn']} removed={'shown' if st['show_removed'] else 'hidden'}",
              flush=True)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"raw cloud — {args.camera} ({args.frame} frame)", width=1280,
                      height=800)

    def _key(k, fn):
        def cb(_vis):
            fn()
            _report()
            return False        # False = don't force a redraw; the frame loop already does
        vis.register_key_callback(ord(k), cb)

    _key("F", lambda: st.__setitem__("on", not st["on"]))
    _key("R", lambda: st.__setitem__("show_removed", not st["show_removed"]))
    _key("]", lambda: st.__setitem__("minn", st["minn"] + 1))
    _key("[", lambda: st.__setitem__("minn", max(1, st["minn"] - 1)))
    _key("=", lambda: st.__setitem__("voxel", round(st["voxel"] + 0.002, 4)))
    _key("-", lambda: st.__setitem__("voxel", max(0.002, round(st["voxel"] - 0.002, 4))))

    opt = vis.get_render_option()
    opt.point_size = float(args.point_size)
    opt.background_color = np.array([0.05, 0.05, 0.07])     # dark: grey/red points read clearly
    pcd = o3d.geometry.PointCloud()
    added = False
    n = 0
    if args.show_axes:
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))

    KEPT = np.array([0.62, 0.68, 0.78], np.float32)         # cool grey — surviving points
    CUT = np.array([0.90, 0.22, 0.18], np.float32)          # red — what the filter removes
    print("[raw-cloud] keys: F=filter on/off  R=show/hide removed  [ ]=min_neighbors  "
          "- ==voxel size  Q/ESC=quit")
    _report()

    try:
        while True:
            depth, _rgb, K = cam.get_frame()                # RGB deliberately unused (geometry only)
            pts, valid = depth_to_pointcloud(depth[None], K, extrinsics,
                                             depth_min=args.min_depth, depth_max=max_depth)
            if st["on"]:
                # the SHARED production denoiser, on the raw cloud, in whichever frame is active
                _, kept = remove_outliers_voxel(pts, valid, st["voxel"], st["minn"])
            else:
                kept = valid
            keep_m = kept[0]
            cut_m = valid[0] & ~keep_m                      # valid but filtered out

            xyz = pts[0][keep_m]
            col = np.tile(KEPT, (len(xyz), 1))
            if st["show_removed"] and cut_m.any():
                xyz = np.concatenate([xyz, pts[0][cut_m]])
                col = np.concatenate([col, np.tile(CUT, (int(cut_m.sum()), 1))])

            n += 1
            if n % 60 == 1:
                nv = int(valid[0].sum())
                print(f"[raw-cloud] frame {n}: valid={nv} kept={int(keep_m.sum())} "
                      f"removed={int(cut_m.sum())} "
                      f"({100.0 * cut_m.sum() / max(nv, 1):.1f}% of valid)", flush=True)

            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(col)
            if not added:
                # only seed once there ARE points — adding an empty cloud gives Open3D a
                # degenerate bounding box and the camera never frames the scene afterwards
                if len(xyz):
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
