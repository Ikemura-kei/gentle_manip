#!/usr/bin/env python3
"""Detect the table ArUco (5x5 dict, id 1, 80 mm) in the external camera: corners in pixels, camera frame
(PnP and depth) and robot frame via the current WORLD_T_CAM_EXT; saves an annotated PNG. Camera only.

    uv run --project envs/deploy python -m gentle_manip.diagnostics.aruco_check [--live]
"""
from __future__ import annotations

import argparse
import datetime

import cv2
import numpy as np

from gentle_manip.diagnostics import calibration as cal
from gentle_manip.robot import xarm7_config as cfg

MARKER_M = 0.080
OBJ = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]], float) * MARKER_M / 2   # cv2 corner order


def detect(bgr, depth_m, K, dist, marker_id, dictionary):
    prm = cv2.aruco.DetectorParameters(); prm.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX   # sub-pixel corners
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dictionary), prm)
    corners, ids, _ = det.detectMarkers(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    if ids is None or marker_id not in ids.ravel():
        return None
    c = corners[list(ids.ravel()).index(marker_id)][0]                      # (4,2) px
    ok, rvec, tvec = cv2.solvePnP(OBJ, c, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    R = cv2.Rodrigues(rvec)[0]; pnp_cam = (R @ OBJ.T).T + tvec.ravel()      # (4,3) camera frame
    dep = []
    for (u, v) in c:                                                        # depth-backprojected corner
        ui, vi = int(round(u)), int(round(v)); z = float(np.median(depth_m[max(vi-1,0):vi+2, max(ui-1,0):ui+2]))
        dep.append([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z])
    return c, pnp_cam, np.array(dep), tvec.ravel()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cam-serial", default="335522071488")
    p.add_argument("--id", type=int, default=1)
    p.add_argument("--dict", default="DICT_5X5_50")
    p.add_argument("--frames", type=int, default=15, help="frames to warm up / median over")
    p.add_argument("--live", action="store_true", help="keep showing the detection until ESC")
    p.add_argument("--ref", type=float, nargs=4, action="append", default=None, metavar=("IDX", "X", "Y", "Z"),
                   help="a corner measured with the robot TCP (index 0-3 as labelled, robot-frame m); prints its "
                        "residual through the current extrinsic. Repeatable.")
    a = p.parse_args()
    pipe, align, K, dist = cal._start_realsense(a.cam_serial)
    sc = pipe.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
    T = np.asarray(cfg.WORLD_T_CAM_EXT, float)
    try:
        while True:
            hits = []
            for _ in range(a.frames):
                f = align.process(pipe.wait_for_frames()); bgr = np.asarray(f.get_color_frame().get_data())
                depth = np.asarray(f.get_depth_frame().get_data()).astype(np.float32) * sc
                r = detect(bgr, depth, K, dist, a.id, getattr(cv2.aruco, a.dict))
                if r is not None:
                    hits.append(r)
            vis = bgr.copy()
            if not hits:
                print("marker not detected"); cv2.putText(vis, f"id {a.id} NOT detected", (10, 24), 0, 0.7, (0, 0, 255), 2)
            else:
                c = np.median([h[0] for h in hits], axis=0); pnp = np.median([h[1] for h in hits], axis=0)
                dep = np.median([h[2] for h in hits], axis=0)
                base_pnp = (T[:3, :3] @ pnp.T).T + T[:3, 3]; base_dep = (T[:3, :3] @ dep.T).T + T[:3, 3]
                print(f"marker id {a.id} ({a.dict}, {1e3*MARKER_M:.0f} mm) — {len(hits)}/{a.frames} frames; distance {np.linalg.norm(np.median([h[3] for h in hits],axis=0)):.3f} m")
                print(f"  side lengths (PnP, mm): {np.round([1e3*np.linalg.norm(pnp[i]-pnp[(i+1)%4]) for i in range(4)],1)}   depth-based: {np.round([1e3*np.linalg.norm(dep[i]-dep[(i+1)%4]) for i in range(4)],1)}")
                print("  corner  px(u,v)        cam PnP (m)                cam depth (m)              base PnP (m)  [current extrinsic — obsolete after the move]")
                for i, name in enumerate(["top-left", "top-right", "bottom-right", "bottom-left"]):
                    print(f"  {name:12s} ({c[i,0]:5.1f},{c[i,1]:5.1f})  {np.round(pnp[i],4)}  {np.round(dep[i],4)}  {np.round(base_pnp[i],4)}")
                    cv2.circle(vis, tuple(np.round(c[i]).astype(int)), 5, (0, 255, 0), 2)
                    cv2.putText(vis, f"{i}:{name}", tuple(np.round(c[i]).astype(int) + [6, -6]), 0, 0.45, (0, 255, 0), 1)
                print(f"  base-frame z of corners (PnP): {np.round(1e3*base_pnp[:,2],1)} mm   (depth): {np.round(1e3*base_dep[:,2],1)} mm")
                for ref in (a.ref or []):
                    i, meas = int(ref[0]), np.asarray(ref[1:]); err = (base_pnp[i] - meas) * 1e3
                    print(f"  REF corner {i}: measured {np.round(meas,4)}  camera->base {np.round(base_pnp[i],4)}  "
                          f"error {np.round(err,1)} mm  |{np.linalg.norm(err):.1f} mm|  (camera minus robot)")
            if not a.live:                                      # one capture: save the annotated frame
                stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                out = cal.DEFAULT_OUT / "eye-to-hand" / f"aruco_check_{stamp}.png"; cv2.imwrite(str(out), vis); print(f"  saved {out}")
                break
            cv2.imshow("aruco_check", vis)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break
    finally:
        pipe.stop(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
