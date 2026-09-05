#!/usr/bin/env python3
"""Camera DRIFT check against the pinned reference frame (dataset/camera_calibration/reference/aruco_ref.npz).
Camera only, extrinsic-independent: re-detects the table ArUco and reports how far its corners moved in
pixels and in the camera frame, plus the rigid camera motion that explains it. You judge the threshold.

    uv run --project envs/deploy python -m gentle_manip.diagnostics.drift_check          # check
    uv run --project envs/deploy python -m gentle_manip.diagnostics.drift_check --pin    # after a recalibration: make today's frame the reference
"""
from __future__ import annotations

import argparse
import datetime
import shutil

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from gentle_manip.diagnostics import aruco_check, calibration as cal

REF_DIR = cal.DEFAULT_OUT / "reference"
REF = REF_DIR / "aruco_ref.npz"


def capture(serial, frames):
    pipe, align, K, dist = cal._start_realsense(serial)
    sc = pipe.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
    hits, bgr = [], None
    try:
        for _ in range(frames):
            f = align.process(pipe.wait_for_frames()); bgr = np.asarray(f.get_color_frame().get_data())
            depth = np.asarray(f.get_depth_frame().get_data()).astype(np.float32) * sc
            r = aruco_check.detect(bgr, depth, K, dist, 1, cv2.aruco.DICT_5X5_50)
            if r is not None:
                hits.append(r)
    finally:
        pipe.stop()
    if not hits:
        return None, bgr
    return (np.median([h[0] for h in hits], 0), np.median([h[1] for h in hits], 0), len(hits)), bgr


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cam-serial", default=None, help="default: the reference's serial")
    p.add_argument("--frames", type=int, default=15)
    p.add_argument("--pin", action="store_true", help="overwrite the reference with this capture (keeps ref_tcp / park)")
    a = p.parse_args()
    ref = dict(np.load(REF, allow_pickle=True)); serial = a.cam_serial or str(ref["cam_serial"])
    det, bgr = capture(serial, a.frames)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if det is None:
        print("ArUco NOT detected — is the marker in view (640x480 is a centre crop of the sensor)?"); return
    px, cam, n = det
    dpx = px - ref["px_corners"]; dcam = (cam - ref["cam_corners"]) * 1e3
    Z = float(np.linalg.norm(ref["cam_corners"].mean(0))); fx = 607.0                       # marker range, focal (px)
    lateral_mm = np.linalg.norm(dpx, axis=1) * Z / fx * 1e3
    R, _ = Rotation.align_vectors(cam - cam.mean(0), ref["cam_corners"] - ref["cam_corners"].mean(0))
    print(f"reference: {ref['date']} ({ref['source_round']}), marker id {int(ref['marker_id'])} at {Z:.2f} m — today: {n}/{a.frames} frames")
    print("  corner drift in the image:   " + "  ".join(f"{i}:({d[0]:+.1f},{d[1]:+.1f})px" for i, d in enumerate(dpx))
          + f"   max {np.abs(dpx).max():.1f} px = {lateral_mm.max():.1f} mm lateral   <- the number to judge (noise ~0.5 px / 0.6 mm)")
    print("  PnP corners, camera frame:   " + "  ".join(f"{i}:{np.linalg.norm(d):.1f}mm" for i, d in enumerate(dcam))
          + f"   (depth along the ray is +-5 mm at this range: informative only if >> 5 mm)   rotation {np.degrees(np.linalg.norm(R.as_rotvec())):.2f} deg")
    vis = bgr.copy()
    for i in range(4):
        cv2.circle(vis, tuple(np.round(ref["px_corners"][i]).astype(int)), 6, (255, 0, 0), 1)     # blue = reference
        cv2.circle(vis, tuple(np.round(px[i]).astype(int)), 4, (0, 255, 0), 2)                    # green = now
    out = REF_DIR / f"drift_{stamp}.png"; cv2.imwrite(str(out), vis); print(f"  saved {out}  (blue = reference, green = now)")
    if a.pin:
        shutil.copy(REF, REF_DIR / f"aruco_ref_superseded_{stamp}.npz")
        ref.update(date=stamp[:10], px_corners=px, cam_corners=cam); np.savez(REF, **ref)
        cv2.imwrite(str(REF_DIR / "aruco_ref.png"), bgr); print(f"  PINNED: reference updated (old one kept as aruco_ref_superseded_{stamp}.npz)")


if __name__ == "__main__":
    main()
