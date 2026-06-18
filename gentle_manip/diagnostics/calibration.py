#!/usr/bin/env python3
"""
Collect hand-eye calibration data using a ChAruco board.

Board spec (calib.io PDF — must match the physical printout):
  6x6 squares, checker size 20 mm, marker size 15 mm, AruCo DICT_4X4

The board is clamped in the robot's parallel-jaw gripper (close it manually
before running). Move the arm to diverse poses while in joint-teaching mode.
Press 'c' at each pose to capture; 'q' to quit and save.

Modes
-----
eye-to-hand  (default) — L515 is world-fixed, board rides on the gripper.
                          Solves for WORLD_T_CAM_EXT (T_base→cam).
eye-in-hand             — D405 is wrist-mounted, board is fixed in the workspace.
                          Solves for EE_T_CAM_WRIST (T_ee→cam).

Usage
-----
    uv run --project envs/deploy python gentle_manip/diagnostics/calibration.py
    uv run --project envs/deploy python gentle_manip/diagnostics/calibration.py \\
        --mode eye-in-hand --cam-serial 230322271104
"""

from __future__ import annotations

import argparse
import datetime
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from xarm.wrapper import XArmAPI

# ── ChAruco board parameters ──────────────────────────────────────────────────
# Must match the printed board exactly.
SQUARES_X     = 6
SQUARES_Y     = 6
SQUARE_LEN    = 0.020   # meters  (20 mm checker size)
MARKER_LEN    = 0.015   # meters  (15 mm marker size)
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50

# ── Camera serials ────────────────────────────────────────────────────────────
SERIAL_CAM_EXT   = "f1120484"      # L515  — world-fixed  (eye-to-hand)
SERIAL_CAM_WRIST = "230322271104"  # D405  — wrist-mounted (eye-in-hand)

STREAM_W   = 640
STREAM_H   = 480
STREAM_FPS = 30

# ── Output ────────────────────────────────────────────────────────────────────
_REPO_ROOT     = Path(__file__).parent.parent.parent
DEFAULT_OUT    = _REPO_ROOT / "dataset" / "camera_calibration"


# ── Board / detector helpers ──────────────────────────────────────────────────

def _make_board() -> Tuple[cv2.aruco.CharucoBoard, cv2.aruco.ArucoDetector]:
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    board      = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), SQUARE_LEN, MARKER_LEN, aruco_dict
    )
    # calib.io boards use the pre-OpenCV-4.8 corner ordering; must match.
    board.setLegacyPattern(True)
    # ArucoDetector is used separately only for the debug marker overlay.
    # Actual ChAruco corner detection uses CharucoDetector (see _make_charuco_detector).
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    return board, aruco_detector


def _make_charuco_detector(
    board: cv2.aruco.CharucoBoard,
    K:     np.ndarray,
    dist:  np.ndarray,
) -> cv2.aruco.CharucoDetector:
    params = cv2.aruco.CharucoParameters()
    params.cameraMatrix = K
    params.distCoeffs   = dist
    return cv2.aruco.CharucoDetector(board, params)


# ── Camera helpers ────────────────────────────────────────────────────────────

def _start_realsense(serial: str):
    """Start pipeline; return (pipeline, align, K, dist)."""
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    cfg      = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.depth, STREAM_W, STREAM_H, rs.format.z16,  STREAM_FPS)
    cfg.enable_stream(rs.stream.color, STREAM_W, STREAM_H, rs.format.bgr8, STREAM_FPS)

    profile = pipeline.start(cfg)
    align   = rs.align(rs.stream.color)

    # Read intrinsics from first frame
    frames = pipeline.wait_for_frames()
    frames = align.process(frames)
    intr   = frames.get_color_frame().profile.as_video_stream_profile().intrinsics
    K      = np.array(
        [[intr.fx, 0.,       intr.ppx],
         [0.,       intr.fy, intr.ppy],
         [0.,       0.,      1.      ]],
        dtype=np.float64,
    )
    # RealSense streams are pre-rectified; distortion is nominally zero.
    dist = np.zeros(5, dtype=np.float64)
    return pipeline, align, K, dist


def _get_bgr(pipeline, align) -> np.ndarray:
    frames = pipeline.wait_for_frames()
    frames = align.process(frames)
    return np.asarray(frames.get_color_frame().get_data())  # (H,W,3) BGR


# ── Board detection ───────────────────────────────────────────────────────────

def _detect(
    bgr:              np.ndarray,
    gray:             np.ndarray,
    board:            cv2.aruco.CharucoBoard,
    charuco_detector: cv2.aruco.CharucoDetector,
    aruco_detector:   cv2.aruco.ArucoDetector,
    K:                np.ndarray,
    dist:             np.ndarray,
    debug:            bool = False,
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Detect ChAruco board and estimate pose.

    Returns (success, rvec, tvec, charuco_corners, marker_corners, marker_ids).
    marker_corners/marker_ids come from the separate ArucoDetector for overlay.
    """
    # Run ArucoDetector independently just for the overlay
    marker_corners, marker_ids, _ = aruco_detector.detectMarkers(gray)

    # Run CharucoDetector (with K/dist baked into params) for corner interpolation
    charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(bgr)

    if debug:
        nm = len(marker_ids)   if marker_ids      is not None else 0
        nc = len(charuco_corners) if charuco_corners is not None else 0
        print(f"  [debug] aruco markers={nm}  charuco_corners={nc}"
              f"  corners_type={type(charuco_corners)}  ids_type={type(charuco_ids)}")
        if charuco_corners is not None and nc > 0:
            print(f"  [debug] corners shape={charuco_corners.shape}")

    if charuco_corners is None or charuco_ids is None or len(charuco_ids) < 6:
        return False, None, None, charuco_corners, marker_corners, marker_ids

    obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
    # solvePnP with DLT needs at least 6 point correspondences.
    if obj_pts is None or len(obj_pts) < 6:
        return False, None, None, charuco_corners, marker_corners, marker_ids

    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist)
    if not ok:
        return False, None, None, charuco_corners, marker_corners, marker_ids

    return True, rvec.flatten(), tvec.flatten(), charuco_corners, marker_corners, marker_ids


# ── Robot helpers ─────────────────────────────────────────────────────────────

def _get_ee_R_t(arm: XArmAPI) -> Tuple[np.ndarray, np.ndarray]:
    """Return (R_grip2base, t_grip2base) in meters from SDK get_position_aa."""
    _, aa = arm.get_position_aa(is_radian=True)
    aa = np.asarray(aa, dtype=np.float64)
    t = aa[:3] * 1e-3                          # mm → m
    R = Rotation.from_rotvec(aa[3:]).as_matrix()
    return R, t


# ── Interactive collection loop ───────────────────────────────────────────────

def _collect(
    arm:              XArmAPI,
    pipeline,
    align,
    K:                np.ndarray,
    dist:             np.ndarray,
    board:            cv2.aruco.CharucoBoard,
    charuco_detector: cv2.aruco.CharucoDetector,
    aruco_detector:   cv2.aruco.ArucoDetector,
    mode:             str,
    debug:            bool = False,
) -> Dict[str, List]:
    data: Dict[str, List] = {
        "R_gripper2base": [],
        "t_gripper2base": [],
        "R_board2cam":    [],
        "t_board2cam":    [],
    }

    print("\n=== ChAruco hand-eye calibration — data collection ===")
    print(f"  mode  : {mode}")
    print(f"  board : {SQUARES_X}×{SQUARES_Y}  "
          f"checker {SQUARE_LEN*1e3:.0f} mm  marker {MARKER_LEN*1e3:.0f} mm  DICT_4X4")
    print("\n  c — capture   d — delete last   s — save frame to disk   q — quit & save\n")

    cv2.namedWindow("ChAruco hand-eye", cv2.WINDOW_NORMAL)

    while True:
        bgr  = _get_bgr(pipeline, align)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        ok, rvec, tvec, charuco_corners, marker_corners, marker_ids = \
            _detect(bgr, gray, board, charuco_detector, aruco_detector, K, dist, debug)
        n   = len(data["R_gripper2base"])
        vis = bgr.copy()

        n_markers  = len(marker_ids)  if marker_ids  is not None else 0
        n_charuco  = len(charuco_corners) if charuco_corners is not None else 0

        if ok:
            # Only draw axes when all projected endpoints are inside the frame
            # to avoid the solvepnp.cpp "endpoints out of frame" warning spam.
            h_img, w_img = vis.shape[:2]
            axis_pts, _ = cv2.projectPoints(
                np.float32([[0,0,0],[SQUARE_LEN*2,0,0],[0,SQUARE_LEN*2,0],[0,0,-SQUARE_LEN*2]]),
                rvec, tvec, K, dist,
            )
            axis_pts = axis_pts.reshape(-1, 2)
            if (axis_pts[:, 0].min() >= 0 and axis_pts[:, 0].max() < w_img and
                    axis_pts[:, 1].min() >= 0 and axis_pts[:, 1].max() < h_img):
                cv2.drawFrameAxes(vis, K, dist, rvec, tvec, SQUARE_LEN * 2)
            if charuco_corners is not None:
                cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners)
            label = f"Board OK  [{n} captured]"
            color = (0, 220, 0)
        else:
            # Draw whatever partial detections exist so the user can see what's happening
            if marker_corners:
                cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)
            if charuco_corners is not None:
                cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners)
            if n_markers == 0:
                label = f"No ArUco markers detected  [{n} captured]"
            elif n_charuco < 6:
                label = f"Markers:{n_markers}  ChAruco corners:{n_charuco}/6  [{n} captured]"
            else:
                label = f"solvePnP failed ({n_charuco} corners)  [{n} captured]"
            color = (0, 0, 220)

        cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("ChAruco hand-eye", vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('c'):
            if not ok:
                print(f"  [skip] board not detected  (markers={n_markers}, charuco_corners={n_charuco})")
                continue
            R, t = _get_ee_R_t(arm)
            data["R_gripper2base"].append(R)
            data["t_gripper2base"].append(t)
            data["R_board2cam"].append(Rotation.from_rotvec(rvec).as_matrix())
            data["t_board2cam"].append(tvec)
            print(f"  [{n+1:2d}] t_grip2base = {np.round(t, 4)}")

        elif key == ord('s'):
            p = Path(f"/tmp/charuco_frame_{int(time.time())}.png")
            cv2.imwrite(str(p), bgr)
            print(f"  [saved] {p}  (markers={n_markers}, charuco_corners={n_charuco})")

        elif key == ord('d'):
            if data["R_gripper2base"]:
                for k in data:
                    data[k].pop()
                print(f"  [{n-1:2d}] deleted last capture")
            else:
                print("  [skip] nothing to delete")

    cv2.destroyAllWindows()
    return data


# ── Hand-eye solver ───────────────────────────────────────────────────────────

def _calibrate(data: Dict[str, List], mode: str) -> np.ndarray:
    """Run cv2.calibrateHandEye and return the solved 4×4 transform.

    eye-to-hand : output is T_base→cam  (= WORLD_T_CAM_EXT)
    eye-in-hand : output is T_ee→cam    (= EE_T_CAM_WRIST)
    """
    R_g2b = data["R_gripper2base"]
    t_g2b = [t.reshape(3, 1) for t in data["t_gripper2base"]]
    R_b2c = data["R_board2cam"]
    t_b2c = [t.reshape(3, 1) for t in data["t_board2cam"]]

    if mode == "eye-in-hand":
        # Camera on gripper, board fixed.
        # calibrateHandEye(R_g2b, t_g2b, R_b2c, t_b2c) → T_cam2gripper = T_ee→cam
        R, t = cv2.calibrateHandEye(
            R_g2b, t_g2b, R_b2c, t_b2c,
            method=cv2.CALIB_HAND_EYE_TSAI,
        )
        label = "EE_T_CAM_WRIST  (T_ee→cam)"
    else:
        # Camera fixed, board on gripper.
        # Invert robot transform: pass T_base→gripper so the output is T_base→cam.
        R_b2g = [R.T              for R in R_g2b]
        t_b2g = [-R.T @ t         for R, t in zip(R_g2b, t_g2b)]
        R, t = cv2.calibrateHandEye(
            R_b2g, t_b2g, R_b2c, t_b2c,
            method=cv2.CALIB_HAND_EYE_TSAI,
        )
        label = "WORLD_T_CAM_EXT  (T_base→cam)"

    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = t.flatten()

    print(f"\n--- {label} ---")
    print(T)
    print("\nPaste into gentle_manip/robot/xarm7_config.py:")
    rows = ",\n".join(
        f"    [{', '.join(f'{v:15.8f}' for v in row)}]" for row in T
    )
    print(f"[\n{rows},\n]")
    return T


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect ChAruco hand-eye calibration data for XArm7."
    )
    parser.add_argument("--robot-ip",    default="192.168.1.241")
    parser.add_argument("--mode",        default="eye-to-hand",
                        choices=["eye-to-hand", "eye-in-hand"])
    parser.add_argument("--cam-serial",  default=SERIAL_CAM_EXT,
                        help="RealSense serial (default: L515 for eye-to-hand)")
    parser.add_argument("--output-dir",  default=str(DEFAULT_OUT))
    parser.add_argument("--no-calibrate", action="store_true",
                        help="Skip running the hand-eye solver after collection")
    parser.add_argument("--debug", action="store_true",
                        help="Print per-frame detection details to stdout")
    args = parser.parse_args()

    # ── Robot setup ────────────────────────────────────────────────────────────
    print(f"Connecting to XArm at {args.robot_ip} …")
    arm = XArmAPI(args.robot_ip, is_radian=True)
    arm.motion_enable(enable=True)
    arm.set_mode(2)   # joint-teaching: free-drive
    time.sleep(0.5)
    arm.set_state(0)
    time.sleep(0.5)
    print("XArm in joint-teaching mode — you can move the arm freely.")

    # ── Camera setup ───────────────────────────────────────────────────────────
    print(f"Starting RealSense {args.cam_serial} …")
    pipeline, align, K, dist = _start_realsense(args.cam_serial)

    board, aruco_detector = _make_board()
    charuco_detector      = _make_charuco_detector(board, K, dist)

    # ── Collect ────────────────────────────────────────────────────────────────
    try:
        data = _collect(arm, pipeline, align, K, dist,
                        board, charuco_detector, aruco_detector,
                        args.mode, debug=args.debug)
    finally:
        pipeline.stop()
        arm.disconnect()

    n = len(data["R_gripper2base"])
    print(f"\nCollected {n} pose(s).")

    if n < 3:
        print("Need ≥ 3 poses to calibrate — exiting without saving.")
        return

    # ── Save raw data ──────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir) / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)
    ts       = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    raw_path = out_dir / f"charuco_hand_eye_{ts}.npz"

    np.savez(
        str(raw_path),
        **{k: np.stack(v) for k, v in data.items()},
        camera_matrix=K,
        dist_coeffs=dist,
    )
    print(f"Saved raw data  → {raw_path}")

    # ── Calibrate and save result ──────────────────────────────────────────────
    if not args.no_calibrate:
        T = _calibrate(data, args.mode)
        res_path = out_dir / f"charuco_hand_eye_result_{ts}.npz"
        np.savez(str(res_path), T=T, camera_matrix=K)
        print(f"Saved result    → {res_path}")


if __name__ == "__main__":
    main()
