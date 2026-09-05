#!/usr/bin/env python3
"""Replay a recorded eye-to-hand calibration round: drive the arm through the stored TCP poses,
capture the ChAruco board at each, save a new round, solve it (calib_select), optionally check the
table plane from a high park pose. For re-calibrating after the camera was moved.

Gentle by design: Cartesian moves in position mode capped by --speed/--mvacc, a --dwell pause at
every pose before capturing, and ESC on the live window stops the arm (set_state 4) at any moment.
Dry run (default) only lists what would happen; --live moves the robot.

    uv run --project envs/deploy python -m gentle_manip.diagnostics.calib_replay                # dry run
    uv run --project envs/deploy python -m gentle_manip.diagnostics.calib_replay --live --table-check --table-z 0.0138

Sequence: lift straight up to --start-z (0.35 m: board out of view) -> reference frame + table ArUco
(camera-frame corners saved as reference_<stamp>_aruco_cam.npy for the later correction fit) -> the stored
poses -> back to the start pose -> calib_select (+ table check). Check a solved extrinsic against the
TCP-measured corner with `aruco_check.py --ref`.
"""
from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from gentle_manip.diagnostics import calibration as cal
from gentle_manip.diagnostics import aruco_check
from gentle_manip.robot import xarm7_config as cfg

ROUND_DIR = cal.DEFAULT_OUT / "eye-to-hand"
ESC = 27


def _newest_round() -> Path:
    raws = [p for p in ROUND_DIR.glob("charuco_hand_eye_*.npz")
            if "_result" not in p.name and "_selected" not in p.name]
    return max(raws, key=lambda p: p.stat().st_mtime)


def _load_poses(path: Path, drop, dz: float):
    """(R (N,3,3), t (N,3), keep_idx) after dropping outliers, raising by dz, EE-box filtering."""
    z = np.load(path); R, t = z["R_gripper2base"], z["t_gripper2base"].copy()
    sel = path.with_name(path.stem + "_selected.npz")
    if drop is None and sel.exists():
        drop = np.nonzero(~np.load(sel)["inliers"])[0].tolist()
    drop = set(drop or [])
    t[:, 2] += dz
    lo, hi = np.asarray(cfg.EE_BOUNDS_MIN), np.asarray(cfg.EE_BOUNDS_MAX)
    keep = [i for i in range(len(t)) if i not in drop]
    out = []
    for i in keep:
        inside = bool(np.all(t[i] >= lo) and np.all(t[i] <= hi))
        print(f"  pose {i:2d}: t={np.round(t[i], 3)}  rpy={np.round(Rotation.from_matrix(R[i]).as_euler('xyz', degrees=True), 0)}"
              f"  {'ok' if inside else 'OUTSIDE EE box -> skipped'}")
        if inside:
            out.append(i)
    if drop:
        print(f"  dropped (outliers): {sorted(drop)}")
    return R, t, out


class Live:
    """Camera + window; `pump()` shows a frame and returns True if ESC was pressed."""
    def __init__(self, serial):
        self.pipe, self.align, self.K, self.dist = cal._start_realsense(serial)
        self.board, self.aruco = cal._make_board()
        self.charuco = cal._make_charuco_detector(self.board, self.K, self.dist)
        self.frame = None

    def pump(self, text: str = "") -> bool:
        self.frame = cal._get_bgr(self.pipe, self.align)
        vis = self.frame.copy()
        cv2.putText(vis, text + "   [ESC = stop]", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("calib_replay", vis)
        return (cv2.waitKey(1) & 0xFF) == ESC

    def detect(self):
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        return cal._detect(self.frame, gray, self.board, self.charuco, self.aruco, self.K, self.dist)

    def close(self):
        self.pipe.stop(); cv2.destroyAllWindows()


def _move(arm, live, t, R, args, label) -> bool:
    """Cartesian move with speed/acc caps; poll the window while moving. False = ESC."""
    aa = np.r_[t * 1e3, Rotation.from_matrix(R).as_rotvec()]
    arm.set_position_aa(aa.tolist(), speed=args.speed, mvacc=args.mvacc, is_radian=True, wait=False)
    t0 = time.time()
    while True:
        if live.pump(f"{label}: moving"):
            arm.set_state(4); print("  ESC -> arm stopped"); return False
        if not arm.get_is_moving() and time.time() - t0 > 0.5:
            return True
        if time.time() - t0 > args.move_timeout:
            arm.set_state(4); print("  move timeout -> arm stopped"); return False


def _wait(live, secs, label) -> bool:
    t0 = time.time()
    while time.time() - t0 < secs:
        if live.pump(f"{label}: settling {secs - (time.time() - t0):.0f}s"):
            return False
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--round", type=Path, default=None, help="raw round npz (default: newest)")
    p.add_argument("--drop", type=int, nargs="*", default=None, help="pose indices to skip (default: the round's recorded outliers)")
    p.add_argument("--dz", type=float, default=0.0, help="raise every pose by this (m)")
    p.add_argument("--live", action="store_true", help="MOVE the robot (default: dry run)")
    p.add_argument("--robot-ip", default="192.168.1.241")
    p.add_argument("--cam-serial", default="335522071488")
    p.add_argument("--speed", type=float, default=30.0, help="TCP speed cap, mm/s")
    p.add_argument("--mvacc", type=float, default=200.0, help="TCP acceleration cap, mm/s^2")
    p.add_argument("--dwell", type=float, default=5.0, help="pause at each pose before capturing (s)")
    p.add_argument("--frames", type=int, default=5, help="frames tried per pose; the detection with most corners is kept")
    p.add_argument("--move-timeout", type=float, default=60.0)
    p.add_argument("--start-z", type=float, default=0.35,
                   help="first lift straight up to this height (board out of view) and save a clean reference frame")
    p.add_argument("--park", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                   help="after the round, move here before the table check (default: the start pose)")
    p.add_argument("--table-check", action="store_true", help="run calib_select --table-check on the result")
    p.add_argument("--table-z", type=float, default=0.0, help="height of the surface the camera sees (m)")
    args = p.parse_args()

    src = args.round or _newest_round()
    print(f"round: {src}")
    R, t, idx = _load_poses(src, args.drop, args.dz)
    print(f"{len(idx)} poses to replay, dz {args.dz:+.3f} m, speed {args.speed:.0f} mm/s, dwell {args.dwell:.0f} s")
    if not args.live:
        print("dry run — pass --live to move the robot"); return

    print(f"connecting to XArm {args.robot_ip} …")
    from xarm.wrapper import XArmAPI
    arm = XArmAPI(args.robot_ip, is_radian=True)
    arm.motion_enable(enable=True); arm.set_mode(0); time.sleep(0.3); arm.set_state(0); time.sleep(0.3)
    live = Live(args.cam_serial)
    data = {k: [] for k in ("R_gripper2base", "t_gripper2base", "R_board2cam", "t_board2cam")}
    stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    try:
        # ── start: straight up to --start-z (board out of view) -> clean reference frame + ArUco ──
        R0, t0 = cal._get_ee_R_t(arm); start = np.r_[t0[:2], args.start_z]
        if not _move(arm, live, start, R0, args, "start") or not _wait(live, args.dwell, "start"):
            return
        ref_png = ROUND_DIR / f"reference_{stamp}.png"; cv2.imwrite(str(ref_png), live.frame)
        f = live.pipe.wait_for_frames(); f = live.align.process(f)
        depth = np.asarray(f.get_depth_frame().get_data()).astype(np.float32) * live.pipe.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
        det = aruco_check.detect(live.frame, depth, live.K, live.dist, 1, cv2.aruco.DICT_5X5_50)
        print(f"  reference frame -> {ref_png.name};  ArUco id 1 {'detected, corners px ' + str(np.round(det[0]).astype(int).tolist()) if det else 'NOT detected'}")
        if det:
            np.save(ROUND_DIR / f"reference_{stamp}_aruco_cam.npy", det[1])       # (4,3) camera-frame corners (PnP)
        for n, i in enumerate(idx):
            label = f"pose {i} ({n + 1}/{len(idx)})"
            if not _move(arm, live, t[i], R[i], args, label) or not _wait(live, args.dwell, label):
                break
            best = None
            for _ in range(args.frames):
                if live.pump(f"{label}: capturing"):
                    best = "esc"; break
                ok, rvec, tvec, corners, _, _ = live.detect()
                if ok and (best is None or len(corners) > best[0]):
                    best = (len(corners), rvec, tvec)
            if best == "esc":
                break
            if best is None:
                print(f"  {label}: board NOT detected -> skipped"); continue
            Rg, tg = cal._get_ee_R_t(arm)
            data["R_gripper2base"].append(Rg); data["t_gripper2base"].append(tg)
            data["R_board2cam"].append(Rotation.from_rotvec(best[1]).as_matrix()); data["t_board2cam"].append(best[2])
            print(f"  {label}: captured ({best[0]} corners)  t_grip2base={np.round(tg, 4)}")
        if len(data["R_gripper2base"]):                                     # park (default: the start pose)
            _move(arm, live, np.asarray(args.park) if args.park is not None else start, R0, args, "park")
    finally:
        live.close(); arm.disconnect()

    n = len(data["R_gripper2base"])
    if n < 3:
        print(f"only {n} capture(s) — nothing saved"); return
    out = ROUND_DIR / f"charuco_hand_eye_{stamp}.npz"
    np.savez(out, **{k: np.asarray(v) for k, v in data.items()}, camera_matrix=live.K, dist_coeffs=live.dist,
             replay_of=str(src.name), dz=args.dz)
    print(f"saved {n} poses -> {out}")
    cmd = [sys.executable, "-m", "gentle_manip.diagnostics.calib_select", "--raw", str(out)]
    if args.table_check:
        cmd += ["--table-check", "--cam-serial", args.cam_serial, "--table-z", str(args.table_z)]
    print("solving:", " ".join(cmd)); subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
