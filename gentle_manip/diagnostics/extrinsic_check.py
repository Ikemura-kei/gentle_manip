#!/usr/bin/env python3
"""Validity of the CURRENT WORLD_T_CAM_EXT against external truth: the TCP-measured ArUco corner(s) and the
table plane, from the pinned park pose. `--move` drives the arm to the park pose first (gentle, ESC stops);
without it the arm is assumed parked already (board out of view).

    uv run --project envs/deploy python -m gentle_manip.diagnostics.extrinsic_check [--move]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.spatial.transform import Rotation

from gentle_manip.diagnostics import drift_check, extrinsic_correct as ec
from gentle_manip.robot import xarm7_config as cfg


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--move", action="store_true", help="move the arm to the pinned park pose first")
    p.add_argument("--robot-ip", default="192.168.1.241")
    p.add_argument("--speed", type=float, default=30.0); p.add_argument("--mvacc", type=float, default=200.0)
    a = p.parse_args()
    ref = dict(np.load(drift_check.REF, allow_pickle=True)); serial = str(ref["cam_serial"])
    if a.move:
        from gentle_manip.diagnostics.calib_replay import Live, _move
        from xarm.wrapper import XArmAPI
        arm = XArmAPI(a.robot_ip, is_radian=True); arm.motion_enable(True); arm.set_mode(0); time.sleep(0.3); arm.set_state(0); time.sleep(0.3)
        live = Live(serial); a.move_timeout = 60.0
        try:
            ok = _move(arm, live, ref["park_pos_m"], Rotation.from_rotvec(ref["park_rotvec"]).as_matrix(), a, "park")
        finally:
            live.close(); arm.disconnect()
        if not ok:
            return
        time.sleep(1.0)
    X = np.asarray(cfg.WORLD_T_CAM_EXT, float)
    det, _ = drift_check.capture(serial, 15)
    print(f"current WORLD_T_CAM_EXT vs external truth (park pose {np.round(ref['park_pos_m'], 3)} m):")
    if det is None:
        print("  ArUco NOT detected")
    else:
        cam = det[1]
        for idx, x, y, z in ref["ref_tcp"]:
            e = (ec.apply(X, cam[int(idx):int(idx) + 1])[0] - np.array([x, y, z])) * 1e3
            print(f"  ArUco corner {int(idx)}: camera->base minus TCP-measured = {np.round(e, 1)} mm  |{np.linalg.norm(e):.1f} mm|")
    cloud = ec.table_cloud(serial, 0.33); n_cam, c_cam, k = ec.fit_plane(cloud); n, c = ec.plane_in(X, n_cam, c_cam)
    print(f"  table plane: tilt {np.degrees(np.arccos(np.clip(n[2], -1, 1))):.2f} deg (expect 0), height {1e3*c[2]:.1f} mm (expect {1e3*float(ref['table_z']):.1f}), {k} pts")
    print("  noise floor (measured 2026-09-05, arm parked): corner +-4 mm, tilt +-0.2 deg, height +-0.2 mm; a stale extrinsic read 122 mm / 10 deg")


if __name__ == "__main__":
    main()
