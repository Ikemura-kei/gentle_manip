#!/usr/bin/env python3
"""Correct a hand-eye extrinsic with EXTERNAL truth: the table plane (tilt + height) and one or more
TCP-measured ArUco corners (x/y). Camera only — park the arm out of view first.

    uv run --project envs/deploy python -m gentle_manip.diagnostics.extrinsic_correct \\
        --selected <round>_selected.npz --aruco-cam reference_<stamp>_aruco_cam.npy \\
        --ref 1 0.1613 0.0829 0.0136 --table-z 0.0138
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from gentle_manip.diagnostics import calibration as cal

ROUND_DIR = cal.DEFAULT_OUT / "eye-to-hand"


def table_cloud(serial, min_depth, n_frames: int = 10):
    """Camera-frame points beyond min_depth from the per-pixel MEDIAN of n_frames depth frames."""
    from gentle_manip.perception.depth_to_pointcloud import depth_to_pointcloud
    pipe, align, K, _ = cal._start_realsense(serial)
    sc = pipe.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
    try:
        for _ in range(20):
            align.process(pipe.wait_for_frames())
        ds = [np.asarray(align.process(pipe.wait_for_frames()).get_depth_frame().get_data()).astype(np.float32) * sc
              for _ in range(n_frames)]
    finally:
        pipe.stop()
    d = np.stack(ds); d[d <= 0] = np.nan
    with np.errstate(all="ignore"):
        import warnings; warnings.simplefilter("ignore", RuntimeWarning)
        depth = np.nan_to_num(np.nanmedian(d, axis=0), nan=0.0).astype(np.float32)
    pts, valid = depth_to_pointcloud(depth[None], K.astype(np.float32), np.eye(4, dtype=np.float32),
                                     depth_min=min_depth, depth_max=2.0)
    return pts[0][valid[0]].astype(np.float64)                      # CAMERA frame


def fit_plane(p):
    """Plane (unit normal, centroid, n_inliers) fitted ONCE: seeded RANSAC for the inlier set, SVD refinement."""
    import open3d as o3d
    o3d.utility.random.seed(0)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
    _, inl = pc.segment_plane(0.004, 3, 5000)      # 4 mm: 8 mm merged the bare table (0 mm) with the board (13.8) into a tilted plane
    q = p[inl]; c = q.mean(0)
    n = np.linalg.svd(q - c, full_matrices=False)[2][-1]
    return n / np.linalg.norm(n), c, len(inl)


def plane_in(T, n_cam, c_cam):
    n = T[:3, :3] @ n_cam; n = n if n[2] > 0 else -n
    return n, T[:3, :3] @ c_cam + T[:3, 3]


def apply(X, p_cam):
    return (X[:3, :3] @ p_cam.T).T + X[:3, 3]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selected", type=Path, required=True)
    p.add_argument("--aruco-cam", type=Path, required=True, help="(4,3) camera-frame ArUco corners (reference_*_aruco_cam.npy)")
    p.add_argument("--ref", type=float, nargs=4, action="append", required=True, metavar=("IDX", "X", "Y", "Z"))
    p.add_argument("--table-z", type=float, default=0.0138)
    p.add_argument("--cam-serial", default="335522071488")
    p.add_argument("--min-depth", type=float, default=0.33)
    a = p.parse_args()

    X = np.load(a.selected)["T"]; corners = np.load(a.aruco_cam); refs = [(int(r[0]), np.asarray(r[1:])) for r in a.ref]
    cloud = table_cloud(a.cam_serial, a.min_depth)
    n_cam, c_cam, k = fit_plane(cloud)                                # fitted once, in the camera frame
    rms = np.std((cloud[np.abs((cloud - c_cam) @ n_cam) < 0.004] - c_cam) @ n_cam); print(f"table plane: {k} inliers of {len(cloud)} pts, inlier residual std {1e3*rms:.1f} mm (median of 10 frames)")

    def report(tag, T):
        n, c = plane_in(T, n_cam, c_cam); tilt = np.degrees(np.arccos(np.clip(n[2], -1, 1)))
        errs = [(i, (apply(T, corners[i:i + 1])[0] - m) * 1e3) for i, m in refs]
        print(f"  {tag:22s} table tilt {tilt:5.2f} deg  height {1e3*c[2]:6.1f} mm (expect {1e3*a.table_z:.1f}, {k} pts)  |  "
              + "  ".join(f"corner {i}: err {np.round(e,1)} mm" for i, e in errs))
        return n, c

    print("hand-eye solve vs external truth (camera minus robot):")
    n, c = report("raw hand-eye", X)
    # 1. tilt + height: rotate about the plane centroid so its normal becomes +z, then lift to table_z
    Rc = Rotation.align_vectors([[0, 0, 1]], [n])[0].as_matrix()
    C = np.eye(4); C[:3, :3] = Rc; C[:3, 3] = c - Rc @ c
    X1 = C @ X; X1[2, 3] += a.table_z - plane_in(X1, n_cam, c_cam)[1][2]
    report("+ plane (tilt,height)", X1)
    # 2. x/y from the measured corner(s): mean in-plane residual
    d = np.mean([apply(X1, corners[i:i + 1])[0] - m for i, m in refs], axis=0)
    X2 = X1.copy(); X2[0, 3] -= d[0]; X2[1, 3] -= d[1]
    report("+ ArUco x/y", X2)
    print(f"\n  correction vs raw: rotation {np.degrees(np.linalg.norm(Rotation.from_matrix(Rc).as_rotvec())):.2f} deg, "
          f"camera moved {1e3*np.linalg.norm(X2[:3,3]-X[:3,3]):.1f} mm  (x/y shift from ArUco {np.round(1e3*d[:2],1)} mm; "
          f"corner z after plane fix {1e3*(apply(X1, corners[refs[0][0]:refs[0][0]+1])[0][2]):.1f} vs measured {1e3*refs[0][1][2]:.1f} mm)")
    out = a.selected.with_name(a.selected.name.replace("_selected.npz", "_corrected.npz"))
    np.savez(out, T=X2, T_hand_eye=X, table_normal=n, table_z=a.table_z, refs=np.array([np.r_[i, m] for i, m in refs]))
    print(f"\n  saved -> {out}\n\nPaste into gentle_manip/robot/xarm7_config.py:\n[")
    for row in X2:
        print("    [" + ", ".join(f"{v:15.8f}" for v in row) + "],")
    print("]")


if __name__ == "__main__":
    main()
