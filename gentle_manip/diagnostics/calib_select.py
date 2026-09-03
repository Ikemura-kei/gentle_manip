#!/usr/bin/env python3
"""Robust hand-eye extrinsic selection: search POSE SUBSETS x SOLVERS, score by consensus.

Why this exists
---------------
`calibration.py` solves once on every captured pose with Tsai. That is fragile: on the
2026-09-03 D435i round a SINGLE bad sample out of 11 moved the answer by 126 mm and put
the table 97 mm above the robot base (see NOTES_2026-09-03_d435i.md). Tsai degrades
sharply when the relative rotations between poses are small, which is exactly when one
outlier can dominate.

Selection principle (no ground truth needed)
-------------------------------------------
The board is rigidly clamped in the gripper, so

    T_board2gripper_i = inv(T_gripper2base_i) @ X @ T_board2cam_i

is a PHYSICAL CONSTANT — identical for every pose i. So a candidate X can be scored on
poses it was NOT fitted with: fit X on subset S, then measure how many of ALL N poses
agree with it. This is RANSAC's consensus criterion and, crucially, it does NOT reward
small subsets — a subset that overfits its own members scores badly on the rest.

    score = (number of inliers at --inlier-mm / --inlier-deg,   <- maximize first
             median residual over those inliers)                <- tie-break, minimize

Modes
-----
exhaustive  every subset of size >= --min-size (auto-selected when the count is small)
ransac      random subsets, then refit on the consensus set (for large N)
Both sweep every OpenCV solver (TSAI, PARK, HORAUD, ANDREFF, DANIILIDIS) unless --methods
narrows it.

Usage
-----
    uv run --project envs/deploy python -m gentle_manip.diagnostics.calib_select
    uv run --project envs/deploy python -m gentle_manip.diagnostics.calib_select \
        --raw dataset/camera_calibration/eye-to-hand/charuco_hand_eye_<ts>.npz --table-check

`--table-check` adds the decisive EXTERNAL validation: it opens the camera, fits the
dominant plane to points beyond --table-min-depth, and reports the table height/tilt in
robot-base coordinates. The table is at z=0 by definition, so this catches a globally
wrong X that internal consistency alone cannot. Clear the board out of the near field and
move the arm aside before using it.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}
_DEFAULT_GLOB = "dataset/camera_calibration/eye-to-hand/charuco_hand_eye_2*.npz"


# Degenerate subsets make OpenCV log "Not enough informative motions" per failed solve,
# which floods stderr during a subset sweep. Those fits are rejected by the scoring anyway.
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except AttributeError:
    pass


def _T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = np.asarray(t).flatten()
    return M


class Data:
    """The captured pose pairs, plus the derived transforms used throughout."""

    def __init__(self, path: str):
        d = np.load(path)
        self.path = path
        self.Rg, self.tg = d["R_gripper2base"], d["t_gripper2base"]
        self.Rb, self.tb = d["R_board2cam"], d["t_board2cam"]
        self.N = len(self.Rg)
        self.K = d["camera_matrix"] if "camera_matrix" in d.files else None
        self.T_g2b = [_T(self.Rg[i], self.tg[i]) for i in range(self.N)]
        self.T_g2b_inv = [np.linalg.inv(M) for M in self.T_g2b]
        self.T_b2c = [_T(self.Rb[i], self.tb[i]) for i in range(self.N)]

    def solve(self, idx: Sequence[int], method: int) -> np.ndarray | None:
        """Eye-to-hand solve on `idx`. Mirrors calibration.py::_calibrate exactly."""
        R_g2b = [self.Rg[i] for i in idx]
        t_g2b = [self.tg[i].reshape(3, 1) for i in idx]
        try:
            R, t = cv2.calibrateHandEye(
                [R.T for R in R_g2b],
                [-R.T @ t for R, t in zip(R_g2b, t_g2b)],
                [self.Rb[i] for i in idx],
                [self.tb[i].reshape(3, 1) for i in idx],
                method=method,
            )
        except cv2.error:
            return None
        if R is None or not np.all(np.isfinite(R)) or not np.all(np.isfinite(t)):
            return None
        # reject non-rotations (some solvers can return garbage on degenerate input)
        if abs(np.linalg.det(R) - 1.0) > 1e-3:
            return None
        return _T(R, t.flatten())

    def board_in_gripper(self, X: np.ndarray) -> np.ndarray:
        """(N, 4, 4) — the quantity that must be constant across poses."""
        return np.stack([self.T_g2b_inv[i] @ X @ self.T_b2c[i] for i in range(self.N)])


def residuals(data: Data, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-pose deviation (mm, deg) of board-in-gripper from the robust consensus."""
    M = data.board_in_gripper(X)
    t_ref = np.median(M[:, :3, 3], axis=0)
    q = Rotation.from_matrix(M[:, :3, :3]).as_quat()
    q *= np.sign(q @ q[0])[:, None]                    # hemisphere-align before averaging
    R_ref = Rotation.from_quat(q.mean(0) / np.linalg.norm(q.mean(0)))
    dt = np.linalg.norm(M[:, :3, 3] - t_ref, axis=1) * 1000.0
    dr = np.degrees(Rotation.from_matrix(
        R_ref.as_matrix().T @ M[:, :3, :3]).magnitude())
    return dt, dr


def score(data: Data, X: np.ndarray, mm: float, deg: float) -> Tuple[int, float, np.ndarray]:
    """(inlier count, median inlier residual mm, inlier mask) over ALL poses."""
    dt, dr = residuals(data, X)
    ok = (dt <= mm) & (dr <= deg)
    med = float(np.median(dt[ok])) if ok.any() else float("inf")
    return int(ok.sum()), med, ok


def search(data: Data, args) -> List[dict]:
    """Return candidates sorted best-first."""
    names = args.methods or list(METHODS)
    n_sub = sum(len(list(itertools.combinations(range(data.N), k)))
                for k in range(args.min_size, data.N + 1))
    mode = args.mode
    if mode == "auto":
        mode = "exhaustive" if n_sub * len(names) <= args.max_solves else "ransac"
    print(f"  poses={data.N}  subsets(size>={args.min_size})={n_sub}  "
          f"methods={len(names)}  -> mode={mode}")

    out: List[dict] = []
    seen: set = set()
    t0 = time.time()

    def consider(idx, mname):
        X = data.solve(idx, METHODS[mname])
        if X is None:
            return
        n_in, med, ok = score(data, X, args.inlier_mm, args.inlier_deg)
        key = (mname, tuple(np.flatnonzero(ok)))
        if key in seen:
            return
        seen.add(key)
        out.append({"X": X, "method": mname, "fit_idx": list(idx),
                    "n_inliers": n_in, "med_mm": med, "inliers": ok})

    if mode == "exhaustive":
        for k in range(args.min_size, data.N + 1):
            for idx in itertools.combinations(range(data.N), k):
                for m in names:
                    consider(idx, m)
    else:
        rng = np.random.default_rng(args.seed)
        for _ in range(args.iters):
            k = int(rng.integers(args.min_size, data.N + 1))
            idx = sorted(rng.choice(data.N, size=k, replace=False).tolist())
            for m in names:
                consider(idx, m)

    # refit each candidate on its own consensus set (the RANSAC polish step)
    for c in list(out):
        inl = list(np.flatnonzero(c["inliers"]))
        if len(inl) >= args.min_size and inl != c["fit_idx"]:
            X2 = data.solve(inl, METHODS[c["method"]])
            if X2 is not None:
                n2, m2, ok2 = score(data, X2, args.inlier_mm, args.inlier_deg)
                out.append({"X": X2, "method": c["method"], "fit_idx": inl,
                            "n_inliers": n2, "med_mm": m2, "inliers": ok2,
                            "refit": True})
    out.sort(key=lambda c: (-c["n_inliers"], c["med_mm"]))
    print(f"  evaluated {len(seen)} distinct fits in {time.time()-t0:.1f}s")
    return out


def table_check(X: np.ndarray, args) -> None:
    """EXTERNAL validation: the imaged surface must lie at a KNOWN height, normal up.

    `--table-z` is the expected height of whatever surface the camera actually sees, in
    robot-base metres. It is 0 for the bare table, but must be raised when something is
    laid on top (e.g. a 14 mm board) — otherwise a correct extrinsic is reported as failing.
    """
    import open3d as o3d
    import pyrealsense2 as rs
    from gentle_manip.perception.depth_to_pointcloud import depth_to_pointcloud

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(args.cam_serial)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    prof = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    sc = prof.get_device().first_depth_sensor().get_depth_scale()
    try:
        for _ in range(40):
            f = align.process(pipe.wait_for_frames(10000))
        depth = np.asarray(f.get_depth_frame().get_data()).astype(np.float32) * sc
        intr = f.get_color_frame().profile.as_video_stream_profile().intrinsics
        K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], np.float32)
    finally:
        pipe.stop()

    pts, valid = depth_to_pointcloud(depth[None], K, X.astype(np.float32),
                                     depth_min=args.table_min_depth, depth_max=2.0)
    p = pts[0][valid[0]]
    if len(p) < 1000:
        print("  [table] too few points beyond the min depth — skipped")
        return
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p.astype(np.float64)))
    model, inl = pc.segment_plane(0.008, 3, 3000)
    n = np.array(model[:3])
    if n[2] < 0:
        n = -n
    tilt = np.degrees(np.arccos(np.clip(n[2], -1, 1)))
    z = np.asarray(pc.points)[inl][:, 2]
    dz_mm = (z.mean() - args.table_z) * 1000.0
    print(f"  [table] normal {np.round(n,3)}  tilt {tilt:.2f} deg from horizontal "
          f"(should be ~0)")
    print(f"  [table] measured height {z.mean()*1000:+.1f} mm   expected "
          f"{args.table_z*1000:+.1f} mm   -> off by {dz_mm:+.1f} mm")
    print(f"  [table] plane std {z.std()*1000:.1f} mm over {len(inl)} inliers")
    bad = abs(dz_mm) > args.table_tol_mm or tilt > args.table_tol_deg
    print(f"  [table] {'*** FAILS the physical check — do not trust this X ***' if bad else 'PASSES'}"
          f"  (tolerance {args.table_tol_mm:.0f} mm / {args.table_tol_deg:.1f} deg)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default=None, help="raw npz (default: newest eye-to-hand)")
    p.add_argument("--min-size", type=int, default=6,
                   help="smallest subset to fit (>=3 is the algebraic minimum; 6+ is sane)")
    p.add_argument("--inlier-mm", type=float, default=4.0)
    p.add_argument("--inlier-deg", type=float, default=1.5)
    p.add_argument("--mode", choices=("auto", "exhaustive", "ransac"), default="auto")
    p.add_argument("--max-solves", type=int, default=400_000,
                   help="auto mode switches to ransac above this many solves")
    p.add_argument("--iters", type=int, default=20_000, help="ransac subsets to try")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--methods", nargs="*", choices=list(METHODS), default=None)
    p.add_argument("--top", type=int, default=8)
    p.add_argument("--table-check", action="store_true",
                   help="validate the winner against the real table plane (needs the camera)")
    p.add_argument("--cam-serial", default="335522071488")
    p.add_argument("--table-z", type=float, default=0.0,
                   help="expected height (m, robot base) of the surface the camera sees. "
                        "0 = bare table; raise it if something is laid on top, e.g. 0.014 "
                        "for a 14 mm board")
    p.add_argument("--table-tol-mm", type=float, default=20.0)
    p.add_argument("--table-tol-deg", type=float, default=3.0)
    p.add_argument("--table-min-depth", type=float, default=0.33,
                   help="ignore points nearer than this (excludes a board still in the gripper)")
    args = p.parse_args()

    # exclude this tool's OWN outputs (and the solver's result file) from the auto-pick,
    # or a second run selects its own artifact instead of the raw capture
    raw = args.raw or max(
        (f for f in glob.glob(_DEFAULT_GLOB)
         if not f.endswith("_selected.npz") and "_result_" not in f),
        key=os.path.getmtime)
    data = Data(raw)
    print(f"raw: {raw}\n  {data.N} poses")
    if data.N < args.min_size:
        sys.exit(f"only {data.N} poses; need >= --min-size ({args.min_size})")

    cands = search(data, args)
    if not cands:
        sys.exit("no valid solution found")

    print(f"\ntop {args.top} candidates "
          f"(inliers of {data.N} @ {args.inlier_mm}mm/{args.inlier_deg}deg, then median residual):")
    print(f"{'#':>2} {'method':>11} {'inl':>4} {'med_mm':>7} {'fit_n':>6}  camera_pos(m)      excluded")
    for r, c in enumerate(cands[:args.top]):
        out = [i for i in range(data.N) if not c["inliers"][i]]
        print(f"{r:2d} {c['method']:>11} {c['n_inliers']:4d} {c['med_mm']:7.2f} "
              f"{len(c['fit_idx']):6d}  {np.round(c['X'][:3,3],3)}  {out}")

    # RANSAC finish: the winner selects the CONSENSUS SET; the final estimate is refit on
    # all of it. A subset that merely scores well is not the estimate — using every
    # consistent pose is strictly better conditioned than the minimal fit that found them.
    best = cands[0]
    inl0 = [int(i) for i in np.flatnonzero(best["inliers"])]
    if len(inl0) > len(best["fit_idx"]):
        Xr = data.solve(inl0, METHODS[best["method"]])
        if Xr is not None:
            n_r, med_r, ok_r = score(data, Xr, args.inlier_mm, args.inlier_deg)
            print(f"\n  consensus refit on all {len(inl0)} inlier poses "
                  f"(winner was a {len(best['fit_idx'])}-pose fit): "
                  f"inliers {best['n_inliers']}->{n_r}, med {best['med_mm']:.2f}->{med_r:.2f} mm")
            if n_r >= best["n_inliers"]:
                print("  -> refit ACCEPTED (uses more data, no consensus lost)")
                inl0 = [int(i) for i in inl0]
                best = {"X": Xr, "method": best["method"], "fit_idx": inl0,
                        "n_inliers": n_r, "med_mm": med_r, "inliers": ok_r}
            else:
                print(f"  -> refit REJECTED: it loses {best['n_inliers']-n_r} inlier(s), i.e. the "
                      f"consensus set is not mutually consistent at {args.inlier_mm}mm.\n"
                      f"     Keeping the subset fit. If this happens on good data, the threshold "
                      f"is too tight — re-run with a larger --inlier-mm.")
    X = best["X"]
    dt, dr = residuals(data, X)
    print(f"\nSELECTED: {best['method']} fitted on {len(best['fit_idx'])} poses "
          f"{best['fit_idx']}")
    print(f"  camera position {np.round(X[:3,3],4)} m   "
          f"tilt {np.degrees(np.arcsin(-X[:3,2][2])):.1f} deg below horizontal")
    print(f"  inliers {best['n_inliers']}/{data.N}, median residual {best['med_mm']:.2f} mm")
    print(f"  EXCLUDED poses: {[i for i in range(data.N) if not best['inliers'][i]]}")
    print(f"\n  {'pose':>4} {'mm':>8} {'deg':>7}")
    for i in range(data.N):
        print(f"  {i:4d} {dt[i]:8.2f} {dr[i]:7.3f}"
              f"{'' if best['inliers'][i] else '   <-- excluded'}")

    # agreement across independent solvers on the winning consensus set = extra confidence
    inl = list(np.flatnonzero(best["inliers"]))
    print("\n  cross-solver agreement on the selected pose set:")
    for m in METHODS:
        Xm = data.solve(inl, METHODS[m])
        if Xm is None:
            print(f"    {m:>11}  failed")
            continue
        dp = np.linalg.norm(Xm[:3, 3] - X[:3, 3]) * 1000
        da = np.degrees(np.linalg.norm(
            Rotation.from_matrix(X[:3, :3].T @ Xm[:3, :3]).as_rotvec()))
        print(f"    {m:>11}  dpos {dp:7.2f} mm   drot {da:6.3f} deg")

    print("\nPaste into gentle_manip/robot/xarm7_config.py:")
    print("[")
    for row in X:
        print("    [" + ", ".join(f"{v:15.8f}" for v in row) + "],")
    print("]")

    out_npz = Path(raw).with_name(Path(raw).stem + "_selected.npz")
    np.savez(str(out_npz), T=X, method=best["method"],
             fit_idx=np.array(best["fit_idx"]), inliers=best["inliers"])
    print(f"\nsaved -> {out_npz}")

    if args.table_check:
        print("\nEXTERNAL table-plane check (clear the board / move the arm aside):")
        table_check(X, args)


if __name__ == "__main__":
    main()
