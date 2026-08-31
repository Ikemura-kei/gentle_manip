"""E1 baselines: gentleness-blind grasp synthesis alternatives, executed by the SAME v4 executor.

Two baselines for the paper's comparison (docs/paper/synthesis_experiments.md, E1):

  naive     — top-down at the settled object centre, uniform-random yaw inside the same hard yaw
              bound the FEM synthesis uses; width = cross-section at the grasp − 2 mm (the
              SQUEEZE_M convention of the vision-only scripted baseline, dppo/scripted/).
  antipodal — DefGraspSim-style candidate generation: sample surface point pairs whose normals
              oppose within the friction cone (Nguyen force-closure criterion), keep the
              geometrically feasible ones, rank by CONE MARGIN (the honest two-hard-contact
              stand-in for Ferrari-Canny epsilon, which is identically zero for two frictional
              point contacts in 6D). Width = pair distance − 2 mm.
  rigid     — the STRONG rigid-body planner (so a v4.1 win is attributable to
              gentleness-awareness, not search budget): stage 1 sweeps a LARGE antipodal
              candidate set (n_samples 4000) by cone margin; stage 2 re-ranks the top-K
              candidates with the full geometric scorer — flush alignment, worst-pad contact
              area, COM lever, holdability — i.e. every quantity our FEM scorer computes EXCEPT
              stress. GPD-style two-stage architecture on the privileged mesh.

Both are gentleness-blind BY DESIGN: no stress term anywhere. Geometric validity (table, gross
penetration, jaw capture) is checked with the same ladder the FEM synthesis uses, so failures of
the baselines are attributable to SELECTION, not to being handed an invalid pose.

Interface matches fg.synthesize_grasp: returns {"x": 7-DOF TCP grasp, "stress_top10": None-safe
placeholder metrics} so `collect_demos_baseline.py` can monkeypatch it in.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import time

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from smgrasp import finger_grasp as fg
from smgrasp.width_grasp import boundary_normals
from smgrasp.viz import boundary_faces

SQUEEZE_M = 0.002          # width = contact width − this (scripted_topdown convention)
AXIS_Z_MAX = 0.35          # antipodal closing axes must be near-horizontal (top-down jaw)


def _topdown_x(cx, cy, yaw, width, obj, pad_geo, table_z):
    """7-DOF top-down grasp at (cx, cy), z set exactly like the CMA seed loop: raise until the
    lowest finger point clears the table by ground_buf + 3 mm."""
    x = np.array([cx, cy, 0.0, np.pi, 0.0, yaw, max(0.008, width)], float)
    x[2] += (table_z + 0.0035 + 0.003) - fg.finger_min_world_z(x, pad_geo)
    return x


def _cross_section(obj, obj_quat, yaw):
    """Object cross-section along the world closing axis of a top-down grasp at `yaw`."""
    q = np.asarray(obj_quat, float)
    Rinv = Rot.from_quat([q[1], q[2], q[3], q[0]]).inv()
    ax = Rinv.apply(np.array([np.sin(yaw), -np.cos(yaw), 0.0]))
    proj = obj.verts @ (ax / (np.linalg.norm(ax) + 1e-12))
    return float(proj.max() - proj.min())


def naive_topdown(obj, pad_geo, obj_com, obj_quat_wxyz, *, E=3e5, density=1000.0, mu=0.7,
                  table_z=0.0, seed=0, yaw_max_deg=None, **_ignored):
    rng = np.random.default_rng(seed)
    hi = np.pi / 2 if yaw_max_deg is None else np.radians(float(yaw_max_deg))
    yaw = float(rng.uniform(-hi, hi))
    com = np.asarray(obj_com, float)
    w = _cross_section(obj, obj_quat_wxyz, yaw) - SQUEEZE_M
    x = _topdown_x(com[0], com[1], yaw, w, obj, pad_geo, table_z)
    return {"x": x, "stress_top10": 1.0, "grip": 0.0, "align": 0.0, "pressure": None,
            "min_pad_area": 0.0, "width_face": None, "baseline": "naive"}


def antipodal(obj, pad_geo, obj_com, obj_quat_wxyz, *, E=3e5, density=1000.0, mu=0.7,
              table_z=0.0, seed=0, n_samples=600, yaw_max_deg=None, **_ignored):
    """Antipodal pair sampling + cone-margin ranking (gentleness-blind established baseline)."""
    rng = np.random.default_rng(seed)
    q = np.asarray(obj_quat_wxyz, float)
    R = Rot.from_quat([q[1], q[2], q[3], q[0]])
    com = np.asarray(obj_com, float)

    bidx = np.unique(boundary_faces(obj.tets)[0])
    P_l = obj.verts[bidx]                                # object-local boundary points
    N_l = boundary_normals(obj)[bidx]                    # outward normals
    P = R.apply(P_l) + com                               # world
    N = R.apply(N_l)
    cone = np.arctan(mu)
    hi = np.pi / 2 if yaw_max_deg is None else np.radians(float(yaw_max_deg))

    best = None
    ii = rng.integers(0, len(P), n_samples)
    jj = rng.integers(0, len(P), n_samples)
    for i, j in zip(ii, jj):
        if i == j:
            continue
        d = P[j] - P[i]
        L = np.linalg.norm(d)
        if L < 0.008 or L > 0.079:
            continue
        d /= L
        if abs(d[2]) > AXIS_Z_MAX:                       # need a near-horizontal closing axis
            continue
        # Nguyen: the connecting line must lie inside both friction cones
        a1 = np.arccos(np.clip(np.dot(N[i], -d), -1, 1))
        a2 = np.arccos(np.clip(np.dot(N[j], d), -1, 1))
        margin = cone - max(a1, a2)
        if margin <= 0:
            continue
        yaw = float(np.arctan2(d[0], -d[1]))             # closing axis(yaw) = [sin, -cos, 0]
        yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2
        if abs(yaw) > hi + 1e-6:
            continue
        mid = 0.5 * (P[i] + P[j])
        x = _topdown_x(mid[0], mid[1], yaw, L - SQUEEZE_M, obj, pad_geo, table_z)
        sc = fg.score_finger_grasp(obj, x, obj_com=com, obj_quat_wxyz=q, pad_geo=pad_geo,
                                   E=E, density=density, mu=mu, table_z=table_z)
        if sc.get("status") not in ("ok",):              # geometric validity only
            continue
        if best is None or margin > best[0]:
            best = (margin, x)
    if best is None:
        return {"x": None, "stress_top10": None, "baseline": "antipodal"}
    return {"x": best[1], "stress_top10": 1.0, "grip": 0.0, "align": 0.0, "pressure": None,
            "min_pad_area": 0.0, "width_face": None, "baseline": "antipodal"}


def rigid_planner(obj, pad_geo, obj_com, obj_quat_wxyz, *, E=3e5, density=1000.0, mu=0.7,
                  table_z=0.0, seed=0, n_samples=4000, top_k=40, yaw_max_deg=None,
                  cam_pos=None, cam_azimuth_max_deg=None, **_ignored):
    """Two-stage rigid-body planner: large antipodal sweep -> geometric re-rank (no stress).

    rigid_score = align − 5·lever/obj_size + 0.3·min(min_pad/50mm2, 1), holdable candidates only.
    All quantities from the same scorer v4.1 uses, with the stress terms EXCLUDED — the exact
    gentleness-awareness ablation at matched candidate quality."""
    rng = np.random.default_rng(seed)
    q = np.asarray(obj_quat_wxyz, float)
    R = Rot.from_quat([q[1], q[2], q[3], q[0]])
    com = np.asarray(obj_com, float)
    bidx = np.unique(boundary_faces(obj.tets)[0])
    P = R.apply(obj.verts[bidx]) + com
    N = R.apply(boundary_normals(obj)[bidx])
    cone = np.arctan(mu)
    hi = np.pi / 2 if yaw_max_deg is None else np.radians(float(yaw_max_deg))
    obj_size = float((obj.verts.max(0) - obj.verts.min(0)).max())

    # Optional HARD camera-occlusion bound — the same structural constraint v4.1's search
    # carries (fg.cam_azimuth_deg semantics: angle between the closing axis and the vertical
    # plane perpendicular to the camera ray, horizontal projections). Off (None) by default so
    # the plain E1 baseline stays occlusion-unconstrained; enabled via --baseline-occ for the
    # apples-to-apples confound check.
    if cam_pos is not None and cam_azimuth_max_deg is not None:
        d_h = (com - np.asarray(cam_pos, float))[:2]
        d_h = d_h / (np.linalg.norm(d_h) + 1e-12)
        az_lim = np.sin(np.radians(float(cam_azimuth_max_deg)))

        def _az_ok(yaw):
            a_h = np.array([np.sin(yaw), -np.cos(yaw)])
            return abs(float(a_h @ d_h)) <= az_lim + 1e-9
    else:
        def _az_ok(yaw):
            return True

    cands = []
    ii = rng.integers(0, len(P), n_samples)
    jj = rng.integers(0, len(P), n_samples)
    for i, j in zip(ii, jj):
        if i == j:
            continue
        d = P[j] - P[i]
        L = np.linalg.norm(d)
        if L < 0.008 or L > 0.079:
            continue
        d /= L
        if abs(d[2]) > AXIS_Z_MAX:
            continue
        a1 = np.arccos(np.clip(np.dot(N[i], -d), -1, 1))
        a2 = np.arccos(np.clip(np.dot(N[j], d), -1, 1))
        margin = cone - max(a1, a2)
        if margin <= 0:
            continue
        yaw = (float(np.arctan2(d[0], -d[1])) + np.pi / 2) % np.pi - np.pi / 2
        if abs(yaw) > hi + 1e-6:
            continue
        if not _az_ok(yaw):
            continue
        mid = 0.5 * (P[i] + P[j])
        cands.append((margin, mid, yaw, L))
    cands.sort(key=lambda c: -c[0])

    best = None
    for margin, mid, yaw, L in cands[:top_k]:
        x = _topdown_x(mid[0], mid[1], yaw, L - SQUEEZE_M, obj, pad_geo, table_z)
        sc = fg.score_finger_grasp(obj, x, obj_com=com, obj_quat_wxyz=q, pad_geo=pad_geo,
                                   E=E, density=density, mu=mu, table_z=table_z)
        if sc.get("status") != "ok" or not sc.get("holdable"):
            continue
        align = float(sc.get("align") or 0.0)
        lever = float(sc.get("com_lever") or 0.0)
        pad = float(sc.get("min_pad_area") or 0.0)
        rigid_score = align - 5.0 * lever / max(obj_size, 1e-6) + 0.3 * min(pad / 50e-6, 1.0)
        if best is None or rigid_score > best[0]:
            best = (rigid_score, x)
    if best is None:
        return {"x": None, "stress_top10": None, "baseline": "rigid"}
    return {"x": best[1], "stress_top10": 1.0, "grip": 0.0, "align": 0.0, "pressure": None,
            "min_pad_area": 0.0, "width_face": None, "baseline": "rigid"}


# ── GPD (ten Pas et al., IJRR 2017) — ESTABLISHED external rigid-body grasp planner ──────────────
GPD_ROOT = os.path.join(os.path.dirname(__file__), "..", "third_party", "gpd")
GPD_BIN = os.path.join(GPD_ROOT, "build", "detect_grasps")
GPD_CFG = os.path.join(GPD_ROOT, "cfg", "gm_gpd.cfg")
GPD_HAND_DEPTH = 0.045      # must match cfg/gm_hand_xarm.cfg


def _surface_pcd(obj, R, com, n_pts=15000, seed=0):
    """Dense area-weighted sample of the tet-boundary surface, world frame."""
    rng = np.random.default_rng(seed)
    F = boundary_faces(obj.tets)[0]
    V = obj.verts
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    fi = rng.choice(len(F), n_pts, p=areas / areas.sum())
    r1, r2 = rng.random(n_pts), rng.random(n_pts)
    s1 = np.sqrt(r1)
    pts = (1 - s1)[:, None] * a[fi] + (s1 * (1 - r2))[:, None] * b[fi] + (s1 * r2)[:, None] * c[fi]
    return R.apply(pts) + com


def _write_pcd(pts, path):
    with open(path, "w") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n"
                "FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
                f"WIDTH {len(pts)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
                f"POINTS {len(pts)}\nDATA ascii\n")
        np.savetxt(f, pts, fmt="%.6f")


def gpd_planner(obj, pad_geo, obj_com, obj_quat_wxyz, *, E=3e5, density=1000.0, mu=0.7,
                table_z=0.0, seed=0, yaw_max_deg=None, **_ignored):
    """GPD as an off-the-shelf rigid-body baseline: privileged dense object cloud -> GPD's
    candidate sweep + CNN scoring -> best hand mapped to our 7-DOF TCP grasp.

    Frame map (GPD hand columns [approach, binormal, axis] -> our TCP [x, y, z] with
    y = closing, z = approach): R_ours = [-axis, binormal, approach]. GPD's position is the
    hand-base centre; the enclosed-object slice sits ~hand_depth/2 further along approach, which
    we equate with our pad mid-plane centre and invert to the TCP via fg's pad placement.
    Candidates are taken in GPD score order; the first that passes the SAME geometric validity
    ladder (score_finger_grasp status == ok) wins. Width = GPD aperture - SQUEEZE_M (own
    convention, like the other baselines). No camera-occlusion bound is applied (GPD knows
    nothing of our camera; E1 scores grasp quality, not data-collection viability)."""
    q = np.asarray(obj_quat_wxyz, float)
    R = Rot.from_quat([q[1], q[2], q[3], q[0]])
    com = np.asarray(obj_com, float)
    pts = _surface_pcd(obj, R, com, seed=seed)
    fd, pcd = tempfile.mkstemp(suffix=".pcd", prefix="gm_gpd_")
    os.close(fd)
    try:
        _write_pcd(pts, pcd)
        env = {k: v for k, v in os.environ.items() if k not in ("LD_LIBRARY_PATH", "LIBRARY_PATH")}
        env["PATH"] = "/usr/local/bin:/usr/bin:/bin"
        out = subprocess.run([GPD_BIN, GPD_CFG, pcd], capture_output=True, text=True,
                             timeout=300, env=env).stdout
    finally:
        os.unlink(pcd)

    hands = []
    for line in out.splitlines():
        if not line.startswith("GRASP_POSE"):
            continue
        t = line.split()
        sc = float(t[3])
        pos = np.array(t[5:8], float)
        app = np.array(t[9:12], float)
        bin_ = np.array(t[13:16], float)
        ax = np.array(t[17:20], float)
        w = float(t[21])
        hands.append((sc, pos, app, bin_, ax, w))
    hands.sort(key=lambda h: -h[0])

    return _rank_to_tcp(hands, obj, pad_geo, com, q, E, density, mu, table_z, "gpd",
                        depth_off=GPD_HAND_DEPTH * 0.5)


def _local_xsec(obj, q, com, pad_centre_w, closing_w, approach_w):
    """Object cross-section along the closing axis, restricted to the slab of surface the pads
    actually touch (|delta approach| < 15 mm, |delta third-axis| < 12 mm around the pad centre).
    Used to convert a learned planner's PRE-SHAPE opening into the faithful width-command
    equivalent of 'close until contact' (rigid execution semantics)."""
    Rinv = Rot.from_quat([q[1], q[2], q[3], q[0]]).inv()
    c = Rinv.apply(pad_centre_w - com)
    a = Rinv.apply(closing_w)
    ap = Rinv.apply(approach_w)
    third = np.cross(a, ap)
    d = obj.verts - c
    sel = (np.abs(d @ ap) < 0.015) & (np.abs(d @ third) < 0.012)
    if sel.sum() < 10:
        return None
    proj = d[sel] @ a
    return float(proj.max() - proj.min())


def _rank_to_tcp(hands, obj, pad_geo, com, q, E, density, mu, table_z, name, depth_off,
                 width_from_xsec=False):
    """Shared tail for external planners: score-ordered [(score, pos, approach, binormal,
    axis, width)] world-frame hands -> first 7-DOF TCP grasp passing the validity ladder.
    `width_from_xsec`: planner width is a PRE-SHAPE opening (close-until-contact execution
    model); replace it with the local object cross-section at the pose − SQUEEZE_M."""
    for sc, pos, app, bin_, ax, w_ap in hands:
        if app[2] > -0.45:                               # keep tilt <= ~63 deg (executor regime)
            continue
        if width_from_xsec:
            xs = _local_xsec(obj, q, com, pos + app * depth_off, bin_, app)
            if xs is None:
                continue
            w_ap = xs
        R_ours = Rot.from_matrix(np.column_stack([-ax, bin_, app]))
        rpy = R_ours.as_euler("xyz")
        width = float(np.clip(w_ap - SQUEEZE_M, 0.008, 0.079))
        pad_centre = pos + app * depth_off               # planner origin -> enclosed-slice centre
        tcp = pad_centre - R_ours.apply([0.0, 0.0, fg._z_off(width) + pad_geo["z_center"]])
        x = np.concatenate([tcp, rpy, [width]])
        # Table clearance: back off ALONG -approach (raising along +z destroys tilted grasps —
        # measured: a 47 deg gn1b grasp lifted clear off the object -> no_contact). After the
        # backoff the pads sit at a HIGHER object slice, so re-measure the local cross-section
        # there and rebuild the width/tcp (measured: without this, w stays sized for the old
        # slice -> no_contact on every deep gn1b grasp).
        deficit = (table_z + 0.0035 + 0.003) - fg.finger_min_world_z(x, pad_geo)
        if deficit > 0.0:
            step = deficit / max(-app[2], 0.3)
            if step > 0.025:
                continue                                  # too deep below table -> infeasible pose
            pad_centre = pad_centre - app * step
            if width_from_xsec:
                xs = _local_xsec(obj, q, com, pad_centre, bin_, app)
                if xs is None:
                    continue
                width = float(np.clip(xs - SQUEEZE_M, 0.008, 0.079))
            tcp = pad_centre - R_ours.apply([0.0, 0.0, fg._z_off(width) + pad_geo["z_center"]])
            x = np.concatenate([tcp, rpy, [width]])
            if fg.finger_min_world_z(x, pad_geo) < table_z + 0.002:
                continue                                  # still below clearance after rebuild
        sc_geo = fg.score_finger_grasp(obj, x, obj_com=com, obj_quat_wxyz=q, pad_geo=pad_geo,
                                       E=E, density=density, mu=mu, table_z=table_z)
        if sc_geo.get("status") != "ok":
            continue
        return {"x": x, "stress_top10": 1.0, "grip": 0.0, "align": 0.0, "pressure": None,
                "min_pad_area": 0.0, "width_face": None, "baseline": name, "gpd_score": sc}
    return {"x": None, "stress_top10": None, "baseline": name}


# ── Learned planners: Contact-GraspNet (ICRA 2021) + GraspNet-1Billion (CVPR 2020) ──────────────
# Original released code + pretrained weights, run in their own venvs via subprocess CLIs
# (learned_baselines/{cgn,gn1b}_infer.py — pure I/O glue, zero algorithmic edits). Input is a
# SINGLE-VIEW cloud (front-facing surface points + a table disc) from a virtual camera in their
# native tabletop viewing regime, in CAMERA frame (x right, y down, z forward) — matching their
# training distribution, unlike GPD which got the full privileged cloud.
LB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "learned_baselines")
TP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party")
CGN_TCP_OFF = 0.1034        # panda hand-base -> enclosed-slice centre along approach (CGN frame)


def _lookat_cam(cam_pos, target):
    """world_T_cam for an OpenCV-style camera (x right, y down, z forward) looking at target."""
    z = np.asarray(target, float) - np.asarray(cam_pos, float)
    z /= np.linalg.norm(z) + 1e-12
    up = np.array([0.0, 0.0, 1.0])
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x) + 1e-12
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, cam_pos
    return T


def _single_view_cloud(obj, R, com, table_z, cam_pos, n_pts=15000, seed=0):
    """Front-facing object surface points + table disc, in world frame."""
    rng = np.random.default_rng(seed)
    F = boundary_faces(obj.tets)[0]
    V = obj.verts
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    fn = np.cross(b - a, c - a)
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    fi = rng.choice(len(F), n_pts, p=areas / areas.sum())
    r1, r2 = rng.random(n_pts), rng.random(n_pts)
    s1 = np.sqrt(r1)
    pts = (1 - s1)[:, None] * a[fi] + (s1 * (1 - r2))[:, None] * b[fi] + (s1 * r2)[:, None] * c[fi]
    pw = R.apply(pts) + com
    nw = R.apply(fn[fi])
    vis = np.einsum("ij,ij->i", nw, np.asarray(cam_pos) - pw) > 0.0    # front-facing only
    pw = pw[vis]
    g = np.linspace(-0.16, 0.16, 55)
    gx, gy = np.meshgrid(g, g)
    keep = (gx**2 + gy**2) <= 0.16**2
    table = np.stack([gx[keep] + com[0], gy[keep] + com[1],
                      np.full(int(keep.sum()), table_z)], 1)
    return np.concatenate([pw, table]).astype(np.float32), pw.astype(np.float32)


# Two virtual viewpoints (standard multi-view usage): a ~54 deg tabletop view (their training
# regime) + a near-overhead view (~77 deg) so top-down approaches — the only ones a 45 mm
# finger can execute on a small tabletop object without stabbing the table — are proposed.
# NOTE (measured): the classic ~54 deg tabletop view yields ZERO executable proposals for our
# rig — its side-ish approaches put 45 mm fingers through the table on 3-4 cm objects. Steep
# views are the only ones whose proposals our workspace can execute; itself a finding.
LEARNED_VIEWS = (np.array([-0.10, 0.0, 0.45]), np.array([0.0, 0.0, 0.47]))


def _run_learned(venv_py, infer_script, obj, R, com, table_z, seed, pass_segment=False,
                 cam_off=None):
    """Build the single-view cloud, run the planner CLI in its venv, return raw GRASP_POSE
    token lists + world_T_cam. `pass_segment` additionally hands the object-only points as the
    planner's segment (Contact-GraspNet's segmented local-regions mode, its intended usage)."""
    cam_pos = com + (LEARNED_VIEWS[0] if cam_off is None else cam_off)
    T_wc = _lookat_cam(cam_pos, com)
    cloud_w, obj_w = _single_view_cloud(obj, R, com, table_z, cam_pos, seed=seed)
    Rcw = T_wc[:3, :3].T
    to_cam = lambda pts: ((pts - T_wc[:3, 3]) @ Rcw.T).astype(np.float32)
    fd, npy = tempfile.mkstemp(suffix=".npy", prefix="gm_lb_")
    os.close(fd)
    fd2, npy2 = tempfile.mkstemp(suffix=".npy", prefix="gm_lbseg_")
    os.close(fd2)
    try:
        np.save(npy, to_cam(cloud_w))
        np.save(npy2, to_cam(obj_w))
        env = {k: v for k, v in os.environ.items() if k not in ("LD_LIBRARY_PATH", "LIBRARY_PATH")}
        # cuda-12.1 bin FIRST: without it TF picks up the system's ancient ptxas 10.1, which
        # miscompiles XLA (measured: intermittent EMPTY prediction sets from Contact-GraspNet).
        env["PATH"] = "/usr/local/cuda-12.1/bin:/usr/local/bin:/usr/bin:/bin"
        # our compiled TF ops link against cuda-12.1's libcudart; without this the loader mixes
        # them with TF's bundled 12.2 copy -> CUDA_ERROR_ILLEGAL_ADDRESS / abort (measured,
        # deterministic). Minimal explicit path — deliberately NOT inheriting conda/ROS entries.
        env["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.1/lib64"
        base_cmd = [venv_py, infer_script, npy] + ([npy2] if pass_segment else [])
        out = ""
        for attempt in range(3):
            cmd = base_cmd + [str((int(seed) + 1013 * attempt) % 100000)]
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
            out = p.stdout
            if p.returncode == 0 and "GRASP_POSE" in out:
                break
            # Contact-GraspNet's legacy TF ops are nondeterministically flaky on RTX 4090 +
            # CUDA 12 (measured: identical seeded input -> 0 grasps / N grasps / abort across
            # runs). Retry with a shifted seed; empty output counts as a failed attempt.
            time.sleep(3.0)
    finally:
        os.unlink(npy)
        os.unlink(npy2)
    rows = [line.split() for line in out.splitlines() if line.startswith("GRASP_POSE")]
    return rows, T_wc


def _parse_rot_hand(tok, T_wc, cols):
    """GRASP_POSE token list with a 9-float `rot` -> (score, pos_w, approach_w, binormal_w,
    axis_w, width). `cols` = (approach_col, binormal_col, axis_col) in the planner's frame."""
    sc = float(tok[3])
    pos_c = np.array(tok[5:8], float)
    Rm = np.array(tok[9:18], float).reshape(3, 3)
    w = float(tok[tok.index("width") + 1])
    Rw = T_wc[:3, :3] @ Rm
    pos_w = T_wc[:3, :3] @ pos_c + T_wc[:3, 3]
    return sc, pos_w, Rw[:, cols[0]], Rw[:, cols[1]], Rw[:, cols[2]], w


def cgn_planner(obj, pad_geo, obj_com, obj_quat_wxyz, *, E=3e5, density=1000.0, mu=0.7,
                table_z=0.0, seed=0, yaw_max_deg=None, **_ignored):
    """Contact-GraspNet: frame x=closing, y=axis, z=approach; translation = hand base."""
    q = np.asarray(obj_quat_wxyz, float)
    R = Rot.from_quat([q[1], q[2], q[3], q[0]])
    com = np.asarray(obj_com, float)
    hands = []
    for k, off in enumerate(LEARNED_VIEWS):
        rows, T_wc = _run_learned(os.path.join(TP_DIR, "cgn_venv", "bin", "python"),
                                  os.path.join(LB_DIR, "cgn_infer.py"), obj, R, com, table_z,
                                  seed + k, pass_segment=True, cam_off=off)
        for tok in rows:
            sc, pos, app, bin_, ax, w = _parse_rot_hand(tok, T_wc, cols=(2, 0, 1))
            hands.append((sc, pos, app, bin_, ax, w))
    hands.sort(key=lambda h: -h[0])
    return _rank_to_tcp(hands, obj, pad_geo, com, q, E, density, mu, table_z, "cgn",
                        depth_off=CGN_TCP_OFF, width_from_xsec=True)


def gn1b_planner(obj, pad_geo, obj_com, obj_quat_wxyz, *, E=3e5, density=1000.0, mu=0.7,
                 table_z=0.0, seed=0, yaw_max_deg=None, **_ignored):
    """GraspNet-1Billion: frame x=approach, y=closing, z=axis; translation = grasp point,
    closing-region depth along approach in the `depth` field."""
    q = np.asarray(obj_quat_wxyz, float)
    R = Rot.from_quat([q[1], q[2], q[3], q[0]])
    com = np.asarray(obj_com, float)
    hands = []
    for k, off in enumerate(LEARNED_VIEWS):
        rows, T_wc = _run_learned(os.path.join(TP_DIR, "gn1b_venv", "bin", "python"),
                                  os.path.join(LB_DIR, "gn1b_infer.py"), obj, R, com, table_z,
                                  seed + k, cam_off=off)
        for tok in rows:
            sc, pos, app, bin_, ax, w = _parse_rot_hand(tok, T_wc, cols=(0, 1, 2))
            d = float(tok[tok.index("depth") + 1])
            hands.append((sc, pos, app, bin_, ax, w, d))
    hands.sort(key=lambda h: -h[0])
    return _rank_to_tcp([(sc, pos + app * (d * 0.5), app, bin_, ax, w)
                        for sc, pos, app, bin_, ax, w, d in hands],
                       obj, pad_geo, com, q, E, density, mu, table_z, "gn1b", depth_off=0.0,
                       width_from_xsec=True)
