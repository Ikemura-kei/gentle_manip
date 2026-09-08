"""Graspability geometry filter for the 200-object expansion.

Core question per candidate mesh: is there SOME gripper approach direction whose cross-body
span fits inside the jaw opening? We approximate the true minimum width of the (convex hull
of the) mesh by scanning a dense set of directions (icosphere-refined + every hull face
normal) and taking the smallest support-width. This is a valid SUFFICIENT test: if any scanned
direction clears the cap, a real feasible grasp axis exists (does not need to be the exact
global minimum — existence is all we need). It is mesh-shape agnostic (works for the
non-convex "crazy shapes" too, since convex hull width is an upper bound on what a well-placed
parallel jaw needs, per finger contact being on the hull-ish outer surface for a rigid two-point
grasp approximation used at the filtering stage; the actual grasp planner's FEM stage is the
real arbiter of graspability once a candidate mesh looks promising here).

Gripper cap: WIDTH_MAX = 0.079 m (79 mm) from the frozen planner
(`grasp_synthesis/smgrasp/finger_grasp_final.py:515`) — this is the authoritative number, not
the commonly-quoted "8 cm" (which is close but not exact: 88.9 mm is the raw pad-to-pad travel
at the URDF's fully-open joint angle per `xarm7_config.GRIPPER_CALIB_SEP[0] - GRIPPER_PAD_OFFSET`;
79 mm is the planner's own safety-derated search bound and is what actually gates a synthesizable
grasp, so it is the number this filter uses).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import trimesh

WIDTH_MAX_GRASP = 0.079     # m, gentle_manip/../grasp_synthesis/smgrasp/finger_grasp_final.py WIDTH_MAX
WIDTH_MAX_RAW_TRAVEL = 0.0889  # m, informational only (raw pad travel, NOT the graspability cap)


def _fibonacci_directions(n: int) -> np.ndarray:
    """n roughly-uniform directions on the unit hemisphere+ (antipodal directions give the
    same width, so a hemisphere is enough), via a Fibonacci lattice."""
    i = np.arange(0, n)
    phi = (1 + 5 ** 0.5) / 2
    theta = np.arccos(1 - (i + 0.5) / n)          # covers full sphere in z; fine, cheap
    az = 2 * np.pi * i / phi
    x = np.sin(theta) * np.cos(az)
    y = np.sin(theta) * np.sin(az)
    z = np.cos(theta)
    return np.stack([x, y, z], axis=1)


def min_width_directions(hull: trimesh.Trimesh, n_scan: int = 400) -> np.ndarray:
    """Candidate directions: hull face normals (captures face-parallel minima, the usual
    global minimum for faceted/boxy shapes) UNION a dense Fibonacci sweep (captures
    antipodal-edge minima for rounded/twisted/rope-like shapes where no two faces are
    parallel)."""
    normals = hull.face_normals
    fib = _fibonacci_directions(n_scan)
    dirs = np.concatenate([normals, fib], axis=0)
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs


@dataclass
class GeomReport:
    name: str
    ok_load: bool
    watertight: bool
    body_count: int
    orig_extents: Optional[np.ndarray] = None
    min_width_dir: Optional[np.ndarray] = None
    min_width_raw: Optional[float] = None          # at ORIGINAL scale (m)
    obb_extents_raw: Optional[np.ndarray] = None   # sorted ascending, at ORIGINAL scale
    suggested_scale: Optional[float] = None
    min_width_scaled: Optional[float] = None
    max_extent_scaled: Optional[float] = None
    thin_axis_scaled: Optional[float] = None       # smallest OBB extent after scale (resting thickness proxy)
    verdict: str = "reject"                        # "accept" | "reject" | "borderline"
    reason: str = ""
    error: str = ""


def analyze_mesh(mesh: trimesh.Trimesh, name: str,
                  target_min_width: float = 0.040,   # nominal; DR then scales x[0.6,1.6] (2026-09-08,
                  # user: stronger size randomization) so 0.040*1.6=64mm stays under the 79mm cap and
                  # 0.040*0.6=24mm stays graspable at the small end -- was 0.055 for the old x[0.9,1.2] DR
                  max_overall_extent: float = 0.11,   # don't let rescaling blow past a sane max size
                  min_overall_extent: float = 0.018,  # too-tiny meshes have too few MPM particles
                  min_thin_axis: float = 0.020,       # resting-thickness floor (soft, flagged not hard)
                  n_scan: int = 400) -> GeomReport:
    rep = GeomReport(name=name, ok_load=True, watertight=bool(mesh.is_watertight),
                      body_count=int(mesh.body_count) if hasattr(mesh, "body_count") else 1)
    try:
        if len(mesh.vertices) < 4 or len(mesh.faces) < 4:
            rep.error = "degenerate mesh (too few verts/faces)"
            return rep
        # Raw Objaverse/GLB downloads are almost always multi-body (separate primitives per
        # material/part) -- that is expected, NOT a defect; repair (largest-component + fill
        # holes / voxel remesh) happens at mesh_prep.py Stage 3. Here we just need an accurate
        # size estimate, so take the largest-by-area component before measuring, same as
        # mesh_prep.repair_watertight will do for real. Watertightness itself is informational
        # at this stage (not a reject reason) -- only a truly degenerate largest part is fatal.
        if rep.body_count > 1:
            parts = [p for p in mesh.split(only_watertight=False) if len(p.vertices) >= 4 and len(p.faces) >= 4]
            if len(parts) == 0:
                rep.error = "split() returned no usable parts"
                return rep
            # Keep every component that's a MEANINGFUL fraction of the biggest one by convex-hull
            # volume, then union them -- NOT "just take the single largest part": a real multi-part
            # object (mug body + handle, scissors' two blades) needs every part kept, while a flat
            # background plane/pedestal/decal (common scene dressing bundled into these GLBs) has
            # near-zero volume despite large area and gets dropped by this volume-relative cutoff.
            def _hull_vol(p):
                try:
                    return float(p.convex_hull.volume)
                except Exception:
                    return 0.0
            vols = [(_hull_vol(p), p) for p in parts]
            vmax = max(v for v, _ in vols) if vols else 0.0
            kept = [p for v, p in vols if vmax > 0 and v >= 0.02 * vmax]
            mesh = trimesh.util.concatenate(kept) if len(kept) > 1 else (kept[0] if kept else parts[0])
            if len(mesh.vertices) < 4 or len(mesh.faces) < 4:
                rep.error = "degenerate largest component(s)"
                return rep
        MAX_VERTS_FOR_HULL = 150_000
        hull_source = mesh
        if len(mesh.vertices) > MAX_VERTS_FOR_HULL:
            # A handful of Objaverse assets carry millions of vertices (dense showcase scans);
            # qhull cost/memory on the raw point set scales badly there and was observed driving
            # multi-GB RSS on this pipeline's first run. We only need an APPROXIMATE hull for the
            # min-width/extent estimate here, so subsample the POINTS (hull only needs points,
            # not faces) before hulling.
            idx = np.random.default_rng(0).choice(len(mesh.vertices), MAX_VERTS_FOR_HULL, replace=False)
            hull_source = trimesh.points.PointCloud(np.asarray(mesh.vertices)[idx])
        hull = hull_source.convex_hull
        ext = mesh.extents
        rep.orig_extents = ext
        dirs = min_width_directions(hull, n_scan=n_scan)
        verts = hull.vertices
        proj = verts @ dirs.T                      # (V, D)
        widths = proj.max(axis=0) - proj.min(axis=0)
        i = int(np.argmin(widths))
        rep.min_width_dir = dirs[i]
        rep.min_width_raw = float(widths[i])

        # OBB-ish proxy via PCA of hull vertices, for the "thin resting axis" + "overall size" checks
        c = verts.mean(0)
        cov = np.cov((verts - c).T)
        evals, evecs = np.linalg.eigh(cov)
        proj_pca = (verts - c) @ evecs
        obb_ext = proj_pca.max(0) - proj_pca.min(0)
        rep.obb_extents_raw = np.sort(obb_ext)

        if rep.min_width_raw <= 1e-6:
            rep.error = "degenerate min-width (flat/zero-volume mesh)"
            return rep

        # Pick the largest scale that still keeps min_width <= target AND overall extent <= max.
        s_for_width = target_min_width / rep.min_width_raw
        s_for_max = max_overall_extent / float(ext.max())
        s_for_min = min_overall_extent / float(ext.min()) if ext.min() > 1e-9 else 1e9
        s = min(s_for_width, s_for_max)
        s = max(s, s_for_min) if s_for_min <= s_for_width else s  # don't shrink below the tiny-floor unless forced
        rep.suggested_scale = float(s)

        mw = rep.min_width_raw * s
        maxext = float(ext.max()) * s
        thin = float(rep.obb_extents_raw[0]) * s
        rep.min_width_scaled = mw
        rep.max_extent_scaled = maxext
        rep.thin_axis_scaled = thin

        if mw > WIDTH_MAX_GRASP - 0.005:
            rep.verdict, rep.reason = "reject", f"min-width {mw*1000:.1f}mm too close to/over the {WIDTH_MAX_GRASP*1000:.0f}mm cap even at max useful scale"
        elif maxext < min_overall_extent:
            rep.verdict, rep.reason = "reject", f"too small even scaled up ({maxext*1000:.1f}mm)"
        elif thin < min_thin_axis:
            # HARD reject (user, 2026-09-08): any object whose smallest dimension is under 2cm
            # even after the best useful rescale is filtered out outright -- too thin to sit
            # >=15mm above the board under the TCP floor / too easily lost in MPM grid resolution.
            # (Previously this was only a "borderline" flag; the "== grasp axis" flat/disk-like
            # sub-case, e.g. a coin, and the "distinct axis, could anisotropically thicken" case
            # are now both covered by the same flat 2cm cutoff rather than a softer manual call.)
            rep.verdict, rep.reason = "reject", (
                f"height/thin-axis {thin*1000:.1f}mm < {min_thin_axis*1000:.0f}mm floor (hard cutoff)")
        else:
            rep.verdict, rep.reason = "accept", (
                f"min-width {mw*1000:.1f}mm, max-extent {maxext*1000:.1f}mm, thin-axis {thin*1000:.1f}mm "
                f"at scale {s:.3f}")
    except Exception as e:  # noqa: BLE001
        rep.error = f"{type(e).__name__}: {e}"
    return rep


def report_to_row(rep: GeomReport) -> dict:
    return {
        "name": rep.name,
        "ok_load": rep.ok_load,
        "watertight": rep.watertight,
        "body_count": rep.body_count,
        "orig_extents_mm": None if rep.orig_extents is None else [round(float(x) * 1000, 1) for x in rep.orig_extents],
        "min_width_raw_mm": None if rep.min_width_raw is None else round(rep.min_width_raw * 1000, 2),
        "suggested_scale": None if rep.suggested_scale is None else round(rep.suggested_scale, 4),
        "min_width_scaled_mm": None if rep.min_width_scaled is None else round(rep.min_width_scaled * 1000, 1),
        "max_extent_scaled_mm": None if rep.max_extent_scaled is None else round(rep.max_extent_scaled * 1000, 1),
        "thin_axis_scaled_mm": None if rep.thin_axis_scaled is None else round(rep.thin_axis_scaled * 1000, 1),
        "verdict": rep.verdict,
        "reason": rep.reason,
        "error": rep.error,
    }
