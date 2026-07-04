"""Procedural mesh deformation for object SHAPE domain randomization (genesis-free, trimesh).

Mild, near-diffeomorphic deformations of a nominal mesh so a sim-trained policy sees a
distribution of shapes (banana curvature, tapered thickness, twist, organic lumps) without
the cost/opacity of a full LDDMM diffeomorphism. All operators act along the object's LONG
axis (from the AABB), so `bend` is literally curvature. Magnitudes are meant to be small; a
validity guard (positive signed volume, bounded volume change) rejects+retries degenerate
draws so MPM always gets a clean watertight volume (connectivity is never changed, only
vertex positions, so trimesh watertightness is preserved by construction).

Units: meshes are in meters. Angles in radians. `bend`/`twist` are total angles over the
object's length; `taper` is a fractional thickness change end-to-end; `rbf` is a bump
magnitude as a fraction of the object's size.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import trimesh


def _axes(verts: np.ndarray):
    """(long_axis, perp_axis, third_axis) indices from the AABB extents (longest first)."""
    ext = verts.max(0) - verts.min(0)
    order = np.argsort(ext)[::-1]        # longest -> shortest
    return int(order[0]), int(order[1]), int(order[2])


def bend(verts: np.ndarray, beta: float, L: int, P: int) -> np.ndarray:
    """Barr bend: rotate each vertex about the third axis by an angle proportional to its
    position along L, bending the L-axis into an arc in the L-P plane. beta = total bend angle."""
    if abs(beta) < 1e-6:
        return verts
    v = verts.copy()
    s = v[:, L] - v[:, L].mean()
    length = v[:, L].max() - v[:, L].min()
    if length < 1e-9:
        return verts
    kappa = beta / length               # rad per meter
    phi = kappa * s
    R = 1.0 / kappa
    p = v[:, P] - v[:, P].mean()         # perpendicular offset from centerline
    # centerline arc + the perpendicular offset rotated to stay normal to the arc
    v[:, L] = R * np.sin(phi) - p * np.sin(phi) + v[:, L].mean() - 0.0
    v[:, P] = R * (1.0 - np.cos(phi)) + p * np.cos(phi) + v[:, P].mean() - R * 0.0
    # recenter L so the object doesn't drift off its default pose
    v[:, L] -= v[:, L].mean() - verts[:, L].mean()
    v[:, P] -= v[:, P].mean() - verts[:, P].mean()
    return v


def taper(verts: np.ndarray, factor: float, L: int, P: int, Q: int) -> np.ndarray:
    """Linearly scale the two perpendicular axes from (1-factor) at one end to (1+factor) at
    the other — thicker/thinner along the length."""
    if abs(factor) < 1e-6:
        return verts
    v = verts.copy()
    s = v[:, L] - v[:, L].mean()
    half = max(1e-9, (v[:, L].max() - v[:, L].min()) / 2.0)
    f = 1.0 + factor * (s / half)
    for ax in (P, Q):
        c = v[:, ax].mean()
        v[:, ax] = c + (v[:, ax] - c) * f
    return v


def twist(verts: np.ndarray, angle: float, L: int, P: int, Q: int) -> np.ndarray:
    """Rotate the perpendicular (P,Q) plane about L by an angle proportional to position along L."""
    if abs(angle) < 1e-6:
        return verts
    v = verts.copy()
    s = v[:, L] - v[:, L].mean()
    length = max(1e-9, v[:, L].max() - v[:, L].min())
    phi = angle * s / length
    cp, cq = v[:, P].mean(), v[:, Q].mean()
    dp, dq = v[:, P] - cp, v[:, Q] - cq
    v[:, P] = cp + dp * np.cos(phi) - dq * np.sin(phi)
    v[:, Q] = cq + dp * np.sin(phi) + dq * np.cos(phi)
    return v


def axis_scale(verts: np.ndarray, factor: float, axis: int) -> np.ndarray:
    """Anisotropic scale along ONE world axis (x/y/z) about the centroid — makes the object
    wider/narrower on that axis independent of the uniform size scale."""
    if abs(factor - 1.0) < 1e-6:
        return verts
    v = verts.copy()
    c = v[:, axis].mean()
    v[:, axis] = c + (v[:, axis] - c) * factor
    return v


def rbf_bumps(mesh: trimesh.Trimesh, mag: float, n_bumps: int, rng: np.random.Generator) -> np.ndarray:
    """Add smooth low-frequency lumps: displace vertices along their normals by a sum of a few
    Gaussians centered at random surface points. mag is a fraction of the object's size."""
    if mag < 1e-6 or n_bumps <= 0:
        return mesh.vertices
    v = mesh.vertices.copy()
    size = float(np.linalg.norm(v.max(0) - v.min(0)))
    sigma = 0.25 * size
    centers = v[rng.integers(0, len(v), size=n_bumps)]
    amps = rng.uniform(-1.0, 1.0, size=n_bumps) * mag * size
    disp = np.zeros(len(v))
    for c, a in zip(centers, amps):
        disp += a * np.exp(-((np.linalg.norm(v - c, axis=1) / sigma) ** 2))
    return v + mesh.vertex_normals * disp[:, None]


def _valid(nominal: trimesh.Trimesh, deformed: trimesh.Trimesh) -> bool:
    """Non-degenerate: positive signed volume, volume within 0.3x–3x of nominal (no collapse/blowup)."""
    vol = deformed.volume
    if not np.isfinite(vol) or vol <= 0:
        return False
    ratio = vol / max(1e-12, nominal.volume)
    return 0.3 < ratio < 3.0


def deform_mesh(mesh: trimesh.Trimesh, params: Dict[str, float],
                rng: Optional[np.random.Generator] = None, tries: int = 4) -> trimesh.Trimesh:
    """Apply bend/taper/twist/rbf along the long axis, then an anisotropic scale along one world
    axis (params keys: bend, twist, taper, rbf[, rbf_n], axis_scale[, axis_scale_ax]). Retries
    with halved magnitude on a degenerate result; falls back to the nominal mesh."""
    rng = rng or np.random.default_rng()
    L, P, Q = _axes(mesh.vertices)
    ax = int(params.get("axis_scale_ax", rng.integers(3))) if "axis_scale" in params else None
    scale = 1.0
    for _ in range(tries):
        v = mesh.vertices.copy()
        v = bend(v, params.get("bend", 0.0) * scale, L, P)
        v = taper(v, params.get("taper", 0.0) * scale, L, P, Q)
        v = twist(v, params.get("twist", 0.0) * scale, L, P, Q)
        if ax is not None:                          # anisotropic scale on one axis (retry-tempered)
            v = axis_scale(v, 1.0 + (params["axis_scale"] - 1.0) * scale, ax)
        out = trimesh.Trimesh(vertices=v, faces=mesh.faces, process=False)
        if params.get("rbf", 0.0) > 0:
            out = trimesh.Trimesh(
                vertices=rbf_bumps(out, params["rbf"] * scale, int(params.get("rbf_n", 3)), rng),
                faces=mesh.faces, process=False)
        if _valid(mesh, out):
            return out
        scale *= 0.5                     # too aggressive -> gentler retry
    return mesh                          # give up -> nominal (never spawn a bad mesh)


def save_deformed(nominal_path, params: Dict[str, float], rng, out_dir) -> Path:
    """Deform the mesh at nominal_path and write it to out_dir as a temp .obj; return the path."""
    mesh = trimesh.load(str(nominal_path), process=False, force="mesh")
    out = deform_mesh(mesh, params, rng)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{Path(nominal_path).stem}_deformed_{rng.integers(1_000_000):06d}.obj"
    out.export(str(dst))
    return dst
