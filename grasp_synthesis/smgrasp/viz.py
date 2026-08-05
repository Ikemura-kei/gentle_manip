"""Visualization helpers for the FEM stress field (sanity-check M1–M4 output).

Two outputs:
  * a headless matplotlib PNG — the object's boundary surface colored by von Mises stress
    under a grasp-squeeze load, plus the two contact points;
  * a ParaView `.vtu` (via meshio) carrying per-tet von Mises + the displacement field,
    for full interactive inspection (cross-sections, warping) if a display is available.

The "grasp squeeze" applies two opposing inward contact forces along an axis — the same
stress a parallel-jaw grasp induces, which is exactly what Q_SM minimizes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .stressmap import contact_load_basis


def von_mises(sigma_voigt: np.ndarray) -> np.ndarray:
    """(…,6) Voigt stress -> (…,) von Mises scalar."""
    s = np.asarray(sigma_voigt, float)
    xx, yy, zz, xy, yz, zx = (s[..., i] for i in range(6))
    return np.sqrt(0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2)
                   + 3.0 * (xy ** 2 + yz ** 2 + zx ** 2))


def grasp_squeeze(obj, axis: int = 1, force: float = 0.02):
    """Two opposing inward point contacts on the ±`axis` faces (self-equilibrated).
    Returns (points (2,3), forces (2,3), u (ndof,), sigma_voigt (M,6))."""
    fem = obj.fem
    v = fem.verts
    others = [i for i in range(3) if i != axis]
    radial = v[:, others[0]] ** 2 + v[:, others[1]] ** 2       # distance² from the axis line
    amax, amin = v[:, axis].max(), v[:, axis].min()
    hi_face = np.where(v[:, axis] > amax - 1e-6)[0]
    lo_face = np.where(v[:, axis] < amin + 1e-6)[0]
    hi = hi_face[np.argmin(radial[hi_face])]                   # face point nearest the axis
    lo = lo_face[np.argmin(radial[lo_face])]
    pts = v[[hi, lo]]
    ea = np.eye(3)[axis]
    f = np.stack([-force * ea, force * ea])                    # both push INWARD
    Lc, _ = contact_load_basis(v, fem.tets, pts)
    u, _ = fem.solve_free(Lc @ f.reshape(-1))
    return pts, f, u, fem.element_stress(u)


def squeeze_at(obj, points: np.ndarray, force: Optional[float] = None):
    """Press two given contact points TOWARD each other (self-equilibrated).
    Returns (points (2,3), forces (2,3), u, sigma_voigt (M,6))."""
    fem = obj.fem
    pts = np.asarray(points, float).reshape(2, 3)
    d = pts[1] - pts[0]
    d /= (np.linalg.norm(d) + 1e-12)
    if force is None:
        force = 0.02 * float(np.abs(fem.verts).max())          # scale-relative default
    f = np.stack([force * d, -force * d])                      # each toward the other
    Lc, _ = contact_load_basis(fem.verts, fem.tets, pts)
    u, _ = fem.solve_free(Lc @ f.reshape(-1))
    return pts, f, u, fem.element_stress(u)


def find_ear_contacts(obj, up: int = 1, top_frac: float = 0.75) -> np.ndarray:
    """Two OUTER-surface contact points, ONE per ear, for an inward squeeze of a bunny-like mesh.

    Take the upper region (top_frac of the up-axis range — set it high enough to exclude the eyes),
    find its principal HORIZONTAL axis (the ears' splay direction), SPLIT the region in two along
    that axis (median), and in EACH half take the point most extreme along the axis — i.e. the outer
    face of that ear. Splitting first guarantees one contact per ear (a global two-extreme pick can
    put both on one ear / the head); pressing them toward each other loads the ears from OUTSIDE in.
    Returns (2,3)."""
    v = obj.verts
    a = v[:, up]
    top = v[a > a.min() + top_frac * (a.max() - a.min())]      # ear zone only (exclude eyes/head)
    horiz = [i for i in range(3) if i != up]
    ctr = top[:, horiz].mean(0)
    axis = np.linalg.eigh((top[:, horiz] - ctr).T @ (top[:, horiz] - ctr))[1][:, -1]  # splay axis
    proj = (top[:, horiz] - ctr) @ axis
    med = np.median(proj)
    left, right = top[proj < med], top[proj >= med]            # the two ears
    lp = (left[:, horiz] - ctr) @ axis
    rp = (right[:, horiz] - ctr) @ axis
    return np.stack([left[int(np.argmin(lp))], right[int(np.argmax(rp))]])  # outer face of each ear


find_ear_tips = find_ear_contacts                             # back-compat alias


def boundary_faces(tets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Surface triangles of a tet mesh + the parent tet index of each (faces used once)."""
    faces = np.concatenate([tets[:, [0, 2, 1]], tets[:, [0, 1, 3]],
                            tets[:, [0, 3, 2]], tets[:, [1, 2, 3]]], axis=0)
    parent = np.tile(np.arange(len(tets)), 4)
    key = np.sort(faces, axis=1)
    _, idx, counts = np.unique(key, axis=0, return_index=True, return_counts=True)
    surf = idx[counts == 1]
    return faces[surf], parent[surf]


# Paper (Pan, Gao & Manocha 2020, Fig. 1) uses a blue(low)-white-red(high) map with red
# contact-force arrows — reproduced here.
PAPER_CMAP = "bwr"


def _face_colors(obj, sigma_voigt, cmap):
    import matplotlib.pyplot as plt

    vm = von_mises(sigma_voigt)
    tri, parent = boundary_faces(obj.tets)
    fc = vm[parent]                                            # per-boundary-face von Mises
    vmin, vmax = float(fc.min()), float(np.percentile(fc, 99))
    norm = plt.Normalize(vmin, vmax)
    return obj.verts[tri], plt.get_cmap(cmap)(norm(fc)), norm


def gripper_pads(center, axis, width, pad_half):
    """Two square parallel-jaw pad footprints (each 4 corners) at center ± (width/2)·axis, in the
    plane perpendicular to the closing axis. For drawing the gripper jaws at a grasp pose."""
    a = np.asarray(axis, float); a /= np.linalg.norm(a) + 1e-12
    t = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0.0, 1, 0])
    u = np.cross(a, t); u /= np.linalg.norm(u); v = np.cross(a, u)
    center = np.asarray(center, float)
    corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    quads = []
    for sign in (+1.0, -1.0):
        c = center + sign * (width / 2.0) * a
        quads.append([c + s1 * pad_half * u + s2 * pad_half * v for s1, s2 in corners])
    return quads


def gripper_cubes(center, axis, hw_left, hw_right, pad_half, thickness=None):
    """Two 3D box jaws (a cube proxy for the finger), footprint 2·pad_half square ⟂ the closing axis,
    each of the given `thickness` along the axis. The INNER face of each jaw sits at distance
    hw_left / hw_right from `center` along ∓axis (place these at the OUTERMOST contact point so the jaw
    rests on the surface without penetrating) and the box extends OUTWARD (away from the object).
    Returns a list of 12 quad faces (6 per jaw) for Poly3DCollection."""
    a = np.asarray(axis, float); a /= np.linalg.norm(a) + 1e-12
    t = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0.0, 1, 0])
    u1 = np.cross(a, t); u1 /= np.linalg.norm(u1); u2 = np.cross(a, u1)
    center = np.asarray(center, float)
    thickness = thickness if thickness is not None else pad_half
    corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    faces = []
    for hw, out in ((hw_left, -1.0), (hw_right, +1.0)):          # left jaw outward = −a, right = +a
        inner = center + out * hw * a
        outer = inner + out * thickness * a
        ic = [inner + s1 * pad_half * u1 + s2 * pad_half * u2 for s1, s2 in corners]
        oc = [outer + s1 * pad_half * u1 + s2 * pad_half * u2 for s1, s2 in corners]
        faces += [ic, oc]                                        # inner + outer faces
        faces += [[ic[i], ic[(i + 1) % 4], oc[(i + 1) % 4], oc[i]] for i in range(4)]  # 4 sides
    return faces


def _draw(ax, tris, colors, obj, points, forces, *, edges=True, pads=None):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    polys = list(tris)
    facecolors = np.asarray(colors, float)
    if facecolors.ndim == 2 and facecolors.shape[1] == 3:        # RGB -> RGBA
        facecolors = np.hstack([facecolors, np.ones((len(facecolors), 1))])
    if pads is not None:                                         # merge the two gripper jaws into the
        polys = polys + [np.asarray(q, float) for q in pads]     # SAME collection so matplotlib's
        gray = np.tile([0.15, 0.15, 0.15, 0.4], (len(pads), 1))  # per-face zsort occludes them
                                                                 # (alpha 0.4 = see the stress through the jaws)
        facecolors = np.vstack([facecolors, gray])               # correctly against the object (a
                                                                 # separate collection would z-order
                                                                 # as one unit and vanish behind it)
    ax.add_collection3d(Poly3DCollection(
        polys, facecolors=facecolors, edgecolors=("k" if edges else "none"),
        linewidths=0.08 if edges else 0.0))
    if points is not None:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="k", s=45,
                   marker="o", depthshade=False)
    if forces is not None and points is not None:                # red inward force arrows (paper style)
        span = float(np.abs(obj.verts).max())
        s = 0.5 * span / (np.linalg.norm(forces, axis=1).max() + 1e-12)
        ax.quiver(points[:, 0], points[:, 1], points[:, 2],
                  forces[:, 0] * s, forces[:, 1] * s, forces[:, 2] * s,
                  color="red", linewidth=2.2, arrow_length_ratio=0.35)
    lim = np.abs(obj.verts).max() * 1.02
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()


def _view(ax, elev, azim, up_axis):
    try:
        ax.view_init(elev=elev, azim=azim, vertical_axis=up_axis)   # matplotlib >= 3.5
    except TypeError:
        ax.view_init(elev=elev, azim=azim)


def render_png(obj, sigma_voigt: np.ndarray, out: str, *, points: Optional[np.ndarray] = None,
               forces: Optional[np.ndarray] = None, pads=None, title: str = "von Mises stress",
               cmap: str = PAPER_CMAP, up_axis: str = "z", views=((20, -60), (20, 120))) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tris, colors, norm = _face_colors(obj, sigma_voigt, cmap)
    fig = plt.figure(figsize=(6.0 * len(views), 5.6))
    for k, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, len(views), k + 1, projection="3d")
        _draw(ax, tris, colors, obj, points, forces, pads=pads)
        _view(ax, elev, azim, up_axis)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=fig.axes, shrink=0.6, label="von Mises stress (E=1 units)")
    fig.suptitle(title, y=0.97, fontsize=13)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def render_rotation_video(obj, sigma_voigt: np.ndarray, out: str, *,
                          points: Optional[np.ndarray] = None, forces: Optional[np.ndarray] = None,
                          pads=None, cmap: str = PAPER_CMAP, n_frames: int = 60, fps: int = 20,
                          elev: float = 18.0, up_axis: str = "z", title: str = "") -> str:
    """Turntable video (azimuth 0→360) of the stress-colored mesh. Reuses one collection."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    tris, colors, norm = _face_colors(obj, sigma_voigt, cmap)
    edges = len(tris) < 4000                                   # edges only help on coarse meshes
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    _draw(ax, tris, colors, obj, points, forces, edges=edges, pads=pads)
    if title:
        ax.set_title(title, fontsize=12)
    frames = []
    for az in np.linspace(0.0, 360.0, n_frames, endpoint=False):
        _view(ax, elev, az, up_axis)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=fps)
    return out


def export_vtu(obj, sigma_voigt: np.ndarray, u: Optional[np.ndarray], out: str) -> str:
    """Write a ParaView .vtu: per-tet von Mises + displacement point field."""
    import meshio

    cell_data = {"von_mises": [von_mises(sigma_voigt)]}
    point_data = {}
    if u is not None:
        point_data["displacement"] = np.asarray(u, float).reshape(-1, 3)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    meshio.write_points_cells(out, obj.verts, [("tetra", obj.tets)],
                              cell_data=cell_data, point_data=point_data)
    return out
