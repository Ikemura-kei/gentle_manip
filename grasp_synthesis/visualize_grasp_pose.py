"""Offline grasp-pose visualization — no Genesis required.

Runs CMA-ES to synthesize a grasp, then renders the mushroom mesh + both finger
meshes (positioned at the synthesized pose) as a matplotlib 3D scene.

Usage:
    uv run --project envs/sim python grasp_synthesis/visualize_grasp_pose.py
    uv run --project envs/sim python grasp_synthesis/visualize_grasp_pose.py \
        --obj-pos 0.475 -0.003 0.015   # override object position
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as Rot
import trimesh

ROOT = Path(__file__).resolve().parent.parent
GRASP_DIR = ROOT / "grasp_synthesis"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(GRASP_DIR) not in sys.path:
    sys.path.insert(0, str(GRASP_DIR))

from synth_utils import (          # noqa: E402
    build_object_sdf, grasp_cost, run_cmaes, sample_finger_surface,
    FINGER_SLOPE, FINGER_TO_TCP_Z, FINGER_GRIP_OFF,
)

MUSHROOM_MESH = str(ROOT / "gentle_manip/assets/objects/mushroom.obj")
LEFT_FINGER   = str(ROOT / "gentle_manip/assets/xarm/xarm_gripper/meshes/left_finger.STL")
RIGHT_FINGER  = str(ROOT / "gentle_manip/assets/xarm/xarm_gripper/meshes/right_finger.STL")
OBJ_SIZE = np.array([0.05, 0.05, 0.04])


def _finger_transform(x: np.ndarray, side: str) -> np.ndarray:
    """4×4 transform (world_T_finger) for the given side ('left'/'right')."""
    tcp_pos = np.asarray(x[:3], np.float64)
    tcp_rot = Rot.from_euler('xyz', x[3:6])
    w = float(x[6])
    z_off = FINGER_TO_TCP_Z + FINGER_SLOPE * (0.044 - w / 2.0)
    t_left = np.array([0.0, -(w / 2.0 + FINGER_GRIP_OFF), z_off])
    t_right = t_left + np.array([0.0, -(w + 2.0 * FINGER_GRIP_OFF), 0.0])

    if side == 'left':
        t = t_left
        R_local = np.eye(3)
    else:
        t = t_right
        R_local = Rot.from_quat([0.0, 0.0, 1.0, 0.0]).as_matrix()  # 180° around z

    R_world = tcp_rot.as_matrix()
    T = np.eye(4)
    T[:3, :3] = R_world @ R_local
    T[:3,  3] = tcp_pos + R_world @ t
    return T


def _mesh_triangles(mesh: trimesh.Trimesh, T: np.ndarray) -> np.ndarray:
    """Return (F, 3, 3) triangle vertex array after applying transform T."""
    verts = (T[:3, :3] @ mesh.vertices.T).T + T[:3, 3]  # (N, 3)
    return verts[mesh.faces]                              # (F, 3, 3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--obj-pos', type=float, nargs=3, default=[0.475, -0.003, 0.015],
                        metavar=('X', 'Y', 'Z'),
                        help='Object centroid in world frame (default: %(default)s)')
    parser.add_argument('--maxfevals', type=int, default=800)
    args = parser.parse_args()

    obj_pos = np.asarray(args.obj_pos, dtype=np.float64)
    print(f"Object centroid: {obj_pos}")

    # SDF + finger surface samples
    print("Building mushroom SDF …")
    sdf_fn   = build_object_sdf(MUSHROOM_MESH)
    left_pts = sample_finger_surface(LEFT_FINGER,  n=30)
    right_pts= sample_finger_surface(RIGHT_FINGER, n=30)

    # Synthesis
    tcp_z_min = float(obj_pos[2]) + FINGER_TO_TCP_Z - 0.04
    tcp_z_max = float(obj_pos[2]) + 0.25
    t_lb_xy = (obj_pos[:2] - 1.5 * OBJ_SIZE[:2]).tolist()
    t_ub_xy = (obj_pos[:2] + 1.5 * OBJ_SIZE[:2]).tolist()
    lb = t_lb_xy + [tcp_z_min, 0.8*np.pi, -0.25*np.pi, -0.25*np.pi, 0.028]
    ub = t_ub_xy + [tcp_z_max, 1.0*np.pi,  0.25*np.pi,  0.25*np.pi, 0.088]
    x0 = [(l+u)/2 for l, u in zip(lb, ub)]

    def objective(x):
        return grasp_cost(x, left_pts, right_pts, sdf_fn, obj_pos)

    print(f"Running CMA-ES ({args.maxfevals} evals) …")
    t0 = time.time()
    best_x, score = run_cmaes(objective, x0, 1.0, lb, ub, args.maxfevals)
    print(f"Done {time.time()-t0:.1f}s  cost={score:.4f}")
    print(f"  tcp_pos   = {best_x[:3].round(4)}")
    print(f"  rpy_deg   = {np.degrees(best_x[3:6]).round(2)}")
    print(f"  width     = {best_x[6]*1000:.1f} mm")

    # Finger centroids (world frame) for a quick sanity check
    from synth_utils import finger_world_pts
    lw, rw = finger_world_pts(best_x, left_pts, right_pts)
    print(f"  left  centroid (world): {lw.mean(0).round(4)}")
    print(f"  right centroid (world): {rw.mean(0).round(4)}")

    # Load meshes and build transforms
    mushroom   = trimesh.load(MUSHROOM_MESH, force='mesh')
    left_mesh  = trimesh.load(LEFT_FINGER,  force='mesh')
    right_mesh = trimesh.load(RIGHT_FINGER, force='mesh')

    T_mush  = np.eye(4); T_mush[:3, 3] = obj_pos
    T_left  = _finger_transform(best_x, 'left')
    T_right = _finger_transform(best_x, 'right')

    # Downsample meshes for fast plotting (every 10th face)
    step = 10
    mush_tris  = _mesh_triangles(mushroom,   T_mush )[::step]
    left_tris  = _mesh_triangles(left_mesh,  T_left )[::step]
    right_tris = _mesh_triangles(right_mesh, T_right)[::step]

    # Plot
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')

    def _add(tris, color, alpha=0.4):
        poly = Poly3DCollection(tris, alpha=alpha)
        poly.set_facecolor(color)
        poly.set_edgecolor('none')
        ax.add_collection3d(poly)

    _add(mush_tris,  'tan',       alpha=0.6)
    _add(left_tris,  'royalblue', alpha=0.5)
    _add(right_tris, 'firebrick', alpha=0.5)

    # Mark TCP and finger centroids
    tcp_pos = best_x[:3]
    ax.scatter(*tcp_pos,    c='gold',  s=80, zorder=5)
    ax.scatter(*lw.mean(0), c='blue',  s=60, zorder=5)
    ax.scatter(*rw.mean(0), c='red',   s=60, zorder=5)
    ax.scatter(*obj_pos,    c='black', s=80, zorder=5)

    # Grasp axis arrow
    grasp_axis = rw.mean(0) - lw.mean(0)
    ax.quiver(*lw.mean(0), *grasp_axis, length=1.0, normalize=False,
              color='purple', linewidth=2)

    # Equal aspect trick
    all_pts = np.vstack([mush_tris.reshape(-1, 3),
                         left_tris.reshape(-1, 3),
                         right_tris.reshape(-1, 3)])
    mn, mx = all_pts.min(0), all_pts.max(0)
    ctr = (mn + mx) / 2; rng = (mx - mn).max() / 2 + 0.02
    ax.set_xlim(ctr[0]-rng, ctr[0]+rng)
    ax.set_ylim(ctr[1]-rng, ctr[1]+rng)
    ax.set_zlim(ctr[2]-rng, ctr[2]+rng)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title(f'Grasp pose  cost={score:.4f}  w={best_x[6]*1e3:.0f}mm\n'
                 f'rpy=[{", ".join(f"{d:.1f}" for d in np.degrees(best_x[3:6]))}]°')
    legend_handles = [
        Patch(facecolor='tan',       alpha=0.6, label='mushroom'),
        Patch(facecolor='royalblue', alpha=0.5, label='left finger'),
        Patch(facecolor='firebrick', alpha=0.5, label='right finger'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='gold',  markersize=8, label='TCP'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='blue',  markersize=7, label='L centroid'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='red',   markersize=7, label='R centroid'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='black', markersize=8, label='object'),
        Line2D([0],[0], color='purple', linewidth=2, label='grasp axis'),
    ]
    ax.legend(handles=legend_handles, loc='upper left')
    plt.tight_layout()
    plt.savefig(str(GRASP_DIR / "grasp_pose_vis.png"), dpi=150)
    print(f"\nFigure saved → grasp_synthesis/grasp_pose_vis.png")
    plt.show()


if __name__ == '__main__':
    main()
