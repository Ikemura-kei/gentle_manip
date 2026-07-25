"""
examples/grasp_synthesis_qsm.py

Synthesize a grasp pose by optimizing the analytic stress-minimization metric Q_SM
directly with CMA-ES over the 7-DOF grasp x = [tx,ty,tz, roll,pitch,yaw, width].

Q_SM is a DROP-IN REPLACEMENT for the hand-crafted SDF grasp-QUALITY metric. No
simulator is used at any stage. The change vs. the user's existing pipeline:

    keep    : feasibility penalties  -> penetration, ground-plane SDF, tcp_height
    DROP    : quality terms          -> w_nearness, w_normal, w_align   (Q_SM subsumes)
    add     : - w_qsm * Q_SM(contacts(x))

Q_SM cannot see the table / arm reach / a finger buried in the object, so those stay
as penalties. Everything Q_SM *can* see (contact geometry, force closure, fragility)
it scores better than the hand-tuned quality terms.

Cost note: Q_SM in the inner loop REQUIRES the active-set acceleration (CLAUDE.md 5.4)
to be tractable (minutes vs. hours per synthesis). A is fixed per object; only the
contact columns of B are rebuilt per eval via back-substitution.

------------------------------------------------------------------------------------
TODO for the coding agent -- wire these to the real modules:
  from smgrasp import build_elastic_object, q_sm, sample_contacts, ContactSet, MetricConfig
  from core.utils.math_utils import homo_from_t_R, homo_transform
  from core.grasp_synthesis.basic import grasp_synthesis_objective   # user's geom obj
  from core.utils.sdf_utils import mesh_to_sdf                       # user's SDF
  import cma
------------------------------------------------------------------------------------
"""

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rot

# ----- placeholders: replace with real imports (see header) ------------------------
from smgrasp import build_elastic_object, q_sm, sample_contacts, ContactSet, MetricConfig  # TODO
from core.utils.math_utils import homo_from_t_R, homo_transform                            # TODO
import cma                                                                                 # TODO


# ----------------------------------------------------------------------------------
# Gripper constants -- verbatim from the user's pipeline so finger placement matches.
# ----------------------------------------------------------------------------------
FINGER_MOVEMENT_SLOPE = -0.23529411763
FINGER_TO_TCP_Z_OFFSET = 0.10645
FINGER_GRIPPER_OFFSET = 0.021


def contact_surface_from_finger(finger_mesh, local_contact_normal, angle_tol_deg=30.0):
    """Extract the gripping-pad submesh: faces whose outward normal (in the finger's
    OWN mesh frame) aligns with `local_contact_normal` (the closing/contact direction)
    within angle_tol_deg. Prefer cropping the pad in CAD instead if you can -- that
    avoids having to know this local direction. Returns a trimesh."""
    n = np.asarray(local_contact_normal, float)
    n = n / (np.linalg.norm(n) + 1e-12)
    cos_thr = np.cos(np.deg2rad(angle_tol_deg))
    keep = np.where(finger_mesh.face_normals @ n > cos_thr)[0]
    if len(keep) == 0:
        raise ValueError("No pad faces found; check local_contact_normal / tolerance.")
    return finger_mesh.submesh([keep], append=True)


def _finger_transforms_in_object(x, T_obj_to_world):
    """(T_left_finger_to_object, T_right_finger_to_object) using the user's exact
    TCP->finger convention. x = [tx,ty,tz, roll,pitch,yaw, gripper_width]."""
    t_world_to_tcp = x[:3][None, ...]
    R_world_to_tcp = Rot.from_euler('xyz', [x[3], x[4], x[5]]).as_matrix()[None, ...]
    T_world_to_tcp = homo_from_t_R(t_world_to_tcp, R_world_to_tcp)
    T_tcp_to_world = np.linalg.inv(T_world_to_tcp)

    w = x[6]
    t_left = np.array([0.0,
                       -w / 2.0 - FINGER_GRIPPER_OFFSET,
                       FINGER_TO_TCP_Z_OFFSET + FINGER_MOVEMENT_SLOPE * (0.044 - w / 2.0)])[None, ...]
    R_left = Rot.from_quat(np.array([0.0, 0.0, 0.0, 1.0])[None, ...]).as_matrix()
    T_left_finger_to_tcp = homo_from_t_R(t_left, R_left)

    t_right = np.array([0.0, -w - FINGER_GRIPPER_OFFSET * 2.0, 0.0])[None, ...]
    R_right = Rot.from_quat(np.array([0.0, 0.0, 1.0, 0.0])[None, ...]).as_matrix()
    T_right_finger_to_left = homo_from_t_R(t_right, R_right)

    T_left_finger_to_world = T_left_finger_to_tcp @ T_tcp_to_world
    T_right_finger_to_world = T_right_finger_to_left @ T_left_finger_to_world
    T_left_finger_to_object = T_left_finger_to_world @ np.linalg.inv(T_obj_to_world)
    T_right_finger_to_object = T_right_finger_to_world @ np.linalg.inv(T_obj_to_world)
    return T_left_finger_to_object, T_right_finger_to_object


def _place_finger_com(finger_mesh, T_finger_to_object, obj_com):
    """Finger mesh -> object COM frame (mirrors the user's homo_transform usage)."""
    V = finger_mesh.vertices[None, ...]
    V_obj = homo_transform(V, np.linalg.inv(T_finger_to_object))[0]
    return trimesh.Trimesh(vertices=V_obj - obj_com[None, :], faces=finger_mesh.faces, process=False)


def contacts_from_pose(x, obj_mesh_com, left_pad_mesh, right_pad_mesh,
                       T_obj_to_world, obj_com, mu=0.5, n_per_patch=6):
    """Place both gripping PADS at x, sample patch contacts on the object surface
    (COM frame). Pass the pad submeshes (contact_surface_from_finger or a CAD crop),
    NOT the full finger meshes -- the full mesh would yield spurious back/side contacts.
    Contact normals are taken from the OBJECT surface inside sample_contacts.
    Returns a merged ContactSet, or None if either jaw makes no contact."""
    T_l, T_r = _finger_transforms_in_object(x, T_obj_to_world)
    left_placed = _place_finger_com(left_pad_mesh, T_l, obj_com)
    right_placed = _place_finger_com(right_pad_mesh, T_r, obj_com)

    cs_l = sample_contacts(obj_mesh_com, left_placed, np.eye(4), mu=mu, n_per_patch=n_per_patch)
    cs_r = sample_contacts(obj_mesh_com, right_placed, np.eye(4), mu=mu, n_per_patch=n_per_patch)
    if cs_l is None or cs_r is None or len(cs_l.points) == 0 or len(cs_r.points) == 0:
        return None
    return ContactSet(
        points=np.concatenate([cs_l.points, cs_r.points], axis=0),
        normals=np.concatenate([cs_l.normals, cs_r.normals], axis=0),
        mu=mu,
    )


# ----------------------------------------------------------------------------------
# The CMA-ES objective (minimize).  Q_SM is a QUALITY -> subtract it.
#
# `feasibility_penalty(x)` should reuse the user's EXISTING SDF machinery with the
# quality weights zeroed:  grasp_synthesis_objective(x, ...) with
# w_nearness = w_normal = w_align = 0, keeping w_penetration, ground SDF, w_tcp_height.
# Optionally keep a SMALL w_nearness as pure landscape shaping (see CLAUDE.md 9.4).
# ----------------------------------------------------------------------------------
def qsm_grasp_objective(x, obj, obj_mesh_com, left_pad_mesh, right_pad_mesh,
                        T_obj_to_world, obj_com, cfg, feasibility_penalty,
                        w_qsm=1.0, mu=0.5, n_per_patch=6, min_contacts=2, big=1e9):
    feas = float(feasibility_penalty(x))                    # penetration (full mesh) + ground + tcp_height
    contacts = contacts_from_pose(x, obj_mesh_com, left_pad_mesh, right_pad_mesh,
                                  T_obj_to_world, obj_com, mu=mu, n_per_patch=n_per_patch)
    if contacts is None or len(contacts.points) < min_contacts:
        return feas + big                                   # no valid contact
    q = q_sm(obj, contacts, cfg)                            # analytic, sim-free
    if q <= 0.0:
        return feas + big * 1e-3                            # not force closure (softer than no-contact)
    return feas - w_qsm * float(q)                          # minimize -> maximize Q_SM


# ----------------------------------------------------------------------------------
# Driver: single CMA-ES run over qsm_grasp_objective -> best grasp pose.
# ----------------------------------------------------------------------------------
def synthesize_grasp_qsm(object_mesh_path, left_pad_path, right_pad_path,
                         T_obj_to_world, obj_center_world, feasibility_penalty,
                         cfg=None, w_qsm=1.0, mu=0.5, n_per_patch=6,
                         maxfevals=800, object_size=np.array([0.05, 0.05, 0.04])):
    """`left_pad_path` / `right_pad_path` are the gripping-PAD meshes (CAD crop, or
    contact_surface_from_finger applied once), NOT the full finger STLs. The full
    finger meshes are still used by your existing penetration penalty inside
    `feasibility_penalty` -- leave that untouched."""
    if cfg is None:
        cfg = MetricConfig(nu=0.33)

    # object precompute (ONCE): tetrahedralize + factor + assemble A, B (+ active set)
    obj = build_elastic_object(object_mesh_path, cfg)
    obj_com = np.asarray(obj.com, dtype=float)
    obj_mesh = trimesh.load(object_mesh_path)
    obj_mesh_com = trimesh.Trimesh(vertices=obj_mesh.vertices - obj_com,
                                   faces=obj_mesh.faces, process=False)
    left_pad_mesh = trimesh.load(left_pad_path)
    right_pad_mesh = trimesh.load(right_pad_path)

    # bounds mirror the user's pipeline (translation around the object; roll ~ pi)
    lb = (obj_center_world - 1.5 * object_size).tolist() + [0.8 * np.pi, -0.2 * np.pi, -0.2 * np.pi, 0.028]
    ub = (obj_center_world + 1.5 * object_size).tolist() + [1.0 * np.pi,  0.2 * np.pi,  0.2 * np.pi, 0.088]
    x0 = [(l + u) / 2 for l, u in zip(lb, ub)]
    stds = np.maximum(0.25 * (np.asarray(ub) - np.asarray(lb)), 1e-6)
    opts = {"bounds": [lb, ub], "maxfevals": maxfevals, "CMA_stds": stds.tolist(), "seed": 2567}

    args = (obj, obj_mesh_com, left_pad_mesh, right_pad_mesh,
            T_obj_to_world, obj_com, cfg, feasibility_penalty, w_qsm, mu, n_per_patch)

    es = cma.CMAEvolutionStrategy(x0, 1.0, opts)
    es.optimize(lambda x: qsm_grasp_objective(np.asarray(x, float), *args))
    best_x = np.asarray(es.result.xbest, dtype=float)
    best_score = float(es.result.fbest)          # = feas - w_qsm * Q_SM at the optimum
    return best_x, best_score


if __name__ == "__main__":
    # Wire `feasibility_penalty` to grasp_synthesis_objective with quality weights = 0,
    # then call synthesize_grasp_qsm(...). This block documents the call shape only.
    print("Wire the TODO imports and feasibility_penalty, then call synthesize_grasp_qsm().")
