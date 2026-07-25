import numpy as np



""""""""""""""
"""GENERAL"""
""""""""""""""


def get_penetration_sdf_score(sdfs, slack=0):
    sdfs_ = sdfs.copy()
    if slack > 0:
        sdfs_[sdfs_>-slack] = 0
    return np.max(np.stack([-sdfs_, np.zeros_like(sdfs)], axis=0), axis=0).sum(axis=-1)
  
def get_nearness_sdf_score(sdfs):
    return np.max(np.stack([sdfs, np.zeros_like(sdfs)], axis=0), axis=0).sum(axis=-1)  

def get_contact_sdf_score(sdfs, contact_threshold=0.005):
    # Penalty for being close to contact (small positive SDF).
    # penalty_contact = 1.0 - num_sdfs_in_contact / total_num_sdfs
    contact_mask = (sdfs >= 0) & (sdfs <= contact_threshold)
    num_in_contact = np.sum(contact_mask, axis=-1)
    # print("!!## num_in_contact", num_in_contact)
    total_num = sdfs.shape[-1]
    penalty_contact = 0.1 - (num_in_contact / total_num)
    # penalty_contact = np.exp(-20.0 * (num_in_contact / total_num))
    return penalty_contact

def sdf_grad_query(points_obj, sdf_grid, bounds, sdf_query_fn, eps=None):
    """
    points_obj: (B, P, 3) points in OBJECT frame
    sdf_grid:   SDF grid (nx, ny, nz) or whatever your mesh_to_sdf returns
    bounds:     (lower, upper), each (3,)
    sdf_query_fn: function(points, sdf_grid, bounds) -> (B, P) sdf values
    eps: finite diff step in meters; if None, choose ~ half a voxel
    returns: (B, P, 3) gradient
    """
    lower, upper = bounds
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    # Choose eps from voxel size if not provided
    if eps is None:
        # assumes sdf_grid is a dense grid with shape (nx, ny, nz)
        grid_shape = np.asarray(sdf_grid.shape, dtype=np.float64)
        voxel = (upper - lower) / (grid_shape - 1.0)
        eps = 0.5 * float(np.min(voxel))  # conservative, isotropic step

    # Central differences
    e = np.eye(3, dtype=np.float64) * eps  # (3,3)

    f_xp = sdf_query_fn(points_obj + e[0], sdf_grid, bounds)
    f_xm = sdf_query_fn(points_obj - e[0], sdf_grid, bounds)
    f_yp = sdf_query_fn(points_obj + e[1], sdf_grid, bounds)
    f_ym = sdf_query_fn(points_obj - e[1], sdf_grid, bounds)
    f_zp = sdf_query_fn(points_obj + e[2], sdf_grid, bounds)
    f_zm = sdf_query_fn(points_obj - e[2], sdf_grid, bounds)

    grad_x = (f_xp - f_xm) / (2.0 * eps)
    grad_y = (f_yp - f_ym) / (2.0 * eps)
    grad_z = (f_zp - f_zm) / (2.0 * eps)

    grad = np.stack([grad_x, grad_y, grad_z], axis=-1)  # (B, P, 3)
    return grad





""""""""""""""
"""GRASP ENV"""
""""""""""""""

def normal_closure_penalty(
    left_points_obj, right_points_obj,
    left_sdf, right_sdf,
    sdf_grid, bounds, sdf_query_fn,
    w_normal=1.0,
    w_align=1.0,
    eps=None,
):
    """
    Normal opposition penalty: small when nL ≈ -nR.
    Closure-direction consistency: normals point “toward the fingers” along the line connecting contacts.

    left_points_obj, right_points_obj: (B, Pl, 3), (B, Pr, 3)
    left_sdf, right_sdf:               (B, Pl), (B, Pr)  SDF values to object
    Returns: (B,) penalty (lower is better)
    """
    B = 1

    # 1) pick "closest-to-contact" points on each finger (min |sdf|)
    idxL = np.argmin(np.abs(left_sdf), axis=1)   # (B,)
    idxR = np.argmin(np.abs(right_sdf), axis=1)  # (B,)

    # gather contact points (B, 3)
    pL = left_points_obj[np.arange(B), idxL]
    pR = right_points_obj[np.arange(B), idxR]

    # 2) estimate normals via SDF gradient at the contact points
    # shape to (B, 1, 3) so sdf_grad_query works
    gradL = sdf_grad_query(pL[:, None, :], sdf_grid, bounds, sdf_query_fn, eps=eps)[:, 0, :]
    gradR = sdf_grad_query(pR[:, None, :], sdf_grid, bounds, sdf_query_fn, eps=eps)[:, 0, :]

    nL = gradL / (np.linalg.norm(gradL, axis=1, keepdims=True) + 1e-12)
    nR = gradR / (np.linalg.norm(gradR, axis=1, keepdims=True) + 1e-12)

    # 3) closure direction: from left contact to right contact
    d = pR - pL
    d_hat = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)

    # (A) antipodal / opposing normals: nL ≈ -nR  -> dot ≈ -1
    # penalty in [0, 1]: 0 when dot=-1, 1 when dot=+1
    dot_lr = np.sum(nL * nR, axis=1)
    oppose_pen = 0.5 * (1.0 + dot_lr)

    # (B) normals oppose closure direction: object normal at left should point toward left finger,
    # which is roughly opposite to d_hat; at right it aligns with d_hat.
    # penalty in [0, 2]: 0 when perfectly aligned, larger when misaligned.
    alignL = np.sum(nL * (-d_hat), axis=1)  # want +1
    alignR = np.sum(nR * ( d_hat), axis=1)  # want +1
    align_pen = (0.5 * (1.0 - alignL)) + (0.5 * (1.0 - alignR))

    # if not ret_aux:
    return w_normal * oppose_pen + w_align * align_pen
    # else:
    #     return w_normal * oppose_pen + w_align * align_pen, oppose_pen, align_pen

def grasp_score_calc(
    sdf_values_to_object, sdf_values_to_plane,
    w_nearness, w_penetration,
    left_points_obj=None, right_points_obj=None,
    left_sdf=None, right_sdf=None,
    sdf_grid=None, bounds=None, t_world_to_tcp=None, sdf_query_fn=None,
    w_normal=0.01, w_align=0.01, w_tcp_height=0.01, grad_eps=None, penetration_slack=0, include_plane_nearness=True,
):
    to_object_penetration_score = get_penetration_sdf_score(sdf_values_to_object, penetration_slack)
    to_object_nearness_score = get_nearness_sdf_score(sdf_values_to_object)
    
    to_plane_penetration_score = get_penetration_sdf_score(sdf_values_to_plane, penetration_slack)
    if not include_plane_nearness:
        to_plane_nearness_score = 0
    else:
        to_plane_nearness_score = get_nearness_sdf_score(sdf_values_to_plane)

    penalty = (
        w_nearness * (to_object_nearness_score + to_plane_nearness_score)
        + w_penetration * (to_object_penetration_score + to_plane_penetration_score)
        + np.exp(w_tcp_height * np.array([t_world_to_tcp[2],]))
    )

    # Add normal-based penalty if inputs provided
    assert left_points_obj is not None and right_points_obj is not None
    assert left_sdf is not None and right_sdf is not None
    assert sdf_grid is not None and bounds is not None and sdf_query_fn is not None

    norm_closure_score = normal_closure_penalty(
        left_points_obj, right_points_obj,
        left_sdf, right_sdf,
        sdf_grid, bounds, sdf_query_fn,
        w_normal=w_normal, w_align=w_align,
        eps=grad_eps,
    )
    penalty = penalty + norm_closure_score

    return penalty[0]



""""""""""""""
"""PUSH ENV"""
""""""""""""""

def get_finger_pose_score_push(left_finger_points_object, t_world_to_obj, push_subgoal, w_finger_pos=1.0, w_finger_orient=1.0, w_subgoal_align=1.0):
    """
    The function calculates a penalty based on the finger pointclouds in object frame, the object position and the push subgoal.
    Encourages the finger principle plane normal (x,y,0) colinear with finger-to-object vector

    left_finger_points_object: (1, N, 3) array of left finger pointcloud in object frame
    t_world_to_obj: (3,) translation of object in world frame
    push_subgoal: (2,) array of push subgoal position.
    """
    # finger pointcloud com in object frame
    finger_com_obj_3d = np.mean(left_finger_points_object[0,:,:], axis=0)  # (3,)

    # push subgoal in object frame
    obj_pos_2d = t_world_to_obj[:2]
    push_subgoal_obj = push_subgoal - obj_pos_2d  # (2,)

    # desired pre-push position in object frame - 6cm away from object opposite to push subgoal direction
    des_prepush_dist_to_obj = 0.06
    des_prepush_pos_obj = - push_subgoal_obj / (np.linalg.norm(push_subgoal_obj) + 1e-12) * des_prepush_dist_to_obj  # (2,)
    des_prepush_pos_obj_3d = np.array([des_prepush_pos_obj[0], des_prepush_pos_obj[1], -t_world_to_obj[2]])  # (3,). Make the finger lower than the object center in z axis

    # finger com to desired pre-push position vector in object frame
    finger_to_des_prepush_vec = des_prepush_pos_obj_3d - finger_com_obj_3d  # (3,)
    finger_to_des_prepush_dist = np.linalg.norm(finger_to_des_prepush_vec) + 1e-12 # i

    # finger principle plane fitted with PCA, with normal (x,y,0)
    finger_points_2d = left_finger_points_object[0,:,:2]  # (N,2)
    finger_points_2d_centered = finger_points_2d - np.mean(finger_points_2d, axis=0, keepdims=True)  # (N,2)
    cov_matrix = np.cov(finger_points_2d_centered.T)  # (2,2)
    evals, evecs = np.linalg.eigh(cov_matrix)  # evals ascending
    finger_normal_2d = evecs[:, 0]  # (2,)

    # compute angle between finger normal and finger com to object norm vector in 2D
    finger_normal_2d = finger_normal_2d / (np.linalg.norm(finger_normal_2d) + 1e-12)
    finger_com_obj_2d = finger_com_obj_3d[:2]
    finger_com_obj_2d = finger_com_obj_2d / (np.linalg.norm(finger_com_obj_2d) + 1e-12)
    finger_orient_score = np.abs(np.dot(finger_normal_2d, finger_com_obj_2d))  # scalar in [0,1], ii

    # compute angle between subgoal to object norm vector, and finger com to object norm vector in 2D
    push_subgoal_obj_2d = push_subgoal_obj / (np.linalg.norm(push_subgoal_obj) + 1e-12)
    subgoal_to_obj_score = np.abs(np.dot(push_subgoal_obj_2d, finger_com_obj_2d))  # scalar in [0,1], iii

    finger_pos_term = w_finger_pos * finger_to_des_prepush_dist
    finger_orient_term = - w_finger_orient * finger_orient_score
    subgoal_align_term = - w_subgoal_align * subgoal_to_obj_score

    return finger_pos_term, finger_orient_term, subgoal_align_term

def push_score_calc(
    sdf_values_to_object, sdf_values_to_plane, 
    t_world_to_obj, push_subgoal,
    left_finger_points_object,
    w_penetration=1.0, w_finger_pos=1.0, w_finger_orient=1.0, w_subgoal_align=1.0,
    penetration_slack=-0.005,
):
    """
    The function calculates a push penalty (not a reward) based on SDF values to an object, a plane,
    as well as the relative height of the tool center point (TCP) to the object.

    push_subgoal: (2,) array of push subgoal position.
    """
    to_object_penetration_score = get_penetration_sdf_score(sdf_values_to_object, penetration_slack)
    to_plane_penetration_score = get_penetration_sdf_score(sdf_values_to_plane, penetration_slack)

    finger_pos_term, finger_orient_term, subgoal_align_term = \
        get_finger_pose_score_push(left_finger_points_object, t_world_to_obj, push_subgoal,
                              w_finger_pos, w_finger_orient, w_subgoal_align)

    pen_term = w_penetration * (to_object_penetration_score + to_plane_penetration_score)
    pen_term = float(pen_term)
    penalty = (pen_term + finger_pos_term + finger_orient_term + subgoal_align_term)
    # print("## penalty pen_term", pen_term)
    # print("## penalty finger_pos_term", finger_pos_term)
    # print("## penalty finger_orient_term", finger_orient_term)
    # print("## penalty subgoal_align_term", subgoal_align_term)
    # print("# total penalty", penalty)

    return penalty



""""""""""""""
"""SCOOP ENV"""
""""""""""""""

def get_finger_pose_score_scoop(left_finger_points_object, w_finger_pos=1.0, w_finger_orient=1.0):
    """
    The function calculates a penalty based on the finger pointclouds in object frame.
    Four criteria:
    i. rz: long pca principle axis of finger pcd pointing y axis of object frame (same rotation as world frame)
    ii. x: pcd center on y axis (center_x=0) of object frame
    iii. z: pcd center_z + half pcd extent_z + offset (0.01) = -filet_half_thickness (0.01) * (max_randomization with slack (1.1))
    iv. y: pcd center_y + half pcd extent_y = - board edge_y in object frame (0.02)

    left_finger_points_object: (1, N, 3) array of left finger pointcloud in object frame
    """
    # finger pointcloud com in object frame
    finger_com_obj_3d = np.mean(left_finger_points_object[0, :, :], axis=0)  # (3,)

    # fit oriented bounding box (OBB) to finger pointcloud in object frame, with one principle axis aligned with object frame z axis, the other two in x-y plane
    finger_points_2d = left_finger_points_object[0, :, :2]  # (N,2)
    finger_points_2d_centered = finger_points_2d - np.mean(finger_points_2d, axis=0, keepdims=True)  # (N,2)
    cov_matrix = np.cov(finger_points_2d_centered.T)  # (2,2)
    evals, evecs = np.linalg.eigh(cov_matrix)  # evals ascending
    finger_rz_2d = evecs[:, 1]  # (2,), the major principle axis
    finger_rz_2d = finger_rz_2d / (np.linalg.norm(finger_rz_2d) + 1e-12)
    rz_score = np.abs(finger_rz_2d[1])  # scalar in [0,1], i, the larger the better (aligned with +/−y)

    # finger com distance to desired x position (y axis of object frame)
    x_score = np.abs(finger_com_obj_3d[0])  # scalar >=0, ii, the smaller the better

    # finger com distance to desired z position in object frame
    finger_points_z = left_finger_points_object[0, :, 2]
    finger_extent_z = finger_points_z.max() - finger_points_z.min()
    offset_z = 0.01
    filet_half_thickness_w_slack = 0.01 * 1.1
    desired_finger_com_z = -filet_half_thickness_w_slack - (finger_extent_z / 2.0) - offset_z
    z_score = np.abs(finger_com_obj_3d[2] - desired_finger_com_z)  # scalar >=0, iii, the smaller the better

    # finger com distance to desired y position in object frame
    finger_points_y = left_finger_points_object[0, :, 1]
    finger_extent_y = finger_points_y.max() - finger_points_y.min()
    board_edge_y_in_obj = -0.025 # TODO: dependent on t_board_to_object y in simulator.py
    desired_finger_com_y = board_edge_y_in_obj - (finger_extent_y / 2.0)
    y_score = np.abs(finger_com_obj_3d[1] - desired_finger_com_y)  # scalar >=0, iv, the smaller the better

    # print("desired prescoop position", np.array([0.0, desired_finger_com_y, desired_finger_com_z]))

    # combine scores
    finger_pos_term = w_finger_pos * (x_score + z_score + y_score)
    finger_orient_term = - w_finger_orient * rz_score

    return finger_pos_term, finger_orient_term


def scoop_score_calc(
    sdf_values_to_object, sdf_values_to_board, sdf_values_to_plane,
    left_finger_points_object,
    w_penetration=1.0, w_finger_pos=1.0, w_finger_orient=1.0,
    penetration_slack=0
):
    """
    The function calculates a scoop penalty (not a reward) based on SDF values to an object, a plane, and a board,
    as well as the relative height of the tool center point (TCP) to the object.
    """
    to_object_penetration_score = get_penetration_sdf_score(sdf_values_to_object, penetration_slack)
    to_board_penetration_score = get_penetration_sdf_score(sdf_values_to_board, penetration_slack)
    to_plane_penetration_score = get_penetration_sdf_score(sdf_values_to_plane, penetration_slack)

    finger_pos_term, finger_orient_term = \
        get_finger_pose_score_scoop(left_finger_points_object, w_finger_pos, w_finger_orient)

    pen_term = w_penetration * (to_object_penetration_score + to_board_penetration_score + to_plane_penetration_score)
    pen_term = float(pen_term)
    penalty = (pen_term + finger_pos_term + finger_orient_term)

    # print("## penalty w_penetration", pen_term)
    # print("## penalty w_finger_pos", finger_pos_term)
    # print("## penalty w_finger_orient", finger_orient_term)
    # print("# total penalty", penalty)

    return penalty