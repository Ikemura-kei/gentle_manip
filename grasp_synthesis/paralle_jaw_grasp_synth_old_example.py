import numpy as np
from core.wrappers.flatten_obs import FlattenObservation
from tqdm import tqdm
import trimesh
import argparse
import cma
import torch
import time
from datetime import datetime
import csv
import os
from skopt import gp_minimize
from skopt.space import Real

from scripts.runners.utils import prep
from core.utils.math_utils import homo_from_t_R, homo_transform
from scipy.spatial.transform import Rotation as Rot
from core.grasp_synthesis.basic import *
from core.grasp_synthesis.grasp_evaluation import *
from core.utils.sdf_utils import mesh_to_sdf, sdf_query, ground_sdf
from scripts.demos.bilevel_utility import *

"""
Example usage:
python scripts/demos/jaw_grasp_synthesis.py --cfg_file cfgs/debugger/jaw_grasp_synthesis.yaml --run_suffix jaw_grasp_synthesis
"""
# ----------------------------
# Hyperparams
# ----------------------------
# inner-loop (grasp synthesis) objective weights
w_nearness = 0.02
w_penetration = 100.0
w_normal = 30.0
w_align = 30.0
w_tcp_height = 10.0

seed = 2567

# outer-loop (design) objective weight
w_stress = 1e-5

object_size = np.array([0.050, 0.050, 0.040])  # meters. TODO
maxfevals_synthesis = 800
mean_stress_thres = 1600  # threshold above which finger will no more close

# top-k grasp candidate selection
TOPK_GRASP = 0
do_quick_eval = False if TOPK_GRASP == 0 else True  
TOPK_MIN_DIST = 0.01  # "meter-equivalent" min distance for grasp synthesis top-k diversity TODO: too large?
task_success_lift_z_thres = 0.10  # meters

# quick evaluation config (skip full approach; use short settle)
SETTLE_STEPS = 5
APPROACH_ITERS = 45
GRASP_ITERS = 15
LIFT_ITERS = 25

def grasp_candidate_distance(x1, x2, rot_w=0.02, width_w=1.0):
    """
    Distance in "meter-equivalent" units:
      d = ||t1 - t2|| + rot_w * angle(q1,q2) + width_w * |w1-w2|
    Where angle is in radians.
    x = [tx,ty,tz,qx,qy,qz,qw,w] with quaternion xyzw.
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    # translation
    t1, t2 = x1[:3], x2[:3]
    tdist = float(np.linalg.norm(t1 - t2))

    def _rot_from_x(x):
        roll, pitch, yaw = float(x[3]), float(x[4]), float(x[5])
        q_xyzw = Rot.from_euler('xyz', [roll, pitch, yaw], degrees=False).as_quat()
        return Rot.from_quat(q_xyzw)

    # rotation angle
    try:
        r1 = _rot_from_x(x1)
        r2 = _rot_from_x(x2)
        angle = (r1.inv() * r2).magnitude()
    except Exception:
        angle = np.pi

    # gripper width
    w1, w2 = x1[6], x2[6]
    width_dist = float(abs(w1 - w2))

    return tdist + rot_w * float(angle) + width_w * float(width_dist)


# ----------------------------
# Grasp synthesis objective (minimize)
# ----------------------------
def grasp_synthesis_objective(x, object_sdf, bounds, left_finger_points, right_finger_points, T_obj_to_world,
                            w_nearness, w_penetration, w_normal, w_align):
    t_world_to_tcp = x[:3][None, ...]
    roll, pitch, yaw = x[3], x[4], x[5]
    q_xyzw = Rot.from_euler('xyz', [roll, pitch, yaw], degrees=False).as_quat()
    q_world_to_tcp = q_xyzw[None, ...]
    R_world_to_tcp = Rot.from_quat(q_world_to_tcp).as_matrix()

    gripper_width = x[6]

    # Compute transforms
    T_world_to_tcp = homo_from_t_R(t_world_to_tcp, R_world_to_tcp)
    T_tcp_to_world = np.linalg.inv(T_world_to_tcp)

    # TCP->fingers
    t_left_finger_to_tcp = np.array([
        0.0,
        -gripper_width / 2.0 - finger_gripper_offset,
        finger_to_tcp_z_axis_offset + finger_movement_slope * (0.044 - gripper_width / 2.0)
    ])[None, ...]
    q_left_finger_to_tcp = np.array([0.0, 0.0, 0.0, 1.0])[None, ...]
    R_left_finger_to_tcp = Rot.from_quat(q_left_finger_to_tcp).as_matrix()
    T_left_finger_to_tcp = homo_from_t_R(t_left_finger_to_tcp, R_left_finger_to_tcp)

    t_right_finger_to_left_finger = np.array([0.0, -gripper_width - finger_gripper_offset * 2.0, 0.0])[None, ...]
    q_right_finger_to_left_finger = np.array([0.0, 0.0, 1.0, 0.0])[None, ...]
    R_right_finger_to_left_finger = Rot.from_quat(q_right_finger_to_left_finger).as_matrix()
    T_right_finger_to_left_finger = homo_from_t_R(t_right_finger_to_left_finger, R_right_finger_to_left_finger)

    # Finger points -> object frame
    T_left_finger_to_world = T_left_finger_to_tcp @ T_tcp_to_world
    T_right_finger_to_world = T_right_finger_to_left_finger @ T_left_finger_to_world
    T_left_finger_to_object = T_left_finger_to_world @ np.linalg.inv(T_obj_to_world)
    left_finger_points_object = homo_transform(left_finger_points, np.linalg.inv(T_left_finger_to_object))

    T_right_finger_to_object = T_right_finger_to_world @ np.linalg.inv(T_obj_to_world)
    right_finger_points_object = homo_transform(right_finger_points, np.linalg.inv(T_right_finger_to_object))

    # Evaluate SDF to object
    left_sdf = sdf_query(left_finger_points_object, object_sdf, bounds)
    right_sdf = sdf_query(right_finger_points_object, object_sdf, bounds)
    to_object_sdf_value = np.concatenate([left_sdf, right_sdf], axis=1)

    # Evaluate SDF to ground plane (world frame)
    left_finger_points_world = homo_transform(left_finger_points, np.linalg.inv(T_left_finger_to_world))
    right_finger_points_world = homo_transform(right_finger_points, np.linalg.inv(T_right_finger_to_world))
    to_plane_sdf_value = ground_sdf(np.concatenate([left_finger_points_world, right_finger_points_world], axis=1))

    # Evaluate the grasp pose
    t_world_to_tcp = t_world_to_tcp.squeeze()
    score = grasp_score_calc(
        to_object_sdf_value, to_plane_sdf_value,
        w_nearness, w_penetration,
        left_points_obj=left_finger_points_object,
        right_points_obj=right_finger_points_object,
        left_sdf=left_sdf,
        right_sdf=right_sdf,
        sdf_grid=object_sdf,
        bounds=bounds,
        t_world_to_tcp=t_world_to_tcp,
        sdf_query_fn=sdf_query,
        w_normal=w_normal,
        w_align=w_align,
        w_tcp_height=w_tcp_height,
        grad_eps=None,
    )
    return score

# ----------------------------
# Quick evaluation for all envs at the same time
# ----------------------------
def evaluate_grasp_pose_quick(
    cand_grasps,
    t_world_to_obj,
    empty_val=1e9,
    very_high_score=1e9,
):
    """
    Evaluate candidate grasps for ALL envs together.

    Args:
        cand_grasps: (E, G, 7) array. Each grasp x = [t(3), q_xyzw(4), w(1)].
                    "Empty" grasp is np.array([empty_val]*7).
        t_world_to_obj: (E, 3) array, initial object positions per env.
        Returns:
        scores: (E, G) array, score per env per grasp index.
                Score definition matches your outer metric:
                score_sum = lift_dist + (0.1 - max_mean_stress * w_stress)
    """
    cand_grasps = np.asarray(cand_grasps, dtype=float)
    assert cand_grasps.ndim == 3 and cand_grasps.shape[2] == 7, \
        f"cand_grasps must be (E, G, 7), got {cand_grasps.shape}"

    E, G, _ = cand_grasps.shape
    scores = np.ones((E, G), dtype=float) * very_high_score

    def _is_empty(x7):
        # strict equality is fine because you construct exactly [1e9]*7
        return np.all(x7 == empty_val)

    # init_obj_z = t_world_to_obj[:, 2].astype(float).copy() # (E,)
    success_across_g = []
    for g in range(G):
        # mask envs with valid candidate
        valid_mask = np.array([not _is_empty(cand_grasps[e, g]) for e in range(E)], dtype=bool)

        # if everyone is empty -> early stop
        if not np.any(valid_mask):
            print(f"Quick-eval: all envs empty at g={g}, early stop.")
            # remaining are already filled with very_high_score
            return scores

        # reset robot+object after each g
        actions = np.zeros((E, 7), dtype=float)
        env.step(actions)  # refresh feedback

        # --- hard set EE pose for valid envs; invalid envs keep current pose ---
        ee_pose = orig_env.obs_dict['vec_states']['ee_pose']  # (E,7) with quat in wxyz in your obs
        t_targets = ee_pose[:, :3].copy()
        q_targets_wxyz = ee_pose[:, 3:].copy()

        for e in range(E):
            if not valid_mask[e]:
                continue
            x = cand_grasps[e, g]
            t_targets[e] = x[:3]

            roll, pitch, yaw = float(x[3]), float(x[4]), float(x[5])
            q_xyzw = Rot.from_euler('xyz', [roll, pitch, yaw], degrees=False).as_quat()
            q_targets_wxyz[e] = np.roll(q_xyzw, 1)  # xyzw -> wxyz

        # Hard-set robot and object poses for all envs in one call
        env.reset_to_ee_pose(t_targets, q_targets_wxyz)
        env.reset_object_pos(t_world_to_obj)

        # one dummy step to sync observations
        env.step(np.zeros((E, 7), dtype=float))

        # print("Quick-eval GRASP phase step")
        mean_stress_hist = []
        top_stress_hist = []
        top_5_max_stress_median_hist = []

        print("Quick-eval GRASP/LIFT phase step")
        obj_z_hist = []
        stop_closing = np.zeros(runner_cfg.env_cfg.n_envs, dtype=bool)
        for _ in range(GRASP_ITERS + LIFT_ITERS):
            curr_mean_stress = orig_env.obs_dict['vec_states']['mean_stress'].astype(float)
            curr_top_stress = orig_env.obs_dict['vec_states']['top_stress'].astype(float)
            curr_top_5_max_stress_median = orig_env.obs_dict['vec_states']['top_5_max_stress_median'].astype(float)
            curr_obj_z = orig_env.obs_dict['vec_states']['soft_body_center'][:, 2].astype(float)
            mean_stress_hist.append(curr_mean_stress)
            top_stress_hist.append(curr_top_stress)
            top_5_max_stress_median_hist.append(curr_top_5_max_stress_median)
            obj_z_hist.append(curr_obj_z)

            curr_finger_pos = orig_env.obs_dict['vec_states']['gripper_width'].astype(float) # not the separation of two fingers: 0.0436 closed, 0.0 fully open

            actions = np.zeros((E, 7), dtype=float)
            stop_closing |= (curr_mean_stress.astype(float) > mean_stress_thres)
            actions[:, -1] = np.where(
                valid_mask & (~stop_closing),
                1.0,
                0.0
            )
            actions[:, 2] = np.where(
                valid_mask & ((stop_closing) | (curr_finger_pos > 0.043).reshape(-1)),
                1.0,
                0.0
            )
            env.step(actions)

        # -------------------
        # Score per env for this candidate g
        # -------------------
        score_sum, lift_dist, success_binary, top_5_mean_stress_median, top_5_top_stress_median, top_5_max_stress_median = \
            compute_grasp_design_score(mean_stress_hist, top_stress_hist, top_5_max_stress_median_hist, obj_z_hist, task_success_lift_z_thres=task_success_lift_z_thres)
        success_across_g.append(success_binary)

        # valid envs get computed score; invalid envs remain very_high_score
        for e in range(E):
            if valid_mask[e]:
                scores[e, g] = float(score_sum[e])
            else:
                scores[e, g] = very_high_score

        print(f"!!!Quick-eval: finished g={g}, valid_envs={int(valid_mask.sum())}/{E}")

    # do not do full trajectory rollout if success rate is too low
    success_across_g = np.stack(success_across_g, axis=1)  # (E, G)
    abort = True if np.mean(success_across_g) < 0.05 else False
    print("success_across_g", success_across_g.shape, np.mean(success_across_g))

    return scores, abort

def write_to_csv(row):
    if current_episode_no == 1:
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerow(row)
    else:
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

# ----------------------------
# Full grasp synthesis + execution
# ----------------------------
def grasp_synthesis_and_execution():
    global current_episode_no
    global best_episode_no_so_far
    global best_score_so_far

    t0 = time.time()

    _obs, _ = env.reset(hard_reset=True)

    # Load assets
    t1 = time.time()
    left_finger_points = trimesh.load('./assets/xarm/mjcf/left_finger.STL').vertices[None, ...]
    right_finger_points = trimesh.load('./assets/xarm/mjcf/right_finger.STL').vertices[None, ...]
    object_sdf, bounds = mesh_to_sdf(trimesh.load('./assets/soft_body/simulation_object.obj'))
    print("Time spent in asset loading and SDF computation:", time.time() - t1)

    best_t = np.zeros((runner_cfg.env_cfg.n_envs, 3), dtype=float)
    best_q_xyzw = np.zeros((runner_cfg.env_cfg.n_envs, 4), dtype=float)

    # Run a few steps to settle the object and the scene
    actions = np.zeros((runner_cfg.env_cfg.n_envs, 7))
    for s in range(SETTLE_STEPS):
        _next_obs, _rew, _done, _extra = env.step(actions)

    # Record soft body state in the rest state after the dummy steps
    env.get_object_state()

    t_world_to_obj = orig_env.obs_dict['vec_states']['soft_body_center']
    q_world_to_obj = np.tile(np.array([[0.0, 0.0, 0.0, 1.0]]), (runner_cfg.env_cfg.n_envs, 1))
    R_world_to_obj = Rot.from_quat(q_world_to_obj).as_matrix()
    T_world_to_obj = homo_from_t_R(t_world_to_obj, R_world_to_obj)
    T_obj_to_world = np.linalg.inv(T_world_to_obj)

    # Record ee initial pose
    ee_pos_init = orig_env.obs_dict['vec_states']['ee_pose'][:, :3]
    ee_orient_init = orig_env.obs_dict['vec_states']['ee_pose'][:, 3:]

    # Grasp synthesis: top-k best is selected
    cand_grasps = np.zeros((runner_cfg.env_cfg.n_envs, TOPK_GRASP+1, 7), dtype=float)
    for b in range(runner_cfg.env_cfg.n_envs):
        translation_lb = (t_world_to_obj[b, :] - 1.5 * object_size).flatten().tolist()
        translation_ub = (t_world_to_obj[b, :] + 1.5 * object_size).flatten().tolist()

        # roll, pitch, yaw: roll rotates around world x, set to pi, ee pointing downward, default pose; pitch around world y; yaw around world z
        lb = translation_lb + [0.8*np.pi, -0.2*np.pi, -0.2*np.pi, 0.028] # TODO
        ub = translation_ub + [1*np.pi, 0.2*np.pi, 0.2*np.pi, 0.088] # roll = ±pi, pitch = ±pi/2, yaw = ±pi
        x0 = [(l+u)/2 for l, u in zip(lb, ub)]
    
        ranges = np.asarray(ub, dtype=float) - np.asarray(lb, dtype=float)
        stds = 0.25 * ranges  # 20% of each dimension range (tune 0.05~0.3)
        stds = np.maximum(stds, 1e-6)  # avoid zeros / tiny ranges
        sigma0 = 1.0  # global; actual per-dim = sigma0 * CMA_stds
        opts = {
            "bounds": [lb, ub],
            "maxfevals": maxfevals_synthesis,
            "CMA_stds": stds.tolist(),
        }

        topk, res = cmaes_topk_diverse(
            objective_fn=grasp_synthesis_objective,
            x0=x0, sigma0=sigma0, opts=opts,
            args=(object_sdf, bounds, left_finger_points, right_finger_points, T_obj_to_world[b],
                  w_nearness, w_penetration, w_normal, w_align),
            k=TOPK_GRASP, min_dist=TOPK_MIN_DIST, 
            dist_fn=grasp_candidate_distance, 
        )
        print("!!! top-k grasp synthesis results", len(topk), topk)

        # cand_grasps[b, 0, :] = np.array(res.xbest, dtype=float) if res.fbest < TOPK_PENALTY_CUTOFF else np.array([1e9,] * 8, dtype=float)
        for k_idx in range(TOPK_GRASP):
            if k_idx < len(topk):
                cand_grasps[b, k_idx, :] = topk[k_idx][1]
            else:
                cand_grasps[b, k_idx, :] = np.array([1e9,] * 7, dtype=float)
        cand_grasps[b, -1, :] = np.array(res.xbest, dtype=float)

    # Quick evaluation to select best candidate per env
    E = cand_grasps.shape[0]
    if do_quick_eval:
        cand_scores, abort = evaluate_grasp_pose_quick(cand_grasps, t_world_to_obj) # (E, G)
        best_idx = np.argmin(cand_scores, axis=1)          # (E,)
        best_x = cand_grasps[np.arange(E), best_idx, :]    # (E, 6)

        if abort:
            print("!!! Aborting full evaluation due to low quick-eval success rate.")
            current_episode_no += 1
            current_episode_no_str = str(current_episode_no)
            _id = current_episode_no_str + '_' + 'aborted_in_quick_eval'
            row = (
                [_id, 100, best_episode_no_so_far, best_score_so_far] 
                + list([100] * (-1 + 6*(1+runner_cfg.env_cfg.n_envs)))
                # + np.round(new_design, 5).tolist()
            )
            write_to_csv(row)
            return 100
    else:
        # Without quick eval, just take the last candidate (res.xbest)
        best_x = cand_grasps[:, -1, :]    # (E, 6)
    best_t = best_x[:, :3] # (E, 3)

    # Convert best orientations    
    for e in range(E):
        x_e = best_x[e]
        roll, pitch, yaw = float(x_e[3]), float(x_e[4]), float(x_e[5])
        best_q_xyzw[e] = Rot.from_euler('xyz', [roll, pitch, yaw], degrees=False).as_quat()
    
    # Final execution
    print("!!!Final full evaluation run: Resetting environment...")
    env.reset_to_ee_pose(ee_pos_init, ee_orient_init)
    env.reset_object_pos(t_world_to_obj)
    actions = np.zeros((runner_cfg.env_cfg.n_envs, 7))
    env.step(actions)

    path = env.plan_trajectory(best_t, np.roll(best_q_xyzw, 1, axis=-1), APPROACH_ITERS)
    record_save_path = f"{log_dir}/{timestamp}/episode_{args.run_suffix}_{current_episode_no + 1}.mp4"
    
    env.start_recording(record_save_path)

    mean_stress_hist = []
    top_stress_hist = []
    top_5_max_stress_median_hist = []
    obj_z_hist = []
    stop_closing = np.zeros(runner_cfg.env_cfg.n_envs, dtype=bool)
    for it in range(orig_env._max_episode_iter):
        if it < len(path):
            print(f"Final execution approach phase, step {it+1}/{len(path)}") if it % 20 == 0 else None
            qpos = path[it]
            qpos[:, -2:] = 0
            target_pose = env.forward_kinematics(qpos)
            delta_pos = (target_pose[:, :3] - orig_env.obs_dict['vec_states']['ee_pose'][:, :3])
            
            # Rotation logic
            prev_rot = Rot.from_quat(np.roll(orig_env.obs_dict['vec_states']['ee_pose'][:, 3:], -1, axis=-1))
            next_rot = Rot.from_quat(np.roll(target_pose[:, 3:], -1, axis=-1))
            delta_orient = (prev_rot.inv() * next_rot).as_rotvec() * 180.0 / np.pi
            
            actions = np.zeros((runner_cfg.env_cfg.n_envs, 7))
            actions[:, :6] = np.concatenate([delta_pos, delta_orient], axis=-1) / np.tile(np.array(runner_cfg.env_cfg.action_scales)[:6], (runner_cfg.env_cfg.n_envs, 1))
            actions[:, -1] = -0.2 - np.random.rand() * 0.1
            env.step(np.clip(actions, -1, 1))

        elif it < len(path) + GRASP_ITERS + LIFT_ITERS:
            print(f"Final execution GRASP/LIFT phase, step {it - len(path) + 1}/{GRASP_ITERS + LIFT_ITERS}") if (it - len(path)) % 20 == 0 else None
            curr_mean_stress = orig_env.obs_dict['vec_states']['mean_stress']
            mean_stress_hist.append(curr_mean_stress)
            top_stress_hist.append(orig_env.obs_dict['vec_states']['top_stress'])
            top_5_max_stress_median_hist.append(orig_env.obs_dict['vec_states']['top_5_max_stress_median'])
            obj_z_hist.append(orig_env.obs_dict['vec_states']['soft_body_center'][:, 2])

            curr_finger_pos = orig_env.obs_dict['vec_states']['gripper_width'].astype(float) # not the separation of two fingers: 0.0436 closed, 0.0 fully open
            actions = np.zeros((runner_cfg.env_cfg.n_envs, 7))
            stop_closing |= (curr_mean_stress.astype(float) > mean_stress_thres)
            print("stop_closing", stop_closing)
            print("curr_mean_stress", curr_mean_stress)
            time.sleep(1)
            actions[:, -1] = np.where(
                (~stop_closing),
                0.5,
                0.0
            )
            actions[:, 2] = np.where(
                (stop_closing),
                1.0,
                0.0
            )
            env.step(actions)
        
        else:
            break

    env.end_recording()
    score_sum, lift_dist, success_binary, top_5_mean_stress_median, top_5_top_stress_median, top_5_max_stress_median = \
        compute_grasp_design_score(mean_stress_hist, top_stress_hist, top_5_max_stress_median_hist, obj_z_hist, task_success_lift_z_thres=task_success_lift_z_thres)

    # Logging
    current_episode_no += 1
    _id = current_episode_no

    row = (
            [_id]
            + [np.round(np.mean(score_sum), 5)]
            + [np.round(np.mean(lift_dist), 5)]
            + [np.round(np.mean(success_binary), 5)]
            + [np.round(np.mean(top_5_mean_stress_median), 5)]
            + [np.round(np.mean(top_5_top_stress_median), 5)]
            + [np.round(np.mean(top_5_max_stress_median), 5)]
            + np.round(score_sum, 5).tolist()
            + np.round(lift_dist, 5).tolist()
            + np.round(success_binary, 5).tolist()
            + np.round(top_5_mean_stress_median, 5).tolist()
            + np.round(top_5_top_stress_median, 5).tolist()
            + np.round(top_5_max_stress_median, 5).tolist()
            # + np.round(new_design, 5).tolist()
        )

    write_to_csv(row)
    print("Time spent in this design iteration:", current_episode_no, time.time() - t0)

    return float(np.mean(score_sum))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, default="cfgs/debugger/jaw_grasp_synthesis.yaml")
    parser.add_argument('--run_suffix', type=str, default='jaw_grasp_synthesis')
    parser.add_argument('--seed', type=int, default=seed)
    args = parser.parse_args()

    # global
    finger_movement_slope = -0.23529411763
    finger_to_tcp_z_axis_offset = 0.10645 # -0.005
    finger_gripper_offset = 0.021 # 0.01545

    np.random.seed(args.seed)

    orig_env, runner_cfg, log_dir = prep(args, algorithm='design_optimization', log_parent_dir='./logs')
    env = FlattenObservation(orig_env)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_dir = f"{log_dir}/{timestamp}"
    os.makedirs(csv_dir, exist_ok=True)
    csv_file_path = f"{csv_dir}/grasp_synthesis_results.csv"

    headers = (
        ["no_episode", ]
        + ["score", "lift_distance", "success_binary", "top_5_mean_stress_median", "top_5_top_stress_median", "top_5_max_stress_median"]
        + [f"score_env_{i}" for i in range(runner_cfg.env_cfg.n_envs)]
        + [f"lift_env_{i}" for i in range(runner_cfg.env_cfg.n_envs)]
        + [f"success_env_{i}" for i in range(runner_cfg.env_cfg.n_envs)]
        + [f"top_5_mean_stress_median_env_{i}" for i in range(runner_cfg.env_cfg.n_envs)]
        + [f"top_5_top_stress_median_env_{i}" for i in range(runner_cfg.env_cfg.n_envs)]
        + [f"top_5_max_stress_median_env_{i}" for i in range(runner_cfg.env_cfg.n_envs)]
    )
    current_episode_no = 0
    best_episode_no_so_far = 0
    best_score_so_far = 1000

    # Test the best design found
    print("Testing the best design found...")
    grasp_synthesis_and_execution()
    for i in range(4):
        print(f"Retesting the best design found, run {i+1}/4...")
        grasp_synthesis_and_execution()
    print("Done.")
    exit()