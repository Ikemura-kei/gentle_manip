from yacs.config import CfgNode
import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot
from dataclasses import dataclass
from typing import Optional

def np2torch(input, device):
    return torch.tensor(input).to(device)

def torch2np(input):
    return input.detach().cpu().numpy()

@dataclass
class BaseRobotCfgDoc:
    """
    Base configuration schema for all robots.

    Attributes
    ----------
    ee_bound : np.ndarray
        End-effector workspace bounds.

        Shape ``(2, 3)``, dtype ``float``.
        First row is minimum, second row is maximum.
        Columns correspond to ``x, y, z``.

    ee_randomization_bound : np.ndarray
        Randomization bounds for end-effector position.

        Shape ``(2, 3)``, dtype ``float``.

    friction_noise : float, optional
        Standard deviation of friction noise.
    """

    ee_bound: np.ndarray
    ee_randomization_bound: np.ndarray
    friction_noise: Optional[float] = None

class BaseRobot:
    """Base interface for robot handlers.

    Subclasses must define the attributes listed in ``REQUIRED_ATTRIBUTES``.

    Args:
        n_envs: Number of environments.
        robot_cfg: Configuration node. See :class:`~core.simulator.robots.base_robot.BaseRobotCfgDoc`.

    Attributes:
        n_envs: Number of environments.
        robot_cfg: The config used by this robot.
    
    Note:
        Subclasses may require additional configuration fields.
    """
    REQUIRED_ATTRIBUTES = ['_file_type', '_file_path', '_end_effector_link',
                           '_links_to_keep', '_joint_names', '_default_joint_angles', '_kp', '_kv']

    def __init__(self, n_envs: int, robot_cfg: CfgNode):
        self._check_variables() # Check for if REQUIRED_ATTRIBUTES are all defined.
        self.n_envs = n_envs
        self.robot_cfg = robot_cfg

    def setup(self, gs_entity):
        """Setup the robot handler when the scene is created.

        Args:
            gs_entity: The robot entity created by ``scene.add_entity``.
        """
        
        self.robot_local_dofs_idx = [gs_entity.get_joint(
            name).dof_idx_local for name in self.joint_names]

        assert len(self.default_joint_angles) == len(self.robot_local_dofs_idx)

        if self.kp is not None:
            gs_entity.set_dofs_kp(
                kp=self.kp,
                dofs_idx_local=self.robot_local_dofs_idx,
            )
        if self.kv is not None:
            gs_entity.set_dofs_kv(
                kv=self.kv,
                dofs_idx_local=self.robot_local_dofs_idx,
            )
            
        gs_entity.set_dofs_position(np.tile(self.default_joint_angles[None,...], (self.n_envs,1)), self.robot_local_dofs_idx)
        gs_entity.control_dofs_position(np.tile(self.default_joint_angles[None,...], (self.n_envs,1)), self.robot_local_dofs_idx)
        
        self.end_effector = gs_entity.get_link(self.end_effector_link) 
        
        self.gs_entity = gs_entity
        
        self.target_ee_pose = None # Initialize this to be none, upon first call for movement, this will be set properly.
        
        self.device = self.end_effector.get_pos().device
        
        self.ee_bound = torch.tile(torch.tensor(self.robot_cfg.ee_bound, device=self.device)[:,None,...], (1, self.n_envs, 1))
        self.ee_randomization_bound = torch.tile(torch.tensor(self.robot_cfg.ee_randomization_bound, device=self.device)[:,None,...], (1, self.n_envs, 1))
        if "ee_euler_randomization_bound" in self.robot_cfg:
            self.ee_euler_randomization_bound = torch.tile(torch.tensor(self.robot_cfg.ee_euler_randomization_bound, device=self.device)[:,None,...], (1, self.n_envs, 1))
        
        self.reset()
        
    def reset(self):
        """Resets the robot to initial pose, randomization is included on the joint space.
        """
        preturbed_default_joint_angles = np.tile(self.default_joint_angles[None,...], (self.n_envs,1)) + (np.random.rand(*(np.tile(self.default_joint_angles[None,...], (self.n_envs,1)).shape)) - 0.5) * 2 * 0.02
        self.gs_entity.set_dofs_position(preturbed_default_joint_angles, self.robot_local_dofs_idx)
        self.gs_entity.control_dofs_position(preturbed_default_joint_angles, self.robot_local_dofs_idx)
        
        self.target_ee_pose = None

    def reset_to_ee_pose(self, ee_pose: np.ndarray):
        """Resets the robot to a specific end-effector pose.

        Args:
            ee_pose (np.ndarray): the target end-effector pose, of shape (n_envs, 7), each in the format [x, y, z, qw, qx, qy, qz].
        """
        self._ee_set(ee_pose, n_arm_link=len(self.joint_names))
        self.target_ee_pose = None
        
    def control(self, mode: str, input: np.ndarray):
        """Abstract method that controls a robot, implementation shall be provided at the child classes.

        Args:
            mode (str): a string that specify the mode of control, this depends on per-child-class implementation.
            input (np.ndarray): the control input, of shape (n_envs, K), where K is the control dimension.

        Raises:
            NotImplementedError: must be implemented at the child class.
        """
        raise NotImplementedError
    
    def forward_kinematics(self, qpos):
        """Forward kinematics of the robot given joint positions.

        Args:
            qpos (np.ndarray): the joint positions of shape (n_envs, J), where J is the number of joints on the robot.

        Returns:
            np.ndarray: returns the pose as an array of shape (n_envs, 7), each in the format [x, y, z, qw, qx, qy, qz].
        """
        pos, quat = self.gs_entity.forward_kinematics(qpos, links_idx_local=self.end_effector.idx_local)
        pos = pos.detach().cpu().numpy()[:, 0, :]
        quat = quat.detach().cpu().numpy()[:, 0, :]
        
        return np.concatenate([pos, quat], axis=-1)
    
    def plan_trajectory(self, goal_pos, goal_q, n_waypoints=300):
        """Plans a trajectory to execute given goal pose, obstacle avoidance is present as provided by Genesis.

        Args:
            goal_pos (np.ndarray): the goal position of the end-effector, in shape (n_envs, 3)
            goal_q (np.ndarray): the goal orientation of the end-effector, in shape (n_envs, 4), quaternion format [qw, qx, qy, qz]
            n_waypoints (int, optional): the number of waypoints in the generated trajectory. Defaults to 300.

        Returns:
            np.ndarray: the planned trajectory of shape (n_waypoints, n_envs, J), where J is the number of joints on the robot.
        """
        qpos = self.gs_entity.inverse_kinematics(
            link=self.end_effector,
            pos=goal_pos,
            quat=goal_q,
        )
        
        path = self.gs_entity.plan_path(
            qpos_goal     = qpos,
            num_waypoints = n_waypoints,
        )
        
        return path.detach().cpu().numpy()

    # Below are internal helper functions.
        
    def _check_variables(self):
        # print(self._file_path)
        for attr in self.REQUIRED_ATTRIBUTES:
            assert hasattr(self, attr)

        assert isinstance(self._file_type, str)
        assert self._file_type in ['URDF', 'MJCF']

        assert isinstance(self._file_path, str)
        
    def _ee_ctrl(self, target_ee_pose: np.ndarray, n_arm_link: int = 7, noise=False):
        assert len(target_ee_pose) == self.n_envs
        assert target_ee_pose.shape[1] == 7
        
        q = self.gs_entity.inverse_kinematics(
            link = self.end_effector,
            pos  = target_ee_pose[:, :3],
            quat = target_ee_pose[:, 3:],
        )
        if noise:
            q += (torch.rand_like(q) - 0.5) * 2 * 0.01
        
        self.gs_entity.control_dofs_position(q[:, :n_arm_link], self.robot_local_dofs_idx[:n_arm_link])
        
    def _dls_ik(self, action: np.ndarray, n_arm_link=7):
        """
        Damped least squares inverse kinematics
        """
        action = torch.tensor(action, device='cuda', dtype=torch.float32)
        delta_pose = action[:, :6]
        lambda_val = 0.01
        jacobian = self.gs_entity.get_jacobian(link=self.end_effector)
        
        jacobian_T = jacobian.transpose(1, 2)
        lambda_matrix = (lambda_val**2) * torch.eye(n=jacobian.shape[1], device='cuda')
        delta_joint_pos = (
            jacobian_T @ torch.inverse(jacobian @ jacobian_T + lambda_matrix) @ delta_pose.unsqueeze(-1)
        ).squeeze(-1)
        
        qpos = self.gs_entity.get_qpos() + delta_joint_pos
    
        self.gs_entity.control_dofs_position(qpos[:, :n_arm_link], self.robot_local_dofs_idx[:n_arm_link])
        
    def _ee_set(self, target_ee_pose: np.ndarray, n_arm_link: int = 7, noise=False):
        assert len(target_ee_pose) == self.n_envs
        assert target_ee_pose.shape[1] == 7
        
        q = self.gs_entity.inverse_kinematics(
            link = self.end_effector,
            pos  = target_ee_pose[:, :3],
            quat = target_ee_pose[:, 3:],
        )
        if noise:
            q += (torch.rand_like(q) - 0.5) * 2 * 0.01
            
        self.gs_entity.set_dofs_position(q[:, :n_arm_link], self.robot_local_dofs_idx[:n_arm_link])
        
    def _delta_ee_ctrl(self, delta_ee_pose: np.ndarray, n_arm_link: int = 7):
        
        assert delta_ee_pose.shape[1] == 6 # dx, dy, dz, aax, aay, aaz
        
        if self.target_ee_pose is None:
            self.target_ee_pose = torch.zeros((self.n_envs, 7)).to(self.device)
            self.target_ee_pose[:, :3] = self.end_effector.get_pos()
            self.target_ee_pose[:, 3:] = self.end_effector.get_quat()
        
        cur_rot_mat = Rot.from_quat(torch2np(self.target_ee_pose[:, 3:]), scalar_first=True).as_matrix()
        delta_rot_mat = Rot.from_rotvec(delta_ee_pose[:, 3:], degrees=False).as_matrix()
        target_rot_mat = cur_rot_mat @ delta_rot_mat
        
        target_quat = Rot.from_matrix(target_rot_mat).as_quat(scalar_first=True)
        
        self.target_ee_pose[:, :3] += np2torch(delta_ee_pose[:, :3], self.device)
        self.target_ee_pose[:, :3] = torch.clip(self.target_ee_pose[:, :3], self.ee_bound[0], self.ee_bound[1])
        
        self.target_ee_pose[:, 3:] = np2torch(target_quat, self.device)
        
        self._ee_ctrl(self.target_ee_pose, n_arm_link=n_arm_link)
    
    @property
    def ee_pose(self):
        """End-effector pose.

        Returns:
            torch.tensor: tensor of shape (n_envs, 7): [x, y, z, qw, qx, qy, qz].
        """
        pos = self.end_effector.get_pos()
        quat = self.end_effector.get_quat()
        return torch.cat([pos, quat], dim=-1)
    
    @property
    def joint_pos(self):
        """Joint positions.

        Returns:
            torch.tensor: joint positions of shape (n_envs, J).
        """
        return self.gs_entity.get_dofs_position()
    
    @property
    def ee_vel(self):
        return self.end_effector.get_vel()
    
    @property
    def ee_twist(self):
        lin_vel = self.end_effector.get_vel()
        ang_vel = self.end_effector.get_ang()
        return torch.cat([lin_vel, ang_vel], dim=-1)

    @property
    def file_type(self):
        return self._file_type

    @property
    def file_path(self):
        return self._file_path

    @property
    def links_to_keep(self):
        return self._links_to_keep

    @property
    def joint_names(self):
        return self._joint_names

    @property
    def default_joint_angles(self):
        return self._default_joint_angles

    @property
    def kp(self):
        return self._kp

    @property
    def kv(self):
        return self._kv

    @property
    def end_effector_link(self):
        return self._end_effector_link