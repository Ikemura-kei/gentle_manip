import cv2
import torch
import numpy as np
from scipy.spatial.transform import Rotation

import imageio


class MetaWorldTask(object):
    def __init__(self, task_name, vary_ctrl=None, seed=1, max_env_horizon=100, save_video=False):
        from tests.metaworld.envs.mujoco.sawyer_xyz.test_scripted_policies import ALL_ENVS
        self.env = ALL_ENVS[task_name]()
        self.env._partially_observable = False
        self.env._freeze_rand_vec = False
        self.env._set_task_called = True
        self.env.seed(seed)
        self.max_env_horizon = max_env_horizon
        self.save_video=save_video

    def run_eval(self, eval_num, model, n_obs_steps=5):
        device = next(model.parameters()).device
        
        grasp_num = 0
        success_num = 0
        eval_text = []
        eval_frames = []
        for rollout in range(eval_num):
            self.env.reset()

            third_obs = self.preprocess_img(self.env.sim.render(84, 84, mode='offscreen', camera_name='corner')[:,:,::-1])
            eef_pos = self.env.get_endeff_pos()
            geom_xmat = self.env.data.get_geom_xmat('rightpad_geom').reshape(3, 3)
            eef_quat = Rotation.from_matrix(geom_xmat).as_quat()
            rightFinger, leftFinger = self.env._get_site_pos("rightEndEffector"), self.env._get_site_pos("leftEndEffector")
            gripper_qpos = np.array([np.sum((rightFinger - self.env.tcp_center) ** 2, axis=0), -1*np.sum((leftFinger - self.env.tcp_center) ** 2, axis=0)])
            proprio_state = np.concatenate([eef_pos, eef_quat, gripper_qpos])[np.newaxis, :]

            img_store = [third_obs] * (n_obs_steps-1)
            proprio_store = [proprio_state] * (n_obs_steps-1)
            
            step_i = 0
            grasp_succ = 0
            episode_frames = []
            while step_i < self.max_env_horizon:
                third_obs = self.preprocess_img(self.env.sim.render(84,84, mode='offscreen', camera_name='corner')[:,:,::-1])     
                img_store.append(third_obs)

                if self.save_video:
                    rendered = self.env.sim.render(960, 960, mode='offscreen', camera_name='corner')[:,:,::-1]
                    rendered = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
                    episode_frames.append(rendered)

                eef_pos = self.env.get_endeff_pos()
                geom_xmat = self.env.data.get_geom_xmat('rightpad_geom').reshape(3, 3)
                eef_quat = Rotation.from_matrix(geom_xmat).as_quat()
                rightFinger, leftFinger = self.env._get_site_pos("rightEndEffector"), self.env._get_site_pos("leftEndEffector")
                gripper_qpos = np.array([np.sum((rightFinger - self.env.tcp_center) ** 2, axis=0), -1*np.sum((leftFinger - self.env.tcp_center) ** 2, axis=0)])
                proprio_state = np.concatenate([eef_pos, eef_quat, gripper_qpos])[np.newaxis, :]
                proprio_store.append(proprio_state)
                
                imgs = torch.from_numpy(np.concatenate(img_store[-1*n_obs_steps:])).float().to(device).unsqueeze(0)
                pros = torch.from_numpy(np.concatenate(proprio_store[-1*n_obs_steps:])).float().to(device).unsqueeze(0)

                actions = model.get_action(imgs, pros)
                action = actions.cpu().numpy()[0][0]
                action = np.clip(action, -1, 1)
                _, _, done, info = self.env.step(action)

                step_i += 1
                success = info['success']
                
                if info['grasp_success']:
                    grasp_succ = 1
                if success:
                    success_num += success
                    break

            eval_frames = eval_frames + episode_frames
        
        eval_text.append(f"total succ rate: {success_num / eval_num}\n")
        if self.save_video:
            return eval_text, eval_frames
        return eval_text
    
    def preprocess_img(self, ori_img):
        img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        img = img / 255.0 - 0.5
        img = img.transpose(2,0,1)
        img = img[np.newaxis, :]
        return img

    def __del__(self):
        self.env.close()


class ObservationWrapper(object):
    def __init__(self, env, n_obs_steps=5, horizon=9, max_path_length=400):
        self._env = env
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.max_path_length = max_path_length
        self.proprio_list = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']

        self.obs_store = {}

        obs_dict = self.reset()
        dummy_img = obs_dict['pixels'][0]
        dummy_proprio = obs_dict['proprio'][0]

    def reset(self, **kwargs):
        self.episode_step = 0

        obs = {}
        all_obs = self._env.reset(**kwargs)

        target_proprio = np.concatenate([all_obs[target_state] for target_state in self.proprio_list], axis=-1).astype(np.float32)
        first_proprio = target_proprio[np.newaxis, :]
        raw_img = all_obs['agentview_image']
        rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).copy()
        view_obs = np.flip(np.flip(rgb_img, axis=0).copy(), axis=1).copy()
        first_obs = view_obs[np.newaxis, :]

        self.obs_store['pixels'] = [first_obs for _ in range(self.n_obs_steps)]
        self.obs_store['proprio'] = [first_proprio for _ in range(self.n_obs_steps)]
        

        obs['pixels'] = np.concatenate(self.obs_store['pixels'][-1 * self.n_obs_steps : ])
        obs['proprio'] = np.concatenate(self.obs_store['proprio'][-1 * self.n_obs_steps : ])
        obs['goal_achieved'] = False
        return obs

    def step(self, action):
        obs = {}
        obs['goal_achieved'] = 0
    
        first_step = action[0]
        all_obs, reward, done, info = self._env.step(first_step)
        
        cur_proprio = np.concatenate([all_obs[target_state] for target_state in self.proprio_list], axis=-1).astype(np.float32)
        self.obs_store['proprio'].append(cur_proprio[np.newaxis, :])

        raw_img = all_obs['agentview_image']
        rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).copy()
        cur_img = np.flip(np.flip(rgb_img, axis=0).copy(), axis=1).copy()
        self.obs_store['pixels'].append(cur_img[np.newaxis, :])

        obs['goal_achieved'] = max(obs['goal_achieved'], reward)
        self.episode_step += 1

        obs['pixels'] = np.concatenate(self.obs_store['pixels'][-1 * self.n_obs_steps : ])
        obs['proprio'] = np.concatenate(self.obs_store['proprio'][-1 * self.n_obs_steps : ])

        if self.episode_step >= self.max_path_length:
            done = True
        return obs, reward, done, info

    def close(self):
        self._env.close()

    def __getattr__(self, name):
        return getattr(self._env, name)


class MimicgenTask(object):
    def __init__(self, task_name, max_env_horizon=300, depth=False, save_video=False):
        import mimicgen
        import robosuite_task_zoo
        import robosuite
        from robosuite.controllers import load_controller_config
        controller_config = load_controller_config(default_controller="OSC_POSE")
        self.depth = depth
        self.max_env_horizon = max_env_horizon

        self.proprio_list = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']
        env_args = {
            "env_name": task_name,
            "robots": "Panda",
            "controller_configs": controller_config,
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "use_camera_obs": True,
            "camera_names": "agentview",
            "camera_heights": 84,
            "camera_widths": 84,
        }
        if depth: env_args["camera_depths"] = True
        
        self.env = robosuite.make(**env_args)
        
        all_obs = self.env.reset()

        self.save_video=save_video

    def run_eval(self, eval_num, model, n_obs_steps=5):
        device = next(model.parameters()).device

        success_num = 0
        eval_text = []
        eval_frames = []

        for rollout in range(eval_num):
            all_obs = self.env.reset()
            
            third_obs, proprio_state = self.get_obs(all_obs)
            img_store = [third_obs] * n_obs_steps
            proprio_store = [proprio_state[np.newaxis, :]] * n_obs_steps

            if self.depth:
                depth_obs = self.preprocess_depth(all_obs["agentview_depth"])
                depth_store = [depth_obs] * n_obs_steps

            step_i = 0
            episode_frames = []
            while step_i < self.max_env_horizon:
                imgs = torch.from_numpy(np.concatenate(img_store[-1*n_obs_steps:])).float().to(device).unsqueeze(0)
                pros = torch.from_numpy(np.concatenate(proprio_store[-1*n_obs_steps:])).float().to(device).unsqueeze(0)

                if self.depth:
                    depths = torch.from_numpy(np.concatenate(depth_store[-1*n_obs_steps:])).float().to(device).unsqueeze(0)
                    actions = model.get_action(imgs, pros, depths)
                else:
                    actions = model.get_action(imgs, pros)
                action = actions.cpu().numpy()[0][0]
                action = np.clip(action, -1, 1)
                
                if self.save_video:
                    rendered = all_obs['agentview_image'].astype(np.uint8)
                    rendered = np.flip(rendered, axis=0)
                    episode_frames.append(rendered)
                
                all_obs, reward, done, _ = self.env.step(action)
                third_obs, proprio_state = self.get_obs(all_obs)
                img_store.append(third_obs)
                proprio_store.append(proprio_state[np.newaxis, :])

                if self.depth:
                    depth_obs = self.preprocess_depth(all_obs["agentview_depth"])
                    depth_store.append(depth_obs)
                step_i += 1

                success = reward >= 0.95

                if success:
                    success_num += success
                    break

            eval_frames = eval_frames + episode_frames
            print(f"env_{rollout}: final {success}")

        eval_text.append(f"total succ rate: {success_num / eval_num}\n")
        if self.save_video:
            return eval_text, eval_frames
        return eval_text

    def get_obs(self, all_obs):
        third_obs = self.preprocess_img(all_obs['agentview_image'].astype(np.uint8))
        proprio_state = np.concatenate([all_obs[target_state] for target_state in self.proprio_list], axis=-1).astype(np.float32)
        return third_obs, proprio_state

    def preprocess_img(self, ori_img):
        img = np.flip(ori_img, axis=0)
        img = cv2.resize(img, (84, 84))
        img = img.transpose(2, 0, 1) / 255.0 - 0.5
        img = img[np.newaxis, :]
        return img
    
    def preprocess_depth(self, depth):
        depth = np.flip(depth, axis=0)
        max_values = np.max(depth)
        min_values = np.min(depth)
        depth = (depth - min_values) / (max_values - min_values)
        depth = np.repeat(depth, 3, axis=-1)
        depth = depth.transpose(2, 0, 1)
        depth = depth[np.newaxis, :]
        return depth

    def __del__(self):
        self.env.close()


if __name__ == "__main__":
    import gymnasium as gym
    import mani_skill2.envs

    env = gym.make("PickCube-v0",
                obs_mode="sensor_data",
                control_mode="pd_ee_delta_pose",
                render_mode="rgb_array")
    print("Observation space", env.observation_space)
    print("Action space", env.action_space)
    
    print(env.obs_mode)

    obs, reset_info = env.reset(seed=0) 
    for i in range(10):
        print("qpos:", obs['agent']['qpos'][-2:])
        print("tcp_pose:", obs['extra']['tcp_pose'])
        action = env.action_space.sample()
        action[:-1] = 0
        action[-1] = 1
        obs, reward, terminated, truncated, info = env.step(action)

    env.close()

    print("done")
