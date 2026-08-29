import os
import io
import yaml
import json
import time
import hydra
from hydra.utils import instantiate
import torch
import warnings
import numpy as np
from easydict import EasyDict
from omegaconf import OmegaConf

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

import utils
import policies.bc
import envs

import warnings
warnings.filterwarnings("ignore")


class WorkSpace:
    def __init__(self, cfg):
        from pathlib import Path
        self.work_dir = Path.cwd()
        self.cfg = cfg
        self.device = cfg.device

        self.env = instantiate(self.cfg.task.TaskEnv)
        
        self.policy = eval(cfg.policy._target_)(cfg.policy).to(self.device)
        if cfg.load_bc:
            print("load bc path: ", cfg.snapshot)
            self.load_snapshot(cfg.snapshot)

    def load_snapshot(self, snapshot):
        with open(snapshot, "rb") as f:
            data = torch.load(f)

        target_load_dict = {key.replace('policy_head', 'head'): data['policy'][key] for key in data['policy'].keys()}
        self.policy.load_state_dict(target_load_dict)

    def run_eval(self):
        if self.cfg.save_video:
            eval_text, eval_frames = self.env.run_eval(self.cfg.eval_num, self.policy, self.cfg.n_obs_steps)
            from video import VideoRecorder
            recorder = VideoRecorder(self.work_dir)
            recorder.init()
            for frame in eval_frames:
                recorder.record(frame)
            recorder.save('eval_video.mp4')
        else:
            eval_text = self.env.run_eval(self.cfg.eval_num, self.policy, self.cfg.n_obs_steps)
        
        with open(str(self.work_dir)+'/sr.txt', 'a') as fout: fout.writelines(eval_text)


@hydra.main(config_path='cfgs', config_name='eval')
def main(cfg):
    np.random.seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    w = WorkSpace(cfg)
    w.run_eval()

if __name__ == '__main__':
	main()
