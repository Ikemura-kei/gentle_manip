import os
import io
import copy
import yaml
import time
import hydra
import torch
import warnings
import numpy as np
from easydict import EasyDict
from omegaconf import OmegaConf

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

import utils
import dataset
import policies.bc

import warnings
warnings.filterwarnings("ignore")
from torch.utils.tensorboard import SummaryWriter


class WorkSpace:
    def __init__(self, cfg):
        from pathlib import Path
        self.work_dir = Path.cwd()
        self.cfg = cfg
        # =====
        self.device = cfg.device
        
        demo_num = self.cfg.task.demo_num
        
        train_cfg = copy.deepcopy(cfg.dataset)
        train_cfg.kwargs.demo_range = [0, demo_num]
        train_dataset = eval(train_cfg._target_)(**train_cfg.kwargs)
        self.train_dataloader = train_dataset.get_dataloader()
        
        valid_demo_num = int(0.1*demo_num)
        valid_cfg = copy.deepcopy(cfg.dataset)
        valid_cfg.kwargs.demo_range = [0, valid_demo_num]
        valid_dataset = eval(valid_cfg._target_)(**valid_cfg.kwargs)
        self.valid_dataloader = valid_dataset.get_dataloader(shuffle=False)

        self.policy = eval(cfg.policy._target_)(cfg.policy).to(self.device)

        self.optimizer = eval(cfg.optimizer._target_)(filter(lambda p: p.requires_grad, self.policy.parameters()), **cfg.optimizer.network_kwargs)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, cfg.epoch)

        self.sw = SummaryWriter(log_dir=str(self.work_dir)+'/tb')

    def save_snapshot(self, idx):
        save_path = os.path.join(self.work_dir, f'snapshot_{idx}.pth')
        save_data = self.policy.state_dict()
        data = {
            "policy": save_data,
            "optimizer": self.optimizer.state_dict(),
            "cfg": self.cfg
        }
        with open(save_path,"wb") as f:
            torch.save(data,f)

    def train(self):
        modulation_starts = 0
        modulation_ends = 50
        
        min_valid_loss = 1e5
        for epoch in range(self.cfg.epoch):
            train_losses = []
            for batch in self.train_dataloader:
                img = batch['img'].to(self.cfg.device, non_blocking=True)
                proprio = batch['proprio'].to(self.cfg.device, non_blocking=True)
                gt_action = batch['actions'].to(self.cfg.device, non_blocking=True)

                self.optimizer.zero_grad()

                loss = self.policy.compute_loss(img, proprio, gt_action)
                train_losses.append(loss.item())
                loss.backward()
                
                # gradient adjustment
                phase_p = torch.max(batch['phase']).item()

                lambda = self.cfg.lambda
                coeff_p = 1 - lambda * phase_p

                if  modulation_starts <= epoch <= modulation_ends:
                    for name, parms in self.policy.encoder.named_parameters():
                        if 'pro' in name:
                            parms.grad *= coeff_p
                
                self.optimizer.step()
                self.lr_scheduler.step()

            epoch_loss = np.mean(train_losses)
            self.sw.add_scalar('Loss/train', epoch_loss, epoch)

            valid_loss = self.valid()
            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                self.save_snapshot("valid")
                self.sw.add_scalar('Loss/valid', valid_loss, epoch)
            if epoch == self.cfg.epoch-1:
                self.save_snapshot(epoch)

    def valid(self):
        valid_losses = []
        for batch in self.valid_dataloader:
            img = batch['img'].to(self.cfg.device, non_blocking=True)
            proprio = batch['proprio'].to(self.cfg.device, non_blocking=True)
            gt_action = batch['actions'].to(self.cfg.device, non_blocking=True)

            with torch.no_grad():
                loss = self.policy.compute_loss(img, proprio, gt_action)
                valid_losses.append(loss.item())

        valid_loss = np.mean(valid_losses)
        return valid_loss
        

@hydra.main(config_path='cfgs', config_name='gap')
def main(cfg):
    np.random.seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    w = WorkSpace(cfg)
    w.train()


if __name__ == '__main__':
	main()
