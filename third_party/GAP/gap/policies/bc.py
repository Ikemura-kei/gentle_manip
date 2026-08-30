from policies.policy import BasePolicy
from policies.visual_encoder import *
import hydra
import policies
from policies.head import *
from itertools import chain

import torch.nn.functional as F
import utils
import torch.nn as nn
import torch


class BCPolicy(BasePolicy):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        encoder_cfg = cfg.encoder
        # get visual encoder
        self.encoder = eval(encoder_cfg._target_)(**encoder_cfg.network_kwargs).to(cfg.device)

        self.cfg.head.network_kwargs.feature_dim = self.encoder.feature_dim
        self.head = eval(self.cfg.head._target_)(
            **self.cfg.head.network_kwargs
        )

    def get_action(self, img, proprio, depth=None, mask_flag=None):
        with torch.no_grad():
            if img.shape[0] != 1: img = img.unsqueeze(0)
            if proprio.shape[0] != 1: proprio = proprio.unsqueeze(0)
            if depth is not None and depth.shape[0] != 1: depth = depth.unsqueeze(0)
            hidden = self.encoder(img, proprio, depth, mask_flag) # (Batch=1,T,E)
            action = self.head.get_action(hidden)
        
        return action
    
    def compute_loss(self, img, proprio, gt_action, depth=None, mask_flag=None):
        B,T,C,H,W = img.shape

        hidden = self.encoder(img, proprio, depth, mask_flag) # (B,T,E)
        loss = self.head.loss_fn(hidden, gt_action)
        return loss
