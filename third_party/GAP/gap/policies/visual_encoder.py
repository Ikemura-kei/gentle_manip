import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision import transforms

import numpy as np
import utils
from policies.modules.transformer_modules import *

class SpatialSoftmax(nn.Module):
    """
    The spatial softmax layer (https://rll.berkeley.edu/dsae/dsae.pdf)
    """

    def __init__(self, in_c, in_h, in_w, num_kp=None):
        super().__init__()
        self._spatial_conv = nn.Conv2d(in_c, num_kp, kernel_size=1)

        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1, 1, in_w).float(),
            torch.linspace(-1, 1, in_h).float(),
        )

        pos_x = pos_x.reshape(1, in_w * in_h)
        pos_y = pos_y.reshape(1, in_w * in_h)
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)

        if num_kp is None:
            self._num_kp = in_c
        else:
            self._num_kp = num_kp

        self._in_c = in_c
        self._in_w = in_w
        self._in_h = in_h

    def forward(self, x):
        # print("the x shape is:", x.shape)
        assert x.shape[1] == self._in_c
        assert x.shape[2] == self._in_h
        assert x.shape[3] == self._in_w

        h = x
        if self._num_kp != self._in_c:
            h = self._spatial_conv(h)
        h = h.contiguous().view(-1, self._in_h * self._in_w)

        attention = F.softmax(h, dim=-1)
        keypoint_x = (
            (self.pos_x * attention).sum(1, keepdims=True).view(-1, self._num_kp)
        )
        keypoint_y = (
            (self.pos_y * attention).sum(1, keepdims=True).view(-1, self._num_kp)
        )
        keypoints = torch.cat([keypoint_x, keypoint_y], dim=1)
        return keypoints


class SpatialProjection(nn.Module):
    def __init__(self, input_shape, out_dim):
        super().__init__()

        assert (
            len(input_shape) == 3
        ), "[error] spatial projection: input shape is not a 3-tuple"
        in_c, in_h, in_w = input_shape
        num_kp = out_dim // 2
        self.out_dim = out_dim
        self.spatial_softmax = SpatialSoftmax(in_c, in_h, in_w, num_kp=num_kp)
        self.projection = nn.Linear(num_kp * 2, out_dim)

    def forward(self, x):
        out = self.spatial_softmax(x)
        out = self.projection(out)
        return out

    def output_shape(self, input_shape):
        return input_shape[:-3] + (self.out_dim,)


class ImgEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        net = resnet18(pretrained=False, progress=False)
        self.spatial_encoder = nn.Sequential(*list(net.children())[:-2])
        self.projection = SpatialProjection(input_shape=[512,3,3], out_dim=self.hidden_dim)

        self.temporal_position_encoding_fn = SinusoidalPositionEncoding(self.hidden_dim)
        self.temporal_transformer = TransformerDecoder(
            input_size=self.hidden_dim,
            num_layers=4,
            num_heads=6,
            head_output_size=64,
            mlp_hidden_size=256,
            dropout=0.1,
        )

    def forward(self, image):
        # Encode image inputs
        B,T,C,H,W = image.shape
        img = image.reshape(-1, *image.shape[2:]) # (Batch_size, n_obs_steps, ) -> (Batch_size*n_obs_steps, ) for encoder
        x = self.spatial_encoder(img) # (B*T, 512, 7, 7)
        x = self.projection(x) # (B*T, hidden_dim)
        x = x.view(B,T,1,-1) #unsqueeze(-2) for temporal
        pos_emb = self.temporal_position_encoding_fn(x)
        sh = x.shape
        x = x + pos_emb.unsqueeze(1)  # (B, T, 1, E)
        self.temporal_transformer.compute_mask(x.shape)
        x = utils.join_dimensions(x, 1, 2) # N, T * 1, E
        x = self.temporal_transformer(x)
        x = x.reshape(*sh)  # N,T,1,E

        return x[:,:,-1] # N,T,E


class ProEncoder(nn.Module):
    def __init__(self, hidden_dim, pro_dim=9, raw=False):
        super().__init__()
        self.proprio_dim = pro_dim
        self.hidden_dim = hidden_dim if not raw else self.proprio_dim
        
        # proprio encoder
        self.proprio_encoder = nn.Sequential(
            nn.Linear(self.proprio_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.hidden_dim),
            nn.ReLU(),
        ) if not raw else nn.Sequential(nn.Identity())

        self.temporal_position_encoding_fn = SinusoidalPositionEncoding(self.hidden_dim)
        self.temporal_transformer = TransformerDecoder(
            input_size=self.hidden_dim,
            num_layers=4,
            num_heads=6,
            head_output_size=64,
            mlp_hidden_size=256,
            dropout=0.1,
        )
        
    def forward(self, proprio):
        # Encode proprio inputs
        B,T,pd = proprio.shape
        proprio = proprio.reshape(-1, *proprio.shape[2:]) # (Batch_size, n_obs_steps, ) -> (Batch_size*n_obs_steps, ) for encoder
        x = self.proprio_encoder(proprio) # (B*T, E)
        x = x.view(B,T,1,-1) #unsqueeze(-2) for temporal
        pos_emb = self.temporal_position_encoding_fn(x)
        pos_emb = pos_emb[:,:x.shape[-1]]
        sh = x.shape
        x = x + pos_emb.unsqueeze(1)  # (B, T, 1, E)
        self.temporal_transformer.compute_mask(x.shape)
        x = utils.join_dimensions(x, 1, 2) # N, T * 1, E
        x = self.temporal_transformer(x)
        x = x.reshape(*sh)  # N,T,1,E
        
        return x[:,:,-1] # N,T,E


class DepthEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        net = resnet18(pretrained=False, progress=False)
        self.depth_encoder = nn.Sequential(*list(net.children())[:-2])
        self.projection = SpatialProjection(input_shape=[512,3,3], out_dim=self.hidden_dim)

        self.temporal_position_encoding_fn = SinusoidalPositionEncoding(self.hidden_dim)
        self.temporal_transformer = TransformerDecoder(
            input_size=self.hidden_dim,
            num_layers=4,
            num_heads=6,
            head_output_size=64,
            mlp_hidden_size=256,
            dropout=0.1,
        )

    def forward(self, image):
        # Encode image inputs
        B,T,C,H,W = image.shape
        img = image.reshape(-1, *image.shape[2:]) # (Batch_size, n_obs_steps, ) -> (Batch_size*n_obs_steps, ) for encoder
        x = self.depth_encoder(img) # (B*T, 512, 7, 7)
        x = self.projection(x) # (B*T, hidden_dim)
        x = x.view(B,T,1,-1) #unsqueeze(-2) for temporal
        pos_emb = self.temporal_position_encoding_fn(x)
        sh = x.shape
        x = x + pos_emb.unsqueeze(1)  # (B, T, 1, E)
        self.temporal_transformer.compute_mask(x.shape)
        x = utils.join_dimensions(x, 1, 2) # N, T * 1, E
        x = self.temporal_transformer(x)
        x = x.reshape(*sh)  # N,T,1,E

        return x[:,:,-1] # N,T,E


class FeatureExtractor(nn.Module):
    def __init__(self, hidden_dim, image=True, proprio=True, pro_dim=9, raw=False, depth=False):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.image = image
        self.proprio = proprio
        self.depth = depth

        self.feature_dim = 0

        if image:
            self.imgencoder = ImgEncoder(hidden_dim)
            self.feature_dim += self.imgencoder.hidden_dim            
        if proprio:
            self.proencoder = ProEncoder(hidden_dim, pro_dim, raw)
            self.feature_dim += self.proencoder.hidden_dim
        if depth:
            self.depencoder = DepthEncoder(hidden_dim)
            self.feature_dim += self.depencoder.hidden_dim

    def forward(self, image=None, proprio=None, depth=None, mask_flag=None):
        modal_list = []
        if self.image:
            modal_list.append(self.imgencoder(image))
        if self.depth:
            modal_list.append(self.depencoder(depth))
        if self.proprio:
            modal_list.append(self.proencoder(proprio))

        if mask_flag is not None and mask_flag < len(modal_list):
            modal_list[mask_flag] = torch.zeros_like(modal_list[mask_flag])

        if len(modal_list) > 1:
            concated = torch.cat(modal_list, dim=-1) # (B,T,E)
            return concated
        return modal_list[0]


if __name__ == '__main__':
    pass