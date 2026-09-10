"""RGB baseline pieces: an ImageNet-pretrained ResNet-18 encoder and train-only image augmentation.

WHY, and the references:
  * Diffusion Policy (Chi et al., RSS 2023) uses an ImageNet-pretrained ResNet-18 for its REAL
    experiments and replaces BatchNorm with GroupNorm, because BatchNorm interacts badly with the
    EMA the training loop keeps. We use EMA, so the same substitution applies here.
  * ACT (Zhao et al., RSS 2023) likewise uses a pretrained ResNet-18 backbone on real data.
  * robomimic (Mandlekar et al., CoRL 2021) found random crop/shift the most impactful image
    augmentation for manipulation; DPPO already ships RandomShiftsAug and we keep it.
  * Photometric jitter is standard in large-scale real pipelines (RT-1, Octo) and is what covers
    illumination drift across collection days; DPPO ships none, so it is added here.
DPPO's stock encoder is a from-scratch depth-1, 128-dim ViT, which is a weak choice at ~120
demonstrations and would make the RGB baseline easy to dismiss as underpowered.

PREPROCESSING CONTRACT — the reason normalization lives INSIDE the encoder:
VisionDiffusionMLP hands the backbone float pixels on the RAW 0-255 scale (its forward only calls
.float()), and deploy_real_dppo.py feeds the same 0-255 range. Doing the /255 and the ImageNet
mean/std here means TRAINING AND DEPLOY CANNOT DIVERGE: neither side has to remember to normalize.
Do not move it into the data pipeline.
"""
from __future__ import annotations

import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _bn_to_gn(module: nn.Module, groups: int = 16) -> nn.Module:
    """Replace every BatchNorm2d with GroupNorm, in place (Diffusion Policy's fix for EMA)."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            c = child.num_features
            setattr(module, name, nn.GroupNorm(num_groups=max(1, c // groups), num_channels=c))
        else:
            _bn_to_gn(child, groups)
    return module


class ResNet18Encoder(nn.Module):
    """ImageNet-pretrained ResNet-18 trunk -> (B, 512). Consumes 0-255 float, normalizes internally.

    Interface matches what VisionDiffusionMLP needs from a backbone: `.repr_dim` and a forward
    returning features it can flatten. `spatial_emb > 0` is NOT supported (that path wants
    `.num_patch` / `.patch_repr_dim`, which are ViT concepts) -- keep spatial_emb at 0.
    """

    def __init__(self, obs_shape, num_channel: int = 3, pretrained: bool = True,
                 groupnorm: bool = True, freeze_stem: bool = False, freeze_conv: bool = False):
        super().__init__()
        if int(num_channel) != 3:
            raise ValueError(
                f"ResNet18Encoder needs 3 input channels, got {num_channel}. A pretrained stem is "
                "3-channel, so img_cond_steps>1 (which stacks frames on channels) cannot use it; "
                "either keep img_cond_steps=1 or inflate the stem deliberately.")
        from torchvision.models import ResNet18_Weights, resnet18

        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        if groupnorm:
            _bn_to_gn(net)      # AFTER loading weights: GroupNorm has no running stats to lose
        net.fc = nn.Identity()
        self.net = net
        self.repr_dim = 512
        self.register_buffer("_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1) * 255.0)
        self.register_buffer("_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1) * 255.0)
        if freeze_stem:
            for p in (list(net.conv1.parameters()) + list(net.layer1.parameters())):
                p.requires_grad = False
        if freeze_conv:
            # FROZEN backbone (user, 2026-09-10): every conv/linear weight fixed at its ImageNet
            # value, GroupNorm affine LEFT TRAINABLE on purpose. _bn_to_gn installs FRESH GroupNorms
            # (identity init, ImageNet running stats gone), so freezing those too would push the
            # pretrained filters through an uncalibrated normalization -- that would confound
            # "frozen features" with "miscalibrated norms". Freezing the filters is the variable.
            for m in net.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    for p in m.parameters():
                        p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x.float() - self._mean) / self._std        # 0-255 -> ImageNet-normalized
        return self.net(x)


class TrainOnlyImageAug(nn.Module):
    """RandomShiftsAug + photometric jitter, applied ONLY in training mode.

    Being an nn.Module that checks `self.training` matters: DPPO's RandomShiftsAug is a plain
    callable and VisionDiffusionMLP.forward applies it with NO train/eval check, so a model built
    with augment=True would augment at inference too. Deploy currently escapes that only because it
    never passes `augment`. This class is correct by construction under model.eval().

    Photometric terms operate on the 0-255 scale, per sample, and identically across every frame in
    a stacked chunk (channels are (T*3)) -- per-frame jitter would break temporal consistency the
    same way per-frame crops would.

    `channel_gain` stands in for hue rotation deliberately: the measured drift across our collection
    days is a WHITE-BALANCE shift (R/G 1.4 %, B/G 0.4 %, luminance 0.5 %), which per-channel gain
    models directly, while a hue rotation would move colours the policy legitimately uses to tell a
    red tomato from a brown mushroom from white tofu.
    """

    def __init__(self, pad: int = 4, brightness: float = 0.25, contrast: float = 0.25,
                 saturation: float = 0.2, channel_gain: float = 0.05):
        super().__init__()
        from model.common.modules import RandomShiftsAug
        self.shift = RandomShiftsAug(pad=pad) if pad and pad > 0 else None
        self.brightness, self.contrast = float(brightness), float(contrast)
        self.saturation, self.channel_gain = float(saturation), float(channel_gain)

    @staticmethod
    def _u(b, lo, hi, device, dtype):
        return torch.empty(b, 1, 1, 1, device=device, dtype=dtype).uniform_(lo, hi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        if self.shift is not None:
            x = self.shift(x)
        B, C, H, W = x.shape
        if C % 3 != 0:
            return x
        d, t = x.device, x.dtype
        v = x.view(B, C // 3, 3, H, W)                    # (B, frames, 3, H, W)
        if self.brightness > 0:
            v = v * self._u(B, 1 - self.brightness, 1 + self.brightness, d, t).unsqueeze(1)
        if self.contrast > 0:
            gray = (0.299 * v[:, :, 0] + 0.587 * v[:, :, 1] + 0.114 * v[:, :, 2])
            m = gray.mean(dim=(-1, -2), keepdim=True).unsqueeze(2)
            v = (v - m) * self._u(B, 1 - self.contrast, 1 + self.contrast, d, t).unsqueeze(1) + m
        if self.saturation > 0:
            gray = (0.299 * v[:, :, 0] + 0.587 * v[:, :, 1] + 0.114 * v[:, :, 2]).unsqueeze(2)
            v = (v - gray) * self._u(B, 1 - self.saturation, 1 + self.saturation, d, t).unsqueeze(1) + gray
        if self.channel_gain > 0:                          # white-balance drift
            g = torch.empty(B, 1, 3, 1, 1, device=d, dtype=t).uniform_(
                1 - self.channel_gain, 1 + self.channel_gain)
            v = v * g
        return v.view(B, C, H, W).clamp_(0.0, 255.0)


def VisionDiffusionMLPAug(backbone, *, aug_pad: int = 4, aug_brightness: float = 0.25,
                          aug_contrast: float = 0.25, aug_saturation: float = 0.2,
                          aug_channel_gain: float = 0.05, **kw):
    """DPPO's VisionDiffusionMLP with its augmentation swapped for TrainOnlyImageAug.

    A FACTORY, not a subclass or a wrapper, so the returned module IS a VisionDiffusionMLP: the
    state_dict keys are byte-identical to a stock one, and deploy can rebuild the policy with the
    plain class and load these weights. A wrapper would have prefixed every key with `inner.`;
    a subclass would have changed the class name recorded nowhere but still risked divergence.

    Neither RandomShiftsAug nor TrainOnlyImageAug contributes parameters, so swapping them cannot
    affect the checkpoint at all -- augmentation is purely a training-time input transform.
    """
    from model.diffusion.mlp_diffusion import VisionDiffusionMLP

    kw["augment"] = True                       # allocate the aug slot, then replace it
    net = VisionDiffusionMLP(backbone=backbone, **kw)
    net.aug = TrainOnlyImageAug(pad=aug_pad, brightness=aug_brightness, contrast=aug_contrast,
                                saturation=aug_saturation, channel_gain=aug_channel_gain)
    return net
