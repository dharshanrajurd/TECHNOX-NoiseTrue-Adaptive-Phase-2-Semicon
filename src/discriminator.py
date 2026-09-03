"""
Small PatchGAN discriminator, used only during training (never at inference,
so it costs zero inference speed). Classifies overlapping patches as
real/fake rather than the whole image -- standard choice for restoration
GANs (pix2pix, Real-ESRGAN) since it penalizes local texture quality
without needing a global real/fake judgment that's hard to define for
degraded-image restoration.

Kept intentionally small (4 conv layers) since its only job is to provide a
gradient signal, not to be a strong classifier -- an overly strong
discriminator can dominate training and push the generator toward
hallucinated texture, exactly what the original design avoided by skipping
GAN loss entirely. Use a LOW adv_weight (0.01-0.02) in the combined loss to
keep it subordinate to the reconstruction losses.
"""

import torch.nn as nn


class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=1, base_ch=32):
        super().__init__()

        def block(in_c, out_c, stride=2, norm=True):
            layers = [nn.Conv2d(in_c, out_c, 4, stride=stride, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_c, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(in_ch, base_ch, norm=False),
            *block(base_ch, base_ch * 2),
            *block(base_ch * 2, base_ch * 4),
            *block(base_ch * 4, base_ch * 8, stride=1),
            nn.Conv2d(base_ch * 8, 1, 4, stride=1, padding=1),  # patch logit map
        )

    def forward(self, x):
        return self.net(x)
