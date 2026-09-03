"""
RRDBNetAdaptive: an ESRGAN-style generator using RRDB (Residual-in-Residual
Dense Block) blocks, adapted to your pipeline.

What's borrowed from ESRGAN: the RRDB block itself -- each one contains
3 densely-connected sub-blocks (every conv layer sees the concatenated
outputs of all earlier layers in that sub-block), each wrapped in a
residual connection scaled down by 0.2 (a stability trick -- without this
scaling, stacking many RRDBs tends to make training unstable). This dense,
heavy-connectivity design is what gives ESRGAN-family models their strong
fine-texture/detail generation, which is your current weak point.

What's kept from your existing pipeline, unchanged in spirit:
  - FiLM + DegradationEstimator conditioning (your project's novelty)
  - Global residual skip (bicubic-upsampled input added to output) --
    same idea as your NAFNetProAdaptive, helps preserve simple/undamaged
    regions instead of redrawing everything from scratch.
  - Same interface: (B,1,H,W) noisy in, (B,1,2H,2W) restored out.
  - Drop-in compatible with your existing PatchDiscriminator, VGG
    perceptual loss, and combined_loss_v3/combined_generator_loss --
    nothing about your loss/training code needs to change, only the
    generator itself.

Usage in train_colab.py: swap the model import/instantiation line, same
as with model_nafnet_pro.py. Needs a FRESH checkpoint directory -- weight
shapes are completely different from your existing checkpoints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_nafnet import DegradationEstimator, FiLM


class ResidualDenseBlock(nn.Module):
    """5 conv layers with dense connections (each layer sees all previous
    layers' outputs concatenated), residual-scaled by 0.2 for stability."""

    def __init__(self, ch=64, growth=32):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, growth, 3, padding=1)
        self.conv2 = nn.Conv2d(ch + growth, growth, 3, padding=1)
        self.conv3 = nn.Conv2d(ch + 2 * growth, growth, 3, padding=1)
        self.conv4 = nn.Conv2d(ch + 3 * growth, growth, 3, padding=1)
        self.conv5 = nn.Conv2d(ch + 4 * growth, ch, 3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.act(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.act(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x + 0.2 * x5   # residual scaling for training stability


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block: 3 ResidualDenseBlocks in sequence,
    wrapped in an outer residual connection, also scaled by 0.2."""

    def __init__(self, ch=64, growth=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(ch, growth)
        self.rdb2 = ResidualDenseBlock(ch, growth)
        self.rdb3 = ResidualDenseBlock(ch, growth)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + 0.2 * out


class RRDBNetAdaptive(nn.Module):
    def __init__(self, ch=64, growth=32, num_blocks=12):
        """
        ch: main feature channel width throughout the network
        growth: growth channels inside each dense block (ESRGAN default: 32)
        num_blocks: how many RRDBs to stack (ESRGAN paper uses 23 for large
            models; 12 is a lighter but still substantial depth, a
            reasonable starting point given your dataset size -- can be
            increased if training is stable and you have GPU budget to spare)
        """
        super().__init__()
        self.conv_first = nn.Conv2d(1, ch, 3, padding=1)

        self.body = nn.Sequential(*[RRDB(ch, growth) for _ in range(num_blocks)])
        self.trunk_conv = nn.Conv2d(ch, ch, 3, padding=1)

        # FiLM conditioning applied to the trunk features, same idea as before
        self.estimator = DegradationEstimator(out_dim=32)
        self.film = FiLM(cond_dim=32, feat_ch=ch)

        # 2x upsample via pixel shuffle (standard ESRGAN-style upsampling)
        self.upsample = nn.Sequential(
            nn.Conv2d(ch, ch * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_hr = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv_last = nn.Conv2d(ch, 1, 3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        base = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)

        cond = self.estimator(x)

        feat = self.conv_first(x)
        trunk = self.trunk_conv(self.body(feat))
        trunk = self.film(trunk, cond)
        feat = feat + trunk   # global residual around the whole RRDB body

        feat = self.upsample(feat)
        feat = self.act(self.conv_hr(feat))
        residual = self.conv_last(feat)

        return torch.clamp(base + residual, 0, 1)


if __name__ == "__main__":
    model = RRDBNetAdaptive(ch=64, growth=32, num_blocks=12)
    x = torch.randn(2, 1, 128, 128)
    out = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
