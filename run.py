"""
NoiseTrue-Adaptive -- official Phase 2 submission entry script.

Required submission component #1 (KLA AI Hackathon problem statement):
  - Standalone Python script (not a notebook) -- this is that.
  - Accepts a test images directory and an output directory.
  - Loads the trained model, runs inference on all input images.
  - Writes denoised outputs to the specified directory.
  - Runs with zero manual edits (aside from --model_path, which is a
    legitimate CLI arg, not a code edit).
  - Scored partly on end-to-end speed (script startup + disk I/O +
    inference + disk I/O, timed) -- this script times and prints all of
    that, matching how the evaluator will measure it.

Usage:
    python evaluate.py --input_dir /path/to/test/NoisyLR --output_dir /path/to/outputs

Fully self-contained: the model architecture is defined directly in this
file, so there is no dependency on src/ or any other project file. The
only external requirement is the trained checkpoint at
models/nafnet+gram_best.pth (relative to this script, overridable with
--model_path), loaded once at startup. No internet access, API keys, or
additional downloads are used or required at any point.
"""

import time
_T_START = time.time()   # capture true script-startup time before heavy imports

import os
import sys
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn


# =====================================================================
# Model architecture (self-contained -- no external src/ imports needed)
# Matches src/model_nafnet.py exactly: NAFNetLiteAdaptive, base_ch=48.
# =====================================================================

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, ch, expand=2):
        super().__init__()
        dw_ch = ch * expand

        self.norm1 = LayerNorm2d(ch)
        self.conv1 = nn.Conv2d(ch, dw_ch, 1)
        self.conv2 = nn.Conv2d(dw_ch, dw_ch, 3, padding=1, groups=dw_ch)
        self.sg1 = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_ch // 2, dw_ch // 2, 1),
        )
        self.conv3 = nn.Conv2d(dw_ch // 2, ch, 1)

        ffn_ch = ch * expand
        self.norm2 = LayerNorm2d(ch)
        self.conv4 = nn.Conv2d(ch, ffn_ch, 1)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_ch // 2, ch, 1)

        self.beta = nn.Parameter(torch.zeros(1, ch, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, ch, 1, 1))

    def forward(self, x):
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sg1(y)
        y = y * self.sca(y)
        y = self.conv3(y)
        x = x + y * self.beta

        y = self.norm2(x)
        y = self.conv4(y)
        y = self.sg2(y)
        y = self.conv5(y)
        x = x + y * self.gamma
        return x


class NAFNetLite(nn.Module):
    def __init__(self, base_ch=48):
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8

        self.stem = nn.Conv2d(1, c1, 3, padding=1)

        self.enc1 = NAFBlock(c1)
        self.down1 = nn.Conv2d(c1, c2, 2, stride=2)
        self.enc2 = NAFBlock(c2)
        self.down2 = nn.Conv2d(c2, c3, 2, stride=2)
        self.enc3 = NAFBlock(c3)
        self.down3 = nn.Conv2d(c3, c4, 2, stride=2)

        self.bottleneck = NAFBlock(c4)

        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.reduce3 = nn.Conv2d(c3 * 2, c3, 1)
        self.dec3 = NAFBlock(c3)

        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.reduce2 = nn.Conv2d(c2 * 2, c2, 1)
        self.dec2 = NAFBlock(c2)

        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.reduce1 = nn.Conv2d(c1 * 2, c1, 1)
        self.dec1 = NAFBlock(c1)

        self.final_up = nn.Sequential(
            nn.Conv2d(c1, c1 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, 1, 3, padding=1),
        )

    def _forward_encoder(self, x):
        x0 = self.stem(x)
        e1 = self.enc1(x0)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        b = self.down3(e3)
        return e1, e2, e3, b

    def _forward_decoder(self, e1, e2, e3, b):
        d3 = self.up3(b)
        d3 = self.reduce3(torch.cat([d3, e3], dim=1))
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self.reduce2(torch.cat([d2, e2], dim=1))
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self.reduce1(torch.cat([d1, e1], dim=1))
        d1 = self.dec1(d1)

        return self.final_up(d1)

    def forward(self, x):
        e1, e2, e3, b = self._forward_encoder(x)
        b = self.bottleneck(b)
        return self._forward_decoder(e1, e2, e3, b)


class DegradationEstimator(nn.Module):
    def __init__(self, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class FiLM(nn.Module):
    def __init__(self, cond_dim, feat_ch):
        super().__init__()
        self.to_scale = nn.Linear(cond_dim, feat_ch)
        self.to_shift = nn.Linear(cond_dim, feat_ch)
        # zero-init: FiLM starts as identity, only learns to deviate if it helps
        # (irrelevant at inference time since real weights get loaded below,
        # kept for architectural fidelity with training).
        nn.init.zeros_(self.to_scale.weight)
        nn.init.zeros_(self.to_scale.bias)
        nn.init.zeros_(self.to_shift.weight)
        nn.init.zeros_(self.to_shift.bias)

    def forward(self, feat, cond):
        scale = self.to_scale(cond).unsqueeze(-1).unsqueeze(-1)
        shift = self.to_shift(cond).unsqueeze(-1).unsqueeze(-1)
        return feat * (1 + scale) + shift


class NAFNetLiteAdaptive(NAFNetLite):
    def __init__(self, base_ch=48):
        super().__init__(base_ch=base_ch)
        c4 = base_ch * 8
        self.estimator = DegradationEstimator(out_dim=32)
        self.film = FiLM(cond_dim=32, feat_ch=c4)

    def forward(self, x):
        cond = self.estimator(x)
        e1, e2, e3, b = self._forward_encoder(x)
        b = self.bottleneck(b)
        b = self.film(b, cond)
        return self._forward_decoder(e1, e2, e3, b)


# =====================================================================
# Entry point
# =====================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(HERE, "models", "nafnet+gram_best.pth")


def parse_args():
    p = argparse.ArgumentParser(description="NoiseTrue-Adaptive restoration -- Phase 2 entry point")
    p.add_argument("--input_dir", required=True, help="Directory containing degraded .npy images")
    p.add_argument("--output_dir", required=True, help="Directory to write restored .npy images (created if missing)")
    p.add_argument("--model_path", default=DEFAULT_MODEL_PATH,
                   help="Path to the trained checkpoint (default: models/nafnet+gram_best.pth)")
    p.add_argument("--base_ch", type=int, default=48,
                   help="Model width used at training time (48 for the submitted checkpoint)")
    p.add_argument("--batch_size", type=int, default=16,
                   help="Maximum images per batch when input shapes match")
    return p.parse_args()


def load_model(model_path, base_ch, device):
    model = NAFNetLiteAdaptive(base_ch=base_ch)
    ckpt = torch.load(model_path, map_location=device)
    # handles raw state_dict, or a wrapped checkpoint dict (key "model" or "state_dict"),
    # since Stage 2/3 training saves wrapped resumable checkpoints (see train.py)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_and_normalize(path):
    arr = np.load(path).astype(np.float32)
    # normalize if not already in [0,1] -- protects against test images
    # provided as 0-255 uint-range floats instead of pre-normalized
    if arr.max() > 1.0:
        arr = arr / 255.0 if arr.max() <= 255 else arr / arr.max()
    return arr


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.model_path):
        print(f"ERROR: model checkpoint not found at {args.model_path}")
        sys.exit(1)

    print(f"Loading model from: {args.model_path}")
    model = load_model(args.model_path, args.base_ch, device)
    t_model_loaded = time.time()
    print(f"Model loaded in {t_model_loaded - _T_START:.3f}s (from script start)")

    input_files = sorted(f for f in os.listdir(args.input_dir) if f.lower().endswith(".npy"))
    if not input_files:
        print(f"No .npy files found in {args.input_dir}. Nothing to do.")
        return
    print(f"Found {len(input_files)} degraded images to restore.")

    # load each file from disk exactly once: read + normalize here, then
    # group by shape so same-size inputs can be batched together for speed
    groups = defaultdict(list)          # shape -> [(fname, normalized_array), ...]
    for fname in input_files:
        arr = load_and_normalize(os.path.join(args.input_dir, fname))
        groups[arr.shape].append((fname, arr))

    t_inference_start = time.time()
    processed = 0
    with torch.no_grad():
        for shape, items in groups.items():
            for i in range(0, len(items), args.batch_size):
                batch_items = items[i:i + args.batch_size]
                batch_files = [f for f, _ in batch_items]

                arrays = [a for _, a in batch_items]
                batch = torch.from_numpy(np.stack(arrays)).unsqueeze(1).to(device)

                out = model(batch)

                # sanitize: no NaN/Inf, clipped to valid [0,1] range
                out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
                out = out.clamp(0, 1).cpu().numpy()

                for j, fname in enumerate(batch_files):
                    restored = out[j, 0].astype(np.float32)  # shape (H, W)
                    np.save(os.path.join(args.output_dir, fname), restored)

                processed += len(batch_files)
                print(f"  Processed {processed}/{len(input_files)} images...", flush=True)

    t_inference_end = time.time()

    print(f"\nInference on {len(input_files)} images took "
          f"{t_inference_end - t_inference_start:.3f}s "
          f"({(t_inference_end - t_inference_start) / max(len(input_files), 1) * 1000:.2f} ms/image)")
    print(f"Total script time (startup + model load + inference + I/O): "
          f"{t_inference_end - _T_START:.3f}s")
    print(f"Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()