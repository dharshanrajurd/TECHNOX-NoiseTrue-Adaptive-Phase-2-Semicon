"""
Prepares a fresh checkpoint directory seeded from best.pth's weights (NOT
increased_quality.pth -- that checkpoint's GAN pass cost PSNR without a
real texture gain, so we're branching off the pre-GAN best instead), so
train_colab.py's resume logic can pick it up as a starting point for the
Gram-matrix-loss fine-tune.

Run this ONCE before the fine-tune command.

Usage:
    python prep_gram_finetune_ckpt.py
"""

import torch
import os
from model_nafnet import NAFNetLiteAdaptive

SOURCE_BEST_PTH = r"C:\Users\dhars\Desktop\semicon phase 2\180epoch_nafnet_.pth"
NEW_CKPT_DIR = r"C:\Users\dhars\Desktop\semicon phase 2\gram_finetune_checkpoints"
BASE_CH = 48
LR = 1e-4          # conservative LR for fine-tuning, matches earlier fine-tune approach
EPOCHS = 15         # matches --epochs in the training command below

os.makedirs(NEW_CKPT_DIR, exist_ok=True)

device = torch.device("cpu")  # just repackaging weights, no GPU needed here

model = NAFNetLiteAdaptive(base_ch=BASE_CH)
loaded = torch.load(SOURCE_BEST_PTH, map_location=device)

# handle both raw state_dict (best.pth-style) and wrapped checkpoint dict
# (latest.pth-style, with keys like "model", "opt_g", "epoch", etc.)
if isinstance(loaded, dict) and "model" in loaded:
    state_dict = loaded["model"]
    print("Detected wrapped checkpoint format -- extracting weights from 'model' key.")
else:
    state_dict = loaded
    print("Detected raw state_dict format.")

model.load_state_dict(state_dict)

opt_g = torch.optim.Adam(model.parameters(), lr=LR)
sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=EPOCHS)

checkpoint = {
    "epoch": -1,              # so start_epoch becomes 0 on resume
    "model": model.state_dict(),
    "disc": None,
    "opt_g": opt_g.state_dict(),
    "opt_d": None,
    "sched_g": sched_g.state_dict(),
    "best_val": 0.12017,      # your known best_val from the NAFNet 150-epoch run
}

torch.save(checkpoint, os.path.join(NEW_CKPT_DIR, "latest.pth"))
print(f"Prepped fresh checkpoint at {NEW_CKPT_DIR}\\latest.pth")
print("Seeded from best.pth (pre-GAN), fresh optimizer/scheduler, ready for the Gram fine-tune.")
