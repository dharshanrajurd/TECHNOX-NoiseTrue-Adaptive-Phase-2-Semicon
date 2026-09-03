"""
=============================================================================
TRAINING SCRIPT — TECHNOX NoiseTrue-Adaptive
=============================================================================
Required submission component #2: reproduces the training of our submitted
model, weights/nafnet+gram_best.pth

Runs on any machine with PyTorch installed (GPU strongly recommended for
realistic runtimes; falls back to CPU automatically if no GPU is found).
No cloud-specific setup, paths, or notebooks required.

Our final model was produced in THREE sequential stages using this single
script with different flags. Run them in order to reproduce the submitted
checkpoint from scratch.

-----------------------------------------------------------------------------
PREREQUISITES
-----------------------------------------------------------------------------
    pip install -r requirements.txt

Dataset layout expected (paired .npy files, matching filenames):
    <data_root>/GT/         256x256 ground truth images
    <data_root>/NoisyLR/    128x128 noisy low-resolution inputs

-----------------------------------------------------------------------------
STAGE 1 — Base training (150 epochs, reconstruction losses only)
-----------------------------------------------------------------------------
Trains NAFNetLiteAdaptive from scratch using the composite reconstruction
loss (Charbonnier + Sobel edge + VGG perceptual + FFT + MS-SSIM).
No adversarial term, no Gram term at this stage.

    python train.py \
        --gt_dir <data_root>/GT \
        --noisy_dir <data_root>/NoisyLR \
        --ckpt_dir ./checkpoints_stage1 \
        --epochs 150

    Result: best val_loss 0.12017

-----------------------------------------------------------------------------
STAGE 2 — Adversarial fine-tune (epochs 150 -> 180)
-----------------------------------------------------------------------------
Resumes from Stage 1's checkpoint (same --ckpt_dir) and continues for 30
more epochs with a PatchDiscriminator (LSGAN) active, to sharpen fine
texture. Adversarial weight kept deliberately low (0.015) so it refines
detail without dominating the reconstruction objective.

    python train.py \
        --gt_dir <data_root>/GT \
        --noisy_dir <data_root>/NoisyLR \
        --ckpt_dir ./checkpoints_stage1 \
        --epochs 180 \
        --use_gan \
        --gan_start_epoch 150

    Result: modest visual sharpening; slight PSNR trade-off, consistent
    with the known perception-distortion tradeoff.

-----------------------------------------------------------------------------
STAGE 3 — Gram matrix texture fine-tune (15 epochs)  <-- PRODUCES FINAL MODEL
-----------------------------------------------------------------------------
Resumes from Stage 2's weights with the Gram matrix texture loss enabled
and loss weights rebalanced (Charbonnier/MS-SSIM reduced, FFT raised) to
push back against over-smoothing.

First, repackage Stage 2's checkpoint into a fresh resumable checkpoint
(resets optimizer/scheduler for a clean fine-tune phase):

    python src/prep_gram_finetune_ckpt.py

Then run the fine-tune:

    python train.py \
        --gt_dir <data_root>/GT \
        --noisy_dir <data_root>/NoisyLR \
        --ckpt_dir ./checkpoints_stage3 \
        --epochs 15 \
        --gram_weight 0.05 \
        --vgg_weight 0.05 \
        --fft_weight 0.07 \
        --msssim_weight 0.15

    Result: best val_loss 0.11536
    This produces weights/nafnet+gram_best.pth -- OUR SUBMITTED MODEL.
    Validation: beat the Stage 2 model on 30/30 test images
    (avg PSNR 23.561 vs 23.086 dB).

-----------------------------------------------------------------------------
RUNTIME NOTES
-----------------------------------------------------------------------------
Checkpoints save every epoch, and re-running any stage's command
automatically resumes from the last saved checkpoint in its --ckpt_dir,
making the pipeline resilient to interruptions -- just re-run the same
command to continue where it left off. Runs on CPU if no GPU is present,
just proportionally slower; a CUDA-capable GPU is recommended for the
full 150+30+15 epoch schedule.
=============================================================================
"""

import os
import sys
import argparse

import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model_nafnet import NAFNetLiteAdaptive
from src.discriminator import PatchDiscriminator
from src.advanced_loss import VGGPerceptualLoss, GramMatrixLoss
from src.dataset_augmented_v2 import RestorationDatasetAugmentedV2
from src.losses_v3 import combined_loss_v3, combined_generator_loss, discriminator_loss


def parse_args():
    p = argparse.ArgumentParser(description="Train NoiseTrue-Adaptive (see module docstring for the 3 stages)")
    p.add_argument("--gt_dir", required=True)
    p.add_argument("--noisy_dir", required=True)
    p.add_argument("--ckpt_dir", required=True, help="Directory for resumable checkpoints")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--disc_lr", type=float, default=1e-4)
    p.add_argument("--base_ch", type=int, default=48)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--synth_prob", type=float, default=0.3)
    p.add_argument("--elastic_prob", type=float, default=0.2)
    p.add_argument("--vgg_weight", type=float, default=0.05)
    p.add_argument("--fft_weight", type=float, default=0.05)
    p.add_argument("--msssim_weight", type=float, default=0.2)
    p.add_argument("--adv_weight", type=float, default=0.015)
    p.add_argument("--gram_weight", type=float, default=0.0,
                    help="Weight for the Gram matrix (texture) loss. 0.0 disables it. "
                         "Stage 3 uses 0.05.")
    p.add_argument("--use_gan", action="store_true",
                    help="Enable the adversarial term (Stage 2).")
    p.add_argument("--gan_start_epoch", type=int, default=20,
                    help="Warm up on reconstruction losses alone before introducing the "
                         "discriminator. Stage 2 uses 150.")
    p.add_argument("--num_workers", type=int, default=2,
                    help="DataLoader worker processes. Set to 0 on machines where "
                         "multiprocessing workers are unavailable/restricted.")
    return p.parse_args()


def save_checkpoint(path, epoch, model, disc, opt_g, opt_d, sched_g, best_val):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "disc": disc.state_dict() if disc is not None else None,
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict() if opt_d is not None else None,
        "sched_g": sched_g.state_dict(),
        "best_val": best_val,
    }, path)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type != "cuda":
        print("WARNING: no GPU detected -- training will be slow. This will still "
              "run correctly on CPU, just proportionally longer per epoch.")

    torch.manual_seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    latest_path = os.path.join(args.ckpt_dir, "latest.pth")
    best_path = os.path.join(args.ckpt_dir, "best.pth")

    dataset = RestorationDatasetAugmentedV2(
        args.gt_dir, args.noisy_dir, augment=True,
        synth_prob=args.synth_prob, elastic_prob=args.elastic_prob,
    )
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                             num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    print(f"Train pairs: {len(train_ds)} | Val pairs: {len(val_ds)}")

    model = NAFNetLiteAdaptive(base_ch=args.base_ch).to(device)
    print(f"Generator parameters: {sum(p.numel() for p in model.parameters()):,}")

    disc = PatchDiscriminator().to(device) if args.use_gan else None

    vgg_loss_fn = VGGPerceptualLoss().to(device)
    # Gram matrix loss reuses the SAME vgg module as VGGPerceptualLoss --
    # no double-loading of VGG weights. Only instantiated if gram_weight > 0.
    gram_loss_fn = GramMatrixLoss(vgg_loss_fn.vgg).to(device) if args.gram_weight > 0 else None
    if gram_loss_fn is not None:
        print(f"Gram matrix loss enabled, weight={args.gram_weight}")

    opt_g = torch.optim.Adam(model.parameters(), lr=args.lr)
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.disc_lr, betas=(0.5, 0.999)) if disc else None
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=args.epochs)

    start_epoch = 0
    best_val = float("inf")

    # --- resume from the last checkpoint if one exists ---
    if os.path.exists(latest_path):
        print(f"Resuming from {latest_path}")
        ckpt = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt_g.load_state_dict(ckpt["opt_g"])
        sched_g.load_state_dict(ckpt["sched_g"])
        best_val = ckpt["best_val"]
        start_epoch = ckpt["epoch"] + 1
        if disc is not None and ckpt.get("disc") is not None:
            disc.load_state_dict(ckpt["disc"])
            opt_d.load_state_dict(ckpt["opt_d"])
        print(f"Resumed at epoch {start_epoch}, best_val so far: {best_val:.5f}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        dataset.augment = True
        gan_active = args.use_gan and epoch >= args.gan_start_epoch
        if disc is not None:
            disc.train()

        total_g, total_d = 0.0, 0.0
        for noisy, gt in train_loader:
            noisy, gt = noisy.to(device), gt.to(device)

            pred = model(noisy)

            if gan_active:
                # --- discriminator step ---
                opt_d.zero_grad()
                disc_real = disc(gt)
                disc_fake = disc(pred.detach())
                d_loss = discriminator_loss(disc_real, disc_fake)
                d_loss.backward()
                opt_d.step()
                total_d += d_loss.item()

                # --- generator step, now including the adversarial term ---
                opt_g.zero_grad()
                disc_fake_for_g = disc(pred)
                g_loss = combined_generator_loss(
                    pred, gt, vgg_loss_fn, disc_fake_for_g, gram_loss_fn,
                    vgg_weight=args.vgg_weight, fft_weight=args.fft_weight,
                    msssim_weight=args.msssim_weight, adv_weight=args.adv_weight,
                    gram_weight=args.gram_weight,
                )
            else:
                # --- reconstruction-only step (warmup, or --use_gan not set) ---
                opt_g.zero_grad()
                g_loss = combined_loss_v3(
                    pred, gt, vgg_loss_fn, gram_loss_fn,
                    vgg_weight=args.vgg_weight, fft_weight=args.fft_weight,
                    msssim_weight=args.msssim_weight, gram_weight=args.gram_weight,
                )

            g_loss.backward()
            opt_g.step()
            total_g += g_loss.item()

        sched_g.step()

        model.eval()
        dataset.augment = False
        val = 0.0
        with torch.no_grad():
            for noisy, gt in val_loader:
                noisy, gt = noisy.to(device), gt.to(device)
                pred = model(noisy)
                val += combined_loss_v3(
                    pred, gt, vgg_loss_fn, gram_loss_fn,
                    vgg_weight=args.vgg_weight, fft_weight=args.fft_weight,
                    msssim_weight=args.msssim_weight, gram_weight=args.gram_weight,
                ).item()
        val /= len(val_loader)

        gan_tag = f"d_loss: {total_d/len(train_loader):.5f} " if gan_active else ""
        print(f"Epoch {epoch+1}/{args.epochs} - g_loss: {total_g/len(train_loader):.5f} "
              f"- {gan_tag}- val_loss: {val:.5f} - lr: {sched_g.get_last_lr()[0]:.6f}", flush=True)

        is_best = val < best_val
        if is_best:
            best_val = val

        # save every epoch -- makes interruptions non-fatal, just re-run to resume
        save_checkpoint(latest_path, epoch, model, disc, opt_g, opt_d, sched_g, best_val)
        if is_best:
            torch.save(model.state_dict(), best_path)

    print(f"\nBest val_loss: {best_val:.5f}")
    print(f"Best model weights: {best_path}")
    print(f"Full resumable state: {latest_path}")


if __name__ == "__main__":
    main()