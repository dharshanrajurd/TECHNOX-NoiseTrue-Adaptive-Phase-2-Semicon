# TECHNOX — NoiseTrue-Adaptive

**SEMICON India Hackathon 2026 — KLA Problem Statement**
*AI-Based Restoration of Degraded Images for Semiconductor Inspection*

Team TECHNOX | Vellore Institute of Technology

---

## Problem

Given a noisy, low-resolution SEM image (128×128), restore it to a clean,
full-resolution image (256×256). This is a combined denoising +
2× super-resolution task on scanning electron microscopy images of
nanoscale structures, degraded by speckle noise, Gaussian noise, and
spatial downsampling.

## Our Approach — Degradation-Aware FiLM Conditioning

Most restoration networks apply one fixed processing strategy to every
image, regardless of how badly degraded it is. **Ours estimates
degradation severity per-image and adapts its internal processing
accordingly.**

Two components inside `NAFNetLiteAdaptive`:

- **`DegradationEstimator`** — a small CNN that reads the noisy input and
  produces a 32-dimensional conditioning vector representing how degraded
  that specific image is.
- **`FiLM` (Feature-wise Linear Modulation)** — converts that vector into
  a per-channel scale and shift applied to the network's bottleneck
  features: `modulated = feature * (1 + scale) + shift`. Zero-initialized,
  so training starts equivalent to the plain baseline and only deviates if
  it measurably helps.

Backbone: **NAFNet-lite** (encoder–bottleneck–decoder with NAFBlocks —
LayerNorm + depthwise convolution + simplified channel attention, using
SimpleGate instead of conventional activations), `base_ch=48`, with a
PixelShuffle 2× upsampling head.

### Validation of the novelty

We didn't just assert that the mechanism works — we tested it. Full
methodology and results in [`docs/degradation_film_full_results.md`](docs/degradation_film_full_results.md).
Summary:

- **FiLM is genuinely active**, not stuck at its zero-initialization
  (learned weight norms: scale 15.64, shift 2.89).
- **Its response correlates with objectively measured degradation
  severity** across a 45-image sample spanning the full noise range
  (r = 0.576), and a 2,000-iteration permutation test confirms this is
  **statistically non-random (p < 0.0001)**.
- **Direct ablation testing** (manually overriding FiLM's scale) shows
  the mechanism performs necessary protective work under heavy
  degradation — disabling it produces visible grid artifacts on
  high-noise inputs.

## Loss Function

Our final model is trained with a composite loss targeting all three
scored metrics:

| Term | Purpose |
|---|---|
| Charbonnier | pixel-level fidelity (targets PSNR) |
| Sobel edge | structural/edge preservation |
| VGG16 perceptual | perceptual feature similarity (targets LPIPS) |
| FFT magnitude | high-frequency content preservation |
| MS-SSIM | structural similarity (targets SSIM directly) |
| **Gram matrix** | **texture statistics — counters the over-smoothing that position-based losses reward** |

The Gram matrix term was our final and most effective addition: unlike
the other perceptual terms, it matches *texture statistics regardless of
exact spatial position*, which directly counteracts the tendency of
averaging losses to blur away fine detail the model is uncertain about.

---

## Repository Structure

```
├── evaluate.py              # REQUIRED: standalone inference script for benchmarking
├── train.py                 # REQUIRED: reproduces training of the submitted model
├── requirements.txt         # REQUIRED: pip freeze environment specification
├── src/                     # core model, loss, and dataset code
├── weights/                 # final trained model checkpoint
├── outputs/                 # REQUIRED: denoised test set outputs
├── experiments/             # alternative approaches tested and rejected (with evidence)
└── docs/                    # problem statement + novelty validation writeup
```

---

## Usage

### Running inference (evaluation script)

```bash
python evaluate.py \
    --input_dir /path/to/test/noisy/images \
    --output_dir /path/to/write/outputs \
    --model_path weights/nafnet+gram_best.pth
```

Accepts `.npy` input images, writes restored `.npy` outputs to the
specified directory. Prints end-to-end timing breakdown (model load,
inference, per-image average).

### Reproducing training

See [`train.py`](train.py) — documents all three training stages in order.

---

## Experiments — what we tested and rejected

We validated our architecture choice empirically rather than assuming it.
All alternatives were tested on 40 random images across PSNR, SSIM, and
LPIPS. Code and results in [`experiments/`](experiments/).

| Approach | Result | Decision |
|---|---|---|
| **RRDB (ESRGAN-style, 8.9M params)** | Lost PSNR (22.85 vs 23.65 dB) and SSIM (0.543 vs 0.600) on **0/40 wins**, won LPIPS 35/40 | Rejected — confirmed our smaller model isn't capacity-limited |
| **Cascaded refinement head** (frozen backbone + trained detail head) | Statistically and visually indistinguishable from baseline; slight consistent LPIPS regression | Rejected — no measurable benefit |
| **Global residual skip** (borrowed from RRDB) | Near-identical scores (PSNR 23.66 vs 23.70), no detail-preservation gain | Rejected — the benefit didn't transfer |
| **Gram matrix loss fine-tune** | **Won 30/30 images** vs the previous best | **Adopted — this is our final model** |

These negative results are included deliberately: they document that our
final architecture was chosen on evidence, not convenience.

---

## Final Model

`weights/nafnet+gram_best.pth` — NAFNetLiteAdaptive (`base_ch=48`,
2,731,921 parameters), trained in three stages (see `train.py`), with
degradation-aware FiLM conditioning and Gram matrix texture loss.

Deliberately lightweight: at 2.7M parameters with early spatial
downsampling, the model is fast at inference — relevant given that
end-to-end inference time on an H100 is an explicitly scored dimension
of this competition.
