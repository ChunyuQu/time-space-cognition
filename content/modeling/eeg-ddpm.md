---
title: 'Diffusion models for EEG augmentation'
date: 2026-10-15
weight: 70
draft: true
summary: 'A DDPM learns to reverse progressive noise, but useful synthetic EEG requires more than visually plausible samples.'
---

A denoising diffusion probabilistic model (DDPM) gradually corrupts an EEG trial with Gaussian noise and trains a neural network to predict the added noise. Sampling starts from noise and repeatedly applies the learned reverse process until a synthetic trial emerges.

{{< flow steps="clean EEG x0|sample time t + noise|closed-form q(xt|x0)|time-conditioned U-Net|predict noise ε|reverse steps to synthetic EEG" caption="Training samples one diffusion time; generation iterates the learned reverse transition from noise." >}}

The implementation uses a U-Net with residual blocks, attention, skip connections, and a sinusoidal embedding of diffusion time. Different noise schedules determine how quickly information disappears during the forward process.

## Augmentation is the real test

For EEG, the goal is usually not generation for its own sake. Synthetic trials are intended to reduce class imbalance or improve decoding with limited data. That claim should be tested downstream:

1. Train a classifier on real data only.
2. Add synthetic data generated using training participants only.
3. Evaluate on untouched real participants or sessions.
4. Compare against simpler augmentation such as noise, cropping, or channel dropout.

Spectra, covariance, event-related time courses, and spatial topographies can diagnose obvious failures, but resemblance is not sufficient. A generator may reproduce smooth-looking EEG while collapsing participant diversity or copying training examples. Privacy and nearest-neighbor checks therefore belong beside classification accuracy.

## Reading `DDPM.py`

`SinusoidalPositionEmbeddings` converts integer diffusion times into sine/cosine vectors. The U-Net maps them through an MLP and injects the result into every ConvNeXt or ResNet block. Each down stage contains two conditioned blocks, residual linear attention, and downsampling; the bottleneck uses full attention; each up stage concatenates a saved skip feature and upsamples.

The file provides cosine, linear, quadratic, and sigmoid beta schedules. With `timesteps=200`, it precomputes cumulative alphas and posterior variance. `extract()` gathers the coefficient for each example's sampled time and reshapes it for broadcasting across an EEG image.

`DDPM_train.py` implements the forward sample:

```python
noise = torch.randn_like(x_start)
sqrt_a = extract(sqrt_alphas_cumprod, t, x_start.shape)
sqrt_om = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
x_t = sqrt_a * x_start + sqrt_om * noise
```

Training should minimize MSE between sampled `noise` and `model(x_t, t)`. Reverse sampling then uses the predicted noise and posterior variance from `t=199` down to zero.

## Current implementation issues

In `ConvNextBlock.forward()`, `self.net(h)` is called but its result is discarded; use `h = self.net(h)`. The full-attention block uses `np.einsum` on Torch tensors and then calls Torch methods on the result; both contractions should be `torch.einsum`. Finally, the provided training file stops after forward-noise visualization,it does not yet contain a complete optimizer loop or reverse sampler. The architecture is a scaffold until those pieces are added and tested on EEG-shaped tensors.
