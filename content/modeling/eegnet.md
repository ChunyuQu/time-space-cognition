---
title: 'From a shallow CNN to EEGNet'
date: 2026-10-15
weight: 30
summary: 'How temporal, spatial, depthwise, and separable convolutions create a compact baseline for motor-imagery EEG.'
---

A shallow convolutional network treats an EEG trial as an image with channels on one axis and time on the other. It is a useful sanity check, but an unconstrained 2D kernel does not express the structure of EEG particularly well.

{{< flow steps="EEG [B,1,C,T]|temporal Conv2d|depthwise spatial Conv2d|pool + dropout|separable Conv2d|flatten + classifier" caption="EEGNet factorizes temporal filtering, electrode mixing, and feature recombination." >}}

EEGNet introduces a more deliberate factorization:

1. A temporal convolution learns frequency-selective filters.
2. A depthwise spatial convolution learns a channel projection for each temporal filter.
3. A separable convolution recombines the resulting features efficiently.
4. Pooling and a small classifier produce the class probabilities.

This separation mirrors a familiar signal-processing pipeline: first ask *when or at what frequency* a pattern occurs, then ask *where across electrodes* it is expressed. Depthwise and separable operations keep the parameter count low, which matters when a participant contributes only a few hundred trials.

## Why it is a strong baseline

EEGNet is compact enough to train on BCI Competition IV 2A while still learning temporal and spatial filters jointly. It also provides a common front end for more elaborate models in this collection.

The essential comparisons are not only accuracy. Check cross-session and cross-subject generalization, variability across random seeds, calibration, and whether learned spatial patterns are physiologically plausible. A larger network that wins only on a trial-wise random split may simply be better at identifying the recording session.

## How the repository builds the model

`EEGNet_Module.py` does not hard-code one `nn.Sequential`. `EEGNet.__init__()` loads `EEGNet_config.yaml`, deep-copies it, and passes it to `parse_model()`. Each YAML row has a source-channel specification, a module name, and arguments. `eval(Module_name)` resolves wrappers such as `conv`, `DepthConv`, `SeparableConv`, `Pool`, and `FC`; `choose_out_ch()` resolves symbolic widths such as `F1`, `F2`, `C`, and `num_class`.

The key implementation is `SeparableConv`: a grouped temporal convolution keeps channels independent, then a `1×1` pointwise convolution mixes them.

```python
nn.Conv2d(in_channels, in_channels, kernel_size=k,
          groups=in_channels, padding=p),
nn.Conv2d(in_channels, out_channels, kernel_size=1),
nn.BatchNorm2d(out_channels)
```

The custom `DepthConv` similarly uses `groups=in_channel`, but repeats the same convolution output with `torch.concat` when `out_channel > in_channel`. That duplicates feature maps rather than learning a depth multiplier; a production implementation should use `out_channels=in_channel * D` directly in `nn.Conv2d`.

## The shallow CNN is a debugging example

`CNNEEG.py` defines three 2D convolutions followed by `220864 → 64 → 32 → 4`. In its current state it should not be treated as a working baseline: `forward()` applies `argmax` before `CrossEntropyLoss`, `nn.logSoftmax` is misspelled, and the training loop constructs a new model inside every batch. The correct contract is to return four logits, instantiate the model once, and apply `argmax` only during evaluation.
