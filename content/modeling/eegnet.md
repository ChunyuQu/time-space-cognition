---
title: 'From a shallow CNN to EEGNet'
date: 2026-10-15
weight: 30
cover: /images/modeling/eegnet-paper-figure1.png
cover_credit: 'Figure 1 from Lawhern et al. (2018), EEGNet'
cover_source: 'https://arxiv.org/abs/1611.08024'
math: true
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

## Mathematical idea: factor a large convolution

Let one EEG trial be \(X\in\mathbb{R}^{C\times T}\), with \(C\) electrodes and \(T\)
time samples. In PyTorch the batch has shape `[B, 1, C, T]`. A temporal kernel
\(h_f\in\mathbb{R}^{K_t}\) produces a feature

$$
z_{f,c,t}=\sum_{\tau=0}^{K_t-1}h_{f,\tau}X_{c,t-\tau}.
$$

The same temporal filter is applied at every electrode. The next kernel spans all electrodes,

$$
s_{f,d,t}=\sum_{c=1}^{C}w_{f,d,c}z_{f,c,t},
$$

so it acts as a frequency-specific spatial filter. This ordering is the central EEGNet assumption:
learn a temporal basis first, then learn how each temporal feature is distributed over the scalp.

A dense convolution from \(F_1\) inputs to \(F_2\) outputs with kernel width \(K\) needs
\(F_1F_2K\) weights. A depthwise convolution plus a \(1\times1\) pointwise convolution needs
approximately \(F_1K+F_1F_2\). The saving is substantial when both channel counts are large.

## Shape walkthrough

For an illustrative trial with `C=22`, `T=1000`, `F1=8`, depth multiplier `D=2`, and
`F2=16`, the canonical flow is:

| stage | operation | output shape |
|---|---|---|
| input | add singleton image channel | `[B, 1, 22, 1000]` |
| temporal filters | `Conv2d(1, 8, (1, Kt))` | `[B, 8, 22, T1]` |
| spatial filters | depthwise `Conv2d(8, 16, (22, 1), groups=8)` | `[B, 16, 1, T1]` |
| temporal summary | average pooling | `[B, 16, 1, T2]` |
| separable block | depthwise temporal + pointwise mixing | `[B, 16, 1, T3]` |
| classifier | flatten + linear layer | `[B, Nclass]` |

The exact values of `T1`, `T2`, and `T3` depend on padding, kernel width, and pooling. For one
dimension, a convolution produces

$$
T_{out}=\left\lfloor\frac{T_{in}+2p-d(K-1)-1}{s}+1\right\rfloor.
$$

Use this equation before choosing the classifier input size; do not discover the size by repeatedly
changing a hard-coded number until the model runs.

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

## Verify the implementation before training

Registering hooks makes the tensor contract visible:

```python
def show_shape(name):
    def hook(module, inputs, output):
        print(f"{name:24s} -> {tuple(output.shape)}")
    return hook

for name, layer in model.named_modules():
    if isinstance(layer, (nn.Conv2d, nn.AvgPool2d, nn.Linear)):
        layer.register_forward_hook(show_shape(name))

x = torch.randn(2, 1, 22, 1000)
logits = model(x)
assert logits.shape == (2, 4)
```

Then check three invariants: the spatial convolution must reduce the electrode dimension to one;
the model must return logits rather than class indices; and a backward pass must give finite,
nonzero gradients to the temporal and spatial kernels.

## What an experiment must report

Report the trial window, sampling rate, filtering, electrode order, normalization scope, split unit,
random seeds, parameter count, and receptive field. Split by session or participant before fitting
normalization statistics. Otherwise a compact architecture can still benefit from leakage and give a
misleadingly strong result.

## The shallow CNN is a debugging example

`CNNEEG.py` defines three 2D convolutions followed by `220864 → 64 → 32 → 4`. In its current state it should not be treated as a working baseline: `forward()` applies `argmax` before `CrossEntropyLoss`, `nn.logSoftmax` is misspelled, and the training loop constructs a new model inside every batch. The correct contract is to return four logits, instantiate the model once, and apply `argmax` only during evaluation.
