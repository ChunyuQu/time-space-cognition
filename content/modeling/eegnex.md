---
title: 'EEGNex: multi-scale temporal context with dilation'
date: 2026-10-15
weight: 40
cover: /images/modeling/eegnex-paper-figure4.jpg
cover_credit: 'Figure 4 from Chen et al., EEGNeX'
cover_source: 'https://arxiv.org/abs/2207.12369'
math: true
summary: 'Dilated depthwise convolutions extend EEGNet’s temporal field without a proportional increase in parameters.'
---

EEGNex keeps the compact spatial-temporal logic of EEGNet but replaces parts of the backbone with dilated depthwise convolutions. A dilated kernel samples the input with gaps, so its receptive field grows without requiring a much wider kernel.

{{< flow steps="EEG [B,1,C,T]|standard temporal conv|dilated conv d=2|dilated conv d=4|depthwise / separable blocks|classifier" caption="EEGNex expands temporal context with two explicit dilation scales." >}}

Using more than one dilation scale allows the network to combine local waveform detail with slower temporal structure. This is useful in motor imagery, where discriminative information can include both transient changes and sustained modulation in sensorimotor rhythms.

## Mathematical idea: receptive field without a wide kernel

A dilated one-dimensional convolution with dilation \(d\) is

$$
y_t=\sum_{j=0}^{K-1}w_jx_{t-dj}.
$$

The kernel still contains \(K\) learned values, but it covers

$$
K_{eff}=1+(K-1)d
$$

samples. With `K=16`, dilation 2 covers 31 samples and dilation 4 covers 61. At 250 Hz these
correspond to 124 ms and 244 ms. Converting samples to milliseconds makes it easier to decide
whether the receptive field matches the neural phenomenon rather than merely looking large in code.

For stacked layers with stride one, the total receptive field is

$$
R=1+\sum_{\ell}(K_\ell-1)d_\ell.
$$

If pooling or strided convolutions intervene, each later contribution must also be multiplied by the
accumulated stride of the earlier layers.

## From EEGNet to EEGNeX

The original paper's Figure 4 separates five convolutional units. The useful conceptual progression is:

| stage | role | question to check |
|---|---|---|
| temporal units 1-2 | build a richer temporal filter bank | what frequency response do the kernels learn? |
| spatial depthwise unit | combine electrodes within each temporal feature | does the kernel span exactly `C` electrodes? |
| dilated units 4-5 | expand temporal context | what is the receptive field in milliseconds? |
| pooling and classifier | compress and classify | is temporal resolution discarded too early? |

This is more precise than saying that dilation simply gives the model “long memory.” Dilation changes
which samples can interact; it does not guarantee that the learned representation uses physiologically
meaningful time scales.

## Transfer across participants

The architecture also supports a simple transfer-learning strategy: train the full model on source participants, freeze early convolutional layers, and adapt later layers to a new participant. The assumption is that early temporal filters capture reusable EEG structure while the final representation is more participant-specific.

That assumption should be tested rather than taken for granted. Compare full fine-tuning, frozen-feature transfer, and training from scratch using participant-held-out evaluation. Dilation adds a useful inductive bias, but it does not by itself solve differences in electrode contact, anatomy, strategy, or recording session.

## Reading `EEGNex_Module.py`

The file defines `dilation_conv1` and `dilation_conv2` as ordinary `Conv2d` layers with `dilation=2` and `dilation=4`. Kernel, stride, padding, bias, and channel counts come from `EEGNex_config.yaml`. The effective one-dimensional kernel width is `1 + (k - 1) * dilation`, so a width-16 kernel covers 31 samples at dilation 2 and 61 samples at dilation 4.

`parse_model()` follows the same YAML-to-`nn.Sequential` mechanism as EEGNet. It tracks the previous output width in `ch`, resolves symbolic widths with `choose_out_ch()`, and instantiates dilation layers alongside standard, depthwise, and separable convolutions.

```python
if m in [conv, DepthConv, SeparableConv, FC,
         dilation_conv2, dilation_conv1]:
    in_channels = ch[f] if f == -1 else f
    out_channels = choose_out_ch(args[0], yaml_cfg['params'])
    param = [in_channels, out_channels, args[1:]]
```

Two implementation details matter. First, sufficient padding must be chosen for each dilation or the time axis shrinks unexpectedly. Second, the custom `DepthConv` again concatenates repeated calls to the same grouped convolution; it does not learn independent depth-multiplier filters. The tutorial should therefore log every intermediate shape and compare parameter counts before interpreting the architecture as canonical EEGNeX.

## A minimal receptive-field and shape audit

```python
def effective_kernel(kernel, dilation):
    return 1 + (kernel - 1) * dilation

for kernel, dilation in [(16, 2), (16, 4)]:
    samples = effective_kernel(kernel, dilation)
    print(samples, "samples", samples / 250 * 1000, "ms")

x = torch.randn(2, 1, 22, 1000)
with torch.no_grad():
    logits = model(x)
assert logits.ndim == 2 and logits.shape[0] == 2
```

Also inspect the kernel spectra with an FFT, compare an equal-parameter non-dilated model, and run an
ablation that removes each dilation stage. If dilation helps only when parameter count also increases,
the experiment has not isolated the proposed mechanism.

## Evaluation across participants

Transfer learning requires a nested protocol. Choose hyperparameters using source participants and a
validation portion of the target participant; evaluate once on untouched target trials. Report training
from scratch, frozen features, partial fine-tuning, and full fine-tuning with identical trial budgets.
Trial-wise random splitting is not evidence of participant transfer because adjacent trials share session,
hardware, and noise structure.
