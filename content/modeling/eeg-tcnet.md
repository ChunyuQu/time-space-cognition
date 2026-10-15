---
title: 'EEG-TCNet: causal temporal memory without recurrence'
date: 2026-10-15
weight: 50
summary: 'An EEGNet-style front end followed by a dilated temporal convolutional network for long-range dependencies.'
---

EEG-TCNet separates feature extraction from temporal integration. An EEGNet-like front end first learns compact spatial and spectral features. A temporal convolutional network (TCN) then processes those features with causal, dilated convolutions and residual connections.

{{< flow steps="EEG [B,1,C,T]|EEGNet spatial-temporal front end|squeeze electrode axis|TCN d=1|TCN d=2|TCN d=4|classifier" caption="The front end extracts compact features; the TCN integrates them over exponentially larger temporal ranges." >}}

For layer \(i\), a dilation of \(2^i\) expands the receptive field exponentially. The network can therefore integrate information across a long interval without recurrent state and can compute time points in parallel during training.

## What the TCN contributes

- Causal convolutions prevent a representation at time \(t\) from using later samples.
- Residual blocks stabilize a deeper temporal model.
- Increasing dilation combines short- and long-range dependencies.
- Convolutional execution is often simpler and faster than an RNN.

The main design question is whether the receptive field matches the physiological time scale of the task. Too short, and the model misses sustained motor-imagery dynamics; too long, and padding or irrelevant context can dominate. An ablation against the same front end without the TCN is necessary to show that temporal memory—not merely extra parameters—produces the gain.

## Reading the TCN code

`TemporalBlock` contains two weight-normalized `Conv1d` layers. Both use the same dilation, followed by `Chomp1d`, ReLU, and dropout. `Chomp1d` removes right-side padding so output length matches input length and future samples do not leak backward. A `1×1` convolution aligns the residual when channel counts change.

```python
dilation_size = 2 ** i
padding = (kernel_size - 1) * dilation_size
layers += [TemporalBlock(in_channels, out_channels,
                         kernel_size, 1, dilation_size,
                         padding, dropout)]
```

With two convolutions per block and kernel width `k`, each level adds `2 * (k - 1) * dilation` samples to the receptive field. This lets the YAML `num_channels` list determine both depth and temporal reach.

`EEG_TCNet.__init__()` loads `EEG_TCNet.yaml`; `parse_model()` resolves CNN modules until it encounters `TCNNet`, at which point the previous channel count becomes `num_inputs` and the final TCN width becomes the next tracked width.

The file contains two different `TemporalConvNet` definitions, so the later definition shadows the earlier name. That is a maintenance hazard: rename them and test the instantiated class explicitly. Also verify the 4D-to-3D squeeze before `Conv1d`; the TCN expects `[batch, feature, time]`, not `[batch, feature, electrode, time]`.
