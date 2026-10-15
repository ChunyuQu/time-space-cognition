---
title: 'EEG-TCNet: causal temporal memory without recurrence'
date: 2026-10-15
weight: 50
cover: /images/modeling/eeg-tcnet-paper-architecture.svg
cover_credit: 'Architecture figure from Ingolfsson et al., EEG-TCNet'
cover_source: 'https://arxiv.org/abs/2006.00622'
math: true
summary: 'An EEGNet-style front end followed by a dilated temporal convolutional network for long-range dependencies.'
---

EEG-TCNet separates feature extraction from temporal integration. An EEGNet-like front end first learns compact spatial and spectral features. A temporal convolutional network (TCN) then processes those features with causal, dilated convolutions and residual connections.

{{< flow steps="EEG [B,1,C,T]|EEGNet spatial-temporal front end|squeeze electrode axis|TCN d=1|TCN d=2|TCN d=4|classifier" caption="The front end extracts compact features; the TCN integrates them over exponentially larger temporal ranges." >}}

For layer \(i\), a dilation of \(2^i\) expands the receptive field exponentially. The network can therefore integrate information across a long interval without recurrent state and can compute time points in parallel during training.

## Mathematical idea: causal temporal convolution

For feature sequence \(x_t\), a causal dilated convolution is

$$
y_t=\sum_{j=0}^{K-1}w_jx_{t-dj}.
$$

Every index is at or before \(t\), so the output cannot use future samples. Padding by
\((K-1)d\) preserves sequence length, but ordinary symmetric padding also creates values on the
right. `Chomp1d` removes those rightmost values and restores the causal alignment.

Each EEG-TCNet residual block contains two convolutions at the same dilation. For blocks with
dilations \(1,2,4,\ldots,2^{L-1}\) and kernel width \(K\), the TCN receptive field is

$$
R=1+2(K-1)\sum_{i=0}^{L-1}2^i
 =1+2(K-1)(2^L-1).
$$

For `K=4` and `L=3`, \(R=43\) feature steps. This must be converted back to EEG samples using
the pooling factor of the EEGNet front end. If the front end reduces time by a factor of 32, those
43 steps span up to 1376 input samples. Reporting only “three TCN blocks” hides this important scale.

## Tensor contract between the two networks

The architectural hand-off is easy to implement incorrectly:

| stage | expected shape |
|---|---|
| raw batch | `[B, 1, C, T]` |
| EEGNet feature map | `[B, F2, 1, Tred]` |
| remove electrode axis | `[B, F2, Tred]` |
| TCN blocks | `[B, Ft, Tred]` |
| select/aggregate time | `[B, Ft]` |
| classifier | `[B, Nclass]` |

Use `squeeze(2)`, not an unrestricted `squeeze()`: when `B=1`, the latter can accidentally remove
the batch dimension as well.

## What the TCN contributes

- Causal convolutions prevent a representation at time \(t\) from using later samples.
- Residual blocks stabilize a deeper temporal model.
- Increasing dilation combines short- and long-range dependencies.
- Convolutional execution is often simpler and faster than an RNN.

The main design question is whether the receptive field matches the physiological time scale of the task. Too short, and the model misses sustained motor-imagery dynamics; too long, and padding or irrelevant context can dominate. An ablation against the same front end without the TCN is necessary to show that temporal memory, not merely extra parameters, produces the gain.

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

## Test causality rather than assuming it

Two inputs that are identical up to time \(t\) must produce identical outputs up to \(t\):

```python
model.eval()
x1 = torch.randn(1, n_features, 64)
x2 = x1.clone()
x2[:, :, 40:] = torch.randn_like(x2[:, :, 40:])

with torch.no_grad():
    y1, y2 = model(x1), model(x2)

torch.testing.assert_close(y1[:, :, :40], y2[:, :, :40])
```

Run this test on the TCN module before attaching the final classifier. A model can have the right
padding sizes and still leak future information through an incorrect crop or tensor rearrangement.

## Separate mechanism from capacity

The key ablation is not merely EEGNet versus EEG-TCNet. Compare models with matched parameter
counts: an EEGNet front end alone, a non-causal temporal CNN, a causal non-dilated CNN, and the
causal dilated TCN. Report mean and dispersion across seeds and participants. This separates gains
from causality, dilation, and additional capacity instead of attributing every improvement to
“temporal memory.”
