---
title: 'FB-EEGViT: filter banks, compact convolutions, and attention'
date: 2026-10-15
weight: 80
summary: 'A hybrid model that moves from multi-scale frequency extraction to EEGNet features and transformer-based integration.'
---

FB-EEGViT combines three modeling ideas in sequence. A filter-bank module first applies parallel temporal convolutions with different kernel widths. An EEGNet block then learns compact spatial filters. Finally, EfficientViT blocks use self-attention and feed-forward layers to integrate the resulting representation.

{{< flow steps="EEG|parallel temporal Conv1d filters|concatenate bands|EEGNet Conv2d + pooling|flatten / tokens|4 attention blocks|classification head" caption="The implementation combines multi-scale filtering, compact convolutional features, and transformer blocks." >}}

```text
EEG → multi-scale temporal filters → spatial feature extraction → attention → class
```

Each stage addresses a different uncertainty: which temporal scale matters, which electrode pattern carries it, and which extracted features should interact. The architecture is attractive when relevant rhythms span several bands and their relationships vary across a trial.

## Proving that the hybrid is useful

Hybrid models can improve accuracy simply because they contain more routes and parameters. A convincing evaluation removes one component at a time: no filter bank, no attention, and an EEGNet-only baseline. Parameter-matched controls help separate architectural benefit from model size.

Attention weights are also not automatically explanations. They can be inspected as diagnostics, but claims about neurophysiological importance require perturbation or feature-removal tests. The model is most valuable when its added components yield reproducible cross-participant improvements rather than a single best run.

## Reading `FBEEGViTNet.py`

`FBCSP` creates a `ModuleList` of `Conv1d(in_channels, 1, kernel_size=k)` layers for `k in range(1, num_filters + 1)` and concatenates their outputs. This is a learned multi-kernel filter bank; despite the class name, it is not classical FBCSP because it does not solve a CSP eigenvalue problem.

The local `EEGNet` is a compact stack of two `Conv2d` layers, batch normalization, ELU, max pooling, dropout, flattening, and a linear classifier. `EfficientViTBlock` applies `nn.MultiheadAttention`, adds the residual, then applies a two-layer feed-forward network with another residual.

`FBCSP_EEGNet_EfficientViT.forward()` is intended to pass `x` through all three components:

```python
x = self.fbcsp(x)
x = self.eegnet(x)
x = self.efficient_vit(x)
return x
```

That contract currently needs redesign. `FBCSP` uses `Conv1d`, whereas typical input is documented as `[B,1,22,T]`; EEGNet then expects 4D input but its classifier already reduces features to class logits, leaving no token sequence for attention. `MultiheadAttention` also expects an embedding dimension matching `in_channels` and a clear sequence/batch convention. A working version should define shapes explicitly—for example, filter each electrode as `[B*C,1,T]`, restore `[B,bands,C,T']`, let EEGNet output tokens rather than logits, and only then apply attention and the final classifier.
