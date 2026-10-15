---
title: 'EEGKANet: learning nonlinearities with splines'
date: 2026-10-15
weight: 90
draft: true
summary: 'Replacing a fixed-activation classifier with KAN layers turns each connection into a learnable smooth function.'
---

Most neural networks learn connection weights while keeping the activation function fixed. A Kolmogorov-Arnold Network (KAN) changes that balance: each connection can learn a smooth univariate function, commonly represented with B-splines.

{{< flow steps="EEG [B,1,C,T]|EEGNet-style conv backbone|flattened feature vector|base SiLU linear path|B-spline basis path|sum paths|class logits" caption="KANLinear adds a learned spline contribution to a conventional activated linear projection." >}}

EEGKANet uses an EEGNet-style convolutional front end to extract spatial-temporal features, then replaces ordinary fully connected layers with `KANLinear` layers. Grid size controls the number of spline intervals, spline order controls smoothness, and adaptive grid updates place knots where the observed features need resolution.

## Why this may help

EEG decision boundaries may depend on nonlinear feature transformations that a fixed ReLU or ELU represents inefficiently. Learned splines can express those transformations directly and can be plotted, offering a form of functional inspection.

The flexibility also creates risks. With small EEG datasets, high-order splines and dense grids can overfit. Comparisons should match parameter count, use participant-level validation, and report variability across seeds. Plotting the learned functions is useful only after checking that they remain stable across folds.

KAN layers are therefore best treated as a testable alternative classifier head, not as a guaranteed replacement for conventional multilayer perceptrons. Their value lies in whether the learned nonlinearities are both generalizable and scientifically interpretable.

## Inside `KANLinear`

The layer registers a knot grid per input feature. It learns `base_weight[out, in]` and `spline_weight[out, in, basis]`, plus an optional scaler. `b_splines()` builds basis values recursively; `curve2coeff()` uses least squares to convert sampled curves into spline coefficients.

The forward pass has two parallel terms:

```python
base_output = F.linear(self.base_activation(x), self.base_weight)
spline_output = F.linear(
    self.b_splines(x).view(x.size(0), -1),
    self.scaled_spline_weight.view(self.out_features, -1),
)
return base_output + spline_output
```

`update_grid()` sorts observed activations, mixes adaptive quantile knots with a uniform grid using `grid_eps`, then refits coefficients so the represented functions change as little as possible. `regularization_loss()` penalizes spline magnitude and the entropy of connection-wise activation strength.

## How EEGKANet is assembled

`EEGKanNet` loads `kan_config.yaml`. `parse_model()` builds convolution, depthwise, separable, pooling, flattening, and `KANLinear` modules. The current `forward()` prints every layer's input and output shape, which is useful for construction debugging but should become an optional hook for normal training.

The `updata_grid` argument is currently unused because the update-aware forward path is commented out. If adaptive knots are desired, call `layer.update_grid(x)` only on KAN layers and on a controlled schedule; updating every mini-batch can make optimization unstable. The high defaults (`grid_size=9`, `spline_order=7`) are flexible, so parameter-matched MLP baselines and participant-held-out validation are essential.
