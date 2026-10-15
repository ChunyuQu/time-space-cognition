---
title: 'Graph convolution for electrode relationships'
date: 2026-10-15
weight: 60
summary: 'Representing electrodes as nodes makes spatial assumptions explicit—and exposes how much depends on the graph itself.'
---

A standard CNN assumes a regular grid. EEG electrodes instead lie on an irregular surface with meaningful neighborhood relations. A graph convolutional network represents each electrode as a node, its signal-derived features as node attributes, and relationships between electrodes as weighted edges.

{{< flow steps="EEG trial|PSD / band-power per electrode|32 electrode nodes|Gaussian-similarity edges|GCN 32→64→128→256|global add pool|2-class logits" caption="Each trial becomes a weighted electrode graph before graph convolution and trial-level pooling." >}}

One practical pipeline computes power in delta, theta, alpha, beta, and gamma bands for every electrode. For each trial, pairs of electrodes receive Gaussian-kernel weights based on feature similarity. Stacked graph convolutions exchange information between connected electrodes, and global pooling produces a trial-level representation for classification.

## The graph is a hypothesis

Performance depends as much on graph construction as on the neural network. A similarity graph may capture functional coupling, but it can also vary with noise and inadvertently encode condition-specific information. A geometry-based graph is more stable but assumes that physical proximity defines relevant interaction.

Useful controls include:

- fixed anatomical versus trial-specific functional edges;
- weighted versus binary adjacency;
- random or degree-matched graphs;
- graph models versus the same classifier on concatenated spectral features.

Graph convolution is most informative when the edge rule is interpretable and survives these controls. Otherwise the graph can become an expensive way to hide an arbitrary preprocessing choice.

## How graphs are constructed

`get_brain_waves_power()` collapses Welch PSD estimates into delta, theta, alpha, lower-beta, higher-beta, and gamma power. `MyOwnDataset.process()` then assumes 32 nodes and compares every electrode pair. For feature vectors `x_i` and `x_j`, it assigns a Gaussian similarity when their Euclidean distance is at most `t=15`; `v=6` controls kernel width.

```python
similarity = torch.exp(-torch.norm(x[i] - x[j])**2 / (2 * v**2))
weight = similarity.item() if torch.norm(x[i] - x[j]) <= t else 0
edge_index.append([i, j])
edge_weight.append(weight)
```

The graph is stored as a PyTorch Geometric `Data` object and cached by `InMemoryDataset`. `EEGGraphConvNet` intends three `GCNConv` stages (`32→64→128→256`), batch normalization, `global_add_pool`, dropout, and a `256→128→2` classifier.

## Current implementation issues

The code only adds `[i, j]`, not the reverse edge `[j, i]`; make the graph explicitly undirected. Zero-weight edges are still stored and should be removed. Device placement is hard-coded to `cuda:0` during preprocessing, which prevents CPU use and makes cached data device-dependent.

The current `forward()` also nests `self.conv2(self.conv2(...))` and calls `self.conv3(x)` without `edge_index`; `global_add_pool(x)` is missing the batch vector. The intended form is `conv1(x, edge_index, edge_weight)`, then `conv2`, then `conv3`, followed by `global_add_pool(x, batch)`. These corrections are prerequisites before reporting model performance.
