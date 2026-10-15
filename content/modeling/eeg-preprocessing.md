---
title: 'EEG preprocessing: from continuous signals to model-ready trials'
date: 2026-10-15
weight: 10
draft: true
summary: 'Windowing, overlap, labels, and spectral features, the decisions that determine what an EEG model is actually allowed to learn.'
---

An EEG classifier never sees “the experiment.” It sees an array created by a chain of choices: channels, sampling rate, epoch boundaries, window length, overlap, normalization, and labels. Those choices define the statistical problem before a model is fitted.

{{< flow steps="CSV recording|loadEEGCSV / cut_data|overlapping windows|EEG_filter or handcrafted features|PyTorch tensor" caption="The implemented preprocessing path: continuous tables become windows, then either raw tensors or explicit features." >}}

## The basic data flow

```text
continuous recording → event-aligned epochs → sliding windows → features or tensors → train/test split
```

A window of length \(L\) moved by a step \(S\) converts each trial into several examples. Overlap increases the number of examples, but neighboring windows are strongly dependent. They must remain in the same data split; otherwise almost identical signal segments can appear in both training and testing data.

Two useful representations are retained. Raw windows preserve temporal structure for CNNs and temporal models. Power spectral density compresses each channel into interpretable frequency-band features and works well with linear models or graph networks.

## What to check

- Infer labels from experimental metadata, not fragile folder names.
- Split by trial, session, or participant before creating overlapping windows.
- Fit normalization only on training data.
- Record channel order, sampling rate, filter settings, and rejected trials.
- Compare every learned model with a simple spectral baseline.

Preprocessing is therefore part of the model, not housekeeping around it. A sophisticated decoder cannot rescue a target that leaked into the features or a validation set that contains copies of its training windows.

## Reading the implementation

`Cut_Processing_Data.py` contains the short segmentation path. `get_filelist()` recursively collects CSV files, `cut_data()` moves a fixed window through a recording, and `cut_data_label()` assigns left/right labels from the parent directory. The more extensive `Get_EEG_Feature.py` exposes the same logic through `loadEEGCSV()`:

```python
FEATURE_NUM = int(window * sample_rate)
FRAME_NUM = int(frame * sample_rate)
X = np.zeros((int(data.shape[0] / FRAME_NUM), FEATURE_NUM, channels))
for i in range(0, data.shape[0] - FEATURE_NUM, FRAME_NUM):
    X[j, :, :] = data[i:i + FEATURE_NUM, :]
```

For `window=1.0`, `frame=0.5`, and `sample_rate=256`, each example contains 256 samples and adjacent examples overlap by 128 samples. The intended output is `[window, time, channel]`.

`EEG_filter()` then calls `neurokit2.signal_filter` for delta (1-3 Hz), theta (4-7), alpha (8-13), beta (14-30), and gamma (31 Hz upward). The file also implements descriptive statistics, entropy, Hjorth parameters, FFT summaries, wavelet features, zero crossings, and absolute or relative band power.

## Code-level cautions

The allocation in `loadEEGCSV()` uses `data.shape[0] / FRAME_NUM`, but the loop stops at `data.shape[0] - FEATURE_NUM`; this can leave unused zero-filled rows. Allocate from the actual number of loop starts instead. Also, `if len(X.shape) == 1 or 2` is always truthy in Python; write `if X.ndim in (1, 2)`. Finally, create participant/session splits before windowing so overlapping segments cannot cross the split.
