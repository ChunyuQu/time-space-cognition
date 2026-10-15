---
title: 'CCA and FBCCA: transparent decoding for SSVEP'
date: 2026-10-15
weight: 20
summary: 'A training-light SSVEP decoder built by matching multichannel EEG to harmonic reference signals.'
---

Steady-state visual evoked potentials provide a rare case in which the stimulus itself supplies a strong model of the neural response. If a target flickers at frequency \(f\), the EEG should contain activity at \(f\) and its harmonics.

{{< flow steps="EEG window [channels × time]|reference sin/cos at each target|CCA per target|largest canonical correlation|SSVEP class" caption="CCA scores each candidate stimulus frequency against the same multichannel EEG window." >}}

Canonical correlation analysis (CCA) compares multichannel EEG `X` with sine-and-cosine reference signals `Y_f`. It finds projections `w_x` and `w_y` that maximize `corr(Xw_x, Y_fw_y)`.

The predicted target is the frequency with the largest canonical correlation. This makes CCA fast, interpretable, and usable without subject-specific training data.

## Why a filter bank helps

Filter-bank CCA (FBCCA) repeats the comparison in multiple frequency bands. It then combines the band-specific correlations with weights that emphasize informative harmonics. This is especially useful for short decoding windows, where a single broadband estimate is noisy.

Template CCA replaces synthetic sinusoids with averaged training trials. It can capture participant-specific phase and waveform structure, at the cost of calibration data.

## Practical trade-off

CCA is an excellent baseline for an online SSVEP system because every prediction can be traced to a frequency match. Performance depends on window duration, number of harmonics, selected channels, and latency alignment. Before adding a neural network, CCA and FBCCA show how much of the task is already solved by known stimulus structure.

## Reading `CCA.py`

`CCA_Base.__init__()` reads `Fs`, window length, number of frequencies, channels, and trials from `opt`. `get_Reference_Signal()` creates two rows per harmonic:

```python
reference_signals = []
for j in range(1, num_harmonics + 1):
    reference_signals.append(np.sin(2 * np.pi * j * target * t))
    reference_signals.append(np.cos(2 * np.pi * j * target * t))
```

`find_correlation()` fits `sklearn.cross_decomposition.CCA` to the transposed EEG and reference matrices, transforms both, and stores the correlation between their first canonical components. `cca_classify()` repeats that score for every candidate frequency and returns `argmax`.

When `template=True`, `get_Template_Signal()` averages training trials for each target and uses those empirical waveforms instead of sinusoids. This adds subject-specific phase and morphology.

## Reading `FBCCA.py`

`filter_bank()` designs Chebyshev type-I band-pass filters whose lower cut-off increases across sub-bands while the upper cut-off remains near 90 Hz. `fbcca_classify()` runs CCA inside every sub-band and combines squared correlations with weights of the form `m**(-1.25) + 0.25`.

The important tensor contract is `[target/trial, channel, sample]`; CCA receives `[sample, channel]` after transpose. Before online use, replace hard-coded filter limits with values checked against `Fs`, and verify that reference duration exactly matches the sliced EEG window.
