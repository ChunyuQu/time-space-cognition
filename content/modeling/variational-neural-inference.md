---
title: 'Variational Neural Inference: an executable tutorial'
date: 2026-09-22
summary: Eleven notebooks that run from tensors and latent variables to sequential VAEs and interpretable nonlinear dynamics, written for neuroscientists who want to fit these models rather than read about them.
---

Most introductions to variational inference either stop at the ELBO or jump
straight to a published model with ten thousand lines of scaffolding. This
tutorial is my attempt at the path between the two: eleven notebooks that build
up from probability and latent variables to sequential VAEs on spike trains,
where every step runs and every model is small enough to read.

It grew out of teaching material for our lab, and it assumes you know Python but
not necessarily PyTorch, and that you care about neural data specifically —
the examples are spikes, behaviour, and latent dynamics rather than MNIST digits.

**The notebooks live on GitHub:
[msenselab/Variational-Neural-Inference](https://github.com/msenselab/Variational-Neural-Inference).**
They are meant to be executed, so this page is a map rather than a copy.

## The progression

```text
probability
    -> latent variables
    -> mixture models and EM
    -> temporal states and dynamic programming
    -> variational inference
    -> recurrent neural dynamics
    -> nonlinear but locally interpretable dynamics
```

Each step adds exactly one idea. Mixture models introduce hidden variables.
HMMs add temporal structure. VAEs add amortized inference. LFADS adds recurrent
latent dynamics for spike trains. gpSLDS adds uncertain nonlinear dynamics
assembled from local linear regimes.

## The notebooks

| # | Notebook | What it covers |
|---|---|---|
| 00 | [PyTorch Primer](https://github.com/msenselab/Variational-Neural-Inference/blob/main/VAE-Tutorial/Part0_Foundations/00_pytorch_basics.ipynb) | Exercise-based PyTorch primer |
| 01 | [PyTorch for Neuroscience](https://github.com/msenselab/Variational-Neural-Inference/blob/main/VAE-Tutorial/Part0_Foundations/01_pytorch_neuroscience_introduction.ipynb) | Visual introduction, seminar prerequisite |
| 02 | [Probabilistic Modeling](https://github.com/msenselab/Variational-Neural-Inference/blob/main/VAE-Tutorial/Part1_Basics/02_probabilistic_modeling.ipynb) | Probability and latent-variable foundations |
| 02b | [Mixture Models and EM](https://github.com/msenselab/Variational-Neural-Inference/blob/main/VAE-Tutorial/Part1_Basics/02b_mixtures_em.ipynb) | Mixtures, EM, the ELBO, stochastic EM |
| 03a | [HMM Foundations](https://github.com/msenselab/Variational-Neural-Inference/blob/main/VAE-Tutorial/Part2_Dynamics/03a_hmm_foundations.ipynb) | Forward-backward, decoding, sampling, Baum-Welch |
| 03 | [From Behaviour to Latent Dynamics](https://github.com/msenselab/Variational-Neural-Inference/blob/main/VAE-Tutorial/Part2_Dynamics/03_hmm_lds.ipynb) | Gaussian and AR-HMM extended workshop |
| 04 | [Standard VAE](https://github.com/msenselab/Variational-Neural-Inference/blob/main/VAE-Tutorial/Part3_Variational/04_standard_vae.ipynb) | The ELBO and amortized inference |
| 05 | [Variational EM](https://github.com/msenselab/Variational-Neural-Inference/blob/main/VAE-Tutorial/Part3_Variational/05_variational_em.ipynb) | CAVI and coordinate-ascent variational EM |
| 06 | [Sequential VAEs (LFADS)](https://github.com/msenselab/Variational-Neural-Inference/blob/main/VAE-Tutorial/Part3_Variational/06_lfads.ipynb) | Transparent PyTorch LFADS for spike trains |
| 07 | [Advanced LFADS Reference](https://github.com/msenselab/Variational-Neural-Inference/blob/main/VAE-Tutorial/Part3_Variational/07_FULL_LFADS_Tutorial.ipynb) | Full JAX workflow, inferring inputs to an integrator RNN |
| 08 | [gpSLDS](https://github.com/msenselab/Variational-Neural-Inference/blob/main/VAE-Tutorial/Part4_Advanced/08_gpslds.ipynb) | Interpretable nonlinear dynamics, conceptual capstone |

## Where to start

If you read four of them, read these: **02** for why latent variables are
useful, **04** for the ELBO and amortized inference, **06** for a temporal VAE
on real spike trains, and **08** for interpretable nonlinear latent dynamics.
Notebooks 00 and 01 are prerequisites rather than lessons. The complete
classical inference track is 02 → 02b → 03a.

Notebook 04 is the one I would hand to someone who has read about VAEs and
still does not feel they could write one.

## Running them

Notebooks 07 and 08 depend on external implementations registered as Git
submodules, so downloading single files is not enough for those two:

```bash
git clone https://github.com/msenselab/Variational-Neural-Inference.git
cd Variational-Neural-Inference
python scripts/setup_all.py
```

The setup script initializes the submodules, applies the JAX and NumPy
compatibility patches, and installs a CPU-compatible environment for all
eleven notebooks. It is idempotent, and `--dry-run` shows what it would do.
For a smaller install, `requirements-core.txt` covers notebooks 00–06.

A note on honesty about what runs: notebook 05 does its full synthetic CAVI
tutorial without any download, and only fetches the Kato dataset if you opt in.
Notebook 06 splits 950 training and 50 genuinely held-out test trials. Notebook
08 demonstrates the gpSLDS computational core rather than reproducing the full
upstream fitting pipeline. Those boundaries are marked in the notebooks
themselves.

## Attribution

The tutorial adapts material from Stanford's
[STATS 320](https://slinderman.github.io/ml4nd/) and from the original model
implementations; `references/ATTRIBUTION.md` in the repository has the details.
The underlying papers are Kingma and Welling (2013) for the VAE, Pandarinath et
al. (2018) for LFADS, Vyas et al. (2020) for computation through dynamics, and
Hu et al. (2024) for gpSLDS.
