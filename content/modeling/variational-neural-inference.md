---
title: 'Variational Neural Inference: an executable tutorial'
date: 2026-09-22
cover: /images/vni/vae-architecture.webp
summary: Eleven notebooks that run from tensors and gradients to sequential VAEs and interpretable nonlinear dynamics — a walkthrough of the models, the figures they produce, and what each one adds.
---

Most introductions to variational inference either stop at the ELBO or jump
straight to a published model with ten thousand lines of scaffolding. This
tutorial is my attempt at the path between the two: eleven notebooks that build
from gradients and latent variables to sequential VAEs on spike trains, where
every step runs and every model is small enough to read.

It grew out of teaching material for our lab. It assumes you know Python but not
necessarily PyTorch, and that you care about neural data specifically — the
examples are spikes, posture, and latent dynamics rather than MNIST digits.

**The notebooks live on GitHub:
[msenselab/Variational-Neural-Inference](https://github.com/msenselab/Variational-Neural-Inference).**
What follows is a walkthrough of where they go and what each stage buys you.
Every figure below is real output from the notebook it names — click any of
them to open the full-size version.

## The progression

```text
gradients
    -> latent variables
    -> mixture models and EM
    -> temporal states and dynamic programming
    -> amortized variational inference
    -> recurrent neural dynamics
    -> nonlinear but locally interpretable dynamics
```

Each step adds exactly one idea, and each notebook is where that idea is made
to work on data.

## 1. Gradients, before anything else

The first two notebooks exist so that nobody has to pretend they already know
PyTorch. Tensors, autograd, a loss surface, and a network small enough that
every weight can be drawn on the page.

{{< fig src="net-architecture"
    alt="Diagram of a small feedforward network: ten input units, a linear layer to 64 hidden units, a ReLU activation, and a linear output, with arrows marking the forward pass and the backward gradient pass."
    caption="A network with nothing hidden: the forward pass computes a prediction, the backward pass computes gradients for every weight."
    nb="01_pytorch_neuroscience_introduction.ipynb"
    href="https://github.com/msenselab/Variational-Neural-Inference/blob/main/VAE-Tutorial/Part0_Foundations/01_pytorch_neuroscience_introduction.ipynb" >}}

By the end of this part you have fitted a GLM to neural responses by gradient
descent. That is the whole toolkit the later models are built from.

## 2. Latent variables, and the algorithm that finds them

A mixture model is the smallest interesting model with something hidden in it:
the data are observed, the cluster label is not. Fitting it by EM is where the
ELBO first appears, and it appears as something you derive rather than something
you are handed.

{{< fig src="em-fit"
    alt="Left: a scatter of two-dimensional data with three fitted Gaussian components drawn as two-standard-deviation ellipses. Right: log marginal likelihood rising steeply over about eight EM iterations and then flattening."
    caption="EM on a three-component Gaussian mixture. The right panel is the guarantee that makes EM worth teaching: the model evidence never goes down."
    nb="02b_mixtures_em.ipynb"
    href="https://github.com/msenselab/Variational-Neural-Inference/blob/main/VAE-Tutorial/Part1_Basics/02b_mixtures_em.ipynb" >}}

The part worth slowing down for is what the E step actually produces. Not a
label per point — a *distribution* over labels, which is soft exactly where the
data are ambiguous.

{{< fig src="em-responsibilities"
    alt="Two scatter plots of the same clustered data. In the left plot point colour is a blend of the three component colours, showing mixed assignments between clusters. In the right plot points are coloured by maximum posterior probability, which is near one inside clusters and drops sharply at the boundaries between them."
    caption="Responsibilities are the posterior over the hidden label. Colour blending (left) shows points that belong partly to two components; posterior confidence (right) collapses precisely at the boundaries. This picture is the one that makes variational inference feel inevitable later."
    nb="02b_mixtures_em.ipynb"
    href="https://github.com/msenselab/Variational-Neural-Inference/blob/main/VAE-Tutorial/Part1_Basics/02b_mixtures_em.ipynb" >}}

The same notebook ends with stochastic EM, which is the first hint that these
algorithms survive being run on minibatches — the thing that will later make
amortized inference practical.

{{< fig src="em-stochastic"
    alt="Log likelihood against iteration for two fitting procedures. Batch EM rises over roughly twenty iterations; stochastic EM reaches the same plateau within about three epochs."
    caption="Batch versus stochastic sufficient-statistic updates, on the same mixture and to the same optimum."
    nb="02b_mixtures_em.ipynb"
    href="https://github.com/msenselab/Variational-Neural-Inference/blob/main/VAE-Tutorial/Part1_Basics/02b_mixtures_em.ipynb" >}}

## 3. Adding time

An HMM is a mixture model whose hidden label is allowed to persist. That single
change buys dynamic programming: forward-backward, smoothing, Viterbi decoding,
Baum-Welch.

{{< fig src="hmm-regimes"
    alt="Top: a coloured band showing a discrete latent state sequence that stays in one state for long stretches before switching. Bottom: the noisy one-dimensional observation generated from it, with horizontal lines marking each state's emission mean."
    caption="Persistent latent regimes generate a noisy sequence. The observation alone is ambiguous at any single time point; the temporal structure is what makes the state recoverable."
    nb="03a_hmm_foundations.ipynb"
    href="https://github.com/msenselab/Variational-Neural-Inference/blob/main/VAE-Tutorial/Part2_Dynamics/03a_hmm_foundations.ipynb" >}}

And it is recoverable. Running Baum-Welch on that sequence recovers the emission
parameters and decodes the state path at 98.8% accuracy, from the observations
alone.

{{< fig src="hmm-baum-welch"
    alt="Three panels. Left: log likelihood rising and converging over about fifteen EM iterations. Middle: true and estimated emission means for three states, plotted against each other and lying nearly on top of one another. Right: the smoothed state decoding compared against the true state sequence, labelled 98.8 percent accuracy."
    caption="Baum-Welch: EM for an HMM. Parameters and state path, both recovered from the observations alone."
    nb="03a_hmm_foundations.ipynb"
    href="https://github.com/msenselab/Variational-Neural-Inference/blob/main/VAE-Tutorial/Part2_Dynamics/03a_hmm_foundations.ipynb" >}}

The extended workshop takes the same machinery to real behavioural recordings,
where an AR-HMM segments continuous posture dynamics into discrete, recurring
syllables.

{{< fig src="arhmm-behaviour"
    alt="Ten principal components of postural data plotted as overlaid time series across thirty seconds, with the background shaded in blocks of different colours marking the discrete state the AR-HMM inferred at each moment."
    caption="An AR-HMM on real posture data: background colour is the inferred discrete state, and state changes line up with changes in the dynamics rather than with the raw signal level."
    nb="03_hmm_lds.ipynb"
    href="https://github.com/msenselab/Variational-Neural-Inference/blob/main/VAE-Tutorial/Part2_Dynamics/03_hmm_lds.ipynb" >}}

## 4. The VAE, and what amortization actually means

Everything so far computed a posterior by running an algorithm per data point.
The VAE's move is to train a network to *predict* the posterior instead — one
forward pass, any data point, including ones never seen during fitting.

{{< fig src="vae-architecture" wide="true"
    alt="Six-panel schematic of a standard variational autoencoder: observed two-dimensional data, an inference network producing a mean and log variance, a stochastic bottleneck with the reparameterization trick and a standard normal prior, a generative network, and the reconstruction. The ELBO objective is written along the bottom."
    caption="The standard VAE end to end: inference network, stochastic bottleneck, generative network, and the ELBO that couples them. The reparameterization trick in panel c is what lets a gradient pass through a sample."
    nb="04_standard_vae.ipynb"
    href="https://github.com/msenselab/Variational-Neural-Inference/blob/main/VAE-Tutorial/Part3_Variational/04_standard_vae.ipynb" >}}

Training it on clustered data shows what the latent space is for: structure that
was spread across the observation space gets folded into two dimensions, with
the clusters still separated.

{{< fig src="vae-latent-space"
    alt="Scatter plot of encoded data in a two-dimensional latent space, points coloured by their original first coordinate, forming three separated groups with a smooth colour gradient within each."
    caption="The learned latent space, coloured by the original coordinate. Nothing told the model about the clusters; they fall out of fitting the ELBO."
    nb="04_standard_vae.ipynb"
    href="https://github.com/msenselab/Variational-Neural-Inference/blob/main/VAE-Tutorial/Part3_Variational/04_standard_vae.ipynb" >}}

If you have read about VAEs and still do not feel you could write one, this is
the notebook I would hand you.

## 5. Doing the inference by hand

Before trusting a network to produce posteriors, it is worth deriving them once.
Notebook 05 does coordinate-ascent variational inference and variational EM
explicitly, then turns the result on the Kato *et al.* whole-brain recordings
and asks whether the inferred discrete states mean anything.

{{< fig src="cavi-overlap"
    alt="Column-normalized overlap matrix. Rows are hand-annotated behavioural labels from Kato et al., columns are inferred discrete states. Bright cells form a near-diagonal band, showing each annotated behaviour maps onto a small group of inferred states."
    caption="Inferred states against hand-annotated behaviour. The band structure is the claim: unsupervised states line up with labels nobody gave the model."
    nb="05_variational_em.ipynb"
    href="https://github.com/msenselab/Variational-Neural-Inference/blob/main/VAE-Tutorial/Part3_Variational/05_variational_em.ipynb" >}}

## 6. Dynamics: the sequential VAE

LFADS is where the two halves meet. The latent variable becomes the state of a
recurrent generator, inference becomes a bidirectional encoder over the whole
trial, and the observation model becomes Poisson spikes.

{{< fig src="lfads-architecture" wide="true"
    alt="Six-panel schematic of LFADS: a spike train input, a bidirectional GRU encoder, latent posteriors for the initial condition and time-varying inputs, a GRU generator, Poisson firing rates, and generated spike counts. The training objective with its two KL terms is written along the bottom."
    caption="LFADS as a sequential VAE. A trial-level initial condition and time-varying inferred inputs jointly drive a recurrent generator; the objective is still an ELBO, now with one KL term per latent."
    nb="06_lfads.ipynb"
    href="https://github.com/msenselab/Variational-Neural-Inference/blob/main/VAE-Tutorial/Part3_Variational/06_lfads.ipynb" >}}

Because the data are synthetic here, the learned dynamics can be checked against
the oscillator that actually produced them — including a Lyapunov estimate and a
phase portrait of the learned autonomous system.

{{< fig src="lfads-oscillator"
    alt="Left column: two sinusoidal ground-truth latent states over time and the sparse input pulses driving them. Right column: the model's latent trajectories projected onto the top two principal components, and the inferred input over the same window."
    caption="Ground-truth oscillatory dynamics against what the model recovered, on held-out trials. Notebook 06 keeps fifty trials genuinely held out rather than reporting training-set reconstruction."
    nb="06_lfads.ipynb"
    href="https://github.com/msenselab/Variational-Neural-Inference/blob/main/VAE-Tutorial/Part3_Variational/06_lfads.ipynb" >}}

## 7. Nonlinear, but still interpretable

A recurrent generator is expressive and opaque. gpSLDS is the capstone because
it gives up some of that expressiveness on purpose: the vector field is a
Gaussian process built from locally linear regimes, so the fitted dynamics can
still be read.

{{< fig src="gpslds-architecture" wide="true"
    alt="Six-panel schematic of a Gaussian process switching linear dynamical system: neural data, a latent path with trajectory uncertainty, a sparse GP with inducing locations, a smooth partition into regimes, local linear dynamics with interpretable fixed points, and a Poisson readout. The smoothly switching linear kernel is written along the bottom."
    caption="gpSLDS: smooth switching, local linear structure, and uncertainty in one model. The kernel is a weighted sum of linear kernels, one per regime."
    nb="08_gpslds.ipynb"
    href="https://github.com/msenselab/Variational-Neural-Inference/blob/main/VAE-Tutorial/Part4_Advanced/08_gpslds.ipynb" >}}

{{< fig src="gpslds-switching"
    alt="Three panels: a vector field over a two-dimensional latent space with trajectories rotating around two separate fixed points; an example latent trajectory showing two state variables over five seconds; and the resulting Poisson spike raster."
    caption="Synthetic switching dynamics, from vector field to latent trajectory to spikes."
    nb="08_gpslds.ipynb"
    href="https://github.com/msenselab/Variational-Neural-Inference/blob/main/VAE-Tutorial/Part4_Advanced/08_gpslds.ipynb" >}}

{{< fig src="gpslds-local-linear"
    alt="Three vector-field panels showing the fitted dynamics linearized around three different points in latent space, each annotated with the eigenvalues of the local Jacobian. The outer two show rotation, the middle one shows a saddle."
    caption="The payoff: linearize the fitted vector field anywhere and read off the eigenvalues. Rotation at the flanks, a saddle in between — the kind of statement an RNN generator cannot make about itself."
    nb="08_gpslds.ipynb"
    href="https://github.com/msenselab/Variational-Neural-Inference/blob/main/VAE-Tutorial/Part4_Advanced/08_gpslds.ipynb" >}}

## All eleven notebooks

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

If you read four of them, read **02**, **04**, **06** and **08**. Notebooks 00
and 01 are prerequisites rather than lessons, and the complete classical
inference track is 02 → 02b → 03a.

## Running them

Notebooks 07 and 08 depend on external implementations registered as Git
submodules, so downloading single files is not enough for those two:

```bash
git clone https://github.com/msenselab/Variational-Neural-Inference.git
cd Variational-Neural-Inference
python scripts/setup_all.py
```

The setup script initializes the submodules, applies the JAX and NumPy
compatibility patches, and installs a CPU-compatible environment for all eleven
notebooks. It is idempotent, and `--dry-run` shows what it would do. For a
smaller install, `requirements-core.txt` covers notebooks 00–06.

Where the notebooks are honest about their limits, so is this page: notebook 05
runs its full synthetic CAVI tutorial without any download and only fetches the
Kato dataset if you opt in; notebook 06 splits 950 training and 50 genuinely
held-out test trials; notebook 08 demonstrates the gpSLDS computational core
rather than reproducing the full upstream fitting pipeline.

## Attribution

The tutorial adapts material from Stanford's
[STATS 320](https://slinderman.github.io/ml4nd/) and from the original model
implementations; `references/ATTRIBUTION.md` in the repository has the details.
The underlying papers are Kingma and Welling (2013) for the VAE, Pandarinath et
al. (2018) for LFADS, Vyas et al. (2020) for computation through dynamics, and
Hu et al. (2024) for gpSLDS.
