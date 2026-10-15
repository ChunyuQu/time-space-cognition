---
title: 'Computational Neuroscience: From Data to Neural Models'
date: 2026-10-15
weight: 5
cover: /images/foundations/foundations-overview.webp
math: true
summary: 'Executable tutorials that move from cognitive phenomena and behavioral experiments to data, mathematical models, neural mechanisms, and testable predictions.'
---

This series starts with phenomena that can be experienced and measured: biases in perceived time,
limits of attention and memory, learning from prediction errors, and the representation of a stimulus
in neural activity. Behavioral experiments turn those questions into data. Mathematical models make
their assumptions explicit, and neural models ask how the same computation might be implemented.

Wherever possible, the examples use real timing and visual-search experiments from
[MSense Lab](https://github.com/msenselab) rather than generic benchmark tables. Learning rules are
first written by hand in NumPy before a library implementation is introduced, so that the code does
not conceal the mechanism.

The notebooks live in the
[`notebooks/foundations`](https://github.com/ChunyuQu/time-space-cognition/tree/main/notebooks/foundations)
folder of this site's repository. Each page below is generated from the executed notebook, so the
figures are the notebook's real output.

The explanations, derivations, code, simulations, and figures are developed specifically for this
tutorial collection. Each topic follows the same scientific path:

1. **Cognitive phenomenon**, what needs explaining?
2. **Behavioral experiment**, how can it be measured?
3. **Empirical signature**, what structure should appear in the data?
4. **Computational model**, what is the smallest explicit account of that structure?
5. **Neural implementation**, how could a neural system perform the computation?
6. **Prediction and failure**, what new result would distinguish or falsify the model?

## A small mathematical toolkit

You do not need advanced mathematics to begin, but four ideas recur throughout the series.

### Vectors, matrices, and shapes

A trial is a vector when it contains one measurement per time point, and a matrix when it
contains several variables or channels. I write a behavioral design matrix as
\(X\in\mathbb{R}^{N\times P}\): \(N\) observations in rows and \(P\) predictors in columns.
An EEG trial is usually \(X\in\mathbb{R}^{C\times T}\): \(C\) electrodes by \(T\) samples.
Writing the shape beside every intermediate result is one of the simplest ways to prevent a
correct formula from becoming incorrect code.

### Probability, expectation, and variability

For a random variable \(Y\), \(\mathbb{E}[Y]\) is its long-run mean and
\(\mathrm{Var}(Y)=\mathbb{E}[(Y-\mathbb{E}[Y])^2]\) measures dispersion. A probability model
\(p(y\mid\theta)\) says how plausible an observation is under parameters \(\theta\). Statistical
inference asks which parameter values are compatible with the observations and how uncertain
those values remain.

### Likelihood and loss

For independent observations, the likelihood is

$$
L(\theta)=\prod_{i=1}^{N}p(y_i\mid\theta).
$$

Products are numerically awkward, so code normally minimizes the negative log-likelihood
\(-\sum_i\log p(y_i\mid\theta)\). Squared error, binary cross-entropy, and multiclass
cross-entropy are not unrelated recipes: each is a negative log-likelihood under a different
observation model.

### Derivatives and learning

A derivative describes local sensitivity. The gradient
\(\nabla_\theta\mathcal{L}\) collects the derivative of a loss with respect to every parameter.
Gradient descent applies

$$
\theta_{k+1}=\theta_k-\eta\nabla_\theta\mathcal{L}(\theta_k),
$$

where \(\eta\) is the learning rate. Backpropagation is an efficient application of the chain
rule for computing this gradient through a sequence of operations; it is not a separate learning
principle.

## Suggested routes through the notebooks

- **Behavioral modelling:** 1 → 2 → 3 → 4.
- **Neural networks:** 3 → 5 → 6 → 7.
- **Learning and neural coding:** 5 → 8 → 9.
- **BCI preparation:** 3 → 6 → 7, followed by the EEG model notes on the parent page.

Each notebook begins with a question, develops the minimum mathematics needed to answer it,
implements the method, and ends by interpreting a result rather than treating code output as a
conclusion.
