"""Create the Bayesian observer tutorial notebook (Foundations 10).

    python scripts/create_bayesian_observer_tutorial.py
    python scripts/nb_to_page.py notebooks/foundations/10_bayesian_observer.ipynb
"""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "foundations" / "10_bayesian_observer.ipynb"


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text, tags=None):
    cell = nbf.v4.new_code_cell(text.strip())
    if tags:
        cell.metadata["tags"] = tags
    return cell


nb = nbf.v4.new_notebook()
nb.metadata.update(
    {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "page": {
            "title": "10: When expectations change perceived time",
            "summary": "A duration feels longer among short ones. We turn that into a reproduction experiment, derive a Bayesian observer, find out which of its parameters the experiment can identify, give it a spiking implementation, and design the test that could reject it.",
            "weight": 100,
            "slug": "10-bayesian-observer",
            "section": "modeling/foundations",
            "date": "2026-10-15",
        },
    }
)

nb.cells = [
    md(
        r"""
# When expectations change perceived time

Hold a stimulus on screen for 0.9 s. Among intervals that are mostly shorter, it seems long. Among
intervals that are mostly longer, it seems short. Asked to reproduce it, people press a key for
different lengths of time, although the physical interval never changed.

This tutorial follows that observation as deep as one model can take it. Each stage answers one
question:

| stage | question |
|---|---|
| 1. Cognitive phenomenon | What exactly needs explaining? |
| 2. Behavioral experiment | How do we make it measurable? |
| 3. Empirical signature | What pattern in the data is the phenomenon? |
| 4. Computational model | What minimal computation produces that pattern, and can we identify its parameters? |
| 5. Neural implementation | Could spiking neurons carry out that computation? |
| 6. Prediction and failure | Which experiment could prove the model wrong? |

**About the data.** Stages 2 to 6 use data simulated from a model whose parameters we chose. That is
deliberate: with known ground truth we can check whether each analysis recovers it. Stage 6 then
confronts the model with real reproduction data.
"""
    ),
    code(
        r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

plt.rcParams.update({
    "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "font.size": 10,
})
colors = {"short context": "#2563eb", "long context": "#dc2626"}
rng = np.random.default_rng(12)
"""
    ),
    md(
        r"""
## 1. Cognitive phenomenon

Judgments of magnitude drift toward the middle of the range a person has recently experienced.
Short intervals are overestimated, long ones underestimated. Hollingworth (1910) called this the
*central tendency of judgment*; in timing research it is also known as Vierordt's law.

Two features make it a good first case:

- **It is about context, not the stimulus.** The same physical interval is judged differently
  depending on what else was shown.
- **It is graded.** People are not simply wrong. Their judgments still track the stimulus, only less
  steeply than they should.

A useful explanation must account for both: why judgments follow the stimulus, and why they are
pulled toward the context.
"""
    ),
    md(
        r"""
## 2. Behavioral experiment

**Task: temporal reproduction.** A disc appears for a duration $s$. After a short blank, the
participant presses and holds a key for as long as the disc was shown. The hold time is the
response $r$. Because $r$ is continuous, a single trial shows both the size and the direction of the
error.

**Manipulation: context.** Two blocks contain the same five *test* durations, 0.60 to 1.20 s. They
differ in their *filler* durations. Fillers in the short-context block are mostly short and fillers
in the long-context block are mostly long. Only test trials are analysed, so any difference between
blocks cannot come from the stimuli themselves.

**Controls.**

- Randomise trial order within a block; counterbalance block order across participants.
- Give practice trials so the key press itself is stable before the test begins.
- Analyse context as a block-level factor. Trials within a block are not independent samples of
  the manipulation.
"""
    ),
    code(
        r"""
phases = [("fixation", 0.5, "#cbd5e1"), ("stimulus s", 0.9, "#2563eb"),
          ("blank", 0.4, "#e2e8f0"), ("reproduction r", 0.85, "#16a34a")]
fig, ax = plt.subplots(figsize=(8, 1.4))
t = 0
for label, dur, color in phases:
    ax.barh(0, dur, left=t, color=color, edgecolor="white")
    ax.text(t + dur / 2, 0, label, ha="center", va="center", fontsize=9,
            color="white" if color in ("#2563eb", "#16a34a") else "black")
    t += dur
ax.set(xlabel="time within a trial (s)", yticks=[], xlim=(0, t), title="One reproduction trial")
ax.grid(False)
plt.tight_layout()
plt.show()
"""
    ),
    md(
        r"""
The simulation below generates the test trials of 24 participants. Each block's durations define
what the observer comes to expect: a mean of 0.75 s in the short context and 1.05 s in the long
context. How the observer turns a measurement into a response is the model of stage 4; here it simply
generates data. Each participant also gets a small constant offset, because people differ in how long
they tend to hold a key.
"""
    ),
    code(
        r"""
test_durations = np.array([0.60, 0.75, 0.90, 1.05, 1.20])
context_means = {"short context": 0.75, "long context": 1.05}
truth = {"sensory_sd": 0.16, "prior_sd": 0.20, "motor_sd": 0.045}

def simulate_observer(duration, prior_mean, sensory_sd, prior_sd, motor_sd, rng):
    m = duration + rng.normal(0, 1, len(duration)) * sensory_sd       # m | s ~ N(s, sigma_s^2)
    w = prior_sd**2 / (sensory_sd**2 + prior_sd**2)                     # weight on the measurement
    estimate = w * m + (1 - w) * prior_mean                             # posterior mean
    return estimate + rng.normal(0, motor_sd, len(duration))            # key press adds motor noise

def make_design(n_participants=24, reps=18, reliabilities=("clear",)):
    rows = [(p, c, mu, rel, d)
            for p in range(n_participants)
            for c, mu in context_means.items()
            for rel in reliabilities
            for d in test_durations
            for _ in range(reps)]
    return pd.DataFrame(rows, columns=["participant", "context", "prior_mean", "reliability", "duration"])

data = make_design()
offsets = rng.normal(0, 0.035, data["participant"].nunique())
data["response"] = simulate_observer(data["duration"].to_numpy(), data["prior_mean"].to_numpy(),
                                     rng=rng, **truth) + offsets[data["participant"]]
print(data.sample(5, random_state=1))
print(f"\n{len(data)} test trials: {data['participant'].nunique()} participants x 2 contexts x "
      f"5 durations x 18 repetitions")
"""
    ),
    md(
        r"""
## 3. Empirical signature

What should the phenomenon look like in this table? Average within each participant first, then
across participants, and look for three things:

1. responses still **increase** with the presented duration;
2. the response function is **shallower than the identity line** (central tendency);
3. at the **same** test duration, the long context gives **longer** responses (context shift).
"""
    ),
    code(
        r"""
participant_means = data.groupby(["participant", "context", "duration"], as_index=False)["response"].mean()
summary = participant_means.groupby(["context", "duration"])["response"].agg(["mean", "sem"]).reset_index()

fig, ax = plt.subplots(figsize=(6, 4.2))
for context, g in summary.groupby("context"):
    ax.errorbar(g["duration"], g["mean"], yerr=g["sem"], marker="o", capsize=3, lw=2,
                color=colors[context], label=context)
ax.plot([0.55, 1.25], [0.55, 1.25], color="0.5", ls=":", label="veridical")
ax.set(xlabel="Presented duration s (s)", ylabel="Mean reproduction r (s)",
       title="Same durations, different contexts")
ax.legend(frameon=False)
plt.show()
""",
        tags=["plot"],
    ),
    md(
        r"""
All three features are present. Before proposing a mechanism, put numbers on them with a
description that makes no claims about mechanism:

$$r = \beta_0 + \beta_s\, s + \beta_c\, c + \varepsilon, \qquad c = -\tfrac12 \text{ (short)},\ +\tfrac12 \text{ (long)}.$$

- $\beta_s$: how strongly responses follow the stimulus. A value of 1 means no central tendency.
- $\beta_c$: how far, in seconds, the long context shifts responses relative to the short one.
"""
    ),
    code(
        r"""
c = data["context"].map({"short context": -0.5, "long context": 0.5}).to_numpy()
X = np.column_stack([np.ones(len(data)), data["duration"], c])      # columns: 1, s, c
beta, *_ = np.linalg.lstsq(X, data["response"].to_numpy(), rcond=None)
print(f"beta_s (duration slope) = {beta[1]:.3f}")
print(f"beta_c (context shift)  = {beta[2]:.3f} s")
"""
    ),
    md(
        r"""
The slope is about 0.6 and the context shift is over 100 ms. That is the signature. The regression
says nothing about *why* the slope is 0.6 or *why* the shift has this size. A model should explain
both with the same parameters.
"""
    ),
    md(
        r"""
## 4. Computational model

### The assumptions

The observer never sees $s$ directly. Every symbol below is something we could, in principle,
manipulate or measure:

| symbol | meaning | what changes it |
|---|---|---|
| $s$ | physical duration on this trial | the experimenter |
| $m$ | internal measurement of $s$ | sensory noise on each trial |
| $\sigma_s$ | how blurred the internal clock is | stimulus quality, attention |
| $\mu_p$ | mean duration the observer expects in this block | the context |
| $\sigma_p$ | how broad that expectation is | the spread of durations in the block |
| $w$ | weight given to the measurement | the ratio of $\sigma_s$ to $\sigma_p$ |
| $\sigma_m$ | motor noise of the key press | the response method |

Two assumptions turn this into a model:

$$m \mid s \sim \mathcal N(s, \sigma_s^2) \quad\text{(likelihood)}, \qquad s \sim \mathcal N(\mu_p, \sigma_p^2) \quad\text{(prior)}.$$

### The derivation

By Bayes' rule the posterior is proportional to likelihood times prior. Taking logarithms and
keeping only the terms that depend on $s$:

$$\log p(s \mid m) = -\frac{(m - s)^2}{2\sigma_s^2} - \frac{(s - \mu_p)^2}{2\sigma_p^2} + \text{const}.$$

This is a quadratic in $s$, so the posterior is Gaussian. Setting its derivative to zero gives the
peak, which for a Gaussian is also the mean:

$$\frac{m - s}{\sigma_s^2} = \frac{s - \mu_p}{\sigma_p^2}
\quad\Longrightarrow\quad
\hat s = w\, m + (1 - w)\, \mu_p, \qquad w = \frac{\sigma_p^2}{\sigma_s^2 + \sigma_p^2}.$$

The estimate is a weighted average of what was measured and what was expected. The weight on the
measurement is high when the measurement is precise ($\sigma_s$ small) or the expectation is vague
($\sigma_p$ large).

### What it predicts for the data

The response adds motor noise: $r = \hat s + \varepsilon_m$, with $\varepsilon_m \sim \mathcal N(0, \sigma_m^2)$.
Averaging over $m$ gives the distribution of responses to a given duration:

$$r \mid s \sim \mathcal N\big(\underbrace{w\, s + (1-w)\,\mu_p}_{\text{mean}},\ \underbrace{w^2\sigma_s^2 + \sigma_m^2}_{\text{variance}}\big).$$

Compare this with the regression. The model **explains** both descriptive numbers with one quantity:

- the slope is $\beta_s = w$;
- the context shift is $\beta_c = (1 - w)(\mu_{\text{long}} - \mu_{\text{short}})$.

With $w \approx 0.6$ and a 0.3 s difference between context means, the shift should be about
$0.4 \times 0.3 = 0.12$ s, which is what the regression found.

### From equations to code

Each line of `predict` is one equation above. The likelihood of the data is the product of Gaussian
densities over trials; we minimise its negative logarithm. Standard deviations are optimised on a
log scale so they stay positive, and each fit starts from several points.
"""
    ),
    code(
        r"""
def predict(s, mu_p, sensory_sd, prior_sd, motor_sd, offset):
    w = prior_sd**2 / (sensory_sd**2 + prior_sd**2)          # w = sigma_p^2 / (sigma_s^2 + sigma_p^2)
    mean = w * s + (1 - w) * mu_p + offset                   # E[r | s] = w s + (1 - w) mu_p
    sd = np.sqrt((w * sensory_sd) ** 2 + motor_sd**2)        # Var[r | s] = w^2 sigma_s^2 + sigma_m^2
    return mean, sd

def unpack(theta, n_levels):
    # theta = [log sigma_s per reliability level..., log sigma_p, log sigma_m, offset]
    return np.exp(theta[:n_levels]), np.exp(theta[n_levels]), np.exp(theta[n_levels + 1]), theta[-1]

def bayes_nll(theta, frame, n_levels):
    sens, prior_sd, motor_sd, offset = unpack(theta, n_levels)
    level = frame["level"].to_numpy() if n_levels > 1 else np.zeros(len(frame), int)
    mean, sd = predict(frame["duration"].to_numpy(), frame["prior_mean"].to_numpy(),
                       sens[level], prior_sd, motor_sd, offset)
    return -norm.logpdf(frame["response"].to_numpy(), mean, sd).sum()

def fit_bayes(frame, n_levels=1, fixed_sensory=None):
    starts = [np.r_[np.log([a] * n_levels), np.log([b, 0.05]), 0.0]
              for a in (0.08, 0.15, 0.3) for b in (0.1, 0.25, 0.5)]
    if fixed_sensory is not None:           # profile: hold sigma_s, optimise the rest
        f = lambda th: bayes_nll(np.r_[np.log(fixed_sensory), th], frame, 1)
        best = min((minimize(f, s0[1:], method="Nelder-Mead", options={"maxiter": 4000})
                    for s0 in starts[:3]), key=lambda r: r.fun)
        return best
    f = lambda th: bayes_nll(th, frame, n_levels)
    return min((minimize(f, s0, method="Nelder-Mead", options={"maxiter": 6000}) for s0 in starts),
               key=lambda r: r.fun)

fit = fit_bayes(data)
sens, prior_sd, motor_sd, offset = unpack(fit.x, 1)
w_hat = prior_sd**2 / (sens[0]**2 + prior_sd**2)
print(f"{'':12s}{'fitted':>8s}{'true':>8s}")
for name, est, tru in [("sigma_s", sens[0], truth["sensory_sd"]), ("sigma_p", prior_sd, truth["prior_sd"]),
                       ("sigma_m", motor_sd, truth["motor_sd"])]:
    print(f"{name:12s}{est:8.3f}{tru:8.3f}")
print(f"{'w':12s}{w_hat:8.3f}{truth['prior_sd']**2 / (truth['sensory_sd']**2 + truth['prior_sd']**2):8.3f}")
"""
    ),
    md(
        r"""
### Diagnosis: what is the experiment allowed to tell us?

The weight $w$ is recovered well. The individual noise terms may or may not be, and a good fit does
not tell us which. Count what the data constrain. The mean response pins down $w$, which depends only
on the **ratio** of $\sigma_s$ to $\sigma_p$. The response variance pins down one number,
$w^2\sigma_s^2 + \sigma_m^2$. That is two constraints on three noise parameters.

A **profile likelihood** makes this visible. Fix $\sigma_s$ at a value, let all other parameters adjust
to fit as well as they can, and record how bad the best fit is. A sharp valley means $\sigma_s$ is
identified. A flat floor means many values explain the data equally well.
"""
    ),
    code(
        r"""
grid = np.linspace(0.04, 0.32, 15)
profile = np.array([fit_bayes(data, fixed_sensory=g).fun for g in grid])

fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot(grid, profile - profile.min(), "o-", color="#7c3aed")
ax.axhline(1.92, color="0.5", ls="--", label="95% interval threshold")
ax.axvline(truth["sensory_sd"], color="k", ls=":", label="true sigma_s")
ax.set(xlabel="sigma_s held fixed", ylabel="increase in -log likelihood",
       title="One reliability condition: a flat valley", ylim=(-1, 40))
ax.legend(frameon=False, fontsize=8)
plt.show()
""",
        tags=["plot"],
    ),
    md(
        r"""
Every value of $\sigma_s$ whose curve lies under the dashed line is compatible with the data at
the 95% level. Small values fit as well as the true one: the model compensates by shrinking $\sigma_p$
to keep $w$ fixed and enlarging $\sigma_m$ to keep the variance fixed. With this design, a fitted
$\sigma_s$ should not be interpreted as "how noisy this person's clock is."

**The fix is a manipulation, not a better optimiser.** Add a second, noisier version of the stimulus
(for example a dimmer disc) in both contexts. Sensory noise now differs between conditions while the
prior and motor noise are shared. That gives four constraints (two weights, two variances) for four
noise parameters.

Parameter recovery shows the difference. Simulate 12 datasets with different true $\sigma_s$, fit
each, and compare the estimates with the truth under both designs.
"""
    ),
    code(
        r"""
def simulate_dataset(design, sensory_by_level, seed):
    r = np.random.default_rng(seed)
    d = design.copy()
    d["level"] = d["reliability"].map({"clear": 0, "noisy": 1}).astype(int)
    sd = np.asarray(sensory_by_level)[d["level"].to_numpy()]
    off = r.normal(0, 0.035, d["participant"].nunique())
    d["response"] = simulate_observer(d["duration"].to_numpy(), d["prior_mean"].to_numpy(), sd,
                                      truth["prior_sd"], truth["motor_sd"], r) + off[d["participant"]]
    return d

one_design = make_design(reliabilities=("clear",))
two_design = make_design(reliabilities=("clear", "noisy"), reps=9)    # same total trial count
true_values = np.linspace(0.07, 0.25, 12)
recovered = {"one condition": [], "clear + noisy": []}
for i, sd_true in enumerate(true_values):
    d1 = simulate_dataset(one_design, [sd_true], seed=200 + i)
    recovered["one condition"].append(unpack(fit_bayes(d1, 1).x, 1)[0][0])
    d2 = simulate_dataset(two_design, [sd_true, 2 * sd_true], seed=300 + i)
    recovered["clear + noisy"].append(unpack(fit_bayes(d2, 2).x, 2)[0][0])

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
for ax, (name, est) in zip(axes, recovered.items()):
    ax.plot([0.04, 0.27], [0.04, 0.27], color="0.5", ls=":")
    ax.scatter(true_values, est, color="#7c3aed", zorder=3)
    r = np.corrcoef(true_values, est)[0, 1]
    ax.set(xlabel="true sigma_s", title=f"{name}: r = {r:.2f}")
axes[0].set_ylabel("recovered sigma_s")
plt.tight_layout()
plt.show()
for name, est in recovered.items():
    err = np.sqrt(np.mean((np.array(est) - true_values) ** 2))
    print(f"{name:14s} RMS recovery error = {err:.3f}")
"""
    ),
    md(
        r"""
With one condition, estimates are scattered or biased because the data cannot separate the noise
terms. With two reliability conditions and the same number of trials, they line up with the truth.
**The design, not the model, decides what can be learned.**
"""
    ),
    md(
        r"""
## 5. Neural implementation

The model of stage 4 states a computation. It does not say how neurons would perform it. Here is one
way they could.

**A population of duration-tuned neurons.** Give each of 40 neurons a preferred duration and a
Gaussian tuning curve $f_i(s)$. On a trial, neuron $i$ emits a Poisson spike count $n_i$ with mean
$g\,f_i(s)$. The gain $g$ stands for stimulus quality: a clear stimulus drives the neurons strongly,
a dim one weakly.

**Reading out a posterior.** For independent Poisson neurons the log likelihood of a duration is a
weighted sum of spike counts,

$$\log p(\mathbf n \mid s) = \sum_i n_i \log f_i(s) - g \sum_i f_i(s) + \text{const}.$$

If the tuning curves tile the duration axis evenly, the second sum hardly depends on $s$. A
downstream area can then add the log prior for the current context and read out the posterior mean.
Nothing in the code below knows about $w$: the weighting of measurement against expectation emerges
from the spike counts.
"""
    ),
    code(
        r"""
preferred = np.linspace(0.2, 1.8, 40)           # preferred durations, shape [neurons]
tuning_width = 0.15
s_axis = np.linspace(0.3, 1.6, 600)             # durations the readout considers, shape [grid]

def tuning(s):
    return np.exp(-0.5 * ((np.asarray(s)[..., None] - preferred) / tuning_width) ** 2) + 0.05

def spikes(s, gain, n_trials, rng):
    return rng.poisson(gain * tuning(np.full(n_trials, s)))                # [trials, neurons]

def decode(counts, gain, prior_mean, prior_sd=truth["prior_sd"]):
    F = gain * tuning(s_axis)                                              # [grid, neurons]
    log_like = counts @ np.log(F).T - F.sum(1)                             # [trials, grid]
    log_post = log_like + norm.logpdf(s_axis, prior_mean, prior_sd)        # add the context prior
    post = np.exp(log_post - log_post.max(1, keepdims=True))
    post /= post.sum(1, keepdims=True)
    return post @ s_axis, post                                             # posterior mean, posterior

n_one = spikes(0.9, gain=3.0, n_trials=1, rng=rng)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].bar(preferred, n_one[0], width=0.035, color="0.35")
axes[0].axvline(0.9, color="k", ls=":")
axes[0].set(xlabel="preferred duration (s)", ylabel="spike count", title="One trial, s = 0.9 s")
for context, mu in context_means.items():
    est, post = decode(n_one, 3.0, mu)
    axes[1].plot(s_axis, post[0] / post[0].max(), color=colors[context],
                 label=f"{context}: estimate {est[0]:.2f} s")
axes[1].axvline(0.9, color="k", ls=":")
axes[1].set(xlabel="duration (s)", ylabel="posterior (scaled)", title="Same spikes, two priors")
axes[1].legend(frameon=False, fontsize=8)
plt.tight_layout()
plt.show()
""",
        tags=["plot"],
    ),
    md(
        r"""
The same spike pattern yields two different estimates once the context prior is added. Now repeat
this over many trials at two gains. Fewer spikes make a broader likelihood, so the prior should pull
harder.
"""
    ),
    code(
        r"""
rows = []
for gain, label in [(3.0, "high gain (clear)"), (0.8, "low gain (dim)")]:
    for s in test_durations:
        counts = spikes(s, gain, 400, rng)
        for context, mu in context_means.items():
            rows.append((label, context, s, decode(counts, gain, mu)[0].mean()))
neural = pd.DataFrame(rows, columns=["gain", "context", "duration", "estimate"])
shift = (neural.pivot_table(index=["gain", "duration"], columns="context", values="estimate")
               .eval("`long context` - `short context`").groupby("gain").mean())
print("context shift of the decoded estimate (s):")
print(shift.round(3))
"""
    ),
    md(
        r"""
The decoded estimates show the same signature as behavior, and the context shift is larger when
the gain is low. The population carries the reliability of the evidence in its total activity, and
a downstream readout that adds a prior weights the two correctly without being told how.
Ma, Beck, Latham and Pouget (2006) developed this idea as *probabilistic population codes*.

Three claims must be kept apart:

1. **Consistent with neural data.** Recordings in monkey medial premotor cortex show timing signals
   that change with the prior (Jazayeri & Shadlen, 2010, studied the behavioral side of this
   paradigm). Consistency does not mean the code above is used.
2. **Implementable.** The simulation shows spiking neurons *can* perform the computation. It does
   not show that they do.
3. **Causal.** Only perturbing the neurons and observing the predicted change in behavior would
   show that the brain relies on this mechanism.

This tutorial reaches level 2.
"""
    ),
    md(
        r"""
## 6. Prediction and failure

### A rival that fits the same data

A simpler account needs no probabilities. An **anchoring** observer mixes each stimulus with the
block mean using a fixed weight $a$:

$$r = a\,s + (1 - a)\,\mu_p + \varepsilon.$$

On the data of stage 3 this is indistinguishable from the Bayesian observer: both give a slope
below 1 and a context shift of $(1 - a)(\mu_{\text{long}} - \mu_{\text{short}})$. A good fit to the
original experiment cannot tell them apart.

### The experiment that separates them

The models disagree about **reliability**. The Bayesian weight $w$ falls when the stimulus gets
noisier, so the context shift should grow. The anchoring weight $a$ is fixed, so the shift should
not change. The clear and noisy design of stage 4 tests exactly this. Simulate it under each model.
"""
    ),
    code(
        r"""
def simulate_anchoring(design, a, noise_by_level, seed):
    r = np.random.default_rng(seed)
    d = design.copy()
    d["level"] = d["reliability"].map({"clear": 0, "noisy": 1}).astype(int)
    noise = np.asarray(noise_by_level)[d["level"].to_numpy()]
    off = r.normal(0, 0.035, d["participant"].nunique())
    d["response"] = (a * d["duration"] + (1 - a) * d["prior_mean"]
                     + r.normal(0, 1, len(d)) * noise + off[d["participant"]])
    return d

bayes_data = simulate_dataset(two_design, [0.10, 0.22], seed=7)
anchor_data = simulate_anchoring(two_design, a=0.6, noise_by_level=[0.10, 0.15], seed=7)

def context_shift(d):
    m = d.groupby(["participant", "reliability", "context"])["response"].mean().unstack("context")
    return (m["long context"] - m["short context"]).groupby("reliability").agg(["mean", "sem"])

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, (name, d) in zip(axes, [("data from the Bayesian observer", bayes_data),
                                ("data from the anchoring observer", anchor_data)]):
    cs = context_shift(d).loc[["clear", "noisy"]]
    ax.bar(cs.index, cs["mean"], yerr=cs["sem"], capsize=4, color=["#93c5fd", "#1e3a8a"])
    ax.set(title=name, xlabel="stimulus")
axes[0].set_ylabel("context shift (s)")
plt.tight_layout()
plt.show()
""",
        tags=["plot"],
    ),
    md(
        r"""
The prediction is qualitative and easy to see. The Bayesian observer's context shift roughly doubles
for the noisy stimulus; the anchoring observer's does not change.

To decide between the models formally, fit both and compare how well they predict participants they
were **not** fitted to. We fit on half of the participants and score the log likelihood of the other
half. As a check of the procedure itself (*model recovery*), we do this for data generated by each
model: the comparison should pick the generating model in both cases.
"""
    ),
    code(
        r"""
def anchoring_nll(theta, frame):
    a = 1 / (1 + np.exp(-theta[0]))                  # weight a in (0, 1)
    offset = theta[1]
    noise = np.exp(theta[2:4])[frame["level"].to_numpy()]
    mean = a * frame["duration"] + (1 - a) * frame["prior_mean"] + offset
    return -norm.logpdf(frame["response"], mean, noise).sum()

def fit_anchoring(frame):
    starts = [np.array([x, 0.0, np.log(0.1), np.log(0.15)]) for x in (-1.0, 0.4, 1.5)]
    return min((minimize(anchoring_nll, s0, args=(frame,), method="Nelder-Mead",
                         options={"maxiter": 4000}) for s0 in starts), key=lambda r: r.fun)

def held_out_comparison(d):
    train = d[d["participant"] < 12]
    test = d[d["participant"] >= 12]
    bayes = bayes_nll(fit_bayes(train, 2).x, test, 2)
    anchor = anchoring_nll(fit_anchoring(train).x, test)
    return bayes, anchor

for name, d in [("Bayesian", bayes_data), ("anchoring", anchor_data)]:
    b, a = held_out_comparison(d)
    winner = "Bayesian" if b < a else "anchoring"
    print(f"data from {name:9s}: held-out -log L  Bayesian {b:8.1f}  anchoring {a:8.1f}  -> prefers {winner}")
"""
    ),
    md(
        r"""
### Where the model fails on real data

The models above were tested on data they generated. Real data are less polite. Take the reproduction
experiment from notebook 3 (16 participants, durations 0.5 to 1.7 s, from
[msenselab/working_memory_reproduction](https://github.com/msenselab/working_memory_reproduction)).
The Gaussian observer predicts that the **spread** of reproductions is the same at every duration,
because $w^2\sigma_s^2 + \sigma_m^2$ does not depend on $s$.
"""
    ),
    code(
        r"""
URL = ("https://raw.githubusercontent.com/msenselab/working_memory_reproduction/"
       "HEAD/data/AllValidData.csv")
real = pd.read_csv(URL).query("Exp == 'Exp1'")
spread = (real.groupby(["NSub", "curDur"])["repDur"].std()
              .groupby("curDur").agg(["mean", "sem"]))

fig, ax = plt.subplots(figsize=(5.5, 3.6))
ax.errorbar(spread.index, spread["mean"], yerr=spread["sem"], marker="o", capsize=3,
            color="#0f172a", label="real data (mean of participant SDs)")
ax.axhline(spread["mean"].mean(), color="#dc2626", ls="--",
           label="Gaussian observer: one constant SD")
ax.set(xlabel="presented duration (s)", ylabel="SD of reproductions (s)",
       title="Variability grows with duration")
ax.legend(frameon=False, fontsize=8)
plt.show()
print((spread["mean"] / spread.index).round(3).rename("SD / duration"))
""",
        tags=["plot"],
    ),
    md(
        r"""
The standard deviation grows from about 0.17 s to 0.26 s across the range. Timing noise is roughly
proportional to the duration being timed (the *scalar property*), and the model assumed it was
constant. The mean predictions can still look right while the variance is systematically wrong.

The standard repair keeps the Bayesian logic and changes the likelihood: let $\sigma_s$ grow with
$s$, or equivalently assume the measurement is Gaussian on a logarithmic time scale. The posterior
then has no closed form, but it can be computed on a grid exactly as the neural readout in stage 5
did. That repaired model makes its own testable predictions, which is where the next tutorial can
start.

### The model should be rejected or revised if

- the context shift is the same for clear and noisy stimuli (anchoring wins);
- its noise parameters cannot be recovered in simulation from the design actually used;
- it predicts the mean reproductions but misses how their spread changes with duration (it does,
  for the version built here);
- a simpler model predicts held-out participants equally well.
"""
    ),
    md(
        r"""
## What to carry forward

- Start from identical stimuli in different contexts, so the effect cannot come from the stimuli.
- Describe the signature before naming a mechanism; then require the mechanism to explain the
  descriptive numbers.
- A Bayesian observer is a weighted average of measurement and expectation, $\hat s = w m + (1-w)\mu_p$,
  with the weight set by relative reliability.
- Count what the data constrain. With one reliability level, the noise terms are not identifiable;
  a second level fixes that.
- A Poisson population plus an added log prior implements the computation without representing $w$.
- Two models that fit the same data must be separated by the experiment where they disagree.
- Check the variance, not only the mean. Here the constant-noise model fails on real data.

## References

- Hollingworth, H. L. (1910). The central tendency of judgment. *Journal of Philosophy, Psychology
  and Scientific Methods*, 7(17), 461 to 469.
- Jazayeri, M., & Shadlen, M. N. (2010). Temporal context calibrates interval timing. *Nature
  Neuroscience*, 13(8), 1020 to 1026.
- Ma, W. J., Beck, J. M., Latham, P. E., & Pouget, A. (2006). Bayesian inference with probabilistic
  population codes. *Nature Neuroscience*, 9(11), 1432 to 1438.
"""
    ),
]

OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(OUT)
