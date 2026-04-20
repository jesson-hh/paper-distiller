---
title: "Minimax Filter for Financial Stylised-Fact Preservation under Synthetic Augmentation"
category: "open-problems"
slug: "minimax-filter-financial-stylised-facts"
tags: ["synthetic-data", "finance", "wasserstein", "filter-design", "stylised-facts", "minimax"]
refs: ["arxiv:2505.04992", "arxiv:2307.01850"]
links: ["boosting-with-synthetic-pretrained", "synthetic-augmentation-financial-timeseries", "diversity-collapse-sde-framework", "self-consuming-diffusion-stability", "score-contamination-lemma", "jiao-yuling", "waveletdiff-multilevel-2510-11839", "fts-diffusion-iclr-2024"]
created: "2026-04-17T13:00:00"
updated: "2026-04-17T13:00:00"
---

# Minimax Filter for Financial Stylised-Fact Preservation

> **Seeded from**: [[boosting-with-synthetic-pretrained]] (Jiang, Hu, Huang,
> Jiao, Liu 2025, arxiv:2505.04992) — Theorem 3.1 and its filter design.
>
> **Direction of relevance**: [[synthetic-augmentation-financial-timeseries]],
> [[diversity-collapse-sde-framework]].
>
> **Status**: open problem (2026-04-17) — no known solution tailored to
> time-series / finance.

## The Problem

Jiang et al. (2025) prove the **key inequality** governing boosted
statistical learning with synthetic data:

$$\mathbb{E}_{\mathcal{P}_{\text{real}}}[\ell(h, z)] \le \mathbb{E}_{\mathcal{P}_{\text{synth}}}[\ell(h, z)] + L_\ell\, \varepsilon + 2\, \mathfrak{R}_n(\mathcal{H}) + M\sqrt{\tfrac{\log(1/\delta)}{2n}}$$

where $\varepsilon = W_1(\mathcal{P}_{\text{synth}}, \mathcal{P}_{\text{real}})$,
and show that a **Wasserstein filter in a VAE latent space** shrinks
$\varepsilon$ enough to beat the $1/\sqrt n$ real-data baseline on 8
i.i.d. image / tabular benchmarks.

**The problem**: the theorem requires **i.i.d. data**; its direct
translation to time-series fails because:

1. Financial returns are **not i.i.d.** — autocorrelation in
   squared returns, volatility clustering, leverage effect.
2. The "right" distance $\varepsilon$ for a financial process is not
   $W_1$ on marginal distributions but rather **a joint distance that
   captures stylised facts**.
3. VAE latents are not an obvious encoder for returns — DWT or
   Wiener-Itô wavelets might be better, but we don't know which.

**Formal open question**:

> Given a real financial process $\mathbf{X}^{\text{real}}_{1:T}$ and a
> pre-trained generator $G$ producing $\mathbf{X}^{\text{synth}}_{1:T}$,
> what filter $\phi$ minimises the worst-case
> downstream excess risk over a class of stylised-fact functionals
> $\mathcal{F} = \{F_k\}$?
>
> $$\phi^\star = \arg\min_\phi\, \sup_{F \in \mathcal{F}}\, \Big|\mathbb{E}[F(\mathbf{X}^{\text{synth,filtered by }\phi})] - \mathbb{E}[F(\mathbf{X}^{\text{real}})]\Big| + \lambda\,\text{cost}(\phi)$$

## Decomposition into sub-problems

### Sub-problem 1. Which stylised-fact functionals?

We know a finite list of *qualitative* facts (fat tails, leverage
effect, autocorrelation decay, volatility clustering, aggregational
Gaussianity). The question is **which ones to encode in $\mathcal{F}$**
and how to weight them.

Candidate $\mathcal{F}$:
- **Heavy-tail exponent** $\alpha$ from a Hill estimator on absolute
  returns.
- **ACF of $|r_t|$** at lags 1, 5, 10, 21 (daily) or 60 (hourly).
- **Leverage correlation** $\mathrm{corr}(r_t, |r_{t+k}|)$ for small
  $k > 0$.
- **Multi-scale wavelet energy distribution** (from
  [[waveletdiff-multilevel-2510-11839]]).
- **Intraday U-curve** (intraday volatility seasonality).

### Sub-problem 2. What distance on paths?

Possible $\varepsilon$ surrogates beyond $W_1$:

| Distance | Strength | Weakness |
|---|---|---|
| **$W_1$ on empirical marginal** | simple | ignores temporal structure |
| **Sliced-$W_p$ over lag-windows** | captures finite-lag structure | computationally heavy |
| **Maximum-mean-discrepancy (MMD)** with stylised-fact kernel | direct stylised-fact matching | kernel-design-dependent |
| **Signature-kernel distance** (rough-paths) | theoretically principled path distance | high constants |
| **Wavelet coefficient $W_1$** (using DWT as in [[latent-schrodinger-bridge-diffusion]]) | decouples scales, DWT is orthogonal | coefficient distribution still has tails |

### Sub-problem 3. Does the bound still hold under mixing?

Theorem 3.1 relies on i.i.d. sampling for the $M\sqrt{\log(1/\delta)/(2n)}$
term (Hoeffding). For $\beta$-mixing / $\alpha$-mixing processes, this
tail bound loses a factor of $(1/\beta_n)^{1/2}$ typical of
non-i.i.d. concentration. The full paper's filter guarantee **will
shrink correspondingly** — quantifying this loss is part of the open
problem.

### Sub-problem 4. Iterated filtering = self-consuming?

If we **iterate**: filter round 1, re-train generator on round-1 output,
filter round 2, … — does the process stabilise or collapse?

- Single-round filtered boosting (Jiang 2025): *helps*.
- Unfiltered iterative (Shumailov 2024, `cite:shumailov-2024-nature-collapse`):
  *collapses*.
- **Filtered iterative = ?** Likely a recursion similar to
  [[recursive-chi2-inequality]] with an explicit filter-shrinkage factor
  $\rho_\phi$ per round. This ties the open problem directly into
  [[non-perturbative-collapse-recursion]].

## What's needed to solve it

1. **Theory**: extend Theorem 3.1 to $\beta$-mixing sequences and
   stylised-fact kernel MMD.
2. **Experiment**: pick a fixed target $F$ (say, heavy-tail exponent
   $\alpha$), test several candidate filters on CSI300 synthetic
   output from [[waveletdiff-multilevel-2510-11839]], and measure
   downstream regression excess risk. Which filter wins?
3. **Lower bound**: construct a family of generators for which any
   fixed filter $\phi$ incurs a fixed worst-case
   stylised-fact error. This gives a minimax benchmark.

## Expected difficulty

- Sub-problem 1 (choosing $\mathcal{F}$): low — mostly a modelling
  decision, work can start now.
- Sub-problem 2 (path distance): medium — pick a candidate, do the
  math on bounding $W_1 \le f(\text{sig-kernel MMD})$ type inequalities.
- Sub-problem 3 (mixing extension): medium — known techniques from
  Rio / Doukhan, but application-specific.
- Sub-problem 4 (iterated filter dynamics): **hard** — likely needs a
  custom version of our χ² recursion analysis.

## Connections

- **directly extends** [[boosting-with-synthetic-pretrained]] — main
  theorem is the seed.
- **enables** [[synthetic-augmentation-financial-timeseries]] — without
  a good $\phi$, the whole direction is empirical.
- **interacts with** [[diversity-collapse-sde-framework]] and
  [[non-perturbative-collapse-recursion]] via the iterated-filter
  question.
- **uses tools from** [[score-contamination-lemma]] (for the mixture
  decomposition of synthetic-plus-real) and
  [[waveletdiff-multilevel-2510-11839]] (for the wavelet-encoded filter
  candidate).

## Current Notes

- Start with Sub-problem 1 + 2. A 1-week prototype: CSI300 synthetic
  returns from WaveletDiff, filter by wavelet-coefficient $W_1$,
  measure downstream quantile-regression excess risk.
- Sub-problem 4 is the "paper-worthy" piece — probably our next
  theoretical contribution if sub-problems 1–3 land.
