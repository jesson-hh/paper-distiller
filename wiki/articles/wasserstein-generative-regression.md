---
title: "Wasserstein Generative Regression (arxiv:2306.15163, JRSSB 2025)"
category: "articles"
slug: "wasserstein-generative-regression"
tags: ["theory", "conditional-generation", "regression", "wasserstein", "unified-framework", "huang-group", "jrssb-2025", "arxiv-2023"]
authors: ["huang-jian"]
refs: ["arxiv:2306.15163", "doi:10.1093/jrsssb/qkaf053"]
links: ["huang-jian", "cnf-convergence-distribution-learning", "conditional-follmer-flow-distribution-learning", "conditional-stochastic-interpolation", "gaussian-interpolation-flows", "interdiff-inter-stock-correlations", "synthetic-augmentation-financial-timeseries"]
created: "2026-04-17T13:30:00"
updated: "2026-04-17T13:30:00"
---

# Wasserstein Generative Regression

> **Authors**: Shanshan Song · Tong Wang · Guohao Shen · Yuanyuan Lin · Jian Huang (see [[huang-jian]])
> **arxiv**: 2306.15163, Jun 2023
> **Published**: Journal of the Royal Statistical Society Series B (JRSSB), 2025
> **Status in our wiki**: L2 deep note — **Huang's flagship paper**

## TL;DR

A single training objective that **simultaneously fits a nonparametric
regression** and **learns the full conditional distribution** — the first
framework to unify these two goals with one provable rate. Handles
multivariate response, gives valid prediction intervals for free, and
dominates cWGANs empirically (on CT-Slices, coverage 96% vs cWGAN's 48%).

Rate:
$$\boxed{\;\mathbb{E}_\mathcal{S}\big\{\mathbb{E}_X \|\mathbb{E}_\eta \hat g(X, \eta) - \mathbb{E}_\eta g^\star(X, \eta)\|^2\big\} \;\le\; C_1\, n^{-\beta / (2\beta + [3(m+d)] \vee [2\beta(d+q+1)])}\;}$$

where $\beta$ = Hölder smoothness of $g^\star$, $m$ = noise dim, $d$ =
predictor dim, $q$ = response dim.

---

## 1. Setting

- **Data**: i.i.d. $(X_i, Y_i) \in \mathbb{R}^d \times \mathbb{R}^q$
  from a joint $P_{X, Y}$.
- **Goal**: learn a generator $g^\star(X, \eta)$ such that (a) the
  *conditional mean* recovers the regression function and (b) the
  *conditional distribution* $g^\star(X, \cdot) \# \mathcal{N}$ matches
  $P_{Y \mid X}$.
- **Noise injection**: $\eta \in \mathbb{R}^m$ is exogenous Gaussian
  noise — making $g$ a stochastic function of $X$.

## 2. Method — The Unified Loss

The hallmark contribution: **one objective** combining both goals:

$$L(g, f) = \lambda_w\, L_W(g, f) + \lambda_\ell\, L_{\text{LS}}(g)$$

where
- $L_W$ = **1-Wasserstein dual** (via a discriminator $f$): forces
  $P_{X, g} \approx P_{X, Y}$ at the distribution level.
- $L_{\text{LS}}$ = standard **least-squares regression**:
  $\mathbb{E} \|Y - \mathbb{E}_\eta g(X, \eta)\|^2$ — pins the
  conditional mean.

**Extremes**:
- $\lambda_\ell = 1, \lambda_w = 0$: pure nonparametric regression (no
  distribution matching).
- $\lambda_\ell = 0, \lambda_w = 1$: conditional WGAN (no mean
  regularisation).
- Interior of the simplex: **the new framework** — both goals
  simultaneously.

Empirical loss (Eq. 6):
$$\hat L(g_\theta, f_\phi) = \lambda_w\, \hat L_W(g_\theta, f_\phi) + \lambda_\ell\, \hat L_{\text{LS}}(g_\theta)$$

with $\hat L_{\text{LS}}$ approximated using $J$ noise samples per data
point.

## 3. Assumptions

- **Conditions 1-2 (bounded support)**: $(X, Y)$ supported on compact
  $\Omega \subseteq [-B_0, B_0]^{d+q}$; noise $\eta$ on compact set.
- **Condition 3 (Hölder smoothness)**:
  $g^\star_k \in \mathcal{H}^\beta(\Omega_\eta \times \Omega_X, B_1)$
  for each response component $k = 1, \ldots, q$.
- **Condition 4**: technical condition linking conditional expectations
  to point evaluations.
- **Conditions ND1, NG1**: deep ReLU network width & depth scaling
  polynomially in $(d, q, m)$.

## 4. Main Results

**Theorem 1 (excess risk on conditional mean)**:
$$\mathbb{E}_\mathcal{S}\Big\{\mathbb{E}_X \|\mathbb{E}_\eta \hat g(X,\eta) - \mathbb{E}_\eta g^\star(X,\eta)\|^2\Big\} \le C_1\, n^{-\beta/(2\beta + [3(m+d)] \vee [2\beta(d+q+1)])}$$

**Theorem 2 (distributional error)**:
$$\mathbb{E}_\mathcal{S}\big\{d_{\mathcal{F}_B^1}(P_{X, \hat g}, P_{X, Y})\big\} \le C_2\, n^{-\beta/(2\beta + [3(m+d)] \vee [2\beta(d+q+1)])}$$

**The key observation**: *both objectives achieve the same rate* — the
unified loss does not degrade either goal relative to its pure
counterpart.

### Reading the exponent

The denominator is
$$[3(m+d)] \;\vee\; [2\beta(d + q + 1)]$$

- If $\beta$ is small (rough target): $3(m + d)$ dominates — rate
  $\approx n^{-\beta/(2\beta + 3m + 3d)}$.
- If $\beta$ is large (smooth target): $2\beta(d + q + 1)$ dominates —
  rate $\approx n^{-1/(2(d + q + 1) + 1/\beta)}$.
- Response dim $q$ only shows up in the smooth-target regime; the
  rough-target regime is $q$-free. *This is nice: if the conditional
  distribution is rough but the response is high-dim, the rate is not
  catastrophically worse than the univariate case.*

## 5. Proof Technique

- **Two-term analysis**: decompose excess risk into (a) regression
  component (standard deep-net nonparametric rate with Hölder $\beta$)
  and (b) Wasserstein-1 component (empirical-process bound on the
  discriminator class).
- The unified loss's **regulariser** structure means the two components
  interact additively, not multiplicatively — the rate is governed by
  the worse of the two.
- Network approximation: standard Yarotsky-type bounds for deep ReLU.

## 6. Novelty vs Prior Work

- **vs. classical quantile regression**: captures full conditional
  distribution; handles multivariate $Y$ (quantile regression requires
  picking quantile levels component-wise).
- **vs. diffusion models**: deterministic generator + Wasserstein
  matching rather than iterative reverse SDE — much cheaper at
  inference.
- **vs. cWGANs**: explicitly regularises the conditional mean via the
  LS term — avoids the common cWGAN failure of matching marginal-ish
  distributions but missing the mean.
- Key quote from the paper:
  > "*unified approach for nonparametric regression and conditional
  > distribution learning*"

## 7. Empirical Evidence

- **Simulations (Models 1-5)**: WGR matches standard nonparametric least
  squares (NLS) on $L_1/L_2$ error; dominates cWGAN on conditional
  quantiles and conditional std.
- **CT-Slices**: **95% prediction-interval coverage = 0.96 vs cWGAN's
  0.48** — dramatic gap; indicates cWGAN's estimated conditional
  distributions are far from calibrated.
- **UJIndoorLoc**: multivariate response (indoor localisation).
- **MNIST image reconstruction**: sharper and more diverse samples than
  cWGAN.

## 8. Limitations

- **Bounded support** excludes heavy-tailed data (finance).
- Hölder smoothness of $g^\star$ is difficult to verify in practice.
- Discriminator class must be properly chosen — the 1-Wasserstein dual
  requires 1-Lipschitz critics; practical implementations use spectral
  normalisation.
- Optimisation (minimax) non-trivial; paper assumes ERM is achieved.

---

## Connections

- **unifies** the lineage
  [[cnf-convergence-distribution-learning]] (pure distribution learning)
  and classical nonparametric regression into one loss.
- **same-group alternative to** [[conditional-follmer-flow-distribution-learning]]:
  both address conditional generation; Föllmer uses ODE + velocity
  regression, WGR uses Wasserstein + regression.
- **precursor to** [[conditional-stochastic-interpolation]] (Huang's
  2023-12 paper) — the stochastic-interpolation framing is one way to
  make the generator side of WGR more flexible.
- **applies to** [[interdiff-inter-stock-correlations]] and
  [[fts-interdiff-fusion]] — if we view factor-conditional return
  generation as both "regression on factors" + "learn tail
  distribution", the WGR objective is a natural choice.
- **method-class overlaps with** WDRO
  ([[jiao-yuling]] cluster B) — both use Wasserstein distance but in
  opposite directions (WDRO bounds risk; WGR matches distributions).

## Contradictions

_None — the framework is additive; claims don't conflict with
neighbouring work._

## Open questions

- **Heavy-tail extension**: financial returns violate bounded-support
  assumption; does the rate deteriorate gracefully or catastrophically?
- **Multi-modal predictor**: conditioning on regime-dependent factors
  — does the single-generator form suffice or do we need mixtures?
- **Iterative / streaming version**: can we do *online* WGR with the
  same rate? Ties into streaming-data extensions seen in
  [[huang-jian]]'s debiased-SGD line.

## My take / relevance

**Most promising architectural template for factor-conditional finance
generators.** The CT-Slices coverage gap (96% vs 48%) is the empirical
evidence we want — cWGAN-style approaches do badly on calibrated
intervals, WGR does well. For our application we want calibrated
quantiles of future returns given factor state; WGR does this directly.

Concrete action items:
1. Re-implement WGR on our CSI300 setup (30-day return windows, 5-10
   factor inputs) and compare calibration against the InterDiff baseline.
2. Study whether the paper's $J$-sample-per-anchor hack causes a bias
   in our high-variance finance setting.
3. If WGR calibrates well, it could replace the cWGAN-ish path in
   [[fts-interdiff-fusion]] entirely.
