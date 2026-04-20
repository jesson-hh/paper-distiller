---
title: "Score-Based Sequential Langevin Sampling for Nonlinear Assimilation (arxiv:2411.13443)"
category: "articles"
slug: "score-based-sequential-langevin-sampling"
tags: ["theory", "diffusion", "data-assimilation", "bayesian-filtering", "langevin", "log-sobolev", "jiao-group", "arxiv-2024"]
authors: ["jiao-yuling"]
refs: ["arxiv:2411.13443"]
links: ["jiao-yuling", "diffusion-doob-matching-inference-alignment", "girsanov-path-kl-bound", "recursive-chi2-inequality"]
created: "2026-04-17T12:00:00"
updated: "2026-04-17T12:30:00"
---

# Nonlinear Assimilation via Score-Based Sequential Langevin Sampling (SSLS)

> **Authors**: Zhao Ding · Chenguang Duan · Yuling Jiao · Jerry Zhijian Yang · Cheng Yuan · Pingwen Zhang (see [[jiao-yuling]])
> **arxiv**: 2411.13443 (v4, Nov 2024)
> **Status in our wiki**: L2 deep note

## TL;DR

Recursive Bayesian filter where each posterior update is realised by
Langevin dynamics driven by a **learned score** of the current filter
density. Proves a cumulative-error bound in TV distance — error grows
only polynomially in the number of steps with a logarithmic factor, not
exponentially.

$$\boxed{\;(\varepsilon^{TV}_{k+1})^2 \;\le\; \mathcal{O}\!\big((\varepsilon^2_{\text{init}} + \varepsilon^2)\, \log^{k+1}(\varepsilon^{-1})\big)\;}$$

---

## 1. Setting

- **State-space model**:
  $x_{t+1} = f(x_t) + \eta_t$, $y_t = h(x_t) + \xi_t$.
- **Goal**: filter $p(x_t \mid y_{1:t})$ for highly nonlinear $f, h$.
- **Online**: observations arrive one at a time; no smoother.

## 2. Method

Alternate **prediction** and **update** steps, each sub-step is a
score-based Langevin sampler:

1. **Prediction** — propagate current particles through $f$ plus
   process noise; score of the prior is re-estimated.
2. **Update** — multiply by likelihood $p(y_{t+1} \mid x)$, sample from
   the product via Langevin using the *sum* of prior-score and
   likelihood-gradient.
3. **Annealing**: the Langevin target is warmed up across sub-iterations
   to improve mixing.
4. **Repeat**.

## 3. Assumptions (verbatim)

- **Assumption 1 (Lipschitz score)**: the posterior score is $\lambda$-
  Lipschitz on $\mathbb{R}^d$.
- **Assumption 2 (Log-Sobolev inequality)**: the posterior satisfies a
  log-Sobolev inequality with constant $C_{\text{LSI}}$.
- **Assumption 3 (Boundedness)**: transition, prediction, and likelihood
  densities and their gradients are bounded.
- **Assumption 4 (Score matching tolerance)**: there exists
  $\Delta \in (0, 1)$ such that
  $$\mathbb{E}\big[\|\nabla \log \hat q - \hat s\|_2^2\big] \le \Delta^2.$$

**No contraction factor $\rho < 1$** in the classical sense. The log-
Sobolev inequality provides the implicit "contraction" mechanism.

## 4. Main Result

**Theorem 3.5 (assimilation error bound)**:

$$\boxed{\;(\varepsilon^{TV}_{k+1})^2 \;\le\; \mathcal{O}\!\big((\varepsilon^2_{\text{init}} + \varepsilon^2)\, \log^{k+1}(\varepsilon^{-1})\big)\;}$$

- $\varepsilon^{TV}_{k+1}$: TV distance between true filter and computed
  filter at step $k+1$.
- $\varepsilon_{\text{init}}$: initial prior shift.
- $\varepsilon$: per-step tolerance (combines score-estimation $\Delta$ +
  Langevin step-size).
- **Error grows polynomially in $k$ with $\log^{k+1}$ factor** — *much*
  better than the exponential blow-up typical of raw particle filters.

### Per-step error decomposition (Theorem 3.3)

Four terms:
1. **Langevin-diffusion convergence** — exponential decay under LSI.
2. **Discretisation error** — linear in step size $h$.
3. **Score estimation error** — proportional to $\Delta$ (Assumption 4).
4. **Prior-propagation error** — quadratic in the previous-step TV.

**Corollary 3.4** (the recursion):
$$(\varepsilon^{TV}_{k+1})^2 \;\lesssim\; C_{\text{LSI}}\, B^4 D^4 \log(\eta \chi^2 / \varepsilon^2)\, (\varepsilon^{TV}_k)^2 \;+\; \varepsilon^2.$$

## 5. Proof Technique

- **Girsanov-type change of measure** bounds the score-approximation
  error's effect on the Langevin path.
- **Log-Sobolev inequality** contracts the Langevin dynamics in
  $\mathrm{KL}$ / TV.
- Recursion (Corollary 3.4) is a **quadratic self-map** on the previous
  TV — analysed by a standard "sum of a geometric + polynomial" style
  argument yielding the $\log^{k+1}$ factor.

## 6. Novelty vs Prior Work

From the paper:

> "We present a novel method for nonlinear assimilation, named score-
> based sequential Langevin sampling (SSLS), within a recursive Bayesian
> framework."
>
> "We analyse the convergence of SSLS in TV-distance under certain mild
> conditions … characterise precisely how the assimilation error is
> affected."
>
> "SSLS yields significant advantages in high-dimensional and nonlinear
> data assimilation, even with only sparse observations."

Against baselines:
- **Auxiliary Particle Filter (APF)**: suffers weight-collapse in
  high-d; SSLS avoids this by re-sampling from a Langevin-targeted
  posterior.
- **Ensemble Kalman Filter (EnKF)**: linear / Gaussian approximation;
  SSLS handles full non-linearity.

## 7. Experiments (tested on)

1. **Linear Gaussian** (1D) — sanity check, §4.1.
2. **Double-well potential** with Langevin dynamics — §4.2.
3. **Lorenz-96** (chaotic, high-d) — §4.3.
4. **Kolmogorov flow** with **sparse observations** — §4.4.

Baselines: APF, EnKF.

## 8. Limitations

- Per-step score re-training is non-trivial; amortised gains only when
  dynamics are slow-varying.
- Log-Sobolev assumption is strong; financial return distributions
  typically don't satisfy LSI uniformly.
- Markovian state — not directly applicable to long-memory financial
  series without modification.

---

## Connections

- **shares proof scaffold with** [[girsanov-path-kl-bound]] — Girsanov
  for the score approximation error; log-Sobolev for Langevin mixing.
- **sequential counterpart to** [[diffusion-doob-matching-inference-alignment]]
  (static alignment) — same ingredients, different control structure.
- **single-round version is** [[conditional-follmer-flow-distribution-learning]] —
  one conditioning = one observation; here we do a *chain* of
  conditionings.
- **contrast with** classical SMC / particle filters: no weight
  collapse, at the cost of needing a good score network.
- **mirrors** [[recursive-chi2-inequality]] in structure — both produce
  a polynomial-in-$k$ error growth via a quadratic self-map.

## Contradictions

_None from the abstract and result. A deeper comparison to EnKF's
explicit variance propagation would be worth doing but no direct
conflict is visible._

## Open questions

- Does SSLS extend to **non-Markovian** state (finance with long memory)?
- What is the **tightest per-step** $\Delta$ budget that still yields
  cumulative stability? This is the same question
  [[recursive-chi2-inequality]] asks about the noise-schedule.
- **Replacing LSI** with a weaker Poincaré-type condition — how much
  do the constants blow up?

## My take / relevance

Less directly tied to our synthetic-augmentation programme than the
other six L2 papers, but it's the **canonical template for sequential
score-based learning**. If our factor-conditional generator ever needs
to update under streaming regime signals (intraday market updates, new
trades arriving), this is the architecture + analysis to clone.

Specifically: the **quadratic-self-map + log factor** machinery in
Corollary 3.4 maps onto our [[recursive-chi2-inequality]] structure. A
cross-pollination exercise: can we re-derive our χ² bound using their
log-Sobolev + Girsanov route? Possibly tighter constants.
