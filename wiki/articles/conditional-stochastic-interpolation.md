---
title: "Conditional Stochastic Interpolation for Generative Learning (arxiv:2312.05579)"
category: "articles"
slug: "conditional-stochastic-interpolation"
tags: ["theory", "conditional-generation", "stochastic-interpolation", "drift-score-estimation", "minimax-rate", "huang-group", "arxiv-2023"]
authors: ["huang-jian"]
refs: ["arxiv:2312.05579"]
links: ["huang-jian", "wasserstein-generative-regression", "gaussian-interpolation-flows", "conditional-follmer-flow-distribution-learning", "cnf-convergence-distribution-learning"]
created: "2026-04-17T13:30:00"
updated: "2026-04-17T13:30:00"
---

# Conditional Stochastic Interpolation for Generative Learning

> **Authors**: Ding Huang · Jian Huang · Ting Li · Guohao Shen (see [[huang-jian]])
> **arxiv**: 2312.05579 (v3, Dec 2023)
> **Status in our wiki**: L2 deep note

## TL;DR

Conditional-generation framework using **stochastic interpolants**
$Y_t = \mathcal{I}(Y_0, Y_1, t) + \gamma(t)\eta$ — more flexible than
pure-ODE flows because the noise channel $\gamma(t)\eta$ adds
expressiveness. Minimax-optimal rate for the drift estimator:

$$\boxed{\;\mathbb{E}_\mathcal{S}\|\hat{\mathbf{b}}_n - \mathbf{b}^\star\|^2 \;\le\; O\!\big(n^{-2\beta/(2\beta + k + d + 1)}\, \log^5 n\big)\;}$$

where $\beta$ = Hölder smoothness, $k$ = conditioning dim, $d$ = response
dim. Parallel work to Jiao's
[[conditional-follmer-flow-distribution-learning]] (the two papers do
not cite each other).

---

## 1. Setting

- **Target**: conditional distribution $p(y \mid x)$ of response
  $y \in \mathbb{R}^d$ given predictor $x \in \mathbb{R}^k$.
- **Interpolant** (Definition with flexibility):
  $$Y_t = \mathcal{I}(Y_0, Y_1, t) + \gamma(t)\eta$$
  - $\mathcal{I}(\cdot, \cdot, t)$: interpolation path satisfying
    $\mathcal{I}(\cdot, \cdot, 0) = Y_0$ (Gaussian start) and
    $\mathcal{I}(\cdot, \cdot, 1) = y_x$ (target).
  - $\gamma(t)$: time-varying noise strength; may be zero (pure ODE
    flow = rectified flow) or non-zero (stochastic).
  - $\eta$: truncated-normal noise.
- **SDE form**: the conditional distribution evolves under a drift
  $\mathbf{b}^\star$ and score $\mathbf{s}^\star$.

## 2. Method

- **Drift loss** (Lemma 2.5, verbatim):
  $$\mathcal{L}_b(Y_0, Y_1, X, \eta) := \int_0^1 \mathbb{E}\big\|\partial_t \mathcal{I}(Y_0, Y_1, t) + \dot\gamma(t)\eta - \mathbf{b}(X, Y_t, t)\big\|^2\, \mathrm{d}t.$$
- **Score loss**:
  $$\mathcal{L}_s(Y_0, Y_1, X, \eta) := \int_0^1 \mathbb{E}\big\|\gamma(t)^{-1}\eta + \mathbf{s}(X, Y_t, t)\big\|^2\, \mathrm{d}t.$$
- **Adaptive diffusion** $u(t)$ in the reverse SDE — prevents score-
  function explosion near $t = 0$ or $t = 1$.

### Interpolation choices (examples)

- **Rectified flow**: $\mathcal{I}(y_0, y_x, t) = (1-t) y_0 + t\, y_x$,
  $\gamma(t) \equiv 0$.
- **Stochastic linear**:
  $\mathcal{I}(\cdot, \cdot, t) = (1-t) y_0 + t\, y_x$,
  $\gamma(t) = \log(t - t^2 + 1)$.
- **More general**: any pair $(\mathcal{I}, \gamma)$ satisfying the
  boundary conditions works — *flexibility is a design feature*.

## 3. Assumptions (numbered)

1. **Assumption 1**: gradient of conditional density $\nabla p(y \mid x)$
   exists and is $L^1$-integrable; conditional expectations of
   interpolant and $Y_t$ are bounded.
2. **Assumption 2**: regularity on drift $\mathbf{b}^\star$ and
   diffusion $u \cdot \mathbf{s}^\star$ — Sobolev-type spaces,
   integrability, polynomial-growth bounds. *Ensures Fokker-Planck
   well-posedness.*
3. **Assumption 3**: noise $\eta$ is truncated-normal with $\|\eta\|_\infty \le B_\eta$.
4. **Assumption 4** (boundary behaviour): $\gamma(t)$ decays
   polynomially at both endpoints,
   $\gamma(t) \ge t^{1/5 - \zeta}$ near $t = 0$ and
   $\gamma(t) \ge (1-t)^{1/5 - \zeta}$ near $t = 1$.

## 4. Main Results

**Theorem 5.6 (non-asymptotic drift bound)**:
$$\mathbb{E}_\mathcal{S}\|\hat{\mathbf{b}}_n - \mathbf{b}^\star\|^2 \;\le\; 722\, d\, B_0^2\, (\lfloor\beta\rfloor+1)^4\, (k+d+1)^{2\lfloor\beta\rfloor + (\beta \vee 1)}\, (UV)^{-4\beta/(k+d+1)} + c_0\, d\, \mathcal{B}^5\, \mathcal{SD}\,\tfrac{\log(\mathcal{S})}{n}$$

**Corollary 4 (minimax rate)**:
$$\boxed{\;\text{drift rate} \;=\; O\!\big(n^{-2\beta/(2\beta + k + d + 1)}\, \log^5 n\big)\;}$$

For the score estimator, the rate is the same drift rate multiplied by
a time-dependent factor $\Gamma(t)$ that blows up near the boundaries
(hence Assumption 4's boundary-decay $\gamma(t)$).

**No explicit $W_2$ bound** on the sampled conditional distribution —
the paper stops at $L^2$-drift-error and score-error. To get a
distribution-level bound, one would combine these with a standard
Girsanov / coupling argument (as
[[conditional-follmer-flow-distribution-learning]] does).

## 5. Proof Technique — **pure empirical process, no Girsanov, no LSI, no Hessian** (verified 2026-04-17)

Proof skeleton for Theorem 5.6:

1. **Lemma 5.2** — excess-risk decomposition into stochastic +
   approximation components.
2. **Theorem 5.3** — stochastic error bounded via Rademacher complexity
   of the deep-ReLU class; Assumption 4 ($\gamma(t)$ polynomial decay)
   controls the boundary factor $t^{-(1-5\zeta)}$.
3. **Theorem 5.5** — approximation error via Yarotsky-style bounds on
   Hölder functions.
4. **Combine** — plain addition gives Theorem 5.6.

### What the proof does **NOT** use (this is the key finding)

Verified by tracing the proof chain — **none of the following appear**:

- **No Girsanov / change of measure**: the argument is purely
  statistical learning, not stochastic-calculus.
- **No log-concavity, no LSI**: no functional inequalities on the
  conditional density.
- **No Hessian bounds**: the proof works with Hölder smoothness
  (Definition 5.4) and never invokes $\nabla^2 \log p$.
- **No $W_2$ bound**: the result is purely on drift $L^2$ error.
  *This gap is precisely the target of
  [[unified-conditional-generation-rate]].*

Assumptions 1-4 as stated are all the regularity the proof touches.

### What Assumption 4 actually does

Not for $W_2$ (there is no $W_2$ bound); instead it controls
$\mathbb{E}[\|\nabla \log p_t\|^2]$ near the endpoints — without the
polynomial $\gamma(t) \ge t^{1/5 - \zeta}$ decay, the factor
$t^{-(1-5\zeta)}$ in Theorem 5.3 would be unbounded.

## 6. Novelty vs Prior Work

**vs. Albergo-Boffi-Vanden-Eijnden (2023)** (unconditional stochastic
interpolants):
> "They also establish the connection between the velocity field and
> the distribution field, along with providing an explicit form of the
> score function. However, the proof techniques rely on strict
> assumptions on the smoothness of density functions…"

The paper claims:
- Extends to **conditional** distributions (adds covariate $X$).
- **Weaker regularity**: only gradient integrability (Assumption 1)
  rather than density smoothness.
- **Adaptive diffusion** $u(t)$ stabilises score-function explosion.
- **Explicit boundary behaviour** (Theorem 2.4) — rarely made explicit
  in prior work.

## 7. Comparison to Jiao's Conditional Föllmer Flow (2402.01460)

**The two papers do not cite each other** — developed in parallel:

| Aspect | Jiao's Conditional Föllmer ([[conditional-follmer-flow-distribution-learning]]) | This paper (Huang's CSI) |
|---|---|---|
| Interpolant | $X_t = t X_1 + \sqrt{1-t^2} X_0$ (specific) | $Y_t = \mathcal{I}(Y_0, Y_1, t) + \gamma(t)\eta$ (flexible) |
| Noise channel | deterministic ODE | optionally stochastic via $\gamma(t)$ |
| Bound target | $\mathbb{E}_y[W_2^2]$ on conditional distribution | $L^2$ drift/score error |
| Rate | $n^{-4/(9(d + d_Y + 5))}$ | $n^{-2\beta/(2\beta + k + d + 1)}\,\log^5 n$ |
| Assumption on score | $\alpha$-bounded Hessian of $-\log p$ | $L^1$-integrable gradient |
| Main use | W₂ coupling to distribution | drift/score estimation |

**Key difference**: Jiao's analysis delivers a *distributional* bound;
this paper's analysis is *$L^2$-on-drift*. These complement rather than
conflict — combining them would give a tighter end-to-end result.

## 8. Limitations

- **No explicit $W_2$-to-target conditional distribution bound.** This is
  *the* gap that [[unified-conditional-generation-rate]] exists to fill.
  The drift $L^2$ rate is already strong; a coupling lift (e.g. via
  Jiao's Proposition 1 from
  [[conditional-follmer-flow-distribution-learning]]) would close it.
- Rate exponent $2\beta/(2\beta + k + d + 1)$ is *minimax-optimal* per
  Corollary 4 — but only **for Hölder-smooth drifts**; rough-drift
  extensions are open.
- $\log^5 n$ factor is large; constants in $(k + d + 1)^{2\beta + \text{etc.}}$
  also grow fast.
- Optimisation is not analysed.

---

## Connections

- **parallel work to** [[conditional-follmer-flow-distribution-learning]]
  — same problem, different analysis target.
- **unconditional counterpart is** [[gaussian-interpolation-flows]] —
  same Huang-group lineage; conditional version adds $X$.
- **flagship for** [[huang-jian]]'s stochastic-interpolation line —
  followed up by [[huang-jian]]'s 2025 SDR extension (arxiv:2512.18971).
- **uses tools from** [[wasserstein-generative-regression]] in spirit
  (Wasserstein / Hölder machinery).

## Contradictions

- **Rate comparison with Jiao's Conditional Föllmer**: this paper gets
  $n^{-2\beta/(2\beta + k + d + 1)}$ on drift, Jiao gets
  $n^{-4/(9(d + d_Y + 5))}$ on $W_2^2$. These are **not directly
  comparable** (different quantities, different assumption sets) — no
  genuine contradiction but users need to cite carefully.

## Open questions

- **$W_2$ bound**: can the drift/score rates here be lifted to a
  distributional bound via Girsanov? Candidate for `techniques/`.
- **Adaptive $\gamma(t)$ selection**: paper assumes $\gamma(t)$ is
  prescribed. What's the optimal $\gamma$ for a given target class?
- **Heavy tails**: Assumption 3 restricts to truncated noise — can
  this be relaxed to unbounded sub-Gaussian or sub-exponential $\eta$?

## My take / relevance

The **stochastic-vs-deterministic interpolant choice** is a genuinely
useful degree of freedom for finance. Noiseless ODE flows (rectified
flow, Föllmer flow) struggle to represent multi-modal financial
distributions; the $\gamma(t) \eta$ term gives the right flexibility.

Concrete action items:
1. For our [[fts-interdiff-fusion]], try **stochastic-linear** with
   $\gamma(t) = \log(t - t^2 + 1)$ as a baseline; compare against
   deterministic rectified flow on calibration.
2. The adaptive $u(t)$ diffusion is essentially a *learned noise
   schedule* — we should check whether it correlates with the *realised
   volatility schedule* of the market (intuitive guess: it should).
3. Extension line: write the "adaptive $\gamma(t)$ optimisation" paper
   — instead of prescribed $\gamma$, learn it end-to-end.
