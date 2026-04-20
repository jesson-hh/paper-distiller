---
title: "Deep Conditional Distribution Learning via Conditional Föllmer Flow (arxiv:2402.01460)"
category: "articles"
slug: "conditional-follmer-flow-distribution-learning"
tags: ["theory", "conditional-generation", "follmer-flow", "ode-generative", "end-to-end-analysis", "jiao-group", "arxiv-2024"]
authors: ["jiao-yuling"]
refs: ["arxiv:2402.01460"]
links: ["jiao-yuling", "cnf-convergence-distribution-learning", "characteristic-learning-one-step-generation", "girsanov-path-kl-bound", "interdiff-inter-stock-correlations", "fts-interdiff-fusion", "cofindiff-controllable-financial-diffusion"]
created: "2026-04-17T12:00:00"
updated: "2026-04-17T12:30:00"
---

# Deep Conditional Distribution Learning via Conditional Föllmer Flow

> **Authors**: Jinyuan Chang · Zhao Ding · Yuling Jiao · Ruoxuan Li · Jerry Zhijian Yang (see [[jiao-yuling]])
> **arxiv**: 2402.01460 (v3, Feb 2024)
> **Status in our wiki**: L2 deep note — **template for factor-conditional generation**

## TL;DR

ODE-based generative model for $p(y \mid x)$ via a **Conditional Föllmer
Flow**. First *end-to-end* convergence analysis for ODE-based conditional
distribution learning, with a clean rate
$\mathbb{E}_y[W_2^2(\hat p(\cdot|y), p(\cdot|y))] = \tilde O(n^{-4/(9(d+d_Y+5))})$.

This is the **母体** for our factor-conditional generation pipeline —
conditioning dimension $d_Y$ sits symmetrically with target dimension
$d$ in the exponent.

---

## 1. Setting

- **Target**: $\pi_1(x \mid y)$ — conditional distribution of response
  $x \in [0,1]^d$ given predictor $y \in [0, B_Y]^{d_Y}$.
- **Samples**: $n$ i.i.d. pairs $(x_i, y_i)$ from the joint.
- **Flow**: unit-time ODE on $[0,1]$, starting from $X_0 \sim
  \mathcal{N}(0, I_d)$ and ending at $X_1 \sim \pi_1(\cdot \mid y)$.
- **Coupling**: interpolation path
  $$X_t = tX_1 + \sqrt{1 - t^2}\, X_0.$$

## 2. Method

- **Conditional Föllmer velocity** (from the interpolation):
  $$v_F(x, y, t) := \frac{x + s(x, y, t)}{t}, \quad t \in (0, 1]$$
  where $s$ is the conditional score of the marginal.
- **Training loss** (equation from paper):
  $$\mathcal{L}(v) = \frac{1}{T}\int_0^T \mathbb{E}\Big[\big\|(X_1 - \tfrac{t}{\sqrt{1-t^2}} X_0) - v(X_t, Y, t)\big\|_2^2\Big]\,\mathrm{d}t.$$
- **Architecture**: velocity net $\hat v(x, y, t)$ — **conditioning is
  via the extra input argument**, not a specified attention/concatenation
  choice. The analysis requires only that the net can jointly approximate
  a Lipschitz-in-$y$ function.

## 3. Assumptions (verbatim)

1. **Assumption 1 (Bounded predictor)**: $Y$ takes values in
   $[0, B_Y]^{d_Y}$.
2. **Assumption 2 (Bounded response)**: $\pi_1(x \mid y)$ is supported on
   $[0, 1]^d$ for every $y$.
3. **Assumption 3 (Lipschitz conditional score)**:
   $-\alpha I \preceq \nabla_x^2 U(x, y) \preceq \alpha I$ with $\alpha > 1$
   (i.e. the log-density is $\alpha$-semi-log-concave / log-convex).
4. **Assumption 4 (Lipschitz in conditioning)**: the velocity $v_F(x, y, t)$
   is $\omega$-Lipschitz in $y$.

## 4. Main Result

**Theorem 2 (main bound)**:
$$\boxed{\;\mathbb{E}_{y \sim \pi(y)}\big[W_2^2(\tilde \pi_T^y(n),\; \pi_1^y)\big] = \tilde O\!\left(n^{-\frac{4}{9(d + d_Y + 5)}}\right)\;}$$

- **$d$ and $d_Y$ enter symmetrically** in the exponent — no magic "only
  the larger one matters". A key practical consequence: high
  conditioning dimension (many factor signals) degrades the rate at the
  same speed as high target dimension.
- $\tilde O$ hides polylogarithmic factors in $n$.
- Rate is *worse* than the unconditional CNF's $\tilde O(n^{-1/(d+5)})$
  (see [[cnf-convergence-distribution-learning]]) — the conditioning
  costs a $\times 4/9$ factor in the exponent.

## 5. Proof Technique — **coupling, not Girsanov** (verified 2026-04-17)

**Important correction**: an earlier version of this note said the proof
used Girsanov / path-KL. **That was wrong.** The actual proof uses
**coupling via Lipschitz pushforward**, a strictly cleaner route that
does *not* need LSI or Talagrand T2.

The chain (4 components) matches
[[cnf-convergence-distribution-learning]] closely:

1. **Velocity-field estimation error** (Theorem 4): $L^2$-type bound
   on $\int_0^T \|\hat{\mathbf v} - \mathbf v_F\|_{L^2(\pi_t)}^2\,dt$.
2. **Continuous-ODE lift** (**Proposition 1**, Appendix D.1): converts
   velocity $L^2$ error to $W_2^2$ via the **Lipschitz pushforward of
   the flow map**.
3. **Discretisation error** (Lemma 5): $\tilde O(e^{\gamma_1} \max_k (\Delta t_k)^3)$.
4. **Early-stopping error** (Lemma 6): $T(n) = 1 - n^{-4/[9(d+d_Y+5)]}$.

### The lift mechanism (Proposition 1 — this is what we'd reuse)

If velocity error is $\varepsilon$ then
$$W_2^2\big(\pi_T^{\mathbf y},\,\hat \pi_T^{\mathbf y}\big) \le C \cdot \varepsilon^2,\qquad C = e^{2\sqrt{d}\,\gamma_1}$$
where $\gamma_1$ is the **Lipschitz constant of the velocity
$\mathbf v_F(\cdot, \mathbf y, t)$ in $\mathbf x$** (Lemma 1 in the
paper).

The constant $C$ is **exponential in ambient dim $d$** (via
$\sqrt{d}\,\gamma_1$). Rate exponent is clean; absolute constants are
loose.

### Where Assumption 3 (bounded Hessian) is actually used

Verified from the proof — Assumption 3 enters in **exactly 3 places**:

1. **Lemma 1 (velocity Lipschitz)**: $-\alpha I \preceq \nabla^2 U \preceq \alpha I$
   gives $\|\nabla^2 \log \pi_t\| \le \alpha$ → score is $\alpha$-Lipschitz
   → $\mathbf v_F$ is $\gamma_1$-Lipschitz with $\gamma_1 = \zeta(\alpha, d)$ polynomial.
2. **Appendix E.1 (ODE well-posedness)**: bounded Hessian gives unique
   solution.
3. **Proposition 1 (coupling propagation)**: the constant
   $C = e^{2\sqrt d\,\gamma_1}$ propagates — bigger $\gamma_1$ means
   bigger amplification.

*Not used* in: early stopping (Lemma 6), discretisation (Lemma 5),
generalisation (Theorem 4). Those depend only on $\gamma_1, \gamma_2, \gamma_3$
constants — which could in principle be obtained without the Hessian
bound, through any other Lipschitz-velocity route. **This is the leverage
point for [[unified-conditional-generation-rate]].**

## 6. Novelty vs Prior Work

Direct quote from the paper:

> "(ii) We prove the convergence property of the proposed method,
> accompanied with a comprehensive end-to-end error analysis, which
> represents the **first study in the field of ODE-based conditional
> distribution learning**."

Against other approaches:
- Classifier-free guidance for SDE diffusion (as used in
  [[cofindiff-controllable-financial-diffusion]]): *empirical*; no
  end-to-end rate.
- Unconditional CNF ([[cnf-convergence-distribution-learning]]): this
  paper extends it to conditional by carrying $y$ through every step
  of the decomposition.

## 7. Limitations

- Rate degrades with $d_Y$ — bad news for high-factor finance settings
  where we want to condition on 30–50 factors simultaneously.
- The $(1 - T)^{-7/2}$ blow-up means you cannot push the integration
  endpoint too close to 1 without destroying the rate.
- The analysis treats the velocity net as a generic Lipschitz-class
  approximator; does **not** instruct on attention vs. FiLM vs. cross-
  attention. So the theoretical guarantee covers concatenation /
  additive conditioning but specific architectures would need case
  analysis.
- ERM is assumed achieved; optimisation-landscape punted.
- **Heavy tails**: bounded Hessian Assumption 3 excludes heavy-tailed
  conditional densities (finance). Sub-problem 2 of
  [[unified-conditional-generation-rate]] asks whether this can be
  replaced by a weaker condition — verified 2026-04-17 that the
  proof's use is *confined to* Lemma 1, App E.1, and Prop 1, so the
  answer is a plausible "yes".
- **Exponential constant** $C = e^{2\sqrt d\,\gamma_1}$ in Prop 1 —
  rate exponent is clean but absolute constants are loose in $d$.

---

## Connections

- **provides the lift lemma** for [[unified-conditional-generation-rate]]
  — **Proposition 1** is exactly the drift-$L^2$ → $W_2^2$ bridge we need,
  provided we can establish velocity-Lipschitz $\gamma_1$ without going
  through the Hessian-bounded route.
- **conditional extension of** [[cnf-convergence-distribution-learning]]
  — the unconditional rate is $n^{-1/(d+5)}$; conditioning gives
  $n^{-4/(9(d+d_Y+5))}$, so the conditioning cost is a
  $\times 4/9$ shrinkage in the exponent's numerator.
- **theoretical underlay for** [[interdiff-inter-stock-correlations]] —
  factor-conditional sampling. Our Stage A (market factor) and Stage B
  (industry factor) fit the bounded-conditioning assumption.
- **companion to** [[cofindiff-controllable-financial-diffusion]]: same
  goal (controllable generation), different technique (ODE flow vs
  SDE diffusion), *this one has the rate*.
- **template for** [[fts-interdiff-fusion]] — we can plug in
  $d = $ (series length), $d_Y = $ (pattern + factor embedding dim)
  to estimate the sample size we need for a target W₂.

## Contradictions

_None identified._

## Open questions

- **Attention-based conditioning**: does the rate tighten when $y$
  enters via cross-attention rather than concatenation? The analysis
  would need to be re-run with attention-class approximation bounds.
- **Hierarchical conditioning** (market → industry → stock): does the
  composition analysis compound linearly or multiplicatively in error?
- **Rate in intrinsic $d_Y^\star$**: can we escape the $d_Y$ curse under
  a low-intrinsic-dim factor assumption (most financial factors live on
  a low-d regime manifold)?

## My take / relevance

**Single most important Jiao paper for our factor-conditional
generation work.** The rate $\tilde O(n^{-4/(9(d + d_Y + 5))})$
gives us a *concrete, computable* sample-size estimate for any
factor-conditional experiment.

Quick calculation for our CSI300 setup (target dim $d \approx 30$
days × 1 log-return, conditioning $d_Y \approx 5$ factors):
$d + d_Y + 5 \approx 40$ ⇒ rate $n^{-4/360} \approx n^{-0.011}$ —
**extremely slow** in the worst case. Two possibilities:
1. Constants are tight and we really need $n \to \infty$.
2. The exponent is pessimistic; the paper's empirical rate is faster.

Next action: **run the paper's experiments to benchmark actual rate
versus theoretical**, then decide whether to adopt the analysis wholesale
or search for a tighter version.
