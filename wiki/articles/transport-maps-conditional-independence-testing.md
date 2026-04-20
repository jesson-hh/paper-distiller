---
title: "Conditional Independence Testing via Transport Maps (arxiv:2504.09567)"
category: "articles"
slug: "transport-maps-conditional-independence-testing"
tags: ["theory", "conditional-independence", "hypothesis-testing", "transport-maps", "cnf", "distance-correlation", "generative-inference-hybrid", "huang-group", "arxiv-2025"]
authors: ["huang-jian"]
refs: ["arxiv:2504.09567"]
links: ["huang-jian", "cnf-convergence-distribution-learning", "conditional-follmer-flow-distribution-learning", "conditional-stochastic-interpolation"]
created: "2026-04-17T13:30:00"
updated: "2026-04-17T13:30:00"
---

# From Conditional to Unconditional Independence: Testing Conditional Independence via Transport Maps

> **Authors**: Chenxuan He · Yuan Gao · Liping Zhu · Jian Huang (see [[huang-jian]])
> **arxiv**: 2504.09567 (v3, Apr 2025)
> **Status in our wiki**: L2 deep note — **rare generative-into-inference bridge**

## TL;DR

Test the hypothesis $\mathbf{X} \perp \mathbf{Y} \mid \mathbf{Z}$ by:
1. Learning two conditional CNFs that transform $\mathbf{X} \mid \mathbf{Z}$
   and $\mathbf{Y} \mid \mathbf{Z}$ into **$Z$-independent** random
   variables $\xi, \eta$.
2. Running a standard **unconditional** independence test (distance
   correlation) on the transformed $(\xi, \eta)$.

Wins against classical CI tests (KCIT, CDC, FCIT, CCIT, CLZ) up to
$\dim(\mathbf{Z}) = 100$.

This is a **rare case** of using generative models as a tool *for
inference* rather than prediction — worth drawing lessons from.

---

## 1. Setting

- **Null**: $\mathbf{X} \perp \mathbf{Y} \mid \mathbf{Z}$.
- **Alternative**: dependence remains after conditioning.
- **Observations**: i.i.d. $(\mathbf{X}_i, \mathbf{Y}_i, \mathbf{Z}_i)$
  for $i = 1, \ldots, n$.
- **High-dim regime**: $\dim(\mathbf{Z})$ can be large (tested up to
  100) — exactly where classical kernel-based CI tests degrade.

## 2. Method

### The transport-map recipe (Lemma 1)

**Key insight**:
> "Conditional independence in the original space is equivalent to
> independence in the transformed space:
> $\mathbf{X} \perp \mathbf{Y} \mid \mathbf{Z} \iff \xi \perp \eta$"

**Construction**: two conditional CNFs define
$$\hat\xi_i = \mathbf{X}_i + \int_1^0 \hat v_x(t,\,\mathbf{X}_{t,i}^\dagger,\,\mathbf{Z}_i)\,\mathrm{d}t,$$
$$\hat\eta_i = \mathbf{Y}_i + \int_1^0 \hat v_y(t,\,\mathbf{Y}_{t,i}^\dagger,\,\mathbf{Z}_i)\,\mathrm{d}t.$$

That is, **run the reverse conditional flow** to push
$\mathbf{X} \mid \mathbf{Z}$ to a $\mathbf{Z}$-free base noise, same for
$\mathbf{Y}$. Under $H_0$, the transformed $(\xi, \eta)$ become
jointly independent.

### The velocity estimators

Minimise squared velocity-matching losses (Eqs. 2.8-2.9 in the paper) —
deep-neural-network parametrisation.

### The test statistic (Eq. 2.12)

$$\boxed{\;T_n = \mathrm{dcorr}_n^2(\hat\xi, \hat\eta) = \frac{\mathrm{dcov}_n^2(\hat\xi, \hat\eta)}{\mathrm{dcov}_n(\hat\xi, \hat\xi) \cdot \mathrm{dcov}_n(\hat\eta, \hat\eta)}\;}$$

where $\mathrm{dcorr}$ is **distance correlation** computed from
pairwise distances. Alternative: **improved projection correlation**
(IPC) gives comparable performance.

## 3. Assumptions

From Lemmas 2-3:

- **Velocity Lipschitz in space and time**:
  $\|\hat v_x(t, \mathbf{x}, \mathbf{z}) - \hat v_x(s, \mathbf{x}, \mathbf{z})\| \le L_t |t - s|$.
- **Support conditions on $\mathbf{Z}$** to ensure the transport is
  marginal-preserving.
- **Standard nonparametric neural-network approximation** (width
  $\mathcal{W}$, size $\mathcal{S}$, depth $\mathcal{T}$).

## 4. Main Results

**Theorem 5 (test consistency)**:
- Under $H_0$: $T_n \to_p 0$.
- Under $H_1$: $T_n \to_p \zeta = \mathrm{dcorr}(\xi, \eta) > 0$.

**Theorem 6 (p-value consistency)**: permutation-based p-value is
consistent under both hypotheses.

**Asymptotic regime**: $n \to \infty$ with properly scaling network
architecture.

## 5. Proof Technique

- **Consistency route**: approximation error of the CNF velocity →
  residual dependence in $(\hat\xi, \hat\eta)$ → distance-correlation
  bias.
- **Key lemma**: if the velocity approximation error shrinks at a
  standard nonparametric rate, the induced dependence bias shrinks at
  the same rate.
- Combine with **distance-correlation concentration** (classical
  result of Székely-Rizzo) to get type-I / power guarantees.

## 6. Novelty vs Classical CI Tests

From the paper:
> "Our method exhibits superior performance across univariate,
> multivariate, and moderately high-dimensional pairs $\mathbf{X},
> \mathbf{Y}$ in the presence of low or moderately high-dimensional
> confounders $\mathbf{Z}$."

Comparisons:
- **KCIT (kernel-based CI test)**: struggles when $\dim(\mathbf{Z}) > 3$;
  kernel matrix is $O(n^2)$ or worse.
- **CDC (conditional distance correlation)**: power degrades as
  $\dim(\mathbf{Z})$ grows.
- **FCIT, CCIT, CLZ**: various recent neural-net-based tests.
- **FlowCIT (this paper)**: maintains power up to $\dim(\mathbf{Z}) = 100$
  in simulations; avoids explicit kernel matrices.

## 7. Experiments

**4 simulation families, 16 settings**:
- Model 1: univariate $(\mathbf{X}, \mathbf{Y})$,
  $\dim(\mathbf{Z}) = 2$, $n = 500$.
- Model 2: multivariate $(d_X, d_Y, d_Z) = (3, 3, 3)$, $n = 500$.
- Model 3: $(d_X, d_Y, d_Z) = (5, 5, 50)$, $n = 1000$.
- Model 4: $(d_X, d_Y, d_Z) = (50, 50, 100)$, $n = 1000$.

**Finding** (paraphrased):
> "Our proposed test exhibits superior performance across all simulation
> models. Additionally, our method effectively controls the type-I
> error under $H_0$ and achieves the highest statistical power."

**Real data**: Wine-quality (n = 4898) — evaluating whether dimension-
reduction methods preserve sufficiency.

## 8. Limitations

- Requires training **two** CNFs — expensive; but amortised when the
  test is repeated on slightly different data.
- CNF training must succeed — if the conditional flow is mis-trained,
  the test can falsely reject (power + type-I control are both
  affected).
- No explicit finite-sample type-I error bound (only asymptotic).
- Continuous variables only; categorical / mixed types not handled.

---

## Connections

- **uses** conditional CNFs as in
  [[cnf-convergence-distribution-learning]] and
  [[conditional-follmer-flow-distribution-learning]] — the flow is
  plug-and-play.
- **novel direction**: *generative → inference* bridge. Most of the
  [[huang-jian]] and [[jiao-yuling]] line goes generative → prediction;
  this paper does generative → statistical testing.
- **contrast with** classical kernel-based CI testing (KCIT, CDC):
  dimension-scaling advantage.
- **could underwrite** a finance application: test conditional
  independence between two stocks' returns given a factor vector, i.e.,
  **whether factors fully explain the cross-sectional dependence**.

## Contradictions

_None identified._

## Open questions

- **Finite-sample type-I error**: paper only proves asymptotic
  consistency. Non-asymptotic control is important for practitioners.
- **Robustness to CNF mis-training**: how does a slightly wrong flow
  bias the test statistic? A sensitivity analysis is missing.
- **Categorical confounders**: what if $\mathbf{Z}$ has discrete
  components?

## My take / relevance

**Genuinely interesting direction for our wiki.** The *generative-into-
inference* theme is under-developed across our existing entries —
[[jiao-yuling]] and [[huang-jian]] lines both mostly go generative →
prediction, not generative → test.

Concrete action items:
1. Apply FlowCIT to **check factor adequacy** — given a factor model,
   test whether residual returns of two stocks are independent
   conditional on the factors. If the test rejects, the factor model is
   incomplete.
2. Use this logic for **regime identification** — test
   $\text{return} \perp \text{macro-state} \mid \text{regime-indicator}$.
3. Promote a `techniques/` entry titled *"Generative tests of
   statistical hypotheses"* — a new category that sits between our
   `articles/` paper notes and `techniques/` proof tricks.
