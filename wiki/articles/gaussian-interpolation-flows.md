---
title: "Gaussian Interpolation Flows (arxiv:2311.11475)"
category: "articles"
slug: "gaussian-interpolation-flows"
tags: ["theory", "continuous-normalizing-flows", "interpolation", "lipschitz-velocity", "well-posedness", "huang-group", "jiao-group", "arxiv-2023"]
authors: ["huang-jian", "jiao-yuling"]
refs: ["arxiv:2311.11475"]
links: ["huang-jian", "jiao-yuling", "cnf-convergence-distribution-learning", "conditional-stochastic-interpolation", "conditional-follmer-flow-distribution-learning", "girsanov-path-kl-bound"]
created: "2026-04-17T13:30:00"
updated: "2026-04-17T13:30:00"
---

# Gaussian Interpolation Flows

> **Authors**: Yuan Gao · Jian Huang · Yuling Jiao (see [[huang-jian]], [[jiao-yuling]])
> **arxiv**: 2311.11475 (v2, Nov 2023)
> **Status in our wiki**: L2 deep note — **foundational well-posedness paper**

## TL;DR

The **regularity foundation** for continuous normalizing flows built
from Gaussian-denoising interpolants. Proves well-posedness (unique
solution to the flow ODE), Lipschitz regularity of the velocity field,
and stability under velocity perturbation. Explicit precursor to
[[cnf-convergence-distribution-learning]].

No finite-sample convergence rate — this paper is the *functional-
analytic prerequisite*; rates come in the follow-up 2404.00551.

---

## 1. Setting

- **Target**: a probability measure $\nu$ on $\mathbb{R}^d$.
- **Source**: standard Gaussian $\mu = \gamma_d$.
- **Interpolant** (Definition 18):
  $X_t \stackrel{d}{=} a_t Z + b_t X_1$,
  $Z \sim \gamma_d$, $X_1 \sim \nu$, with schedule
  - $\dot a_t \le 0$, $\dot b_t \ge 0$,
  - $a_0 > 0$, $b_0 \ge 0$, $a_1 = 0$, $b_1 = 1$.
- **Flow ODE** (Definition 25):
  $$\frac{\mathrm{d}X_t}{\mathrm{d}t}(x) = V(t, X_t(x)), \qquad X_0(x) \sim \mu$$
  where the velocity field is the **conditional expectation**:
  $$V(t, x) := \mathbb{E}[\dot a_t Z + \dot b_t X_1 \mid X_t = x]$$
  (Theorem 20).

## 2. Framework Unification

The schedule flexibility covers many existing frameworks as special
cases:

| Existing method | Schedule |
|---|---|
| **VE SDE** (variance-exploding) | $a_t^2 + b_t^2 > 1$, $b_t = 1$ fixed |
| **VP SDE** (variance-preserving) | $a_t^2 + b_t^2 = 1$ |
| **Rectified flow / linear** | $a_t = 1 - t$, $b_t = t$ |
| **Trigonometric interpolant** | $a_t = \cos(\pi t/2)$, $b_t = \sin(\pi t/2)$ |
| **Föllmer flow** | specific Gaussian-score schedule |

All absorbed under one regularity analysis.

## 3. Assumptions (Assumption 2)

Target $\nu$ satisfies **one of**:

1. $\beta$-semi-log-convex AND $\kappa$-semi-log-concave with $\kappa > 0$
   (strong log-concavity).
2. $\beta$-semi-log-convex AND $\kappa$-semi-log-concave with $\kappa \le 0$,
   **on bounded support**.
3. $\nu = \gamma_{a, \sigma^2} \star \rho$ where $\rho$ has support on a
   Euclidean ball of radius $R$ (Gaussian-convolution of a compactly-
   supported measure).
4. $\beta$-semi-log-convex, $\kappa \le 0$, plus $\mathrm{d}\nu / \mathrm{d}\gamma_a$
   is $L$-log-Lipschitz.

**Strong log-concavity** appears only in case (i); the other cases handle
**multimodal** / **bounded-support** / **Gaussian-smoothed** distributions.

## 4. Main Results

### (a) Well-Posedness (Theorem 38)

> "There exists a unique solution $(X_t)_{t \in [0, 1]}$ to the IVP"
> with push-forward $X_t \# \mu \stackrel{d}{=} a_t Z + b_t X_1$.

### (b) Lipschitz Velocity (Proposition 29)

Bounds on $\nabla_x V(t, x)$ under each case of Assumption 2. Under
$\beta$-semi-log-convexity:
$$\nabla_x V(t, x) \succeq \frac{\beta a_t \dot a_t + b_t \dot b_t}{\beta a_t^2 + b_t^2} I_d.$$

### (c) Stability (Proposition 54(ii))

$$W_2^2(Y_1 \# \mu, \nu) \le \frac{e^{2 C_3} - 1}{2 C_3} \int\!\!\int \|V - \tilde V\|^2\, p_t\, \mathrm{d}x\, \mathrm{d}t$$

where $\tilde V$ is any perturbed velocity and $C_3$ is a constant
depending on the Lipschitz bound. **This is the lemma that connects
velocity-estimation error to Wasserstein-distribution error** — used in
every follow-up rate paper.

No explicit $W_2$ *convergence rate* appears in this paper; it stops at
the stability bound.

## 5. Proof Technique

**Key lemma (Lemma 26, covariance representation)**:
$$\nabla_x S(t, x) = \frac{b_t^2}{a_t^4} \mathrm{Cov}(Y \mid X_t = x) - \frac{1}{a_t^2} I_d$$

**Bounding the covariance**:
- **Upper bound**: Brascamp-Lieb inequality under $\kappa$-semi-log-
  concavity.
- **Lower bound**: Cramér-Rao inequality under $\beta$-semi-log-
  convexity.

This avoids moment bounds; instead, **the Hessian of the potential $U$
is leveraged directly** to control the velocity's Lipschitz constant.

## 6. Novelty vs Prior Work

| Reference | This paper's advance |
|---|---|
| Albergo-Vanden-Eijnden (2023) stochastic interpolants | extends to **adaptive starting** ($a_0 > 0, b_0 \ge 0$ vs. $a_0 = 1, b_0 = 0$); proves **well-posedness for broad measure classes** |
| Föllmer flow (general) | covers **multimodal** targets ($\kappa < 0$) explicitly on bounded domains |
| Stochastic interpolants | provides **deterministic ODE derandomisation** via continuity equation |
| Diffusion denoising | **unifies VE, VP, linear, trigonometric** in one framework (Table 1 in paper) |

Key quote:
> "The regularising effect of Gaussian denoising would ensure the
> Lipschitz smoothness of the velocity field… since the standard
> Gaussian distribution is both 1-semi-log-concave and 1-semi-log-
> convex, its convolution… will maintain its high regularity."

## 7. Limitations

- **No sample-level rate**: stops at functional-analytic foundations.
  (Finite-sample rates come in [[cnf-convergence-distribution-learning]].)
- **Assumption 2** excludes heavy-tailed targets (no case covers power-
  law tails without extra smoothing).
- Proofs rely on Hessian conditions — hard to verify practically.

---

## Connections

- **precursor to** [[cnf-convergence-distribution-learning]] — that
  paper adds finite-sample rates on top of this paper's regularity
  foundations.
- **unconditional counterpart to** [[conditional-stochastic-interpolation]]
  — same group extends this to conditional distributions.
- **stability lemma (Prop 54(ii)) is used by** essentially every
  Huang-group / Jiao-group flow-matching paper after 2023 — it's the
  $W_2$-from-velocity-error bridge.
- **tools used in** [[girsanov-path-kl-bound]] overlap (Lipschitz
  velocity → bounded drift for Girsanov).
- **generalises** the schedule families:
  [[conditional-follmer-flow-distribution-learning]] specialises to
  $a_t = \sqrt{1 - t^2}, b_t = t$;
  [[characteristic-learning-one-step-generation]] uses Gaussian-
  smoothed case (3).

## Contradictions

_None; this paper is all infrastructure._

## Open questions

- **Heavy-tail extension**: does Assumption 2 admit a relaxation to
  sub-exponential or power-law tails? None of the four cases currently
  does. Connects to our financial-data concern.
- **Tight stability constant**: the $(e^{2 C_3} - 1)/(2 C_3)$ factor in
  Proposition 54(ii) may be loose. A sharper version would tighten every
  downstream rate.
- **Non-Gaussian base**: can we replace the Gaussian source $\mu$ by
  another log-concave base, and still get the same regularity?

## My take / relevance

This is the **infrastructure paper** we'd rely on implicitly whenever we
claim "our CNF is well-posed and has Lipschitz velocity." Worth citing
even when we don't quote a theorem — it's the reason the follow-up
rate papers *can* be written.

Concrete uses for us:
1. **Choice of $(a_t, b_t)$ schedule**: this paper's framework lets us
   pick freely; Theorem 20 gives the velocity in every case. For
   finance, we might want the *trigonometric* schedule (softer
   endpoints than linear).
2. **Stability lemma (Prop 54(ii))** is directly what we'd cite when
   claiming our generator's distribution $W_2$-error is controlled by
   its velocity $L^2$-error.
3. **Covariance-representation lemma (Lemma 26)** is a technical gem —
   the Brascamp-Lieb / Cramér-Rao sandwich is a nice structural tool
   we might re-use in our collapse analyses.
