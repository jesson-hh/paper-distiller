---
title: "Can a Non-Singular Interpolation Recover $n^{-1/(d+3)}$ for CNFs?"
category: "open-problems"
slug: "cnf-rate-without-time-singularity"
tags: ["theory", "continuous-normalizing-flows", "convergence-rate", "time-singularity", "interpolation-design"]
refs: ["arxiv:2404.00551", "arxiv:2405.05512", "arxiv:2402.01460"]
links: ["cnf-convergence-distribution-learning", "characteristic-learning-one-step-generation", "conditional-follmer-flow-distribution-learning", "girsanov-path-kl-bound", "jiao-yuling"]
created: "2026-04-17T13:00:00"
updated: "2026-04-17T13:00:00"
---

# Can a Non-Singular Interpolation Recover $n^{-1/(d+3)}$ for CNFs?

> **Seeded from**: [[cnf-convergence-distribution-learning]]
> (Gao, Huang, Jiao, Zheng 2024, arxiv:2404.00551) — Remark 2.
>
> **Contrast**: [[characteristic-learning-one-step-generation]] achieves
> the *better* rate $n^{-2/(d+3)}$ in a *one-step* formulation, apparently
> because it avoids the time singularity that degrades the multi-step CNF.
>
> **Status**: open problem (2026-04-17) — the "right" interpolation
> schedule that yields $n^{-1/(d+3)}$ for multi-step CNFs has not, to our
> knowledge, been proposed.

## The Gap

From Remark 2 of [[cnf-convergence-distribution-learning]]:

> *"Without the time singularity of the velocity field, the error would
> be $\tilde O(n^{-1/(d+3)})$. The time singularity reduces the rate to
> $\tilde O(n^{-1/(d+5)})$."*

So the multi-step CNF's headline rate is $n^{-1/(d+5)}$, but a version
**without** the time singularity would be $n^{-1/(d+3)}$ — a $d+2$
improvement in the exponent. This is a **large** gap: at $d = 30$ the
ratio $n^{-1/35}$ vs $n^{-1/33}$ means we need on the order of
$n^{35/33} \approx 1.06 \cdot n$ more samples to match — but
the gap grows with $d$ and the *constants* may differ dramatically.

Meanwhile, the *one-step* characteristic learner
([[characteristic-learning-one-step-generation]]) gets **even better**:
$n^{-2/(d+3)}$, also without time singularity. So it appears that the
time singularity is a specific cost of the *classical Föllmer
interpolation* $X_t = t X_1 + \sqrt{1 - t^2} X_0$, not a fundamental
limit.

## Formal Open Question

> Does there exist a CNF interpolation schedule
> $$X_t = \alpha(t) X_1 + \beta(t) X_0,\qquad t \in [0,1]$$
> with $\alpha, \beta \in C^\infty$ on $[0, 1]$ (no singularity at the
> endpoints) such that the velocity field
> $$v^\star(x, t) = \dot\alpha(t) \mathbb{E}[X_1 \mid X_t = x] + \dot\beta(t) \mathbb{E}[X_0 \mid X_t = x] \cdot \tfrac{\alpha(t)}{\beta(t)}$$
> (or its appropriate Föllmer analogue) is Lipschitz in $(x, t)$
> uniformly on $[0, 1]$ and the full Gao-Huang-Jiao-Zheng
> tri-decomposition yields rate $\tilde O(n^{-1/(d+3)})$?

## Why this might be achievable

1. **Rectified-flow** ($X_t = t X_1 + (1-t) X_0$) has a *constant*
   velocity and no singularity — but has its own regularity issues
   (discontinuous velocity for non-smooth data).
2. **Cosine** ($\alpha(t) = \sin(\pi t/2)$) is smooth everywhere and
   may avoid the Föllmer $1/t$ blow-up at $t \to 0$.
3. **Variance-preserving** schedules
   ($\alpha^2 + \beta^2 = 1$) with different $\alpha$
   may control the velocity gradient uniformly.

## Why it might not be achievable

- **Information-theoretic lower bound**: if the minimax rate for
  estimating a Lipschitz-score density in $d$ dimensions is
  $n^{-1/(d+4)}$ (or similar), then no interpolation can beat it —
  $n^{-1/(d+3)}$ would violate the bound.
- **Tri-decomposition cost**: the velocity-estimation rate
  $(n \bar t^2)^{-1/(d+3)}$ in
  [[cnf-convergence-distribution-learning]] depends on *both* $n$ and
  $\bar t$; removing the singularity lets us use $\bar t = 0$ but might
  break the discretisation analysis.
- **Non-smooth transport**: the optimal transport map between Gaussian
  and a compactly-supported target is generically not $C^\infty$, so
  even a smooth schedule may not yield a globally Lipschitz velocity.

## Sub-questions

### (a) Rectified-flow analogue
Does $X_t = t X_1 + (1 - t) X_0$ give a full tri-decomposition with rate
$n^{-1/(d+3)}$? Some empirical evidence (e.g. from the flow-matching
literature) suggests yes, but no published clean theorem covers the
whole decomposition.

### (b) Matching the one-step rate
Can a multi-step scheme achieve $n^{-2/(d+3)}$ — the one-step
characteristic-learning rate? This would require a *different* Föllmer
analogue where the regression target has better concentration. The
semi-group penalty in
[[characteristic-learning-one-step-generation]] is suggestive.

### (c) Conditional version
Does the conditional analogue of this question have the same answer?
I.e., can [[conditional-follmer-flow-distribution-learning]]'s
$n^{-4/(9(d+d_Y+5))}$ be improved to $n^{-c/(d+d_Y+3)}$ by a non-singular
conditional interpolation?

## Connection to our own work

- A tighter CNF rate **directly improves** the denominator in any χ²
  recursion analysis of [[diversity-collapse-sde-framework]] — our
  collapse threshold would move.
- **Finance-relevant**: at realistic $d$ (e.g., 30-day × 1-channel =
  30 dim), $1/(d+5)$ vs $1/(d+3)$ is a meaningful difference in sample
  requirements for a tolerance $W_2 \le \varepsilon$.
- The result, if obtained, is a **publishable piece** — the
  tri-decomposition technology of
  [[cnf-convergence-distribution-learning]] is the template; we need to
  identify the interpolation family that makes the velocity field
  globally Lipschitz.

## What's needed to solve it

1. **Start with rectified-flow**: specialise the tri-decomposition to
   $\alpha(t) = t$, $\beta(t) = 1 - t$. Does the velocity-estimation
   term improve?
2. **Experimentally check**: train CNFs with different $\alpha, \beta$
   on a Gaussian-mixture benchmark; measure $W_2$ vs $n$; fit the
   exponent.
3. **Lower bound**: for Lipschitz-score compactly-supported targets, is
   $n^{-1/(d+3)}$ achievable or is there a hidden floor?

## Expected difficulty

- Sub-question (a) (rectified-flow analogue): **medium** — the
  tri-decomposition technology transfers; the main work is checking the
  velocity-estimation empirical process.
- Sub-question (b) (match one-step $n^{-2/(d+3)}$): **hard** — not clear
  why multi-step should lose a factor of 2 in the numerator.
- Lower bound: **medium** — minimax rates for Lipschitz-score density
  estimation are classical.

## Current Notes

- Next concrete action: specialise the proofs of Theorem 4.4 in
  [[cnf-convergence-distribution-learning]] to rectified-flow and see
  which step in the current bound uses the $1/t$ singularity. That's
  the leverage point.
