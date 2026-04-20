---
title: "Convergence of Continuous Normalizing Flows for Learning Probability Distributions (arxiv:2404.00551)"
category: "articles"
slug: "cnf-convergence-distribution-learning"
tags: ["theory", "continuous-normalizing-flows", "ode-generative", "wasserstein-rate", "tri-decomposition", "jiao-group", "arxiv-2024"]
authors: ["jiao-yuling"]
refs: ["arxiv:2404.00551"]
links: ["jiao-yuling", "characteristic-learning-one-step-generation", "conditional-follmer-flow-distribution-learning", "latent-schrodinger-bridge-diffusion", "girsanov-path-kl-bound", "recursive-chi2-inequality", "diversity-collapse-sde-framework"]
created: "2026-04-17T12:00:00"
updated: "2026-04-17T12:30:00"
---

# Convergence of Continuous Normalizing Flows for Learning Probability Distributions

> **Authors**: Yuan Gao · Jian Huang · Yuling Jiao · Shurong Zheng (see [[jiao-yuling]])
> **arxiv**: 2404.00551, Mar 2024
> **Status in our wiki**: L2 deep note — **foundational rate we cite**

## TL;DR

Non-asymptotic $W_2$ rate for CNFs trained via velocity-regression
(flow-matching-style), with the signature Jiao-group tri-decomposition.

$$\boxed{\; \mathbb{E}\,W_2(\hat \nu_{1 - \bar t}, \nu) \;=\; \tilde O\!\big(n^{-1/(d+5)}\big) \;}$$

The $1/(d+5)$ exponent comes from a time-singularity in the velocity
field; without the singularity it would be $1/(d+3)$. This is
**the reference rate** we plug into our collapse analyses when the
generator is a CNF.

---

## 1. Setting

- **Target**: $\nu$ on $\mathbb{R}^d$ — estimated from $n$ i.i.d. samples.
- **Estimator**: CNF $\phi_\theta: \mathbb{R}^d \to \mathbb{R}^d$;
  integrate $\dot x(t) = v_\theta(x(t), t)$ from $t = 0$ ($\mathcal{N}(0,I)$)
  toward $t = 1$ (target).
- **Training**: velocity regression against a flow-matching target.

## 2. Method

- **Deep ReLU velocity network**
  $v_\theta(x, t): \mathbb{R}^d \times [0,1] \to \mathbb{R}^d$
  trained by MSE on an empirical matching objective.
- **Sampling**: integrate the ODE with a numerical scheme; the analysis
  tracks this step-size as an explicit source of error.
- **Early stopping**: integrate up to $1 - \bar t$, not all the way to
  $t = 1$ — the final time $\bar t$ is an explicit hyper-parameter that
  must be tuned.

## 3. Assumptions

- **Assumption 2 (target class)**: $\nu$ "either has a bounded support,
  is strongly log-concave, or is a finite or infinite mixture of Gaussian
  distributions."
- **Semi-log-concavity**: the log-density potential $U$ satisfies a
  Hessian condition $-\kappa I \preceq \nabla^2 U \preceq \beta I$ —
  equivalent to Lipschitz score.
- **No explicit Hölder / Sobolev assumption** — this is essentially a
  Lipschitz-score world.

## 4. Main Result

**Theorem 1.2 / 4.4** (informal, then formal):

$$\boxed{\;\mathbb{E}\,W_2(\hat p_{1-\bar t},\; p_1) \;=\; \tilde O\!\big(n^{-1/(d+5)}\big)\;}$$

where $\tilde O$ hides $\mathrm{polylog}(n)$ factors.

**Remark 2 from the paper** (important for future use):
> "Without the time singularity of the velocity field, the error would be
> $\tilde O(n^{-1/(d+3)})$. The time singularity reduces the rate to
> $\tilde O(n^{-1/(d+5)})$."

So: the exponent is **worse by $d + 2$** in the denominator because of
the $1/t$ singularity in the Föllmer drift at $t \to 0$.

### Network prescription (Theorem 4.4)

Width, depth, size, and network Lipschitz constants are explicitly
prescribed:

- **Depth** $\mathtt{D} \asymp L \log L$
- **Width** $\mathtt{W} \asymp \bar t^{-2} (NL)^{2/d} N \log N$
- **Size** $\mathtt{S} \asymp \bar t^{-2} (NL)^{2/d} (N \log N)^2 L \log L$
- **Lipschitz in $x$**: $L_x \asymp \log \log n$
- **Lipschitz in $t$**: $L_t \asymp \bar t^{-2} \log \log n$
- With $L \asymp \log \log n$

Practically: the network grows as $n^{d/(2d+10)}$ in width — the width
part of the rate is carried by this prescription.

## 5. Proof Technique — the Tri-Decomposition

The paper explicitly names and bounds three terms (equation 4.1):

$$W_2(\hat p_{1-\bar t}, p_1) \;\le\; \underbrace{W_2(\hat p_{1-\bar t}, \tilde p_{1-\bar t})}_{\text{discretisation}} \;+\; \underbrace{W_2(\tilde p_{1-\bar t}, p_{1-\bar t})}_{\text{velocity estimation}} \;+\; \underbrace{W_2(p_{1-\bar t}, p_1)}_{\text{early stopping}}$$

Individual rates:

| Component | Rate |
|---|---|
| Discretisation | $O(\bar t^{-2}\, \Upsilon)$ with $\Upsilon \lesssim n^{-3/(d+5)}$ |
| Velocity estimation | $(n \bar t^2)^{-1/(d+3)} \mathrm{polylog}(n) \log(1/\bar t)$ |
| Early stopping | $\bar t$ |

Optimal $\bar t \asymp n^{-1/(d+5)}$ balances the velocity-estimation and
early-stopping terms.

This decomposition is **the template** we now recognise as the
Jiao-group signature; also appears in [[conditional-follmer-flow-distribution-learning]]
and [[characteristic-learning-one-step-generation]].

## 6. Novelty vs Prior Work

- Previous CNF analyses were asymptotic or treated only the velocity
  estimation; this is the **first fully non-asymptotic, tri-decomposed
  $W_2$ rate** for CNFs.
- Diffusion literature (`cite:benton-2024-d-linear-diffusion`,
  `cite:chen-2023-low-d-score-approximation`): comparable flavour but
  for reverse-SDE, not deterministic ODE. Different technical route
  (path-KL / Girsanov for diffusion vs. direct coupling here).

## 7. Limitations

- **Bounded support / log-concave / Gaussian mixture** — leaves out
  heavy-tailed targets. **Financial returns violate this**. Extending
  to sub-exponential / power-law tails is wide open.
- **ReLU only** — other activations (RePU, SiLU, GELU) are not
  prescribed; presumably the analysis goes through for any
  Lipschitz-class activation but constants change.
- **ERM assumed achieved** — optimisation-landscape analysis is punted.
- Cannot directly compare with the *conditional* rate
  [[conditional-follmer-flow-distribution-learning]]; the conditional
  paper uses a different Hessian-condition assumption (α-bounded rather
  than the semi-log-concavity here), so the two rates aren't directly
  addable.

---

## Connections

- **supplies the reference rate for** [[recursive-chi2-inequality]] at
  each generation when the generator is a CNF — plug in
  $\varepsilon_\text{gen} = n^{-1/(d+5)}$.
- **sibling of** [[characteristic-learning-one-step-generation]] (same
  group) — **one-step** version with $n^{-2/(d+3)}$ rate (strictly
  better because no time singularity!); consider why.
- **single-round version of** [[conditional-follmer-flow-distribution-learning]] —
  adding conditioning degrades the exponent by $\times 4/9$ and adds
  $d_Y$ to the denominator.
- **latent extension is** [[latent-schrodinger-bridge-diffusion]] —
  rate becomes $n^{-1/(6(d^* + 3))}$ in the latent case.
- **supports** [[synthetic-augmentation-financial-timeseries]] baseline
  rate analysis on light-tailed synthetic targets.

## Contradictions

_None. Result is consistent with the broader diffusion-rate literature._

## Open questions

- **Heavy tails**: extend to sub-exponential / power-law tails. Critical
  for finance. Candidate for `open-problems/`.
- **Time-singularity removal**: the $d + 5$ vs $d + 3$ gap is caused by
  the $1/t$ singularity. Can a different interpolation schedule avoid
  this and recover $n^{-1/(d+3)}$ while keeping the Föllmer analysis?
- **Co-training of velocity and integrator**: current analysis treats the
  integrator as external; what happens when the integrator is learned
  (e.g. consistency distillation)?
- **Sharpness**: is $1/(d+5)$ minimax-optimal for this problem class?
  Paper does not match a lower bound.

## My take / relevance

One of the cleanest provable rates for a practically-used generator.
Use this as:

1. **The citable rate** in the references of
   [[diversity-collapse-sde-framework]] — every collapse argument
   needs an input rate; this is a clean candidate.
2. **The template tri-decomposition** for our own analyses — when we
   write the χ² version of a flow-based rate, we mirror this three-term
   split.
3. **The "worst case" baseline** to improve against. The $d + 5$
   exponent is *slow* in high dimension ($d = 30$ → $n^{-1/35}$).
   Whatever we invent for finance needs to beat this on a matched
   assumption set — or weaken the assumptions in return.
