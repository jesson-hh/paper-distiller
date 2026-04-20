---
title: "Wasserstein Generative Regression Framework — a reusable technique"
category: "techniques"
slug: "wasserstein-generative-regression-framework"
tags: ["technique", "conditional-generation", "wasserstein", "dual-objective", "calibration", "leverage", "discriminator-design"]
refs: ["arxiv:2306.15163", "arxiv:2505.07967"]
links: ["wasserstein-generative-regression", "conditional-follmer-flow-distribution-learning", "cnf-convergence-distribution-learning", "conditional-stochastic-interpolation", "interdiff-inter-stock-correlations", "cofindiff-controllable-financial-diffusion", "fts-interdiff-fusion", "synthetic-augmentation-financial-timeseries", "huang-jian", "jiao-yuling"]
created: "2026-04-17T14:00:00"
updated: "2026-04-17T14:00:00"
---

# Wasserstein Generative Regression Framework

> **Origin**: distilled from [[wasserstein-generative-regression]] (Song
> et al. 2023, published JRSSB 2025) — lifted out of a single paper into
> a **reusable training-loss technique** we can plug into other
> generators.
>
> **Use when**: you need a *conditional generator* whose (a) conditional
> mean matches a regression target *and* (b) conditional distribution is
> calibrated (valid prediction intervals, correct tail shape,
> asymmetric features). This is exactly the regime where standard cWGAN
> or pure MSE-on-noise DDPM underperforms.

---

## The problem this technique solves

Most conditional generators optimise **one** of these objectives:

| Path | Objective | What it gets right | What it misses |
|---|---|---|---|
| **Nonparametric regression** | $\min \mathbb{E}\|Y - \mathbb{E}_\eta g(X, \eta)\|^2$ | conditional mean | **distribution shape** — no variance / tails / skew |
| **Classifier-free DDPM** / **cWGAN** | minimise distributional loss (e.g. score-MSE, Wasserstein dual) | distribution shape *sometimes* | **mean drift** — empirical PI coverage often far off (cWGAN on CT-Slices: **48 %** actual vs 95 % nominal) |
| **Flow-matching + velocity MSE** | $\|\hat v - v^\star\|^2$ | velocity (then $W_2$ via Girsanov) | same calibration issue when rolled out |

WGR says: **use both**. One simple additive loss, provable rate on both
objectives simultaneously, no extra theory cost.

---

## The unified loss

Given data $(X_i, Y_i)$ and a stochastic generator $g_\theta(X, \eta)$
with noise $\eta \sim \mathcal{N}(0, I_m)$:

$$\boxed{\;\mathcal{L}_{\text{WGR}}(\theta, \phi) \;=\; \lambda_w\, \mathcal{L}_W(g_\theta, f_\phi) \;+\; \lambda_\ell\, \mathcal{L}_{\text{LS}}(g_\theta) \;}$$

- $\mathcal{L}_W$ = **1-Wasserstein dual** via a 1-Lipschitz critic
  $f_\phi$:
  $$\mathcal{L}_W = \mathbb{E}_{(X, Y)}[f_\phi(X, Y)] \;-\; \mathbb{E}_{X, \eta}[f_\phi(X, g_\theta(X, \eta))].$$
  Forces **distribution matching** $P_{X, g} \approx P_{X, Y}$.

- $\mathcal{L}_{\text{LS}}$ = **plug-in regression**:
  $$\mathcal{L}_{\text{LS}} = \mathbb{E}_{X}\Big\|Y - \tfrac{1}{J}\sum_{j=1}^J g_\theta(X, \eta_j)\Big\|^2.$$
  Forces **conditional mean** to match. $J \ge 16$ per anchor is typical.

- **Interior of the simplex** $(\lambda_w, \lambda_\ell) \in (0, 1)^2$
  gives the framework. The paper's headline result is that no
  λ-rebalancing is needed to get **the same provable rate on both
  objectives**:
  $$n^{-\beta / (2\beta + [3(m+d)]\, \vee\, [2\beta(d+q+1)])}$$

## Why both terms matter — intuition

- **LS term alone** is fine for mean; the conditional distribution is
  unconstrained. A generator satisfying it can collapse to a
  deterministic mapping that ignores $\eta$.
- **Wasserstein term alone** gets distribution-level calibration but
  does not anchor the mean — in practice this is why cWGANs famously
  under-cover prediction intervals despite matching marginals (the
  48 % coverage failure in the paper's CT-Slices experiment).
- **Together**, each regularises the pathology of the other. LS
  prevents PI under-coverage; Wasserstein prevents mean-only collapse.

## Provable guarantees (pointers)

See [[wasserstein-generative-regression]] for:
- **Theorem 1**: excess risk on $\mathbb{E}_\eta g$ (conditional mean).
- **Theorem 2**: $d_{\mathcal{F}_B^1}$ on the joint distribution.
- Both achieve the rate above under Hölder smoothness.
- Rate is **the same** for both targets — not a trade-off.

---

## Three instantiations

The framework is *generator-agnostic*. Below: three concrete ways to plug
in, ranked by implementation effort.

### (I) Direct — flow-matching or rectified-flow generator

If you already have a **one-pass** generator
$g_\theta(X, \eta) = \Phi_\theta(\eta; X)$ (rectified flow, flow
matching, one-step consistency model), WGR plugs in directly:

```
for each batch (X, Y):
    noise = sample Gaussian of size (B, J, m)
    y_gen = g_theta(X, noise)            # (B, J, ...)
    y_mean = y_gen.mean(dim=1)            # (B, ...)
    L_LS = ||Y - y_mean||^2
    critic_out_real = f_phi(X, Y)
    critic_out_fake = f_phi(X, y_gen.flatten_B_J())
    L_W = critic_out_real.mean() - critic_out_fake.mean()
    loss_G = lambda_w * (-L_W) + lambda_ell * L_LS
    # maximise L_W for critic (separate opt step)
```

**Cost**: $J$ generator passes per batch for the LS term. Typical
$J = 8$–$16$.

### (II) Hybrid — standard DDPM + WGR on denoised sample

If you want to keep the DDPM reverse chain (e.g. because you've already
trained one), the clean way is:

```
DDPM loss:  L_DDPM = ||epsilon - epsilon_theta(x_t, t, X)||^2
sample path: pick a small set of timesteps {t_k} (or just t=0 after
             partial denoise via DDIM ~ 5-10 steps)
get hat_x_0 from epsilon_theta at those steps
apply WGR on (X, hat_x_0) vs (X, Y):
    L_LS  = || Y - average over noise-seed of hat_x_0 ||^2
    L_W   = critic(X, Y) - critic(X, hat_x_0)
total: L_DDPM + mu * (lambda_w * (-L_W) + lambda_ell * L_LS)
```

$\mu$ is a small mixing weight ($0.1$–$0.3$); the DDPM term stays
dominant so the reverse-SDE training signal isn't lost.

**Cost**: adds one partial reverse pass per batch + critic forward +
backward. On your setup (InterDenoiser, B=16, L=64), DDIM-5 costs
< 2× a standard DDPM step.

### (III) Distillation — one-step student + WGR

Distil an existing DDPM to a one-step generator (consistency distillation
style), then do (I) on the student. This is heavier but gives the
cheapest inference. Recommended only after (II) is validated.

---

## Instantiation for finance factor-conditional generators

Mapping onto [[fts-interdiff-fusion]] / [[interdiff-inter-stock-correlations]]
pipelines:

- **$X$** = $(m_t, s_t, r_t)$ — market factor, sector factor, regime
  label — all three of M4/M5's factor inputs.
- **$Y$** = $(B, N, L, C)$ panel of log-returns (your current target).
- **$\eta$** = standard Gaussian over the panel + possibly a factor-
  level noise (one noise vector per window is often enough).
- **Generator** $g_\theta$ = either your InterDenoiser used as a DDIM
  sampler (option II) or a re-factored one-step generator (option III).

### Critic design for panel data

- **Panel-symmetric critic**: should be invariant to stock order within
  a panel (permutation equivariance) — same InterBlock structure as
  the generator is the natural choice; take global pool over the N-axis
  at the end.
- **1-Lipschitz enforcement**:
  - **Spectral normalisation** (Miyato 2018) on all critic layers —
    simplest, works out of the box.
  - **Gradient penalty** (Gulrajani WGAN-GP) — more expensive but
    known-tight; recommended when spectral norm leaves numeric slack.
  - **Bounded Lipschitz via layerwise L2 regulariser** — cheap, bias
    toward under-enforcement; use only if you can verify via a
    Lipschitz-monitoring callback.

### Hyperparameter defaults (starting point)

| Hyperparameter | Default | Notes |
|---|---|---|
| $\lambda_w$ | $1.0$ | scale-match LS by eyeballing the two-term ratio early in training |
| $\lambda_\ell$ | $1.0$ | same |
| $J$ (noise samples per anchor for LS) | $8$ | 16 if mean is hard to match |
| Critic update ratio | $5 : 1$ | 5 critic steps per generator step — standard WGAN-GP lore |
| $\mu$ (mixing weight, option II only) | $0.1$ | increase only after DDPM loss has stabilised |
| Lipschitz-enforcement | spectral norm first, add gradient penalty if asymmetry metrics don't improve | |

Start with the hybrid (option II) on a **single milestone** (e.g.
length=64, CSI300, bootstrap one critic). Only scale to CSI800 once the
two-term balance is stable.

---

## Why this specifically helps with **leverage** (the M8 stuck problem)

The leverage effect $\mathrm{corr}(r_t, |r_{t+k}|)$ is a **sign-
sensitive** feature of the joint distribution of returns. Under pure
MSE-on-noise DDPM:

$$\nabla_\theta\, \mathbb{E}\|\epsilon - \epsilon_\theta\|^2 \;=\; -2\,\mathbb{E}[(\epsilon - \epsilon_\theta) \nabla_\theta \epsilon_\theta]$$

The gradient is **symmetric in the sign of noise** — there is no
directional pressure toward asymmetric marginals. Your M8 sign-cond
branch *added* capacity for asymmetry but the loss still didn't
*reward* asymmetric generation; the branch stayed near inactive.

The WGR Wasserstein term, by contrast, operates on the **samples**
directly. A critic with mild non-linearity detects the sign asymmetry
and penalises the generator for failing to reproduce it — directional
gradient flows back into the asymmetric branches (sign-cond, negative-
factor projections, etc.).

Empirical prediction: **WGR + sign-cond > WGR alone > DDPM + sign-cond
(your M8)** on the leverage metric. This is testable within your
existing eval harness.

---

## Common pitfalls

1. **Critic over-powers generator**. Early training, the critic
   dominates; generator collapses to ignore $\eta$. Mitigation: start
   with $\lambda_w$ small, warm up over first 2k steps.
2. **Trivial critic on panel structure**. If the critic easily
   distinguishes (X, Y) from (X, g(X, η)) via obvious artefacts (e.g.
   wrong marginal variance), it never gives useful signal on
   higher-order features. Mitigation: train DDPM to reasonable quality
   *before* adding the WGR terms (hybrid II); or warm-start the
   generator.
3. **LS term with $J = 1$** makes the mean gradient noisy. $J = 1$ is
   equivalent to adding a no-op regulariser; $J \ge 8$ restores proper
   mean-matching pressure.
4. **Spectral-norm false-sense-of-security**. Spectral norm of weight
   matrices bounds the linear Lipschitz of each layer but does **not**
   tightly bound the full-network Lipschitz constant. Add a
   Lipschitz-monitoring callback: sample two random pairs, compute
   $|f(x) - f(x')| / \|x - x'\|$; if > 1.5, switch to gradient penalty.
5. **Mis-scaled two-term loss**. Early on, one term's magnitude is 10×
   the other and gets ignored. Fix: rescale by running-statistics so
   each term contributes roughly equally in the first 1000 steps.

---

## Open questions this framework leaves

- **Heavy-tailed $Y$** (finance!): WGR's rate assumes bounded support.
  Empirically the hybrid (option II) should still work with log-return
  data, but the rate theorem doesn't cover it. Candidate for
  `open-problems/`.
- **Dependent $(X, Y)_{t}$**: the rate assumes i.i.d. We use mixing
  time-series data. The empirical process bound should degrade by a
  mixing-factor (known technique) — but the formal result isn't in
  the paper.
- **Multi-stock panel factorisation**: $q$ in the WGR rate is the
  response dim. For $Y \in \mathbb{R}^{N \times L \times C}$ (your
  panel), $q = N \cdot L \cdot C$ is huge. But because the critic
  exploits panel-permutation symmetry, the *effective* $q$ is smaller.
  Formalising this is open.

---

## Connections

- **direct lift from** [[wasserstein-generative-regression]] — the
  paper is the specific instantiation; this note is the generalised
  recipe.
- **alternative to** the classifier-free-guidance path used in
  [[cofindiff-controllable-financial-diffusion]] and
  [[interdiff-inter-stock-correlations]].
- **compatible with**
  [[conditional-follmer-flow-distribution-learning]] and
  [[conditional-stochastic-interpolation]] as the underlying
  generator (option I) — the LS+W loss replaces their plain velocity-
  regression.
- **applies to** [[fts-interdiff-fusion]] — specifically as a
  methodology pivot after M8's leverage deadlock. See the direction
  doc for the concrete M9 experiment.
- **related framework**: [[jiao-yuling]]'s WDRO work (2505.07967) adds
  *robustness* to WGR's calibration; could be composed if we want
  both calibration and regime-robustness.

## Contradictions

_None identified. This is a clean technique lift._

## Tracking

- [ ] Implement hybrid option (II) on CSI300 InterDiff M1 baseline
      and measure leverage on L=64 windows.
- [ ] If leverage moves toward positive, reproduce on CSI800 and
      compare to M6/M8.
- [ ] Add a Lipschitz monitor + two-term balance logger to the train
      loop.
- [ ] Consider whether WGR can be combined with the Doob-matching path
      from [[diffusion-doob-matching-inference-alignment]] at
      *inference* time, as a post-hoc calibration layer.
