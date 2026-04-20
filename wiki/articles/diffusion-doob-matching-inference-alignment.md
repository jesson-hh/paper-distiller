---
title: "Diffusion Inference-Time Alignment via Doob's Matching (arxiv:2601.06514)"
category: "articles"
slug: "diffusion-doob-matching-inference-alignment"
tags: ["theory", "diffusion", "controllable-generation", "doob-h-transform", "inference-time-steering", "intrinsic-dim-adaptive", "jiao-group", "arxiv-2026"]
authors: ["jiao-yuling"]
refs: ["arxiv:2601.06514"]
links: ["jiao-yuling", "cofindiff-controllable-financial-diffusion", "girsanov-path-kl-bound", "score-contamination-lemma", "synthetic-augmentation-financial-timeseries", "score-based-sequential-langevin-sampling"]
created: "2026-04-17T12:00:00"
updated: "2026-04-17T12:30:00"
---

# Inference-Time Alignment for Diffusion Models via Variationally Stable Doob's Matching

> **Authors**: Jinyuan Chang · Chenguang Duan · Yuling Jiao · Yi Xu · Jerry Zhijian Yang (see [[jiao-yuling]])
> **arxiv**: 2601.06514, Jan 2026
> **Status in our wiki**: L2 deep note

## TL;DR

Align a **pre-trained, frozen** diffusion to a target distribution via a
guidance drift derived from **Doob's h-transform**, with guidance
estimated by a **gradient-regularised least-squares regression** that
gives non-asymptotic rates and is stable under model mismatch.

Headline rate for the guidance estimator (Theorem 5.3):

$$\boxed{\;\mathbb{E}\Big[\|\nabla \log \hat h_t^\lambda - \nabla \log h_t^\star\|_{L^2(p_{T-t})}^2\Big] \;\le\; C\, \sigma_{T-t}^{-8}\, n^{-2/(d+8)}\, \log^4 n\;}$$

and in Theorem 5.9 this adapts to the **intrinsic dimension** $d^\star$
under a low-dim subspace assumption.

---

## 1. Setting

- **Reference diffusion**: pre-trained, frozen — parametrises a prior
  over $x_0$ via reverse SDE.
- **Target**: distribution $q$ that is a tilted version of the reference:
  $q \propto p_{\text{ref}}\, e^{r(x)}$ for some reward / weight
  function $r$ (or equivalently $w = e^r$).
- **Hard constraint**: no re-training of the score network. Only modify
  the reverse-SDE drift at inference.

## 2. Method

- **Doob's h-transform** (background): tilting the reverse dynamics by a
  weight induces a modified drift
  $\text{drift}_q = \text{drift}_{\text{ref}} + \sigma^2 \nabla \log h$,
  where $h_t(x) = \mathbb{E}[w(x_0) \mid x_t = x]$ under the reference.
- **Variationally stable Doob's matching**:
  - Express $\nabla \log h$ as the gradient of an implicitly defined
    least-squares problem.
  - Estimate $h$ and $\nabla h$ **jointly** via a regression loss
    **augmented with gradient regularisation** — this is the
    "variationally stable" part.
  - Plug the learned $\nabla \log \hat h$ into the reverse SDE at
    inference.

## 3. Assumptions (verbatim)

1. **Assumption 1 (bounded support)**: support of $q_0$ is a compact
   subset of $\{x_0 \in \mathbb{R}^d : \|x_0\|_\infty \le 1\}$.
2. **Assumption 2 (bounded weight)**: constants
   $0 < \underline{B} < 1 < \bar{B} < \infty$ with
   $\underline{B} \le w(x) \le \bar{B}$ for all $x \in \mathrm{supp}(q_0)$.
3. **Assumption 3 (reference score error)**:
   $$\frac{1}{T}\sum_{k=0}^{K-1} h\,\mathbb{E}^{\mathbb{P}}\Big[\|\hat s(kh, X^\leftarrow_{kh}) - \nabla \log p_{T-kh}(X^\leftarrow_{kh})\|_2^2\Big] \le \varepsilon_{\text{ref}}^2.$$

The bounded-weight condition is what makes "variational stability" work
— unbounded rewards blow up the bound.

## 4. Main Results

**Theorem 5.3 (guidance estimation rate)**:
$$\mathbb{E}\Big[\|\nabla \log \hat h_t^\lambda - \nabla \log h_t^\star\|_{L^2(p_{T-t})}^2\Big] \;\le\; C\, \sigma_{T-t}^{-8}\, n^{-2/(d+8)}\, \log^4 n$$

- Rate $n^{-2/(d+8)}$ is *worse* than the best-possible score rate —
  the $+8$ comes from the gradient-regularisation penalty and the extra
  Lipschitz control needed for variational stability.
- The $\sigma_{T-t}^{-8}$ factor says the bound blows up as the noise
  level $\sigma_{T-t}$ shrinks at the end of reverse sampling. Must
  stop early or schedule carefully.

**Theorem 5.6 (end-to-end $W_2$ bound for controllable generation)**:
$$W_2\big((\mathcal{M} \circ \mathcal{T}_R)_{\sharp} \hat q_{T - T_0},\; q_0\big) \;\le\; [\text{guidance} + \text{reference score} + \text{init} + \text{discretisation}]$$

Four-way decomposition — each term traceable:
- **Guidance error** from Theorem 5.3.
- **Reference score error** $\varepsilon_{\text{ref}}$ from Assumption 3.
- **Initialisation error** from mismatch at $t = T$.
- **Discretisation error** from the Euler–Maruyama step.

**Theorem 5.9 (adaptivity to intrinsic dimension)**: under a low-dim
subspace assumption on $q_0$, the rates above transform
$d \to d^\star$ — curse-of-dimensionality escapes when the target lives
on a meaningful low-dim subspace.

## 5. Proof Technique

- **Variational stability** = the augmented loss has a strongly-convex
  lower bound *in the function-space Hilbert norm*, so approximation
  errors don't amplify pathologically. Concretely, the gradient
  regularisation term prevents the naive $L^2$ least-squares from
  exploding when the weight $w$ has steep gradients.
- **Girsanov** on the reverse SDE (same pillar as
  [[girsanov-path-kl-bound]]) converts per-step drift mismatch into
  path-KL, which bounds $W_2$ via Talagrand-type inequalities.
- **Empirical process** bounds the finite-sample approximation of
  $\nabla \log h^\star$ by the regularised regressor.
- **Low-dim escape** (Theorem 5.9): parametrise the reward $w$ via a
  low-dim factor, propagate the effective dimension through the
  regression.

## 6. Novelty vs Prior Work

Verbatim (§1.1):

> "We introduce **variationally stable Doob's matching**, a novel
> guidance estimation framework for controllable diffusion models
> grounded in Doob's h-transform. The Doob h-function is estimated via a
> least-squares regression approach augmented with a **gradient
> regularisation**, and the plug-in gradient of the logarithm of the
> resulting h-function estimator yields an estimator for the Doob's
> guidance."

Against existing alignment methods:
- **Classifier guidance** (Dhariwal–Nichol): pointwise score gradient of
  a classifier; can diverge in high-reward regions. This paper's
  gradient regularisation prevents divergence.
- **RLHF / fine-tuning the reference**: expensive; this paper freezes
  the reference.
- **Importance-sampling reweighting**: variance blows up in high-d;
  Doob's h-transform is an implicit reweighting with better sample
  efficiency.

## 7. Limitations

- **Bounded reward** (Assumption 2) — strong rewards blow up the
  constant $\bar B / \underline B$.
- **$\sigma^{-8}$ noise factor** — need careful noise-schedule design to
  avoid the bound exploding at the reverse-sampling endpoint.
- **Guidance network is an extra component** — must be trained; total
  memory higher than a naked reference model.
- **Static reward** — does not (yet) handle time-varying or streaming
  rewards.

---

## Connections

- **uses** [[girsanov-path-kl-bound]] for the path-KL conversion.
- **extends** the inference-time-guidance idea in
  [[cofindiff-controllable-financial-diffusion]] — crucially *without*
  jointly training the reference score net. Much cheaper if you want
  many reward heads on one base model.
- **parallels** [[score-based-sequential-langevin-sampling]] — same
  group, both are inference-time schemes; SSLS = sequential filtering,
  this = static alignment.
- **applies to** [[synthetic-augmentation-financial-timeseries]]:
  condition a market-trained diffusion on arbitrary regime rewards
  *post hoc* by training a lightweight Doob head.

## Contradictions

- **$\sigma^{-8}$ blow-up vs SSLS's $\log^{k+1}$ growth**: both are
  inference-time analyses of score-based models, but the degeneracies
  are very different. No direct conflict, but worth noting that
  static-alignment pays in noise-schedule brittleness whereas sequential
  filtering pays in step-count log factor.

## Open questions

- Does variational stability hold for **heavy-tailed reward**
  functions? The $\bar B$ bound plus Lipschitz requirements on $r$ would
  need to be relaxed — critical for finance (reward = drawdown or
  Sharpe is unbounded). Candidate for `open-problems/`.
- **Self-consuming regime**: what happens when the reference score net
  is itself trained on Doob-aligned outputs?  Ties into
  [[self-consuming-diffusion-stability]].
- **Time-varying reward**: replace $w(x_0)$ with $w(x_0, t)$ — does the
  variational stability argument still close?

## My take / relevance

**Strong candidate for replacing CoFinDiff's classifier-free-guidance
training in our pipeline.** The economics:

- **CoFinDiff**: jointly train conditional score on $(x, c)$ — $O(n)$
  per regime $c$.
- **This paper**: train *one* reference score on $x$ + a cheap Doob
  head per regime. Total cost is $O(n) + K \cdot O(n_{\text{Doob}})$
  with $n_{\text{Doob}} \ll n$.

The $\sigma^{-8}$ caveat is manageable if we use a staircase schedule
that keeps $\sigma$ above a floor.

**Next action**: prototype a Doob head on our market-trained
[[fts-interdiff-fusion]] backbone to compare against the existing
classifier-free factor pipeline. The theorem gives a concrete sample-
size target: $n_{\text{Doob}} \sim \varepsilon_{\text{guide}}^{-(d+8)}$.
