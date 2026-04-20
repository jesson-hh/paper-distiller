---
title: "Boosting Statistical Learning with Synthetic Data from Pretrained Large Models (arxiv:2505.04992)"
category: "articles"
slug: "boosting-with-synthetic-pretrained"
tags: ["theory", "synthetic-data", "self-consuming", "diffusion", "wasserstein-filter", "statistical-learning", "jiao-group", "arxiv-2025"]
authors: ["jiao-yuling"]
refs: ["arxiv:2505.04992"]
links: ["jiao-yuling", "synthetic-augmentation-financial-timeseries", "diversity-collapse-sde-framework", "self-consuming-diffusion-stability", "recursive-chi2-inequality", "score-contamination-lemma"]
created: "2026-04-17T12:00:00"
updated: "2026-04-17T12:30:00"
---

# Boosting Statistical Learning with Synthetic Data from Pretrained Large Models

> **Authors**: Jialong Jiang · Wenkang Hu · Jian Huang · Yuling Jiao · Xu Liu (see [[jiao-yuling]])
> **arxiv**: 2505.04992, May 2025
> **Status in our wiki**: L2 deep note — **directly on our synthetic-augmentation axis**

## TL;DR

A risk bound that is a *linear function* of the **$W_1$ distance between
synthetic and real** distributions. Combined with a concrete
**distance-based filter** (Wasserstein-1 in a VAE latent for images,
$p$-value tests for tabular), this turns vague "synthetic data may help
or hurt" folklore into a testable condition on the generator.

Evaluated on 8 datasets across tabular, medical, and vision.

---

## 1. Setting

- **Task**: supervised learning — regression or classification.
- **Data**:
  - Real $\mathcal{P}_{\text{real}}$, $n$ samples.
  - Synthetic $\mathcal{P}_{\text{synth}}$ from a pre-trained generator
    (Stable Diffusion, large VAEs, GANs, LLM generators, etc.).
- **Question**: when does appending filtered synthetic data to real
  strictly reduce excess risk?

## 2. Method — a two-layer pipeline

1. **Generate** — draw many synthetic samples from the pre-trained
   model.
2. **Filter** — score each sample with a domain-specific statistic;
   keep only the high-fidelity ones:
   - **Images**: Wasserstein-1 in the VAE **latent** space (not the
     pixel space, which is too high-dim to estimate $W_1$ well).
     > "We compute the Wasserstein distance not directly in the
     > high-dimensional image space, but in the **latent space** derived
     > from images."
   - **Tabular**: $p$-value based hypothesis testing.
   - **Alternatives tested**: MMD, TV — with comparative results.
3. **Train downstream ERM** on filtered synthetic ∪ real.

## 3. Assumptions

The paper does **not** present a numbered Assumption block. Theorem 3.1
requires:

- Loss $\ell: \mathcal{H} \times \mathcal{Z} \to [0, M]$ is
  $L_\ell$-Lipschitz, bounded by $M$.
- Synthetic distribution satisfies a $W_1$ proximity condition:
  $W_1(\mathcal{P}_{\text{synth}}, \mathcal{P}_{\text{real}}) \le \varepsilon$.

That's it — remarkably light. All the statistical power comes from
Rademacher complexity $\mathfrak{R}_n(\mathcal{H})$ and standard
concentration.

## 4. Main Result

**Theorem 3.1 (Generalization Error Bound)**:

$$\boxed{\;\mathbb{E}_{\mathcal{P}_{\text{real}}}\big[\ell(h, z)\big] \;\le\; \mathbb{E}_{\mathcal{P}_{\text{synth}}}\big[\ell(h, z)\big] \;+\; L_\ell\, \varepsilon \;+\; 2\, \mathfrak{R}_n(\mathcal{H}) \;+\; M\sqrt{\tfrac{\log(1/\delta)}{2n}} \;}$$

where $\varepsilon = W_1(\mathcal{P}_{\text{synth}}, \mathcal{P}_{\text{real}})$.

**Reading the bound**:
- First term: what training on synthetic alone buys you.
- Second term: **linear-in-$\varepsilon$ penalty for the distribution
  shift** — this is the point of the filter, to shrink $\varepsilon$.
- Third + fourth terms: standard generalisation gap on real data.

**Key consequence**: synthetic data *strictly helps* iff the real-data
risk decrease from using more samples (terms 3 + 4) exceeds the
$L_\ell \varepsilon$ penalty of the filter — a **testable inequality**.

## 5. Filter (verbatim where possible)

> "Despite the ability to produce large volumes of synthetic data, the
> proportion that effectively improves model performance is limited."

Concrete filter design:
- **Image pipeline**: encoder $E$ (a pre-trained VAE) maps both real and
  synthetic into latent $\mathcal{Z}_{\text{real}}, \mathcal{Z}_{\text{synth}}$;
  compute $W_1$ in latent; reject synthetic samples whose per-instance
  latent distance to its real-data nearest-neighbour exceeds a
  threshold.
- **Tabular pipeline**: hypothesis tests (e.g. two-sample tests) used to
  retain feature subsets that match the real distribution.

## 6. Experiments

Eight datasets:
1. **Boston Housing** (tabular regression)
2. **GTEx** (118 AD-associated genomic predictors)
3. **German Credit** (tabular classification)
4. **MNIST** (vision)
5. **CIFAR-10**
6. **CIFAR-100** (ResNet-18, ImageNet weights)
7. **ISIC** (skin cancer, 10,015 training samples)
8. **Cassava Leaf Disease** (5 classes, EfficientNet-B0)

Consistent improvement reported; precise deltas in the paper.

## 7. Novelty vs Prior Work

- Contra **model-collapse literature** (Shumailov 2024, etc.): those
  show unfiltered iterative use collapses diversity; this paper shows
  single-round use *with filtering* helps.
- Contra **covariate-shift importance reweighting**: doesn't scale when
  $m \gg n$; the $W_1$-based filter is a structural projection instead
  of a soft reweighting.
- Quote:
  > "We propose a **novel end-to-end framework that generates and
  > systematically filters synthetic data through domain-specific
  > statistical methods**, selectively integrating high-quality samples."

## 8. Limitations

- Requires designing $\phi$ per domain — no universal filter.
- I.i.d. data assumed — **time-series extension is non-trivial** (our
  problem).
- Filter introduces selection bias the paper handles only for bounded
  Lipschitz $\phi$.
- Rate hidden in $\mathfrak{R}_n(\mathcal{H})$; no explicit $n^{-\alpha}$
  expression.

---

## Connections

- **complements** [[diversity-collapse-sde-framework]] — unfiltered
  iterative use collapses (our framework), filtered single-round use
  helps (this paper). Cite both for the full picture.
- **directly feeds** [[synthetic-augmentation-financial-timeseries]] —
  our "augmentation helps" gets replaced by "augmentation helps iff
  $L_\ell W_1(\mathcal{P}_{\text{synth}}, \mathcal{P}_{\text{real}})$ is
  small".
- **structurally parallel to** [[score-contamination-lemma]] — both
  linearly decompose downstream damage into a "distance-from-real" term.
- **parallels** [[self-consuming-diffusion-stability]]: both ask "is
  synthetic safe?"; single-round (this paper) vs iterative (our
  direction).

## Contradictions

- On **"synthetic data helps"**:
  - Shumailov 2024 (`cite:shumailov-2024-nature-collapse`): **no**
    (unfiltered, iterative).
  - This paper: **yes** (filtered, single round).
  - Both are correct under their premises. Our wiki should flag this
    side-by-side whenever citing either.

## Open questions

- **Finance-specific filter $\phi$**: what's the right $W_1$ proxy for
  financial series?
  - Candidate 1: **$W_1$ on log-return empirical CDF** — captures tail
    shape.
  - Candidate 2: **autocorrelation-weighted $W_1$** — captures temporal
    structure.
  - Candidate 3: **stylised-fact vector distance** — matches higher
    moments + intermittency.
- **Iterative filtered** use: does the single-round bound extend to
  $K$ rounds of filter-and-augment? Likely a recursion similar to
  [[recursive-chi2-inequality]] with a filter shrinkage factor —
  candidate for `open-problems/`.

## My take / relevance

**Highest-priority L2 paper for our programme.** The filter schema is
*directly* actionable:

Concrete action plan:
1. Generate synthetic CSI300 log-returns with
   [[waveletdiff-multilevel-2510-11839]] or our
   [[fts-interdiff-fusion]] architecture.
2. Apply a moment-matching or stylised-fact-aware $\phi$ to reject
   low-quality samples.
3. Check whether $L_\ell W_1(\text{filtered synth}, \text{real})$ is
   small enough to beat the $1/\sqrt{n}$ real-data baseline.
4. If yes, ship. If no, iterate on the filter.

**Next research move**: draft an `open-problems/` entry on *"minimax
filter for financial stylised-fact preservation under synthetic
augmentation"*.
