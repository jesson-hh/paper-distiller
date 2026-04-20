# Theory-Article Template

> For papers whose core contribution is a **theorem** (rate, bound, convergence
> guarantee, approximation result). Complements the generic article pattern
> with seven theory-specific fields. Copy under `wiki/articles/<slug>.md`.
>
> `_templates/` is outside `CATEGORIES` — not indexed, safe to edit here.

---

## Frontmatter

```yaml
---
title: "<Short Name> (<venue/arxiv>) — <one-line contribution>"
category: "articles"
slug: "<descriptive-slug-lowercase>"
tags: ["theory", "<subfield>", "<arxiv-year>"]
authors: ["<author-slug-1>", "<author-slug-2>"]     # NEW: link to authors/
refs: ["arxiv:XXXX.XXXXX"]
links: ["<related-technique-slug>", "<related-article-slug>"]
created: "YYYY-MM-DDTHH:MM:SS"
updated: "YYYY-MM-DDTHH:MM:SS"
---
```

## Body skeleton

```markdown
# <Paper Title>

> **Authors**: First-1, First-2, … (see [[author-slug]] for their programme)
> **Venue / arxiv**: <arxiv:ID or conference>
> **Status in our wiki**: L2 deep note (theory-template)

## TL;DR
One or two sentences: *what's proven*, plus the rate. A reader
should know whether to keep reading.

---

## 1. Setting
- **Target object**: what is being estimated / sampled / approximated
  (regression function, distribution, score field, transport map, …)
- **Function/distribution class**: Hölder $\mathcal{C}^{\beta}$, Sobolev
  $H^s$, log-concave, low intrinsic dim $d^*$, mixing coefficient $\alpha$, …
- **Observation model**: i.i.d. / dependent (β-mixing, C-mixing) / online /
  adversarial / partial / misspecified
- **Sample size**: $n$; **dimension**: ambient $d$ vs intrinsic $d^* \ll d$

## 2. Method
- **Estimator / algorithm**: deep ReLU NN, RePU NN, Transformer, flow-ODE,
  Langevin, DRM variational objective, …
- **Loss / variational form**: MSE, Föllmer matching, Doob's h-transform,
  Ritz energy, Galerkin weak form, adversarial risk, …
- **Training**: GD / SGD / PGD / over-parameterised vs NTK / …

## 3. Assumptions (the fine print)
1. Smoothness / regularity ofif the target
2. Domain (compact / $\mathbb{R}^d$ with tail decay)
3. Data structure (low-d manifold / mixing)
4. Network / model capacity (width, depth, norm constraints)
5. Any additional conditions (Lipschitz score, log-Sobolev, bounded second
   moments, etc.) — **list explicitly so we can compare to competitors**

## 4. Main Result(s)
- **Theorem <n>**: estimator $\hat{f}$ satisfies
  $$\mathbb{E}\|\hat{f} - f\|^2 \le C\, n^{-\frac{2\beta}{2\beta + d^*}} \log^k n$$
  — or whichever rate/bound applies. Be concrete.
- **Corollaries** / special cases (e.g. curse-of-dimension escape when
  $d^* \ll d$).
- **Sharpness**: is the rate tight / minimax-optimal? Matched to a lower
  bound? (Often not — note if open.)

## 5. Proof Technique
One paragraph on the **new** technical ingredient (the rest is usually
standard decomposition). Options that tend to recur in this programme:

- **Approximation + generalization + optimization** tri-decomposition
- **Girsanov / path-KL** for SDE-based methods
- **NTK / lazy-training** argument for optimisation error
- **Kolmogorov–Arnold** representation for multivariate approximation
- **Bernstein-type concentration** under dependent data
- **Reverse Poincaré / log-Sobolev** to lift score error to distribution error

## 6. Novelty vs Prior Work
- Closes gap **X** in [[related-article-slug]]: previously rate was $n^{-?}$
  under stronger assumption Y; this paper weakens Y.
- Extends [[technique-slug]] from case A to case B.
- First analysis of this estimator class for this setting.

## 7. Limitations
- Where does the bound degrade? (boundary behaviour, non-compact domains,
  heavier tails, non-smooth targets, …)
- What's assumed away? (dependencies ignored, hidden constants huge,
  over-parametrisation regime only, …)
- Practical vs theoretical gap: does the paper match empirical evidence?

---

## Connections (human-readable)

- **uses** [[technique-slug]] for Step X of the proof
- **extended by** [[later-paper-slug]] which relaxes assumption Y
- **parallels** [[sibling-paper-slug]] (same rate via a different argument)
- **applies to** [[direction-slug]] — we can plug this bound into our …

## Contradictions

- On **<topic>**: this paper's rate requires assumption X, whereas
  [[competitor-paper]] claims the same rate without X — reconcile or flag.
- Leave empty if none genuine.

## Open questions surfacing from this paper
- (For each, tag whether it belongs in `open-problems/`.)

## My take / relevance
- How this connects to my direction ([[direction-slug]]).
- Starred quotes / equations we may reuse.
