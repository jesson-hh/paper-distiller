---
title: "Jian Huang 黄健 — 深度生成统计 + 非参表征的资深 PI"
category: "authors"
slug: "huang-jian"
tags: ["author", "statistical-learning", "generative-models", "nonparametric", "survival-analysis", "representation-learning", "hk-polyu"]
affiliation: "Department of Applied Mathematics & DSAI, The Hong Kong Polytechnic University"
aliases: ["Jian Huang", "J. Huang", "黄健"]
homepage: "https://sites.google.com/view/prof-jian-huang"
scholar: "VKYAVUMAAAAJ"
papers: ["arxiv:2512.18971", "arxiv:2505.07967", "arxiv:2505.04992", "arxiv:2504.09567", "arxiv:2504.01031", "arxiv:2504.01030", "arxiv:2503.01728", "arxiv:2502.20414", "arxiv:2411.01833", "arxiv:2410.19226", "arxiv:2410.18021", "arxiv:2408.14036", "arxiv:2408.00602", "arxiv:2406.13197", "arxiv:2406.03683", "arxiv:2405.18284", "arxiv:2404.00551", "arxiv:2402.16661", "arxiv:2312.05579", "arxiv:2311.11475", "arxiv:2309.00771", "arxiv:2306.15163", "arxiv:2305.00608"]
refs: ["arxiv:2306.15163", "arxiv:2312.05579", "arxiv:2311.11475", "arxiv:2504.09567"]
links: ["jiao-yuling", "synthetic-augmentation-financial-timeseries", "diversity-collapse-sde-framework", "cnf-convergence-distribution-learning", "boosting-with-synthetic-pretrained"]
first_seen: "2026-04-17"
last_updated: "2026-04-17"
---

# Jian Huang 黄健

## Profile

- **Affiliation**: Chair Professor, Department of Applied Mathematics
  (cross-appointment in DSAI), **The Hong Kong Polytechnic University**.
- **PhD**: Statistics, University of Washington, Seattle.
- **Prior**: University of Iowa before moving to HK PolyU.
- **Recognition**: ASA Fellow, IMS Fellow; Highly Cited Researcher
  (Math) 2015-2019; Top-2% cited scientist (Stanford/Elsevier 2021).
- **Homepage**: [sites.google.com/view/prof-jian-huang](https://sites.google.com/view/prof-jian-huang)
- **Scholar**: [VKYAVUMAAAAJ](https://scholar.google.com/citations?user=VKYAVUMAAAAJ)
- **Research focus** (recent 3 years): the *statistical inference / theory*
  side of **deep generative modelling**, **nonparametric regression with
  neural networks**, **representation learning with statistical
  guarantees**, and classical strengths in semiparametric / survival /
  high-dim stats.

## Relation to [[jiao-yuling]]

Huang is **the statistical-inference elder** to Jiao's applied-PDE /
scientific-computing machinery. Of Huang's 23 papers in our window, **6
are joint with Jiao** — the joint subset (CNF convergence, Gaussian
interpolation flows, synthetic-data boosting, RePU score estimation,
adversarial excess risk, Wasserstein DRO) is exactly the
"provable-generative-stats" intersection of the two groups.

The important complement: Huang brings **Wasserstein Generative
Regression** (`arxiv:2306.15163`, JRSSB 2025), a unifying framework
that Jiao's papers don't touch, and a whole **sufficient-dimension-
reduction / transfer-learning / fair-rep** line that is independently
his.

## Scope of this distillation

- **Window**: 2023-04-17 .. 2025-12-22 (last 3 years)
- **Stat/ML categories only**: `stat.ML`, `stat.ME`, `math.ST`,
  plus a few `cs.LG` papers that cross-listed
- **Total in window**: 23 (count is a lower bound; some `cs.LG` items
  may exist outside the filter — to revisit)
- **Core kept (A + B + C)**: 17 — tied to our diffusion / synthetic
  programme or to the shared stats-theory toolkit
- **Peripheral (D + E)**: 6 survival / change-plane / semi-supervised
  papers — listed at L0
- **Depth split**: **L2 = 4** (all new, Huang-signature) ·
  **ref_only = 3** (already have L2 via [[jiao-yuling]]) ·
  **L1 = 10** · **L0 = 6**

## L2 rate cheatsheet (Huang-signature deep notes)

Verified from the papers themselves; see per-article files for assumptions + proofs.

| Paper | Year | Main quantity | **Rate / Guarantee** | Notes |
|---|---|---|---|---|
| [[wasserstein-generative-regression]] | 2023 (JRSSB 2025) | excess risk + $d_{\mathcal F_B^1}$ | $C_1\, n^{-\beta/(2\beta + [3(m+d)] \vee [2\beta(d+q+1)])}$ | same rate on mean *and* distribution |
| [[conditional-stochastic-interpolation]] | 2023 | drift $L^2$ error | $O(n^{-2\beta/(2\beta + k + d + 1)} \log^5 n)$ | minimax optimal (drift); no $W_2$ |
| [[transport-maps-conditional-independence-testing]] | 2025 | test $T_n \to \zeta$ | consistent under $H_0/H_1$ (Thm 5, 6) | no finite-sample type-I bound |
| [[gaussian-interpolation-flows]] | 2023 | $W_2^2 \le \text{const} \cdot \|V-\tilde V\|^2$ (stability) | Prop 54(ii) | **no sample rate** — regularity foundation |

All exponents / guarantees hold on compact support (or semi-log-concave
targets for GIF). None extended to heavy-tailed financial data.

---

## Cluster A — Generative model theory (9 papers)

The heart of Huang's recent line. Three sub-threads:

1. **Stochastic interpolation / flows with provable rates** —
   Gaussian Interpolation Flows (2311.11475), Conditional Stochastic
   Interpolation (2312.05579, 2512.18971), CNF convergence
   (2404.00551), Wasserstein Generative Regression (2306.15163).
2. **Deriving statistical inference tools from generative models** —
   Transport Maps for CI Testing (2504.09567), Penalized Generative
   Variable Selection (2402.16661).
3. **Using generative models for domain adaptation** — Bayesian Power
   Steering (2406.03683), Boosting with Synthetic Data (2505.04992).

### L2 deep notes (4 — open each article for theorems)

- **2025-12** — 2512.18971 "Conditional Stochastic Interpolation for
  Generative Nonlinear Sufficient Dimension Reduction". **[L0 here;
  treated as extension of 2312.05579 in the L2 note for the earlier
  paper.]**
- **2025-04** — [[transport-maps-conditional-independence-testing]]
  Hypothesis testing for $X \perp Y \mid Z$ via transport-map
  transformations that reduce conditional to unconditional
  independence. *Generative ML put to work in inference.*
- **2023-12** — [[conditional-stochastic-interpolation]]
  Conditional distribution learning via stochastic interpolants —
  the Huang-group's foundational paper in this line. Precursor and
  parallel to Jiao's conditional Föllmer flow
  ([[conditional-follmer-flow-distribution-learning]]).
- **2023-11** — [[gaussian-interpolation-flows]]
  Well-posedness, Lipschitz velocity, and $W_2$ convergence for
  Gaussian-denoising CNFs. Foundational; many later Huang/Jiao papers
  lean on this.
- **2023-06** — [[wasserstein-generative-regression]]
  **Flagship.** Unified framework that covers *both* nonparametric
  regression and conditional distribution learning as special cases.
  Published JRSSB 2025.

### Cross-referenced to [[jiao-yuling]] L2 (already deep, no duplication)

- 2404.00551 — [[cnf-convergence-distribution-learning]]
- 2505.04992 — [[boosting-with-synthetic-pretrained]]

### L1 summaries (2 remaining in cluster A)

- **2024-06 · Bayesian Power Steering** (arxiv:2406.03683) · *Ding Huang
  · Ting Li · Jian Huang*
  - **Problem**: domain adaptation of a pre-trained diffusion model.
  - **Path**: a Bayesian framework with *learnable intervention modules*
    inserted at specific denoising steps; the modules are trained on a
    small target-domain dataset.
  - **Insight**: the target-distribution shift can be absorbed by
    sparse modular interventions without retraining the whole net —
    parallels [[diffusion-doob-matching-inference-alignment]] in spirit
    but with different (Bayesian) machinery.
  - **Relevance**: alternative path to CoFinDiff-style conditioning;
    worth a close read to compare against Doob's matching.

- **2024-02 · Penalized Generative Variable Selection**
  (arxiv:2402.16661) · *Tong Wang · Jian Huang · Shuangge Ma*
  - **Problem**: variable selection in conditional Wasserstein GANs for
    survival data.
  - **Path**: group-Lasso penalty on the generator input; only relevant
    covariates pass through.
  - **Insight**: generative models + sparsity = a clean sample-efficient
    selection method in sparse-signal regimes.
  - **Relevance**: if we ever need to shortlist factors for our
    [[fts-interdiff-fusion]] conditioning, this is a candidate
    mechanism.

---

## Cluster B — Statistical learning theory (5 core papers)

Non-generative stats-theory results. Several joint with Jiao's group
(already summarised in [[jiao-yuling]] L1 list); listed here for
completeness.

- **2025-05 · Wasserstein DRO Nonparametric Regression**
  (arxiv:2505.07967) · *Liu · Jiao · Wang · Huang* · **L1**
  (full summary in [[jiao-yuling]])
- **2024-05 · Adaptive Debiased SGD in High-Dim GLMs with Streaming
  Data** (arxiv:2405.18284) · *Han · Luo · Luo · Lin · Huang* · **L0**
  - Online inference for high-dim GLMs. Tangential to us.
- **2023-09 · Non-Asymptotic Adversarial Excess Risk under
  Misspecification** (arxiv:2309.00771) · *Liu · Jiao · Wang · Huang*
  · **L1** (in [[jiao-yuling]])
- **2023-05 · RePU NN for Score Estimation** (arxiv:2305.00608) ·
  *Shen · Jiao · Lin · Huang* · **L1** (in [[jiao-yuling]])

---

## Cluster C — Transfer & representation learning (6 papers)

Huang's *independent* signature beyond the Jiao intersection. The
core question: **how do we carry information across domains, tasks,
or protected attributes with statistical control?**

- **2025-04 · Density-Ratio Estimation (Unbounded) for Covariate
  Shift** (arxiv:2504.01031) · *Xu · Yu · Huang* · **L1**
  - **Problem**: estimating density ratios when the ratio itself is
    unbounded (heavy-tailed shift).
  - **Path**: a regularised loss that remains stable without the
    bounded-ratio assumption.
  - **Insight**: unlocks covariate-shift methods for heavier-tailed
    domains than previously tractable.
  - **Relevance**: directly useful for **regime-shift-robust finance**
    experiments.

- **2025-04 · Fair Sufficient Representation Learning**
  (arxiv:2504.01030) · *Zhou · Ip · Huang* · **L1**
  - **Problem**: representation that is simultaneously *sufficient for
    the task* and *fair w.r.t. protected attributes*.
  - **Path**: distance-covariance-based loss that balances the two
    objectives.
  - **Insight**: fairness and sufficiency trade off on a measurable
    axis; no heuristic needed.

- **2025-02 · Transfer Learning via Enhanced Sufficient Representation**
  (arxiv:2502.20414) · *Ge · Zhou · Huang* · **L1**
  - **Problem**: TL via invariant *sufficient* reps across domains.
  - **Path**: augment the SDR objective with a domain-invariance
    penalty.
  - **Insight**: sufficient-dim-reduction is a natural lens for TL —
    "what is the minimum rep that works for all domains".

- **2024-06 · Representation Transfer Learning for Semiparametric
  Regression** (arxiv:2406.13197) · *He · Liu · Zhang · Huang* · **L1**
  - **Problem**: TL in semiparametric regression where the unknown
    function differs across domains but shares a representation.
  - **Path**: profile-likelihood-style joint estimation.
  - **Insight**: classical semiparametric machinery reapplied to
    neural reps.

- **2025-03 · DeepSuM Deep Sufficient Modality Learning**
  (arxiv:2503.01728) · *Z. Gao · Huang · Li · Wang* · **L0**
- **2024-10 · Deep Nonparametric Conditional Hazard** (arxiv:2410.18021)
  · *Su · Liu · Yin · Huang · Zhao* · **L1** (cluster D, survival)

---

## Cluster D — Classical stats + deep-net extensions (4 papers)

Peripheral to our programme. Listed for completeness.

- **2024-10 · Deep Transformation Model** (arxiv:2410.19226) · **L1**
  Nonparametric transformation models with neural networks for
  regression and survival.
- **2024-10 · Deep Conditional Hazard** (arxiv:2410.18021) · **L1**
  DNN-based nonparametric estimation for conditional hazards with
  censored data.
- **2024-08 · Robust Change-Plane Regression** (arxiv:2408.14036) · **L0**
- **2024-08 · Efficient Change-Plane Testing** (arxiv:2408.00602) · **L0**

---

## Cluster E — Semi-supervised / miscellaneous (1 paper)

- **2024-11 · OwMatch** (arxiv:2411.01833) · **L0**
  Open-world semi-supervised learning with self-labelling.

---

## Connections

Huang's work plugs into ours along 3 seams:

- **unifies regression + conditional generation** via
  [[wasserstein-generative-regression]] — this is the natural host
  framework when we argue that conditional financial generators are
  "just regression plus noise". We might base a technique or
  methodology doc on it.
- **provides the stability backbone** for stochastic-interpolant methods
  ([[gaussian-interpolation-flows]], [[conditional-stochastic-interpolation]])
  — Lipschitz velocities, $W_2$ bounds, well-posedness results that we
  can cite when claiming regularity in our pipelines.
- **imports generative models into classical inference** via
  [[transport-maps-conditional-independence-testing]] — an example of
  the opposite direction (generative → inference). Suggests a route
  for using our synthetic-augmented generators to do *statistical
  tests* on real data, not just prediction.

## Contradictions

- **Independent of the "collapse" discussion** in
  [[diversity-collapse-sde-framework]] — Huang's papers *assume good
  generators* and derive stats guarantees; our collapse line asks
  *when generators go bad*. The two complement rather than conflict.

## Tracking

- [ ] Deep-read 4 L2 picks (verified rates filled into cheatsheet).
- [ ] Watch **Wasserstein Generative Regression** follow-ups — the
  JRSSB publication may spawn an extension line in 2026.
- [ ] Monitor **transport-map inference** (2504.09567) — if it works
  out, it could seed an entire new `techniques/` category.
- [ ] Re-run arxiv author query with `cs.LG` included too, not just
  `stat.*` — may have missed papers.

---

## Coauthor log (record-only)

Frequency in the 3-year window:

| Coauthor | # papers with Huang | Affiliation guess | Priority |
|---|---|---|---|
| Yuling Jiao | 6 | Wuhan | **already an [[jiao-yuling]] page** |
| Guohao Shen | 3 | HK PolyU stats | medium |
| Yuanyuan Lin | 3 | HK PolyU stats | medium |
| Yuan Gao | 3 | unknown | low-med |
| Ting Li | 3 | HK PolyU | low-med |
| Tong Wang | 3 | Renmin / Xiamen | low |
| Xu Liu | 3 | SUFE | low |
| Shuangge Ma | 2 | Yale biostat | low |
| Zhou Yu | 2 | ECNU | low |
| Junhui Wang | 2 | CUHK | low |
| Changyu Liu | 2 | (student) | low |
| Shuntuo Xu | 2 | HK PolyU (student) | low |
| Xueyu Zhou | 2 | HK PolyU (student) | low |
| Ding Huang | 2 | HK PolyU | low |

Shen + Lin form the HK PolyU stats trio with Huang; any of them could
be the third author-page to build if we want to complete the HK PolyU
cluster.

## Raw metadata pointers

- Author query cache: `wiki/raw/author/huang-jian.json`
- Individual paper caches (added during L2 deep-read): `wiki/raw/arxiv/<arxiv-id>.json`
