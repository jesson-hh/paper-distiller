---
title: "Yuling Jiao 焦雨领 — 深度学习逼近理论 + 可证明生成模型"
category: "authors"
slug: "jiao-yuling"
tags: ["author", "deep-learning-theory", "generative-models", "approximation-theory", "scientific-computing", "wuhan-university"]
affiliation: "School of Mathematics and Statistics, Wuhan University"
aliases: ["Yuling Jiao", "焦雨领", "Y. Jiao"]
homepage: "http://jszy.whu.edu.cn/jiaoyuling/en/index.htm"
scholar: "yFDDsVgAAAAJ"
papers: ["arxiv:2602.10587", "arxiv:2601.08527", "arxiv:2601.06514", "arxiv:2512.08022", "arxiv:2505.07967", "arxiv:2505.04992", "arxiv:2504.13558", "arxiv:2504.12175", "arxiv:2502.14424", "arxiv:2411.13443", "arxiv:2410.09383", "arxiv:2409.05577", "arxiv:2408.08533", "arxiv:2407.09032", "arxiv:2405.12684", "arxiv:2405.11451", "arxiv:2405.05512", "arxiv:2404.13309", "arxiv:2404.02538", "arxiv:2404.00551", "arxiv:2403.19090", "arxiv:2402.01460", "arxiv:2401.04535", "arxiv:2312.11863", "arxiv:2311.11475", "arxiv:2311.03660", "arxiv:2309.03490", "arxiv:2309.00771", "arxiv:2306.13881", "arxiv:2305.00608", "arxiv:2304.07947"]
refs: ["arxiv:2405.05512", "arxiv:2404.13309", "arxiv:2404.00551", "arxiv:2402.01460", "arxiv:2601.06514", "arxiv:2411.13443", "arxiv:2505.04992"]
links: ["synthetic-augmentation-financial-timeseries", "diversity-collapse-sde-framework", "girsanov-path-kl-bound", "ntk-score-error-decomposition", "recursive-chi2-inequality"]
first_seen: "2026-04-17"
last_updated: "2026-04-17"
---

# Yuling Jiao 焦雨领

## Profile

- **Affiliation**: School of Mathematics and Statistics, Wuhan University
- **Rank**: Full Professor (promoted 2023)
- **PhD**: Wuhan University, 2014
- **Homepage**: [jszy.whu.edu.cn/jiaoyuling](http://jszy.whu.edu.cn/jiaoyuling/en/index.htm)
- **Scholar**: [yFDDsVgAAAAJ](https://scholar.google.com/citations?user=yFDDsVgAAAAJ)
- **Research focus**: theoretical foundations of deep learning — approximation
  theory, generative models with provable guarantees, scientific computing
  (PINN / DRM), statistical learning theory. Also a quantum-ML side line we
  skip here.

## Scope of this distillation

- **Window**: 2023-04-17 .. 2026-02-16 (last 3 years of arxiv preprints)
- **Total papers found**: 37
- **Core kept (A/B/C/D)**: **31** — clusters tied to our programme
  ([[diversity-collapse-sde-framework]],
  [[synthetic-augmentation-financial-timeseries]])
- **Dropped (E/F)**: 6 quantum-ML + LLM papers — listed at the bottom for
  completeness, not distilled
- **Depth split**: L2 deep notes = 7 · L1 in-place summaries = 19 · L0 list
  = 5

Levels:
- **L0** — title + one-line topic only (peripheral to our interests)
- **L1** — inline here: *Problem → Path → Insight → Relevance*
- **L2** — full `articles/<slug>.md`, linked via `[[…]]`

Machine record of paper metadata is in
`wiki/raw/author/jiao-yuling.json` — keep that in sync when new papers
appear.

---

## Cluster A — Provable generative modelling (14 papers)

His flagship line. Three-step research arc visible:
1. **Föllmer flow basics (2023)** — construct a unit-time ODE flow from
   Gaussian to target, prove Lipschitz / well-posedness / Wasserstein
   rate.
2. **Conditional & latent extensions (2024)** — lift to conditional
   distributions, flow-matching with transformers, Schrödinger bridge in
   latent space.
3. **Inference-time & assimilation (2025-26)** — Doob's-matching for
   controllable generation, Bayesian posterior sampling, Langevin-based
   stochastic interpolants, deep bootstrap.

Within each extension, the proof scaffold stays: *approximation error of a
velocity/score network* → *Girsanov path-KL* → *coupling to Wasserstein
distance on distributions*. This is the same pillar structure as our
[[diversity-collapse-sde-framework]]; Jiao's group provides ready-to-cite
bounds.

### L2 rate cheatsheet (from the 7 deep notes below)

| Paper | Year | Main quantity | **Rate** | Notes |
|---|---|---|---|---|
| [[conditional-follmer-flow-distribution-learning]] | 2024 | $\mathbb{E}_y[W_2^2(\hat p(\cdot\|y), p(\cdot\|y))]$ | $\tilde O(n^{-4/(9(d+d_Y+5))})$ | symmetric in $d, d_Y$ |
| [[cnf-convergence-distribution-learning]] | 2024 | $\mathbb{E}\,W_2$ | $\tilde O(n^{-1/(d+5)})$ | time-singularity adds $+2$ |
| [[characteristic-learning-one-step-generation]] | 2024 | $\mathbb{E}\,\mathcal{D}(\hat g)$ | $\tilde O(n^{-2/(d+3)} \log^2 n)$ | **one-step**, beats multi-step |
| [[latent-schrodinger-bridge-diffusion]] | 2024 | $\mathbb{E}\,W_2$ | $\tilde O(n^{-1/(6(d^\star+3))})$ | uses intrinsic $d^\star$ |
| [[score-based-sequential-langevin-sampling]] | 2024 | $(\varepsilon^{TV}_{k+1})^2$ | $O((\varepsilon^2_0+\varepsilon^2)\log^{k+1}(\varepsilon^{-1}))$ | polynomial in $k$, not exp |
| [[boosting-with-synthetic-pretrained]] | 2025 | excess risk | $L_\ell\, W_1(\text{synth,real}) + \mathfrak R_n(\mathcal H) + \ldots$ | Wasserstein-filter bound |
| [[diffusion-doob-matching-inference-alignment]] | 2026 | guidance $L^2$ err | $C\sigma^{-8} n^{-2/(d+8)} \log^4 n$ | adapts to intrinsic $d^\star$ (Thm 5.9) |

All exponents assume compact support + Lipschitz-score class. None
proven minimax-optimal (as of Apr 2026).

### L2 deep notes (7 — all generative-theory)

Open the linked article for full theorem statements and proof notes.

- **2026-01** — [[diffusion-doob-matching-inference-alignment]]
  Inference-time alignment of a *frozen* diffusion via variationally
  stable Doob's h-matching. Directly relevant to controllable generation
  ([[cofindiff-controllable-financial-diffusion]]).
- **2025-05** — [[boosting-with-synthetic-pretrained]]
  How synthetic data from pre-trained generators can **boost** downstream
  statistical learning with a formal statement on when it helps / hurts.
  Directly relevant to
  [[synthetic-augmentation-financial-timeseries]] and
  [[diversity-collapse-sde-framework]].
- **2024-11** — [[score-based-sequential-langevin-sampling]]
  Recursive Bayesian filtering with score-based Langevin steps; proves
  asymptotic stability. Template for how to combine multiple
  score networks over time.
- **2024-05** — [[characteristic-learning-one-step-generation]]
  One-step generator with convergence rate depending on **intrinsic**
  (not ambient) dimension — a key ingredient for any low-d synthetic
  augmentation argument.
- **2024-04** — [[latent-schrodinger-bridge-diffusion]]
  Theory of diffusion via Schrödinger bridges in a learned latent space.
  Pairs with [[girsanov-path-kl-bound]].
- **2024-04** — [[cnf-convergence-distribution-learning]]
  Non-asymptotic Wasserstein bounds for continuous normalizing flows
  (CNFs). Foundational; this is the rate we'd cite if we need a CNF
  analogue of our χ² chain.
- **2024-02** — [[conditional-follmer-flow-distribution-learning]]
  Conditional-distribution learning via Föllmer's drift with full
  end-to-end error analysis (approximation + generalization +
  optimization). Template for our [[fts-interdiff-fusion]] conditional
  design.

### L1 summaries (7 remaining in cluster A)

- **2026-02 · Deep Bootstrap** (arxiv:2602.10587) · *Chang · Jiao · Kang
  · Shi*
  - **Problem**: nonparametric regression with principled uncertainty.
  - **Path**: embed bootstrap in a *conditional diffusion* — learn
    $p(y \mid x)$, then sample bootstrap replicates via the diffusion.
  - **Insight**: unifies estimation + sampling + inference in one net;
    confidence bands come for free once the conditional diffusion is
    trained.
  - **Relevance**: an alternative to [[model-free-prediction-uncertainty]]
    flavour of prediction bands — worth considering for
    synthetic-augmented finance.

- **2026-01 · Sampling via Stochastic Interpolants + Langevin in Flow
  ODEs** (arxiv:2601.08527) · *Duan · Jiao · Steidl · Wald · Yang ·
  Zhang*
  - **Problem**: sample from unnormalised Boltzmann densities.
  - **Path**: probability-flow ODE from *linear stochastic interpolants*
    with velocity & initialisation estimated by Langevin samplers.
  - **Insight**: avoids the KL-divergence training target; Langevin
    pre-training gives robust velocity estimates even when mode collapse
    would kill MLE.
  - **Relevance**: candidate drop-in for any Boltzmann-density subproblem
    we encounter (posterior over collapse trajectories?).

- **2025-12 · Provable Diffusion Posterior Sampling for Bayesian
  Inversion** (arxiv:2512.08022) · *Chang · Duan · Jiao · Li · Yang ·
  Yuan*
  - **Problem**: Bayesian inverse problems where the prior is a
    pre-trained diffusion.
  - **Path**: plug-and-play posterior sampler with explicit probability
    transport; non-asymptotic error bound.
  - **Insight**: bridges PnP imaging with diffusion-based priors while
    keeping the guarantees (rare combo).
  - **Relevance**: same *inference-time steering* philosophy as Doob's
    matching but for the posterior-sampling setting.

- **2024-04 · Flow Matching in Latent Space with Transformers**
  (arxiv:2404.02538) · *Jiao · Lai · Wang · Yan*
  - **Problem**: when does latent-space flow matching converge?
  - **Path**: decompose W₂ error = approximation (Transformer) + latent
    encoding error + flow-ODE discretisation.
  - **Insight**: transformers actually give tighter approximation bounds
    than ReLU MLPs in this regime, because of the KA-like representation.
  - **Relevance**: backs the choice of transformer backbones for our
    conditional generators (InterDiff, WaveletDiff).

- **2023-11 · Gaussian Interpolation Flows** (arxiv:2311.11475) · *Gao ·
  Huang · Jiao*
  - **Problem**: a principled treatment of simulation-free CNFs.
  - **Path**: Gaussian denoising in continuous time; prove well-posedness
    and Lipschitz regularity of the learned velocity.
  - **Insight**: Lipschitz velocity is the hidden enabler for every
    downstream Wasserstein/KL bound — worth isolating as a lemma.
  - **Relevance**: would pair with [[girsanov-path-kl-bound]] for finer
    quantitative χ² analysis of GIF-trained models.

- **2023-11 · Sampling via Föllmer Flow** (arxiv:2311.03660) · *Ding ·
  Jiao · Lu · Yang · Yuan*
  - **Problem**: sample Gaussian → target via a 1-time ODE.
  - **Path**: preconditioned Föllmer drift (score of the noisy marginals);
    non-asymptotic Wasserstein rate.
  - **Insight**: this **is** the sampling-as-learning-score story, but
    with explicit constants — useful when we want to re-prove our
    [[score-contamination-lemma]] with named rates.
  - **Relevance**: high — cite as the rate we need for
    [[recursive-chi2-inequality]] at each generation.

- **2023-09 · Lipschitz Transport Maps via Föllmer Flow**
  (arxiv:2309.03490) · *Dai · Gao · Huang · Jiao · Kang · Liu*
  - **Problem**: under what conditions on the target does the Föllmer
    flow map have Lipschitz push-forward?
  - **Path**: log-concavity-like conditions on the target; bound the
    velocity field's gradient.
  - **Insight**: Lipschitz transport is what converts sample-level
    convergence into distribution-level convergence without extra
    regularity.
  - **Relevance**: a standing lemma we should import when we need to
    upgrade any W₂ bound to a functional bound.

---

## Cluster B — Neural-network approximation theory (5 core papers, all L1)

Central question: *how big a network do we need to approximate a class of
functions to accuracy ε, and does the depth–width trade-off escape curse
of dimensionality?* Jiao's team has systematic answers for Transformers,
RNNs, and constrained ReLU/RePU networks.

- **2025-04 · Transformers Can Overcome the Curse of Dimensionality**
  (arxiv:2504.13558) · *Jiao · Lai · Wang · Yan*
  - **Problem**: do Transformers really break the curse?
  - **Path**: approximate Hölder $\mathcal{C}^\beta$ functions via a
    KA-representation-inspired decomposition routed through attention.
  - **Insight**: rate depends on a *latent* intrinsic dimension of the
    function's argument structure, not $d$.
  - **Relevance**: theoretical backing for "it's OK to use wide
    Transformers for high-d financial series".

- **2025-04 · Approximation Bounds for Transformer Networks with
  Application to Regression** (arxiv:2504.12175) · *Jiao · Lai · Sun ·
  Wang · Yan*
  - **Problem**: non-parametric regression rate for Transformers under
    *dependent* observations.
  - **Path**: marry KA-representation-based approximation with
    β-mixing-adapted generalisation bounds.
  - **Insight**: close-to-minimax rate even under mixing, without ad-hoc
    assumptions.
  - **Relevance**: probably the cleanest citation when we justify
    Transformer backbones on financial time-series (which are mixing).

- **2024-09 · Approximation Bounds for RNNs with Application to
  Regression** (arxiv:2409.05577) · *Jiao · Wang · Yan*
  - **Problem**: what function class can deep ReLU RNNs approximate at
    what rate?
  - **Path**: direct construction; express hidden state as a composition
    of Hölder pieces.
  - **Insight**: minimax-optimal rates for standard smoothness classes —
    surprising because RNNs are often dismissed as theoretically opaque.
  - **Relevance**: baseline we could cite when comparing RNN vs
    Transformer backbones.

- **2023-05 · Differentiable NNs with RePU Activation: Score Estimation &
  Isotonic Regression** (arxiv:2305.00608) · *Shen · Jiao · Lin · Huang*
  - **Problem**: smoother activations → smoother networks → better score
    / gradient approximation.
  - **Path**: rectified-power-unit activations; analyse the regularity of
    the network class; apply to score matching and isotonic regression.
  - **Insight**: RePU is a sweet spot — smooth enough for gradient-based
    downstream use, still piecewise-analytic.
  - **Relevance**: candidate activation for our score networks if we
    need provable smoothness.

- **2023-04 · DNN Approximation of Composition Functions (with
  PINNs)** (arxiv:2304.07947) · *Duan · Jiao · Lu · Yang · Yuan*
  - **Problem**: approximate $f \circ g$ where both are smooth — tighter
    than each separately.
  - **Path**: compositional approximation argument; apply to PINN
    convergence.
  - **Insight**: composition often beats the single-Sobolev rate because
    network depth matches composition depth.
  - **Relevance**: the composition-class lens could be imported into our
    wavelet-level analysis (WaveletDiff stacks DWT + score + iDWT).

---

## Cluster C — Scientific computing / PINN / DRM (5 core papers)

Mostly tangential to our programme, but `DRM Revisited` is the clearest
complete-error-analysis template in the whole list.

- **2024-07 · DRM Revisited: A Complete Error Analysis**
  (arxiv:2407.09032) · *Jiao · Li · Wu · Yang · Zhang* · **L1**
  - **Problem**: settle the over-parameterised DRM: when does it
    converge, at what rate?
  - **Path**: the complete tri-decomposition (approximation +
    generalization + optimization) with explicit constants for each.
  - **Insight**: gives a *recipe* — how to pick width, depth, steps,
    learning rate — that matches the theoretical rate.
  - **Relevance**: the cleanest template we've seen for a unified
    tri-decomposition. Structure we might steal for finance-side
    analyses.

- **2024-05 · Error Analysis of 3-Layer NN Trained with PGD for DRM**
  (arxiv:2405.11451) · *Jiao · Lai · Wang* · **L0**
  - Specialisation to 3-layer case; not critical for us.

- **2024-03 · Stabilized PINNs for Wave Equations** (arxiv:2403.19090) ·
  *Jiao · Liu · Yang · Yuan* · **L0**

- **2023-06 · CDII-PINN** (arxiv:2306.13881) · *Duan · Jiao · Lu · Yang*
  · **L0**
  - Current-density impedance imaging via PINN + Tikhonov.

Note: the composition-approximation paper (arxiv:2304.07947) is also
partially in cluster C.

---

## Cluster D — Statistical learning theory (8 core papers, all L1)

Mostly applied-statistics results. A few worth tracking:

- **2025-05 · Wasserstein DRO Nonparametric Regression**
  (arxiv:2505.07967) · *Liu · Jiao · Wang · Huang*
  - **Problem**: nonparametric regression robust to distribution shift.
  - **Path**: local worst-case risk over a Wasserstein ball; analyse with
    deep nets as function class.
  - **Insight**: WDRO radius acts as an implicit regulariser; same rate
    as standard nonparam up to log factors.
  - **Relevance**: robust-regression flavour we could plug into
    out-of-regime finance experiments.

- **2025-02 · Distribution Matching for Self-Supervised Transfer
  Learning** (arxiv:2502.14424) · *Jiao · Ma · Sun · Wang · Wang*
  - **Problem**: SSL with a formal generalisation guarantee.
  - **Path**: drive representation distribution toward a *predefined
    reference* distribution while preserving augmentation invariance.
  - **Insight**: a principled target distribution sidesteps collapse in
    SSL; turns SSL into a distribution-matching problem.
  - **Relevance**: echoes our own concern about *distributional*
    collapse in [[self-consuming-diffusion-stability]].

- **2024-10 · Deep Transfer Learning: Framework + Error Analysis**
  (arxiv:2410.09383) · *Jiao · Lin · Luo · Yang*
  - **Problem**: quantify what we gain moving from upstream (large $n$)
    to downstream (small $m$).
  - **Path**: framework where both domains are learned nets; explicit
    error bounds.
  - **Insight**: downstream rate improves by a factor $\sqrt{n/m}$ when
    upstream representation is faithful — formalises the "pre-train +
    fine-tune" intuition.
  - **Relevance**: pretrain-on-market, fine-tune-per-stock story can
    reuse this bound.

- **2024-08 · Adv-SSL: Adversarial Self-Supervised Rep. Learning**
  (arxiv:2408.08533) · *Duan · Jiao · Lin · Ma · Yang*
  - **Problem**: SSL with unbiased risk estimation + theory.
  - **Path**: adversarial inner-max over augmentations.
  - **Insight**: adversarial reformulation gives unbiased risk and an
    explicit generalisation bound; SGD suffices for optimisation.

- **2024-05 · Model-Free Prediction with Uncertainty Assessment**
  (arxiv:2405.12684) · *Jiao · Kang · Liu · Peng · Zuo*
  - **Problem**: CI for deep nonparametric regression.
  - **Path**: asymptotic normality of the prediction estimator.
  - **Insight**: enables plug-in confidence regions without quantile
    regression.

- **2024-01 · Semi-Sup. Deep Sobolev Regression (ReQU NN)**
  (arxiv:2401.04535) · *Ding · Duan · Jiao · Yang*
  - **Problem**: simultaneous regression + gradient estimation + variable
    selection.
  - **Path**: Sobolev loss (value + gradient) with ReQU nets.
  - **Insight**: one loss, two estimators — gradient estimate comes with
    a matching rate.

- **2023-12 · NN Approximation for Pessimistic Offline RL**
  (arxiv:2312.11863) · *Wu · Jiao · Shen · Yang · Lu*
  - **Problem**: offline RL with C-mixing data and neural nets.
  - **Path**: pessimistic value estimation; non-asymptotic bounds.
  - **Insight**: mixing-aware concentration handles temporal dependence
    without iid assumptions.
  - **Relevance**: closest relative to a finance-decision-making story;
    the mixing-data tooling here is transferable.

- **2023-09 · Non-Asymptotic Bounds for Adversarial Excess Risk under
  Misspecification** (arxiv:2309.00771) · *Liu · Jiao · Wang · Huang*
  - **Problem**: adversarial-loss regression when the model is misspecified.
  - **Path**: decompose excess risk into estimation + misspecification.
  - **Insight**: adversarial robustness doesn't blow up under
    misspecification if the ambiguity set is chosen by structure.

---

## Clusters E & F (dropped — listed for completeness)

Not distilled. Keep metadata in
`wiki/raw/author/jiao-yuling.json` for future pickups.

- **E: Quantum ML** — `arxiv:2510.18208`, `arxiv:2507.07601`,
  `arxiv:2406.12195`, `arxiv:2310.07528`.
- **F: LLM theory** — `arxiv:2603.10000`, `arxiv:2505.10594`.

---

## Connections

Jiao's programme plugs into ours along 3 seams:

- **grounds** [[girsanov-path-kl-bound]] — his Föllmer-flow rate papers
  (`arxiv:2311.03660`, `arxiv:2309.03490`, `arxiv:2311.11475`) give the
  explicit constants we currently treat as placeholders.
- **bounds-input-for** [[ntk-score-error-decomposition]] — his
  tri-decomposition style in
  [[cnf-convergence-distribution-learning]] and DRM-Revisited is the
  exact pattern we use; citing these strengthens our proof.
- **extends** [[cofindiff-controllable-financial-diffusion]] via
  [[diffusion-doob-matching-inference-alignment]] — inference-time
  alignment *without retraining*. Potentially cheaper alternative to
  CoFinDiff's classifier-free path.
- **parallels** [[synthetic-augmentation-financial-timeseries]] — his
  `arxiv:2505.04992` (synthetic-data boosting) asks the exact same
  question we care about, with provable guarantees.

## Contradictions

_None identified yet — needs closer reading of the 7 L2 papers before we
claim any._

## Tracking

- [ ] Deep-read the 7 L2 picks (fill articles/ fully).
- [ ] Create stub entries for top coauthors: `yang-jerry-zhijian`,
      `duan-chenguang`, `huang-jian`. (Deferred per user instruction.)
- [ ] Watch for: (a) any new Föllmer-flow / stochastic-interpolant paper,
      (b) any extension of the synthetic-data boosting line to time-series.
- [ ] Re-run arxiv author query monthly; diff against `raw/author/jiao-yuling.json`.

---

## Coauthor log (record-only, promote to `authors/` later)

Frequency in our 3-year window (>= 3 co-authorships). These are candidates
for their own `authors/<slug>.md` pages when we get there.

| Coauthor | # papers with Jiao | Affiliation guess | Priority |
|---|---|---|---|
| Jerry Zhijian Yang | 17 | Wuhan Math | **high** — almost every paper |
| Chenguang Duan | 8 | Wuhan (student?) | **high** — Föllmer-flow driver |
| Jian Huang | 6 | HK PolyU / Iowa | medium |
| Cheng Yuan | 5 | Wuhan | medium |
| Xiliang Lu | 5 | Wuhan | medium |
| Yang Wang | 5 | HKUST | medium |
| Yanming Lai | 4 | Wuhan | low |
| Bokai Yan | 4 | HKUST | low |
| Zhao Ding | 4 | Wuhan (student?) | medium (Föllmer) |
| Ruoxuan Li | 4 | Wuhan | low |
| Huazhen Lin | 4 | SWUFE? | low |
| Lican Kang | 4 | ? | low |
| Jinyuan Chang | 3 | SWUFE? | medium (recent diffusion line) |
| Defeng Sun | 3 | HK PolyU | low |
| Pingwen Zhang | 3 | Peking | low |

## Raw metadata pointers

- Author query cache: `wiki/raw/author/jiao-yuling.json`
- Individual paper caches (to be added during L2 deep-read):
  `wiki/raw/arxiv/<arxiv-id>.json`
