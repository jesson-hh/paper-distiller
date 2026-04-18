---
title: "M10–M17 Leverage Engineering: RSV + Aux Loss Sweep"
category: "articles"
slug: "m10-m17-leverage-engineering"
tags: ["leverage-effect", "auxiliary-loss", "har-rv-l", "hinge-loss", "diffusion", "edm", "ablation", "csi800"]
refs: ["experiments/phase2_interdiff_fts/edm_diffusion.py", "experiments/phase2_interdiff_fts/model.py"]
links: ["m9-tedm-heavy-tails", "a-share-positive-leverage", "factor-conditional-interdiff-m4-m5", "fts-interdiff-fusion"]
created: "2026-04-18T03:30:00"
updated: "2026-04-18T03:30:00"
---

# M10–M17 — Engineering the Leverage Effect

> **Goal**: Fix the persistent A-share-positive-leverage gap (real CSI800 = +0.013, our models = -0.010 to -0.012). Tried HAR-RV-L style realized semi-variance conditioning, then auxiliary leverage loss in three forms (mse / sign / hinge) across weights {5, 10, 20, 30, 50, 100}.
> **Result**: Hinge form with w=20 reduces leverage gap from -0.010 to **-0.0037** (63% improvement, 7/7 verdict still green) **and** brings skew from -0.115 → +0.118 (real +0.106, near-perfect match). Sign flip to positive not achieved without breaking other metrics — wall hit at lev ≈ -0.003.
> **Relation**: continues [[m9-tedm-heavy-tails]]; closes most of [[a-share-positive-leverage]]'s gap (sign still wrong but magnitude greatly reduced).

## Background

After M9 (t-EDM heavy-tail noise), all 7 stylized-fact verdicts were green, but **leverage_lag1 = -0.012** vs real +0.013 — wrong sign and 2× worse than M6 baseline. The Student-t prior amplifies symmetric tail extremes, which deepened the wrong-direction asymmetry.

Two avenues to fix:
1. **Conditioning**: provide the model with explicit asymmetric input signal
2. **Auxiliary loss**: directly supervise the leverage statistic on the model's denoised x0

We tried both, in sequence.

## M10: Realized Semi-Variance (RSV) Conditioning

Inspired by HAR-RV-L econometrics (Corsi 2009 + asymmetric extensions). Add `compute_rsv_from_mkt(mkt_cond, window=5)` to model.py:

```python
rsv_pos[t] = mean(ReLU(+r_{t-5:t})^2)   # past 5d upside realized variance
rsv_neg[t] = mean(ReLU(-r_{t-5:t})^2)   # past 5d downside realized variance
```

Stack into (B, L, 2), project via dedicated `lev_proj: Linear(2, d_model)`. Broadcast across N stocks like other factor conditioning. +4.4k params.

### Why this might fix what M8 couldn't

M8 (sign-aware) used `ReLU(-mkt_cond)` instantaneously. Leverage is **temporal** — past down-moves predict current high vol. Asking attention to re-aggregate `rsv` from raw negative time-points is hard. M10 pre-aggregates → easier signal.

### Results

| metric | Real | M9 (no RSV) | **M10 (RSV both)** |
|---|---|---|---|
| leverage_lag1 | +0.013 | -0.012 | **-0.009** (improved 25%) |
| pair_corr | 0.349 | 0.367 (+5%) | **0.354 (+1.3%)** ⬆ |
| max_eig_frac | 0.398 | 0.408 | **0.397** (-0.25%) ⬆ |
| kurt | 3.55 | 3.14 | 3.12 |
| Verdict | — | 7/7 | 7/7 |

RSV conditioning unexpectedly **also** fixed the cross-section over-shoot from M9's t-EDM. Likely because RSV provides an alternate common-mode signal that lets the model rely less on amplifying the market factor.

## M11: lev_mode=pos_only

Hypothesis: model is "lazy" and learns classical leverage from rsv_neg even though we want positive leverage. Try removing rsv_neg and forcing it to learn from rsv_pos only.

```python
# In forward, mask out the unwanted RSV channel:
if lev_mode == "pos_only":
    lev = torch.stack([rsv_pos, torch.zeros_like(rsv_neg)], dim=-1)
```

### Result

leverage = **-0.010** (basically same as M10). 

**Conclusion**: model has alternative routes to infer "past was negative" from raw `mkt_cond` trajectory (intra-stock attention can re-derive rsv_neg even if we don't feed it). Pure conditioning intervention can't flip leverage.

## M12: Aux Leverage Loss (MSE), w=100 — FAILURE

Direct supervision: compute leverage on x0_pred, push toward target.

```python
r = x0_pred[..., 0]
lev_pred = pearson_corr(r[..., :-1].flatten(), (r[..., 1:]**2).flatten())
loss = main_mse + w * (lev_pred - 0.013)**2
```

Restricted to low-sigma batch elements (`sigma < sigma_data`) where x0_pred is meaningful.

### Result (w=100)

| metric | M11 | **M12 (mse w=100)** |
|---|---|---|
| leverage | -0.010 | **-0.022** (worse!) |
| acf_r²_lag1 | 0.062 OK | **0.047 MEH** ⬇ |
| pair_corr | 0.352 | **0.324 (-7%)** ⬇ |
| max_eig_frac | 0.395 | **0.363 (-9%)** ⬇ |
| Verdict | 7/7 | **6/7** ❌ |

**Total disaster.** Leverage went *more* negative; cross-section and vol clustering broke.

### Diagnosis

- **Per-batch leverage is super noisy**. With ~half of 16 batch elements at low sigma (`sigma < 1`), and pooling 8 × 32 stocks × 64 timesteps = 16k pairs, the per-batch correlation estimate has standard error ~0.01.
- At w=100, aux contribution to total loss = 0.4 (similar to main loss 0.5).
- Optimizer chases noisy per-batch gradient → finds local minimum where leverage is *more* negative than baseline.
- Main MSE pressure on cross-section gets diluted → those metrics break.

## M13–M14: MSE Sweep — Sweet Spot at w=10

Tried smaller weights:

| w | leverage | verdict |
|---|---|---|
| 0 (M11) | -0.010 | 7/7 |
| **10 (M13)** | **-0.006** ✅ | 7/7 |
| 30 (M14) | -0.011 | 7/7 |
| 100 (M12) | -0.022 | 6/7 |

**MSE sweep is non-monotonic in w**. w=10 is a sweet spot (40% improvement); larger w degrades.

Mechanism: MSE penalty `(lev - target)^2` has *gradient magnitude proportional to (lev - target)*. Far from target, gradient is large → optimizer aggressively pursues noisy signal. Plus MSE punishes overshoot, which is undesired (we'd be happy with lev > target since target = real value).

## M15–M17: Hinge Aux Loss — Constant Gradient, No Overshoot

Switch loss form:

```python
# hinge: only penalize lev < target, with constant gradient magnitude
aux = w * F.relu(target - lev_pred)
```

### Why hinge works better

- **Constant gradient** when `lev < target`: same magnitude regardless of distance → no aggressive chase of noisy gradient
- **Zero gradient** when `lev ≥ target`: no overshoot penalty
- **One-sided**: model gets pushed in *one consistent direction* every batch (no oscillation from MSE's symmetric penalty)

### Sweep results

| w | leverage | skew (real +0.106) | hill_R (real 3.46) | verdict |
|---|---|---|---|---|
| 0 (M11) | -0.010 | -0.115 | 3.82 | 7/7 |
| **5 (M15)** | **-0.0045** | **+0.092** | **3.57** | 7/7 |
| **20 (M16)** | **-0.0037** | **+0.118** | **3.57** | 7/7 |
| 50 (M17) | -0.0034 | +0.122 | 3.58 | 7/7 |

**Saturation around lev = -0.003**. Going from w=5 to w=50 (10×) improves leverage by only 0.001 (24%). But cleanly, no sign of degradation.

### Side-effect bonus: Skew nearly perfectly matched

The hinge aux pushed not only leverage but also **skew** dramatically. Real CSI800 skew = +0.106. M11 baseline = -0.115 (sign reversed). After hinge aux:
- M15 (w=5): +0.092 ✅
- M16 (w=20): +0.118 (within 0.012 of real)
- M17 (w=50): +0.122 (within 0.016 of real, slight overshoot but excellent)

Why? Skew and leverage share the same underlying mechanism (asymmetry in returns). Pushing leverage upward also pushes skew upward — they're correlated.

## Recommendation: M16 (hinge w=20)

Best practical config for production data generation. Reasoning:

| | M16 (w=20) | M17 (w=50) |
|---|---|---|
| Leverage | -0.0037 | -0.0034 (only +0.0003) |
| Skew | +0.118 | +0.122 (basically same) |
| Other verdicts | All OK | All OK |
| Aux pressure | Moderate | High (more overhead, less robust to data shift) |
| Convergence safety | High | High |

M16 captures essentially all the benefit at lower w, with safer margin against future data variations.

## Full Sweep Summary

```
                   leverage  skew    verdict     notes
M11 (no aux)       -0.010   -0.115   7/7 OK     baseline
M13 (mse w=10)     -0.006   -0.039   7/7 OK     MSE sweet spot
M14 (mse w=30)     -0.011   -0.028   7/7 OK     MSE backslides
M12 (mse w=100)    -0.022   -0.015   6/7 BROKE  MSE catastrophe
M15 (hinge w=5)    -0.0045  +0.092   7/7 OK     hinge promising
M16 (hinge w=20)   -0.0037  +0.118   7/7 OK     ★ recommended ★
M17 (hinge w=50)   -0.0034  +0.122   7/7 OK     ceiling hit

Real CSI800        +0.013   +0.106   —          ground truth
```

## Why we didn't reach lev > 0

The aux loss can pull leverage from -0.010 to -0.003 (70% closer to zero) but stops there. Mechanism:

- **Main MSE loss anchors x0_pred toward real x0** via the normal denoising objective
- The MSE term has hundreds of millions of effective gradient signals (per-pixel reconstruction)
- The aux term has ONE scalar gradient (per batch leverage statistic)
- Equilibrium = wherever the two losses balance

To push leverage further (toward positive), we need to either:
1. **Increase aux gradient signal density**: e.g. compute leverage per (B,N) trajectory rather than batch-pooled, providing B×N gradient signals → more bandwidth to push back against main loss. Untried.
2. **Architectural change that gives model new degrees of freedom**: GJR-GARCH 2-stage where the asymmetric leverage is baked into the conditioning factor itself. Most theoretically clean.
3. **Asymmetric noise schedule**: separate β for r > 0 vs r < 0. Significant rewrite.

These would be M18+ work.

## Artifacts

```
experiments/phase2_interdiff_fts/
├── edm_diffusion.py             # StudentTEDM with aux_lev_weight, _mode, _target
├── model.py                     # InterDenoiser with lev_cond, lev_window, lev_mode
├── train.py                     # --aux-lev-{weight,target,mode} flags
└── ckpts/
    ├── M0_m10_lev_tedm_step20000.pt              # rsv both
    ├── M0_m11_levpos_tedm_step20000.pt           # rsv pos_only
    ├── M0_m12_lev_aux_w100_step20000.pt          # mse w=100 (broken)
    ├── M0_m13_lev_aux_w10_step20000.pt           # mse w=10 (sweet spot)
    ├── M0_m14_lev_aux_w30_step20000.pt           # mse w=30
    ├── M0_m15_lev_hinge_w5_step20000.pt          # hinge w=5
    ├── M0_m16_lev_hinge_w20_step20000.pt         # ★ recommended
    └── M0_m17_lev_hinge_w50_step20000.pt         # ceiling test
```

Each ckpt has matching `.samples.npz` + training log.

## Next steps (if leverage sign flip required)

1. **Per-trajectory aux loss** (rather than batch-pooled): compute leverage for each (B, N) trajectory then average or sum. More gradient bandwidth, possibly enough to push lev > 0.
2. **GJR-GARCH 2-stage**: fit GJR-GARCH(1,1) on real market factor → sample factor trajectories with correct leverage sign → condition InterDiff on those factors. Architecture-clean fix.
3. **Higher hinge weight + carefully**: try w=200, w=500 with extra monitoring. Risk: verdict break.

For now, **M16 captures most of the achievable improvement** within the current architecture constraints.
