# Factor-Conditional Panel Generation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the 22% market-factor-variance gap in synthetic panels by conditioning InterDiff on an explicit market factor signal.

**Architecture:** Extract equal-weight market factor from real panel windows during training, inject via additive projection into the denoiser. At sample time, bootstrap market factor windows from real data and condition the reverse diffusion on them.

**Tech Stack:** PyTorch, numpy. All changes in `experiments/phase2_interdiff_fts/`.

**Spec:** `docs/superpowers/specs/2026-04-16-factor-conditional-generation-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `panel_windows.py` | Modify | Add market factor computation, yield 3-tuple |
| `model.py` | Modify | Add `mkt_proj`, `mkt_cond` param to forward |
| `diffusion.py` | Modify | Thread `mkt_cond` through `training_loss` and `sample` |
| `train.py` | Modify | Unpack 3-tuple batch, pass `mkt_cond` to diffusion |
| `sample.py` | Modify | Bootstrap market factor windows, pass to sampler |
| `_verify_mkt_cond.py` | Create | Smoke test: load dataset, build model, run 1 fwd/bwd |

---

### Task 1: PanelWindowDataset — yield market factor

**Files:**
- Modify: `experiments/phase2_interdiff_fts/panel_windows.py:142-157`

The dataset currently yields `(window,)` or `(window, regime_labels)`. We add market factor as a third element. The market factor is the equal-weight mean of log_ret (channel 0) across the k sampled stocks.

- [ ] **Step 1: Modify `__iter__` to compute and yield market factor**

In `panel_windows.py`, replace the `__iter__` method (lines 142-157):

```python
    def __iter__(self):
        rng = self._rng
        L = self.length
        k = self.k_stocks
        while True:
            s = int(rng.choice(self._valid_starts))
            ok_stocks = np.where(self._stock_full_mask_per_start[:, s])[0]
            if ok_stocks.size < k:
                continue
            picks = rng.choice(ok_stocks, size=k, replace=False)
            window = self.returns[picks, s : s + L, :]
            # Market factor: equal-weight mean of log_ret across sampled stocks
            mkt = window[:, :, 0].mean(axis=0).astype(np.float32)  # (L,)
            if self.regime_labels is not None:
                lab = self.regime_labels[picks, s : s + L]
                yield torch.from_numpy(window), torch.from_numpy(lab), torch.from_numpy(mkt)
            else:
                yield torch.from_numpy(window), torch.from_numpy(mkt)
```

- [ ] **Step 2: Verify dataset iteration**

```bash
cd experiments/phase2_interdiff_fts
python -c "
from panel_windows import PanelWindowDataset
ds = PanelWindowDataset('data/csi300_2015_2024.npz', length=32, k_stocks=16, seed=0, normalise=True, regime_window=10, n_regimes=8)
it = iter(ds)
batch = next(it)
print(f'items: {len(batch)}')
print(f'window: {batch[0].shape}')
print(f'regime: {batch[1].shape}')
print(f'mkt:    {batch[2].shape}  mean={batch[2].mean():.4f}  std={batch[2].std():.4f}')
"
```

Expected: `items: 3`, `mkt: torch.Size([32])`, mean near 0, std near 1.

- [ ] **Step 3: Commit**

```bash
git add experiments/phase2_interdiff_fts/panel_windows.py
git commit -m "feat(panel_windows): yield market factor as third batch element"
```

---

### Task 2: InterDenoiser — add market factor conditioning

**Files:**
- Modify: `experiments/phase2_interdiff_fts/model.py:101-155`

Add a 2-layer MLP that projects the scalar market factor at each timestep into d_model space, then adds it to all stock embeddings (broadcast across N).

- [ ] **Step 1: Add `mkt_cond` support to `__init__` and `forward`**

In `model.py`, modify `InterDenoiser.__init__` — add after the `time_mlp` definition (after line 130):

```python
        # Market factor conditioning: (B, L) -> (B, 1, L, d_model)
        self.mkt_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
```

Modify `InterDenoiser.forward` signature and body — replace lines 139-154:

```python
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
        mkt_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, L, C = x.shape
        h = self.in_proj(x)
        h = h + self.t_pos[:, :, :L, :] + self.s_pos[:, :N, :, :]

        if cond is not None and self.n_regimes > 0:
            h = h + self.regime_embed(cond)  # (B, N, L, d)

        if mkt_cond is not None:
            # mkt_cond: (B, L) -> (B, 1, L, 1) -> project -> (B, 1, L, d) -> broadcast
            me = self.mkt_proj(mkt_cond[:, None, :, None])  # (B, 1, L, d_model)
            h = h + me  # broadcast across N stocks

        te = sinusoidal_time_embedding(t, self.d_model)
        te = self.time_mlp(te)[:, None, None, :]
        h = h + te

        for blk in self.blocks:
            h = blk(h)

        return self.out(self.ln_out(h))
```

- [ ] **Step 2: Verify model builds and forward passes**

```bash
cd experiments/phase2_interdiff_fts
python -c "
import torch
from model import InterDenoiser, count_params
m = InterDenoiser(n_channels=4, max_length=64, max_stocks=32, d_model=128, n_blocks=6, n_heads=8, n_regimes=8)
print(f'params: {count_params(m):,}')
x = torch.randn(2, 16, 32, 4)
t = torch.randint(0, 500, (2,))
cond = torch.randint(0, 8, (2, 16, 32))
mkt = torch.randn(2, 32)
out = m(x, t, cond=cond, mkt_cond=mkt)
print(f'out: {out.shape}')
# Also check without mkt_cond (backward compat)
out2 = m(x, t, cond=cond)
print(f'out (no mkt): {out2.shape}')
"
```

Expected: param count slightly higher than before (~3M), out shapes `[2, 16, 32, 4]`.

- [ ] **Step 3: Commit**

```bash
git add experiments/phase2_interdiff_fts/model.py
git commit -m "feat(model): add market factor conditioning via mkt_proj MLP"
```

---

### Task 3: GaussianDiffusion — thread mkt_cond

**Files:**
- Modify: `experiments/phase2_interdiff_fts/diffusion.py:52-117`

Both `training_loss` and `sample` need to accept and forward `mkt_cond` to the model.

- [ ] **Step 1: Add `mkt_cond` parameter to `training_loss`**

In `diffusion.py`, modify `training_loss` (lines 52-73) — change signature and the model call:

```python
    def training_loss(
        self,
        model,
        x0: torch.Tensor,
        cond: torch.Tensor | None = None,
        mkt_cond: torch.Tensor | None = None,
        aux_market_weight: float = 0.0,
    ) -> torch.Tensor:
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        eps = model(xt, t, cond=cond, mkt_cond=mkt_cond)
        base = F.mse_loss(eps, noise)
        if aux_market_weight <= 0.0:
            return base
        mkt_true = noise.mean(dim=1, keepdim=True)
        mkt_pred = eps.mean(dim=1, keepdim=True)
        res_true = noise - mkt_true
        res_pred = eps - mkt_pred
        loss_mkt = F.mse_loss(mkt_pred, mkt_true)
        loss_res = F.mse_loss(res_pred, res_true)
        return loss_res + aux_market_weight * loss_mkt
```

- [ ] **Step 2: Add `mkt_cond` parameter to `sample`**

In `diffusion.py`, modify `sample` (lines 75-117) — change signature and the model call inside the loop:

```python
    @torch.no_grad()
    def sample(
        self,
        model,
        shape,
        progress: bool = False,
        clip_denoised: bool = True,
        clip_range: float = 5.0,
        cond: torch.Tensor | None = None,
        mkt_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = torch.randn(shape, device=self.device)
        iters = range(self.T - 1, -1, -1)
        if progress:
            try:
                from tqdm import tqdm
                iters = tqdm(iters, desc="sample")
            except ImportError:
                pass
        for i in iters:
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            eps = model(x, t, cond=cond, mkt_cond=mkt_cond)
            beta = self.betas[i]
            alpha = self.alphas[i]
            ac = self.alpha_cum[i]
            sa_cum = self.sqrt_alpha_cum[i]
            sb_cum = self.sqrt_one_minus_alpha_cum[i]

            x0_pred = (x - sb_cum * eps) / sa_cum
            if clip_denoised:
                x0_pred = x0_pred.clamp(-clip_range, clip_range)

            if i > 0:
                ac_prev = self.alpha_cum[i - 1]
                coef_x0 = (torch.sqrt(ac_prev) * beta) / (1 - ac)
                coef_xt = (torch.sqrt(alpha) * (1 - ac_prev)) / (1 - ac)
                mean = coef_x0 * x0_pred + coef_xt * x
                var = beta * (1 - ac_prev) / (1 - ac)
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(var) * noise
            else:
                x = x0_pred
        return x
```

- [ ] **Step 3: Commit**

```bash
git add experiments/phase2_interdiff_fts/diffusion.py
git commit -m "feat(diffusion): thread mkt_cond through training_loss and sample"
```

---

### Task 4: train.py — unpack and pass market factor

**Files:**
- Modify: `experiments/phase2_interdiff_fts/train.py:182-227`

The batch now has 2 or 3 elements. We need to unpack the market factor and pass it through.

- [ ] **Step 1: Update `_split_batch` to handle 3-tuple**

In `train.py`, replace `_split_batch` (lines 182-185) and update the training loop call:

```python
    def _split_batch(b):
        """Unpack batch -> (panel, regime_cond_or_None, mkt_cond_or_None)."""
        if isinstance(b, (list, tuple)):
            if len(b) == 3:
                # (window, regime_labels, mkt_factor)
                return b[0].to(device), b[1].to(device), b[2].to(device)
            elif len(b) == 2:
                # Could be (window, regime_labels) or (window, mkt_factor)
                # Distinguish: regime_labels is int64, mkt_factor is float32
                if b[1].dtype == torch.long:
                    return b[0].to(device), b[1].to(device), None
                else:
                    return b[0].to(device), None, b[1].to(device)
        return b.to(device), None, None
```

- [ ] **Step 2: Update preflight to unpack 3 values**

In `train.py`, update the preflight block (around lines 190-204). Replace:

```python
            preflight, pre_cond = _split_batch(next(it_pre))
```

with:

```python
            preflight, pre_cond, pre_mkt = _split_batch(next(it_pre))
```

And replace the preflight loss call:

```python
            loss_pre = diff.training_loss(model, preflight, cond=pre_cond, mkt_cond=pre_mkt, aux_market_weight=args.aux_market_weight)
```

And update the `del` line:

```python
            del preflight, loss_pre, it_pre, pre_cond, pre_mkt
```

- [ ] **Step 3: Update training loop to pass mkt_cond**

In `train.py`, replace the training loop unpack and loss call (around lines 225-227):

```python
        batch, cond, mkt = _split_batch(batch)

        loss = diff.training_loss(model, batch, cond=cond, mkt_cond=mkt, aux_market_weight=args.aux_market_weight)
```

- [ ] **Step 4: Save mkt_cond flag in checkpoint**

In `train.py`, in the checkpoint save block (around line 267), add `"mkt_cond": True` to `save_obj`:

```python
            save_obj = {
                "model": model.state_dict(),
                "args": vars(args),
                "step": step,
                "loss_ema": ema_loss,
                "stats": {"mean": ds.mean, "std": ds.std},
                "ds_info": ds.info(),
                "mkt_cond": True,
            }
```

- [ ] **Step 5: Commit**

```bash
git add experiments/phase2_interdiff_fts/train.py
git commit -m "feat(train): unpack market factor from batch, pass to diffusion"
```

---

### Task 5: sample.py — bootstrap market factor for generation

**Files:**
- Modify: `experiments/phase2_interdiff_fts/sample.py`

At sample time we don't have a "real" market factor to condition on — we bootstrap random L-length windows from the real market factor series.

- [ ] **Step 1: Add market factor bootstrap to sample.py**

Replace the full `main()` function in `sample.py`:

```python
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ck["args"]
    print(f"[sample] ckpt step={ck.get('step')} loss_ema={ck.get('loss_ema'):.4f}")

    regime_spec = None
    if "regime_spec" in ck:
        regime_spec = RegimeSpec.from_dict(ck["regime_spec"])
        print(f"[sample] regime: K={regime_spec.n_regimes} window={regime_spec.window}")

    use_mkt_cond = ck.get("mkt_cond", False)
    print(f"[sample] mkt_cond={use_mkt_cond}")

    model = InterDenoiser(
        n_channels=4,
        max_length=cfg["length"],
        max_stocks=cfg["k"],
        d_model=cfg["d_model"],
        n_blocks=cfg["n_blocks"],
        n_heads=cfg["n_heads"],
        n_regimes=regime_spec.n_regimes if regime_spec is not None else 0,
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    diff = GaussianDiffusion(T=cfg["T"], device=device)

    L = cfg["length"]
    K = cfg["k"]
    C = 4

    # --- prepare conditioning sources ---
    ds_source = None
    if regime_spec is not None or use_mkt_cond:
        ds_source = PanelWindowDataset(
            panel_npz=args.panel,
            length=L,
            k_stocks=K,
            seed=args.seed,
            normalise=True,
            time_range=("2015-01-05", cfg.get("train_end", "2022-12-31")),
            regime_window=regime_spec.window if regime_spec else 0,
            n_regimes=regime_spec.n_regimes if regime_spec else 0,
        )
        source_iter = iter(ds_source)

    all_samples = []
    for bi in range(args.n_batches):
        cond_batch = None
        mkt_batch = None

        if ds_source is not None:
            cond_list = []
            mkt_list = []
            for _ in range(args.batch):
                item = next(source_iter)
                if isinstance(item, (list, tuple)):
                    if len(item) == 3:
                        _, lab, mkt = item
                        cond_list.append(lab)
                        mkt_list.append(mkt)
                    elif len(item) == 2:
                        if item[1].dtype == torch.long:
                            cond_list.append(item[1])
                        else:
                            mkt_list.append(item[1])
                else:
                    pass  # no conditioning
            if cond_list:
                cond_batch = torch.stack(cond_list, dim=0).to(device)
            if mkt_list and use_mkt_cond:
                mkt_batch = torch.stack(mkt_list, dim=0).to(device)

        x = diff.sample(model, shape=(args.batch, K, L, C),
                        cond=cond_batch, mkt_cond=mkt_batch)
        all_samples.append(x.cpu().numpy())
        print(f"[sample] batch {bi+1}/{args.n_batches}")

    samples = np.concatenate(all_samples, axis=0)
    print(f"[sample] panel batch tensor: {samples.shape}")

    flat = samples.reshape(-1, L, C)
    print(f"[sample] flattened (stock-trajectory, L, C): {flat.shape}")

    mean = ck["stats"]["mean"]
    std = ck["stats"]["std"]
    avg_mean = mean.mean(axis=0)
    avg_std = std.mean(axis=0)
    flat_denorm = flat * avg_std[None, None, :] + avg_mean[None, None, :]
    print("[sample] de-normalised using avg per-stock stats")

    panels_denorm = samples * avg_std[None, None, None, :] + avg_mean[None, None, None, :]

    out = Path(args.out) if args.out else Path(args.ckpt).with_suffix(".samples.npz")
    np.savez_compressed(
        out,
        windows=flat,
        windows_denorm=flat_denorm,
        panels=samples,
        panels_denorm=panels_denorm,
        ret_fields=np.array(["log_ret", "log_hc", "log_lc", "log_oc"]),
        ckpt=str(args.ckpt),
    )
    print(f"[sample] saved -> {out.resolve()}")
    print(f"[sample] windows stats:  mean={flat_denorm[..., 0].mean():.5f}  "
          f"std={flat_denorm[..., 0].std():.5f}  "
          f"min={flat_denorm[..., 0].min():.4f}  "
          f"max={flat_denorm[..., 0].max():.4f}")
```

- [ ] **Step 2: Commit**

```bash
git add experiments/phase2_interdiff_fts/sample.py
git commit -m "feat(sample): bootstrap market factor from real data for conditional sampling"
```

---

### Task 6: Smoke test — end-to-end forward/backward

**Files:**
- Create: `experiments/phase2_interdiff_fts/_verify_mkt_cond.py`

Quick verification that the full pipeline (dataset -> model -> loss -> backward) works before launching a real training run.

- [ ] **Step 1: Create smoke test script**

```python
"""Smoke test: dataset -> model -> diffusion loss -> backward with mkt_cond."""
import torch
from panel_windows import PanelWindowDataset
from model import InterDenoiser, count_params
from diffusion import GaussianDiffusion

ds = PanelWindowDataset(
    "data/csi300_2015_2024.npz",
    length=32, k_stocks=16, seed=0, normalise=True,
    regime_window=10, n_regimes=8,
)
it = iter(ds)
items = [next(it) for _ in range(4)]

windows = torch.stack([x[0] for x in items])
conds = torch.stack([x[1] for x in items])
mkts = torch.stack([x[2] for x in items])
print(f"windows: {windows.shape}  conds: {conds.shape}  mkts: {mkts.shape}")

model = InterDenoiser(
    n_channels=4, max_length=32, max_stocks=16,
    d_model=64, n_blocks=3, n_heads=4, n_regimes=8,
)
print(f"params: {count_params(model):,}")

diff = GaussianDiffusion(T=200, device="cpu")
loss = diff.training_loss(model, windows, cond=conds, mkt_cond=mkts)
print(f"loss: {loss.item():.4f}")
loss.backward()
print("backward OK")

# Also verify sampling with mkt_cond
model.eval()
with torch.no_grad():
    s = diff.sample(model, shape=(2, 16, 32, 4), cond=conds[:2], mkt_cond=mkts[:2])
    print(f"sample: {s.shape}  mean={s.mean():.4f}  std={s.std():.4f}")

print("ALL OK")
```

- [ ] **Step 2: Run smoke test**

```bash
cd experiments/phase2_interdiff_fts
python _verify_mkt_cond.py
```

Expected: prints shapes, loss ~1.0, "backward OK", sample shape, "ALL OK".

- [ ] **Step 3: Commit**

```bash
git add experiments/phase2_interdiff_fts/_verify_mkt_cond.py
git commit -m "test: add mkt_cond smoke test"
```

---

### Task 7: Train M4 — market-factor-conditioned model

**Files:**
- No code changes; this is a training run.

- [ ] **Step 1: Launch training (GPU, 20k steps)**

```bash
cd experiments/phase2_interdiff_fts
python train.py \
    --panel data/csi300_2015_2024.npz \
    --length 64 --k 32 --batch 16 \
    --steps 20000 --lr 2e-4 \
    --T 500 --d-model 128 --n-blocks 6 --n-heads 8 \
    --regime-window 10 --n-regimes 8 \
    --tag m4_mkt \
    --log-every 100 --ckpt-every 2000 \
    2>&1 | tee ckpts/m4_mkt_train.log
```

Monitor: loss should converge to ~0.19-0.20 (similar to m1/m3).

- [ ] **Step 2: Sample from the trained model**

```bash
python sample.py \
    --ckpt ckpts/M0_m4_mkt_step20000.pt \
    --n-batches 50 --batch 8 \
    --panel data/csi300_2015_2024.npz
```

- [ ] **Step 3: Evaluate**

```bash
python eval_compare.py --samples ckpts/M0_m4_mkt_step20000.samples.npz
python _diag_cs.py m4_mkt
```

Success criteria:
- market_factor_var gap < 10% (from 22%)
- panel_mean_pairwise_corr gap < 8% (from 15%)
- All existing OK metrics must not degrade

- [ ] **Step 4: Commit results log**

```bash
git add ckpts/m4_mkt_train.log
git commit -m "exp: M4 market-factor-conditioned training 20k steps"
```

---

### Task 8: Compare M4 vs M1/M3 and decide on Stage B

**Files:**
- No code changes; analysis only.

- [ ] **Step 1: Run side-by-side eval**

Compare the eval_compare output for m4_mkt against m1_big and m3_big from the earlier runs. Key metrics:

| Metric | m1 | m3 | m4_mkt | Target |
|--------|----|----|--------|--------|
| market_factor_var gap | -22% | -23% | ? | < 10% |
| panel_mean_pair_corr gap | -13% | -14% | ? | < 8% |
| acf_r2_lag1 | OK | OK | ? | OK |
| hill_right | OK | OK | ? | OK |
| leverage_lag1 | 0.011 | 0.011 | ? | > 0.01 |

- [ ] **Step 2: Decision point**

- If gap < 10%: Stage A is a success. Proceed to Stage B (industry factors) as a separate plan.
- If gap 10-15%: Partial success. Try increasing model capacity or training longer before Stage B.
- If gap > 15%: Stage A failed. Investigate why — is the mkt_proj too weak? Is bootstrap too noisy? Debug before Stage B.
