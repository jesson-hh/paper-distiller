---
title: "Unified Rate for Conditional Generation via Interpolation"
category: "directions"
slug: "unified-conditional-generation-rate"
tags: ["theory", "conditional-generation", "stochastic-interpolation", "wasserstein-rate", "end-to-end-analysis", "paper-worthy"]
refs: ["arxiv:2312.05579", "arxiv:2402.01460", "arxiv:2404.00551", "arxiv:2311.11475"]
links: ["conditional-stochastic-interpolation", "conditional-follmer-flow-distribution-learning", "cnf-convergence-distribution-learning", "gaussian-interpolation-flows", "cnf-rate-without-time-singularity", "wasserstein-generative-regression", "wasserstein-generative-regression-framework", "interpolation-families-parameterization", "huang-jian", "jiao-yuling", "synthetic-augmentation-financial-timeseries", "fts-interdiff-fusion", "girsanov-path-kl-bound", "diversity-collapse-sde-framework"]
created: "2026-04-17T14:45:00"
updated: "2026-04-17T14:45:00"
---

# Unified Rate for Conditional Generation via Interpolation

> **Status**: theoretical research programme (seeded 2026-04-17)
> **Owner**: me
> **Parent**: [[synthetic-augmentation-financial-timeseries]]
> **Sibling(s)**: [[cnf-rate-without-time-singularity]],
> [[minimax-filter-financial-stylised-facts]]

## 一句话

Huang 和 Jiao 在 **2023-12 / 2024-02** 各自独立做了"条件随机插值"生成器（互不
引用），用**完全不同的分析路径**给出**不可比的 rate**。把两套技术器拼起来 —— 
用 Huang 的 drift 假设 + Jiao 的 W₂ 机器 —— 应该能得到**严格更紧**的
end-to-end 条件分布收敛率，同时只需要比两者都弱的假设。这是一块**paper-worthy
的空白**。

## 动机：两篇并行、不可比、各留半个缺口

| 维度 | Huang 2312.05579 [[conditional-stochastic-interpolation]] | Jiao 2402.01460 [[conditional-follmer-flow-distribution-learning]] |
|---|---|---|
| 提交日 | 2023-12-09 | 2024-02-02 |
| 插值家族 | **flexible** $Y_t = \mathcal I(Y_0, Y_1, t) + \gamma(t)\eta$ | **specific** $X_t = t X_1 + \sqrt{1 - t^2} X_0$ |
| 分析目标 | drift $L^2$ error $\mathbb E\|\hat{\mathbf b}_n - \mathbf b^\star\|^2$ | 分布 $\mathbb E_y[W_2^2(\hat\pi, \pi)]$ |
| rate | $n^{-2\beta/(2\beta + k + d + 1)} \log^5 n$ | $n^{-4/(9(d + d_Y + 5))}$ |
| 核心假设 | Hölder $\beta$ drift + $\nabla p$ $L^1$-integrable | bounded Hessian $-\alpha I \preceq \nabla^2 U \preceq \alpha I$ |
| 证明机器 | 经验过程 + 显式逼近常数 | Girsanov path-KL + coupling |
| 分布级 bound | ❌ **没有** | ✅ 有 |
| 对重尾的兼容性 | ✅ 比较友好（只要 drift Hölder）| ❌ bounded Hessian 即排除 |
| 互相引用 | **❌ 没有** | **❌ 没有** |

两边各有一个缺口：
- **Huang 有弱假设但没分布级结论** —— 只 bound 住 drift，没把它抬到 $W_2$。
- **Jiao 有分布级结论但假设强** —— $\alpha$-bounded Hessian 即排除金融重尾。

**闭环观察**：把 Huang 的 drift rate 用 Jiao 的 Girsanov+coupling 机器**抬到
$W_2^2$**，在 Huang 的（较弱）假设下即可得到分布级 bound。这是两篇独立工作留下
的"加法空白"。

## 预估：Huang + lift 的 rate 应该 **严格好于** Jiao

Drift $L^2 \to W_2^2$ 的 Girsanov 抬升通常保持指数（至多损掉常数）。用 Huang
的 Thm 5.6 + 抬升:

$$\mathbb E_y\big[W_2^2(\hat \pi, \pi)\big] \lesssim \underbrace{n^{-2\beta/(2\beta + k + d + 1)} \log^5 n}_{\text{Huang drift → Jiao lift}} + \text{(discretisation)}$$

**数值对比**（$d + d_Y = 30$，smooth drift $\beta = 2$）:

| Rate | 指数 | 解释 |
|---|---|---|
| Jiao 2402.01460 | $-4/(9 \cdot 35) \approx -0.0127$ | bounded Hessian 假设 |
| **Huang drift + lift** | $-4/(4 + 35) \approx -0.103$ | Hölder $\beta = 2$ 假设 |
| **Ratio** | **~8× 更好** | 样本等价量 $n \to n^{0.12}$ |

即便 $\beta = 1$（Lipschitz drift，金融可信），也比 Jiao 好 **~4×**。数值说明
这**不是小改进** —— 如果能把抬升做对，方法学上的增益非常显著。

## 核心研究问题

> 给定条件随机插值生成器 $g_\theta$ 训练至 drift $L^2$ error 为
> $\varepsilon_{\text{drift}}^2$，在 *Huang 的假设集*（Hölder drift + 
> integrable 梯度）下，能否推出
>
> $$\mathbb E_y[W_2^2(\hat \pi, \pi)] \le C \cdot \varepsilon_{\text{drift}}^2 + o(\varepsilon_{\text{drift}}^2)$$
>
> 其中 $C$ 只依赖于时间窗口和 drift 的 Lipschitz 模，**不依赖 Hessian 有界
> 性**？

## 分解为 4 个可攻子问题

### Sub-problem 1. 插值族的共同参数化

Huang 的 $Y_t = \mathcal I(Y_0, Y_1, t) + \gamma(t)\eta$ 和 Jiao 的
$X_t = t X_1 + \sqrt{1 - t^2} X_0$ 都是[[gaussian-interpolation-flows]] 统一
框架的特例。写出两者在 $(a_t, b_t, \gamma_t)$ 参数化下的精确位置:

| 方法 | $a_t$ | $b_t$ | $\gamma_t$ |
|---|---|---|---|
| Jiao Föllmer 条件版 | $\sqrt{1 - t^2}$ | $t$ | $0$ |
| Huang rectified 条件版 | $1 - t$ | $t$ | $0$ |
| Huang stochastic linear | $1 - t$ | $t$ | $\log(t - t^2 + 1)$ |
| 通用 Albergo-VdE | adaptive | adaptive | adaptive |

**行动**: 写成 [[gaussian-interpolation-flows]] 的 Definition 18 schedule 
的子情形清单。这是 `techniques/` 级别的一个小条目，后续两个 sub-problem 的
baseline。

### Sub-problem 2. 假设集的正式对齐

Huang Assumption 1-4 vs. Jiao Assumption 1-4 的逐项对比。

预期的观察：
- Jiao 的 Hessian 有界 ⇒ drift Lipschitz（Huang 的特殊情形）
- Huang 的 Hölder drift $\beta = 1$ + integrable 梯度 不蕴含 Hessian 有界
- 因此 Jiao 的假设是 Huang 的**严格子集**

如果这成立，Sub-problem 3 的抬升就在 Huang 的更大假设族上运行，**严格覆盖 
Jiao 的情形**。

### Sub-problem 3. Drift-to-W2 Lift Lemma（核心 —— **M0 后已大幅简化**）

已知：drift estimate $\hat{\mathbf b}$ 满足
$\mathbb E \|\hat{\mathbf b} - \mathbf b^\star\|^2_{L^2} \le \varepsilon^2$。

目标:
$$\mathbb E_y\big[W_2^2(\hat \pi^y, \pi^y)\big] \le C(T, L) \cdot \varepsilon^2.$$

**✅ 2026-04-17 M0 读后关键简化**: 不需要新造引理。**Jiao 2402.01460 
Proposition 1 本身就是这个 lift**:

$$W_2^2(\pi_T^y,\hat \pi_T^y) \le e^{2\sqrt d\,\gamma_1} \cdot \varepsilon^2$$

其中 $\gamma_1$ 是速度场 $\mathbf v_F(\cdot, y, t)$ 在 $\mathbf x$ 上的 
Lipschitz 常数。

**之前的 🚨 Talagrand T2 陷阱预警是错误的** —— Jiao 的证明**不用 Girsanov 
也不用 LSI**，直接走 coupling via Lipschitz pushforward。这是 M0 的最大
发现。

### Sub-problem 3 简化后的真正问题

Jiao 目前通过 bounded Hessian (Assumption 3) 推出 $\gamma_1 = \zeta(\alpha, d)$
polynomial。我们要的是：

> **能否从 Huang 假设集里直接得到 $\gamma_1 < \infty$，绕过 Hessian bound？**

候选路径:

1. **Huang Assumption 2** 已经假定 drift $\mathbf b^\star$ 满足 "polynomial
   growth bounds" —— 这是 Fokker-Planck well-posedness 的标配。如果 
   polynomial-growth 足以推 velocity Lipschitz，问题就解决了。需要查一个
   标准 ODE / SDE 结果。
2. **直接假定** $\gamma_1 < \infty$ 为 sub-problem 3 的输入 —— 这是最弱的
   条件，直接接到 Jiao Prop 1。这样 Sub-problem 2 的工作就是验证 Huang 的 
   drift rate 能在这个假设下成立。

**警告**: $C = e^{2\sqrt d\,\gamma_1}$ 在 ambient $d$ 里**指数爆炸**。rate 
指数仍然是 clean 的 $n^{-2\beta/(2\beta + k + d + 1)}$，但绝对常数在高维无
实用性。这是 *theory paper only*，实际工程价值有限（但作为 theoretical 
clarification 仍然 paper-worthy）。

### Sub-problem 4. End-to-end combination

把 Sub-problem 3 的 lift + Huang 的 drift rate 拼成完整结论:

$$\boxed{\;\mathbb E_y[W_2^2(\hat \pi, \pi)] \le C_1 \cdot n^{-2\beta/(2\beta + k + d + 1)} \log^5 n \;+\; C_2 \cdot h^3 \;+\; C_3 \cdot \bar t\;}$$

三项分别是: drift 估计（Huang）+ 离散化（标准 Grönwall）+ early-stopping 
（[[cnf-convergence-distribution-learning]] 技术移植）。

最优化 $\bar t$ 和 $h$ 应该和 [[cnf-convergence-distribution-learning]] 类似
结构，但系数不同。

## 预期难度（M0 后已更新）

| Sub-problem | 难度 | 时间预估 |
|---|---|---|
| 1. 插值族参数化 | 低 | 1 天 |
| 2. 假设集对齐 | 低-中 | 2-3 天 |
| 3. Drift → W₂ lift（**M0 后大幅简化** —— 用 Jiao Prop 1） | **低-中** | 3-5 天 |
| 4. End-to-end 合并 | 低 | 3-5 天 |

**M0 后瓶颈迁移**: 原本担心 Sub-problem 3 需要造新的 lift lemma（1-2 周的
开放性研究）；M0 发现 Jiao Prop 1 直接可用。新瓶颈变成 Sub-problem 2 + 一
个小引理 "Huang 假设 2 ⇒ $\gamma_1$ 有限"。**整体工期从 2-4 周压缩到 
~2 周**。

## 预期产出（最乐观路径）

- **Technique note**: `techniques/drift-to-w2-lift-lemma.md`（Sub-problem 3 
  的结论）
- **Article** 或 **Direction 扩写**: 整合 Sub-problem 1-4 成一个定理的完整
  陈述
- **可发表 paper**: 如果 lift + Huang-strict-subsumes-Jiao 都成立，**一篇
  theory paper** 的核心结果就有了

如果 Sub-problem 3 被卡住：至少 Sub-problem 1-2 的产出可以独立成文，作为
"clarifying the relationship between two parallel lines" 类型的短篇。

## 风险

| 风险 | 严重度 | 缓解 |
|---|---|---|
| Prop 54(ii) 在 Huang 假设族外失效 | 🔴 | 先做 Sub-problem 2，如果发现假设不相容，放弃这条抬升，改用 Schrödinger bridge coupling |
| Huang 的 drift rate 在分析过程中实际隐含 Jiao 的假设 | 🔴 | Sub-problem 2 的首要目标就是这个 check |
| 抬升的常数 $C(T, L)$ 爆炸（比如含 $e^{O(L^2 T)}$）| 🟡 | 常数爆炸在 *theory paper* 里也能接受，只是实用性低；仍然可发表 |
| 两篇论文互相"隐式引用"了对方某个 arxiv preprint 而我们没看到 | 🟡 | 拉两篇各自版本演化，看 v1 → v_latest 的 reference 变化 |
| lift lemma 早已有人做过（在我们的视野外）| 🟡 | Sub-problem 3 开始前做一次针对性 literature 扫 |

## 里程碑

```
M0. 读通 Huang 2312.05579 第 5 节（drift rate 证明）和 Jiao 2402.01460 第 3 节
    （Girsanov+coupling 证明）—— 对应 [[conditional-stochastic-interpolation]]
    和 [[conditional-follmer-flow-distribution-learning]] 两个 L2 note 里的
    "TODO — verify from full paper" 都填上
        ↓
M1. 完成 Sub-problem 1（插值族共同参数化）→ 短 techniques note
        ↓
M2. 完成 Sub-problem 2（假设集对齐）
    关键分支点: 如果 Jiao ⊊ Huang 不成立，这条 direction 失效；回到
    [[cnf-rate-without-time-singularity]] 那条
        ↓
M3. 完成 Sub-problem 3（lift lemma）
    关键分支点: 如果 Prop 54(ii) 不够用，切换到 Schrödinger bridge coupling
        ↓
M4. 完成 Sub-problem 4（end-to-end 合并）
        ↓
M5. 数值验证: 在高斯 mixture 目标 + 条件向量上训 CSI 生成器，测 drift $L^2$
    和 $W_2^2$，拟合 rate 指数与 M4 预测对比
        ↓
M6. Paper draft
```

每个 milestone 都有明确的 **decision point** → 任何一步 blocker 触发就停。

## 与其它条目的关系

### 深度依赖（必读）

- [[conditional-stochastic-interpolation]] —— Huang 的 drift rate 的源头
- [[conditional-follmer-flow-distribution-learning]] —— Jiao 的 W₂ rate 的
  源头
- [[gaussian-interpolation-flows]] —— 统一插值框架 + Prop 54(ii) 稳定性
- [[cnf-convergence-distribution-learning]] —— 同作者群的**无条件**版，是
  end-to-end rate 的模板

### 连带关系

- [[cnf-rate-without-time-singularity]] —— **sibling open problem**；如果
  我们 pivot 出去的 Huang + lift 路径成功，而且它的插值族不含时间奇点（如
  rectified flow），那么可以顺便解决那个 $n^{-1/(d+3)}$ 恢复问题
- [[wasserstein-generative-regression-framework]] —— 对应**应用**端的技术；
  本 direction 是**理论**端的配套
- [[synthetic-augmentation-financial-timeseries]] —— 应用父方向；**这个
  direction 的理论成果直接替换该方向的引用 rate**
- [[fts-interdiff-fusion]] —— 最近的实际应用；理论 rate 直接决定实验预期
- [[girsanov-path-kl-bound]], [[recursive-chi2-inequality]], 
  [[relative-fisher-poincare]] —— 可能在 Sub-problem 3 里用到的 techniques
  备选

### 和两位作者的个人 programme 的关系

- 对 [[huang-jian]]: 把他的 drift rate 抬到分布级，让他现有的 drift bound
  "活起来"
- 对 [[jiao-yuling]]: 放松他 W₂ rate 的假设集，覆盖他原本不包括的重尾情形
- **战略价值**: 这是一个典型的"两个 programme 合并"题目，研究结果天然地对
  齐两条线，**单独某一个 PI 都不会写这篇论文** —— 我们的外部视角就是竞争
  优势。

## 下一步（现在就能做）

1. **M0**: 拉 Huang 和 Jiao 两篇 PDF 的 §5（Huang）/ §3-4（Jiao），把两个
   L2 note 里的 "TODO — verify from full paper" 标签都逐项填上
2. **M1 skeleton**: 用 [[gaussian-interpolation-flows]] 的 Definition 18 
   schedule，把 Huang/Jiao 两篇的插值族写成参数表
3. 如果 M0-M1 结果显示假设不兼容（比如 Jiao 的技术实际上需要 Huang 禁用的
   条件）→ direction 立即停，转向 [[cnf-rate-without-time-singularity]]

## Progress Log

### 2026-04-17 — seeded

- 从 [[conditional-stochastic-interpolation]] 和 
  [[conditional-follmer-flow-distribution-learning]] 两个 L2 note 的对比
  中识别出这条路径
- 数值 back-of-envelope: Huang + lift 在 $d + d_Y = 30$ 下比 Jiao **~4-8×**
  指数上更好
- 未开工: 以上所有 Sub-problem 仍为待证

### 2026-04-17 — M0 ✅ 精读两篇证明

拉了 Huang 2312.05579 §5 和 Jiao 2402.01460 §3-4 + Appendix 的完整证明结构。

**核查结论**（逐项更新到两个 L2 note):

#### Huang 2312.05579 证明（Theorem 5.6 的 drift $L^2$ bound）

- **完全不用 Hessian bound** ✓
- **完全不用 log-concavity / LSI** ✓
- **不用 Girsanov / 换测度** ✓
- 纯经验过程 ERM 证明（Lemma 5.2 excess-risk decomposition + Thm 5.3 
  stochastic error via Rademacher + Thm 5.5 approximation）
- Assumption 4（$\gamma(t)$ polynomial decay）是真的被用的 —— 控制 Thm 5.3 
  的 $t^{-(1-5\zeta)}$ boundary factor，不是可选的

**Huang 的 drift rate 完全在 Hölder + Sobolev-integrability 假设族下运行。**

#### Jiao 2402.01460 证明（Theorem 2 的 $W_2^2$ bound）

**最大的惊喜**: **不用 Girsanov，不用 Talagrand T2，不用 LSI**。用的是
**Coupling via Lipschitz pushforward**（Proposition 1, Appendix D.1）。

> 我原来在 seeded 版里 🚨 过 "Talagrand T2 要 LSI" 的陷阱 —— **那是错的担心**。
> 人家根本没走那条路。

Assumption 3（bounded Hessian $-\alpha I \preceq \nabla^2 U \preceq \alpha I$）
**只在 3 处被用**：

1. Lemma 1: 推出 velocity $\mathbf v_F$ 的 Lipschitz 常数 $\gamma_1 = \zeta(\alpha, d)$
2. Appendix E.1: 保证 ODE well-posedness
3. Proposition 1: 误差传播常数 $C = e^{2\sqrt d\,\gamma_1}$

不在以下地方被用（**可替换**）: 
- Theorem 4（generalisation error）
- Lemma 5（discretisation）
- Lemma 6（early stopping）

这三项只依赖抽象的 $\gamma_1, \gamma_2, \gamma_3$ 常数；只要我们能从**任何其他
路径**推出 velocity Lipschitz（即 $\gamma_1 < \infty$），整个 pipeline 就能
在没有 Hessian bound 的情况下跑通。

#### 战略更新

1. **Sub-problem 3 大幅简化**: Jiao 的 Proposition 1 就是 lift lemma，我们不
   需要自己造。问题变成"能否绕开 Hessian bound 得到 $\gamma_1 < \infty$"。
2. **Sub-problem 2 的验收标准**: 检查 Huang Assumption 2（drift Sobolev + 
   polynomial growth）是否蕴含 velocity Lipschitz。如果是，这条 direction 
   直接可完成。这是下一个动作（~半天标准 SDE/ODE regularity 结果）。
3. **常数代价**: 即使 Sub-problem 2 通过，最终常数 $e^{2\sqrt d\,\gamma_1}$ 
   在高维指数爆炸。rate 指数是干净的，论文能发，但下游实验预期需要小心（这
   影响 [[fts-interdiff-fusion]] 的 M9 计划的数值预期）。

#### 下一动作 (M1 / Sub-problem 1 + 2 并行)

- **M1a**（1 天）: 在 [[gaussian-interpolation-flows]] 的 $(a_t, b_t, \gamma_t)$ 
  参数化下写出两篇的插值族；已在本 direction 的主文里部分列出，需展开成正式 
  techniques note
- **M1b**（半天）: 查 "polynomial growth of drift ⇒ velocity Lipschitz" 的
  标准 SDE regularity 结果；如果成立，直接进入 Sub-problem 4 的 end-to-end 
  合并
- **M1c**（1 天）: 如果 M1b 失败，Huang Assumption 2 实际上不足以给 $\gamma_1$，
  需要 additional regularity。这是 direction 的 real blocker，若触发则暂停并
  评估是否转向 [[cnf-rate-without-time-singularity]]

#### 已填 TODO

两个 L2 note 的 "proof technique" 节已根据上面核查结果重写:
- [[conditional-follmer-flow-distribution-learning]] 第 5 节: 纠正 "Girsanov"
  → "coupling via Lipschitz pushforward"；加上 Prop 1 的 lift 机制 + Assumption 3
  的 3 处精确使用位置
- [[conditional-stochastic-interpolation]] 第 5 节: 明确"不用 Girsanov / 
  LSI / Hessian"；$W_2$ gap 指向本 direction

### 2026-04-17 — M1 ✅ （两项并行完成）

#### M1a ✅ — 插值族统一参数化

落盘: [[interpolation-families-parameterization]]（techniques note）

把 Huang/Jiao/GIF/rectified-flow/VE/VP/cosine 所有插值族都写成 GIF
$(a_t, b_t, \gamma_t)$ 主参数化的子情形。关键的参数表格已在 techniques note
里。对比的几个要点:

- GIF Definition 18 是 $\gamma_t \equiv 0$ 的 ODE subset
- Huang CSI 扩展到 $\gamma_t \ne 0$ 的 SDE；Assumption 4 要求 $\gamma$ 在端点
  polynomial decay
- Jiao Föllmer 条件版是 $(a_t, b_t, \gamma_t) = (\sqrt{1-t^2}, t, 0)$
- Huang rectified 是 $(1-t, t, 0)$；stochastic linear 是 $(1-t, t, \log(t-t^2+1))$

#### M1a 带来的关键观察（M0 漏掉的）

**Jiao Prop 1 和 Huang drift rate 的参数集不相交**:

| 工具 | $\gamma_t$ 允许范围 |
|---|---|
| Jiao Prop 1 lift | $\gamma_t \equiv 0$ |
| Huang drift rate (Thm 5.6) | $\gamma_t \ge t^{1/5-\zeta}$ 靠近端点 |
| **Intersection** | **空** |

这是一个必须处理的技术 gap。参见 [[interpolation-families-parameterization]] 
列的 3 条 options:
- **Option A**: 把 Jiao Prop 1 抬到 SDE 设定（synchronous coupling）—— 能做但
  要多干一周
- **Option B**: 核对 Huang drift bound 在 $\gamma \equiv 0$ 是否仍成立 —— 
  本质上只是"能否丢掉 Assumption 4"
- **Option C**: 找共享子族（实际退化到 Option A）

#### M1b ✅ — Option B 路径核实

读了 Huang Theorem 5.3 的完整陈述，结论**决定性地支持 Option B**:

1. **Drift bound 和 Score bound 是结构独立的两条 bound**:
   > "the ERMs $\hat{\mathbf b}_n$ 和 $\hat{\mathbf s}_n$ ... satisfy 
   > [drift bound] and [score bound]"（两个分开的数学表达式）

2. **关键的 $t^{-(1-5\zeta)}$ boundary factor 只出现在 score bound**，不在
   drift bound。Drift bound 是 $t$-均匀的:
   $$\mathbb E_S\big\{R_t^{\mathbf b}(\hat{\mathbf b}_n) - 2 R_{t,n}^{\mathbf b}(\hat{\mathbf b}_n) + R_t^{\mathbf b}(\mathbf b^\star)\big\} \le c_0\, d\, \mathcal B^5\, \mathcal{SD}\, \log(\mathcal S) / n$$

3. **$\gamma \equiv 0$ 时 drift loss 保持 well-defined**:
   $R_{t,n}^{\mathbf b}(\mathbf b) = \frac{1}{n}\sum_k \|\partial_t \mathcal I + \dot\gamma(t)\eta_k - \mathbf b(\ldots)\|^2$
   在 $\gamma \equiv 0$ 时退化为标准非参回归，没有任何端点奇点。
   （退化的是 score loss 里的 $\|\gamma(t)^{-1}\eta\|$，但这一项我们不需要。）

4. **Corollary 4 的 minimax rate $O(n^{-2\beta/(2\beta+k+d+1)} \log^5 n)$** 是
   drift bound 推出来的，不需要 score bound。

**结论**: Huang drift rate 在 $\gamma \equiv 0$ 下**应该**原样成立。institutional
coupling（"paper 没显式分解"）是唯一障碍 —— 技术上可以独立抽出来。

#### Sub-problem 2 同时推进: 假设集对齐

借 M1b 顺便核对 Jiao Assumption 3 vs Huang Definition 5.4:

- **Jiao Assumption 3** ($\alpha$-bounded Hessian $-\alpha I \preceq \nabla^2 U \preceq \alpha I$) 通过 Lemma 1 推出 velocity $\mathbf v_F$ 是 $\zeta(\alpha, d)$-Lipschitz in $\mathbf x$
- **Huang Definition 5.4** (Hölder-$\beta$ drift with $\beta \ge 1$) 直接给出 Lipschitz drift

在 $\gamma = 0$ 下 drift $\equiv$ velocity，所以：

$$\text{Jiao Assumption 3} \;\Longrightarrow\; \text{Huang Definition 5.4 at } \beta = 1$$

反向**不成立**（Hölder drift 不蕴含 bounded Hessian of density）。

**所以 Jiao 假设集 $\subsetneq$ Huang 假设集（在 $\beta = 1$ 下）**。Sub-problem
2 验收通过。

#### 拼出来的 rate（预期结论）

组合 Huang Cor 4 drift rate + Jiao Prop 1 lift:

$$\mathbb E_y\big[W_2^2(\hat \pi^y, \pi^y)\big] \;\le\; \underbrace{e^{2\sqrt d\,\gamma_1}}_{\text{Prop 1 lift}}\; \cdot \underbrace{O\!\big(n^{-2\beta/(2\beta + k + d + 1)} \log^5 n\big)}_{\text{Huang Cor 4 (drift)}} \;+\; \text{disc. + ES}$$

- 对于 $\beta = 1$（Lipschitz drift，Jiao 假设的严格超集）:
  rate $= n^{-2/(k + d + 3)}$
  vs Jiao's $n^{-4/(9(d + d_Y + 5))}$
  在 $d + d_Y = 30$ 下: **$n^{-0.061}$ vs $n^{-0.0127}$，指数比 ~4.8×**
- 对于 $\beta = 2$（smooth drift，Huang 自然设置）:
  rate $= n^{-4/(k + d + 5)}$
  在 $d + d_Y = 30$ 下: **$n^{-0.114}$ vs $n^{-0.0127}$，指数比 ~9×**

**常数代价**: $e^{2\sqrt d\,\gamma_1}$ 在高维指数爆炸 —— rate 指数 clean 但
绝对常数不实用。依然是 theory paper 级别的干净结果。

#### 下一步 M2 / Sub-problem 4 （end-to-end 合并）

- **M2a**（1 天）: 正式写出 Lemma "Huang drift bound at $\gamma \equiv 0$" 的
  陈述（声明 + 从 Theorem 5.3 drift 部分重抽出的证明骨架）
- **M2b**（2-3 天）: 写主定理（端到端 $W_2^2$ rate）的证明骨架。合并
  - 数据: M1b 的 drift bound
  - Lift: Jiao Prop 1（引用）
  - 离散化: 标准 Euler + Grönwall
  - Early stopping: Huang Lemma 6（原样引用）
- **M2c**（2 天）: 数值验证 —— 高斯混合目标 + 条件向量上训 CSI 生成器，拟合
  rate 指数，看是否匹配 $2\beta/(2\beta + k + d + 1)$
- **M2d**（1-2 天）: paper outline，确定投稿目标（arxiv + 一个 stat theory
  venue，JMLR 或 EJS）

总工期估计: **~1 周完成 M2**，两周内有 paper draft。

#### 风险重评（M1 后）

- 🟢 Lift lemma 存在：Jiao Prop 1 直接用 （M0 发现）
- 🟢 Huang drift bound 可延拓到 $\gamma \equiv 0$：结构上通（M1b 核实）
- 🟢 假设集对齐：Jiao ⊊ Huang at $\beta = 1$（M1b 副产品）
- 🟡 institutional coupling 是否是 legit 技术障碍：需要 Huang 提供明确
  version 的"drift-only rate under Hölder-$\beta$ drift, no Assumption 4"，或
  我们自己写出其证明的 $\gamma \equiv 0$ 版
- 🔴 **指数常数**: $e^{2\sqrt d\,\gamma_1}$ 在高维无实用价值；论文可发但
  benchmarks 受限
- 🟡 **数值验证**: M2c 的 empirical rate 匹配可能暴露 theory-experiment gap
  （典型问题）
