---
title: "Interpolation Families Unified in the GIF $(a_t, b_t, \\gamma_t)$ Parameterization"
category: "techniques"
slug: "interpolation-families-parameterization"
tags: ["technique", "interpolation", "stochastic-interpolation", "flow-matching", "parameterization", "comparison"]
refs: ["arxiv:2311.11475", "arxiv:2312.05579", "arxiv:2402.01460", "arxiv:2404.00551", "arxiv:2405.05512"]
links: ["gaussian-interpolation-flows", "conditional-stochastic-interpolation", "conditional-follmer-flow-distribution-learning", "cnf-convergence-distribution-learning", "characteristic-learning-one-step-generation", "unified-conditional-generation-rate"]
created: "2026-04-17T15:30:00"
updated: "2026-04-17T15:30:00"
---

# Interpolation Families Unified in the GIF $(a_t, b_t, \gamma_t)$ Parameterization

> **Purpose**: collect the interpolation families from the
> Huang/Jiao literature (CNF, Föllmer, stochastic, rectified flow,
> characteristic) into a single parameterization — so that statements
> proved for one family can be transferred to another via parameter
> matching.
>
> **Motivation**: [[unified-conditional-generation-rate]] Sub-problem 1.
> To combine Huang's drift rate with Jiao's $W_2$ lift, we first need
> to know *whether the two papers' interpolants are even the same
> object under reparameterisation*. They largely are — but with an
> important technical caveat about $\gamma_t$.

## Master parameterization

[[gaussian-interpolation-flows]] introduces the cleanest master form
(Definition 18 there). We extend it slightly with a stochastic channel
$\gamma_t \eta$ to cover [[conditional-stochastic-interpolation]]:

$$\boxed{\;Y_t \;\stackrel{d}{=}\; a_t\, Y_0 \;+\; b_t\, Y_1 \;+\; \gamma_t\, \eta\;}$$

- $Y_0 \sim \mathcal N(0, I_d)$ (standard Gaussian source).
- $Y_1 \sim \nu$ (target) — possibly conditional on $X \sim \pi(x)$.
- $\eta \sim \mathcal N(0, I_d)$ (independent stochastic channel).
- Schedule constraints: $a_0 > 0$, $b_0 \ge 0$, $a_1 = 0$, $b_1 = 1$.
  $\dot a \le 0$, $\dot b \ge 0$.
- $\gamma_t \ge 0$ is optional; $\gamma_0 = \gamma_1 = 0$ enforced by the
  boundary-distribution matching.

**GIF's sub-framework** is the $\gamma_t \equiv 0$ slice. **CSI extends**
to $\gamma_t \not\equiv 0$.

The velocity / drift that drives the ODE/SDE is then:

$$\mathbf v^\star(y, t) \;=\; \mathbb E\big[\dot a_t\, Y_0 + \dot b_t\, Y_1 + \dot \gamma_t\, \eta \,\big|\, Y_t = y\big]$$

(and in the conditional case, also $\,|\, X = x$).

---

## Family parameter table

All of the following are special cases:

| Method | $a_t$ | $b_t$ | $\gamma_t$ | Type | Paper |
|---|---|---|---|---|---|
| **Jiao Föllmer conditional** | $\sqrt{1-t^2}$ | $t$ | $0$ | ODE | [[conditional-follmer-flow-distribution-learning]] |
| **Jiao Föllmer unconditional** | $\sqrt{1-t^2}$ | $t$ | $0$ | ODE | `arxiv:2311.03660` |
| **Huang CSI rectified** | $1-t$ | $t$ | $0$ | ODE | [[conditional-stochastic-interpolation]] Example |
| **Huang CSI stochastic linear** | $1-t$ | $t$ | $\log(t - t^2 + 1)$ | SDE | [[conditional-stochastic-interpolation]] Example |
| **Huang CSI general** | $\mathcal I(\cdot,\cdot,t)$ | — | $\gamma(t) \ge t^{1/5-\zeta}$ | SDE | [[conditional-stochastic-interpolation]] |
| **CNF Gao-Huang-Jiao-Zheng** | schedule-class | schedule-class | $0$ | ODE | [[cnf-convergence-distribution-learning]] |
| **GIF master** | any valid schedule | any valid schedule | $0$ | ODE | [[gaussian-interpolation-flows]] |
| **Rectified flow** (Liu et al 2023) | $1-t$ | $t$ | $0$ | ODE | (external) |
| **VE SDE** | — | $1$ (fixed) | $\sigma_t$ | SDE | (diffusion literature) |
| **VP SDE** | $\cos(\phi_t)$ | $\sin(\phi_t)$ | via $\sigma$ | SDE | (diffusion literature) |
| **Cosine (trigonometric)** | $\cos(\pi t/2)$ | $\sin(\pi t/2)$ | $0$ | ODE | [[gaussian-interpolation-flows]] Table 1 |
| **Characteristic learning** | $\sigma \cdot (1-t)$ | $t$ | $0$ | ODE (one-step target) | [[characteristic-learning-one-step-generation]] |

---

## What velocity expression each case gives

**ODE case** ($\gamma_t \equiv 0$):
$$\mathbf v^\star(y, t) = \dot a_t\, \mathbb E[Y_0 \mid Y_t = y] + \dot b_t\, \mathbb E[Y_1 \mid Y_t = y]$$

**SDE case** ($\gamma_t \not\equiv 0$), Huang's CSI form uses the drift
$\mathbf b^\star$ and score $\mathbf s^\star$ separately:
$$\mathbf b^\star(y, t) = \mathbb E[\partial_t \mathcal I + \dot \gamma_t \eta \mid Y_t = y],\qquad \mathbf s^\star(y, t) = -\tfrac{1}{\gamma_t^2}\mathbb E[\gamma_t \eta \mid Y_t = y]$$

The **reverse SDE** of the CSI form uses both $\mathbf b^\star$ and
$u(t) \cdot \mathbf s^\star$ with an **adaptive diffusion** $u(t)$ that
the paper prescribes (key design feature).

---

## Regularity of the velocity

Under **GIF Proposition 29** (semi-log-concavity), the velocity
gradient is sandwiched:
$$\nabla_y V(t, y) \succeq \frac{\beta a_t \dot a_t + b_t \dot b_t}{\beta a_t^2 + b_t^2}\, I_d$$
— a dimension-free Lipschitz bound whose tightness depends on the
$(\kappa, \beta)$ log-concavity constants of the target.

The **covariance representation** (Lemma 26 of
[[gaussian-interpolation-flows]]):
$$\nabla_y S(t, y) \;=\; \frac{b_t^2}{a_t^4}\, \mathrm{Cov}(Y \mid Y_t = y) - \frac{1}{a_t^2}\, I_d.$$

Two bounds for the conditional covariance:
- **Upper bound** (for Lipschitz): Brascamp-Lieb under $\kappa$-semi-log-
  concavity.
- **Lower bound** (for lower velocity-gradient bound): Cramér-Rao under
  $\beta$-semi-log-convexity.

**All of these invoke Hessian-of-log-density conditions.** That's
exactly what [[unified-conditional-generation-rate]] wants to avoid.

---

## The $\gamma = 0$ vs $\gamma \ne 0$ mismatch (critical for Sub-problem 3)

**The subtle observation that M0 missed**:

- **Jiao's Proposition 1** (the lift lemma from
  [[conditional-follmer-flow-distribution-learning]]) is stated for
  the **ODE case** $\gamma_t \equiv 0$.
- **Huang's drift rate** (Theorem 5.6 from
  [[conditional-stochastic-interpolation]]) is *proven* under
  **Assumption 4**, which requires $\gamma(t) \ge t^{1/5 - \zeta}$ near
  the boundaries — i.e., strictly **$\gamma_t > 0$ on an interval
  around the endpoints**.

So the two tools target **disjoint subsets of the parameterisation**:

| Tool | $\gamma_t$ regime |
|---|---|
| Jiao Prop 1 lift | $\gamma_t \equiv 0$ |
| Huang drift rate | $\gamma_t \ge t^{1/5-\zeta}$ near $t \in \{0, 1\}$ |
| **Intersection** | **empty** |

**Three ways to close the gap** (each is a distinct research sub-task):

### Option A — Lift Jiao's Prop 1 to $\gamma_t \ne 0$ (coupling in SDE setting)

Jiao's coupling argument uses **Lipschitz pushforward through the ODE
flow map**. Extending to an SDE means coupling two *stochastic* flows
with different velocity/score networks. Standard tool: **synchronous
coupling of the two SDEs driven by the same Brownian motion**. Gives:
$$W_2^2(\pi_T^y, \hat \pi_T^y) \le e^{2\sqrt d \cdot \gamma_1^{\text{drift}}} \cdot \varepsilon_{\text{drift}}^2 + \text{score-error contribution}$$

This route:
- ✅ Keeps Huang's setup intact (no retrograding on his rate).
- ✅ Standard SDE-coupling toolkit; textbook-level.
- ❌ Introduces a separate **score-error contribution** that also needs
  to be bounded (Huang's Theorem 5.3 does bound score error — we can
  reuse it).
- ❌ Constants may be worse.

### Option B — Reprove Huang's drift rate at $\gamma \equiv 0$

Does Theorem 5.6's drift bound survive $\gamma \equiv 0$? Reading the
proof's use of Assumption 4:

- Assumption 4 is used in Theorem 5.3, which bounds the **score**
  error. The boundary decay of $\gamma$ is needed because the
  score-estimation loss has a $t^{-(1-5\zeta)}$ factor that blows up
  without it.
- For **drift-only**, Assumption 4 is reported *not* to be invoked in
  Theorem 5.6 directly.

**Tentative read**: if we only need the drift bound and discard the
score bound, Assumption 4 can be dropped. Then $\gamma \equiv 0$ is
admissible, and Jiao Prop 1 directly applies. **But** we lose the
score bound, and Huang's full end-to-end argument needs both.

The question: **does Jiao's lift lemma need score error (it doesn't for
the pure-ODE case) or only drift error?** Answer from the M0 read: it
needs only drift error (velocity = drift when $\gamma = 0$). ✅

So Option B is cleanest if the drift bound survives at $\gamma = 0$.
**Verifying that is the next concrete task** — requires reading the
full proof of Theorem 5.3/5.6 line-by-line in Huang's supplement.

### Option C — Find a shared sub-family

Both families at $(a_t, b_t) = (1-t, t)$ and $\gamma_t = \log(t - t^2 + 1)$:
Huang's stochastic linear. Does Jiao's Prop 1 extend to this? Only via
Option A. So Option C reduces to Option A anyway. Not a genuinely new
path.

### Decision for M1 → Sub-problem 3

Pursue **Option B first** (cheaper; 1-2 days of proof reading). If
Assumption 4 is truly only for the score bound and drops cleanly at
$\gamma = 0$, we're done. If not, fall back to Option A (1 week of
SDE-coupling work).

---

## Implications for other wiki entries

- [[conditional-follmer-flow-distribution-learning]]: needs a footnote
  that Prop 1 as currently stated is ODE-only; extension to SDE is
  Option A above.
- [[conditional-stochastic-interpolation]]: needs a footnote that
  Assumption 4's role in the drift bound vs score bound should be
  distinguished — key open checkpoint.
- [[unified-conditional-generation-rate]]: Progress Log to be updated
  with this subtlety (done in the same edit batch).
- [[cnf-rate-without-time-singularity]]: the rectified-flow case
  ($a_t = 1-t, b_t = t, \gamma_t = 0$) **is exactly** the
  time-singularity-free case that problem asks about. If Option B
  succeeds, both open problems share the same resolution.

---

## Connections

- **master framework**: [[gaussian-interpolation-flows]] — Definition 18
- **specialisations**: [[conditional-follmer-flow-distribution-learning]],
  [[conditional-stochastic-interpolation]],
  [[cnf-convergence-distribution-learning]],
  [[characteristic-learning-one-step-generation]]
- **downstream user**: [[unified-conditional-generation-rate]] needs
  this table to do the assumption-alignment step
- **related open problem**: [[cnf-rate-without-time-singularity]] —
  rectified-flow case sits in this same parameterisation

## Open questions surfacing from this note

1. **Does Huang's drift bound hold at $\gamma \equiv 0$?** (Option B
   above; primary next step.)
2. Do the various $(a_t, b_t)$ schedules yield genuinely different
   rates under matched assumptions? The GIF unification suggests *no*,
   but no one has written the theorem.
3. The **adaptive $u(t)$ diffusion** of CSI's reverse SDE: can its
   optimal form be derived from the $(a_t, b_t, \gamma_t)$ schedule
   alone, or does it need the target's conditional score?
