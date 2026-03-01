Here are the **two explicit objective-function blocks** (Board-advice vs ASA-advice), written in the **sequential ARA style**:

- **Focal actor** optimises.
- **Other strategic players** are treated as *chance nodes* via **predictive distributions** induced by uncertainty over their utilities/parameters and their own EU-maximisation.
- **Vote** and **review findings** remain Nature (chance), but endogenous through the Bayesian submodels.

I'll keep notation tight and aligned to your diagram:
$D_1, D_{\text{rev}}, D_4$ (Board), $A_2$ (ASA), $M_{\text{agm}}, M_{\text{rev}}$ (CEO), $V\equiv V_{2023}$, $R$ (review findings), and feasibility "state flags" $S$.

---

# Common structure (used by both blocks)

## Histories / information sets

Let $h_t$ be the history observed at stage $t$. Under your tree this is essentially the sequence of realised earlier moves and chance outcomes.

Examples:

- Before ASA acts: $h_{A}=(D_1, B_0^{mkt}, B_0^{mgmt}, \text{known state flags})$.
- Before CEO (AGM) acts: $h_{M_{agm}}=(D_1,A_2,V,\ldots)$.
- Before Board final acts: $h_{D4}=(D_1,A_2,V,D_{\text{rev}},R,M_{\text{rev}},S,\ldots)$.

## Chance models (Bayesian)

These are shared:

$$
p(V \mid D_1,A_2,B_0^{mkt},B_0^{mgmt},S,\Theta_V),
\qquad
p(R \mid D_{\text{rev}},D_1,S,\Theta_R).
$$

(And in your implementation, $B_0^{mkt},B_0^{mgmt}$ are Monte Carlo draws from checkpoint `.npz`.)

## Utilities

Each actor $j\in\{B,A,C\}$ has utility $u_j(Z;\Theta_j)$ over terminal outcome summary $Z$ (CEO status, vote, review outcome, implementation costs, etc.).

## ARA predictive distribution template

From focal actor $i$'s perspective, any opponent $j$'s move at node $X$ is a chance variable with predictive distribution

$$
p_i(X=x \mid h)
=
\Pr_{\Theta_j\sim p_i(\Theta_j)}\left[
x \in \arg\max_{x'\in \mathcal{X}(h)}
\Psi_j(x';h,\Theta_j)
\right]
$$

where opponent $j$'s expected utility for candidate action $x$ is

$$
\Psi_j(x;h,\Theta_j)
=
\mathbb{E}_{i}\left[u_j(Z;\Theta_j)\mid h, X=x\right],
$$

and the expectation is taken under the focal actor's uncertainty about downstream chance nodes and (if needed) simplified response beliefs for future players (Level-1 ARA).

That's the "replace opponent decision node with chance node" step.

---

# Block 1 — Advising the Board (Board is focal optimiser)

## Board decisions and objective

Board chooses its policy triple

$$
\pi_B = (d_1, d_{\text{rev}}(\cdot), d_4(\cdot)),
$$

subject to feasibility flags $S$ (e.g., review can only be commissioned once, CEO may already be gone, etc.).

The Board's decision problem is:

$$
\pi_B^*
\in
\arg\max_{\pi_B \in \Pi_B}
\; EU_B(\pi_B),
$$

with

$$
EU_B(\pi_B)
=
\mathbb{E}\left[
u_B(Z;\Theta_B)
\;\middle|\;
\pi_B
\right].
$$

### Expanded expectation (what's inside $EU_B$)

Under Board advice, the random variables are:

- ASA move $A_2$ (strategic opponent → predictive distribution $p_B(A_2\mid \cdot)$)
- CEO moves $M_{\text{agm}}, M_{\text{rev}}$ (strategic opponent → predictive distributions)
- vote $V$ and review findings $R$ (Nature, Bayesian)
- priors/latent draws $B_0^{mkt},B_0^{mgmt}$ and uncertain parameters $\Theta$.

So:

$$
\begin{aligned}
EU_B(\pi_B)
&=
\mathbb{E}_{B_0,\Theta}\Bigg[
\sum_{a_2\in\mathcal{A}_2}
p_B(a_2 \mid h_A)
\int p(V\mid d_1,a_2,B_0,S,\Theta_V) \\
&\qquad \times
\sum_{m_{agm}\in\mathcal{M}_{agm}}
p_B(m_{agm}\mid h_{M_{agm}})
\;\;
\sum_{d_{rev}\in\mathcal{D}_{rev}(h)}
\mathbb{I}[d_{rev}=d_{\text{rev}}(h_{D_{rev}})] \\
&\qquad \times
\int p(R\mid d_{rev},d_1,S,\Theta_R)\;
\sum_{m_{rev}\in\mathcal{M}_{rev}}
p_B(m_{rev}\mid h_{M_{rev}}) \\
&\qquad \times
\sum_{d_4\in\mathcal{D}_4(h)}
\mathbb{I}[d_4=d_4(h_{D4})]
\;\;
u_B(Z;\Theta_B)
\;\; dR \, dV
\Bigg].
\end{aligned}
$$

Interpretation:

- The indicators $\mathbb{I}[\cdot]$ encode that **Board's policy** selects $d_{\text{rev}}$ and $d_4$ deterministically given histories (and feasibility).
- ASA/CEO choices are integrated as chance variables using predictive distributions induced by their own EU-maximisation under parameter uncertainty.

## Predictive distributions needed in Board mode

### ASA predictive distribution (from Board's view)

$$
p_B(A_2=a \mid h_A)
=
\Pr_{\Theta_A\sim p_B(\Theta_A)}
\left[
a\in\arg\max_{a'} \Psi_A(a';h_A,\Theta_A)
\right]
$$

with

$$
\Psi_A(a;h_A,\Theta_A)=
\mathbb{E}_B[u_A(Z;\Theta_A)\mid h_A, A_2=a].
$$

### CEO predictive distributions (from Board's view)

Similarly at AGM:

$$
p_B(M_{\text{agm}}=m \mid h_{M_{agm}})
=
\Pr_{\Theta_C\sim p_B(\Theta_C)}
\left[
m\in\arg\max_{m'} \Psi_C^{agm}(m';h_{M_{agm}},\Theta_C)
\right]
$$

and post-review:

$$
p_B(M_{\text{rev}}=m \mid h_{M_{rev}})
=
\Pr_{\Theta_C\sim p_B(\Theta_C)}
\left[
m\in\arg\max_{m'} \Psi_C^{rev}(m';h_{M_{rev}},\Theta_C)
\right].
$$

That fully specifies the Board-advice ARA objective: **Board is not an adversary**; it's the optimiser.

---

# Block 2 — Advising ASA (ASA is focal optimiser)

Now swap the focal actor.

ASA chooses a mobilisation policy

$$
\pi_A = a_2(\cdot)
$$

(or just a single $a_2$ if ASA acts only once in your tree).

ASA's decision problem:

$$
\pi_A^* \in \arg\max_{\pi_A\in \Pi_A}\; EU_A(\pi_A),
\qquad
EU_A(\pi_A)=\mathbb{E}\left[u_A(Z;\Theta_A)\mid \pi_A\right].
$$

### Expanded expectation (ASA mode)

Random variables now include:

- Board actions $D_1, D_{\text{rev}}, D_4$ as *opponent* moves **unless** you choose "Board policy model" mode.
- CEO moves $M_{\text{agm}}, M_{\text{rev}}$ as opponent moves.
- Vote $V$ and review $R$ as chance nodes.
- Priors/latent draws and parameters.

So the "full ARA Board opponent" version is:

$$
\begin{aligned}
EU_A(\pi_A)
&=
\mathbb{E}_{B_0,\Theta}\Bigg[
\sum_{d_1\in\mathcal{D}_1}
p_A(d_1\mid h_{D1})
\sum_{a_2\in\mathcal{A}_2}
\mathbb{I}[a_2=\pi_A(h_A)] \\
&\qquad\times
\int p(V\mid d_1,a_2,B_0,S,\Theta_V)
\sum_{m_{agm}\in\mathcal{M}_{agm}} p_A(m_{agm}\mid h_{M_{agm}}) \\
&\qquad\times
\sum_{d_{rev}\in\mathcal{D}_{rev}} p_A(d_{rev}\mid h_{D_{rev}})
\int p(R\mid d_{rev},d_1,S,\Theta_R) \\
&\qquad\times
\sum_{m_{rev}\in\mathcal{M}_{rev}} p_A(m_{rev}\mid h_{M_{rev}})
\sum_{d_4\in\mathcal{D}_4} p_A(d_4\mid h_{D4}) \\
&\qquad\times
u_A(Z;\Theta_A)\; dR\, dV
\Bigg].
\end{aligned}
$$

Key differences vs Board mode:

- Board's decisions appear as **predictive distributions** $p_A(\cdot)$ because Board is now an adversary (unless you switch it off).
- ASA's own action is fixed by policy $\pi_A$ via the indicator.

## Predictive distributions needed in ASA mode

### Board predictive distributions (from ASA's view) — if Board is full ARA opponent

At $D_1$:

$$
p_A(D_1=d\mid h_{D1})
=
\Pr_{\Theta_B\sim p_A(\Theta_B)}
\left[
d\in\arg\max_{d'} \Psi_B^{(A)}(d';h_{D1},\Theta_B)
\right]
$$

where $\Psi_B^{(A)}$ is *ASA's model* of Board's expected utility (Board optimises its own $u_B$, but ASA is uncertain about Board's parameters/beliefs).

Similarly for $D_{\text{rev}}$ and $D_4$ given their histories.

### CEO predictive distributions (from ASA's view)

Same form as before, but using ASA's beliefs about CEO parameters:

$$
p_A(M_{\text{agm}}=\cdot\mid h) \text{ induced by } \Theta_C\sim p_A(\Theta_C),
$$

and similarly for $M_{\text{rev}}$.

## Optional simplification: "Board is policy model" in ASA mode

If you choose to *not* model Board as full ARA, then replace

$$
p_A(D_1\mid h_{D1}),\;p_A(D_{\text{rev}}\mid h),\;p_A(D_4\mid h)
$$

with a calibrated policy model

$$
q_A(D_1\mid h_{D1}),\;q_A(D_{\text{rev}}\mid h),\;q_A(D_4\mid h)
$$

(e.g., logistic choice on vote pressure, deference, review findings, etc.). Algebra stays identical—just swap $p_A$ for $q_A$.

---

# Summary in one line each (what you can paste into the paper)

**Board-advice ARA objective:**

$$
\pi_B^* \in \arg\max_{\pi_B}\;
\mathbb{E}\Big[u_B(Z)\Big],
\quad
\text{where } A_2,M_{\text{agm}},M_{\text{rev}}\text{ are integrated using }p_B(\cdot)
\text{ induced by opponents' EU-max under uncertainty.}
$$

**ASA-advice ARA objective:**

$$
\pi_A^* \in \arg\max_{\pi_A}\;
\mathbb{E}\Big[u_A(Z)\Big],
\quad
\text{where } D_1,D_{\text{rev}},D_4,M_{\text{agm}},M_{\text{rev}}\text{ are integrated using }p_A(\cdot)
\text{ (or }q_A(\cdot)\text{ if Board is reduced-form).}
$$

---

Below is the same pair of problems (Board-advice vs ASA-advice), but written as **tree-node indexed Bellman-style recursions**, i.e. a value function per node in the extensive-form tree, with:

- **Decision nodes owned by the focal actor**: take a **max** over feasible actions.
- **Decision nodes owned by an opponent**: take an **expectation** w.r.t. the focal actor's **ARA predictive distribution** for that opponent's move.
- **Chance nodes** (vote $V$, review findings $R$, plus your checkpoint beliefs/parameters): take an **expectation / integral** under the Bayesian submodel.

I'll use the node ordering that matches your tree:

$$
D_1 \rightarrow A_2 \rightarrow V \rightarrow M_{\text{agm}} \rightarrow D_{\text{rev}} \rightarrow R \rightarrow M_{\text{rev}} \rightarrow D_4 \rightarrow \text{Terminal}.
$$

I'll denote:

- histories $h$ (everything observed up to the node),
- feasibility flags $S(h)$,
- terminal outcome summary $Z(h)$ once the game ends.

---

# Common node-indexed operators

## Feasible action sets

Each decision node has a feasible set depending on state flags:

$$
\mathcal{D}_1(h),\;\mathcal{A}_2(h),\;\mathcal{M}_{agm}(h),\;\mathcal{D}_{rev}(h),\;\mathcal{M}_{rev}(h),\;\mathcal{D}_4(h).
$$

## Chance kernels

- Vote kernel:

$$
p(V=v \mid h) \equiv p(v \mid D_1,A_2,B_0^{mkt},B_0^{mgmt},S,\Theta_V).
$$

- Review kernel:

$$
p(R=r \mid h) \equiv p(r \mid D_{rev},D_1,S,\Theta_R),
$$

with degenerate mass at "NoReview" if $D_{rev}=\text{NoReview}$.

## ARA predictive distributions (node-local)

At an opponent decision node $X$ with history $h$, focal actor $i$ uses:

$$
p_i(X=x\mid h)
=
\Pr_{\Theta_{owner(X)}\sim p_i(\Theta_{owner(X)})}\left[
x \in \arg\max_{x'\in\mathcal{X}(h)} \Psi_{owner(X)}(x';h,\Theta_{owner(X)})
\right],
$$

where the inner $\Psi$ is that opponent's expected utility under the focal actor's model (Level-1 ARA).

I will write these predictive distributions as:

- $p_B(A_2\mid h)$, $p_B(M_{\text{agm}}\mid h)$, $p_B(M_{\text{rev}}\mid h)$ in Board-advice mode.
- $p_A(D_1\mid h)$, $p_A(D_{\text{rev}}\mid h)$, $p_A(D_4\mid h)$, $p_A(M_{\text{agm}}\mid h)$, $p_A(M_{\text{rev}}\mid h)$ in ASA-advice mode (or $q_A(\cdot)$ for reduced-form Board).

---

# 1) Board-advice: node-indexed value recursion

Define Board's value function at each node $n$ as:

$$
U_B^{(n)}(h) = \mathbb{E}_B\left[u_B(Z)\mid \text{at node }n,\text{ history }h\right].
$$

## Terminal node

At a terminal history $h_T$:

$$
U_B^{(T)}(h_T) = u_B(Z(h_T)).
$$

## Node $D_4$ (Board decision)

History $h_{D4}=(D_1,A_2,V,M_{\text{agm}},D_{\text{rev}},R,M_{\text{rev}},S,\ldots)$.

$$
U_B^{(D4)}(h_{D4})
=
\max_{d_4\in \mathcal{D}_4(h_{D4})}
U_B^{(T)}(h_{D4}\cup\{D_4=d_4\}).
$$

## Node $M_{\text{rev}}$ (CEO decision treated as chance from Board view)

History $h_{Mrev}=(D_1,A_2,V,M_{\text{agm}},D_{\text{rev}},R,S,\ldots)$.

$$
U_B^{(Mrev)}(h_{Mrev})
=
\sum_{m\in \mathcal{M}_{rev}(h_{Mrev})}
U_B^{(D4)}(h_{Mrev}\cup\{M_{\text{rev}}=m\})\;
p_B(M_{\text{rev}}=m\mid h_{Mrev}).
$$

## Node $R$ (review findings chance node)

History $h_R=(D_1,A_2,V,M_{\text{agm}},D_{\text{rev}},S,\ldots)$.

$$
U_B^{(R)}(h_R)
=
\int
U_B^{(Mrev)}(h_R\cup\{R=r\})\;
p(R=r\mid h_R)\;dr.
$$

(Replace the integral by a sum if $R$ is discrete/ordinal.)

## Node $D_{\text{rev}}$ (Board decision)

History $h_{Drev}=(D_1,A_2,V,M_{\text{agm}},S,\ldots)$.

$$
U_B^{(Drev)}(h_{Drev})
=
\max_{d\in\mathcal{D}_{rev}(h_{Drev})}
U_B^{(R)}(h_{Drev}\cup\{D_{\text{rev}}=d\}).
$$

## Node $M_{\text{agm}}$ (CEO decision treated as chance from Board view)

History $h_{Magm}=(D_1,A_2,V,S,\ldots)$.

$$
U_B^{(Magm)}(h_{Magm})
=
\sum_{m\in \mathcal{M}_{agm}(h_{Magm})}
U_B^{(Drev)}(h_{Magm}\cup\{M_{\text{agm}}=m\})\;
p_B(M_{\text{agm}}=m\mid h_{Magm}).
$$

## Node $V$ (vote chance node)

History $h_V=(D_1,A_2,B_0^{mkt},B_0^{mgmt},S,\ldots)$.

$$
U_B^{(V)}(h_V)
=
\int
U_B^{(Magm)}(h_V\cup\{V=v\})\;
p(V=v\mid h_V)\;dv.
$$

## Node $A_2$ (ASA decision treated as chance from Board view)

History $h_A=(D_1,B_0^{mkt},B_0^{mgmt},S,\ldots)$.

$$
U_B^{(A2)}(h_A)
=
\sum_{a\in\mathcal{A}_2(h_A)}
U_B^{(V)}(h_A\cup\{A_2=a\})\;
p_B(A_2=a\mid h_A).
$$

## Root node $D_1$ (Board decision)

History $h_{D1}=(B_0^{mkt},B_0^{mgmt},S_0,\ldots)$.

$$
U_B^{(D1)}(h_{D1})
=
\max_{d_1\in\mathcal{D}_1(h_{D1})}
U_B^{(A2)}(h_{D1}\cup\{D_1=d_1\}).
$$

## Board's optimal initial action

Given checkpoint draws/priors at the root:

$$
d_1^* \in \arg\max_{d_1\in\mathcal{D}_1}\;
\mathbb{E}_{B_0,\Theta}\left[U_B^{(A2)}(h_{D1}\cup\{D_1=d_1\})\right].
$$

That's the Board-advice tree recursion in explicit node form.

---

# 2) ASA-advice: node-indexed value recursion

Define ASA's value function at each node $n$:

$$
U_A^{(n)}(h) = \mathbb{E}_A\left[u_A(Z)\mid \text{at node }n,\text{ history }h\right].
$$

Everything is identical structurally, except:

- **ASA's own decision node $A_2$** is now a **max** node.
- **Board decision nodes** become **expectations** under $p_A(\cdot\mid h)$ (unless reduced-form $q_A$).
- CEO decision nodes remain expectations (CEO is still full ARA opponent).

## Terminal

$$
U_A^{(T)}(h_T)=u_A(Z(h_T)).
$$

## Node $D_4$ (Board decision as chance from ASA view)

History $h_{D4}=(D_1,A_2,V,M_{\text{agm}},D_{\text{rev}},R,M_{\text{rev}},S,\ldots)$.

$$
U_A^{(D4)}(h_{D4})
=
\sum_{d_4\in\mathcal{D}_4(h_{D4})}
U_A^{(T)}(h_{D4}\cup\{D_4=d_4\})\;
p_A(D_4=d_4\mid h_{D4}).
$$

## Node $M_{\text{rev}}$ (CEO decision as chance)

$$
U_A^{(Mrev)}(h_{Mrev})
=
\sum_{m\in\mathcal{M}_{rev}(h_{Mrev})}
U_A^{(D4)}(h_{Mrev}\cup\{M_{\text{rev}}=m\})\;
p_A(M_{\text{rev}}=m\mid h_{Mrev}).
$$

## Node $R$ (review findings chance)

$$
U_A^{(R)}(h_R)
=
\int
U_A^{(Mrev)}(h_R\cup\{R=r\})\;
p(R=r\mid h_R)\;dr.
$$

## Node $D_{\text{rev}}$ (Board decision as chance from ASA view)

$$
U_A^{(Drev)}(h_{Drev})
=
\sum_{d\in\mathcal{D}_{rev}(h_{Drev})}
U_A^{(R)}(h_{Drev}\cup\{D_{\text{rev}}=d\})\;
p_A(D_{\text{rev}}=d\mid h_{Drev}).
$$

## Node $M_{\text{agm}}$ (CEO decision as chance)

$$
U_A^{(Magm)}(h_{Magm})
=
\sum_{m\in\mathcal{M}_{agm}(h_{Magm})}
U_A^{(Drev)}(h_{Magm}\cup\{M_{\text{agm}}=m\})\;
p_A(M_{\text{agm}}=m\mid h_{Magm}).
$$

## Node $V$ (vote chance)

$$
U_A^{(V)}(h_V)
=
\int
U_A^{(Magm)}(h_V\cup\{V=v\})\;
p(V=v\mid h_V)\;dv.
$$

## Node $A_2$ (ASA decision — focal maximisation)

History $h_A=(D_1,B_0^{mkt},B_0^{mgmt},S,\ldots)$ — **note**: ASA observes $D_1$ before acting in your tree, so $D_1$ is inside $h_A$.

$$
U_A^{(A2)}(h_A)
=
\max_{a\in\mathcal{A}_2(h_A)}
U_A^{(V)}(h_A\cup\{A_2=a\}).
$$

## Root node $D_1$ (Board move before ASA acts — chance from ASA view)

History $h_{D1}=(B_0^{mkt},B_0^{mgmt},S_0,\ldots)$.

$$
U_A^{(D1)}(h_{D1})
=
\sum_{d_1\in\mathcal{D}_1(h_{D1})}
U_A^{(A2)}(h_{D1}\cup\{D_1=d_1\})\;
p_A(D_1=d_1\mid h_{D1}).
$$

## ASA optimal policy / recommendation

Given checkpoint draws/priors:

$$
a_2^*(d_1)
\in
\arg\max_{a_2\in\mathcal{A}_2}\;
\mathbb{E}_{B_0,\Theta}\left[
U_A^{(V)}(h_A\cup\{A_2=a_2\})
\right],
\quad h_A=(h_{D1},D_1=d_1).
$$

So ASA's recommended action is conditional on observed $D_1$ (because ASA moves after Board).

---

# Reduced-form Board option in ASA mode (one-line substitution)

If you decide Board is **not** full ARA opponent when advising ASA, replace:

$$
p_A(D_1\mid h),\;p_A(D_{\text{rev}}\mid h),\;p_A(D_4\mid h)
$$

with a calibrated policy model:

$$
q_A(D_1\mid h),\;q_A(D_{\text{rev}}\mid h),\;q_A(D_4\mid h).
$$

All recursion equations remain identical.

---

# What this buys you (practically)

- In Board mode: the solver is **max–sum–integral** with max at $D_1,D_{\text{rev}},D_4$.
- In ASA mode: it's **sum–max–sum–integral** with max at $A_2$, and Board nodes summed under the Board predictive distribution.

Excellent — this is the core of the ARA machinery.

Below I define the **predictive distributions node-by-node**, separately for:

- **Board-advice mode** → $p_B(\cdot \mid h)$
- **ASA-advice mode** → $p_A(\cdot \mid h)$

At each opponent decision node $X$, the predictive distribution is induced by:

1. Uncertainty over that opponent's parameters $\Theta_j$
2. That opponent solving its own EU-maximisation problem (Level-1 ARA)
3. Integration over downstream chance nodes and simplified beliefs about other players

Throughout, let:

- $h$ = history observed at that node
- $\mathcal{X}(h)$ = feasible action set
- $u_j(Z; \Theta_j)$ = utility of actor $j$
- $Z$ = terminal outcome
- $\Psi_j(x; h, \Theta_j)$ = opponent $j$'s expected utility for action $x$

---

# PART I — Predictive distributions in **Board-advice mode**

Board is focal optimiser.
Opponents: ASA and CEO.

So we must define:

- $p_B(A_2 \mid h)$
- $p_B(M_{\text{agm}} \mid h)$
- $p_B(M_{\text{rev}} \mid h)$

Board decisions are **not predictive** here — they are max nodes.

---

## 1. ASA node $A_2$

History:

$$
h_A = (D_1, B_0^{mkt}, B_0^{mgmt}, S)
$$

Feasible actions:

$$
\mathcal{A}_2(h_A)=\{\text{DoNothing}, \text{RecStrike}\}
$$

### Step 1 — ASA's expected utility for candidate action $a$

From Board's perspective:

$$
\Psi_A(a; h_A, \Theta_A)
=
\mathbb{E}_B\left[
u_A(Z; \Theta_A)
\mid h_A, A_2=a
\right]
$$

The expectation integrates over:

- Vote $V\sim p(V\mid D_1,a,\cdot)$
- CEO behaviour $M_{\text{agm}},M_{\text{rev}}$
  (Board may assume CEO uses a simple policy inside ASA's model — Level-1 ARA)
- Board review/final decisions
  (ASA's model of Board behaviour — can be simplified)

---

### Step 2 — Induced predictive distribution

Board is uncertain about ASA's parameters $\Theta_A \sim p_B(\Theta_A)$.

$$
p_B(A_2=a\mid h_A)
=
\Pr_{\Theta_A \sim p_B(\Theta_A)}
\Big[
a \in
\arg\max_{a' \in \mathcal{A}_2(h_A)}
\Psi_A(a'; h_A, \Theta_A)
\Big]
$$

Equivalent softmax form (optional):

$$
p_B(A_2=a\mid h_A)
=
\int
\frac{\exp\{\lambda_A \Psi_A(a;h_A,\Theta_A)\}}
{\sum_{a'}\exp\{\lambda_A \Psi_A(a';h_A,\Theta_A)\}}
\; p_B(\Theta_A)\; d\Theta_A
$$

---

## 2. CEO node $M_{\text{agm}}$

History:

$$
h_{Magm} = (D_1, A_2, V, S)
$$

Feasible actions:

$$
\mathcal{M}_{agm}(h)=\{\text{Stay},\text{Resign}\}
$$

### CEO expected utility (AGM stage)

$$
\Psi_C^{agm}(m; h_{Magm}, \Theta_C)
=
\mathbb{E}_B\left[
u_C(Z; \Theta_C)
\mid h_{Magm}, M_{\text{agm}}=m
\right]
$$

Expectation integrates over:

- Board review decision (Board policy)
- Review findings $R$
- Final board action $D_4$

### Predictive distribution

$$
p_B(M_{\text{agm}}=m \mid h_{Magm})
=
\Pr_{\Theta_C\sim p_B(\Theta_C)}
\Big[
m \in
\arg\max_{m'} \Psi_C^{agm}(m'; h_{Magm}, \Theta_C)
\Big]
$$

---

## 3. CEO node $M_{\text{rev}}$

History:

$$
h_{Mrev}=(D_1,A_2,V,M_{\text{agm}},D_{\text{rev}},R,S)
$$

### CEO expected utility (post-review stage)

$$
\Psi_C^{rev}(m; h_{Mrev}, \Theta_C)
=
\mathbb{E}_B\left[
u_C(Z; \Theta_C)
\mid h_{Mrev}, M_{\text{rev}}=m
\right]
$$

Expectation integrates over:

- Final board decision $D_4$

### Predictive distribution

$$
p_B(M_{\text{rev}}=m \mid h_{Mrev})
=
\Pr_{\Theta_C\sim p_B(\Theta_C)}
\Big[
m \in
\arg\max_{m'} \Psi_C^{rev}(m'; h_{Mrev}, \Theta_C)
\Big]
$$

---

# PART II — Predictive distributions in **ASA-advice mode**

ASA is focal optimiser.
Opponents: Board and CEO.

Now we must define:

- $p_A(D_1\mid h)$
- $p_A(D_{\text{rev}}\mid h)$
- $p_A(D_4\mid h)$
- $p_A(M_{\text{agm}}\mid h)$
- $p_A(M_{\text{rev}}\mid h)$

---

## 1. Board initial node $D_1$

History:

$$
h_{D1}=(B_0^{mkt}, B_0^{mgmt}, S_0)
$$

Feasible:

$$
\mathcal{D}_1(h_{D1})=\{D0,D1,D2\}
$$

### Board expected utility (from ASA's perspective)

$$
\Psi_B^{D1}(d; h_{D1}, \Theta_B)
=
\mathbb{E}_A\left[
u_B(Z; \Theta_B)
\mid h_{D1}, D_1=d
\right]
$$

Expectation integrates over:

- ASA future action (ASA assumes it will play optimally)
- Vote $V$
- CEO reactions
- Review & final board stage

### Predictive distribution

$$
p_A(D_1=d\mid h_{D1})
=
\Pr_{\Theta_B\sim p_A(\Theta_B)}
\Big[
d \in \arg\max_{d'} \Psi_B^{D1}(d'; h_{D1}, \Theta_B)
\Big]
$$

---

## 2. Board review node $D_{\text{rev}}$

History:

$$
h_{Drev}=(D_1,A_2,V,M_{\text{agm}},S)
$$

$$
\Psi_B^{rev}(d; h_{Drev}, \Theta_B)
=
\mathbb{E}_A\left[
u_B(Z; \Theta_B)
\mid h_{Drev}, D_{\text{rev}}=d
\right]
$$

$$
p_A(D_{\text{rev}}=d\mid h_{Drev})
=
\Pr_{\Theta_B\sim p_A(\Theta_B)}
\Big[
d \in \arg\max_{d'} \Psi_B^{rev}(d'; h_{Drev}, \Theta_B)
\Big]
$$

---

## 3. Board final node $D_4$

History:

$$
h_{D4}=(D_1,A_2,V,M_{\text{agm}},D_{\text{rev}},R,M_{\text{rev}},S)
$$

$$
\Psi_B^{D4}(d; h_{D4}, \Theta_B)
=
u_B(Z(h_{D4},D_4=d); \Theta_B)
$$

$$
p_A(D_4=d\mid h_{D4})
=
\Pr_{\Theta_B\sim p_A(\Theta_B)}
\Big[
d \in \arg\max_{d'} \Psi_B^{D4}(d'; h_{D4}, \Theta_B)
\Big]
$$

---

## 4. CEO nodes in ASA mode

Identical structure to Board mode, but using ASA's beliefs about CEO parameters:

### AGM stage

$$
p_A(M_{\text{agm}}=m\mid h)
=
\Pr_{\Theta_C\sim p_A(\Theta_C)}
\Big[
m \in \arg\max_{m'} \Psi_C^{agm}(m';h,\Theta_C)
\Big]
$$

### Post-review stage

$$
p_A(M_{\text{rev}}=m\mid h)
=
\Pr_{\Theta_C\sim p_A(\Theta_C)}
\Big[
m \in \arg\max_{m'} \Psi_C^{rev}(m';h,\Theta_C)
\Big]
$$

---

# Structural summary

For **any opponent node $X$** owned by player $j$, from focal $i$'s perspective:

$$
\boxed{
p_i(X=x\mid h)
=
\int
\mathbb{I}\left[
x \in \arg\max_{x'} \Psi_j(x';h,\Theta_j)
\right]
\; p_i(\Theta_j)\; d\Theta_j
}
$$

with

$$
\Psi_j(x;h,\Theta_j)
=
\mathbb{E}_i
\left[
u_j(Z;\Theta_j)
\mid h, X=x
\right].
$$

This is the exact ARA predictive step — opponent choice uncertainty induced by parameter uncertainty and EU-maximisation.

---
