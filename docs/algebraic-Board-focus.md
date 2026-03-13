# Algebraic Notation — Board-Focal ARA

This document specifies the mathematical structure used when the **Board is the focal (maximising) actor**. It is a Board-specific subset of the full `algebraic.md`; ASA-advice mode recursion is omitted but all utility functions are retained (they are consumed by the predictive distribution engine when modelling opponents).

---

## 1. Game tree structure

### 1.1 Node order

The tree is **not** a fixed linear sequence. Node traversal is conditional on game state, principally on whether the CEO is still present. The base sequence is:

$$
D_0^{\text{ceo}} \rightarrow D_1 \rightarrow A_2 \rightarrow V \rightarrow M_{\text{agm}} \rightarrow D_4 \rightarrow D_{\text{rev}} \rightarrow R \rightarrow M_{\text{rev}} \rightarrow [\text{conditional } D_4 \rightarrow D_{\text{rev}}] \rightarrow \text{Terminal}
$$

**Key ordering principle:** $D_4$ (CEO response) precedes $D_{\text{rev}}$ (Board response). The CEO has initiative to resign before the Board decides whether to act.

**Post-review conditional round:** After $R$ and $M_{\text{rev}}$, if $\text{review\_adverse} = \text{true}$ AND $\text{CEO\_present} = \text{true}$, a second $D_4 \rightarrow D_{\text{rev}}$ round occurs. Otherwise, the game proceeds directly to Terminal.

| Node                 | Type     | Owner  | Board role       |
| -------------------- | -------- | ------ | ---------------- |
| $D_0^{\text{ceo}}$ | Decision | CEO    | Opponent (ARA)   |
| $D_1$              | Decision | Board  | **Focal (max)**  |
| $A_2$              | Decision | ASA    | Opponent (ARA)   |
| $V$                | Chance   | Nature | MC integration   |
| $M_{\text{agm}}$   | Chance   | Nature | Pass-through     |
| $D_4$              | Decision | CEO    | Opponent (ARA)   |
| $D_{\text{rev}}$   | Decision | Board  | **Focal (max)**  |
| $R$                | Chance   | Nature | MC integration   |
| $M_{\text{rev}}$   | Chance   | Nature | Pass-through     |
| $D_4'$             | Decision | CEO    | Opponent (ARA)   |
| $D_{\text{rev}}'$  | Decision | Board  | **Focal (max)**  |
| Terminal             | Terminal | —     | Compute $u_B$   |

### 1.2 Action sets

| Node                                    | Actions                                                           | Feasibility                              |
| --------------------------------------- | ----------------------------------------------------------------- | ---------------------------------------- |
| $D_0^{\text{ceo}}$                    | `CEO_resign`, `CEO_stay`                                      | always                                   |
| $D_1$                                 | `D0_minimal`, `D1_review`, `D3_ceo_transition`              | always (`D3` requires `CEO_present`) |
| $A_2$                                 | `A2_no_strike`, `A2_rec_strike`                               | always                                   |
| $D_4$, $D_4'$                       | `D4_stay`, `D4_resign`, `D4_negotiate_exit`                 | `CEO_present`                          |
| $D_{\text{rev}}$, $D_{\text{rev}}'$ | `Drev_no_action`, `Drev_commission_review`, `Drev_sack_ceo` | see below                                |

Feasibility rules evaluated dynamically on the game state $S$:

| Code                        | Condition                                                                                             |
| --------------------------- | ----------------------------------------------------------------------------------------------------- |
| `always`                  | $\text{true}$                                                                                       |
| `CEO_present`             | $S.\text{CEO\_present} = \text{true}$                                                               |
| `CEO_not_removed`         | $S.\text{CEO\_removed} = \text{false}$                                                              |
| `review_not_commissioned` | $S.\text{review\_commissioned} = \text{false}$                                                      |
| `review_commissioned`     | $S.\text{review\_commissioned} = \text{true}$                                                       |
| `review_completed`        | $S.\text{review\_completed} = \text{true}$                                                          |
| `not_reviewed_yet`        | $S.\text{review\_commissioned} = \text{false} \;\wedge\; S.\text{review\_completed} = \text{false}$ |
| `post_review_round`       | $S.\text{post\_review\_round} = \text{true}$                                                        |
| `post_review_round_and_CEO_present` | $S.\text{post\_review\_round} \;\wedge\; S.\text{CEO\_present}$                          |
| `post_review_round_and_review_not_commissioned` | $S.\text{post\_review\_round} \;\wedge\; S.\text{review\_commissioned} = \text{false}$ |

### 1.3 State transitions

The game state $S$ is an immutable record:

$$
S = (\text{CEO\_present},\; \text{review\_commissioned},\; \text{review\_completed},\; \text{CEO\_removed},\; \text{CEO\_resigned\_early},\; \text{review\_adverse},\; \text{post\_review\_round},\; \text{headline\_incident},\; \text{checkpoint\_id})
$$

Key transitions:

| Action                                    | Effect                                                                                                                                              |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CEO_resign`                            | $\text{CEO\_present} \leftarrow \text{false},\; \text{CEO\_removed} \leftarrow \text{true},\; \text{CEO\_resigned\_early} \leftarrow \text{true}$ |
| `D3_ceo_transition`, `Drev_sack_ceo`  | $\text{CEO\_present} \leftarrow \text{false},\; \text{CEO\_removed} \leftarrow \text{true}$                                                       |
| `D1_review`, `Drev_commission_review` | $\text{review\_commissioned} \leftarrow \text{true}$                                                                                              |
| `D4_resign`, `D4_negotiate_exit`      | $\text{CEO\_present} \leftarrow \text{false},\; \text{CEO\_removed} \leftarrow \text{true}$                                                       |
| `adverse` (at $R$)                    | $\text{review\_adverse} \leftarrow \text{true},\; \text{review\_completed} \leftarrow \text{true},\; \text{post\_review\_round} \leftarrow \text{CEO\_present}$ |
| `no_adverse` (at $R$)                 | $\text{review\_adverse} \leftarrow \text{false},\; \text{review\_completed} \leftarrow \text{true}$                                               |

### 1.4 Scenario branching at $D_0^{\text{ceo}}$

The solver evaluates both pre-game scenarios separately:

$$
\text{scenario} \in \{\text{ceo\_stayed},\; \text{ceo\_resigned}\}
$$

The probability of each scenario is computed via Bayesian-updated ARA predictive distribution over $D_0^{\text{ceo}}$ (see §9.2):

$$
\Pr_B(\text{CEO\_resign}) = \frac{\alpha_0 + n_{\text{resign}}}{\alpha_0 + \beta_0 + N}
$$

where $\alpha_0 = 12.5$, $\beta_0 = 0.5$ (Beta prior from 12 Australian no-remorse CEO observations), and $n_{\text{resign}}$ is ARA soft evidence.

### 1.5 Conditional branching structure

When $D_0^{\text{ceo}} = \text{CEO\_stay}$ and $D_1 \ne \text{D3\_ceo\_transition}$, the full tree unfolds:

$$
D_1 \rightarrow A_2 \rightarrow V \rightarrow M_{\text{agm}} \rightarrow D_4 \rightarrow D_{\text{rev}} \rightarrow R \rightarrow M_{\text{rev}} \rightarrow \begin{cases} D_4' \rightarrow D_{\text{rev}}' \rightarrow \text{Terminal} & \text{if adverse} \wedge \text{CEO\_present} \\ \text{Terminal} & \text{otherwise} \end{cases}
$$

When CEO is removed early ($D_0^{\text{ceo}} = \text{CEO\_resign}$ or $D_1 = \text{D3\_ceo\_transition}$):

$$
D_1 \rightarrow A_2 \rightarrow V \rightarrow M_{\text{agm}} \rightarrow [\text{D4 skip}] \rightarrow D_{\text{rev}} \rightarrow R \rightarrow M_{\text{rev}} \rightarrow \text{Terminal}
$$

---

## 2. Belief checkpoint draws

Each checkpoint $c \in \{C_{\text{pre}}, C_0, C_1, C_2, C_3\}$ contains $N$ draws (typically 500) from upstream Stan models:

$$
\theta_c^{(i)} = \bigl(B_{\text{mkt}}^{(i)},\; B_{\text{mgmt}}^{(i)},\; \alpha_V^{(i)},\; \gamma_A^{(i)},\; \gamma_{AH}^{(i)},\; \gamma_D^{(i)},\; \sigma_V^{(i)},\; r_1^{(i)},\; r_2^{(i)}\bigr), \quad i = 1,\ldots,N
$$

Opponent parameter priors $\Theta_j \sim p_B(\Theta_j)$ for $j \in \{\text{ASA}, \text{CEO}\}$ are specified in opponent_priors.xlsx. Supported families: Normal, LogNormal, Beta, Uniform, Gamma.

---

## 3. Chance models

### 3.1 Vote model — logit-normal with headline interaction and crisis floor

$$
B_{\text{agm}} = B_{\text{mkt}}^{(i)} + \gamma_A^{(i)} \cdot \mathbb{1}[A_2 = \text{rec\_strike}] + \gamma_{AH}^{(i)} \cdot \mathbb{1}[A_2 = \text{rec\_strike}] \cdot \mathbb{1}[\text{headline}] + \gamma_D^{(i)} \cdot f(D_1)
$$

Governance effect:

$$
f(D_1) = \begin{cases}
0 & D_1 = \text{D0\_minimal} \\
U(\ell_1, u_1) & D_1 = \text{D1\_review} \\
U(\ell_3, u_3) & D_1 = \text{D3\_ceo\_transition}
\end{cases}
$$

Unbiased bounds: $(\ell_1, u_1) = (0, 1)$, $(\ell_3, u_3) = (-1, 0.5)$.

Vote fraction from logit-normal:

$$
\text{logit}(V) \sim \mathcal{N}\!\bigl(\alpha_V^{(i)} + B_{\text{agm}},\; \sigma_V^{(i)}\bigr), \qquad V_{\text{logit}} = \text{expit}(\text{logit}(V))
$$

**Structural crisis floor** (headline incidents only):

$$
V_{\text{floor}} \sim \text{Beta}(50, 150), \quad V_{\text{final}} = \max(V_{\text{logit}}, V_{\text{floor}})
$$

$V_{\text{floor}}$ drawn once per belief draw (epistemic).

Derived indicators:

$$
\text{strike} = \mathbb{1}[V_{\text{final}} \ge 0.25], \qquad \text{overwhelming} = \mathbb{1}[V_{\text{final}} \ge 0.50]
$$

**Epistemic/aleatoric separation:** Three epistemic quantities drawn once per belief draw:
1. $f(D_1) \sim U(\ell, u)$ — governance effect
2. $V_{\text{floor}} \sim \text{Beta}(50, 150)$ — crisis floor
3. $\kappa_\sigma$ — sigma scale (when Board overconfidence bias is active)

### 3.2 Review findings — two-component model

**Component 1: Outcome rating (trinary).**

$$
(p_{\text{neg}}, p_{\text{bal}}, p_{\text{pos}}) \sim \text{Dirichlet}(38, 160, 1)
$$

$$
\mathbb{E} = (0.191, 0.804, 0.005)
$$

Balanced/neutral dominates (~80%) because board-commissioned reviews in crisis contexts admit "mistakes were made" without conceding legal liability. Negative is material (~19%) due to ACCC severity. Positive is negligible (<1%).

$$
\text{review\_outcome} \sim \text{Categorical}(p_{\text{neg}}, p_{\text{bal}}, p_{\text{pos}})
$$

$(p_{\text{neg}}, p_{\text{bal}}, p_{\text{pos}})$ drawn once per belief draw (epistemic). Outcome values: `"negative"`, `"balanced"`, `"positive"` (or `"none"` if review not commissioned).

The post-review conditional round ($D_4' \rightarrow D_{\text{rev}}'$) activates only for **negative** outcomes (not balanced or positive).

**Component 2: CAR.**

$$
\mu_f \sim t_4(-0.05, 0.03), \quad \sigma_f \sim |\mathcal{N}(0, 0.10)|, \quad \text{CAR} \sim t_3(\mu_f, \sigma_f)
$$

The outcome rating and CAR are independent: adverse reviews can produce mildly positive CAR (market had priced in bad news) and vice versa.

If review not commissioned: $\text{CAR} = 0$, $\text{adverse} = \text{false}$.

### 3.3 Review direct cost

$$
C_{\text{direct}} \sim \text{Gamma}(4.55, 4741), \quad \text{mean} \approx 9.6 \text{ bps}
$$

Drawn once per scenario (epistemic).

---

## 4. Board overconfidence bias

When Board is focal, three cognitive biases are applied:

### 4.1 Overestimation on governance effects

$$
f(D_1 = \text{D1\_review}) \sim U(0.63, 1.0), \quad f(D_1 = \text{D3}) \sim U(-0.62, 0.5)
$$

($\beta = 0.625$ midpoint overestimation.)

### 4.2 Overprecision on vote uncertainty

$$
\hat{\sigma}_V = \sigma_V \cdot \kappa_\sigma, \qquad \kappa_\sigma = 1/\sqrt{\kappa}, \quad \kappa = 3.5
$$

Production default: $\kappa_\sigma = 0.53$.

### 4.3 Overestimation on review

**CAR location:**

$$
\hat{\mu}_f = \mu_f + 0.03
$$

**Outcome rating:** Board believes positive outcomes are more likely:

$$
\hat{\alpha}_{\text{pos}} = 1 \times (1 + 10 \times 0.03) = 1.3
$$

$$
(p_{\text{neg}}, p_{\text{bal}}, p_{\text{pos}})^{\text{biased}} \sim \text{Dirichlet}(38, 160, 1.3)
$$

Default bias (0.03): slight tilt toward positive outcomes.

### 4.4 Bias propagation

All biases apply consistently in both the Board's own EU calculation and in rollout simulations for predictive distributions, producing self-consistent decision-making.

---

## 5. Utility functions

### 5.1 Terminal outcome

$$
Z = (d_1, a_2, d_{\text{rev}}, d_4, d_4', d_{\text{rev}}', V, \text{strike}, \text{overwhelming}, \text{CAR}, C_{\text{direct}}, \text{adverse}, \text{CEO\_removed}, \text{CEO\_resigned\_early}, \text{review\_commissioned})
$$

### 5.2 Board utility (focal)

Board minimises opposition and disruption. The utility function has three structural layers.

Define:
- $\text{ceo\_at\_end} = \neg\text{CEO\_removed} \;\wedge\; \neg\text{CEO\_resigned\_early}$
- $\text{removed\_involuntary} = \text{CEO\_removed} \;\wedge\; \neg\text{CEO\_resigned\_early}$
- $\text{board\_inactive} = (d_1 = \text{D0\_minimal}) \;\wedge\; d_{\text{rev}} \notin \{\text{sack}, \text{review}\} \;\wedge\; d_{\text{rev}}' \ne \text{sack}$

$$
\begin{aligned}
u_B(Z) &= \underbrace{
  -w_{\text{inact\_base}} \cdot \mathbb{1}[\text{board\_inactive}]
  - w_{\text{inact\_no\_rev}} \cdot \mathbb{1}[\neg\text{review\_commissioned}]
  - w_{\text{inact\_ceo}} \cdot \mathbb{1}[\text{ceo\_at\_end}]
  - w_{\text{inact\_no\_sack}} \cdot \mathbb{1}[\neg\text{removed\_involuntary}]
}_{\text{1. Inaction components (unconditional)}} \\[6pt]
&\quad \underbrace{
  - w_{\text{v\_strike}} \cdot \frac{(V - 0.25)_+}{0.75}
  - w_{\text{v\_over}} \cdot \frac{(V - 0.50)_+}{0.50}
}_{\text{2. Vote penalties (linear in normalized excess)}} \\[6pt]
&\quad - w_{\text{pass}} \cdot \mathbb{1}[\text{CEO\_resigned\_early}] \\
&\quad + w_{\text{CAR}}^{+} \cdot (\text{CAR})_+ \cdot \mathbb{1}[\text{review\_comm}]
  - w_{\text{CAR}}^{-} \cdot (-\text{CAR})_+ \cdot \mathbb{1}[\text{review\_comm}] \\
&\quad - w_{\text{cost}} \cdot C_{\text{direct}} \cdot \mathbb{1}[\text{review\_comm}] \\
&\quad - w_{\text{impl}} \cdot \bigl(\mathbb{1}[d_1 = \text{D3}] + \mathbb{1}[d_{\text{rev}} = \text{sack}] + \mathbb{1}[d_{\text{rev}}' = \text{sack}]\bigr) \\
&\quad - \max\!\bigl(0,\; w_{\text{loss}} - w_{\text{loss\_over}} \cdot \mathbb{1}[\text{overwhelming}]\bigr) \cdot \mathbb{1}[\text{removed\_involuntary}] \\
&\quad - w_{\text{rev\_neg}} \cdot \mathbb{1}[\text{review\_comm} \;\wedge\; \text{outcome} = \text{negative}] \\
&\quad - w_{\text{rev\_bal}} \cdot \mathbb{1}[\text{review\_comm} \;\wedge\; \text{outcome} = \text{balanced}] \\
&\quad - w_{\text{rev\_post}} \cdot \mathbb{1}[\text{removed\_involuntary} \;\wedge\; \neg\text{review\_comm}]
\end{aligned}
$$

**Review CAR loss aversion:**

$$
w_{\text{CAR}}^{+} = \frac{w_{\text{CAR}}}{\frac{1 + \lambda_{\text{la}}}{2}}, \qquad w_{\text{CAR}}^{-} = \lambda_{\text{la}} \cdot w_{\text{CAR}}^{+}
$$

where $w_{\text{CAR}} = 15.0$ (anchor) and $\lambda_{\text{la}} = 2.25$ (loss aversion). Positive CARs receive weight $\approx 9.23$; negative CARs receive weight $\approx 20.77$.

Full default weights:

| Parameter                          | Symbol                       | Default |
| ---------------------------------- | ---------------------------- | ------- |
| `inaction_base_penalty`            | $w_{\text{inact\_base}}$     | 3.0     |
| `inaction_no_review_penalty`       | $w_{\text{inact\_no\_rev}}$  | 2.0     |
| `inaction_ceo_present_penalty`     | $w_{\text{inact\_ceo}}$      | 5.0     |
| `inaction_no_sack_penalty`         | $w_{\text{inact\_no\_sack}}$ | 3.0     |
| `vote_strike_penalty`              | $w_{\text{v\_strike}}$       | 2.0     |
| `vote_overwhelming_penalty`        | $w_{\text{v\_over}}$         | 3.0     |
| `board_passivity_after_departure`  | $w_{\text{pass}}$            | 0.5     |
| `review_car_weight`                | $w_{\text{CAR}}$             | 15.0    |
| `review_car_loss_aversion`         | $\lambda_{\text{la}}$        | 2.25    |
| `review_direct_cost_weight`        | $w_{\text{cost}}$            | 15.0    |
| `implementation_cost_sack`         | $w_{\text{impl}}$            | 1.0     |
| `ceo_loss_cost`                    | $w_{\text{loss}}$            | 1.5     |
| `ceo_loss_shock_overwhelming`      | $w_{\text{loss\_over}}$      | 0.5     |
| `negative_review_finding_penalty`  | $w_{\text{rev\_neg}}$        | 5.0     |
| `balanced_review_finding_penalty`  | $w_{\text{rev\_bal}}$        | 2.5     |
| `review_after_removal_penalty`     | $w_{\text{rev\_post}}$       | 3.0     |

### 5.3 ASA utility (opponent — used in predictive distribution)

Seven-dimensional weighted assessment model. Each dimension is scored on a [1, 5] Likert scale with dimension-specific weights summing to 1.0:

| Dimension | Weight | Description |
|-----------|--------|-------------|
| FW        | 0.10   | Financial Welfare — share price, legal exposure |
| PPL       | 0.30   | Pay/Performance Linkage — remuneration outcomes, clawback |
| TD        | 0.10   | Transparency/Disclosure — governance disclosure quality |
| EGR       | 0.15   | ESG/Governance Risk — regulatory, labour, ESG signals |
| BA        | 0.20   | Board Accountability — consequences imposed on management |
| OL        | 0.10   | Organizational Legitimacy — ASA member trust, credibility |
| PF        | 0.05   | Procedural Fairness — AGM process, share trading norms |

$$
u_A(Z) = \sum_{d \in \mathcal{D}} w_d \cdot \text{clip}\!\bigl(s_d^{\text{base}}(Z) + \Delta s_d(Z),\; 1,\; 5\bigr) - w_{\text{mob}} \cdot \mathbb{1}[a_2 = \text{rec\_strike}]
$$

**Base scores** $s_d^{\text{base}}$ depend on the path to $A_2$: (CEO\_resigned\_early, $d_1$) determines the lookup key into a 5-row table of base Likert scores per dimension.

| Path | FW | PPL | TD | EGR | BA | OL | PF | Weighted Mean |
|------|-----|-----|-----|------|-----|-----|-----|---------------|
| CEO resign → D0\_minimal | 2.1 | 1.3 | 1.8 | 1.5 | 2.3 | 2.0 | 3.0 | 1.84 |
| CEO resign → D1\_review | 2.2 | 1.3 | 2.2 | 1.9 | 2.9 | 2.4 | 3.0 | 2.09 |
| CEO stay → D0\_minimal | 1.7 | 1.2 | 1.5 | 1.3 | 1.2 | 1.4 | 2.9 | 1.43 |
| CEO stay → D1\_review | 1.9 | 1.2 | 2.0 | 1.6 | 1.9 | 1.7 | 2.9 | 1.67 |
| CEO stay → D3\_ceo\_transition | 2.5 | 1.6 | 2.3 | 2.2 | 3.3 | 2.7 | 3.0 | 2.27 |

**Post-A2 adjustments** $\Delta s_d$:
- Strike: $\Delta \text{BA} += 1.5$, $\Delta \text{OL} += 1.0$
- Overwhelming: $\Delta \text{BA} += 1.0$, $\Delta \text{OL} += 0.5$
- CEO removed (involuntary): $\Delta \text{BA} += 1.0$, $\Delta \text{FW} += 0.5$
- Negative review: $\Delta \text{TD} += 1.0$, $\Delta \text{EGR} += 0.5$
- Balanced review: $\Delta \text{TD} += 0.3$
- Market alignment (rec\_strike $\wedge$ strike): $\Delta \text{OL} += 1.0$, $\Delta \text{PF} += 0.5$

Default mobilisation cost: $w_{\text{mob}} = 0.3$.

### 5.4 CEO utility (opponent — used in predictive distribution)

$$
u_C(Z) = U_{\text{money}}(W) - \lambda_D \cdot D_{\text{raw}}
$$

CRRA with Kahneman–Tversky loss aversion:

$$
\text{CRRA}(W) = \begin{cases}
W^{1-\gamma}/(1-\gamma) & \gamma \ne 1 \\
\ln(W) & \gamma = 1
\end{cases}
$$

$$
U_{\text{money}}(W) = \begin{cases}
\text{CRRA}(W) & W \ge W_{\text{ref}} \\
\lambda \cdot \text{CRRA}(W) - (\lambda - 1) \cdot \text{CRRA}(W_{\text{ref}}) & W < W_{\text{ref}}
\end{cases}
$$

**Resign path:** $W = W_{\text{resign}}$, $D_{\text{raw}} = D_{\text{resign}}$.

**Stay path — wealth:**

$$
W = \begin{cases}
W_{\text{stay\_kept}} & \text{CEO not removed} \\
\frac{1}{2}(W_{\text{stay\_sacked}} + W_{\text{stay\_kept}}) & \tilde{d}_4 = \text{negotiate\_exit} \\
1.3 \cdot W_{\text{stay\_sacked}} & \tilde{d}_4 = \text{resign} \\
W_{\text{stay\_sacked}} & \text{otherwise (forced removal)}
\end{cases}
$$

**Stay path — disutility (departure-mode-dependent):**

$$
D_{\text{raw}} = D_{\text{stay}} + D_{\text{departure}}(\tilde{d}_4) \cdot \mathbb{1}[\text{CEO\_removed}] + D_{\text{agm}} \cdot \mathbb{1}[V > 0.25] + D_{\text{disgrace}} \cdot \mathbb{1}[\text{overwhelming}] + D_{\text{adverse\_review}} \cdot \mathbb{1}[\text{review\_commissioned} \;\wedge\; \text{outcome} = \text{negative}]
$$

$$
D_{\text{departure}}(\tilde{d}_4) = \begin{cases}
D_{\text{negotiate}} = 45 & \tilde{d}_4 = \text{negotiate\_exit} \\
D_{\text{resign\_late}} = 60 & \tilde{d}_4 = \text{resign} \\
D_{\text{sacked}} = 100 & \text{otherwise (Board fires CEO)}
\end{cases}
$$

$\tilde{d}_4$ is the effective CEO departure action ($d_4'$ overrides $d_4$ if the CEO acted post-review).

Default parameters:

| Parameter | Default |
|-----------|---------|
| $\gamma$ | 1.5 |
| $\lambda = \lambda_D$ | 2.25 |
| $W_{\text{ref}}$ | 16.0 |
| $W_{\text{resign}}$ | 8.0 |
| $W_{\text{stay\_kept}}$ | 7.0 |
| $W_{\text{stay\_sacked}}$ | 1.5 |
| $D_{\text{resign}}$ | 40.0 |
| $D_{\text{stay}}$ | 25.0 |
| $D_{\text{sacked}}$ | 100.0 |
| $D_{\text{resign\_late}}$ | 60.0 |
| $D_{\text{negotiate}}$ | 45.0 |
| $D_{\text{agm}}$ | 30.0 |
| $D_{\text{disgrace}}$ | 30.0 |
| $D_{\text{adverse\_review}}$ | 10.0 |

---

## 6. ARA predictive distributions (Board perspective)

### 6.1 General form

At an opponent decision node $X$ owned by player $j \in \{\text{ASA}, \text{CEO}\}$:

$$
\boxed{
p_B(X = x \mid h) = \frac{1}{K} \sum_{k=1}^{K} \mathbb{1}\!\left[x \in \arg\max_{x' \in \mathcal{X}(h)} \Psi_j(x'; h, \Theta_j^{(k)})\right]
}
$$

- $K = 200$ opponent parameter samples: $\Theta_j^{(k)} \sim p_B(\Theta_j)$
- $\Psi_j(x; h, \Theta_j) = \frac{1}{R} \sum_{r=1}^{R} u_j(Z^{(r)}; \Theta_j)$, $R = 20$

Rollouts use fixed policies for non-evaluated actors and the Board's (possibly biased) model of Nature.

### 6.2 Fixed policies (Level-1 rollout fallbacks)

**Board at $D_1$:** $\pi_B^{\text{fix}}(D_1) = \text{D0\_minimal}$

**Board at $D_{\text{rev}}$ / $D_{\text{rev}}'$:**

$$
\pi_B^{\text{fix}}(D_{\text{rev}} \mid h) = \begin{cases}
\text{Drev\_sack\_ceo} & V_\% \ge \tau_{\text{sack}} \;\wedge\; \text{CEO\_present} \\
\text{Drev\_commission\_review} & V_\% \ge \tau_{\text{review}} \;\wedge\; \text{review\_not\_commissioned} \\
\text{Drev\_no\_action} & \text{otherwise}
\end{cases}
$$

$\tau_{\text{sack}} = 0.25$, $\tau_{\text{review}} = 0.25$ (from `policy_parameters` sheet).

**ASA at $A_2$ — Beta-Binomial:**

| Board action | Posterior | Mean |
|-------------|-----------|------|
| D0\_minimal | $\text{Beta}(46, 4)$ | 0.920 |
| D1\_review / CEO resigned | $\text{Beta}(44, 4)$ | 0.917 |
| D3\_ceo\_transition | $\text{Beta}(43, 4)$ | 0.914 |

$$
p_{\text{strike}} \sim \text{Beta}(\alpha_d, \beta_d), \quad A_2 \sim \text{Bernoulli}(p_{\text{strike}})
$$

**CEO at $D_0^{\text{ceo}}$:** $\pi_C^{\text{fix}} = \text{CEO\_stay}$

**CEO at $D_4$ / $D_4'$:**

$$
\pi_C^{\text{fix}}(D_4) = \begin{cases}
\text{D4\_resign} & V_\% \ge 0.40 \;\wedge\; \text{feasible} \\
\text{D4\_stay} & \text{otherwise}
\end{cases}
$$

### 6.3 Level-2 recursion

At Level-2, opponents model the Board strategically:
- ASA models Board: $p_{\text{ASA}}^{(1)}(D_1 = d \mid h)$ replaces $\pi_B^{\text{fix}}$ at $D_1$
- CEO models Board: $p_{\text{CEO}}^{(1)}(D_{\text{rev}} = d \mid h)$ replaces $\pi_B^{\text{fix}}$ at $D_{\text{rev}}$

Nested predictions use reduced $K_{\text{nested}}$ and $R_{\text{nested}}$.

---

## 7. Board-focal tree recursion

### 7.1 Node dispatch

| Node type | Owner | Operation |
|-----------|-------|-----------|
| Decision | Board | $\max$ over feasible actions |
| Decision | ASA / CEO | $\sum$ weighted by $p_B(\cdot)$ or fixed policy |
| Chance ($V$) | Nature | MC average over $M_V$ vote samples |
| Chance ($R$) | Nature | MC average over $M_R$ review samples |
| Chance ($M_{\text{agm}}, M_{\text{rev}}$) | Nature | Pass-through |
| Terminal | — | Compute $u_B(Z)$ |

### 7.2 Full recursion (ceo\_stayed scenario)

$$
U_B^{(D_1)}(h_0) = \max_{d_1 \in \mathcal{D}_1} U_B^{(A_2)}(h_0 \cup \{D_1 = d_1\})
$$

$$
U_B^{(A_2)}(h) = \sum_{a \in \mathcal{A}_2} p_B(A_2 = a \mid h) \cdot U_B^{(V)}(h \cup \{A_2 = a\})
$$

$$
U_B^{(V)}(h) = \frac{1}{M_V} \sum_{m} U_B^{(D_4)}(h \cup \{V_m\})
$$

$M_{\text{agm}}$ is a pass-through.

**CEO responds** (opponent — predictive or skip if absent):

$$
U_B^{(D_4)}(h) = \begin{cases}
\displaystyle\sum_{d_4} p_B(D_4 = d_4 \mid h) \cdot U_B^{(D_{\text{rev}})}(h \cup \{D_4 = d_4\}) & \text{if CEO\_present} \\[6pt]
U_B^{(D_{\text{rev}})}(h) & \text{if CEO absent}
\end{cases}
$$

**Board responds** (focal — maximise):

$$
U_B^{(D_{\text{rev}})}(h) = \max_{d \in \mathcal{D}_{\text{rev}}(S)} U_B^{(R)}(h \cup \{D_{\text{rev}} = d\})
$$

**Review findings:**

$$
U_B^{(R)}(h) = \frac{1}{M_R} \sum_{m} U_B^{(\text{post-review})}(h \cup \{R_m, C_{\text{direct}}\})
$$

$M_{\text{rev}}$ is a pass-through.

**Post-review conditional round:**

$$
U_B^{(\text{post-review})}(h) = \begin{cases}
U_B^{(D_4')}(h) & \text{if review\_adverse} \wedge \text{CEO\_present} \\
u_B(Z(h, S)) & \text{otherwise}
\end{cases}
$$

$$
U_B^{(D_4')}(h) = \sum_{d_4} p_B(D_4 = d_4 \mid h) \cdot U_B^{(D_{\text{rev}}')}(h \cup \{D_4 = d_4\})
$$

$$
U_B^{(D_{\text{rev}}')}(h) = \max_{d \in \mathcal{D}_{\text{rev}}(S)} u_B(Z(h \cup \{D_{\text{rev}} = d\}, S))
$$

### 7.3 Board optimal action

$$
d_1^* \in \arg\max_{d_1 \in \mathcal{D}_1} \frac{1}{N} \sum_{i=1}^{N} U_B^{(A_2)}(h_0 \cup \{D_1 = d_1\}; \theta^{(i)})
$$

### 7.4 Predictive distributions required

| Distribution | Node | Opponent | Purpose |
|-------------|------|----------|---------|
| $p_B(A_2 \mid h)$ | $A_2$ | ASA | Strike recommendation |
| $p_B(D_4 \mid h)$ | $D_4$ | CEO | Post-AGM response |
| $p_B(D_4' \mid h)$ | $D_4'$ | CEO | Post-review response (same model) |

---

## 8. Mode configurations (Board only)

| Mode               | ASA model  | CEO model | Level |
| ------------------ | ---------- | --------- | ----- |
| Board Mode         | ARA        | ARA       | 1     |
| Board L2           | ARA        | ARA       | 2     |
| Board (ASA=Policy) | Policy     | ARA       | 1     |

Level-2 strategic counterparts: ASA models Board, CEO models Board.

---

## 9. Solver orchestration

### 9.1 Pipeline

For checkpoint $c$, scenario $s$:

1. Load $\text{BeliefBundle}(c)$, utility weights, policy parameters, overconfidence bias.
2. Construct engine: $\text{ChanceModels}$, $\text{PredictiveDistribution}(K{=}200, R{=}20)$, $\text{TreeEvaluator}(M_V{=}50, M_R{=}20)$.
3. Set initial state: $S_0 = \text{for\_scenario}(s)$.
4. For each $d_1 \in \mathcal{D}_1(S_0)$: $\text{EU}(d_1) = \frac{1}{N} \sum_{i=1}^{N} U_B^{(A_2)}(\{D_1 = d_1\}; \theta^{(i)})$.
5. $d_1^* = \arg\max_{d_1} \text{EU}(d_1)$.

### 9.2 D0\_ceo Bayesian prediction

**Prior:** $p_{\text{resign}} \sim \text{Beta}(12.5, 0.5)$, $\mathbb{E} = 0.962$ (Jeffreys + 12 Australian CEO resignations).

**ARA evidence:** Level-2 predictive over $N$ belief draws yields soft pseudo-counts $n_{\text{resign}}, n_{\text{stay}}$.

**Posterior:**

$$
\Pr_B(\text{CEO\_resign}) = \frac{12.5 + n_{\text{resign}}}{13.0 + N}
$$

Computed with reduced budget: $K_{\text{d0}} = 50$, $R_{\text{d0}} = 10$.

### 9.3 Scenario solver

1. $\Pr_B(D_0^{\text{ceo}}) = \text{predict\_d0\_ceo}()$
2. For each $s \in \{\text{ceo\_stayed}, \text{ceo\_resigned}\}$: solve and attach $\Pr_B$.

### 9.4 Parallelisation

Each $(d_1, i)$ pair submitted to `ProcessPoolExecutor`. Workers initialised once per checkpoint.

---

## 10. Computational budget

| Parameter | Symbol | Default |
|-----------|--------|---------|
| Belief draws | $N$ | 500 (or `--n_draws`) |
| Opponent parameter samples | $K$ | 200 |
| Stochastic rollouts | $R$ | 20 |
| Vote MC samples | $M_V$ | 50 |
| Review MC samples | $M_R$ | 20 |
| D0\_ceo opponent samples | $K_{\text{d0}}$ | 50 |
| D0\_ceo rollouts | $R_{\text{d0}}$ | 10 |

Total tree evaluations per solve: $|\mathcal{D}_1| \times N$.
Total rollouts per predictive call: $K \times |\mathcal{A}| \times R$.
