# Algebraic Notation — Qantas ARA Engine

This document specifies the mathematical structure implemented in `engine/`.
Every formula maps to executable code; section headings reference the source module.

---

## 1. Game tree structure (`state.py`)

### 1.1 Node order

The tree is **not** a fixed linear sequence. Node traversal is conditional on game state, principally on whether the CEO is still present. The base sequence is:

$$
D_0^{\text{ceo}} \rightarrow D_1 \rightarrow A_2 \rightarrow V \rightarrow M_{\text{agm}} \rightarrow D_4 \rightarrow D_{\text{rev}} \rightarrow R \rightarrow M_{\text{rev}} \rightarrow [\text{conditional } D_4 \rightarrow D_{\text{rev}}] \rightarrow \text{Terminal}
$$

**Key ordering principle:** $D_4$ (CEO response) precedes $D_{\text{rev}}$ (Board response). The CEO has initiative to resign before the Board decides whether to act.

**Post-review conditional round:** After $R$ and $M_{\text{rev}}$, if $\text{review\_adverse} = \text{true}$ AND $\text{CEO\_present} = \text{true}$, a second $D_4 \rightarrow D_{\text{rev}}$ round occurs. Otherwise, the game proceeds directly to Terminal.

| Node                 | Type     | Owner  | Purpose                                        |
| -------------------- | -------- | ------ | ---------------------------------------------- |
| $D_0^{\text{ceo}}$ | Decision | CEO    | Pre-game resignation (05-Sep-2023)             |
| $D_1$              | Decision | Board  | Governance reform package                      |
| $A_2$              | Decision | ASA    | Strike recommendation                          |
| $V$                | Chance   | Nature | Shareholder vote (logit-normal)                |
| $M_{\text{agm}}$   | Chance   | Nature | Post-AGM market reaction (pass-through)        |
| $D_4$              | Decision | CEO    | CEO response to AGM outcome                    |
| $D_{\text{rev}}$   | Decision | Board  | Board review response / CEO removal            |
| $R$                | Chance   | Nature | Review findings release (Student-$t$ + Bernoulli) |
| $M_{\text{rev}}$   | Chance   | Nature | Post-review market reaction (pass-through)     |
| $D_4'$             | Decision | CEO    | CEO response to adverse review (conditional)   |
| $D_{\text{rev}}'$  | Decision | Board  | Board response to adverse review (conditional) |
| Terminal             | Terminal | —     | Compute utilities                              |

$M_{\text{agm}}$ and $M_{\text{rev}}$ are pass-through nodes: no sampling or branching occurs.

$D_4'$ and $D_{\text{rev}}'$ share the same action sets and feasibility rules as $D_4$ and $D_{\text{rev}}$ respectively. They are only reached when $\text{review\_adverse} \wedge \text{CEO\_present}$.

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

The game state $S$ is an immutable record with fields:

$$
S = (\text{CEO\_present},\; \text{review\_commissioned},\; \text{review\_completed},\; \text{CEO\_removed},\; \text{CEO\_resigned\_early},\; \text{review\_adverse},\; \text{post\_review\_round},\; \text{headline\_incident},\; \text{checkpoint\_id})
$$

The `headline_incident` flag (default `true` for Qantas) indicates whether the company has experienced a high-profile governance or conduct failure. It conditions the vote model's ASA interaction term and structural crisis floor (§3.1).

Applying an action returns a new copy: $S' = \text{apply}(S, n, a)$.

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

The probability of each scenario is computed via Bayesian-updated ARA predictive distribution over $D_0^{\text{ceo}}$ (see §10.2):

$$
\Pr_i(\text{CEO\_resign}) = \frac{\alpha + n_{\text{resign}}}{\alpha + \beta + N}
$$

where $\alpha, \beta$ are Beta prior pseudo-counts and $n_{\text{resign}}$ is the ARA soft evidence accumulated over $N$ belief draws. Each scenario prunes the action space downstream (e.g., `ceo_resigned` makes $D_4$ infeasible because CEO is already gone).

### 1.5 Conditional branching structure

When $D_0^{\text{ceo}} = \text{CEO\_stay}$ and $D_1 \ne \text{D3\_ceo\_transition}$, the full tree unfolds:

$$
D_1 \rightarrow A_2 \rightarrow V \rightarrow M_{\text{agm}} \rightarrow D_4 \rightarrow D_{\text{rev}} \rightarrow R \rightarrow M_{\text{rev}} \rightarrow \begin{cases} D_4' \rightarrow D_{\text{rev}}' \rightarrow \text{Terminal} & \text{if adverse} \wedge \text{CEO\_present} \\ \text{Terminal} & \text{otherwise} \end{cases}
$$

When CEO is removed early ($D_0^{\text{ceo}} = \text{CEO\_resign}$ or $D_1 = \text{D3\_ceo\_transition}$):

$$
D_1 \rightarrow A_2 \rightarrow V \rightarrow M_{\text{agm}} \rightarrow [\text{D4 skip}] \rightarrow D_{\text{rev}} \rightarrow R \rightarrow M_{\text{rev}} \rightarrow \text{Terminal}
$$

All $D_4$, $D_4'$ nodes are skipped when $\text{CEO\_present} = \text{false}$. $D_{\text{rev}}$ actions are restricted to exclude `Drev_sack_ceo` when CEO is absent.

---

## 2. Belief checkpoint draws (`state.py`)

### 2.1 Posterior draws

Each checkpoint $c \in \{C_{\text{pre}}, C_0, C_1, C_2, C_3\}$ contains $N$ draws (typically 500) from upstream Stan models:

$$
\theta_c^{(i)} = \bigl(B_{\text{mkt}}^{(i)},\; B_{\text{mgmt}}^{(i)},\; \alpha_V^{(i)},\; \gamma_A^{(i)},\; \gamma_{AH}^{(i)},\; \gamma_D^{(i)},\; \sigma_V^{(i)},\; r_1^{(i)},\; r_2^{(i)}\bigr), \quad i = 1,\ldots,N
$$

All parameters are `float64` arrays indexed by draw $i$. The first seven are consumed by the vote model: $\gamma_A$ is the base ASA mobilisation effect, and $\gamma_{AH}$ is the additional headline-interaction shift (§3.1). The review parameters $r_1, r_2$ (`review_param_1`, `review_param_2`) are loaded from checkpoints but not currently consumed by the review chance model (which samples its own hierarchical parameters).

### 2.2 Opponent parameter priors

The focal actor's uncertainty over opponent $j$'s utility parameters $\Theta_j$ is specified via prior distributions (opponent_priors.xlsx):

$$
\Theta_j \sim p_i(\Theta_j), \qquad j \in \{\text{Board}, \text{ASA}, \text{CEO}\} \setminus \{i\}
$$

Supported distribution families: Normal, LogNormal, Beta, Uniform, Gamma.

Keyed by `(perspective_actor, target_actor, parameter_name)`.

---

## 3. Chance models (`chance_models.py`)

### 3.1 Vote model — logit-normal with headline interaction and crisis floor

The AGM belief aggregates four components:

$$
B_{\text{agm}} = B_{\text{mkt}}^{(i)} + \gamma_A^{(i)} \cdot \mathbb{1}[A_2 = \text{rec\_strike}] + \gamma_{AH}^{(i)} \cdot \mathbb{1}[A_2 = \text{rec\_strike}] \cdot \mathbb{1}[\text{headline}] + \gamma_D^{(i)} \cdot f(D_1)
$$

where:
- $\gamma_A^{(i)}$ is the base ASA mobilisation effect (non-crisis baseline)
- $\gamma_{AH}^{(i)}$ is the additional headline-interaction shift: the incremental ASA effect when both a strike is recommended *and* the company has a headline incident. This addresses insufficient ASA differentiation in crisis vs non-crisis cases — a single pooled $\gamma_A$ understates the ASA effect in crises and overstates it in non-crisis cases.
- $\mathbb{1}[\text{headline}] = S.\text{headline\_incident}$

The governance effect $f(D_1)$ is sampled from a Uniform distribution:

$$
f(D_1) = \begin{cases}
0 & D_1 = \text{D0\_minimal} \\
U(\ell_1, u_1) & D_1 = \text{D1\_review} \\
U(\ell_3, u_3) & D_1 = \text{D3\_ceo\_transition}
\end{cases}
$$

Unbiased bounds: $(\ell_1, u_1) = (0, 1)$, $(\ell_3, u_3) = (-1, 0.5)$.

The D3 (CEO transition) effect uses $U(-1, 0.5)$ with $\mathbb{E}[f] = -0.25$: the four historical CEO-exit observations are confounded with crisis severity, and the data do not rule out a modest dampening effect on protest voting ($f > 0$). The upper bound of $0.5$ allows for the possibility that decisive Board action partially mollifies shareholders.

The vote fraction is drawn from the logit-normal:

$$
\text{logit}(V) \sim \mathcal{N}\!\bigl(\alpha_V^{(i)} + B_{\text{agm}},\; \sigma_V^{(i)}\bigr)
$$

$$
V_{\text{logit}} = \text{expit}\!\bigl(\text{logit}(V)\bigr) = \frac{1}{1 + e^{-\text{logit}(V)}}
$$

**Structural crisis floor.** In headline-incident cases, a minimum opposition level is imposed:

$$
V_{\text{floor}} \sim \text{Beta}(50, 150), \quad \mathbb{E}[V_{\text{floor}}] = 0.25, \quad \text{90\% CI} \approx [0.22, 0.28]
$$

$$
V_{\text{final}} = \begin{cases}
\max(V_{\text{logit}},\; V_{\text{floor}}) & \text{if headline\_incident} \\
V_{\text{logit}} & \text{otherwise}
\end{cases}
$$

$V_{\text{floor}}$ is drawn once per belief draw (epistemic: "what is the minimum opposition this crisis generates?") and held fixed across the $M_V$ vote samples within that draw. This prevents the logit-normal from under-producing first strikes in crisis scenarios — without the floor, the model assigns only 3–14% probability mass to $V < 0.25$ even with high median opposition.

Derived indicators (from `vote_thresholds` sheet):

$$
\text{strike} = \mathbb{1}[V_{\text{final}} \ge 0.25], \qquad \text{overwhelming} = \mathbb{1}[V_{\text{final}} \ge 0.50]
$$

**Epistemic/aleatoric separation:** Three epistemic quantities are drawn once per belief draw and held fixed across MC vote samples:
1. The governance effect $f(D_1)$ — "how effective is this reform?"
2. The crisis floor $V_{\text{floor}}$ — "what is the minimum opposition?"
3. (When biased) the sigma scale $\kappa_\sigma$ — Board's perceived precision

Vote samples within a draw reflect aleatoric noise (individual voter responses).

### 3.2 Review findings — two-component model (outcome rating + CAR)

The review produces two independently modelled outputs: a qualitative outcome rating (adverse vs positive) and a quantitative cumulative abnormal return (CAR).

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

**Component 2: CAR (market reaction to findings release).**

Student-$t$ hierarchy calibrated from ASX governance review case studies 2014–2023 (board-background/governance-review-case-studies.md):

$$
\mu_f \sim t_4(-0.05,\; 0.03)
$$

$$
\sigma_f \sim |\mathcal{N}(0,\; 0.10)|
$$

$$
\text{CAR} \sim t_3(\mu_f,\; \sigma_f)
$$

Calibrated from: Star $-13.95\%$, Westpac $-3.00\%$, CBA $+1.75\%$, Qantas $+0.85\%$.

The CAR captures the market's quantitative reaction. It is separate from the qualitative outcome rating: a "positive" review could still produce a mildly negative CAR (market expected more), and an "adverse" review could produce a mildly positive CAR (market had already priced in the bad news).

**Why $t_4$ for $\mu_f$, not Cauchy:** The Cauchy distribution ($t_1$) has no finite mean or variance. With `review_car_weight` $= 15$, a single extreme Cauchy draw (e.g., $\mu_f \approx -0.5$) contributes $15 \times (-0.5) = -7.5$ to Board utility and can dominate the Monte Carlo average, making "commission review" appear far worse than it actually is. $t_4$ is the minimum degrees-of-freedom giving both a finite mean ($\nu > 1$) and finite variance ($\nu > 2$), while still retaining genuinely heavy tails. The outer $\text{CAR} \sim t_3(\mu_f, \sigma_f)$ preserves heavy-tail capability for extreme events (Star $-13.95\%$).

Analytical mean: $\mathbb{E}[\mu_f] = -0.05$ (well-defined since $\nu = 4 > 1$). $\mathbb{E}[\text{CAR}] = \mathbb{E}[\mu_f] = -0.05$.

If review not commissioned: $\text{CAR} = 0$, $\text{review\_adverse} = \text{false}$ (deterministic).

### 3.3 Review direct cost — Gamma

$$
C_{\text{direct}} \sim \text{Gamma}(\alpha = 4.55,\; \beta = 4741)
$$

Properties: mean $\approx 9.6$ bps, SD $\approx 4.5$ bps, mode $\approx 7.5$ bps.

Drawn once per scenario (epistemic) and held fixed across MC review samples.

---

## 4. Board overconfidence bias (`chance_models.py`)

When Board is focal, the engine applies cognitive biases calibrated from the governance overconfidence literature.

### 4.1 Overestimation on governance effects

Boards overestimate the effectiveness of their governance actions by a factor $\beta \sim U(0.25, 1.0)$. Production midpoint $\beta = 0.625$. This shifts the Uniform bounds for $f(D_1)$:

$$
f(D_1 = \text{D1\_review}) \sim U(\ell_1^{\text{bias}}, u_1^{\text{bias}}), \quad f(D_1 = \text{D3}) \sim U(\ell_3^{\text{bias}}, u_3^{\text{bias}})
$$

Default: $(\ell_1^{\text{bias}}, u_1^{\text{bias}}) = (0.63, 1.0)$, $(\ell_3^{\text{bias}}, u_3^{\text{bias}}) = (-0.62, 0.5)$.

Note: the unbiased D3 ceiling is now $0.5$ (not $0$), so the biased Board's overestimation affects only the *floor* (narrowing the range upward from $-1$ to $-0.62$), while the ceiling remains at the unbiased value.

### 4.2 Overprecision on vote uncertainty

Boards perceive $\kappa \sim U(2, 5)$ times more precision than warranted:

$$
\hat{\sigma}_V = \sigma_V \cdot \kappa_\sigma, \qquad \kappa_\sigma = 1/\sqrt{\kappa}
$$

Production default: $\kappa = 3.5$, $\kappa_\sigma = 0.53$.

### 4.3 Overestimation on review

Board overestimates governance quality, affecting both the CAR location and the outcome rating probability.

**CAR location bias:**

$$
\hat{\mu}_f = \mu_f + \delta_{\text{CAR}}, \qquad \delta_{\text{CAR}} = 0.03
$$

Board perceives review CAR $\sim 3$ pp more favourable than actuarial.

**Outcome rating:** Board believes positive outcomes are more likely:

$$
\hat{\alpha}_{\text{pos}} = 1 \times (1 + 10 \times 0.03) = 1.3
$$

$$
(p_{\text{neg}}, p_{\text{bal}}, p_{\text{pos}})^{\text{biased}} \sim \text{Dirichlet}(38, 160, 1.3)
$$

Default bias (0.03): slight tilt toward positive outcomes.

### 4.4 Bias propagation

Biases are applied consistently:

- In the focal actor's own EU calculation (tree value)
- In rollout simulations for predictive distributions

This produces self-consistent decision-making under cognitive bias.

---

## 5. Utility functions (`utilities.py`)

### 5.1 Terminal outcome

A terminal outcome $Z$ is the tuple:

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

**Post-A2 adjustments** $\Delta s_d$:
- Strike: $\Delta \text{BA} += 1.5$, $\Delta \text{OL} += 1.0$
- Overwhelming: $\Delta \text{BA} += 1.0$, $\Delta \text{OL} += 0.5$
- CEO removed (involuntary): $\Delta \text{BA} += 1.0$, $\Delta \text{FW} += 0.5$
- Negative review: $\Delta \text{TD} += 1.0$, $\Delta \text{EGR} += 0.5$
- Balanced review: $\Delta \text{TD} += 0.3$
- Market alignment (rec\_strike $\wedge$ strike): $\Delta \text{OL} += 1.0$, $\Delta \text{PF} += 0.5$

Default mobilisation cost: $w_{\text{mob}} = 0.3$.

### 5.4 CEO utility — reference-dependent CRRA with loss aversion

$$
u_C(Z) = U_{\text{money}}(W) - \lambda_D \cdot D_{\text{raw}}
$$

where the CRRA function is:

$$
\text{CRRA}(W) = \begin{cases}
\dfrac{W^{1-\gamma}}{1-\gamma} & \gamma \ne 1 \\[6pt]
\ln(W) & \gamma = 1
\end{cases}
$$

with risk-aversion parameter $\gamma \in [0.5, 3.0]$ (default 1.5).

Monetary utility uses Kahneman–Tversky loss aversion around a reference point $W_{\text{ref}}$ (pre-crisis expected compensation):

$$
U_{\text{money}}(W) = \begin{cases}
\text{CRRA}(W) & W \ge W_{\text{ref}} \\[6pt]
\lambda \cdot \text{CRRA}(W) - (\lambda - 1) \cdot \text{CRRA}(W_{\text{ref}}) & W < W_{\text{ref}}
\end{cases}
$$

where $\lambda = 2.25$ per Tversky & Kahneman (1992) cumulative prospect theory estimates. Losses below the reference point are amplified by $\lambda$.

Non-monetary penalties $D_{\text{raw}}$ are also scaled by $\lambda_D$ (defaults to $\lambda$): an executive with a large ego evaluates reputational losses relative to their expected status as a powerful CEO, making sacking, public disgrace, and AGM humiliation feel disproportionately worse.

**Resign path** ($D_0^{\text{ceo}} = \text{CEO\_resign}$):

$$
W = W_{\text{resign}}, \qquad D_{\text{raw}} = D_{\text{resign}}
$$

**Stay path** ($D_0^{\text{ceo}} = \text{CEO\_stay}$):

Wealth depends on departure mode. Let $\tilde{d}_4$ be the effective CEO departure action (the last non-stay $D_4$ action, with $d_4'$ overriding $d_4$ if the CEO acted post-review):

$$
W = \begin{cases}
W_{\text{stay\_kept}} & \text{CEO not removed} \\
\frac{1}{2}(W_{\text{stay\_sacked}} + W_{\text{stay\_kept}}) & \tilde{d}_4 = \text{negotiate\_exit} \\
1.3 \cdot W_{\text{stay\_sacked}} & \tilde{d}_4 = \text{resign} \\
W_{\text{stay\_sacked}} & \text{otherwise (forced removal)}
\end{cases}
$$

$W$ values are calibrated for ACCC-era pay erosion (frozen LTIs, reserved clawbacks, legal costs).

Non-monetary disutility starts at a baseline crisis cost and is additive over outcome components, with **departure-mode-dependent penalties** replacing the old blanket $D_{\text{sacked}}$:

$$
D_{\text{raw}} = D_{\text{stay}} + D_{\text{departure}}(\tilde{d}_4) \cdot \mathbb{1}[\text{CEO\_removed}] + D_{\text{agm}} \cdot \mathbb{1}[V > 0.25] + D_{\text{disgrace}} \cdot \mathbb{1}[\text{overwhelming}] + D_{\text{adverse\_review}} \cdot \mathbb{1}[\text{review\_commissioned} \;\wedge\; \text{outcome} = \text{negative}]
$$

where the departure-mode penalty discriminates between involuntary and voluntary removal:

$$
D_{\text{departure}}(\tilde{d}_4) = \begin{cases}
D_{\text{negotiate}} & \tilde{d}_4 = \text{negotiate\_exit} \quad \text{(face-saving terms)} \\
D_{\text{resign\_late}} & \tilde{d}_4 = \text{resign} \quad \text{(controls narrative, but post-AGM)} \\
D_{\text{sacked}} & \text{otherwise} \quad \text{(Board fires CEO — maximum reputational destruction)}
\end{cases}
$$

This ensures that CEO $D_4$ actions (resign, negotiate) produce strictly less disutility than being fired by the Board, giving the CEO a rational incentive to exit voluntarily.

Safety bound: $W \leftarrow \max(W, 0.01)$, $W_{\text{ref}} \leftarrow \max(W_{\text{ref}}, 0.01)$.

Default parameters:

| Parameter                      | Symbol                          | Default |
| ------------------------------ | ------------------------------- | ------- |
| Risk aversion                  | $\gamma$                      | 1.5     |
| Loss aversion (monetary)       | $\lambda$                     | 2.25    |
| Loss aversion (non-monetary)   | $\lambda_D$                   | 2.25    |
| Reference wealth               | $W_{\text{ref}}$              | 16.0    |
| Resign wealth                  | $W_{\text{resign}}$           | 8.0     |
| Stay wealth (kept)             | $W_{\text{stay\_kept}}$       | 7.0     |
| Stay wealth (sacked)           | $W_{\text{stay\_sacked}}$     | 1.5     |
| Resign disutility              | $D_{\text{resign}}$           | 40.0    |
| Stay baseline disutility       | $D_{\text{stay}}$             | 25.0    |
| Sacked disutility              | $D_{\text{sacked}}$           | 100.0   |
| Resign-late disutility         | $D_{\text{resign\_late}}$     | 60.0    |
| Negotiate-exit disutility      | $D_{\text{negotiate}}$        | 45.0    |
| AGM humiliation                | $D_{\text{agm}}$              | 30.0    |
| Public disgrace                | $D_{\text{disgrace}}$         | 30.0    |
| Adverse review                 | $D_{\text{adverse\_review}}$  | 10.0    |

---

## 6. ARA predictive distributions (`predictive.py`)

### 6.1 General form

At an opponent decision node $X$ owned by player $j$, from focal actor $i$'s perspective:

$$
\boxed{
p_i(X = x \mid h) = \frac{1}{K} \sum_{k=1}^{K} \mathbb{1}\!\left[x \in \arg\max_{x' \in \mathcal{X}(h)} \Psi_j(x'; h, \Theta_j^{(k)})\right]
}
$$

where:

- $K = 200$ opponent parameter samples from $\Theta_j^{(k)} \sim p_i(\Theta_j)$
- $\Psi_j(x; h, \Theta_j)$ is opponent $j$'s expected utility for action $x$

### 6.2 Expected utility via stochastic rollout

$$
\Psi_j(x; h, \Theta_j) = \frac{1}{R} \sum_{r=1}^{R} u_j\!\left(Z^{(r)}; \Theta_j\right)
$$

where $R = 20$ and each $Z^{(r)}$ is a terminal outcome obtained by:

1. Forcing action $x$ at node $X$
2. Simulating forward to terminal using fixed policies for other actors
3. Sampling chance nodes from the focal actor's (possibly biased) model of Nature

### 6.3 Fixed policies (Level-1 fallbacks)

Used within rollouts for non-ARA actors:

**Board at $D_1$:**

$$
\pi_B^{\text{fix}}(D_1) = \text{D0\_minimal}
$$

**Board at $D_{\text{rev}}$ (and $D_{\text{rev}}'$):**

$$
\pi_B^{\text{fix}}(D_{\text{rev}} \mid h) = \begin{cases}
\text{Drev\_sack\_ceo} & V_\% \ge \tau_{\text{sack}} \;\wedge\; \text{CEO\_present} \\
\text{Drev\_commission\_review} & V_\% \ge \tau_{\text{review}} \;\wedge\; \text{review\_not\_commissioned} \\
\text{Drev\_no\_action} & \text{otherwise}
\end{cases}
$$

where $\tau_{\text{sack}}$ and $\tau_{\text{review}}$ are policy parameters loaded from governance_spec.xlsx (`policy_parameters` sheet). Defaults: $\tau_{\text{sack}} = 0.25$, $\tau_{\text{review}} = 0.25$.

**ASA at $A_2$ — informative Beta prior (headline-incident conditioning):**

Source: `ranked_voting_recommendations.csv`, restricted to `headline_incident = 1` (Qantas excluded). Pooled observed rate: 14/15 = 0.933.

ASA recommendation is near-automatic given a headline incident. The data provide no statistical evidence for a board-action effect at the recommendation stage; the three Beta distributions are nearly identical and separation is a modelling convention (monotonic decreasing by board accountability level):

| Board action level | Condition                              | Posterior               | Mean $p_{\text{strike}}$ |
| ------------------- | -------------------------------------- | ----------------------- | -------------------------- |
| 0 — Do nothing      | $D_1 = \text{D0\_minimal}$          | $\text{Beta}(46, 4)$  | 0.920                      |
| 1 — Review / CEO resigns | $D_1 = \text{D1\_review}$ or CEO resigned early | $\text{Beta}(44, 4)$ | 0.917 |
| 2 — Sack CEO        | $D_1 = \text{D3\_ceo\_transition}$  | $\text{Beta}(43, 4)$  | 0.914                      |

All 90% credible intervals are entirely above 0.84.

$$
p_{\text{strike}} \sim \text{Beta}(\alpha_d, \beta_d), \qquad A_2 = \begin{cases} \text{rec\_strike} & \text{w.p. } p_{\text{strike}} \\ \text{no\_strike} & \text{w.p. } 1 - p_{\text{strike}} \end{cases}
$$

**CEO at $D_0^{\text{ceo}}$:**

$$
\pi_C^{\text{fix}}(D_0^{\text{ceo}}) = \text{CEO\_stay}
$$

**CEO at $D_4$ (and $D_4'$):**

$$
\pi_C^{\text{fix}}(D_4 \mid h) = \begin{cases}
\text{D4\_resign} & V_\% \ge \tau_{\text{resign}} \;\wedge\; \text{feasible} \\
\text{D4\_stay} & \text{otherwise}
\end{cases}
$$

where $\tau_{\text{resign}} = 0.40$ (from `policy_parameters` sheet).

### 6.4 Level-2 recursion

When `level >= 2` and opponent $j$ has a strategic counterpart $j'$:

$$
p_i^{(L)}(X = x \mid h) \quad\text{where}\quad
\Psi_j^{(L)}(x; h, \Theta_j) \text{ uses } p_j^{(L-1)}(\cdot) \text{ for counterpart } j'
$$

The level decrements on each recursive call, preventing infinite recursion. Nested predictions use reduced $K_{\text{nested}}$ and $R_{\text{nested}}$ to avoid quadratic explosion.

---

## 7. Tree evaluation (`tree.py`)

### 7.1 Value function

Define $U_i^{(n)}(h)$ as focal actor $i$'s expected utility at node $n$ with history $h$:

$$
U_i^{(n)}(h) = \mathbb{E}_i\!\left[u_i(Z) \mid \text{node } n,\; h\right]
$$

### 7.2 Terminal node

$$
U_i^{(\text{Terminal})}(h) = u_i\!\left(Z(h, S)\right)
$$

### 7.3 Focal decision node — maximisation

If node $n$ is owned by focal actor $i$:

$$
U_i^{(n)}(h) = \max_{a \in \mathcal{A}_n(S)} U_i^{(\text{next}(n))}(h \cup \{n \mapsto a\})
$$

### 7.4 Opponent decision node — predictive expectation

If node $n$ is owned by opponent $j \ne i$:

**ARA model:**

$$
U_i^{(n)}(h) = \sum_{a \in \mathcal{A}_n(S)} p_i(n = a \mid h) \cdot U_i^{(\text{next}(n))}(h \cup \{n \mapsto a\})
$$

**Policy model:**

$$
U_i^{(n)}(h) = U_i^{(\text{next}(n))}(h \cup \{n \mapsto \pi_j^{\text{fix}}(n, h, \omega)\})
$$

where $\omega$ is a random seed. Board and CEO fixed policies are deterministic (the action depends only on $h$). The ASA fixed policy at $A_2$ is stochastic: a single action is drawn from a Beta-Binomial posterior and treated as deterministic for that evaluation (see Section 6.3).

### 7.5 Vote chance node — Monte Carlo integration

Three epistemic quantities are drawn once per belief draw and held fixed across the $M_V$ vote samples:

1. $f(D_1) \sim U(\ell, u)$ — governance effect
2. $V_{\text{floor}} \sim \text{Beta}(50, 150)$ — crisis floor (only if `headline_incident`)
3. $\kappa_\sigma$ — sigma scale (only if overconfidence bias is active)

Vote samples are drawn $M_V$ times (aleatoric, default $M_V = 50$):

$$
U_i^{(V)}(h) = \frac{1}{M_V} \sum_{m=1}^{M_V} U_i^{(\text{next}(V))}(h \cup \{V_\% \mapsto v_m^{\text{final}},\; \text{strike}_m,\; \text{overwhelming}_m\})
$$

where each $v_m^{\text{logit}} \sim \text{LogitNormal}(\alpha_V + B_{\text{agm}}, \sigma_V)$ with $B_{\text{agm}}$ conditioned on the pre-drawn governance effect, and:

$$
v_m^{\text{final}} = \begin{cases}
\max(v_m^{\text{logit}},\; V_{\text{floor}}) & \text{if headline\_incident} \\
v_m^{\text{logit}} & \text{otherwise}
\end{cases}
$$

When Board is focal with overconfidence bias:

- Governance effect uses biased bounds $(\ell^{\text{bias}}, u^{\text{bias}})$
- $\sigma_V$ is scaled by $\kappa_\sigma$

### 7.6 Review chance node — Monte Carlo integration

If review not commissioned: deterministic pass-through ($\text{CAR} = 0$, $\text{adverse} = \text{false}$, $C_{\text{direct}} = 0$).

If commissioned, two epistemic quantities are drawn once and held fixed across MC review samples:

1. $C_{\text{direct}} \sim \text{Gamma}(4.55, 4741)$ — review direct cost
2. $p_{\text{adverse}} \sim \text{Beta}(10, 5)$ — adverse probability (with bias if applicable, see §4.3)

Findings are sampled $M_R$ times (default $M_R = 20$):

$$
U_i^{(R)}(h) = \frac{1}{M_R} \sum_{m=1}^{M_R} U_i^{(\text{next}(R))}(h \cup \{\text{CAR}_m, \text{adverse}_m, C_{\text{direct}}\})
$$

where each $\text{CAR}_m \sim t_3(\mu_f, \sigma_f)$ with hierarchical parameters, and $\text{adverse}_m \sim \text{Bernoulli}(p_{\text{adverse}})$.

### 7.7 Pass-through nodes ($M_{\text{agm}}, M_{\text{rev}}$)

$$
U_i^{(M)}(h) = U_i^{(\text{next}(M))}(h)
$$

---

## 8. Full node-indexed recursion — Board-advice mode

Board is focal. Opponents: ASA (ARA or Policy), CEO (ARA or Policy).

### Node sequence (ceo\_stayed scenario, D4 before D\_rev)

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

**CEO responds to AGM outcome** (opponent — predictive or skip if CEO absent):

$$
U_B^{(D_4)}(h) = \begin{cases}
\displaystyle\sum_{d_4 \in \mathcal{D}_4(S)} p_B(D_4 = d_4 \mid h) \cdot U_B^{(D_{\text{rev}})}(h \cup \{D_4 = d_4\}) & \text{if CEO\_present} \\[6pt]
U_B^{(D_{\text{rev}})}(h) & \text{if CEO absent (skip)}
\end{cases}
$$

**Board responds** (focal — maximise, or limited actions if CEO absent):

$$
U_B^{(D_{\text{rev}})}(h) = \max_{d \in \mathcal{D}_{\text{rev}}(S)} U_B^{(R)}(h \cup \{D_{\text{rev}} = d\})
$$

**Review findings:**

$$
U_B^{(R)}(h) = \frac{1}{M_R} \sum_{m} U_B^{(\text{post-review})}(h \cup \{R_m, C_{\text{direct}}\})
$$

$M_{\text{rev}}$ is a pass-through.

**Post-review conditional round** (if adverse AND CEO present, another $D_4 \rightarrow D_{\text{rev}}$):

$$
U_B^{(\text{post-review})}(h) = \begin{cases}
U_B^{(D_4')}(h) & \text{if review\_adverse} \wedge \text{CEO\_present} \\
U_B^{(\text{Terminal})}(h) & \text{otherwise}
\end{cases}
$$

$$
U_B^{(D_4')}(h) = \sum_{d_4 \in \mathcal{D}_4(S)} p_B(D_4 = d_4 \mid h) \cdot U_B^{(D_{\text{rev}}')}(h \cup \{D_4 = d_4\})
$$

$$
U_B^{(D_{\text{rev}}')}(h) = \max_{d \in \mathcal{D}_{\text{rev}}(S)} U_B^{(\text{Terminal})}(h \cup \{D_{\text{rev}} = d\})
$$

$$
U_B^{(\text{Terminal})}(h) = u_B(Z(h, S))
$$

### Board optimal action

$$
d_1^* \in \arg\max_{d_1 \in \mathcal{D}_1} \frac{1}{N} \sum_{i=1}^{N} U_B^{(A_2)}(h_0 \cup \{D_1 = d_1\}; \theta^{(i)})
$$

### Predictive distributions needed

- $p_B(A_2 \mid h)$: ARA predictive over ASA strike recommendation
- $p_B(D_4 \mid h)$: ARA predictive over CEO response (post-AGM)
- $p_B(D_4' \mid h)$: ARA predictive over CEO response (post-review, same model)

---

## 9. Full node-indexed recursion — ASA-advice mode

ASA is focal. Opponents: Board (ARA or Policy), CEO (ARA or Policy).

### Node sequence (ceo\_stayed scenario, D4 before D\_rev)

$$
U_A^{(D_1)}(h_0) = \sum_{d_1 \in \mathcal{D}_1} p_A(D_1 = d_1 \mid h_0) \cdot U_A^{(A_2)}(h_0 \cup \{D_1 = d_1\})
$$

$$
U_A^{(A_2)}(h) = \max_{a \in \mathcal{A}_2} U_A^{(V)}(h \cup \{A_2 = a\})
$$

$$
U_A^{(V)}(h) = \frac{1}{M_V} \sum_{m} U_A^{(D_4)}(h \cup \{V_m\})
$$

**CEO responds to AGM outcome** (opponent — predictive or skip):

$$
U_A^{(D_4)}(h) = \begin{cases}
\displaystyle\sum_{d_4 \in \mathcal{D}_4(S)} p_A(D_4 = d_4 \mid h) \cdot U_A^{(D_{\text{rev}})}(h \cup \{D_4 = d_4\}) & \text{if CEO\_present} \\[6pt]
U_A^{(D_{\text{rev}})}(h) & \text{if CEO absent (skip)}
\end{cases}
$$

**Board responds** (opponent — predictive):

$$
U_A^{(D_{\text{rev}})}(h) = \sum_{d \in \mathcal{D}_{\text{rev}}(S)} p_A(D_{\text{rev}} = d \mid h) \cdot U_A^{(R)}(h \cup \{D_{\text{rev}} = d\})
$$

$$
U_A^{(R)}(h) = \frac{1}{M_R} \sum_{m} U_A^{(\text{post-review})}(h \cup \{R_m, C_{\text{direct}}\})
$$

**Post-review conditional round:**

$$
U_A^{(\text{post-review})}(h) = \begin{cases}
U_A^{(D_4')}(h) & \text{if review\_adverse} \wedge \text{CEO\_present} \\
U_A^{(\text{Terminal})}(h) & \text{otherwise}
\end{cases}
$$

$$
U_A^{(D_4')}(h) = \sum_{d_4 \in \mathcal{D}_4(S)} p_A(D_4 = d_4 \mid h) \cdot U_A^{(D_{\text{rev}}')}(h \cup \{D_4 = d_4\})
$$

$$
U_A^{(D_{\text{rev}}')}(h) = \sum_{d \in \mathcal{D}_{\text{rev}}(S)} p_A(D_{\text{rev}} = d \mid h) \cdot U_A^{(\text{Terminal})}(h \cup \{D_{\text{rev}} = d\})
$$

### ASA optimal action

$$
a_2^*(d_1) \in \arg\max_{a \in \mathcal{A}_2} \frac{1}{N} \sum_{i=1}^{N} U_A^{(V)}(h \cup \{A_2 = a\}; \theta^{(i)})
$$

ASA's recommendation is conditional on the observed $D_1$.

### Predictive distributions needed

- $p_A(D_1 \mid h)$: ARA predictive over Board initial governance reform
- $p_A(D_4 \mid h)$: ARA predictive over CEO response (post-AGM)
- $p_A(D_{\text{rev}} \mid h)$: ARA predictive over Board review response
- $p_A(D_4' \mid h)$: ARA predictive over CEO response (post-review)
- $p_A(D_{\text{rev}}' \mid h)$: ARA predictive over Board response (post-review)

### Reduced-form Board option

If Board uses fixed policy instead of ARA:

$$
p_A(D_1 \mid h) \;\longrightarrow\; \mathbb{1}[D_1 = \pi_B^{\text{fix}}(D_1, h)]
$$

All recursion equations remain identical; only the predictive distribution is replaced by a point mass.

---

## 10. Solver orchestration (`solver.py`)

### 10.1 Pipeline

For a given focal actor $i$, checkpoint $c$, scenario $s$:

1. Load $\text{BeliefBundle}(c)$, $\text{ParameterSampler}$, utility weights, policy parameters, overconfidence bias.
2. Construct engine: $\text{ChanceModels}$, $\text{PredictiveDistribution}(K, R)$, $\text{TreeEvaluator}(M_V, M_R)$.
3. Set initial state: $S_0 = \text{for\_scenario}(s)$.
4. For each initial action $a \in \mathcal{A}_{D_1}(S_0)$:
   - For each draw $i = 1, \ldots, N$:
     - $v_{a,i} = U_i^{(\text{next}(D_1))}(\{D_1 = a\}; \theta^{(i)})$
   - $\text{EU}(a) = \frac{1}{N} \sum_i v_{a,i}$
5. $a^* = \arg\max_a \text{EU}(a)$

### 10.2 D0\_ceo Bayesian prediction

The solver predicts the pre-game CEO resignation probability by combining a Beta prior with ARA evidence:

**Prior:** Based on 12 Australian no-remorse CEO observations (all resigned), using a Jeffreys starting prior:

$$
p_{\text{resign}} \sim \text{Beta}(\alpha_0, \beta_0) = \text{Beta}(12.5, 0.5), \quad \mathbb{E} = 0.962
$$

**ARA evidence:** For each belief draw $i = 1, \ldots, N$, a Level-2 ARA predictive distribution is computed over $D_0^{\text{ceo}}$, yielding soft pseudo-counts:

$$
n_{\text{resign}} = \sum_{i=1}^{N} p_i(D_0^{\text{ceo}} = \text{resign} \mid h), \qquad n_{\text{stay}} = \sum_{i=1}^{N} p_i(D_0^{\text{ceo}} = \text{stay} \mid h)
$$

**Bayesian update:**

$$
\Pr(\text{CEO\_resign}) = \frac{\alpha_0 + n_{\text{resign}}}{\alpha_0 + \beta_0 + N}
$$

The D0\_ceo prediction uses reduced computational budget ($K_{\text{d0}} = 50$, $R_{\text{d0}} = 10$) and Level-2 mode (CEO strategically models Board's $D_1$ response).

### 10.3 Scenario solver

`solve_scenarios()` runs both scenarios and attaches D0\_ceo predictive:

1. $\Pr_i(D_0^{\text{ceo}}) = \text{predict\_d0\_ceo}(i, c)$
2. For each $s \in \{\text{ceo\_stayed}, \text{ceo\_resigned}\}$:
   - $\text{result}_s = \text{solve}(i, c, s)$
   - $\text{result}_s.\text{scenario\_prob} = \Pr_i(D_0^{\text{ceo}} = \text{action}(s))$

### 10.4 Parallelisation

Each $(a, i)$ pair is submitted as an independent task to `ProcessPoolExecutor`. Workers are initialised once per checkpoint with all engine components (beliefs, tree, predictive) to eliminate per-task file I/O. A persistent process pool is reused across predict\_d0\_ceo and solve calls for the same checkpoint.

---

## 11. Mode configurations (`modes.py`)

| Mode               | Focal | ASA model  | Board model | CEO model | Level |
| ------------------ | ----- | ---------- | ----------- | --------- | ----- |
| Board Mode         | Board | ARA        | — (focal)  | ARA       | 1     |
| ASA Mode           | ASA   | — (focal) | ARA         | ARA       | 1     |
| Board L2           | Board | ARA        | — (focal)  | ARA       | 2     |
| ASA L2             | ASA   | — (focal) | ARA         | ARA       | 2     |
| Board (ASA=Policy) | Board | Policy     | — (focal)  | ARA       | 1     |
| ASA (Board=Policy) | ASA   | — (focal) | Policy      | ARA       | 1     |

Level-2 strategic counterparts:

- Board L2: ASA models Board, CEO models Board
- ASA L2: Board models ASA, CEO models Board

---

## 12. Summary of the recursion pattern

At every node:

| Node type                                   | Owned by          | Operation                                    |
| ------------------------------------------- | ----------------- | -------------------------------------------- |
| Decision                                    | Focal actor       | $\max$ over feasible actions               |
| Decision                                    | Opponent (ARA)    | $\sum$ weighted by predictive distribution |
| Decision                                    | Opponent (Policy) | Deterministic fixed policy                   |
| Chance ($V$)                              | Nature            | MC average over $M_V$ vote samples          |
| Chance ($R$)                              | Nature            | MC average over $M_R$ review samples        |
| Chance ($M_{\text{agm}}, M_{\text{rev}}$) | Nature            | Pass-through                                 |
| Terminal                                    | —                | Compute $u_i(Z)$                            |

---

## 13. Computational budget

| Parameter                  | Symbol           | Default                | Purpose                                    |
| -------------------------- | ---------------- | ---------------------- | ------------------------------------------ |
| Belief draws               | $N$            | 500 (or `--n_draws`) | Posterior samples per checkpoint           |
| Opponent parameter samples | $K$            | 200                    | Draws from $p_i(\Theta_j)$ per predictive |
| Stochastic rollouts        | $R$            | 20                     | Forward simulations per $(K, a)$ pair     |
| Vote MC samples            | $M_V$          | 50                     | Samples per vote node evaluation           |
| Review MC samples          | $M_R$          | 20                     | Samples per review node evaluation         |
| D0\_ceo opponent samples   | $K_{\text{d0}}$ | 50                     | Reduced $K$ for D0\_ceo prediction        |
| D0\_ceo rollouts           | $R_{\text{d0}}$ | 10                     | Reduced $R$ for D0\_ceo prediction        |

Total tree evaluations per solve: $|\mathcal{A}_{D_1}| \times N$.
Total rollouts per predictive call: $K \times |\mathcal{A}| \times R$.
