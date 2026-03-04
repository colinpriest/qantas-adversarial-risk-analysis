# Shareholder Vote Model — Design Document

This document specifies the Bayesian distribution, parameters, and design rationale for the shareholder strike vote at the AGM (node $V$ in the game tree). The vote model is the central chance node in the Qantas ARA engine: it converts upstream decisions (Board governance reform, ASA strike recommendation) and latent belief states into a continuous vote percentage, from which discrete thresholds (first strike at 25%, overwhelming at 50%) are derived.

---

## 1. Distribution Specification

### 1.1 The logit-normal model

The vote fraction $V \in (0, 1)$ follows a **logit-normal distribution**:

$$
\text{logit}(V) \sim \mathcal{N}\!\bigl(\mu_V,\; \sigma_V^{(i)}\bigr)
$$

$$
V = \text{expit}\!\bigl(\text{logit}(V)\bigr) = \frac{1}{1 + e^{-\text{logit}(V)}}
$$

where the location parameter $\mu_V$ is composed of three additive terms:

$$
\mu_V = \alpha_V^{(i)} + B_{\text{agm}}^{(i)}
$$

$$
B_{\text{agm}} = B_{\text{mkt}}^{(i)} + \gamma_A^{(i)} \cdot \mathbb{1}[A_2 = \text{rec\_strike}] + \gamma_D^{(i)} \cdot f(D_1)
$$

The superscript $(i)$ denotes posterior draw $i$ from the upstream belief checkpoint.

### 1.2 Parameter definitions

| Symbol | Name | Source | Description |
|--------|------|--------|-------------|
| $\alpha_V^{(i)}$ | `alpha_vote` | Stan posterior | Baseline opposition intercept on the logit scale |
| $B_{\text{mkt}}^{(i)}$ | `B_mkt` | Stan posterior | Latent market belief at the checkpoint date |
| $\gamma_A^{(i)}$ | `gamma_A` | Stan posterior + data-driven prior | Additive shift from ASA public mobilisation |
| $\gamma_D^{(i)}$ | `gamma_D` | Stan posterior | Scaling coefficient for governance action effect |
| $\sigma_V^{(i)}$ | `sigma_vote` | Stan posterior | Logit-scale noise (aleatoric uncertainty in voter behaviour) |
| $f(D_1)$ | Governance effect | Sampled at runtime | Effectiveness of Board's governance reform |

### 1.3 Governance effect (non-monotonic)

The governance effect $f(D_1)$ is drawn from a Uniform distribution whose sign and bounds depend on the Board's action:

$$
f(D_1) = \begin{cases}
0 & D_1 = \text{D0\_minimal} \\
U(0, 1) & D_1 = \text{D1\_review} \\
U(-1, 0) & D_1 = \text{D3\_ceo\_transition}
\end{cases}
$$

When multiplied by $\gamma_D$ (negative from the posterior), the effects are:

- **D1 (review):** positive $f$ $\times$ negative $\gamma_D$ $\to$ lowers $B_{\text{agm}}$ $\to$ **reduces** protest
- **D3 (CEO exit):** negative $f$ $\times$ negative $\gamma_D$ $\to$ raises $B_{\text{agm}}$ $\to$ **increases** protest
- **D0 (no action):** no governance signal; baseline applies

### 1.4 Derived indicators

Two binary thresholds are applied to the sampled $V$ (loaded from the `vote_thresholds` sheet of `governance_spec.xlsx`):

$$
\text{strike} = \mathbb{1}[V \ge 0.25], \qquad \text{overwhelming} = \mathbb{1}[V \ge 0.50]
$$

A **first strike** (25%) triggers the "two strikes" rule under the Corporations Act 2001 s.250U–250V, and an **overwhelming** vote (50%) signals a governance crisis severe enough to threaten a board spill.

---

## 2. Parameter Estimation Pipeline

The vote model parameters are not hard-coded. They flow through a three-stage estimation pipeline, from raw data to runtime draws.

### Stage 1: Upstream Stan models

Two Stan state-space models produce the foundational estimates:

**Media measurement model** (`models/media_better.stan`): Separates latent media *coverage* from *intensity* using sparse monthly observations. An AR(1) process on log-intensity evolves monthly:

$$
\log M_t = \mu_{\log M} + \phi(\log M_{t-1} - \mu_{\log M}) + \sigma_{\log M} \cdot z_t
$$

The log-differences $\Delta \log M_t = \log M_t - \log M_{t-1}$ become the media shock series that enters the belief model.

**Belief dynamics model** (`models/belief_model.stan`): Tracks a latent shareholder distrust state $B_t$ via AR(1) dynamics driven by media shocks, identified through three observation channels:

$$
B_t = \rho \cdot B_{t-1} + \beta \cdot \text{shock}_t + \sigma_B \cdot z_t
$$

Observation likelihoods:

| Channel | Likelihood | Anchoring |
|---------|-----------|-----------|
| Abnormal returns | $\text{abret}_t \sim \mathcal{N}(\lambda_r \cdot B_t, 1)$ | Free loading |
| Remuneration vote | $\text{logit}(y_{\text{rem}}) \sim \mathcal{N}(\alpha_{\text{rem}} + B_t, \sigma_{\text{rem}})$ | $\lambda_{\text{rem}} = 1$ (fixed) |
| Chair vote | $\text{logit}(y_{\text{chair}}) \sim \mathcal{N}(\alpha_{\text{chair}} + \lambda_{\text{chair}} \cdot B_t, \sigma_{\text{chair}})$ | Free loading |

The critical design choice is **anchoring** $\lambda_{\text{rem}} = 1$, which pins the belief state to the logit scale of remuneration-report opposition. This makes $B_t$ directly interpretable: a value of $B_t = 0$ means opposition is at baseline $\alpha_{\text{rem}}$, and each unit increase shifts logit(opposition) by 1. This anchoring propagates into the game tree — the vote model operates natively on the same scale as the belief state.

**Prior for $\alpha_{\text{rem}}$:** $\mathcal{N}(\text{logit}(0.10), 1.0)$. Centred at 10% baseline opposition (logit $\approx -2.2$), reflecting that most ASX companies see single-digit remuneration opposition in normal times. The 1.0 SD on the logit scale allows substantial departure.

**Prior for $\sigma_{\text{rem}}$:** half-$t(4, 0, 0.6)$. Permits moderate vote-to-belief observation noise.

### Stage 2: Data-driven prior for $\gamma_A$

The ASA mobilisation effect $\gamma_A$ is estimated from a cross-company panel of 36 voting recommendations (`data/voting-recommendations.csv`, `data/ranked_voting_recommendations.csv`), using two complementary channels:

**Vote channel (OLS on logit scale):**

$$
\text{logit}(\text{rem\_against\_pct}_i) = b_0 + b_1 \cdot \text{asa\_against}_i + b_2 \cdot \text{prior\_year\_pct}_i + b_3 \cdot \log(\text{mkt\_cap}_i) + \text{GICS FE} + \epsilon_i
$$

The coefficient $b_1$ captures the additive logit-scale shift from an ASA "against" recommendation, controlling for firm size, prior-year opposition, and industry. Prior derived: $\gamma_A^{\text{vote}} \sim \mathcal{N}(b_1, \sqrt{\text{SE}_1^2 + (0.35 \cdot \sigma_{\text{resid}})^2})$. The extra variance term inflates the standard error to account for cross-company heterogeneity.

**Strike channel (Bayesian logistic regression):**

$$
\text{first\_strike}_i \sim \text{Bernoulli}\!\bigl(\text{logistic}(X_i \beta)\bigr), \quad \beta \sim \mathcal{N}(0, 2.5^2 I)
$$

The $\mathcal{N}(0, 2.5)$ prior follows Gelman et al.'s recommendation for weakly informative priors on logistic regression coefficients, preventing separation-induced infinite estimates while remaining uninformative for plausible effect sizes.

**Combined prior (inverse-variance weighting):** The two channels are pooled into a single Normal prior:

$$
\gamma_A \sim \mathcal{N}\!\left(\frac{w_{\text{vote}} \mu_{\text{vote}} + w_{\text{strike}} \mu_{\text{strike}}}{w_{\text{vote}} + w_{\text{strike}}},\; \frac{1}{\sqrt{w_{\text{vote}} + w_{\text{strike}}}}\right)
$$

where $w = 1/\sigma^2$. This is the minimum-variance unbiased combination of two independent Normal estimates of the same quantity.

**Truncation:** The combined prior is truncated below at zero, encoding the domain constraint that public ASA mobilisation cannot *reduce* shareholder pressure.

### Stage 3: Checkpoint construction

The posterior draws from the Stan model are combined with shock priors to produce belief checkpoints at five critical dates:

| Checkpoint | Date | Context | Market belief | Management belief |
|---|---|---|---|---|
| $C_{\text{pre}}$ | Pre-crisis | Baseline | $B_{\text{base}}$ | $B_{\text{base}}$ |
| $C_0$ | 2023-10-01 | Post-CEO resignation | $B_{\text{sep}}$ | $B_{\text{sep}} + \gamma_E$ |
| $C_1$ | 2023-10-10 | Review announced | $B_{\text{sep}} + \gamma_{\text{review}}$ | $B_{\text{sep}} + \gamma_E + \gamma_{\text{review}}$ |
| $C_2$ | 2023-10-18 | ASA mobilisation public | $B_{\text{sep}} + \gamma_{\text{review}} + \gamma_A$ | $B_{\text{sep}} + \gamma_E + \gamma_{\text{review}} + \gamma_A$ |
| $C_3$ | 2023-11-03 | Post-AGM (82.9% observed) | Importance-resampled | Aligned |

Each checkpoint contains 500 draws of all parameters $(\alpha_V, B_{\text{mkt}}, B_{\text{mgmt}}, \gamma_A, \gamma_D, \sigma_V)$ serialised in `.npz` files.

The $C_3$ update uses likelihood-weighted importance resampling against the observed 82.9% vote:

$$
w_i \propto \mathcal{N}\!\bigl(\text{logit}(0.829) \;\big|\; \alpha_{\text{rem}}^{(i)} + B_{C_2}^{(i)},\; \sigma_{\text{rem}}^{(i)}\bigr)
$$

This Bayesian update concentrates the posterior on draws consistent with the extreme observed outcome, sharpening beliefs for post-AGM decision nodes.

---

## 3. Empirical Calibration of Governance Effects

### 3.1 The non-monotonic pattern

The governance effect $f(D_1)$ encodes an empirically observed **non-monotonic** relationship between Board actions and shareholder protest. The calibration data is a cross-company panel of ASX remuneration votes (`data/ranked_voting_recommendations.csv`, 36 company-years from 2021–2023), classified by Board corrective action:

| Board action | $n$ | Mean $\Delta\text{vote}$ | Interpretation |
|---|---|---|---|
| D0 — No action | 18 | +32.8 pp | Baseline escalation: passive board amplifies discontent |
| D1 — Review/announcement | 8 | +14.9 pp | Most effective mitigation: review signals responsiveness |
| D3 — CEO exit | 4 | +46.7 pp | Crisis signal: sacking amplifies protest beyond baseline |

This ranking — **D1 mitigates, D0 is baseline, D3 escalates** — is counterintuitive but robust. Forcing CEO departure signals to the market that the governance problem was more severe than previously believed, triggering a protest amplification effect.

### 3.2 Encoding as scaled Uniform draws

The governance effects are modelled as scaled random variables rather than point estimates, reflecting genuine uncertainty about the magnitude:

- **D1 (review):** $f \sim U(0, 1)$, expected value 0.5. A review *always* reduces protest relative to baseline (positive values, multiplied by negative $\gamma_D$), but the magnitude is uncertain — some reviews are more credible than others.
- **D3 (CEO exit):** $f \sim U(-1, 0)$, expected value $-0.5$. A CEO exit *always* amplifies protest (negative values, multiplied by negative $\gamma_D$), but the severity of the amplification is uncertain.
- **D0 (no action):** $f = 0$ deterministically. No governance signal means no effect on $B_{\text{agm}}$ beyond what $B_{\text{mkt}}$ already captures.

The $\gamma_D$ coefficient (estimated from the Stan posterior, typically negative) controls the overall *scale* of governance effects on the logit of the vote, while $f(D_1)$ controls the *direction and relative magnitude*.

### 3.3 Epistemic vs. aleatoric separation

A critical design choice: the governance effect $f(D_1)$ is drawn **once per scenario per belief draw** and held fixed across Monte Carlo vote samples. This reflects the distinction:

- **Epistemic uncertainty** (drawn once): "How effective *is* this particular governance reform?" This is a property of the reform itself — it does not change between individual voter decisions.
- **Aleatoric uncertainty** (drawn $M_V$ times): "Given the true effectiveness, what fraction of shareholders actually vote against?" This is irreducible noise from aggregating millions of individual voting decisions.

In the tree evaluation, 50 vote samples (aleatoric) are drawn per governance-effect draw (epistemic), and the Monte Carlo average approximates:

$$
\mathbb{E}[U \mid f] = \frac{1}{M_V} \sum_{m=1}^{M_V} U\!\bigl(\text{expit}(\alpha_V + B_{\text{agm}}(f) + \sigma_V \cdot \epsilon_m)\bigr)
$$

---

## 4. Design Justification

### 4.1 Why logit-normal?

The logit-normal distribution is the natural choice for modelling a proportion (vote fraction) that arises from aggregating many individual binary decisions. Four properties make it ideal here:

1. **Bounded support.** $V \in (0, 1)$ by construction. A Normal model on the raw vote percentage would assign positive probability to votes below 0% or above 100%.

2. **Additive effects on the logit scale.** The belief state $B_t$, ASA mobilisation $\gamma_A$, and governance effects $\gamma_D \cdot f$ are all additive on the logit scale. This is natural for latent-variable models where multiple factors contribute to a probability — the logit link linearises interaction effects.

3. **Consistency with the Stan identification.** The belief model anchors $\lambda_{\text{rem}} = 1$, meaning the belief state $B_t$ lives on the logit scale of remuneration opposition. Using a logit-normal vote model means the game tree operates on *exactly the same scale* as the upstream Bayesian estimation — no awkward scale transformations are needed.

4. **Flexible shape.** Unlike the Beta distribution (the other standard bounded-support model), the logit-normal can be parameterised directly through regression-style predictors. The logit-normal also handles extreme probabilities (near 0 or 1) gracefully — the long tails on the logit scale produce appropriate concentration near the boundaries on the probability scale.

**Why not Beta regression?** A Beta regression model could also work, but it would require re-parameterising the belief state and effect coefficients onto a different scale. The logit-normal preserves the additive structure that comes naturally from the Stan state-space model.

### 4.2 Why separate $\gamma_A$ and $\gamma_D$?

The model distinguishes between ASA mobilisation ($\gamma_A$) and governance action ($\gamma_D$) effects for three reasons:

1. **Different causal mechanisms.** ASA mobilisation operates through *information transmission* — retail shareholders who would not otherwise attend to the vote learn about governance concerns and are prompted to vote against. Governance actions operate through *belief revision* — shareholders update their assessment of the severity of the underlying problem.

2. **Different estimation bases.** $\gamma_A$ is estimated from a cross-company panel regression (Section 2, Stage 2), exploiting within-company variation in ASA recommendations. $\gamma_D$ is estimated jointly with the belief state in the Stan model, using the time-series structure of vote outcomes.

3. **Non-monotonic structure.** The governance effect is non-monotonic (D1 mitigates, D3 amplifies), which requires a multiplicative decomposition $\gamma_D \cdot f(D_1)$ where $\gamma_D$ sets the scale and $f$ sets the sign and relative magnitude. The ASA effect is unidirectional (mobilisation always increases opposition) and is purely additive.

### 4.3 Why not estimate $\gamma_D$ from the cross-company panel?

The cross-company panel (`ranked_voting_recommendations.csv`) provides descriptive statistics on governance effects (Section 3.1), but these are *not* used directly as parameter estimates for $\gamma_D$. Instead, $\gamma_D$ is estimated within the Stan state-space model. The reasons:

1. **Endogeneity.** In the cross-company data, boards that face worse crises are more likely to take aggressive action (D3). The raw correlation between D3 and high votes overstates the causal effect of CEO exit on protest, because it confounds the board's response with the severity of the underlying crisis. The Stan model partially addresses this by conditioning on the latent belief state.

2. **Scale consistency.** The Stan model produces $\gamma_D$ on exactly the logit scale needed for the vote model, jointly estimated with $B_t$, $\alpha_V$, and $\sigma_V$. Using a separately estimated coefficient would break the joint posterior structure.

3. **Qualitative constraint only.** The cross-company data is used *qualitatively* — it establishes the non-monotonic sign pattern (D1 positive, D3 negative) and provides a plausibility check on magnitudes. The actual parameter value comes from the Stan posterior.

### 4.4 Why Uniform for governance effects?

The Uniform distribution $f(D_1) \sim U(0, 1)$ or $U(-1, 0)$ reflects maximum ignorance about the *magnitude* of the effect, conditional on its *direction*:

- **Direction is known.** The empirical evidence (Section 3.1) clearly establishes that reviews reduce protest and CEO exits amplify it. The Uniform bounds encode this directional knowledge.

- **Magnitude is genuinely uncertain.** The effectiveness of a governance review depends on its scope, independence, and market credibility — factors that vary enormously across cases. The Star Casino review (leading to $-13.95\%$ CAR) had very different impact from the Qantas review (+0.85% CAR). A Uniform over the full unit interval captures this uncertainty without imposing a preferred magnitude.

- **Interaction with $\gamma_D$.** The overall vote effect is $\gamma_D \cdot f$, where $\gamma_D$ is informed by the posterior. The Uniform on $f$ means the magnitude uncertainty is "resolved" at runtime — each simulation draw commits to a specific effectiveness level, which then interacts with the posterior estimate of how sensitive the vote is to governance signals.

### 4.5 Why five checkpoints?

The checkpoint structure reflects the **information asymmetry** between market participants and Qantas management at different points in the crisis:

| Checkpoint | Information state | Key asymmetry |
|---|---|---|
| $C_{\text{pre}}$ | Pre-crisis baseline | None — common beliefs |
| $C_0$ | CEO resignation announced (05-Sep) | Management knows private ASA engagement ($\gamma_E$); market does not |
| $C_1$ | Review announced (10-Oct) | Common knowledge of review; management still has private $\gamma_E$ signal |
| $C_2$ | ASA goes public (18-Oct) | $\gamma_A$ becomes common knowledge; information gap narrows |
| $C_3$ | AGM observed (03-Nov) | Common knowledge of 82.9% vote; posterior sharpened via importance resampling |

This structure allows the ARA engine to evaluate Board and ASA decisions at each information state, showing how the optimal action changes as beliefs evolve and information asymmetries resolve.

---

## 5. Board Overconfidence Bias on the Vote

When the Board is the focal actor, cognitive biases distort its perception of the vote distribution. These are calibrated from the governance overconfidence literature (Twardawski & Kind 2023, Brahma et al. 2023, Coffeng et al. 2021, Boundy-Singer et al. 2022).

### 5.1 Overestimation of governance effectiveness

The Board overestimates how much its actions reduce protest by a factor $\beta \sim U(0.25, 1.0)$ (production midpoint $\beta = 0.625$):

$$
f^{\text{bias}}(D_1 = \text{review}) \sim U(0.63, 1.0) \quad\text{vs. unbiased } U(0, 1)
$$
$$
f^{\text{bias}}(D_1 = \text{CEO exit}) \sim U(-0.62, 0.0) \quad\text{vs. unbiased } U(-1, 0)
$$

The biased Board believes a review is at least 63% effective (floor raised from 0) and that CEO exit backlash is at most 62% of the true worst case (floor raised from $-1.0$ to $-0.62$).

### 5.2 Overprecision on vote uncertainty

The Board perceives $\kappa \sim U(2, 5)$ times more precision than warranted:

$$
\hat{\sigma}_V = \sigma_V / \sqrt{\kappa}
$$

Production default: $\kappa = 3.5$, giving $\hat{\sigma}_V = 0.53 \cdot \sigma_V$. The Board's subjective confidence intervals are roughly half the width of the true intervals, leading it to underestimate the probability of extreme vote outcomes.

### 5.3 Bias propagation

Both biases are applied consistently:

- In the focal actor's own expected utility calculation (biased $f$ and biased $\sigma_V$ flow into the vote model)
- In rollout simulations for predictive distributions (opponent models see the biased Board acting on its biased beliefs)

This produces self-consistent decision-making under cognitive bias: the Board genuinely believes its optimistic assessment, and opponents model the Board as believing it.

---

## 6. How the Vote Feeds Into Utility Functions

The vote outcome $V$ (and derived indicators) enters all three actors' utility functions, creating the strategic tension that drives the game.

### 6.1 Board utility

The Board is penalised for high opposition through four channels:

$$
u_B \ni -w_{\text{vote}} \cdot (V - 0.25)_+^2 - w_{\text{over}} \cdot \mathbb{1}[\text{overwhelming}] - w_{\text{spill}} \cdot V \cdot \mathbb{1}[\text{strike}] - w_{\text{second}} \cdot \mathbb{1}[\text{strike} \wedge \text{CEO\_present}]
$$

| Component | Weight | Mechanism |
|---|---|---|
| Quadratic vote penalty | $w_{\text{vote}} = 2.0$ | Convex cost: marginal damage increases with opposition |
| Overwhelming penalty | $w_{\text{over}} = 3.0$ | Fixed cost at $V \ge 50\%$: governance crisis threshold |
| Spill risk | $w_{\text{spill}} = 2.5$ | Proportional to $V$ once strike threshold crossed |
| Second-strike spill | $w_{\text{second}} = 8.0$ | CEO still present after first strike $\to$ near-certain spill next year |

The quadratic form for vote penalty reflects that a 30% vote is mildly embarrassing, a 50% vote is a crisis, and an 80% vote is catastrophic — the marginal damage accelerates.

### 6.2 ASA utility

ASA benefits from high opposition through complementary channels:

$$
u_A \ni w_{\text{vote}}^A \cdot V + w_{\text{over}}^A \cdot \mathbb{1}[\text{overwhelming}] + w_{\text{rep}}^A \cdot (V - 0.25)_+ + w_{\text{align}}^A \cdot \mathbb{1}[\text{rec\_strike} \wedge \text{strike}]
$$

| Component | Weight | Mechanism |
|---|---|---|
| Vote reward | $w_{\text{vote}}^A = 2.0$ | Linear benefit from opposition: every percentage point helps |
| Overwhelming bonus | $w_{\text{over}}^A = 2.0$ | Threshold reward for governance crisis signal |
| Reputational gain | $w_{\text{rep}}^A = 1.0$ | Bonus for high-profile outcome above 25% |
| Market alignment | $w_{\text{align}}^A = 1.5$ | Credibility bonus when ASA recommendation matches market action |

The market alignment bonus is empirically motivated: in 100% of headline-incident cases in the panel data, the market votes a first strike. ASA's institutional credibility depends on leading, not lagging, this consensus.

### 6.3 CEO utility

The vote creates non-monetary penalties that scale with loss aversion ($\lambda_D = 2.25$):

$$
D_{\text{raw}} \ni D_{\text{agm}} \cdot \mathbb{1}[V > 0.25] + D_{\text{disgrace}} \cdot \mathbb{1}[\text{overwhelming}]
$$

| Component | Raw value | Loss-aversion-scaled |
|---|---|---|
| AGM humiliation | $D_{\text{agm}} = 30$ | $\lambda_D \cdot 30 = 67.5$ |
| Public disgrace | $D_{\text{disgrace}} = 30$ | $\lambda_D \cdot 30 = 67.5$ |

Under reference-dependent CRRA with loss aversion, the CEO evaluates these penalties relative to their expected status as a powerful executive. The combination creates a strong incentive to resign before the AGM if a high vote is anticipated.

---

## 7. Monte Carlo Integration

### 7.1 Runtime procedure

At the $V$ node in the game tree, the `TreeEvaluator` performs Monte Carlo integration:

1. **Draw governance effect once** (epistemic): $f \sim U(\ell, u)$ based on the D1 action and any overconfidence bias.
2. **Draw $M_V = 50$ vote samples** (aleatoric): For each sample $m$:
   - Construct $B_{\text{agm}}$ from the belief draw and action history
   - Draw $\text{logit}(V_m) \sim \mathcal{N}(\alpha_V + B_{\text{agm}}, \sigma_V)$
   - Apply expit to get $V_m \in (0, 1)$
   - Derive strike and overwhelming indicators
3. **Average downstream utility** across all 50 samples:

$$
U_i^{(V)}(h) = \frac{1}{50} \sum_{m=1}^{50} U_i^{(\text{next})}(h \cup \{V_m, \text{strike}_m, \text{overwhelming}_m\})
$$

### 7.2 Computational budget

Each vote node evaluation requires 50 forward recursions through the remaining game tree. Across the full solver:

- **Per belief draw:** 3 initial actions $\times$ 50 vote samples = 150 tree evaluations at the vote node
- **Per checkpoint:** $N$ belief draws $\times$ 150 = typically 75,000 vote-node evaluations
- **Per predictive distribution call:** $K = 200$ opponent samples $\times$ $R = 20$ rollouts $\times$ 50 vote samples = 200,000 vote draws

The logit-normal sampling is computationally trivial (one Normal draw + one expit transform per sample). The cost is dominated by the downstream tree recursion, not the vote sampling itself.

---

## 8. Numerical Implementation Details

### 8.1 Numerically stable expit

The inverse-logit function uses a branch-stable implementation to avoid overflow:

```python
def _expit(x):
    if x >= 0:
        z = exp(-x)
        return 1 / (1 + z)
    else:
        z = exp(x)
        return z / (1 + z)
```

For large positive $x$, $e^{-x} \to 0$ and the result approaches 1. For large negative $x$, $e^x \to 0$ and the result approaches 0. Neither branch computes $e^{|x|}$ for large $|x|$, avoiding float64 overflow.

### 8.2 Sigma floor

The logit-scale noise has a floor: $\sigma_V \leftarrow \max(\sigma_V, 10^{-6})$. This prevents degenerate zero-variance draws from the posterior from producing division-by-zero or deterministic vote outcomes.

### 8.3 Vote threshold precision

The thresholds (0.25 and 0.50) are loaded from Excel, not hard-coded. This allows sensitivity analysis over alternative threshold definitions (e.g., "what if the first-strike rule used 20%?") without code changes.

---

## 9. Validation

### 9.1 Empirical calibration check

At checkpoint $C_0$ (the primary analysis point), the vote model produces distributions consistent with the observed outcome:

- **Observed:** 82.9% against the remuneration report at the Qantas 2023 AGM
- **Model (C0, ASA rec strike, D1 review, median draw):** The logit-normal with posterior parameters places substantial mass above 50%, with the 82.9% outcome falling within the central 90% of the distribution

The model was *not* fitted to the Qantas vote (Qantas is excluded from the cross-company panel for $\gamma_A$ estimation). The consistency between the model's predictive distribution and the observed outcome provides an out-of-sample validation.

### 9.2 Test coverage

The vote model has 8 dedicated unit tests in `tests/test_engine.py`:

| Test | Validates |
|---|---|
| `test_vote_model_basic` | Output range $V \in (0, 1)$ and boolean indicators |
| `test_vote_increases_with_strike_recommendation` | $\gamma_A > 0$ shifts vote upward |
| `test_vote_non_monotonic_governance_effect` | D1 reduces and D3 increases protest |
| `test_governance_effect_respects_bias` | Biased D1 bounds shift correctly |
| `test_governance_effect_d3_bias` | Biased D3 bounds shift correctly |
| `test_d0_unaffected_by_bias` | D0 always returns 0 regardless of bias |
| `test_sigma_scale_reduces_vote_variance` | Overprecision reduces vote spread |
| `test_sigma_scale_1_is_identity` | No-bias case is identity |

---

## 10. References

**Statistical methodology:**
- Gelman, A., Jakulin, A., Pittau, M. G., & Su, Y.-S. (2008). A weakly informative default prior distribution for logistic and other regression models. *Annals of Applied Statistics*, 2(4), 1360–1383. [Bayesian logistic regression priors]
- Aitchison, J. & Shen, S. M. (1980). Logistic-normal distributions. *Biometrika*, 67(2), 261–272. [Logit-normal distribution properties]

**Overconfidence calibration:**
- Twardawski, T. & Kind, A. (2023). Board overconfidence in mergers and acquisitions. *Journal of Business Research*. [Board governance overestimation]
- Coffeng, T. et al. (2021). Board decision-making quality. *Corporate Governance*. [20% boards choose best option]
- Boundy-Singer, Z. et al. (2022). Metacognitive miscalibration in organizational leaders. *Organizational Behavior and Human Decision Processes*. [Overprecision estimates]

**Australian governance:**
- Corporations Act 2001 (Cth) ss.250U–250V. [Two-strikes rule]
- Ertimur, Y., Ferri, F., & Stubben, S. (2011). Board of directors' responsiveness to shareholders. *Journal of Corporate Finance*. [Shareholder voting corrections]
- Tversky, A. & Kahneman, D. (1992). Advances in prospect theory. *Journal of Risk and Uncertainty*, 5(4), 297–323. [Loss aversion $\lambda = 2.25$]

**Data sources:**
- `data/ranked_voting_recommendations.csv` — 36 ASX company-years (2021–2023) of remuneration vote outcomes and board actions
- `data/voting-recommendations.csv` — Extended cross-company panel with ASA recommendations and proxy adviser positions
- `data/checkpoints/belief_C*.npz` — Posterior draws from Stan estimation pipeline (500 draws per checkpoint)
- `data/governance_spec.xlsx` — Vote thresholds, utility weights, and game tree parameters
