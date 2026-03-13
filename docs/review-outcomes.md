# External Review Outcomes: Data Sources and Stochastic Model Derivation

This document explains the data sources, Bayesian reasoning, and distributional assumptions behind the stochastic model for external governance review outcomes used at the R (Review) chance node in the ARA game tree. The model is implemented in `engine/chance_models.py` as `ReviewModel` and `ReviewDirectCostModel`.

## 1. Overview

When the Board commissions an external governance review (at node D_rev or D1_review), the game tree encounters a chance node R whose outcome determines:

1. **Qualitative outcome rating** — negative, balanced, or positive — which triggers downstream feasibility rules (post-review round activation) and utility penalties.
2. **Cumulative abnormal return (CAR)** — a continuous market reaction to the findings release, entering the Board utility function via `review_car_weight`.
3. **Direct costs** — reviewer fees, management distraction, and internal resource consumption, entering via `review_direct_cost_weight`.

Each component is modelled as a separate stochastic draw, calibrated from distinct empirical sources.

## 2. Empirical Data Sources

### 2.1 ASX Governance Review Case Studies (2014-2023)

The primary empirical basis is a longitudinal event study of external governance reviews for ASX-listed entities, documented in `background/board/governance-review-case-studies.md`. The study applies a market-adjusted abnormal return methodology:

$$AR_{i,t} = R_{i,t} - R_{m,t}$$

where $R_{m,t}$ is the S&P/ASX 200 return, with CARs computed across three event windows (announcement, reviewer revealed, findings released).

Six case studies provide calibration data:

| Company | Period | Total CAR | Findings Window AR |
|---------|--------|-----------|-------------------|
| CBA | 2017-18 | +0.93% | +1.75% |
| Westpac | 2019-20 | -7.30% | -3.00% |
| Rio Tinto | 2020 | -4.05% | -2.65% |
| Star Entertainment | 2021-22 | -19.15% | -13.95% |
| BOQ | 2022-23 | N/A | -5.70% |
| Qantas | 2023-24 | +0.20% | +0.85% |

The findings window exhibits extreme heterogeneity: from +1.75% (CBA relief rally) to -13.95% (Star existential threat). This motivates heavy-tailed distributions.

### 2.2 Cross-Jurisdictional Review Database

A broader panel of 22 ASX 100 reviews and 15+ S&P 100 reviews (2013-2023) is documented in `background/board/external-review-usa-vs-australia.md`. Key patterns:

- **Regulatory reviews dominate**: 19 of 22 ASX 100 reviews were regulator-initiated or regulator-prompted.
- **Negative outcomes predominate in regulatory reviews**: All APRA governance self-assessments (36 firms) produced negative findings.
- **Positive outcomes are rare**: Limited to proactive non-regulatory reviews (e.g., BOQ 2020 board performance review).
- **Board-initiated crisis reviews** occupy a specific niche: they report "mistakes were made" without conceding legal liability.

### 2.3 Bayesian Posterior Analysis for Qantas Context

The context-specific posterior analysis in `background/board/external-review-distributions.md` updates the base rates using evidence specific to the September 2023 Qantas crisis: pre-existing reputational damage (ACCC ghost flights action, Senate inquiry), 30% share price decline since March 2023, and the board's incentive to signal accountability without creating litigation exposure.

The prior (base rate for board-initiated reviews) is approximately 88% positive, 7-10% balanced, 3-5% negative. The posterior, after updating on the Qantas context, shifts dramatically:

| Finding Category | Prior | Posterior | Reasoning |
|-----------------|-------|-----------|-----------|
| Balanced/Neutral | 7-10% | 75-85% | "Mistakes were made" is the dominant rational strategy |
| Negative | 3-5% | 15-20% | Updated upward due to ACCC severity, but limited by board control of scope |
| Positive | ~88% | <1% | A clean bill of health during active litigation would be non-credible |

### 2.4 Direct Cost Estimation

Direct costs are estimated from first principles in `background/board/direct-costs-governance-review.md`, decomposed into three components calibrated for a reference market capitalisation of AUD 10 billion:

| Component | Low (CAR) | Central (CAR) | High (CAR) |
|-----------|-----------|---------------|------------|
| A: Reviewer fees | -0.00020 | -0.00030 | -0.00050 |
| B: Management distraction | -0.00015 | -0.00040 | -0.00100 |
| C: Internal resources | -0.00012 | -0.00026 | -0.00045 |
| **Total** | **-0.00047** | **-0.00096** | **-0.00195** |

Component B (management distraction) is estimated via four independent approaches: executive time costing, cognitive bandwidth multiplier (4x), operational performance deviation, and analogy from M&A distraction literature. The approaches converge on a central estimate of ~40 bps.

### 2.5 Actual Historical Outcome

The Qantas review was conducted by Tom Saar over approximately 10 months (October 2023 to August 2024), producing 32 recommendations. The actual outcome (`data/actual_outcomes.json`):

- **Outcome rating**: Balanced — "mistakes were made" by the board and management, but "no deliberate wrongdoing"
- **Market reaction**: +0.85% AR in the findings window (+0.20% total CAR across all three windows)
- **Consequence**: $9.26 million clawback of former CEO's payout; 12-point reputation score uplift by mid-2024

This outcome matched the modal posterior prediction (balanced, 75-85%).

## 3. Qualitative Outcome Rating Model

### 3.1 Distribution Choice: Dirichlet-Categorical

The outcome rating follows a two-level hierarchical model:

**Level 1 (epistemic):** Draw outcome probabilities once per belief draw:

$$(p_{neg}, p_{bal}, p_{pos}) \sim \text{Dirichlet}(38, 160, 1)$$

**Level 2 (aleatoric):** Each Monte Carlo sample draws:

$$\text{outcome} \sim \text{Categorical}(p_{neg}, p_{bal}, p_{pos})$$

The Dirichlet is drawn once per belief draw because it represents epistemic uncertainty about the review process — the probabilities themselves are uncertain, not just which outcome occurs. Within a single belief draw, the probabilities are fixed and multiple MC samples draw from the same Categorical distribution.

### 3.2 Pseudo-Count Calibration

The Dirichlet pseudo-counts encode both the posterior mode and the concentration (how confident we are in that mode):

| Outcome | Pseudo-count | E[p] | Interpretation |
|---------|-------------|------|----------------|
| Negative | 38 | 0.191 | ACCC severity pushes this well above the 3-5% base rate |
| Balanced | 160 | 0.804 | "Mistakes were made" dominates as the rational strategic path |
| Positive | 1 | 0.005 | Near-zero: clean bill of health non-credible during litigation |

The total pseudo-count (199) controls concentration. At this level, the Dirichlet draws cluster tightly around the expected values — reflecting the strong posterior updating from the Qantas-specific evidence. A lower total (e.g., 20) would allow more diffuse draws, implying less confidence in the posterior.

The pseudo-counts are derived as follows:

- **Balanced = 160**: The posterior places 75-85% mass on balanced. Taking 80% as the point estimate and a total concentration of ~199 gives 0.80 x 199 = 159.2, rounded to 160.
- **Negative = 38**: The posterior places 15-20% on negative. Taking 19% gives 0.19 x 199 = 37.8, rounded to 38.
- **Positive = 1**: The posterior places <1% on positive. The minimum pseudo-count of 1 ensures the outcome is possible but extremely unlikely.

### 3.3 Board Overconfidence Bias

When the Board is the focal actor, overconfidence inflates the positive pseudo-count:

$$\alpha_{pos}^{biased} = 1 \times (1 + 10 \times \beta_{car})$$

With the default `review_car_bias = 0.03`, this gives Dirichlet(38, 160, 1.3) — a slight tilt toward positive that represents the Board's tendency to overestimate governance quality. The impact is marginal (E[p_pos] moves from 0.005 to 0.007) but directionally correct: overconfident boards assign modestly higher probability to favourable findings.

### 3.4 Downstream Effects

The outcome rating feeds into:

- **Post-review round activation**: Only "negative" triggers `state.post_review_round = True`, which activates the D4_post_review and D_rev_post_review decision nodes. "Balanced" and "positive" outcomes do not trigger the post-review round.
- **Board utility penalties**: `negative_review_finding_penalty` (default 0.571) fires for negative outcomes; `balanced_review_finding_penalty` (default 0.285) fires for balanced. These penalties are unconditional on CEO presence — review findings reflect on Board governance quality regardless of whether the CEO has departed.
- **Positive review**: No utility penalty.

## 4. Cumulative Abnormal Return (CAR) Model

### 4.1 Distribution Choice: Hierarchical Student-t

The CAR from the findings release window follows a three-level hierarchy:

$$\mu_f \sim t(\nu=4, -0.05, 0.03)$$
$$\sigma_f \sim \text{Half-Normal}(0.10)$$
$$\text{CAR} \sim t(\nu=3, \mu_f, \sigma_f)$$

### 4.2 Parameter Calibration

**Location parameter** $\mu_f \sim t(4, -0.05, 0.03)$:

- **Centre at -0.05**: The median findings-window AR across the six case studies is approximately -2.8%, but this is pulled down by the Star outlier (-13.95%). The -5% location reflects a moderate negative expectation: governance reviews typically reveal bad news, and the market reaction is mildly negative on average.
- **Scale of 0.03**: Captures uncertainty about the market's read of the findings. The range from CBA (+1.75%) to Westpac (-3.00%), excluding the Star outlier, spans roughly 5 percentage points, consistent with $\pm 2\sigma$ under scale = 0.03 for a t(4).
- **Degrees of freedom = 4**: The t(4) distribution has finite mean and variance but heavier tails than a normal, accommodating the possibility that the market's prior expectation of the review location was substantially wrong.

**Scale parameter** $\sigma_f \sim \text{Half-Normal}(0.10)$:

- Controls the volatility of the CAR around its location. The Half-Normal with scale 0.10 places most mass on $\sigma_f \in [0, 0.20]$, which is consistent with the observed heterogeneity.
- A draw of $\sigma_f = 0.05$ produces relatively tight CARs (concentrated near $\mu_f$); a draw of $\sigma_f = 0.15$ produces diffuse outcomes where CARs of $\pm 20\%$ are plausible.

**Observation-level** $\text{CAR} \sim t(3, \mu_f, \sigma_f)$:

- **Degrees of freedom = 3**: The heaviest tails in the hierarchy. This is the level that must accommodate "black swan" outcomes like Star's -13.95%. With $\nu = 3$, the distribution has finite mean but infinite kurtosis, ensuring that catastrophic findings windows remain in the support.
- The t(3) was chosen over t(5) or normal because the empirical distribution shows extreme negative skewness/kurtosis: one in six observations is a -14% outlier, which a normal or thin-tailed t would assign near-zero probability.

### 4.3 Properties

| Property | Value |
|----------|-------|
| E[CAR] | -0.05 (since E[t(4)] = 0 for $\nu > 1$) |
| CAR is independent of outcome rating | By design (see Section 4.4) |
| Board bias shifts $\mu_f$ upward | +0.03 with default overconfidence |

### 4.4 Independence of CAR and Outcome Rating

The CAR and outcome rating are modelled as conditionally independent given the belief draw. This is a deliberate simplification. In reality, a "negative" finding likely produces a more negative CAR than a "positive" finding. However:

1. **Identification**: The case study panel has only 6 observations, insufficient to estimate a conditional CAR distribution per outcome category.
2. **Market pricing**: The CAR reflects the *surprise* component of the findings relative to market expectations, not the absolute sentiment. A "balanced" finding can produce a positive CAR (if the market feared worse) or a negative CAR (if the market expected better). The Qantas case illustrates this: a balanced finding produced a +0.85% AR because the market had priced in worse outcomes.
3. **Separation of concerns**: The outcome rating drives discrete downstream game tree effects (post-review round, utility penalties), while the CAR captures the continuous market valuation impact. Keeping them independent avoids double-counting.

## 5. Direct Cost Model

### 5.1 Distribution Choice: Gamma

$$C_{direct} \sim \text{Gamma}(\alpha = 4.55, \beta = 4741)$$

where $C_{direct}$ is a positive decimal CAR value (subtracted from utility).

### 5.2 Parameter Calibration

The Gamma parameters are method-of-moments estimates from the aggregated cost analysis:

- **Mean** = $\alpha / \beta$ = 0.00096 (9.6 bps)
- **SD** = $\sqrt{\alpha} / \beta$ = 0.00045 (4.5 bps)
- **The range** [-0.00047, -0.00195] is interpreted as a 90% credible interval, giving $\sigma \approx (0.00195 - 0.00047) / (2 \times 1.645) \approx 0.00045$.

Solving: $\alpha = \mu^2 / \sigma^2 \approx 4.55$, $\beta = \mu / \sigma^2 \approx 4741$.

### 5.3 Properties

| Property | Value |
|----------|-------|
| Mean | 0.00096 (9.6 bps) |
| SD | 0.00045 (4.5 bps) |
| Mode | 0.00075 (7.5 bps) |
| 5th percentile | 0.00031 (3.1 bps) |
| 95th percentile | 0.00185 (18.5 bps) |
| Skewness | 0.94 |

The positive skewness reflects asymmetric risk: management distraction can escalate if the review becomes prolonged or contested, but there is a natural floor on costs (reviewer fees alone).

### 5.4 Materiality

The direct cost (~10 bps) is small relative to:
- The CAR from findings release (E = -500 bps)
- The market capitalisation loss that motivates the review (~3,000 bps for Qantas)
- The utility weight on review outcomes (`review_car_weight = 15.0` amplifies CAR; `review_direct_cost_weight = 15.0` amplifies direct cost similarly, but the direct cost is ~50x smaller than the expected CAR)

Direct costs should therefore rarely be the binding constraint on the Board's decision to commission a review. Their inclusion ensures the model does not treat reviews as costless, but the qualitative outcome rating and CAR dominate the utility calculation.

## 6. Integration with the Game Tree

### 6.1 Sampling Protocol

At the R chance node, the tree evaluator executes:

1. **Once per belief draw (epistemic)**:
   - Draw outcome probabilities $(p_{neg}, p_{bal}, p_{pos}) \sim \text{Dirichlet}(38, 160, 1)$ via `ReviewModel.draw_outcome_probabilities()`
   - These probabilities are held fixed across all MC samples within this belief draw

2. **Per MC sample (aleatoric)**:
   - Draw outcome $\sim \text{Categorical}(p_{neg}, p_{bal}, p_{pos})$ via `ReviewModel.sample()`
   - Draw CAR from hierarchical Student-t via `ReviewModel.sample()`
   - Draw direct cost $\sim \text{Gamma}(4.55, 4741)$ via `ReviewDirectCostModel.sample()`
   - Apply state transition: if outcome = "negative" and CEO present, set `post_review_round = True`

### 6.2 Utility Weights

The review outcomes enter the Board utility function through three channels:

| Channel | Engine Parameter | Default | Mechanism |
|---------|-----------------|---------|-----------|
| CAR impact | `review_car_weight` | 15.0 | Multiplies sampled CAR |
| Direct cost | `review_direct_cost_weight` | 15.0 | Multiplies sampled direct cost (positive = penalty) |
| Negative finding penalty | `negative_review_finding_penalty` | 0.571 | Fires when outcome = "negative", regardless of CEO status |
| Balanced finding penalty | `balanced_review_finding_penalty` | 0.285 | Fires when outcome = "balanced", regardless of CEO status |

The finding penalties are unconditional on CEO presence because review findings reflect on Board governance quality regardless of whether the CEO has departed. This was updated from an earlier specification that conditioned on `ceo_present_at_end`.

## 7. Validation Against Historical Outcome

The actual Qantas review outcome provides a single-point validation:

| Component | Model Prediction | Actual |
|-----------|-----------------|--------|
| Outcome rating | Balanced (E[p] = 0.804) | Balanced |
| CAR | E = -5% (wide uncertainty) | +0.85% (within support of t(3)) |
| Direct cost | E = 9.6 bps | Not disclosed |

The balanced outcome was the modal prediction with ~80% posterior probability. The positive CAR (+0.85%) falls within the support of the t(3) distribution, though above the expected value — consistent with the market having priced in worse outcomes prior to the findings release (the "relief" interpretation documented in the case studies).

## 8. Limitations

1. **Small calibration sample**: Only 6 ASX case studies with complete event-window data. The Student-t hierarchy is chosen partly because it is robust to small samples (heavy tails prevent overconfidence in central estimates).

2. **Independence assumption**: CAR and outcome rating are modelled as independent. A richer model would condition CAR on the outcome category, but this requires more data than currently available.

3. **Stationarity**: The model assumes the market's reaction to governance reviews is stationary across the 2014-2023 period. In practice, market sophistication in pricing governance risk has likely increased, which could compress the CAR distribution over time.

4. **Board control of scope**: The model does not explicitly capture the board's ability to influence the review's terms of reference, which affects the probability of negative findings. This is implicitly captured in the low Dirichlet pseudo-count for positive outcomes (board-initiated reviews in crisis contexts are structurally constrained).

5. **Single-entity calibration**: The Dirichlet posterior is calibrated specifically for the Qantas September 2023 context (pre-existing reputational damage, active litigation). Application to other entities would require re-elicitation of the posterior.

6. **Direct cost uncertainty**: Management distraction (the largest cost component) rests on transferability assumptions from M&A distraction literature that have not been directly validated for governance reviews.
