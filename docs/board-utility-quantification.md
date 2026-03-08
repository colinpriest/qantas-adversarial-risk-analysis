# Board Utility Quantification Pipeline

Technical documentation for `board_utility_quantification.py` — a 6-stage pipeline that estimates the Board utility function parameters using LLM stakeholder simulation.

---

## 1. Purpose and Motivation

The ARA engine (`run/run_board_mode.py`) requires numerical values for 15 Board utility weight parameters. These parameters control how the Board trades off competing concerns: vote penalties, CEO disruption costs, regulatory liability, reputational damage, etc.

**Problem:** There is no historical dataset of Australian board governance decisions with observable utility weights. The parameters in `governance_spec.xlsx` were initially set by expert judgement, but their magnitudes are uncertain and potentially inconsistent.

**Solution:** Use an LLM (gpt-4o-mini) as a calibrated stakeholder simulator. Present it with structured governance scenarios, elicit action probabilities and factor importance ratings, then estimate utility weights via maximum likelihood. This produces data-driven parameter estimates with standard errors, confidence intervals, and diagnostic tests.

**Relationship to the engine:** This script is a *supporting calibration tool*. Its outputs (estimated weights) feed into `governance_spec.xlsx`, which is then consumed by the primary simulation `run/run_board_mode.py`. The quantification script does not replace the engine — it parameterises it.

---

## 2. Pipeline Overview

```
Stage 1: Scenario Generation     → scenarios.csv (~95 scenarios)
Stage 2: LLM Elicitation         → elicitation_results.csv (95 × 40 = 3,800 LLM calls)
Stage 3: Data Preprocessing       → estimation_dataset.csv (aggregated means/variances)
Stage 4A: Softmax MLE             → 8 action-varying weight estimates + lambda
Stage 4B: Factor Rating OLS       → 5 scenario-level weight estimates
Stage 5: Behavioural Diagnostics  → 6 bias tests (loss aversion, non-linearity, etc.)
Stage 6: Validation               → within-sample KL, historical prediction, interaction effects
         ↓
    board_utility_dashboard.html  (self-contained interactive dashboard, 12 tabs)
```

---

## 3. Scenario Pair Design

### 3.1 Identification Strategy

Each scenario is a complete game state (decision node, vote outcome, CEO status, review status, feasible actions). The LLM sees a natural-language description and returns action probabilities.

Parameters are identified by **controlled variation**: scenarios that differ in exactly one feature isolate the corresponding parameter's effect on action choice.

### 3.2 Four Tiers

| Tier | Count | Purpose | Design |
|------|-------|---------|--------|
| 1 | ~40 | Parameter isolation | Single feature varies; all others at baseline |
| 2 | 20 | Joint estimation | Realistic multi-penalty combinations |
| 3 | ~34 | Behavioural probes | Matched pairs testing specific cognitive biases |
| 4 | 1 | Historical calibration | Qantas Nov 2023 AGM (out-of-sample) |

### 3.3 Tier 1 Examples

**Vote penalty (w2):** 9 scenarios at the same decision node (D_rev) with vote percentages from `VOTE_GRID = [0.10, 0.20, 0.26, 0.30, 0.40, 0.50, 0.60, 0.75, 0.83]`. All other state features held constant. The LLM's shifting action probabilities across these scenarios traces the vote penalty response surface.

**CEO departure (w1):** 8 scenarios with CEO resigned early vs present, at varied vote levels (0.30, 0.40, 0.55, 0.65, 0.83) and decision nodes (D1, D_rev). Multiple scenarios are needed because w1 is a scenario-level parameter requiring Stage 4B estimation (see Section 4).

**Inaction penalty (w_inaction):** Contrast pair — both at V=0.35 with strike, but one has CEO present at end (w_inaction fires) and one has CEO removed (w_inaction=0). The difference in action probabilities identifies w_inaction.

### 3.4 Tier 3: Behavioural Probes

| Bias | Scenarios | Design |
|------|-----------|--------|
| Loss aversion | 6 | Matched CAR gain/loss pairs at ±3%, ±5%, ±8% |
| Self-assessment | 10 | 5 vote levels × 2 review origins (board-initiated vs external) |
| Ikea effect | 10 | 5 vote levels × 2 CEO appointment types |
| Optimism bias | 2 | With/without explicit adverse probability |
| Non-linearity | 8 | Dense vote grid around thresholds |

---

## 4. Two-Stage Parameter Estimation

### 4.1 Why Two Stages Are Required

The softmax choice model is:

```
P(action | scenario; w, λ) = exp(λ · EU(action)) / Σ exp(λ · EU(action'))
```

This model identifies parameters whose `phi(scenario, action)` **varies across actions** within a scenario. Five parameters (w1, w2, w3, w4, w9) have phi that depends only on scenario-level features:

| Parameter | Phi basis | Why constant across actions |
|-----------|-----------|---------------------------|
| w1 | I[CEO resigned early] | CEO status is fixed before any action |
| w2 | (V - 0.25)² | Vote outcome is a scenario feature |
| w3 | I[V > 0.50] | Overwhelming is a scenario feature |
| w4 | V × I[V > 0.25] | Spill risk depends on vote only |
| w9 | I[V > 0.50] | Reputational spill = scenario feature |

Since these phi values are the same for every action in a scenario, they cancel in the softmax numerator/denominator ratio. The gradient is exactly zero — no amount of data helps.

These are NOT unimportant parameters. w_inaction (=w10+w11+w14=15.0 at spec default) is the largest penalty in the utility function. They simply need a **different identification strategy**.

### 4.2 Stage 4A: Softmax MLE (8 action-varying parameters)

**Estimated parameters:** w_removal, w8s, w_remove_ceo_overwhelming, w8r, w_inaction, w12, w13, w15, plus lambda (profiled).

**Method:**
1. Build phi matrix: `phi[scenario, action, param]` from `decompose_utility_board()`
2. Compute anchored contribution (CAR + direct cost, not estimated)
3. Profile likelihood over lambda grid `[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]`
4. For each lambda, minimise cross-entropy loss via L-BFGS-B with analytical gradient
5. Multi-start: 10 random starting points per lambda value
6. Select (lambda, weights) pair with lowest cross-entropy
7. Standard errors: Hessian-based (primary) + nonparametric bootstrap (B=500)
8. P-values: Wald test z = |estimate/SE|

**Collinearity handling:** w7 and w8 fire on identical indicator (CEO removed involuntarily) → collapsed to w_removal. w10, w11, w14 fire on identical indicator (strike AND CEO present) → collapsed to w_inaction. Condition number is monitored; ridge regularisation (α=0.01) applied if needed.

### 4.3 Stage 4B: Factor Rating OLS (5 scenario-level parameters)

**Estimated parameters:** w1, w2, w3, w4, w9.

**Data source:** The LLM provides factor ratings (1-5 Likert scale, 10 factors) per scenario. These ratings capture the LLM's perceived importance of scenario features that the softmax model cannot identify.

**Method per parameter:**
1. Extract scenario-level phi values from state vectors
2. Identify mapped factor(s) from `FACTOR_PARAM_MAP` (e.g., w1 → Factor 4 "CEO relationship loss")
3. Run OLS: `mean_factor_rating = α + β · |phi_k(scenario)|`
4. The coefficient β measures Likert-points-per-unit-phi

**Factor-to-parameter mapping:**

| Parameter | Mapped Factor(s) | Rationale |
|-----------|-------------------|-----------|
| w1 | F4 (CEO relationship/knowledge loss) | CEO departure directly drives F4 concern |
| w2 | F1 (second strike risk) | Vote level drives second-strike concern |
| w3 | F9+F10 (activist escalation + board cohesion) | Overwhelming vote drives both |
| w4 | F1+F5 (second strike + market reaction) | Spill risk = vote × strike interaction |
| w9 | F7 (reputational contagion) | Overwhelming → contagion to other boards |

**Special case — w2:** The phi transform `(V-0.25)²` compresses variation near zero, giving poor R². Factor F1 ratings respond more linearly to V. The regression uses V directly as the regressor, then converts via: `w2 = gamma / (2 × (V_ref - 0.25))`, where `V_ref` is the mean V across strike scenarios and `gamma` is the OLS slope.

---

## 5. LLM Prompts

### 5.1 System Prompt Architecture

The system prompt establishes four contexts:

| Section | Content | Design rationale |
|---------|---------|-----------------|
| A: Board Persona | Names all 8 directors, describes professional backgrounds | Grounds simulation in specific perspectives, models internal disagreement |
| B: Legal Context | Three liability channels (personal, spill, corporate), two-strikes rule, ASA role | Ensures LLM reasons about real regulatory mechanisms |
| C: Historical Context | 2023 Qantas crises (ACCC, Senate, customer service) | Provides crisis backdrop WITHOUT revealing outcomes |
| D: Response Format | Factor rating instructions + probability assignment | Structured output for quantitative analysis |

**Critical constraint:** Section C states "The vote outcome and subsequent events are NOT known to the Board at this decision point." This prevents hindsight contamination.

### 5.2 Scenario Prompt Construction

Each scenario prompt is built conditionally from the state vector:

1. **CEO status** — Present or resigned early
2. **Vote outcome** — Percentage, first-strike/overwhelming status
3. **Prior Board action** — D1 decision (for D_rev/D_rev_post nodes)
4. **Review status** — Commissioned, board-initiated vs external, adverse/positive findings, CAR
5. **Decision point** — Feasible actions with descriptions

### 5.3 Factor Rating Design

10 factors rated 1-5 (1=not significant, 5=decisive). Presentation order is randomised per LLM call using `hashlib.sha256(prompt_text + seed)` for deterministic cross-process stability. Factor indices are preserved regardless of presentation position.

The factor ratings serve dual purposes:
1. **Stage 4B estimation** — Identify scenario-level parameters via regression
2. **Behavioural diagnostics** — Detect order effects (primacy/recency bias)

### 5.4 Why This Context Was Needed

The LLM needs sufficient context to produce calibrated probability distributions:
- **Board composition** grounds heterogeneous risk preferences (a lawyer weights liability differently from a media executive)
- **Legal framework** enables reasoning about spill risk thresholds (25%, 50%)
- **Crisis specifics** ensure responses reflect Qantas's actual governance environment, not generic board behaviour
- **Explicit feasibility constraints** prevent the LLM from proposing actions outside the game tree

Without this context, the LLM defaults to generic corporate governance platitudes and produces uniform distributions across actions.

---

## 6. HTML Dashboard

The dashboard is a self-contained HTML file (~5MB with embedded Plotly.js) that presents all pipeline results in 12 interactive tabs.

### 6.1 Tab Reference

| Tab | Content | How to interpret |
|-----|---------|-----------------|
| **Overview** | Run status, model, config, timing | Green = complete, amber = in-progress |
| **Cost & Usage** | Token counts, API cost, encoding stats | Total cost at gpt-4o-mini prices |
| **Scenario Battery** | Table of all scenarios with state vectors | Sortable; check scenario count per tier |
| **Elicitation Results** | Per-scenario success rate, parse failures | Flag scenarios with <70% parse success |
| **Elicited Probabilities** | Mean probability vectors, top action per scenario | Sanity check: do probabilities shift as expected? |
| **Parameter Estimates** | Table + forest plot with 95% CIs | Blue=factor rating, green=softmax MLE. P-values for significance. |
| **Covariance** | Heatmap + numeric grid, high-correlation warnings | Red/blue = high correlation. w8s↔w_inaction ≈ -0.99 is structural (complementary phi). |
| **Behavioural Diagnostics** | 6 bias tests with detail tables | "confirmed" = bias detected (p<0.05). "null" = no evidence. Per-factor order effects table. |
| **Interaction Effects** | Residual scatter, KL by node, heterogeneity tests | Flat residuals = good model fit. Significant Mann-Whitney = model fits differently by subgroup. |
| **Validation** | Within-sample KL, historical prediction, worst scenarios | Mean KL < 0.05 = good. Historical: check if D1_review ranks high. |
| **Linearity Diagnostics** | Q-Q plot, scale-location, phi basis table | Q-Q: points on 45° line = normal residuals. Scale-location: flat = homoscedastic. |
| **Raw Data** | Download links for all CSV/JSON outputs | Click to download for offline analysis |

### 6.2 Key Metrics

- **Forest plot**: All 13 estimated parameters with 95% confidence intervals. Square markers = factor rating (Stage 4B). Circle markers = softmax MLE (Stage 4A). Zero reference line for sign interpretation.
- **Mean KL divergence**: Average information loss between LLM probabilities and model predictions. Target < 0.05.
- **Condition number**: Measures phi matrix collinearity. Values > 100 suggest potential estimation instability.
- **Historical prediction**: Out-of-sample test using Tier 4 (Qantas AGM 2023). Checks whether D1_review (what actually happened) is the top-predicted action.

---

## 7. Parameter Mapping to Engine

The quantification pipeline uses collapsed parameter names that map to the engine's `utilities_board` parameters:

| Pipeline param | Engine parameter(s) | Relationship |
|---------------|--------------------|-|
| w1 | `early_ceo_departure_cost` | Direct: w1 = engine value |
| w2 | `vote_penalty_weight` | Direct: w2 = engine value |
| w3 | `overwhelming_penalty_weight` | Direct: w3 = engine value |
| w4 | `spill_risk_weight` | Direct: w4 = engine value |
| w_removal | `implementation_cost_sack` + `ceo_loss_cost` | Sum: w_removal = w7 + w8 |
| w8s | `ceo_loss_shock_strike` | Direct |
| w_remove_ceo_overwhelming | `ceo_loss_shock_overwhelming` | Direct |
| w8r | `ceo_loss_shock_adverse` | Direct |
| w9 | `reputational_spill_weight` | Direct |
| w_inaction | `second_strike_spill_penalty` + `board_regulatory_liability` + `qantas_legal_d_rev_penalty` | Sum: w_inaction = w10 + w11 + w14 |
| w12 | `board_d1_liability` | Direct |
| w13 | `qantas_legal_d1_penalty` | Direct |
| w15 | `adverse_review_ceo_present_penalty` | Direct |

For collapsed parameters (w_removal, w_inaction), the helper script `run/apply_estimated_weights.py` allocates the estimated total across constituent engine parameters proportionally to their spec defaults.

---

## 8. Usage

### Running the pipeline

```bash
# Full pipeline (default: 40 reps, gpt-4o-mini)
python board_utility_quantification.py --all

# Stages selectively (uses cached elicitation)
python board_utility_quantification.py --stage 4,5,6

# Smoke test (2 reps, fast)
python board_utility_quantification.py --stage 1,2 --n_reps 2

# Scenarios only (no API key needed)
python board_utility_quantification.py --stage 1
```

### Applying estimated weights to the engine

```bash
# After reviewing the dashboard and confirming the estimates:
python -m run.apply_estimated_weights outputs/parameter_estimates.csv

# Preview changes without writing:
python -m run.apply_estimated_weights outputs/parameter_estimates.csv --dry-run
```

### Running the engine with updated weights

```bash
# The engine reads from governance_spec.xlsx automatically
python -m run.run_board_mode --checkpoint C0 --n_draws 100
```

---

## 9. Design Decisions

1. **gpt-4o-mini over gpt-4o**: 17× cheaper per token, fast enough for 3,800 calls. Preliminary tests showed comparable action probability distributions on governance scenarios.

2. **Softmax + factor regression (two-stage) over joint model**: The softmax model is well-established in discrete choice. Factor ratings provide an independent data channel for scenario-level parameters. A joint model would require stronger distributional assumptions.

3. **40 reps per scenario**: Balances precision vs cost. 29% SE reduction per doubling. At 40 reps, even the widest CI (w2) is informative. Cost: ~$0.60/full run.

4. **Deterministic caching via hashlib**: Python's `hash()` is randomised per process (PYTHONHASHSEED). Using `hashlib.sha256` ensures identical cache keys across runs. System prompt excluded from key since factor order is a randomised control, not a semantic variation.

5. **Collinearity collapse**: w7+w8 and w10+w11+w14 are structurally collinear (identical phi). Attempting to estimate them separately would produce infinite standard errors. The collapsed estimates can be allocated back to constituents proportionally to spec defaults.

6. **Loss aversion anchored at λ=2.25**: From Kahneman & Tversky (1992). Not estimated — tested diagnostically. The CAR term is the only component with gain/loss asymmetry, so λ_LA is not separately identifiable from the choice data.
