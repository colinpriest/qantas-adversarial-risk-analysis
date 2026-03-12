# Board Utility Quantification Pipeline

Technical documentation for `board_utility_quantification.py` — a pipeline that estimates the Board utility function parameters using LLM stakeholder simulation and Bayesian estimation, then feeds the posterior weight draws into a stochastic game tree simulation that computes decision probabilities at every Board node.

---

## 1. Purpose and Motivation

The ARA game tree requires numerical values for the Board's utility weight parameters. These parameters control how the Board trades off competing concerns: vote penalties, CEO disruption costs, review findings, inaction liability, and reputational damage.

**Problem:** There is no historical dataset of Australian board governance decisions with observable utility weights. The parameters in `governance_spec.xlsx` were initially set by expert judgement, but their magnitudes are uncertain and potentially inconsistent.

**Solution:** A three-stage approach:

1. **Scenario elicitation** — Use an LLM (gpt-4o-mini) as a calibrated stakeholder simulator to generate Likert-scale severity ratings for structured governance scenarios.
2. **Bayesian estimation** — Fit an ordinal probit model (Stan) to the Likert data, producing posterior draws of all utility weight parameters.
3. **Stochastic simulation** — Feed the posterior weight draws into the game tree, where Board decision probabilities emerge from per-draw utility maximisation (argmax-count).

**Relationship to the engine:** This script is a *supporting calibration tool*. Its outputs (posterior weight draws in `stan_posterior_draws.npz` and point estimates in `parameter_estimates.csv`) feed into the unified game tree (`run/run_unified_ARA.py`). The quantification script does not replace the engine — it parameterises it.

---

## 2. Pipeline Overview

```
Stage 1: Scenario Generation     -> scenarios.csv (~95 scenarios)
Stage 2: LLM Elicitation         -> elicitation_results.csv (95 x 40 = 3,800 LLM calls)
Stage 3: Data Preprocessing       -> estimation_dataset.csv (Likert long-form)
Stage 4: Bayesian Estimation      -> stan_posterior_draws.npz (8,000 posterior draws)
                                  -> parameter_estimates.csv (posterior summaries)
Stage 5: Behavioural Diagnostics  -> 6 bias tests (loss aversion, non-linearity, etc.)
Stage 6: Validation & Dashboard   -> board_utility_dashboard.html (12-tab interactive)
         |
         v
    run/run_unified_ARA.py        -> loads w_draws, builds game tree,
                                     computes Board action probabilities
                                     via per-draw argmax-count
```

---

## 3. Paired Scenario Design

### 3.1 Identification Strategy

Each scenario is a complete game state: a decision node, a vote outcome, CEO status, review status, and a set of feasible actions. The LLM sees a natural-language description and returns Likert ratings (1-5) for each feasible action within that scenario.

Parameters are identified by **controlled variation**: pairs or sets of scenarios that differ in exactly one game-state feature isolate the corresponding parameter's effect on the LLM's severity ratings. When scenario A differs from scenario B only in whether a governance review was commissioned, the difference in Likert ratings between A and B identifies the review-related weight parameters.

### 3.2 The Basis Function Decomposition

Board expected utility for a (scenario, action) pair is decomposed as:

```
EU(scenario, action) = sum_k  w_k * phi_k(scenario, action)  +  anchored(scenario)
```

where:
- `phi_k(scenario, action)` is the **basis function** (indicator or graduated) for parameter k — computed by `decompose_utility_board()`
- `w_k > 0` is the weight being estimated (all weights are strictly positive)
- `anchored(scenario)` captures fixed contributions (CAR, direct costs) that don't vary by action

The key insight: **a parameter is identifiable from choice data only if its basis function varies across actions within at least some scenarios.** Parameters whose phi is constant across all actions within every scenario cancel in the softmax ratio and have zero gradient — no amount of data helps.

### 3.3 Four Scenario Tiers

| Tier | Count | Purpose | Design |
|------|-------|---------|--------|
| 1 | ~55 | Parameter isolation | Single feature varies; all others at baseline |
| 2 | ~28 | Joint estimation + scale anchoring | Realistic multi-penalty combinations + CAR anchors |
| 3 | ~34 | Behavioural probes | Matched pairs testing specific cognitive biases |
| 4 | 1 | Historical calibration | Qantas Nov 2023 AGM (out-of-sample) |

### 3.4 Tier 1: Parameter Isolation Pairs

Each parameter has a set of scenarios designed to isolate its effect. The paired design ensures that the LLM's rating shift is attributable to the target parameter, not confounds.

**w_inaction_base (Board took minimal action at all nodes):**

The key pairing is at D1 where:
- `D0_minimal` action: `phi = -1` (board inactive → penalty fires)
- `D1_review` action: `phi = 0` (board responded → no penalty)
- `D3_ceo_transition` action: `phi = 0` (board responded decisively)

The LLM sees the same vote outcome, same CEO status, same everything — only the action label changes. If the LLM rates `D0_minimal` lower (worse) than `D1_review`, that rating difference identifies w_inaction_base.

Eight vote levels (0.26–0.83) trace how the inaction penalty interacts with vote severity.

**w_passivity (Board passivity after CEO departure):**

Scenario pairs where CEO status varies (present vs resigned_early), everything else constant. When CEO has resigned, `phi = -(1 - response_strength)` — graduated by how strongly the Board responds:
- `D0_minimal`: phi = -1.0 (full penalty — Board did nothing after CEO left)
- `D1_review`: phi = -0.5 (partial penalty — Board commissioned review but didn't replace CEO)
- `D3_ceo_transition`: phi = 0.0 (no penalty — Board acted decisively)

Nine scenarios: a pair at low vote (no strike) for clean identification, plus 5 CEO-resigned scenarios at varied vote levels (0.30–0.83) for robust cross-vote estimation, plus 2 D_rev scenarios for cross-node identification.

**w_removal (CEO involuntary removal cost):**

Contrast pairs where `ceo_present_at_end` varies (True vs False). When CEO is involuntarily removed, `phi = -1`; when retained, `phi = 0`. The rating difference between sacking and retaining the CEO — holding all else constant — identifies the removal cost.

Four scenarios: 2 pairs at different d1_action contexts (D0_minimal vs D1_review) × vote = 0.20–0.30.

**w_review_negative / w_review_balanced (review finding penalties):**

Contrast triplets where review outcome varies (negative / balanced / positive), everything else constant:
- Negative: `phi_negative = -1`, `phi_balanced = 0`
- Balanced: `phi_negative = 0`, `phi_balanced = -1`
- Positive: both phi = 0 (baseline)

Three vote levels (0.35, 0.50, 0.60) × 3 outcomes = 9 scenarios, ensuring identification at both sub-overwhelming and overwhelming vote levels.

**w_ceo_accountability (accountability benefit for removal backed by review):**

This interaction term (`phi = +I[removed_involuntary AND review_commissioned]`) is identified from D_rev_post scenarios where the Board considers sacking after a review. Five D_rev_post(negative) scenarios, 3 D_rev_post(balanced), 3 D_rev_post(positive), plus 4 D_rev(no-review) scenarios to pin down w_removal separately.

### 3.5 Tier 2: Joint Multi-Penalty Scenarios

Twenty-eight realistic scenarios where multiple penalties fire simultaneously. These scenarios serve two purposes:

1. **Joint estimation** — Verify that parameters estimated from Tier 1 isolation pairs generalise to multi-penalty combinations. If the model is well-specified, predicted ratings from Tier 1 weights should match observed Tier 2 ratings.

2. **CAR scale anchoring** — Two anchor scenarios with known CAR differences (-0.01 vs -0.12) at otherwise identical states. The rating difference calibrates the utility-to-CAR ratio (w_CAR_anchor = 15.0).

Configurations span: 9 vote levels × multiple D1 action contexts × review/no-review × negative/balanced/positive outcomes × CEO present/removed.

### 3.6 Tier 3: Behavioural Probes

Matched pairs testing cognitive biases. These do not enter the estimation — they diagnose whether the LLM exhibits biases that would violate model assumptions.

| Bias | Scenarios | Design |
|------|-----------|--------|
| Loss aversion | 10 | Matched CAR gain/loss pairs at ±1%, ±3%, ±5%, ±8%, ±14% |
| Self-assessment | 10 | 5 vote levels × 2 review origins (board-initiated vs external) |
| Ikea effect | 10 | 5 vote levels × 2 CEO appointment types |
| Optimism bias | 2 | With/without explicit adverse probability |
| Non-linearity | 8 | Dense vote grid around thresholds |

### 3.7 Tier 4: Historical Calibration

One out-of-sample scenario: the Qantas November 2023 AGM at V = 0.83 (actual outcome). This tests whether the model's predicted action ranking matches what actually happened (Board chose D1_review).

---

## 4. Bayesian Estimation via Ordinal Probit

### 4.1 Why Ordinal Probit

The LLM produces Likert ratings (1-5 integer scale) for each (scenario, action) pair. These are ordinal observations of an underlying latent utility — the LLM's assessment of how appropriate/necessary each action is given the scenario.

The ordinal probit model maps latent utility to observed Likert categories via cumulative normal thresholds (cutpoints). This is a natural model for the data-generating process: the LLM has an internal severity assessment that gets discretised into 5 categories.

### 4.2 Model Structure

The Stan model (`models/ordinal_utility.stan`) specifies:

**Latent utility:**
```
mu[s] = phi[s] . w  +  anchored[s]
        - I[strike] * w_strike * vote_x_strike[s]
        - I[overwhelming] * w_overwhelming * vote_x_overwhelming[s]
```

where `s` indexes unique (scenario, action) pairs, `w` is the K=10 vector of utility weights, and `phi[s]` is the basis function row for pair s.

**Normalisation:** Raw mu values can span 20+ units (dominated by vote penalties), placing most observations in saturated probit tails with zero gradient. A pre-computed `mu_scale` normalises `eta = (mu + RE) / mu_scale` so the probit sees values in ~[-3, 3].

**Observation model:**
```
y[n] ~ ordered_probit(eta[sa_id[n]], cutpoints)
```

where `eta` includes a scenario-level random intercept (`sigma_scenario * z_scenario[sc]`, non-centred parameterisation) to capture unmodelled scenario heterogeneity.

### 4.3 Estimated Parameters (12 total)

**10 action-varying weights** (lognormal priors centred at spec defaults):

| # | Parameter | Description | Prior median | Phi pattern |
|---|-----------|-------------|--------------|-------------|
| 1 | w_inaction_base | Board did nothing at any node | 3.0 | -I[board_inactive] |
| 2 | w_inaction_no_review | No governance review commissioned | 2.0 | -I[no review AND CEO not removed] |
| 3 | w_inaction_delay | Reactive (not proactive) governance | 1.5 | -I[d1=minimal AND d_rev acted] |
| 4 | w_passivity | Board passivity after CEO departure | 0.5 | -I[CEO resigned] × (1 - response_strength) |
| 5 | w_removal | CEO involuntary removal cost | 1.8 | -I[removed involuntary] |
| 6 | w_remove_ceo_overwhelming | Removal relief at overwhelming vote | 0.5 | +I[removed] × I[overwhelming] |
| 7 | w_review_negative | Negative review finding penalty | 5.0 | -I[review AND negative] |
| 8 | w_review_balanced | Balanced review finding penalty | 2.5 | -I[review AND balanced] |
| 9 | w_review_post_removal | No review after CEO removal | 3.0 | -I[removed AND no review] |
| 10 | w_ceo_accountability | Accountability benefit: removal + review | 3.0 | +I[removed AND review] |

**Ordering constraint:** w_removal > w_remove_ceo_overwhelming, enforced via `w[5] = w[6] + delta_removal` where `delta_removal ~ lognormal(0.26, 1.0)`.

**2 vote penalty weights** (scenario-level, estimated separately):

| Parameter | Description | Prior median |
|-----------|-------------|--------------|
| w_strike | First-strike penalty escalation | 2.0 |
| w_overwhelming | Additional overwhelming penalty | 3.0 |

These enter as linear penalties: `-w_strike * max(0, (V-0.25)/0.75)` and `-w_overwhelming * max(0, (V-0.50)/0.50)`.

### 4.4 Cutpoint Reparameterisation

The 4 ordinal probit cutpoints are reparameterised for numerical stability:

```
cutpoint[1] = 3 * tanh(base_raw)           # location in [-3, 3]
cutpoint[g+1] = cutpoint[g] + 0.25 + 2.0 * inv_logit(gap_raw[g])   # gap in (0.25, 2.25)
```

This avoids: (a) cutpoint degeneracy (gaps collapsing to zero), (b) exp() overflow during warmup, (c) disordered cutpoints that violate the ordered constraint.

### 4.5 MCMC Configuration

- 4 chains × 2,000 sampling draws (+ 1,000 warmup) = 8,000 posterior draws
- `adapt_delta = 0.99` (high, to minimise divergences in the constrained parameter space)
- `max_treedepth = 15`
- Convergence diagnostics: R-hat, bulk ESS, divergences reported per parameter

### 4.6 Posterior Outputs

The estimation produces:

1. **`stan_posterior_draws.npz`** — 8,000 × 10 matrix of weight draws (`w_draws`), plus cutpoint and sigma draws. This is the primary input to the game tree.
2. **`parameter_estimates.csv`** — Posterior mean, SD, 95% CI, and estimation method per parameter.
3. **Covariance matrix** — Full posterior covariance of the 12 parameters, for downstream uncertainty propagation.

---

## 5. From Posterior Draws to Board Decision Probabilities

### 5.1 The Game Tree Simulation

The unified game tree (`run/game_tree.py`) loads the posterior weight draws and uses them to compute Board decision probabilities at every Board decision node. This is where the Bayesian estimation connects to actionable game-theoretic predictions.

### 5.2 Board EU Computation at Terminal Nodes

At each terminal node, Board EU for a single posterior draw `i` is:

```
EU_board[i] = w_draws[i, :] @ phi_vector  +  anchored_value
```

where `phi_vector` contains the basis function values for the terminal state (computed by `decompose_utility_board()`) and `anchored_value` contains the CAR and direct cost contributions scaled by their fixed anchors (w_CAR = 15.0, w_cost = 15.0).

Because `w_draws` is an (n_draws, K) matrix, each posterior draw produces a different EU — reflecting uncertainty in the Board's preferences.

### 5.3 EU Propagation Through the Tree

EU arrays propagate backward from terminals through the tree:

- **Terminal nodes:** Compute `EU_board[i]` for each draw `i` as above.
- **Chance nodes (V, R):** Weighted sum over child EU arrays using per-draw Dirichlet probabilities — `EU_parent[i] = sum_j  p_j[i] * EU_child_j[i]`.
- **Non-Board decision nodes (A2, D4):** Weighted sum using fixed action probabilities from `TREE_DEFAULT_PROBS`, with per-draw Dirichlet epistemic noise.
- **Board decision nodes (D1, D_rev, D_rev_post):** **Argmax-count** — see next section.

### 5.4 Argmax-Count: How Board Probabilities Emerge

At each Board decision node, the game tree has computed EU arrays for every feasible action's subtree. The Board's decision probability for each action is determined by **per-draw utility maximisation**:

```python
# board_eu_mat: (n_draws, n_actions) — EU of each action for each posterior draw
board_eu_mat = column_stack([child_eus["Board"] for action in feasible])

# For each draw, find which action maximises Board EU
best_idx = argmax(board_eu_mat, axis=1)   # (n_draws,) index of best action per draw

# Count how often each action wins across all draws
for j, action in enumerate(feasible):
    count = sum(best_idx == j)
    action_probs[action] = (count + alpha) / (n_draws + K * alpha)
```

where `alpha = 1.0` is the Laplacian smoothing constant (ensures no action gets exactly 0%).

**Interpretation:** On posterior draw `i`, the Board has specific weight values `w_draws[i, :]`. Given those weights, one action has the highest EU — that action gets the Board's "vote" for draw `i`. The final probability is the fraction of draws on which each action wins.

This is fundamentally different from a softmax: there is no temperature parameter. The probabilities reflect **epistemic uncertainty about the Board's preferences**, not bounded rationality. If all 8,000 posterior draws agree that `D1_review` is optimal, it gets ~99.8% (with Laplacian smoothing). If the posterior is diffuse and the EU difference between actions is small relative to posterior variance, the probabilities will be closer to uniform.

### 5.5 Dirichlet Epistemic Noise

After computing the mean action probabilities from argmax-count, the tree adds per-draw Dirichlet noise:

```python
CONC_SUM = 20.0
dir_alpha = [p / prob_sum * CONC_SUM for p in prob_values]
per_draw_probs = rng.dirichlet(dir_alpha, size=n_draws)
```

This captures two forms of uncertainty:
1. **Parameter uncertainty** (from the posterior) — already reflected in the argmax-count
2. **Small-sample epistemic uncertainty** — even with 8,000 draws, the true probability might differ from the count estimate

The per-draw Dirichlet probabilities are used to propagate all three actor EU streams (Board, ASA, CEO) through the node:

```python
for actor in ["Board", "ASA", "CEO"]:
    eu_mat = column_stack([child_eus[actor] for action in feasible])
    node_eus[actor] = sum(per_draw_probs * eu_mat, axis=1)
```

### 5.6 Concrete Example: D1 Node (CEO Stays Branch)

At D1 in the "CEO stays" scenario, the Board has three feasible actions:

| Action | Description |
|--------|-------------|
| D0_minimal | Do nothing — minimal governance response |
| D1_review | Commission a governance review |
| D3_ceo_transition | Force CEO exit |

For each posterior draw `i`, the tree has already computed the EU of the entire subtree below each action (recursing through A2 → V → D4 → D_rev → R → Terminal). The argmax picks the best action per draw:

- Draw 1: w_inaction_base=4.2, w_review_negative=3.1, ... → D1_review wins
- Draw 2: w_inaction_base=1.8, w_removal=0.3, ... → D3_ceo_transition wins
- Draw 3: w_inaction_base=5.1, w_review_negative=6.0, ... → D1_review wins
- ...
- Draw 8000: → D1_review wins

Count: D0_minimal wins 10 times, D1_review wins 4780 times, D3_ceo_transition wins 3210 times.

Probabilities (with Laplacian smoothing, alpha=1):
```
P(D0_minimal)       = (10 + 1) / (8000 + 3) ≈ 0.001
P(D1_review)         = (4780 + 1) / (8000 + 3) ≈ 0.598
P(D3_ceo_transition) = (3210 + 1) / (8000 + 3) ≈ 0.401
```

### 5.7 Why Argmax-Count, Not Softmax

The softmax model `P(a) ~ exp(lambda * EU(a))` requires choosing a rationality parameter `lambda`. Setting lambda too low makes the Board appear random; too high makes it deterministic. The "right" lambda is not known.

Argmax-count sidesteps this by instead saying: "Given our posterior uncertainty about what the Board values, how often does each action come out best?" This is a fully Bayesian approach — the spread of probabilities across actions reflects posterior uncertainty about the utility weights, not bounded rationality.

When the posterior is tight (parameters well-identified), the probabilities concentrate on one action. When the posterior is diffuse, they spread across actions. The data determines the degree of decisiveness, not an ad hoc temperature parameter.

---

## 6. LLM Prompts

### 6.1 System Prompt Architecture

The system prompt establishes four contexts:

| Section | Content | Design rationale |
|---------|---------|-----------------|
| A: Board Persona | Names all 8 directors, describes professional backgrounds | Grounds simulation in specific perspectives, models internal disagreement |
| B: Legal Context | Three liability channels (personal, spill, corporate), two-strikes rule, ASA role | Ensures LLM reasons about real regulatory mechanisms |
| C: Historical Context | 2023 Qantas crises (ACCC, Senate, customer service) | Provides crisis backdrop WITHOUT revealing outcomes |
| D: Response Format | Likert rating instructions + action assessment | Structured output for quantitative analysis |

**Critical constraint:** Section C states "The vote outcome and subsequent events are NOT known to the Board at this decision point." This prevents hindsight contamination.

### 6.2 Scenario Prompt Construction

Each scenario prompt is built conditionally from the state vector:

1. **CEO status** — Present (with severe ESG crisis context and peer benchmark) or resigned early
2. **Vote outcome** — Percentage, first-strike/overwhelming status
3. **Prior Board action** — D1 decision (for D_rev/D_rev_post nodes)
4. **Review status** — Commissioned, board-initiated vs external, negative/balanced/positive findings, CAR
5. **Decision point** — Feasible actions with descriptions

### 6.3 Likert Rating Design

For each feasible action, the LLM rates severity/appropriateness on a 1-5 scale (1 = not significant / inappropriate, 5 = decisive / strongly appropriate). This produces ordinal observations that directly feed the ordinal probit model.

40 independent LLM calls per scenario provide sufficient replication to estimate within-scenario variance and detect systematic biases.

---

## 7. HTML Dashboard

The dashboard is a self-contained HTML file (~5MB with embedded Plotly.js) that presents all pipeline results in 12 interactive tabs.

### 7.1 Tab Reference

| Tab | Content | How to interpret |
|-----|---------|-----------------|
| **Overview** | Run status, model, config, timing | Green = complete, amber = in-progress |
| **Cost & Usage** | Token counts, API cost, encoding stats | Total cost at gpt-4o-mini prices |
| **Scenario Battery** | Table of all scenarios with state vectors | Sortable; check scenario count per tier |
| **Elicitation Results** | Per-scenario success rate, parse failures | Flag scenarios with <70% parse success |
| **Elicited Probabilities** | Mean Likert ratings, top action per scenario | Sanity check: do ratings shift as expected? |
| **Parameter Estimates** | Table + forest plot with 95% CIs | Posterior mean, SD, 95% CI per parameter. |
| **Covariance** | Heatmap + numeric grid, high-correlation warnings | Red/blue = high correlation. Check for structural collinearities. |
| **Behavioural Diagnostics** | 6 bias tests with detail tables | "confirmed" = bias detected (p<0.05). "null" = no evidence. |
| **Interaction Effects** | Residual scatter, KL by node, heterogeneity tests | Flat residuals = good model fit. |
| **Validation** | Within-sample KL, historical prediction, worst scenarios | Historical: check if D1_review ranks high for Tier 4. |
| **Linearity Diagnostics** | Q-Q plot, scale-location, phi basis table | Q-Q: points on 45 line = normal residuals. |
| **Raw Data** | Download links for all CSV/JSON outputs | Click to download for offline analysis |

### 7.2 Key Metrics

- **Forest plot**: All 12 estimated parameters with 95% credible intervals. Posterior summaries from the Stan ordinal probit.
- **R-hat**: All parameters should have R-hat < 1.01 for convergence.
- **Bulk ESS**: Effective sample size; should be > 400 per parameter.
- **Divergences**: Zero is ideal. A few (<10) is acceptable if concentrated in warmup.
- **Historical prediction**: Out-of-sample test using Tier 4 (Qantas AGM 2023). Checks whether D1_review (what actually happened) is the top-predicted action.

---

## 8. Parameter Mapping to Engine

The quantification pipeline parameter names map to the engine's `utilities_board` parameters in `governance_spec.xlsx`:

| Pipeline param | Engine parameter(s) | Relationship |
|---------------|--------------------|-|
| w_inaction_base | `inaction_base_penalty` | Direct |
| w_inaction_no_review | `inaction_no_review_penalty` | Direct |
| w_inaction_delay | `inaction_delay_penalty` | Direct |
| w_passivity | `board_passivity_after_departure` | Direct |
| w_removal | `implementation_cost_sack` + `ceo_loss_cost` | Sum: allocated proportionally to spec defaults |
| w_remove_ceo_overwhelming | `ceo_loss_shock_overwhelming` | Direct |
| w_review_negative | `negative_review_finding_penalty` | Direct |
| w_review_balanced | `balanced_review_finding_penalty` | Direct |
| w_review_post_removal | `review_after_removal_penalty` | Direct |
| w_ceo_accountability | `ceo_accountability_benefit` | Direct |
| w_strike | `vote_penalty_weight` | Direct (linear in vote excess) |
| w_overwhelming | `overwhelming_penalty_weight` | Direct (linear in vote excess) |

For collapsed parameters (w_removal), the helper script `run/apply_estimated_weights.py` allocates the estimated total across constituent engine parameters proportionally to their spec defaults.

---

## 9. End-to-End Data Flow

```
gpt-4o-mini                    Stan MCMC                   Game Tree
-----------                    ---------                   ---------
  |                               |                           |
  | Likert ratings               |                           |
  | (1-5 per action)            |                           |
  v                               |                           |
scenarios.csv  --->  ordinal_utility.stan  --->  stan_posterior_draws.npz
(~95 scenarios       (ordinal probit,           (8000 x 10 weight draws)
 x 40 reps           12 parameters,                    |
 = 3,800 calls)      4 cutpoints,                      |
                     scenario RE)                       v
                                               run_unified_ARA.py
                                                    |
                                                    | loads w_draws
                                                    | builds game tree
                                                    |
                                                    v
                                               Board EU per draw:
                                               EU[i] = w_draws[i,:] @ phi + anchored
                                                    |
                                                    | argmax per draw
                                                    v
                                               P(action) = count(argmax == action) / n_draws
                                                    |
                                                    v
                                               tree_interactive_Board_unified.html
                                               (probabilities shown at each node)
```

---

## 10. Usage

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

### Running the game tree with estimated weights

```bash
# Build all 4 strategic modes with 500 posterior draws
python -m run.run_unified_ARA --n_draws 500

# Fast diagnostic run (no LLM commentary)
python -m run.run_unified_ARA --n_draws 500 --no-commentary
```

### Applying estimated weights to governance_spec.xlsx

```bash
# After reviewing the dashboard and confirming the estimates:
python -m run.apply_estimated_weights outputs/parameter_estimates.csv

# Preview changes without writing:
python -m run.apply_estimated_weights outputs/parameter_estimates.csv --dry-run
```

---

## 11. Design Decisions

1. **gpt-4o-mini over gpt-4o**: 17x cheaper per token, fast enough for 3,800 calls. Preliminary tests showed comparable Likert distributions on governance scenarios.

2. **Ordinal probit over softmax MLE**: The LLM produces ordinal severity ratings, not choice probabilities. Ordinal probit is the natural model for this data type. It also avoids the need to estimate a rationality parameter (lambda).

3. **40 reps per scenario**: Balances precision vs cost. Each rep is an independent LLM call with randomised factor presentation order. At 40 reps, posterior precision is dominated by across-scenario variation, not within-scenario noise. Cost: ~$0.60/full run.

4. **Argmax-count over softmax for game tree probabilities**: Softmax requires choosing a temperature (lambda). Argmax-count derives probabilities purely from Bayesian posterior uncertainty — when posterior draws agree, probabilities concentrate; when they disagree, probabilities spread. No ad hoc parameters.

5. **Lognormal priors centred at spec defaults**: All utility weights are strictly positive. Lognormal priors encode this constraint naturally and centre the posterior at expert-set defaults when data is uninformative. Prior SD = 1.0 gives ~2.7x range per SD on the ratio scale.

6. **w_removal > w_remove_ceo_overwhelming ordering constraint**: Enforced via `w[5] = w[6] + delta_removal` where delta > 0. The base cost of removing the CEO must exceed the relief from removing after an overwhelming vote.

7. **Cutpoint reparameterisation (tanh + inv_logit)**: Prevents numerical overflow/underflow that caused divergences with naive ordered cutpoints. Base location bounded to [-3, 3], gaps bounded to [0.25, 2.25].

8. **Non-centred scenario random effects**: Standard technique to avoid divergences in hierarchical models. `z_scenario ~ N(0,1)`, RE = `sigma_scenario * z_scenario`.

9. **Loss aversion anchored at lambda = 2.25**: From Kahneman & Tversky (1992). Not estimated — tested diagnostically. The CAR term is the only component with gain/loss asymmetry, so lambda_LA is not separately identifiable from the Likert data.
