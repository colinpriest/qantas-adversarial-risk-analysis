# ARA Board Utility Estimation: Experimental Design and Analysis Specification v2

## 1. Overview

This specification describes an experimental pipeline for estimating the magnitude parameters of the Qantas Board utility function using LLM stakeholder simulation. The pipeline produces posterior distributions over the 13 free utility weight parameters, anchored to the two data-derived parameters (w_CAR = 15.0, w_cost = 15.0) via a jointly estimated softmax choice model.

The pipeline has six stages:

1. **Scenario generation** — construct a structured battery of game-state scenarios
2. **LLM elicitation** — obtain chain-of-thought reasoning and action probability distributions from the LLM
3. **Data storage and preprocessing** — parse and validate LLM outputs into a structured dataset
4. **Choice model estimation** — recover utility weights via maximum likelihood under a softmax model
5. **Behavioural diagnostics** — test for non-linearity, loss aversion, optimism bias, self-assessment bias, and the Ikea effect
6. **Adaptive refinement loop** — target additional scenarios at high-uncertainty parameters

---

## 2. Assumptions and Model Structure

### 2.1 Utility Function (Baseline: Additive Linear)

The Board utility function over a terminal outcome Z is:

```
U(Z; w) = -w1·I_early
         - w2·(V - 0.25)²₊
         - w3·I_over
         - w4·V·I_strike
         + w_CAR_pos·max(CAR, 0)·I_review      [ANCHORED via w_CAR = 15.0]
         - w_CAR_neg·max(-CAR, 0)·I_review      [ANCHORED via loss aversion]
         - w_cost·C_direct·I_review              [FIXED = 15.0]
         - w7·(I_d1sack + I_drevsack + I_drevpostsack)
         - w8·I_ceo_removed_involuntary
         - w9·I_over
         - w10·I_strike_ceo_present
         - w11·I_strike_ceo_present
         - w12·I_over_d1_minimal
         - w13·I_strike_d1_minimal
         - w14·I_strike_ceo_present
         - w15·I_adverse_review_ceo_present
```

The CAR term is split into gain and loss components to accommodate loss aversion (see Section 2.2). Parameters w1–w4, w7–w15 are free (13 parameters). w_cost = 15.0 is fixed. w_CAR_pos and w_CAR_neg are constrained via the loss aversion anchor (see Section 2.2).

Note: terms w10, w11, w14 share the same indicator (strike ∧ CEO_present_at_end). Their individual magnitudes are not separately identified from choice probabilities alone without scenarios that isolate each channel. See Section 3.2.

### 2.2 Loss Aversion Modification

The CAR term is the only utility component with a genuine gain/loss asymmetry: CAR can be positive or negative. Standard prospect theory (Kahneman and Tversky 1992) predicts that losses are weighted more heavily than equivalent gains by a factor λ_LA.

**Anchor value**: λ_LA = 2.25, from Kahneman and Tversky (1992) cumulative prospect theory. The published range across subsequent meta-analyses is approximately 1.5 to 2.5 for individual decision-makers; for institutional actors (boards, committees) some research suggests lower values in the range 1.5 to 1.8. The production default uses the K-T point estimate; sensitivity analysis over the full range is specified in Section 9.4.

**Implementation**: 

```
w_CAR_pos = w_CAR / ((1 + λ_LA) / 2)    [normalised so expected contribution is preserved]
w_CAR_neg = λ_LA · w_CAR_pos
```

Given w_CAR = 15.0 and λ_LA = 2.25:
- w_CAR_pos ≈ 9.23
- w_CAR_neg ≈ 20.77

This means the Board is approximately 2.25x more sensitive to a negative review CAR than to an equivalent positive one. The loss aversion modification does not introduce a new free parameter — λ_LA is anchored from the literature. It does, however, change the structure of the choice model and must be propagated through all EU computations.

**Reference point**: The reference point for the CAR gain/loss split is CAR = 0 (no market reaction). A positive CAR is a gain relative to no reaction; a negative CAR is a loss. This is consistent with the Board's ex-ante expectation that commissioning a review is neutral in expectation (review is undertaken for governance reasons, not market timing).

### 2.3 Choice Model

For a decision node with feasible action set A and game state s, the Board selects action a with probability:

```
P(a | s; w, λ) = exp(λ · U(a, s; w)) / Σ_{a' ∈ A} exp(λ · U(a', s; w))
```

where U(a, s; w) is the expected utility of taking action a given state s, integrated over downstream stochastic outcomes using the distributions specified in the ARA engine. λ is the rationality (inverse temperature) parameter estimated jointly with the free weights, identified by fixing w_CAR = 15.0.

### 2.4 Additive Separability Assumption

The baseline utility function assumes full additive separability: when multiple penalty terms are simultaneously active, they stack without interaction. This assumption is tested empirically in the behavioural diagnostics (Section 8). No interaction terms are pre-specified to avoid overfitting.

### 2.5 Behavioural Utility Extensions: Theoretical Basis

The following behavioural effects are tested as departures from the baseline additive linear model. None are pre-imposed on the utility function; all are treated as diagnostics that may motivate model extensions if confirmed.

**Loss aversion**: Already incorporated structurally in Section 2.2. Diagnostic test in Section 8.1.

**Non-linearity**: The utility function is linear in the weight parameters but may be non-linear in the state variables. The quadratic vote penalty (term 2) is the only pre-specified non-linearity. Additional non-linear effects — diminishing marginal disutility as multiple penalties accumulate, threshold effects near indicator boundaries — are tested in Section 8.2.

**Optimism bias**: The Board may systematically underestimate the probability of adverse outcomes (adverse review, high vote, second strike). This is a directional bias distinct from overconfidence (which concerns precision). In the utility function, optimism bias manifests as the Board behaving as if p(adverse) is lower than the Beta(10,5) prior mean of 0.667. Diagnostic test in Section 8.3.

**Self-assessment bias**: The Board may rate the quality of its own governance decisions and practices higher than external assessors would. In this model, self-assessment bias could cause the Board to underestimate the probability of adverse findings specifically for reviews it initiated, relative to externally mandated assessments. Distinct from optimism bias: self-assessment bias is about evaluation of self-produced work, not general outcomes. Diagnostic test in Section 8.4.

**Ikea effect**: People place higher value on outcomes they have invested effort in creating. For the Board, this could manifest as: (a) higher disutility from removal of a CEO the Board appointed and defended, relative to an inherited CEO; (b) higher perceived quality of a governance review the Board commissioned, potentially causing the Board to underreact to adverse findings from its own review. The Ikea effect is operationalised through scenario design that varies the Board's prior investment in the outcome being evaluated. Diagnostic test in Section 8.5.

---

## 3. Scenario Design

### 3.1 State Vector

Each scenario is fully specified by a state vector S with the following fields:

| Field | Type | Values |
|---|---|---|
| decision_node | categorical | D1, D_rev, D_rev_post |
| ceo_status_at_start | binary | present, resigned_early |
| ceo_appointment | categorical | appointed_by_current_board, inherited | [for Ikea effect] |
| d1_action | categorical | minimal, review, ceo_transition |
| review_origin | categorical | board_initiated, externally_mandated, N/A | [for Ikea/self-assessment] |
| vote_outcome_V | float | [0, 1] |
| strike | binary | derived: V ≥ 0.25 |
| overwhelming | binary | derived: V ≥ 0.50 |
| review_commissioned | binary | true, false |
| review_adverse | binary | true, false, N/A |
| car_outcome | float | decimal CAR, or N/A |
| car_sign | categorical | gain, loss, N/A | [derived, for loss aversion scenarios] |
| ceo_present_at_end | binary | determined by action being evaluated |

### 3.2 Decomposed Identification Scenarios

For each free parameter, at least one minimal scenario pair isolates that parameter by holding all other active indicators at zero or constant.

| Target Parameter | Isolation Condition | Node | Notes |
|---|---|---|---|
| w1 (early departure) | CEO resigned early vs present, no other events | Pre-game | |
| w2 (vote penalty) | V varies within [0.25, 0.60], CEO removed | D1 | Eliminates strike-ceo_present penalties |
| w3 (overwhelming) | V crosses 0.50, CEO removed, d1≠minimal | D1 | Isolate from w12, w13 |
| w4 (spill risk) | Strike present, V varies, CEO removed at end | D1 | Eliminates w10, w11, w14 |
| w7 (implementation) | All else constant, sack vs no-sack | D_rev | |
| w8 (CEO loss) | CEO removed vs not, no strike | D_rev | |
| w9 (rep spill) | Overwhelming, CEO removed | D1 | Additive with w3 |
| w10+w11+w14 (joint) | Strike, CEO present, not overwhelming | D_rev | See note below |
| w12 (d1 liability) | Overwhelming, d1=minimal, CEO removed | D1 | |
| w13 (legal d1) | Strike not overwhelming, d1=minimal, CEO removed | D1 | |
| w15 (adverse review) | Review commissioned, adverse, CEO present vs removed | D_rev_post | |

**Note on w10/w11/w14 joint identification**: These three parameters share the indicator (strike ∧ CEO_present_at_end). Separation requires the natural language prompt to contextualise each channel distinctly: regulatory/ASIC exposure (w11), board spill mechanism (w10), and corporate legal/class action exposure (w14). This is the one parameter group where LLM contextual reasoning must carry identification weight. The system prompt (Section 4.1) must describe these three mechanisms with sufficient specificity. Flag this limitation explicitly in the paper.

**Note on w3/w9**: Both trigger on I_over. Separation requires scenarios where overwhelming occurs but CEO is removed (activates w3+w9 but eliminates w10/w11/w14) vs where overwhelming occurs with CEO retained (all terms active). At least 4 scenarios varying this combination are required.

### 3.3 Scenario Battery Structure

The initial battery is organised in four tiers:

**Tier 1: Identification scenarios (minimum 30)**
Decomposed single-parameter isolation scenarios per Section 3.2. Primary identification data.

**Tier 2: Joint scenarios (minimum 20)**
Realistic multi-penalty scenarios. Used for additivity testing and validation.

**Tier 3: Behavioural probe scenarios (minimum 20)**
Designed specifically to test the five behavioural hypotheses in Section 2.5. These include symmetric gain/loss CAR scenarios (loss aversion), scenarios varying ceo_appointment and review_origin fields (Ikea effect, self-assessment bias), and high-penalty-count scenarios (non-linearity, diminishing marginal disutility). Details in Section 8.

**Tier 4: Historical calibration scenario (exactly 1)**
The actual Qantas AGM November 2023 scenario, fully specified. Withheld from all estimation steps and used only for out-of-sample validation. Parameter estimation must be completed before this scenario is evaluated (pre-registration principle).

### 3.4 Vote Outcome Grid

For scenarios where V is a continuous input, use:

V ∈ {0.10, 0.20, 0.26, 0.30, 0.40, 0.50, 0.60, 0.75, 0.83}

The values 0.26 and 0.83 anchor to the Qantas case (near-miss counterfactual and actual outcome respectively).

### 3.5 CAR Outcome Grid for Loss Aversion Scenarios

For Tier 3 loss aversion probe scenarios, use symmetric gain/loss pairs at:

CAR ∈ {-0.14, -0.08, -0.05, -0.03, -0.01, 0.00, +0.01, +0.03, +0.05, +0.08, +0.14}

The negative anchors correspond to the Star Entertainment and BOQ case study observations. Each positive value should be paired with its negative counterpart to enable within-pair comparison.

---

## 4. LLM Elicitation Protocol

### 4.1 Prototype Model

Use **gpt-4o-mini** for the prototype. Full experiment adds additional models (see Section 4.4).

### 4.2 System Prompt (Fixed Across All Scenarios)

The system prompt has four sections, held constant across all scenario queries.

**Section A: Board persona — multi-director deliberation framing**

Describe the Qantas Board as a group of directors in active deliberation, not as a single unified actor. Name the actual board members as of late 2023 (chair, independent directors). Each director brings distinct professional backgrounds and risk tolerances. Prompt the LLM to reason as if it is observing the boardroom discussion: directors raise concerns, weigh competing considerations, and reach a majority position. The probability output should reflect the board's likely collective decision, accounting for internal disagreement where it exists.

This framing is used for two reasons: it produces richer chain-of-thought reasoning (more realistic deliberation), and it captures within-board heterogeneity that a single-actor framing suppresses. Do not assign fixed positions to individual directors — allow the LLM to reason about the deliberation dynamics.

**Section B: Legal and regulatory context**

Describe: the two-strikes rule (ss 250U–250W Corporations Act 2001), ASIC enforcement powers for breach of directors' duties (s 180), class action environment for ASX companies, ASA's role and typical mobilisation patterns, proxy advisor influence (ISS, Glass Lewis). Keep factual and non-leading. Describe the three channels of legal/regulatory exposure separately and distinctly: (1) personal director regulatory liability via ASIC, (2) board spill mechanism under two-strikes rule, (3) corporate-level legal exposure via class actions and company-level ASIC penalties. This separation is necessary for w10/w11/w14 identification.

**Section C: Historical context**

Describe: the ghost flights matter, the ACCC proceedings and settlement, the Senate inquiry, the sequence of events leading to the AGM. Stop before the AGM outcome. Do not describe the vote result.

**Section D: Elicitation instructions**

Instruct the LLM to:
1. Reason step by step as a boardroom deliberation, with directors raising distinct concerns
2. Consider all feasible actions and their likely consequences for each of the three legal/regulatory exposure channels
3. After reasoning, output a structured block containing:
   - Probability assigned to each feasible action (must sum to 1.0, expressed to two decimal places)
   - For each action: a one-sentence justification from the Board's perspective
   - A structured factor importance rating (see Section 4.3)
   - Free-form commentary on the deliberation (no length limit)

### 4.3 Structured Factor Importance Rating

After the free-form reasoning and before the probability output, the LLM must rate a fixed set of factors on a 1–5 scale (1 = not a significant consideration, 5 = decisive consideration).

**Factor list** (10 factors):

1. Risk of a second strike at the next AGM
2. Personal regulatory liability of individual directors (ASIC)
3. Corporate legal exposure (class actions, ASIC company penalties)
4. CEO relationship and institutional knowledge loss
5. Market reaction to governance action
6. Direct costs of governance reform
7. Reputational contagion to directors' other board positions
8. Implementation complexity of the chosen action
9. Shareholder activist escalation risk
10. Board cohesion and internal deliberation costs

**Ordering**: Randomise the presentation order of the 10 factors for each LLM call independently. Generate a random permutation of [1..10] at call time and present factors in that order. Record the permutation used in the elicitation results (see Section 5.2). This avoids primacy and recency effects, prevents experimenter framing bias, and enables testing for order effects as a diagnostic. Do not use prior-expectation ordering as this would introduce experimenter bias into the elicitation.

**Placement**: Factors are rated before the probability output within the same LLM call. The chain-of-thought reasoning should inform the ratings; the ratings should in turn constrain the probability assignment. This ordering ensures the factor ratings serve as an intermediate reasoning step (consistent with chain-of-thought prompting research) rather than a post-hoc rationalisation.

### 4.4 Repetition and Variation Protocol

**Prototype (single model):**
- Model: gpt-4o-mini
- N_rep = 10 repetitions per scenario, fresh context each time
- Random factor ordering generated independently per call
- Record seed/call ID for reproducibility

**Full experiment (added after prototype validation):**
- N_prompt = 3 prompt framings per scenario (same content, different natural language expression)
- N_model ≥ 3 (gpt-4o-mini, Claude Sonnet, one additional)
- Total observations per scenario = N_rep × N_prompt × N_model

### 4.5 Output Format Requirements

The LLM must output a structured block delimited by XML-style tags. The parser must:
- Extract the probability vector and validate it sums to 1.0 (tolerance 0.01)
- Extract the factor importance ratings as a 10-element integer vector along with the presented order permutation
- Extract free-form commentary as a string
- Flag and log any response failing format validation
- Store raw output regardless of parse status (never discard)

---

## 5. Data Storage Format

### 5.1 Scenario Registry

`scenarios.csv` — one row per scenario:

| Column | Type | Description |
|---|---|---|
| scenario_id | string | S{tier}{number} e.g. S1_001 |
| tier | int | 1, 2, 3, or 4 |
| target_parameter | string | Tier 1: parameter being isolated; Tier 3: behavioural hypothesis |
| decision_node | string | D1, D_rev, D_rev_post |
| state_vector | JSON | Full state vector including behavioural fields |
| feasible_actions | JSON | List of feasible action codes |
| prompt_text | string | Full scenario prompt text |
| created_at | timestamp | |

### 5.2 Elicitation Results

`elicitation_results.csv` — one row per LLM call:

| Column | Type | Description |
|---|---|---|
| result_id | string | Unique identifier |
| scenario_id | string | FK to scenarios.csv |
| model | string | LLM model identifier |
| prompt_variant | int | Prompt framing index |
| seed | int | Random seed / call ID |
| factor_order | JSON | 10-element permutation of factor indices as presented |
| raw_output | string | Full LLM response |
| parse_status | string | success, format_error, probability_error |
| prob_vector | JSON | Parsed probability vector keyed by action code |
| factor_ratings | JSON | 10-element integer vector in canonical factor order (re-mapped from presented order) |
| commentary | string | Parsed free-form text |
| called_at | timestamp | |

Note: factor_ratings is always stored in canonical order (factor 1 through 10) regardless of presentation order, to simplify downstream analysis. The raw presentation order is preserved in factor_order.

### 5.3 Computed Dataset

`estimation_dataset.csv` — one row per (scenario_id, model, prompt_variant):
- Probability vector: mean across seeds
- Seed variance: mean variance across action probabilities
- Factor ratings: mean across seeds in canonical order
- Factor rating variance: element-wise variance across seeds
- Include only rows with parse_status = success for ≥ 7 of 10 seeds

---

## 6. Choice Model Estimation

### 6.1 Expected Utility Computation

For each scenario i and each feasible action a, compute U(a, s_i; w) by:

1. Determining the terminal outcome distribution Z(a, s_i) under action a
2. Integrating over stochastic components (V, CAR, C_direct) via Monte Carlo with M = 500 samples using ARA engine distributions
3. Applying the utility function formula from Section 2.1 with the loss aversion modification from Section 2.2

Pre-compute and cache EU[i, a] for the initial parameter vector. Because the utility function is linear in w (given fixed stochastic draws), EU[i, a] can be decomposed as:

```
EU[i, a; w] = Σ_k w_k · φ_k[i, a]
```

where φ_k[i, a] is the Monte Carlo average of the k-th basis function (indicator or continuous term) for scenario i, action a. Compute φ_k[i, a] once and reuse across all optimisation iterations. This reduces the per-iteration cost to a matrix-vector product.

### 6.2 Objective Function

Let p_LLM[i, a] be the mean elicited probability vector for scenario i. The objective is to minimise cross-entropy:

```
L(w, λ) = -Σ_i Σ_a p_LLM[i,a] · log(p_model[i,a; w,λ])
```

where p_model[i, a; w, λ] = softmax(λ · EU[i, a; w]).

### 6.3 Constraints and Identification

Optimisation variables: {w1, w2, w3, w4, w7, w8, w9, w10, w11, w14, w12, w13, w15, λ} — 14 variables.

Constraints:
1. All weight parameters ≥ 0
2. λ > 0
3. w_CAR_pos and w_CAR_neg are derived from w_CAR = 15.0 and λ_LA = 2.25 (not optimisation variables)
4. w_cost = 15.0 (fixed)

Identification: w_CAR = 15.0 pins the absolute utility scale, which jointly identifies λ.

Record the condition number of the Hessian at the solution. If condition number > 1000, apply ridge penalty α = 0.01 to the Hessian diagonal before inversion and report this.

### 6.4 Optimisation Algorithm

Use scipy.optimize.minimize with L-BFGS-B. Multiple starting points:
- Starting point 1: all weights at spec values, λ = 1.0
- Starting points 2–10: weights drawn from Normal(spec_value, 0.3·spec_value) clipped at zero, λ from LogNormal(0, 0.5)

Take global minimum across all converged solutions.

### 6.5 Uncertainty Quantification

Compute Hessian at optimum via finite differences. Invert for asymptotic covariance matrix. Also compute bootstrap standard errors (B = 500 scenario resamples). Where Hessian-based and bootstrap SEs disagree by more than 50%, prefer the bootstrap estimate and flag this in outputs.

### 6.6 Rescaling Check

Verify that the model-implied utility contribution of CAR = -5% under a commissioned review equals -1.038 (= 20.77 × 0.05). This confirms that the loss aversion modification and anchor constraint are correctly implemented.

---

## 7. Interaction Effects Testing

### 7.1 Residual Computation

For each scenario i:
```
r[i, a] = p_LLM[i, a] - p_model[i, a; w*, λ*]
```

### 7.2 Systematic Pattern Detection

Group residuals by:
- Number of simultaneously active penalty terms (1, 2, 3, 4+)
- Whether strike and overwhelming are simultaneously active
- Whether w10/w11/w14 joint condition is active alongside other major penalties

Test whether mean residual per group differs from zero (t-test). A significant negative residual for 4+ active penalty groups is consistent with diminishing marginal disutility. A significant positive residual is consistent with superadditivity.

### 7.3 Interaction Term Addition

If systematic patterns are detected, add pairwise interaction terms w_ij · I_i · I_j for detected pairs. Re-estimate and compare via AIC. Only add terms supported by residual analysis.

---

## 8. Behavioural Diagnostics

### 8.1 Loss Aversion Diagnostic

**Scenario design (Tier 3)**: For each |CAR| value in {0.01, 0.03, 0.05, 0.08, 0.14}, construct a matched pair of scenarios identical in all respects except the sign of the CAR outcome. All other state variables are held at values that make the review decision the relevant choice.

**Test**: For each matched pair, compute the ratio of choice probability sensitivity:

```
sensitivity_ratio = |ΔP(review | CAR=-x)| / |ΔP(review | CAR=+x)|
```

relative to the no-review baseline. Under the K-T loss aversion model, this ratio should be approximately λ_LA = 2.25.

**Model extension trigger**: If the observed mean sensitivity ratio is significantly different from 1.0 (t-test, p < 0.05), the loss aversion modification in Section 2.2 is confirmed. If the estimated ratio differs significantly from λ_LA = 2.25, update λ_LA to the estimated value and report the deviation from the K-T anchor. If the ratio is not significantly different from 1.0, revert to a single w_CAR parameter and report this as a null finding.

**Sensitivity analysis**: Re-estimate the full model with λ_LA ∈ {1.5, 1.75, 2.0, 2.25, 2.5} and report the sensitivity of all free parameter estimates to the loss aversion anchor.

### 8.2 Non-Linearity Diagnostic

**Continuous non-linearity (vote penalty)**: The baseline model specifies a quadratic penalty (V-0.25)². Using the vote outcome grid scenarios (Tier 1, w2 identification), fit the following alternative functional forms to the LLM-elicited choice probabilities and compare via AIC:

- Quadratic: (V - 0.25)²
- Linear above threshold: (V - 0.25)
- Cubic: (V - 0.25)³
- Piecewise linear with break at 0.50: separate slopes for [0.25, 0.50] and [0.50, 1.0]
- Log-linear: log(V / 0.25) for V > 0.25

If any alternative form produces substantially lower AIC (ΔAIC > 4), report this as evidence against the quadratic specification and adopt the superior form.

**Diminishing marginal disutility**: Using Tier 2 joint scenarios grouped by number of active penalty terms, test whether the marginal disutility of adding the k-th active penalty is decreasing in k. Compute the implied disutility increment per additional active term and regress on k. A significant negative slope is evidence for diminishing marginal disutility, suggesting a concave transformation of the summed penalties. If confirmed, test the following modification:

```
U_transformed = -f(|U_penalties|) + U_gains
```

where f is a concave function (e.g., square root or log). Compare via AIC.

### 8.3 Optimism Bias Diagnostic

**Definition in this context**: The Board systematically assigns lower probability to adverse outcomes than the objective priors imply.

**Test**: Design 10 Tier 3 scenarios where the probability of adverse review outcome is made explicit in the prompt (e.g., "Based on comparable ASX governance reviews, approximately 67% of reviews have produced adverse or neutral findings"). Compare LLM-elicited choice probabilities from these scenarios with matched scenarios where the adverse probability is left implicit (the LLM must infer it from context).

If the explicit-probability scenarios produce systematically lower P(commission review) than the implicit scenarios — consistent with the LLM assigning higher subjective p(adverse) when the base rate is not stated — this is evidence that the LLM's default representation of the Board is optimistically biased about review quality.

Additionally, extract the Board's implied p(adverse) from the chain-of-thought commentary using a secondary LLM call that asks: "Based on the deliberation above, what probability did the Board assign to adverse review findings?" Compare the distribution of extracted values to Beta(10, 5) mean = 0.667.

### 8.4 Self-Assessment Bias Diagnostic

**Definition in this context**: The Board assigns higher quality to governance outcomes it has produced relative to identical outcomes produced externally.

**Scenario design**: Construct matched Tier 3 pairs where the review outcome (e.g., "findings indicate significant governance failures in executive accountability") is held constant, but the review origin varies:
- Condition A: "The Board commissioned an independent review in February 2024"
- Condition B: "ASIC commissioned an independent review in February 2024"

Under self-assessment bias, the Board should respond more defensively (lower P(sack CEO), lower urgency in response) when the adverse findings come from its own review than from a mandatorily imposed review.

**Test**: Compare P(sack CEO | adverse, board-initiated) vs P(sack CEO | adverse, externally-mandated) across matched pairs. A significant difference in the predicted direction is evidence of self-assessment bias. If confirmed, consider adding a modifier to w_adverse that reduces the penalty when the Board itself commissioned the review (encoding the bias structurally).

### 8.5 Ikea Effect Diagnostic

**Definition in this context**: The Board places higher value on outcomes associated with its own prior investment of effort and commitment.

**Two operationalisations**:

*Operationalisation A — CEO appointment*: Vary the ceo_appointment field (appointed_by_current_board vs inherited). Under the Ikea effect, the Board should assign higher disutility to removing a CEO it appointed and publicly defended, reflecting sunk effort and psychological ownership. Test: compare P(sack CEO) across matched pairs varying only ceo_appointment. If significant, add a modifier to w_loss or w8 that is higher for inherited CEOs (less Ikea attachment) than for Board-appointed CEOs.

*Operationalisation B — Review ownership*: Vary whether the review was Board-initiated vs externally mandated. Under the Ikea effect, the Board should place higher weight on its own review findings (even adverse ones) because it invested effort in designing the review terms of reference. This is the reverse of self-assessment bias (which predicts discounting adverse own-review findings) — the two effects make opposite predictions and their relative magnitude is an empirical question. Test: compare the urgency of response to adverse findings across matched pairs varying only review origin.

**Interaction between 8.4 and 8.5**: Self-assessment bias (discount own adverse findings) and the Ikea effect (amplify own review salience) are competing hypotheses about the same scenario variation. Design a scenario triplet:
- (i) Board-initiated review, adverse findings
- (ii) Externally-mandated review, adverse findings
- (iii) Board-initiated review, positive findings

The pattern of choice probabilities across these three conditions allows separation of the two effects. Report the pattern and its interpretation regardless of outcome.

### 8.6 Factor Rating Order Effect Test

For each factor, regress the factor importance rating on its position in the presented order (1 = presented first, 10 = presented last). A significant positive coefficient indicates recency effects; a significant negative coefficient indicates primacy effects. If order effects are detected for any factor, include position as a covariate in the factor rating validation regression (Section 9.2).

---

## 9. Adaptive Refinement Loop

### 9.1 Trigger Condition

After initial estimation, identify parameters where bootstrap SE > 30% of point estimate.

### 9.2 Scenario Generation for High-Uncertainty Parameters

For each high-uncertainty parameter k:
1. Identify isolation conditions from Section 3.2
2. Generate 10 additional Tier 1 scenarios varying the relevant state variable
3. Elicit and re-estimate on augmented dataset

### 9.3 Stopping Rule

Stop when:
- All bootstrap SEs < 30% of point estimates, or
- Total scenarios > 150, or
- Three adaptive iterations completed

---

## 10. Validation

### 10.1 Within-Sample Fit

Report mean KL divergence per scenario. Target < 0.05 nats. Report 5 worst-fitting scenarios for qualitative inspection.

### 10.2 Factor Rating Validation

For each of the 10 factors, regress mean factor importance rating (across seeds) on the model-implied expected utility contribution of the corresponding term. Report coefficient, standard error, and R². Significant positive coefficients are confirmatory. Adjust for order effects if detected in Section 8.6.

### 10.3 Historical Scenario Validation (Tier 4)

Present the Qantas AGM November 2023 scenario. The model should assign highest probability to D1_review at D1 and Drev_sack_ceo at D_rev. Record predicted probabilities and rank of historically observed action at each decision node. This is the primary external validity result.

### 10.4 Behavioural Robustness Check

Re-estimate the model with:
- Loss aversion active vs suppressed (λ_LA = 1.0)
- Interaction terms included vs excluded
- Tier 3 behavioural scenarios included vs excluded from estimation

Report the change in all free parameter estimates across these variants. Parameters that are stable across variants are robust; parameters that shift substantially are sensitive to the behavioural model specification.

### 10.5 Anchor Sensitivity Analysis

Re-estimate with w_CAR ∈ {12.0, 15.0, 18.0} and λ_LA ∈ {1.5, 2.0, 2.25, 2.5}. Report sensitivity of all free parameter estimates.

---

## 11. Outputs

| Output | Format | Description |
|---|---|---|
| parameter_estimates.csv | CSV | Point estimates, Hessian SE, Bootstrap SE for all 14 parameters |
| covariance_matrix.csv | CSV | Full 14×14 covariance matrix |
| scenario_fit.csv | CSV | Per-scenario KL divergence, residuals, model vs LLM probabilities |
| interaction_test_results.csv | CSV | Residual group means, t-statistics, AIC comparison |
| behavioural_diagnostics.csv | CSV | Results of Sections 8.1–8.6: test statistics, effect sizes, model extension decisions |
| validation_results.json | JSON | Factor regression results, order effect tests, historical scenario predictions |
| updated_governance_spec.xlsx | XLSX | Original spec with free parameter columns updated to estimated values, SE columns added, behavioural modification flags |

---

## 12. Implementation Architecture

### 12.1 LLM API Integration

**Library**: Use the `instructor` library for all LLM API calls. instructor wraps the underlying API client (openai, anthropic, etc.) and enforces structured output via Pydantic model validation. All LLM responses must be defined as Pydantic models before implementation begins — no raw JSON parsing elsewhere in the codebase.

**Prototype model**: `gpt-4o-mini` via the OpenAI API.

### 12.2 Pydantic Schema Definitions

Define the following Pydantic models. All categorical fields must use `enum` types so that instructor's validation rejects any value not in the permitted set before it reaches application logic.

```
ActionCode(str, Enum):
    D0_MINIMAL = "D0_minimal"
    D1_REVIEW = "D1_review"
    D3_CEO_TRANSITION = "D3_ceo_transition"
    DREV_NO_ACTION = "Drev_no_action"
    DREV_COMMISSION_REVIEW = "Drev_commission_review"
    DREV_SACK_CEO = "Drev_sack_ceo"

DecisionNode(str, Enum):
    D1 = "D1"
    D_REV = "D_rev"
    D_REV_POST = "D_rev_post"

ParseStatus(str, Enum):
    SUCCESS = "success"
    FORMAT_ERROR = "format_error"
    PROBABILITY_ERROR = "probability_error"
    TOKEN_LIMIT = "token_limit"
    REPAIRED = "repaired"

ActionProbability(BaseModel):
    action: ActionCode
    probability: float  [ge=0.0, le=1.0]
    justification: str

FactorRating(BaseModel):
    factor_index: int  [ge=1, le=10]
    rating: int  [ge=1, le=5]

ElicitationResponse(BaseModel):
    prob_vector: list[ActionProbability]
    factor_ratings: list[FactorRating]  [len=10]
    commentary: str

    @validator: sum of probabilities in prob_vector must equal 1.0 within tolerance 0.01
    @validator: all ActionCodes in prob_vector must match the feasible_actions for this scenario
    @validator: all factor indices 1..10 must appear exactly once in factor_ratings
```

instructor handles retry logic for schema validation failures internally (up to 3 retries by default). Set `max_retries=3` on the instructor client.

### 12.3 JSON Repair

Before passing LLM output to Pydantic validation, apply a JSON repair step using the `json-repair` library. This handles common LLM formatting failures: trailing commas, unquoted keys, truncated output, markdown code fences wrapping the JSON. The repair step must:

1. Strip any leading/trailing markdown fences (` ```json ` ... ` ``` `)
2. Pass the stripped string to `json_repair.repair()`
3. Attempt Pydantic validation on the repaired string
4. If validation succeeds after repair, set parse_status = "repaired" and log a warning
5. If validation still fails after repair, set parse_status = "format_error", store raw output, and do not raise — continue to next call

All repaired responses are included in the dataset but flagged. Report the repair rate per model and prompt variant in the dashboard.

### 12.4 Rate Limiting and Retry Logic

Implement retry with exponential backoff for all API calls. Do not rely solely on instructor's built-in retry, which handles schema errors but not HTTP errors.

Retry schedule:
- HTTP 429 (rate limit): wait = min(2^attempt × base_wait, 60s), base_wait = 1s, max_attempts = 6
- HTTP 500/503 (server error): wait = min(2^attempt × 2s, 120s), max_attempts = 4
- HTTP 400 (bad request): do not retry — log error and mark call as failed

Use a `tenacity` decorator on the API call function with the above schedule. Log each retry attempt at WARNING level including wait duration and attempt number.

### 12.5 Token Limit Handling

After each API call, check whether the response was truncated due to token limits (finish_reason == "length" in the OpenAI response). If truncated:

1. Set parse_status = "token_limit"
2. Increment a run-level counter `token_limit_count`
3. Log at ERROR level: scenario_id, model, prompt length in tokens, response length in tokens
4. If `token_limit_count` exceeds 10 during a run, raise a `TokenLimitRunError` exception that terminates the run with the message: "Run aborted: {count} token limit exceedances detected. The system prompt or scenario prompts may be too long for the selected model. Consider compressing the system prompt or switching to a model with a larger context window. Affected scenario IDs: {list}."

Track the prompt token count per call to identify which scenarios are approaching the limit before truncation occurs. Log a WARNING if a prompt exceeds 80% of the model's documented context window.

### 12.6 Token Usage and Cost Tracking

After every API call, extract token usage from the response metadata (prompt_tokens, completion_tokens, total_tokens). Accumulate into a run-level cost tracker:

```
TokenUsage(BaseModel):
    scenario_id: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float  [computed from per-model price table]

RunCostSummary(BaseModel):
    total_calls: int
    successful_calls: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    total_cost_usd: float
    cost_by_model: dict[str, float]
    cost_by_tier: dict[str, float]
    cost_by_scenario: dict[str, float]
```

Maintain a price table as a config file (not hardcoded): model name → (cost_per_1k_prompt_tokens, cost_per_1k_completion_tokens). Update the price table without code changes.

Report full cost summary in the dashboard (see Section 12.8).

### 12.7 Concurrency

**API calls**: Use `concurrent.futures.ThreadPoolExecutor` with `max_workers=10`. All scenarios within a tier can be dispatched concurrently subject to this limit. Submit all calls for a tier as futures, collect results as they complete (use `as_completed`), and update the progress bar on each completion.

**CPU-bound calculations** (EU computation, optimisation, bootstrap resampling): Use `concurrent.futures.ProcessPoolExecutor` with `max_workers = os.cpu_count() - 1`. This applies to:
- EU basis function computation (φ_k[i,a] matrix, Section 6.1): parallelise over scenarios
- Bootstrap resampling (Section 6.5): parallelise over bootstrap samples
- Sensitivity analysis reruns (Section 10.5): parallelise over anchor value combinations

Do not mix thread and process pools in the same pipeline stage. API calls use threads; calculations use processes.

### 12.8 Caching

Cache all API call results to disk. Cache key construction:

```python
cache_key = sha256(
    json.dumps({
        "system_prompt": system_prompt_text,
        "scenario_prompt": scenario_prompt_text,
        "model": model_id,
        "seed": random_seed,
        "temperature": temperature,
        "cache_version": CACHE_VERSION  # integer constant in config
    }, sort_keys=True).encode()
).hexdigest()
```

Cache storage: one JSON file per cache entry in a `cache/` directory, filename = `{cache_key}.json`. Contents: full request parameters + raw response + parsed result + timestamp.

Cache lookup: before making an API call, check whether `cache/{cache_key}.json` exists. If it does, load and return the cached result without making an API call. Log at DEBUG level: "Cache hit: {cache_key[:8]}..."

Cache invalidation: increment `CACHE_VERSION` in config to invalidate the entire cache (e.g., when the system prompt changes). Do not delete cache files — old versions remain on disk and are simply unreachable via new keys.

Cache statistics: track and report hits vs misses per run in the dashboard.

### 12.9 Progress Bars

Use the `tqdm` library. One progress bar per pipeline stage. Requirements:

- Display: scenario count completed / total, elapsed time, estimated time remaining (ETA)
- ETA computation: use tqdm's default rate-based estimate (items/second), which updates dynamically
- Console output discipline: all logging (ERROR, WARNING, INFO) must use `tqdm.write()` rather than `print()` or `logging.StreamHandler` directly. This prevents log messages from overwriting or corrupting the progress bar display. Configure the logging handler to call `tqdm.write` for all levels
- Nested bars: where a stage has sub-stages (e.g., API calls within each tier), use `tqdm(leave=False)` for the inner bar so it disappears on completion and does not clutter the terminal
- Do not show a progress bar for steps completing in under 2 seconds

### 12.10 Text Encoding

**Scope**: applies to all text produced or consumed by the pipeline — LLM outputs, CSV files, JSON files, log files, the HTML dashboard, and all intermediate data structures.

**Standard**: UTF-8 throughout, with ASCII-safe output for all file writes and LLM-derived text fields. The distinction matters: UTF-8 is used as the declared encoding on all file handles (preventing OS-level codec errors on Windows), but LLM-derived text is additionally sanitised to remove or replace characters outside the printable ASCII range (codepoints 32–126) before being stored or rendered.

**Why LLM outputs are the primary risk**: LLM APIs frequently return characters that are legal UTF-8 but cause silent problems when files are moved between operating systems or opened in tools with different default encodings. The most common offenders are:

| Character | Unicode | Common source | Replacement |
|---|---|---|---|
| Smart quotes | U+2018 U+2019 U+201C U+201D | LLM prose | straight quotes ' " |
| Em dash | U+2014 | LLM prose | double hyphen -- |
| En dash | U+2013 | LLM prose | single hyphen - |
| Ellipsis | U+2026 | LLM prose | three dots ... |
| Non-breaking space | U+00A0 | LLM formatting | regular space |
| Bullet | U+2022 | LLM lists | hyphen - |
| Degree / currency symbols | various | LLM numerics | spell out or omit |
| Zero-width space | U+200B | invisible in LLM output | remove |
| BOM | U+FEFF | Windows editors | remove |

**Sanitisation function**: implement a single `sanitise_text(s: str) -> str` function applied to every string field extracted from an LLM response before it is stored in a Pydantic model or written to any file. The function must:

1. Decode bytes to str using UTF-8 with `errors="replace"` if input is bytes
2. Remove BOM (U+FEFF) if present at the start of the string
3. Apply the replacement table above using a single-pass translation (str.translate with a pre-built translation table — do not use repeated str.replace calls)
4. Remove zero-width and other invisible Unicode characters (categories Cf, Cc except tab/newline/carriage return)
5. Normalise Unicode to NFC form (unicodedata.normalize) before the replacement step, to collapse composed and decomposed representations of the same character
6. Replace any remaining non-ASCII characters (codepoints > 126) with their closest ASCII equivalent using the `unidecode` library, falling back to "?" if no equivalent exists
7. Collapse runs of more than one consecutive space to a single space
8. Strip leading and trailing whitespace

Apply `sanitise_text` to: all string fields in `ElicitationResponse`, all commentary strings, all justification strings, all scenario prompt text before caching, and all strings written to CSV or JSON outputs.

**Do not apply** `sanitise_text` to: the raw_output field in elicitation results (this must preserve the original LLM response exactly for debugging), and the HTML dashboard template itself (which is author-controlled).

**File write requirements**:

All file opens must explicitly specify `encoding="utf-8"` and `errors="replace"`:

```python
open(path, "w", encoding="utf-8", errors="replace")
```

Never rely on the OS default encoding (which is cp1252 on Windows, UTF-8 on macOS/Linux). This single omission is the most common source of cross-platform encoding failures.

CSV files: use `csv.writer` with the file handle opened as above. Do not use pandas `to_csv()` without explicitly passing `encoding="utf-8"`.

JSON files: use `json.dumps` with `ensure_ascii=True`. This encodes all non-ASCII characters as `\uXXXX` escape sequences, producing a file that is safe on any platform regardless of the encoding used to open it. Since LLM-derived strings have already been sanitised to ASCII by this point, `ensure_ascii=True` is a safety net rather than a primary mechanism.

Log files: configure the logging FileHandler with `encoding="utf-8"`. On Windows, the default is cp1252, which will raise an exception if any log message contains a non-ASCII character.

**HTML dashboard**: declare `<meta charset="UTF-8">` as the first tag inside `<head>`. Write the HTML file with `encoding="utf-8"`. Since all embedded data strings have been sanitised to ASCII before JSON serialisation, the embedded `RESULTS_DATA` JavaScript variable will be ASCII-safe regardless of the charset declaration.

**Encoding audit at run start**: at pipeline startup, before any API calls, run a short self-test that writes and reads back a file containing a known set of ASCII and near-ASCII characters on the current OS. Log the result at INFO level. If the self-test fails (encoding roundtrip error), abort with an informative error before any work is done.

**Encoding error reporting in the dashboard**: the Cost & Usage tab should include an encoding issues panel reporting: total sanitisation replacements made, count of non-ASCII characters replaced, count of BOM removals, count of zero-width character removals. This provides traceability if an external user notices that quoted text in the dashboard differs slightly from what the LLM produced.

The pipeline produces a single self-contained HTML file (`results_dashboard.html`) designed for two purposes: local monitoring during a run, and sharing with external users after completion. Both use cases are served by the same file with no external dependencies.

**Self-containment requirements (critical for shareability)**:

All assets must be embedded directly in the HTML file. No external dependencies are permitted, because external users may be on restricted networks, the CDN may be unavailable, or the file may be opened years after the run. Specifically:

- **Plotly.js**: download the minified bundle at build time and embed it inline in a `<script>` tag. Do not load from CDN. The Plotly minified bundle is approximately 3.5MB; this is acceptable given the single-file sharing requirement.
- **CSS**: all styles inline in a `<style>` tag. No external stylesheets.
- **Fonts**: use system fonts only (font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif). No Google Fonts or web font loading.
- **Data**: all results data embedded as a JavaScript variable in a `<script>` tag: `const RESULTS_DATA = {...};`. The Pydantic results object is serialised to JSON and injected at render time.
- **Downloadable files**: CSV and JSON outputs listed in Section 11 are embedded as base64-encoded data URIs in `<a href="data:text/csv;base64,...">` download links. This allows external users to download the raw data files directly from the dashboard without needing access to the original run directory.

**In-progress state vs final state**:

The dashboard has two render modes controlled by a field in the results data: `run_status: "in_progress" | "complete" | "aborted"`.

In `in_progress` mode:
- A banner is displayed at the top: "Run in progress — last updated {timestamp}. This page refreshes automatically."
- A `<meta http-equiv="refresh" content="30">` tag is present in the `<head>`
- Tabs with no data yet render a placeholder: "Waiting for data from this stage..."
- The file is overwritten atomically (via temp file + os.replace) after each stage completes

In `complete` mode:
- The banner is absent
- The meta-refresh tag is absent
- All tabs are fully populated
- A header note reads: "Run completed {timestamp}. This file is self-contained and can be shared."

In `aborted` mode:
- A red banner describes the abort reason (e.g. token limit exceedance)
- Data collected up to the abort point is displayed
- The meta-refresh tag is absent

**Critical**: the file distributed to external users must always be the `complete` or `aborted` version. Never share an `in_progress` file. The pipeline should log a clear reminder at run completion: "Dashboard finalised at results_dashboard.html — safe to share."

**Atomic write procedure** (applied on every dashboard update):

```
1. Serialise full results object to JSON string
2. Render complete HTML string from template, embedding JSON and Plotly bundle
3. Write HTML string to results_dashboard.tmp
4. os.replace("results_dashboard.tmp", "results_dashboard.html")
```

Step 4 is atomic on Linux, macOS, and Windows. The browser never reads a partially written file.

**Performance**: The full HTML file will be 4–6MB when Plotly is embedded (dominated by the Plotly bundle at ~3.5MB). JSON serialisation and HTML rendering for a typical results object should complete in under 500ms. If rendering takes longer than 1 second, log a WARNING — this likely indicates the results JSON is unexpectedly large and should be investigated.

**Tab structure**:

| Tab | Contents |
|---|---|
| Overview | Run status banner; run summary (date, model, scenario counts by tier, total cost, cache hit rate); error and abort event log |
| Cost & Usage | Token usage table (sortable by model/tier/scenario); cost breakdown pie chart; cumulative cost line chart over time; rate limit and token limit event log |
| Scenario Battery | Searchable, filterable table of all scenarios with state vectors; filter by tier, decision node, target parameter |
| Elicitation Results | Per-scenario probability distributions (bar charts); seed variance heatmap; parse status breakdown; repair rate by model and prompt variant |
| Parameter Estimates | Table of all 14 parameters: point estimate, Hessian SE, Bootstrap SE, 95% CI; forest plot of estimates with CIs; comparison against original spec values |
| Covariance | Heatmap of the 14×14 correlation matrix; flagged high-correlation pairs (|r| > 0.8) |
| Behavioural Diagnostics | One sub-section per diagnostic (8.1–8.6): test statistic, p-value, effect size, decision (confirmed/null/inconclusive), supporting chart |
| Interaction Effects | Residual group mean chart by number of active penalties; AIC comparison table for baseline vs interaction models |
| Validation | Factor rating regression results (with order effect adjustment if applicable); historical scenario prediction with probability breakdown; anchor sensitivity tornado chart |
| Raw Data | Download links for all output files (CSV/JSON embedded as base64 data URIs); file sizes shown; generated timestamp |

**Chart and table requirements**:
- All Plotly charts: hover tooltips enabled, zoom and pan enabled, download as PNG button visible
- All tables: client-side sort on all columns, filter/search box, pagination for tables exceeding 50 rows
- All charts and tables must render gracefully with null/empty data (placeholder text, no JS errors)
- Colour scheme must be accessible (WCAG AA contrast ratios); do not rely on colour alone to convey information

---

## 13. Open Questions for Implementer

1. **w10/w11/w14 collinearity**: The system prompt must describe the three legal/regulatory exposure channels with sufficient distinctness that the LLM reasons about them separately. Prototype testing should include a qualitative review of chain-of-thought outputs to verify that all three channels appear in the deliberation. If the LLM consistently conflates two channels, the identification strategy for those parameters fails and the spec must be revised.

2. **EU caching**: Pre-compute φ_k[i, a] basis functions (Section 6.1) before beginning optimisation. The full computation should be: (a) load all scenarios, (b) run M=500 MC draws per scenario-action pair, (c) compute and store φ_k matrix, (d) run optimisation using only the φ_k matrix. This avoids any ARA engine calls during optimisation.

3. **Hessian conditioning**: If condition number > 1000 for the joint (w10, w11, w14) submatrix, these three parameters cannot be separately identified from the current scenario battery. In this case, collapse to a single combined weight w_joint = w10 + w11 + w14 for the baseline estimation and note this as a limitation.

4. **Behavioural effect interaction (Sections 8.4 and 8.5)**: The self-assessment bias and Ikea effect diagnostics share scenario structure and make competing predictions. Implement the scenario triplet design in Section 8.5 first, then build the matched pairs for 8.4 as a subset. This avoids duplicating scenario generation effort.

5. **gpt-4o-mini token budget**: The system prompt (Section 4.2) is long. Verify that the combined system prompt plus scenario prompt plus expected response fits within the model's context window with margin. If necessary, compress Section B (legal context) using bullet points rather than prose, without reducing substantive content.
