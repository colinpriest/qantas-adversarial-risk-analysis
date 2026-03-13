# ASA Utility Quantification — Methodology

This document describes the approach used to estimate the Australian Shareholders' Association's utility function parameters and strike recommendation probabilities for the Qantas ARA decision tree.  It covers the pipeline implemented in `asa_utility_quantification.py`.

---

## 1. The Coherence Problem

An adversarial risk analysis (ARA) game tree requires two kinds of numerical input for each non-focal actor:

1. **Utility parameters** — weights that determine how the actor trades off competing concerns when choosing an action.
2. **Action probabilities** — the probability distribution over the actor's feasible actions at each decision node, conditioned on the game state.

These two inputs are not independent.  If ASA's utility function assigns weight *w* to board accountability and weight *v* to mobilisation cost, then the probability of a strike recommendation at any given A2 node is *determined* by those weights applied to the game state at that node.  Assigning probabilities and utility weights separately risks **incoherence**: the probabilities may be inconsistent with the utility function, or vice versa.

The pipeline enforces coherence by deriving action probabilities *from* the utility function via simulation.  The final P(strike) at each A2 node is the fraction of stochastic weight draws for which EU(strike) > EU(no_strike).  This guarantees that probabilities and utilities are always consistent.

### Why this matters

The incoherence problem is not hypothetical.  An earlier version of the engine used path-independent Beta priors for A2 strike probabilities, inherited from cross-company base rates.  At the node `root__ceo_stay__d3_ceo_transition` (Board forces CEO exit), the Beta prior implied a strike probability of ~89%, while the utility function's EU ranking predicted ASA would recommend a strike with probability ~93%.  The 4pp discrepancy arose because the Beta prior was calibrated to a different population of governance crises and did not condition on the specific game state.  When this prior was used in opponent modelling, the Board's ARA evaluation of ASA's behaviour was inconsistent with ASA's actual decision rule.

---

## 2. From Behaviour to Utility: The Identification Challenge

### 2.1 Historical base rates

The starting point for ASA calibration is observed behaviour.  ASA's track record in 15 comparable headline governance incidents in Australian listed companies provides an empirical anchor:

| Board response | Incidents | ASA recommended strike | Rate |
|---------------|-----------|----------------------|------|
| Board did nothing | 10 | 9 | 90.0% |
| Board commissioned review or CEO resigned | 3 | 3 | 100.0% |
| Board sacked CEO | 2 | 2 | 100.0% |
| **Overall** | **15** | **14** | **93.3%** |

These base rates establish that ASA almost always recommends a strike in headline governance crises.  The floor is high: even when boards take the strongest possible action, the historical rate is 100%.

### 2.2 Why base rates are not enough

The base rates provide a coarse anchor but are insufficient for three reasons:

1. **Sparse conditioning.**  With only 15 observations, the conditional rates (e.g., "Board sacked CEO" with n=2) are too noisy to distinguish between nodes.  A binomial 95% confidence interval for 2/2 extends from 34% to 100%.

2. **No utility identification.**  Observing that ASA struck in 14 of 15 cases tells us the strike action dominates, but not *why* or by *how much*.  The utility weights are not identified from the base rates alone — infinitely many weight vectors produce the same 93% aggregate rate.

3. **Case specificity.**  The Qantas crisis (CEO remuneration A\$21.4M, ACCC litigation, Federal Court ruling, CEO share sales) is among the most severe in Australian corporate history.  The base rate from the average headline incident understates the strike probability for this specific case.

The pipeline addresses these gaps through structured elicitation.

---

## 3. Scaffolding the Estimation

Because direct estimation from historical data is infeasible, the pipeline uses a scaffolding approach: multiple layers of structured elicitation that progressively constrain the parameter space until the utility weights are identified.

### 3.1 Step 1 — Range elicitation (usual and rare)

The first scaffold establishes the plausible probability space at each A2 node.  An LLM (gpt-4o-mini), prompted as a calibrated ASA company monitor, is asked to provide three values for P(strike) at each of the five A2 nodes:

- **LOW**: the lowest plausible probability (a "rare but defensible" assessment)
- **BEST**: the point estimate
- **HIGH**: the highest plausible probability (an "almost certain" assessment)

The prompt anchors the LLM to the historical base rate (93.3%) and to case-specific facts (Qantas FY23 remuneration, ACCC action, Federal Court ruling).  It also asks for two floor probabilities:

- **Expected floor**: the minimum P(strike) across all scenarios, reflecting ASA's reputational constraint (typically 0.88–0.95).
- **Absolute floor**: the hard lower bound below which ASA's credibility would be damaged (typically 0.80–0.88).

This step is repeated with *n* independent LLM draws (default 30) to capture the LLM's own uncertainty.  The variance across draws provides the standard deviation used later for Beta distribution fitting.

**Output**: per-node ranges {low, best, high, sd} and floor statistics.

**Five A2 nodes** (game tree paths to the ASA decision):

| Node | Game state | Description |
|------|-----------|-------------|
| Node 1 | CEO resigned, Board did nothing | CEO departed voluntarily, no reform |
| Node 2 | CEO resigned, Board commissioned review | CEO departed + forward-looking governance reform |
| Node 3 | CEO stayed, Board did nothing | Worst case: no action taken |
| Node 4 | CEO stayed, Board commissioned review | Review but CEO remains |
| Node 5 | Board forced CEO exit | Strongest possible accountability action |

### 3.2 Step 2 — Gap elicitation (pairwise probability signals)

The second scaffold establishes the *relative ordering* between nodes.  Rather than asking for absolute probabilities (which are hard to calibrate in the high-probability regime where all values cluster near 95%), the LLM is asked to assess *gaps* — how much each additional Board action reduces P(strike).

Four gaps are elicited:

| Gap | Comparison | What it measures |
|-----|-----------|-----------------|
| **Departure** | Node 3 vs Node 1 | How much does CEO departure reduce P(strike)? |
| **Review (CEO stays)** | Node 3 vs Node 4 | How much does commissioning a review reduce P(strike) when CEO stays? |
| **Review (CEO resigned)** | Node 1 vs Node 2 | How much does a review further reduce P(strike) after CEO departure? |
| **Sacking signal** | Node 2 vs Node 5 | How much additional reduction from Board-forced exit vs voluntary departure + review? |

Each gap is elicited as a range {low, expected, high}, again with *n* independent draws.  The prompt emphasises that the remuneration vote is **retrospective** — Board actions taken after the FY23 pay period are forward-looking signals that may slightly moderate ASA's position, but they do not change the historical pay structure being voted on.  This is why gaps should be small (typically 1–8 percentage points).

**Key design decision**: the gaps are elicited as *pairwise differences*, not absolute probabilities.  This is more cognitively tractable for the LLM (and for human experts) because it reduces the task to "how much does this action help?" rather than "what is the exact probability?".

### 3.3 Step 3 — Enforced ranking and constrained optimisation

The range and gap elicitations provide overlapping, potentially inconsistent constraints.  Step 3 reconciles them via constrained optimisation (SLSQP).

**Objective**: minimise the squared distance from the elicited best estimates, subject to:

1. **Range constraints**: each target probability must lie within its elicited [low, high] range.
2. **Floor constraint**: all targets must exceed the expected floor probability.
3. **Monotonicity**: the ranking must respect the a priori ordering:

   ```
   Node 3 > Node 1 >= Node 4 > Node 2 > Node 5
   ```

   Board inaction (Node 3) produces the highest P(strike); Board-forced CEO exit (Node 5) produces the lowest.  A minimum gap of 0.5 percentage points is enforced between consecutive ranked nodes to ensure the random utility model can produce distinct probabilities.

4. **Gap constraints**: the pairwise gaps must lie within the elicited [low, high] ranges from Step 2.

**Tied ranking**: Node 1 (CEO resigned, Board did nothing) and Node 4 (CEO stayed, Board commissioned review) are allowed to tie.  The elicitation data gives identical best estimates (0.950) and ranges ([0.900, 0.990]) for both nodes, and the utility structure makes them differ only by `w_passive - w_depart`, which the optimizer cannot identify.  Forcing a gap between them distorts the optimisation with no empirical support.

**Output**: a monotone target probability ladder, e.g.:

```
Node 3 (stay+nothing):      0.9800
Node 1 (resign+nothing):    0.9673
Node 4 (stay+review):       0.9623
Node 2 (resign+review):     0.9573
Node 5 (sacked):            0.9273
```

---

## 4. Implying Utility from Probability

### 4.1 The random utility model

The pipeline uses a **random utility model** (McFadden, 1974) to connect utility weights to action probabilities.  The core idea: utility weights are not point values but random variables drawn from truncated normal distributions.  For each simulation draw, the weights are sampled, expected utilities are computed for both actions, and the action with higher EU is selected.  The probability of a strike recommendation is the fraction of draws where EU(strike) > EU(no_strike).

Formally, let `w_k ~ TruncNormal(mu_k, sigma_k, 0.5, 10.0)` for each interaction parameter *k*.  At each A2 node, the delta-EU is:

```
delta_EU = sum_k  w_k * delta_phi_k
```

where `delta_phi_k = phi_k(rec_strike) - phi_k(no_strike)` is the difference in the basis function between the two actions.  Strike wins when `delta_EU > 0`.

### 4.2 Context vs interaction decomposition

The utility parameters are decomposed into two classes:

**Context parameters** (`w_ctx_inaction`, `w_ctx_departure`, `w_ctx_review`) fire equally for both actions.  They capture how good the situation is regardless of what ASA does.  Because they fire equally, they cancel in `delta_EU` and have no effect on action probabilities.  They exist for ordinal probit / Likert prediction but are irrelevant to the A2 decision.

**Interaction parameters** (5 parameters) fire only for `rec_strike` and drive the decision:

| Parameter | Delta-phi | Interpretation |
|-----------|-----------|---------------|
| `w_strike_cost` | -1 for all nodes | Net mobilisation cost of striking |
| `w_strike_vs_passive` | +1 when Board inactive, 0 otherwise | Value of striking against a passive Board |
| `w_departure_dampens` | -1 when CEO departed, 0 otherwise | CEO departure reduces strike value |
| `w_sack_dampens` | -1 when CEO was sacked, 0 otherwise | Board-forced exit further reduces strike value |
| `w_credibility_signal` | +1 for all nodes (high profile) | Repeat-game credibility value of striking |

The delta-phi matrix at the five A2 nodes:

```
                              cost  passive  depart  sack  cred
Node 3 (stay+nothing):         -1      +1       0     0    +1
Node 1 (resign+nothing):       -1      +1      -1     0    +1
Node 4 (stay+review):          -1       0       0     0    +1
Node 2 (resign+review):        -1       0      -1     0    +1
Node 5 (sacked):               -1       0      -1    -1    +1
```

All phi values are binary (0 or 1), which is a structural characteristic of this problem: ASA observes discrete game tree outcomes, not continuous governance quality scores.

### 4.3 Two-stage optimisation

The pipeline optimises the 10 parameters (5 means + 5 sigmas of the truncated normal distributions) to match the target probabilities from Step 3.

**Stage 1 — Analytical CLT (fast initial guess).**  Since `delta_EU` is a sum of independent truncated normals, its distribution is approximately normal by the CLT.  The mean and variance of `delta_EU` are computed from the truncated normal moments:

```
E[delta_EU] = sum_k  delta_phi_k * E[w_k]
Var[delta_EU] = sum_k  delta_phi_k^2 * Var[w_k]
P(strike) = Phi(E[delta_EU] / sqrt(Var[delta_EU]))
```

This gives a smooth, differentiable loss function.  Multi-start L-BFGS-B (50 random restarts) finds a good initial guess.

**Stage 2 — MC refinement with common random numbers (accurate final answer).**  The CLT approximation is systematically biased when parameters are at bounds (e.g., `TruncNormal(0.5, 2.1, 0.5, 10)` is heavily right-skewed, not normal).  Stage 2 pre-draws 100,000 uniform random numbers (common random numbers, fixed seed), then for any candidate `(mu, sigma)` vector, transforms them via the inverse CDF of the truncated normal.  This makes the MC loss function **deterministic** (same draws every call) and **unbiased** (exact truncated normal shape).  Nelder-Mead refines from the Stage 1 solution.

**Typical results**: Stage 1 loss ~0.00013, Stage 2 loss ~0.00008.  Max validation error (independent 50,000-draw simulation) ~0.8 percentage points.

### 4.4 Structural limitations

The binary phi values create structural constraints on how precisely the model can match arbitrary target probabilities.  With 5 interaction parameters and 5 target probabilities, the system is exactly determined in principle.  In practice, 4 of 5 weights are pushed to their lower bound (0.5) with minimum sigma (0.1), leaving essentially one degree of freedom (`w_credibility_signal`) to fit the probability ladder.  This is a consequence of the problem structure: in a high-probability regime (all nodes > 92%), the discrimination between nodes is small and dominated by the shared `w_cred - w_cost` intercept.

---

## 5. Beta Distributions from Simulated Probabilities

### 5.1 Why Beta distributions?

The ARA engine models opponents stochastically.  When ASA is not the focal actor, the engine samples ASA's action at each A2 node from a probability distribution rather than computing a deterministic best response.  The Beta distribution is the natural choice: it is the conjugate prior for a Bernoulli outcome (strike / no strike), is bounded on [0, 1], and is fully characterised by two parameters (alpha, beta) that control both the mean and the spread.

### 5.2 Method-of-moments fitting

Rather than using a fixed concentration parameter (n_eff), the pipeline derives **node-specific** Beta distributions using method of moments:

- **Mean**: the MC-optimised P(strike) from the random utility model (Step 4).
- **Variance**: the SD^2 from the LLM elicitation draws (Step 1), which captures genuine epistemic uncertainty about ASA's behaviour at each node.

The method-of-moments equations:

```
n_eff = mean * (1 - mean) / var  -  1
alpha = mean * n_eff
beta  = (1 - mean) * n_eff
```

Both alpha and beta are rounded to integers (the engine samples from `Beta(alpha, beta)` using NumPy's `rng.beta()`).  The effective sample size n_eff is clipped to [20, 500] to prevent degenerate distributions.

### 5.3 Node-specific concentration

Different nodes have different amounts of uncertainty.  The elicitation SDs vary substantially:

| Node | P(strike) | Elicited SD | n_eff | Beta | 95% CI |
|------|-----------|-------------|-------|------|--------|
| Node 3 (stay+nothing) | 0.979 | 0.009 | 235 | Beta(230, 5) | [0.957, 0.993] |
| Node 1 (resign+nothing) | 0.966 | 0.014 | 175 | Beta(169, 6) | [0.934, 0.987] |
| Node 4 (stay+review) | 0.965 | 0.015 | 143 | Beta(138, 5) | [0.929, 0.989] |
| Node 2 (resign+review) | 0.946 | 0.015 | 220 | Beta(208, 12) | [0.912, 0.971] |
| Node 5 (sacked) | 0.924 | 0.028 | 92 | Beta(85, 7) | [0.862, 0.969] |

The sacking scenario (Node 5) has the widest uncertainty (SD = 0.028, n_eff = 92) because the LLM assessments are most dispersed — there are only 2 historical observations of Board-forced exits, and the implications for ASA's response are genuinely uncertain.  The Board-inaction scenario (Node 3) has the narrowest uncertainty (SD = 0.009, n_eff = 235) because the LLM consistently rates this as near-certain strike territory.

### 5.4 Refit at MC-optimised probabilities

The Beta distributions are fitted twice:

1. **Initial fit** (Step 4a): uses the constrained target probabilities from Step 3 as means.  These Betas define the optimisation targets for the random utility model.

2. **Refit** (Step 4e): after the 2-stage optimisation, the Beta distributions are refitted using the MC-optimised P(strike) as the mean, keeping the same node-specific n_eff.  This ensures the engine's Beta priors reflect the actual model output rather than the (slightly different) optimisation targets.

---

## 6. Engine Integration

The calibrated Beta distributions are saved to `outputs/asa/asa_a2_calibration.json` and consumed by two engine components:

### 6.1 ASA as opponent (stochastic player)

When ASA is not the focal actor, `engine/predictive.py` loads the Beta priors and samples P(strike) at each A2 node:

```python
alpha, beta_param = self._a2_beta_priors[path_key]
p_strike = rng.beta(alpha, beta_param)
```

Each rollout draws a different p_strike from the Beta distribution, so the engine naturally represents uncertainty about ASA's behaviour.  Nodes with more elicitation uncertainty (wider Betas) produce more variable opponent behaviour.

### 6.2 ASA as focal actor (strategic player)

When ASA is the focal actor, the engine uses `utility_asa()` to evaluate terminal outcomes through the full 7-dimensional Likert system.  At A2, the tree recursion computes expected utility for both actions (strike and no-strike) by evaluating all downstream outcomes, and picks the action with higher EU.  The quantification pipeline's interaction weights are an equivalent reduced-form representation of this decision, but the terminal-level utility function is the canonical implementation.

### 6.3 Visualisation tree

The `run/game_tree.py` script loads point estimates of P(strike) from the calibration JSON for the interactive visualisation.  These are the MC-optimised probabilities (not Beta means, which differ slightly due to integer rounding).

---

## 7. Limitations and Assumptions

1. **LLM as surrogate expert.**  The elicitation uses gpt-4o-mini prompted as a calibrated ASA company monitor.  The LLM's probability assessments are informed by its training data (which includes ASA's published reports, proxy voting guidelines, and media coverage of governance incidents), but it is not a human ASA decision-maker.  The elicitation should be validated against expert judgement where possible.

2. **Binary phi values.**  The basis functions are binary (0/1), reflecting discrete game tree outcomes.  This limits the model's ability to distinguish between nodes that differ only in the balance of two opposing effects (e.g., Node 1 vs Node 4, where `w_passive - w_depart` is unidentifiable).

3. **CLT approximation in Stage 1.**  The analytical CLT approximation is systematically biased when truncated normal parameters are at bounds.  Stage 2 corrects this, but the initial guess may be suboptimal if the CLT bias is large.

4. **Fixed high-profile indicator.**  All nodes have `high_profile = True` (the Qantas crisis is unambiguously high-profile).  The `w_credibility_signal` parameter is identified only in this regime; the model does not generalise to low-profile cases.

5. **Retrospective vote assumption.**  The analysis assumes the remuneration vote is primarily retrospective (assessing FY23 pay), with Board actions serving as forward-looking signals that have limited moderating effect.  This is supported by ASA's published methodology but may overstate the rigidity of ASA's position.

---

## 8. References

- McFadden, D. (1974). Conditional logit analysis of qualitative choice behavior. In *Frontiers in Econometrics*, ed. P. Zarembka, 105–142. New York: Academic Press.
- Banks, D., Rios Insua, D., and Rios, J. (2015). *Adversarial Risk Analysis*. CRC Press.
- ASA (2023). *Voting Intentions — Qantas Airways Limited*. Australian Shareholders' Association.
