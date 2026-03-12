# ASA Utility Quantification Pipeline — Technical Specification

Technical documentation for `asa_utility_quantification.py` — a pipeline that estimates the ASA utility function parameters using LLM stakeholder simulation and Bayesian estimation, then feeds the posterior weight draws into the stochastic game tree to compute ASA action probabilities at every A2 node.

This specification is structurally parallel to `board_utility_quantification.md`. Differences from the Board pipeline are flagged explicitly.

---

## 1. Purpose and Motivation

The ARA game tree requires numerical values for ASA's utility weight parameters. These control how ASA trades off competing concerns when deciding whether to recommend a strike vote: the magnitude of pay-performance decoupling, the board's accountability response, ESG governance risk, disclosure quality, financial welfare of retail shareholders, organisational legitimacy, and procedural fairness.

**Problem:** ASA's utility weights cannot be read off from its published voting guidelines (which are qualitative). The weights assigned in the prior analysis — PPL=0.30, BA=0.20, EGR=0.15, FW=0.10, TD=0.10, OL=0.10, PF=0.05 — are expert elicitations, not estimates. Their relative magnitudes are uncertain, and interaction effects between dimensions (notably PPL×BA) are unspecified.

**Problem specific to ASA vs Board:** The conditional prior incoherence identified at `root__ceo_stay__d3_ceo_transition` demonstrates that path-independent Beta priors inherited from cross-company base rates can contradict the utility model's EU ranking. The solution is to derive ASA action probabilities from the same argmax-count mechanism used for the Board: per-draw utility maximisation over posterior weight draws, so that the action probabilities are always coherent with the estimated utility function.

**Solution:** The same three-stage approach as the Board pipeline:

1. **Scenario elicitation** — Use gpt-4o-mini as a calibrated ASA company monitor simulator to generate Likert-scale appropriateness ratings for structured governance scenarios.
2. **Bayesian estimation** — Fit an ordinal probit model (Stan) to the Likert data, producing posterior draws of all ASA utility weight parameters.
3. **Stochastic simulation** — Feed posterior weight draws into the game tree; ASA action probabilities emerge from per-draw argmax-count.

---

## 2. Key Structural Differences from the Board Pipeline

| Feature | Board | ASA |
|---------|-------|-----|
| Action space at decision nodes | 3 actions (D0/D1/D3) | 2 actions (no_strike / rec_strike) |
| Utility argument that binds | w_inaction_base + vote penalties | w_ppl (backward-looking; fixed once remuneration report is public) |
| Mobilisation cost | Absent (governance actions are within mandate) | Present: strike recommendation incurs volunteer mobilisation cost and relationship cost with company |
| OL structure | Symmetric (inaction and action both have consequences) | Asymmetric: OL penalty for visible inaction when salience is high; no symmetric OL reward for striking in low-salience cases |
| Persona | Named 8-director board (collective disagreement modelled) | ASA company monitor team + CEO (collective, but mission-homogeneous) |
| Temporal structure | Board acts before vote is observed | ASA acts after game state (CEO, board actions) is observed but before AGM |
| PPL is backward-looking | N/A | Critical: PPL deficit is locked in once the remuneration report is published; no board action after that point can alter it |

---

## 3. ASA Utility Function Decomposition

### 3.1 Functional Form

```
EU(ASA, scenario, action) = sum_k  w_k * phi_k(scenario, action)
                           + anchored(scenario)
                           - w_mob * I[action == rec_strike]
```

where:
- `phi_k(scenario, action)` is the basis function for parameter k
- `w_k > 0` are the weights being estimated
- `anchored(scenario)` captures fixed scenario-level contributions that do not vary by action (baseline financial welfare, baseline share price)
- `w_mob > 0` is the mobilisation cost: a fixed negative utility term that fires only when `action == rec_strike`, representing volunteer effort, public statement drafting, and relationship friction with the company

### 3.2 Parameters and Basis Functions

**Nine action-varying weights** (lognormal priors centred at prior elicitations):

| # | Parameter | Description | Prior median | Phi pattern |
|---|-----------|-------------|--------------|-------------|
| 1 | w_ppl | Pay-performance link deficit | 3.0 | `-ppl_deficit` (continuous, 0–1 scale) |
| 2 | w_ba | Board accountability signal | 2.0 | `+ba_signal` (continuous, 0–1 scale) |
| 3 | w_ppl_ba | PPL×BA interaction: BA credit is discounted when PPL deficit is severe | 1.5 | `-ppl_deficit * (1 - ba_signal)` |
| 4 | w_egr | ESG governance risk severity | 1.5 | `-egr_severity` (graduated 0–1) |
| 5 | w_td | Transparency deficit | 1.0 | `-td_deficit` (0=fully transparent, 1=opaque) |
| 6 | w_ol | Organisational legitimacy cost of inaction | 2.0 | `-ol_exposure * I[action == no_strike]` |
| 7 | w_pf | Procedural fairness violation | 0.5 | `-pf_violations` (count-based, normalised) |
| 8 | w_mob | Mobilisation cost of strike recommendation | 1.0 | `-I[action == rec_strike]` |
| 9 | w_ol_salience | OL amplifier for high-salience cases | 1.0 | `-salience * ol_exposure * I[action == no_strike]` |

**Two vote-context modifiers** (scenario-level, not action-varying — used for scale anchoring only):

| Parameter | Description | Prior median |
|-----------|-------------|--------------|
| w_strike_context | Utility boost from a successful first strike in a high-PPL case | 2.0 |
| w_second_strike_risk | Utility reduction from strategic risk of triggering second-strike dynamics prematurely | 1.0 |

### 3.3 Basis Function Definitions

**ppl_deficit** — Continuous [0, 1]:
```
ppl_deficit = clip((pay_ratio_normalised - 0.5) / 0.5, 0, 1)
```
where `pay_ratio_normalised` is the pay-to-benchmark ratio normalised to [0, 1] across the scenario battery. At ppl_deficit = 0, pay is at or below benchmark (no PPL concern). At 1.0, pay is maximally decoupled from performance (as in the Qantas FY23 case: near 10-fold increase against ACCC action).

**ba_signal** — Continuous [0, 1]:
```
ba_signal = 0.00   # Board did nothing (D0_minimal)
ba_signal = 0.35   # Board commissioned governance review (D1_review)
ba_signal = 0.65   # Board commissioned review AND CEO resigned voluntarily
ba_signal = 0.85   # Board forced CEO exit (D3_ceo_transition)
ba_signal = 1.00   # Board forced exit + committed to clawback
```
This graduated scale is the key identification device. The difference in ASA's Likert rating between ba_signal = 0.35 and ba_signal = 0.85 identifies `w_ba`. The interaction term `w_ppl_ba` is identified by comparing that BA uplift across low-PPL and high-PPL scenarios.

**Critical constraint:** When `action == no_strike`, ba_signal contributes positively (board has acted, strike is less necessary). When `action == rec_strike`, ba_signal reduces the utility of striking (the primary goal is already partially achieved). Both effects are captured by the `+ba_signal` basis function, which is positive under no_strike and must be negative under rec_strike — this sign flip is handled by including `-ppl_deficit * (1 - ba_signal)` as the interaction term.

**egr_severity** — Graduated [0, 1]:
```
egr_severity = 0.0   # No confirmed ESG violations
egr_severity = 0.3   # Single regulatory investigation (announced, not filed)
egr_severity = 0.6   # Filed court action OR confirmed labour law violation
egr_severity = 0.8   # Both filed action AND confirmed labour violation
egr_severity = 1.0   # Multiple filed actions + confirmed violations + share sale timing concern
```

**td_deficit** — Graduated [0, 1]:
```
td_deficit = 0.0   # Full conduct-linked STI gating, clawback provisions disclosed
td_deficit = 0.4   # Partial disclosure (some metrics disclosed, no clawback)
td_deficit = 0.7   # No conduct gating, no clawback, but report is otherwise legible
td_deficit = 1.0   # Opaque report, no conduct metrics, no clawback, no STI breakdown
```

**ol_exposure** — Continuous [0, 1]: product of case salience and member-customer overlap:
```
ol_exposure = salience_score * member_overlap_score
```
where `salience_score` reflects media coverage intensity and `member_overlap_score` reflects the proportion of ASA's member base likely to be directly affected as customers or shareholders. For Qantas, both are at maximum (salience = 1.0, overlap ≈ 1.0 given Qantas's retail shareholder profile).

**pf_violations** — Integer count, normalised:
```
pf_violations = count of: {AGM mic cutoff, insider share sale timing, 
                            institutional-only capital raising, 
                            non-hybrid AGM format, restricted floor questions}
normalised to [0, 1] by dividing by 5
```

---

## 4. Paired Scenario Design

### 4.1 Identification Strategy

Identical to the Board pipeline: paired or grouped scenarios differing in exactly one game-state feature isolate the corresponding parameter. The LLM sees natural-language scenario descriptions and rates the appropriateness of each action (no_strike / rec_strike) on a 1–5 Likert scale.

**Critical difference from Board:** Because ASA has only two actions, the basis function must vary *between* the two actions within a scenario for a parameter to be identifiable. Parameters whose phi is identical for no_strike and rec_strike cancel in the Likert rating difference and have zero gradient. The design must ensure each target parameter has at least some scenarios where its phi differs meaningfully across the two actions.

For most parameters this is naturally satisfied: `w_ppl` and `w_egr` increase the appropriateness of rec_strike (phi is larger for rec_strike), while `w_ba` decreases it. `w_mob` fires only for rec_strike. `w_ol` fires only for no_strike.

### 4.2 Four Scenario Tiers

| Tier | Count | Purpose |
|------|-------|---------|
| 1 | ~55 | Parameter isolation: single feature varies |
| 2 | ~28 | Joint estimation + historical anchoring |
| 3 | ~24 | Behavioural probes |
| 4 | 1 | Historical calibration: Qantas Nov 2023 AGM |

---

### 4.3 Tier 1: Parameter Isolation Scenarios (~55 scenarios)

#### w_ppl: Pay-Performance Link (12 scenarios)

Six pay ratio levels × 2 BA contexts (do nothing / governance review):

| pay_ratio_normalised | ppl_deficit | ba_context |
|---------------------|-------------|------------|
| 0.10 | 0.00 | do_nothing |
| 0.10 | 0.00 | review |
| 0.30 | 0.00 | do_nothing |
| 0.30 | 0.00 | review |
| 0.55 | 0.10 | do_nothing |
| 0.55 | 0.10 | review |
| 0.70 | 0.40 | do_nothing |
| 0.70 | 0.40 | review |
| 0.85 | 0.70 | do_nothing |
| 0.85 | 0.70 | review |
| 1.00 | 1.00 | do_nothing |
| 1.00 | 1.00 | review |

The rating difference between ppl_deficit=0.00 and ppl_deficit=1.00, holding ba_context constant, identifies w_ppl. The interaction between this shift and ba_context identifies w_ppl_ba.

#### w_ba: Board Accountability Signal (10 scenarios)

Five ba_signal levels × 2 PPL contexts (moderate / severe):

| ba_signal | board_action | ppl_deficit |
|-----------|-------------|-------------|
| 0.00 | Do nothing | 0.70 |
| 0.00 | Do nothing | 1.00 |
| 0.35 | Governance review | 0.70 |
| 0.35 | Governance review | 1.00 |
| 0.65 | Review + voluntary resignation | 0.70 |
| 0.65 | Review + voluntary resignation | 1.00 |
| 0.85 | Board-forced exit | 0.70 |
| 0.85 | Board-forced exit | 1.00 |
| 1.00 | Forced exit + clawback commitment | 0.70 |
| 1.00 | Forced exit + clawback commitment | 1.00 |

The rating shift for rec_strike as ba_signal increases from 0 to 1 (holding ppl_deficit constant) identifies w_ba. The comparison across the two ppl_deficit levels identifies w_ppl_ba.

#### w_egr: ESG Governance Risk (8 scenarios)

Four egr_severity levels × 2 PPL contexts:

| egr_severity | ESG state | ppl_deficit |
|-------------|-----------|-------------|
| 0.0 | No violations | 0.70 |
| 0.0 | No violations | 1.00 |
| 0.6 | Filed court action | 0.70 |
| 0.6 | Filed court action | 1.00 |
| 0.8 | Court action + labour violation | 0.70 |
| 0.8 | Court action + labour violation | 1.00 |
| 1.0 | Full Qantas-equivalent ESG state | 0.70 |
| 1.0 | Full Qantas-equivalent ESG state | 1.00 |

#### w_td: Transparency Deficit (6 scenarios)

Three td_deficit levels × 2 PPL severity contexts:

| td_deficit | disclosure_state | ppl_deficit |
|-----------|-----------------|-------------|
| 0.0 | Full conduct gating + clawback | 0.70 |
| 0.0 | Full conduct gating + clawback | 1.00 |
| 0.7 | No conduct gating, no clawback | 0.70 |
| 0.7 | No conduct gating, no clawback | 1.00 |
| 1.0 | Maximally opaque | 0.70 |
| 1.0 | Maximally opaque | 1.00 |

#### w_ol: Organisational Legitimacy (8 scenarios)

Four ol_exposure levels (varying salience and member overlap) × 2 PPL contexts. Critically: these scenarios hold all other features constant and only vary how publicly prominent the case is and how directly ASA's membership is affected.

| ol_exposure | case_description | ppl_deficit |
|------------|-----------------|-------------|
| 0.1 | Low-profile industrial company, few retail shareholders | 0.70 |
| 0.1 | Low-profile industrial company, few retail shareholders | 1.00 |
| 0.4 | Mid-tier consumer company, moderate media coverage | 0.70 |
| 0.4 | Mid-tier consumer company, moderate media coverage | 1.00 |
| 0.7 | High-profile consumer brand, significant media coverage | 0.70 |
| 0.7 | High-profile consumer brand, significant media coverage | 1.00 |
| 1.0 | National icon, maximum media coverage, near-universal member exposure | 0.70 |
| 1.0 | National icon, maximum media coverage, near-universal member exposure | 1.00 |

The key identification: when ol_exposure is high and action == no_strike, ASA rates no_strike much lower (the OL cost is visible). When ol_exposure is low, the no_strike vs rec_strike Likert gap should be much smaller. This identifies w_ol and w_ol_salience.

#### w_mob: Mobilisation Cost (6 scenarios)

Mobilisation cost is identified by scenarios where PPL deficit is moderate and BA signal is moderate — the "ambiguous zone" where the mobilisation cost is the marginal determinant. Three mobilisation_context levels × 2 PPL deficits:

| mobilisation_context | description | ppl_deficit |
|---------------------|-------------|-------------|
| low | AGM is 8 weeks away, monitors fully available | 0.55 |
| low | AGM is 8 weeks away, monitors fully available | 0.70 |
| medium | AGM is 3 weeks away, partial monitor availability | 0.55 |
| medium | AGM is 3 weeks away, partial monitor availability | 0.70 |
| high | AGM is 1 week away, peak season with 30+ AGMs/week | 0.55 |
| high | AGM is 1 week away, peak season with 30+ AGMs/week | 0.70 |

**Note:** w_mob is the weakest-identified parameter in the battery. It is included for completeness but its prior should receive more weight in the final posterior. Alternative: treat w_mob as fixed at a researcher-set value (e.g., 0.3 utility units) and exclude it from estimation.

#### w_pf: Procedural Fairness (5 scenarios)

Three pf_violation counts at the same PPL/BA baseline:

| pf_violations (raw) | violations present | ppl_deficit | ba_signal |
|--------------------|--------------------|-------------|-----------|
| 0 | None | 0.80 | 0.00 |
| 1 | Insider share sale timing concern | 0.80 | 0.00 |
| 2 | Share sale + AGM mic cutoff | 0.80 | 0.00 |
| 3 | Share sale + mic cutoff + hybrid AGM denied | 0.80 | 0.00 |
| 5 | All violations | 0.80 | 0.00 |

---

### 4.4 Tier 2: Joint Multi-Feature Scenarios (~28 scenarios)

Realistic governance scenarios where multiple parameters fire simultaneously. These serve two purposes:

**Joint estimation:** Verify that Tier 1 isolation parameters generalise to multi-feature combinations. If well-specified, predictions from Tier 1 weights should match Tier 2 ratings.

**Historical anchoring:** Two anchor scenarios with known cross-company outcomes:
- A financial services company post-Royal Commission receiving a ~35% against vote (moderate strike) — known historical outcome
- A mining company with strong ESG governance and well-disclosed remuneration receiving a ~5% against vote — known outcome

These anchors calibrate the Likert-to-utility ratio, analogous to the Board pipeline's CAR-based anchoring.

Tier 2 configurations span:
- 4 PPL deficit levels × 3 BA signal levels × 2 EGR severity levels = 24 base combinations
- Plus 4 special cases: pre-remuneration-report-release (epistemic uncertainty scenarios), share sale scenarios (PF activated independently of PPL), and post-forced-exit-with-clawback scenarios (near-zero strike region)

---

### 4.5 Tier 3: Behavioural Probes (~24 scenarios)

Matched pairs testing cognitive biases that would violate model assumptions if present.

| Bias | Scenarios | Design |
|------|-----------|--------|
| Anchoring / order effects | 8 | Same scenario, different information presentation order (pay revealed first vs ESG revealed first) |
| Recency bias | 6 | Same cumulative ESG record, varied temporal ordering of incidents |
| Attribution | 4 | Same pay quantum, varied framing (industry-wide vs company-specific performance) |
| PPL threshold effects | 6 | Dense PPL deficit grid around the 0.50 midpoint — tests for discontinuity vs linearity |

These do not enter the estimation. They diagnose whether the LLM exhibits biases incompatible with the linear utility decomposition.

**Critical probe — PPL backward-looking lock-in:** One pair of scenarios tests whether the LLM correctly treats PPL as backward-looking once the remuneration report is public. Scenario A: remuneration report released, then board forces CEO exit. Scenario B: board forces CEO exit, then remuneration report released. If the model is well-specified, both should produce the same Likert ratings on the rec_strike action because the PPL deficit is identical. Systematic differences would indicate hindsight or sequence-sensitivity bias.

---

### 4.6 Tier 4: Historical Calibration (1 scenario)

The Qantas November 2023 AGM at the full observed state:
- ppl_deficit = 1.00 (A$21.4M, near 10-fold increase, ACCC action live)
- ba_signal = 0.65 (Joyce resigned voluntarily; governance review commissioned)
- egr_severity = 1.00 (ACCC court action + labour law violation + share sale timing)
- td_deficit = 0.70 (no conduct gating, no clawback, partial holdback signalled)
- ol_exposure = 1.00 (maximum salience, maximum member overlap)
- pf_violations = 2 (share sale timing + mic cutoff at AGM noted retrospectively)

**Observed outcome:** ASA recommended strike. The model should assign rec_strike as the argmax action with high probability (>0.95) under the posterior.

This is the out-of-sample test. It is NOT included in estimation — it validates the model.

---

## 5. Bayesian Estimation via Ordinal Probit

### 5.1 Model Selection

**Difference from Board pipeline:** Because ASA has a binary action space (no_strike / rec_strike), a binary probit model on the action choice could in principle replace the ordinal probit on Likert ratings. However, ordinal probit on Likert ratings is retained for three reasons:

1. **Consistency** with the Board pipeline — same model family, same Stan code structure, comparable posterior diagnostics
2. **Within-scenario variance** — 40 independent LLM ratings per scenario provide within-scenario variance for the random effects, which is lost if we collapse to a single binary choice per scenario
3. **Gradient richness** — A rating of 1 vs 3 for rec_strike encodes more information than a binary 0/1 choice; this is especially valuable for the weak-signal scenarios in the ambiguous zone

### 5.2 Model Structure

The Stan model (`models/asa_ordinal_utility.stan`):

**Latent utility for (scenario s, action a):**
```
mu[s,a] = phi[s,a] . w  +  anchored[s]  -  w_mob * I[a == rec_strike]
```

**Normalisation:** Pre-compute `mu_scale` from the spread of mu values across all (scenario, action) pairs. Normalise `eta = (mu + RE) / mu_scale`.

**Observation model:**
```
y[n] ~ ordered_probit(eta[sa_id[n]], cutpoints)
```

with scenario-level random intercepts (non-centred: `z_scenario ~ N(0,1)`, `RE = sigma_scenario * z_scenario`).

### 5.3 Estimated Parameters (11 total)

**9 action-varying weights** (lognormal priors):

| Parameter | Prior median | Prior SD (log scale) | Constraint |
|-----------|-------------|---------------------|-----------|
| w_ppl | 3.0 | 1.0 | > 0 |
| w_ba | 2.0 | 1.0 | > 0 |
| w_ppl_ba | 1.5 | 1.0 | > 0 |
| w_egr | 1.5 | 1.0 | > 0 |
| w_td | 1.0 | 1.0 | > 0 |
| w_ol | 2.0 | 1.0 | > 0 |
| w_pf | 0.5 | 1.0 | > 0 |
| w_mob | 1.0 | 1.0 | > 0 |
| w_ol_salience | 1.0 | 1.0 | > 0 |

**2 ordering constraints** enforced via decomposition:
- `w_ppl > w_ba`: PPL is the primary driver; BA cannot dominate PPL (enforced via `w_ppl = w_ba + delta_ppl`, `delta_ppl ~ lognormal(0.0, 0.8)`)
- `w_ol + w_ol_salience * salience_max > w_mob`: At maximum salience, the OL cost of inaction must exceed the mobilisation cost of striking (enforced via prior structure, not hard constraint)

### 5.4 Cutpoint Reparameterisation

Identical to the Board pipeline:
```
cutpoint[1] = 3 * tanh(base_raw)
cutpoint[g+1] = cutpoint[g] + 0.25 + 2.0 * inv_logit(gap_raw[g])
```

### 5.5 MCMC Configuration

Identical to Board pipeline:
- 4 chains × 2,000 sampling draws + 1,000 warmup = 8,000 posterior draws
- `adapt_delta = 0.99`
- `max_treedepth = 15`
- Convergence: R-hat < 1.01, bulk ESS > 400, divergences < 10

---

## 6. From Posterior Draws to ASA Action Probabilities

### 6.1 EU Computation at A2 Nodes

At each A2 node, the game tree has already propagated Board and CEO EU through the subtree. For ASA, on posterior draw `i`:

```python
EU_ASA[i, action] = w_draws[i, :] @ phi_vector(scenario, action)
                  + anchored_value(scenario)
                  - w_mob[i] * I[action == rec_strike]
```

where `phi_vector` encodes the game state at the A2 node (ppl_deficit, ba_signal, egr_severity, td_deficit, ol_exposure, pf_violations, and their interaction terms).

### 6.2 Argmax-Count at A2 Nodes

```python
# asa_eu_mat: (n_draws, 2) — EU of no_strike and rec_strike per draw
best_idx = argmax(asa_eu_mat, axis=1)   # 0 = no_strike, 1 = rec_strike
p_rec_strike = (sum(best_idx == 1) + alpha) / (n_draws + 2 * alpha)
```

This replaces the fixed Beta priors currently used at A2 nodes. The resulting probabilities are guaranteed to be coherent with the utility model by construction: if rec_strike has higher EU on 99.8% of draws (as at `root__ceo_stay__d3_ceo_transition`), the probability is 99.8%, not an inherited 60% from a cross-company base rate.

### 6.3 Game State Encoding for Each A2 Node

The five A2 nodes in the current tree map to the following game state vectors:

| Node | ppl_deficit | ba_signal | egr_severity | td_deficit | ol_exposure | pf_violations |
|------|-------------|-----------|-------------|------------|-------------|---------------|
| CEO resigns → do nothing | 1.00 | 0.30 | 1.00 | 0.70 | 1.00 | 1 |
| CEO resigns → review | 1.00 | 0.65 | 1.00 | 0.65 | 1.00 | 1 |
| CEO stays → do nothing | 1.00 | 0.00 | 1.00 | 0.70 | 1.00 | 1 |
| CEO stays → review | 1.00 | 0.40 | 1.00 | 0.65 | 1.00 | 1 |
| CEO stays → board forces exit | 1.00 | 0.85 | 1.00 | 0.65 | 1.00 | 1 |

**Key observation:** ppl_deficit = 1.00 is fixed across all five nodes because the FY23 remuneration report is public at all A2 decision points. This encodes the binding constraint identified in the utility dynamics analysis: PPL is backward-looking and locked in once the report is released. The only parameter that varies substantially across nodes is ba_signal.

### 6.4 Why This Resolves the Prior Incoherence

At `root__ceo_stay__d3_ceo_transition` (CEO stays → board forces exit):
- ba_signal = 0.85 (board has taken the strongest possible action short of clawback)
- But ppl_deficit = 1.00 (the remuneration report cannot be undone)
- The interaction term `w_ppl_ba` captures how much BA credit ASA extends when PPL deficit is at its maximum

If the posterior for w_ppl_ba is estimated with sufficient precision — which the Tier 1 BA×PPL paired scenarios are designed to ensure — the argmax-count will naturally produce a near-degenerate distribution at this node because:

```
EU(rec_strike) = -w_ppl * 1.00  +  w_ba * 0.85  -  w_ppl_ba * 1.00 * (1 - 0.85)  +  ...
EU(no_strike)  = 0               +  w_ba * 0.85  +  0                               +  ...
```

For rec_strike to dominate, we need:
```
-w_ppl * 1.00  -  w_ppl_ba * 0.15  >  w_mob + 0
```

Given w_ppl prior median 3.0, w_mob prior median 1.0, w_ppl_ba prior median 1.5:
```
-3.0 - 0.225  >  -1.0  →  -3.225 > -1.0  →  False
```

Wait — this is the reverse: no_strike EU includes the ba_signal benefit, rec_strike incurs the mobilisation cost. Let me restate:

```
EU(no_strike)  = +w_ba * 0.85  -  w_ol * 1.00  (OL penalty for no_strike fires)
EU(rec_strike) = -w_ppl * 1.00  -  w_ppl_ba * 0.15  -  w_mob
```

For no_strike to dominate (consistent with the EU table showing +2.79 for no_strike):
```
w_ba * 0.85 - w_ol  >  -w_ppl - w_ppl_ba * 0.15 - w_mob
```

This is the correct inequality the posterior must satisfy. The estimation will confirm whether it holds across 99.8% of draws — and if it does, the argmax-count produces p(no_strike) ≈ 0.998, which is coherent. The Beta(9,6) prior producing p(rec_strike) = 0.60 was simply ignoring all of this structure.

---

## 7. LLM Prompts

### 7.1 System Prompt Architecture

| Section | Content | Rationale |
|---------|---------|-----------|
| A: ASA Persona | Mission, member base (retail investors, SMSF trustees), volunteer company monitor structure, not-for-profit status | Grounds the simulation in ASA's specific institutional identity and constraints |
| B: Utility Framework | The seven utility arguments with qualitative descriptions; PPL as primary driver; mobilisation cost as constraint | Ensures the LLM reasons from ASA's actual decision framework, not generic activist logic |
| C: Governance Context | The Qantas governance crisis backdrop WITHOUT revealing ASA's actual voting intention | Provides crisis context; prevents hindsight contamination |
| D: Decision Framework | Two-strikes rule mechanics; what a strike recommendation entails; ASA's published voting guidelines on remuneration | Ensures the LLM reasons about the actual regulatory and governance mechanisms |
| E: Response Format | Likert rating instructions for each action | Structured output for ordinal probit |

**Critical constraint (identical to Board):** Section C must state: "ASA's voting intention for the Qantas 2023 AGM is NOT known at this decision point. You are reasoning prospectively."

### 7.2 Key Difference in Persona Construction

The Board prompt names all 8 directors with professional backgrounds to model internal disagreement. ASA is more mission-homogeneous — the internal disagreement is not between named individuals but between two decision logics:

1. **The accountability logic:** Strike to impose consequences and signal norms
2. **The pragmatic logic:** No strike when the board has already acted; preserve the relationship to influence future behaviour

The system prompt should explicitly describe both logics as legitimate considerations within ASA's framework, to ensure the LLM explores the full utility space rather than defaulting to the accountability logic in all cases.

### 7.3 Scenario Prompt Structure

```
[GAME STATE]
Pay quantum and benchmark: [ppl_description]
Board action taken to date: [ba_description]
Regulatory and ESG status: [egr_description]
Remuneration report disclosure quality: [td_description]
Public profile and member exposure: [ol_description]
Procedural issues: [pf_description]
Time to AGM: [mobilisation_context]

[QUESTION]
ASA's company monitor team is preparing its voting intentions for [Company]'s
upcoming AGM. The remuneration report is now public. Two actions are feasible:

Action A — No strike recommendation: ASA votes open proxies in favour of (or
abstains on) the remuneration report.

Action B — Recommend strike: ASA votes open proxies against the remuneration
report and publishes a voting intention explaining its reasoning.

Please rate the appropriateness of EACH action separately on a 1–5 scale:
1 = Not appropriate / not warranted given the circumstances
3 = Uncertain / could go either way
5 = Strongly appropriate / clearly warranted

Respond in JSON: {"no_strike": <1-5>, "rec_strike": <1-5>, "reasoning": "<2-3 sentences>"}
```

---

## 8. Pipeline Overview

```
Stage 1: Scenario Generation    -> asa_scenarios.csv (~108 scenarios)
Stage 2: LLM Elicitation        -> asa_elicitation_results.csv (108 x 40 = 4,320 calls)
Stage 3: Data Preprocessing     -> asa_estimation_dataset.csv (Likert long-form)
Stage 4: Bayesian Estimation    -> asa_stan_posterior_draws.npz (8,000 x 9 weight draws)
                                -> asa_parameter_estimates.csv
Stage 5: Behavioural Diagnostics -> 4 bias tests
Stage 6: Validation & Dashboard -> asa_utility_dashboard.html (10-tab interactive)
         |
         v
    run/run_unified_ARA.py      -> loads asa_w_draws
                                -> computes phi vectors for each A2 node
                                -> argmax-count produces P(no_strike), P(rec_strike)
                                -> replaces Beta priors at all A2 nodes
```

---

## 9. Behavioural Diagnostics

| Test | Scenarios | Expected result | If violated |
|------|-----------|-----------------|-------------|
| PPL backward-lock | 4 pairs | Identical ratings regardless of event sequence after remuneration report release | Drop sequence-manipulation scenarios; add ordering constraint |
| OL asymmetry | 6 pairs | OL penalty for no_strike should be larger at high salience; no symmetric OL reward for striking at low salience | Introduce asymmetric OL parameterisation |
| BA saturation | 5 pairs | Each additional BA signal increment should produce diminishing returns on Likert shift | Extend model with log(ba_signal) basis instead of linear |
| Mobilisation sensitivity | 4 pairs | Likert difference between no_strike and rec_strike should narrow as mobilisation cost increases | Confirm w_mob is identified; if flat, fix w_mob |

---

## 10. Parameter Mapping to Engine

| Pipeline param | Engine parameter | Relationship |
|---------------|-----------------|--------------|
| w_ppl | `asa_ppl_weight` | Direct |
| w_ba | `asa_ba_weight` | Direct |
| w_ppl_ba | `asa_ppl_ba_interaction` | Direct |
| w_egr | `asa_egr_weight` | Direct |
| w_td | `asa_td_weight` | Direct |
| w_ol | `asa_ol_weight` | Direct |
| w_pf | `asa_pf_weight` | Direct |
| w_mob | `asa_mobilisation_cost` | Direct |
| w_ol_salience | `asa_ol_salience_amplifier` | Direct |

---

## 11. End-to-End Data Flow

```
gpt-4o-mini                Stan MCMC               Game Tree (A2 Nodes)
-----------                ---------               --------------------
  |                           |                           |
  | Likert ratings            |                           |
  | (1-5 per action)          |                           |
  v                           |                           |
asa_scenarios.csv  -->  asa_ordinal_utility.stan --> asa_stan_posterior_draws.npz
(~108 scenarios          (ordinal probit,              (8000 x 9 weight draws)
 x 40 reps               9 weights,                           |
 = 4,320 calls)          2 ordering constraints,              |
                         cutpoints, scenario RE)              v
                                                    run_unified_ARA.py
                                                         |
                                                         | loads asa_w_draws
                                                         | computes phi_vector
                                                         |   per A2 node
                                                         v
                                                    EU_ASA[i, action]
                                                    = asa_w_draws[i,:] @ phi
                                                    - w_mob[i] * I[rec_strike]
                                                         |
                                                         | argmax per draw
                                                         v
                                                    P(action) = count / n_draws
                                                         |
                                                         v
                                                    Replaces Beta priors
                                                    at all A2 nodes
                                                    (prior-coherent by construction)
```

---

## 12. Design Decisions

1. **Ordinal probit retained over binary probit:** Consistency with Board pipeline. The 40-rep Likert elicitation provides richer gradient information than a collapsed binary choice, especially in the ambiguous zone (ba_signal = 0.65, ppl_deficit = 0.70) where the utility difference between actions is small.

2. **ppl_deficit fixed at 1.00 across all A2 nodes in the Qantas case study:** The remuneration report is public at all A2 decision points. This is not a simplification — it encodes the binding constraint identified analytically. It means the posterior weight on w_ppl does not affect the *relative* probability across A2 nodes, only the absolute level of rec_strike utility. The cross-node variation comes entirely from w_ba and the interaction term.

3. **w_ppl > w_ba ordering constraint:** Theory-driven. ASA's published guidelines explicitly treat the pay-performance link as the primary determinant of the remuneration vote, with board accountability as a modifying factor. This ordering prevents the estimation from producing pathological solutions where a sufficiently strong BA signal always overrides the PPL deficit regardless of its magnitude.

4. **Mobilisation cost as an estimated parameter (not fixed):** The original analysis treated mobilisation cost as background noise. However, the 60/40 prior at the forced-exit node was almost certainly driven by an implicit mobilisation cost being too high relative to the marginal accountability benefit of striking when the board had already acted. Estimating w_mob explicitly surfaces this trade-off and makes it falsifiable.

5. **OL asymmetry in the basis function design:** The basis function `w_ol * ol_exposure * I[action == no_strike]` encodes the asymmetry identified in the utility analysis: OL generates a cost for visible inaction in salient cases, but does not generate a symmetric reward for visible action in low-salience cases. ASA does not strike to build brand; it faces reputational costs for failing to strike when members expect it to.

6. **Tier 4 historical calibration as falsification, not estimation:** The Qantas 2023 AGM outcome (ASA recommended strike, 83% result) is out-of-sample. If the estimated model assigns P(rec_strike) < 0.80 at the Tier 4 state vector, the model is misspecified. This is a strong falsifiability criterion given ppl_deficit = 1.00 and ol_exposure = 1.00 in that scenario.

7. **gpt-4o-mini over gpt-4o:** Consistent with Board pipeline. For ASA's binary decision space, the loss in LLM quality from using the mini model is further reduced compared to the Board's three-action space with more complex interaction effects.
