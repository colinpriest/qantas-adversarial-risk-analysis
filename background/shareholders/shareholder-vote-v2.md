# Shareholder Vote Model v2 — Revised Specification

This document specifies the Bayesian distribution, parameters, and design rationale for the shareholder strike vote at the AGM (node V in the game tree). The vote model is the central chance node in the Qantas ARA engine: it converts upstream decisions (Board governance reform, ASA strike recommendation) and latent belief states into a continuous vote percentage, from which discrete thresholds (first strike at 25%, overwhelming at 50%) are derived.

This revision addresses six structural issues identified in review: insufficient ASA differentiation across headline conditions, underproduction of first strikes in crisis scenarios, overly strong sign constraints on CEO-exit governance effects, a bad control problem in the gamma_A estimation, non-independent channel combination, and potential importance sampling degeneracy at checkpoint C3.

---

## 1. Distribution Specification

### 1.1 The logit-normal model

The vote fraction V in (0, 1) follows a logit-normal distribution:

```
logit(V) ~ N(mu_V, sigma_V^(i))
V = expit(logit(V)) = 1 / (1 + exp(-logit(V)))
```

where the location parameter mu_V is composed of additive terms:

```
mu_V = alpha_V^(i) + B_agm^(i)
```

```
B_agm = B_mkt^(i)
      + gamma_A^(i) * I[A2 = rec_strike]
      + gamma_AH^(i) * I[A2 = rec_strike] * I[headline = 1]
      + gamma_D^(i) * f(D1)
```

The superscript (i) denotes posterior draw i from the upstream belief checkpoint.

**Change from v1:** The interaction term gamma_AH * I[rec_strike] * I[headline] is new. It captures the empirical finding that the ASA mobilisation effect is substantially larger in crisis scenarios than in routine ones. In v1, a single pooled gamma_A understated the ASA effect in crises and overstated it in non-crisis cases.

### 1.2 Parameter definitions

| Symbol | Name | Source | Description |
|--------|------|--------|-------------|
| alpha_V^(i) | `alpha_vote` | Stan posterior | Baseline opposition intercept on the logit scale |
| B_mkt^(i) | `B_mkt` | Stan posterior | Latent market belief at the checkpoint date |
| gamma_A^(i) | `gamma_A` | First-differences OLS + prior | Additive logit shift from ASA mobilisation (non-crisis baseline) |
| gamma_AH^(i) | `gamma_AH` | Subgroup OLS + prior | Additional logit shift from ASA mobilisation in headline-incident cases |
| gamma_D^(i) | `gamma_D` | Stan posterior | Scaling coefficient for governance action effect |
| sigma_V^(i) | `sigma_vote` | Stan posterior | Logit-scale noise (aleatoric uncertainty in voter behaviour) |
| f(D1) | Governance effect | Sampled at runtime | Effectiveness of Board's governance reform |

### 1.3 Governance effect (weakened sign constraint)

The governance effect f(D1) is drawn from a Uniform distribution whose bounds depend on the Board's action:

```
f(D1) =
  0                  if D1 = D0_minimal
  U(0, 1)            if D1 = D1_review
  U(-1, 0.5)         if D1 = D3_ceo_transition
```

When multiplied by gamma_D (negative from the posterior), the effects are:

- **D1 (review):** positive f times negative gamma_D lowers B_agm, reducing protest. A board review always helps, but the magnitude is uncertain.
- **D3 (CEO exit):** f ranges from -1 to +0.5. Negative values (multiplied by negative gamma_D) raise B_agm, amplifying protest. Positive values lower B_agm, reducing protest. The asymmetric bounds (-1 to +0.5 rather than -1 to +1) encode a prior belief that amplification is more likely than mitigation, but do not foreclose the possibility that a well-managed CEO transition reduces opposition.
- **D0 (no action):** no governance signal; baseline applies.

**Change from v1:** The D3 bounds were U(-1, 0) in v1, encoding a strict causal claim that CEO exit always amplifies protest. This has been relaxed to U(-1, 0.5) for two reasons:

1. *Identification weakness.* The amplification claim rests on 4 observations in the cross-company panel. These 4 companies had the most severe underlying crises in the sample. The high votes may reflect crisis severity rather than a causal effect of CEO departure. With only 4 observations, the data cannot distinguish "CEO exit amplifies protest" from "the worst crises produce both CEO exits and high votes."

2. *Plausible alternative mechanism.* In some scenarios, a decisive CEO transition signals accountability and draws a line under the crisis. The revised bounds allow f to take small positive values (up to 0.5), representing cases where the transition is well-managed and provides partial mitigation. The asymmetry (range of 1.5 on the negative side vs 0.5 on the positive side) encodes the prior judgment that amplification is roughly three times as likely as mitigation, without making it certain.

**Data anchoring:** The 4 D3 cases in the panel (RIO 2021, QBE 2021, ASX 2022, and one additional case) produced mean delta_vote of +46.7 pp, compared to +32.8 pp for D0 and +14.9 pp for D1. The D3 excess over D0 (+13.9 pp) is confounded with crisis severity. The D1 deficit relative to D0 (-17.9 pp) is more credibly causal because reviews are a less extreme response and are deployed across a wider range of crisis severities.

### 1.4 Structural floor for crisis scenarios

When headline_incident = 1, the model applies a structural floor to the vote fraction:

```
V_final = max(V_logit_normal, V_floor)
```

where V_floor ~ Beta(50, 150), giving mean 0.25 and 95th percentile approximately 0.30.

**Rationale:** In the 36-observation panel, P(first_strike | headline_incident = 1) = 15/15 = 100%. The lowest h=1 vote is MQG 2023 at 25.4%. A pure logit-normal with realistic parameter values assigns 3-14% probability to V < 0.25 even when the median vote is 50-60%, because the logit-normal has infinite support. The structural floor encodes the domain knowledge that headline incidents create a minimum level of opposition below which the vote essentially cannot fall, regardless of other factors. The floor is stochastic rather than fixed to avoid a mass point at exactly 0.25.

**Beta(50, 150) calibration:** The parameters are chosen so that the floor distribution has mean 0.25, mode 0.247, and 90% of its mass between 0.22 and 0.28. This is tight enough to enforce the near-certainty of a first strike in crisis scenarios, while allowing a small amount of uncertainty about the exact floor level. The effective sample size of 200 reflects high confidence in the floor, anchored to the 15/15 empirical rate.

**When the floor is not binding:** In the vast majority of crisis-scenario draws, the logit-normal V will exceed the floor and V_final = V_logit_normal. The floor only activates for draws in the extreme lower tail, preventing the model from generating implausible sub-25% crisis outcomes. For non-crisis scenarios (h=0), no floor is applied and the logit-normal operates without constraint.

### 1.5 Derived indicators

Two binary thresholds are applied to V_final (loaded from the `vote_thresholds` sheet of `governance_spec.xlsx`):

```
strike = I[V >= 0.25]
overwhelming = I[V >= 0.50]
```

A first strike (25%) triggers the "two strikes" rule under the Corporations Act 2001 s.250U-250V, and an overwhelming vote (50%) signals a governance crisis severe enough to threaten a board spill.

---

## 2. Parameter Estimation Pipeline

The vote model parameters flow through a three-stage estimation pipeline, from raw data to runtime draws. This revision changes Stage 2 substantially to address the bad control problem and non-independence issues identified in review.

### Stage 1: Upstream Stan models

Two Stan state-space models produce the foundational estimates. These are unchanged from v1.

**Media measurement model** (`models/media_better.stan`): Separates latent media coverage from intensity using sparse monthly observations. An AR(1) process on log-intensity evolves monthly:

```
log M_t = mu_logM + phi * (log M_{t-1} - mu_logM) + sigma_logM * z_t
```

The log-differences delta_log_M_t = log M_t - log M_{t-1} become the media shock series that enters the belief model.

**Belief dynamics model** (`models/belief_model.stan`): Tracks a latent shareholder distrust state B_t via AR(1) dynamics driven by media shocks, identified through three observation channels:

```
B_t = rho * B_{t-1} + beta * shock_t + sigma_B * z_t
```

Observation likelihoods:

| Channel | Likelihood | Anchoring |
|---------|-----------|-----------|
| Abnormal returns | abret_t ~ N(lambda_r * B_t, 1) | Free loading |
| Remuneration vote | logit(y_rem) ~ N(alpha_rem + B_t, sigma_rem) | lambda_rem = 1 (fixed) |
| Chair vote | logit(y_chair) ~ N(alpha_chair + lambda_chair * B_t, sigma_chair) | Free loading |

**Anchoring design choice:** Fixing lambda_rem = 1 pins the belief state to the logit scale of remuneration-report opposition. This means B_t is directly interpretable: a value of B_t = 0 corresponds to opposition at baseline alpha_rem, and each unit increase shifts logit(opposition) by 1. This anchoring propagates into the game tree without scale transformations.

**Prior for alpha_rem:** N(logit(0.10), 1.0). Centred at 10% baseline opposition (logit approximately -2.2). The 10% centre reflects the median remuneration opposition for ASX 200 companies in non-crisis years (typical range 3-15%, with most companies in single digits). The 1.0 SD on the logit scale allows the baseline to range from approximately 2.5% to 33% at the prior's central 90% interval, accommodating companies with persistently elevated or depressed opposition.

**Prior for sigma_rem:** half-t(4, 0, 0.6). The t(4) distribution has heavier tails than a half-Normal, allowing for occasional large observation noise. The scale of 0.6 reflects that logit-scale vote-to-belief residuals of order 0.3-0.8 are empirically common (corresponding to 5-15 pp of unexplained vote variation at moderate opposition levels). Values above 1.5 (implying 25+ pp of unexplained variation) are permitted but penalised.

### Stage 2: Data-driven prior for gamma_A and gamma_AH (revised)

The ASA mobilisation effects gamma_A (baseline) and gamma_AH (headline interaction) are estimated from the cross-company panel of 36 voting recommendations, using a single estimation channel (vote-channel OLS) in a first-differences specification.

**Change from v1:** Three modifications address issues identified in review:

1. *First-differences specification* replaces the levels regression, removing the bad control problem from conditioning on prior_year_pct.
2. *Single estimation channel* replaces the inverse-variance combination of vote and strike channels, removing the non-independence problem.
3. *ASA-by-headline interaction* is estimated directly, rather than using a pooled effect.

#### 2.1 First-differences specification

For the subset of companies with consecutive-year observations in the panel (approximately 8-10 pairs), estimate:

```
delta_logit(rem_against_pct) = b0 + b1 * delta_asa_against + b2 * delta_asa_against * headline_t
                                + b3 * delta_log_mkt_cap + epsilon
```

where delta denotes the year-on-year change within a company. The first-differencing eliminates time-invariant company characteristics (governance quality, ownership structure, investor base) that confound the ASA effect in the levels regression.

**Why first differences rather than fixed effects:** With only 2 years per company in most pairs, first differences and fixed effects are numerically equivalent. First differences make the identification strategy more transparent: the ASA effect is identified from companies that experience a *change* in ASA recommendation between consecutive years.

**Why prior_year_pct is removed:** In v1, the levels regression included prior_year_pct as a control. This is a bad control in the sense of Angrist and Pischke (2009, Ch. 3): prior-year opposition is itself partly caused by prior-year ASA recommendations. Conditioning on it attenuates the estimated current-year ASA effect by absorbing variation that should be attributed to the ASA's persistent influence. The first-differences specification avoids this entirely by differencing out the lagged outcome.

**Coefficient interpretation:**

- b1 = gamma_A: the logit-scale shift from ASA mobilisation in non-headline cases. Expected sign: positive (ASA recommendation increases opposition). Expected magnitude: 0.8-1.5 on the logit scale, corresponding to roughly 10-20 pp of additional opposition at moderate baseline levels.
- b2 = gamma_AH: the additional logit-scale shift in headline-incident cases. Expected sign: positive (the ASA effect is amplified in crises). Expected magnitude: 0.5-1.5 additional logit units.

**Data anchoring for expected magnitudes:** In the raw panel data, the median vote-against when ASA recommends against and h=1 is approximately 52% (logit 0.08), compared to approximately 12% (logit -2.0) when ASA does not recommend against. The raw gap of approximately 2.0 logit units overstates the causal effect (confounding with crisis severity), but even after attenuation the combined effect gamma_A + gamma_AH should be in the range of 1.0-2.5 logit units for crisis scenarios.

#### 2.2 Prior construction

The OLS estimates b1 and b2 are converted to Normal priors with inflated standard errors:

```
gamma_A ~ N(b1, sqrt(SE_1^2 + tau_A^2))       truncated below at 0
gamma_AH ~ N(b2, sqrt(SE_2^2 + tau_AH^2))     truncated below at 0
```

where tau_A and tau_AH are cross-company heterogeneity terms set to 0.35 * sigma_resid, following v1. The truncation at zero encodes the domain constraints that:

- ASA mobilisation cannot reduce shareholder opposition (gamma_A >= 0)
- The headline amplification cannot be negative (gamma_AH >= 0), because headline incidents increase the salience of ASA's message and the proportion of retail shareholders who follow the recommendation

**Why a single channel:** In v1, the vote-channel and strike-channel estimates were pooled via inverse-variance weighting, which assumes independence. Both channels use the same 36-observation panel, the same companies, and the same ASA recommendation variable. The estimation errors are correlated, and with n=36 the covariance is impractical to estimate reliably. Using the vote channel alone with inflated standard errors is more honest about the true precision. The strike channel serves as a qualitative consistency check but does not enter the formal prior.

#### 2.3 Calibration check: does the interaction produce enough separation?

The interaction term must produce vote distributions that match two empirical patterns:

**Pattern 1: Near-certain first strikes in crisis + ASA scenarios.** With h=1 and ASA recommending against, the combined shift gamma_A + gamma_AH must push the logit-scale mean high enough that the structural floor (Section 1.4) is rarely binding. Target: P(V > 0.25 | h=1, ASA against) > 0.99. This is achieved if the logit-scale mean exceeds approximately -0.5 (corresponding to V approximately 0.38) with sigma_V approximately 0.8, because P(logit(V) < logit(0.25)) = P(Z < (-1.1 - (-0.5))/0.8) = P(Z < -0.75) = 0.23, and the structural floor catches the remaining lower-tail draws. In practice, the logit-scale mean in crisis scenarios should be substantially higher than -0.5 (closer to 0.0-0.5, corresponding to V of 0.50-0.62), making the floor binding in fewer than 5% of draws.

**Pattern 2: Material first-strike risk but not certainty in non-crisis + ASA scenarios.** With h=0 and ASA recommending against, gamma_A alone (without gamma_AH) should produce a logit-scale mean that places the median vote near 35-45%, with a meaningful lower tail extending below 25%. Target: P(V > 0.25 | h=0, ASA against) in the range of 0.65-0.85. This matches the panel observation that most non-crisis ASA-against cases produce first strikes, but not all (some companies with strong institutional shareholder support or a credible board response fall short).

### Stage 3: Checkpoint construction

The posterior draws from the Stan model are combined with shock priors to produce belief checkpoints at five critical dates:

| Checkpoint | Date | Context | Market belief | Management belief |
|---|---|---|---|---|
| C_pre | Pre-crisis | Baseline | B_base | B_base |
| C0 | 2023-10-01 | Post-CEO resignation | B_sep | B_sep + gamma_E |
| C1 | 2023-10-10 | Review announced | B_sep + gamma_review | B_sep + gamma_E + gamma_review |
| C2 | 2023-10-18 | ASA mobilisation public | B_sep + gamma_review + gamma_A + gamma_AH | B_sep + gamma_E + gamma_review + gamma_A + gamma_AH |
| C3 | 2023-11-03 | Post-AGM (82.9% observed) | SMC-resampled | Aligned |

**Change at C2:** The ASA mobilisation checkpoint now includes both gamma_A and gamma_AH, since the Qantas case is a headline-incident scenario. In v1, only gamma_A was added at this checkpoint.

**Change at C3:** Importance resampling has been replaced with sequential Monte Carlo (SMC), as described in Section 2.4 below.

Each checkpoint contains 500 draws of all parameters (alpha_V, B_mkt, B_mgmt, gamma_A, gamma_AH, gamma_D, sigma_V) serialised in `.npz` files.

### 2.4 Sequential Monte Carlo at C3 (revised)

**Change from v1:** Simple importance resampling has been replaced with SMC to address potential degeneracy.

The v1 specification used importance resampling with weights:

```
w_i proportional to N(logit(0.829) | alpha_rem^(i) + B_C2^(i), sigma_rem^(i))
```

The problem: with an observed vote of 82.9% (logit 1.574), only draws where alpha_rem + B_C2 is near 1.574 receive non-negligible weight. If the prior places most mass well below 1.574 (likely, given that 82.9% is extreme), the effective sample size (ESS) after resampling collapses, potentially to single digits out of 500 draws.

**ESS diagnostic:** Before proceeding with any resampling scheme, compute:

```
ESS = (sum(w_i))^2 / sum(w_i^2)
```

If ESS >= 100 (out of 500 draws), simple importance resampling is adequate and the v1 approach can be retained. If ESS < 100, the SMC procedure described below is required.

**SMC procedure:** When ESS is insufficient, use a tempered likelihood sequence:

```
w_i^(t) proportional to [N(logit(0.829) | alpha_rem^(i) + B_C2^(i), sigma_rem^(i))]^(beta_t)
```

where beta_t increases from 0 to 1 across T tempering steps (typically T = 5-10). At each step:

1. Compute weights w_i^(t) using the current beta_t
2. If ESS < N/2, resample with replacement
3. Apply a small MCMC perturbation kernel (random walk on the parameter vector) to diversify particles
4. Increase beta_t

This gradually concentrates the particle cloud on draws consistent with the observed 82.9% vote, maintaining particle diversity throughout.

**Data anchoring for ESS concern:** The 82.9% Qantas vote corresponds to logit 1.574. The prior predictive for alpha_rem + B_C2 at the crisis checkpoint likely has a mean near 0.0-0.5 (corresponding to 50-62% opposition) with sigma approximately 1.0. The likelihood evaluated at logit 1.574 is then approximately N(1.574 | 0.25, 1.0) = 0.13, which is not catastrophically low, but the fraction of draws close enough to receive meaningful weight depends on the joint distribution's shape. The ESS diagnostic resolves this empirically.

---

## 3. Empirical Calibration of Governance Effects

### 3.1 The non-monotonic pattern

The governance effect f(D1) encodes an empirically observed non-monotonic relationship between Board actions and shareholder protest. The calibration data is the cross-company panel of ASX remuneration votes (`data/ranked_voting_recommendations.csv`, 36 company-years from 2021-2023), classified by Board corrective action:

| Board action | n | Mean delta_vote | Interpretation |
|---|---|---|---|
| D0 -- No action | 18 | +32.8 pp | Baseline escalation: passive board amplifies discontent |
| D1 -- Review/announcement | 8 | +14.9 pp | Most effective mitigation: review signals responsiveness |
| D3 -- CEO exit | 4 | +46.7 pp | Ambiguous: confounded with crisis severity |

**Change from v1:** The D3 interpretation has been revised from "crisis signal: sacking amplifies protest beyond baseline" to "ambiguous: confounded with crisis severity." The quantitative encoding has been weakened accordingly (Section 1.3).

**Data anchoring for the D1 effect:** The D1 deficit relative to D0 (-17.9 pp in raw terms) is the most credible estimate in the table because:

- Reviews are deployed across a range of crisis severities (from moderate governance concerns to major incidents), reducing confounding with underlying severity.
- The 8 D1 cases span multiple industries and company sizes, providing reasonable cross-sectional variation.
- The mechanism is intuitive: a review signals responsiveness without confirming that the problem was severe enough to require a CEO change.

**Data anchoring for the D3 effect:** The D3 excess over D0 (+13.9 pp in raw terms) is unreliable as a causal estimate because:

- The 4 D3 cases are the most severe crises in the sample (RIO 2021, QBE 2021, ASX 2022, and one additional). Selection into CEO exit is driven by crisis severity.
- With n=4, the standard error of the mean delta_vote is large (approximately 10 pp), so the D3-D0 difference is not statistically significant.
- The latent belief state B_t in the Stan model partially controls for crisis severity, but a single-dimensional state cannot fully capture the multidimensional nature of governance crises (type of misconduct, media salience, regulatory involvement, prior history).

### 3.2 Encoding as scaled Uniform draws (revised bounds)

- **D1 (review):** f ~ U(0, 1), expected value 0.5. A review always reduces protest relative to baseline. The Uniform over the full unit interval captures genuine uncertainty about the magnitude of mitigation. Some reviews are comprehensive and credible (e.g., Star Casino, leading to major governance reform), while others are perfunctory announcements with limited follow-through.

- **D3 (CEO exit):** f ~ U(-1, 0.5), expected value -0.25. The asymmetric bounds encode three beliefs: (a) amplification is more likely than mitigation (2/3 of the distribution mass is negative), (b) the worst-case amplification (f = -1) is as severe as the best-case review (f = 1 for D1), and (c) a well-managed transition can provide partial mitigation (up to f = 0.5), but not as much as a review at its best.

  **Calibration of the 0.5 upper bound:** The upper bound reflects that even in the best case, a CEO transition carries costs (market disruption, succession uncertainty, implicit confirmation of severity) that limit its mitigating potential relative to a review. Capping at 0.5 rather than 1.0 says "even a perfectly managed CEO exit is at most half as effective as a perfectly credible review."

- **D0 (no action):** f = 0 deterministically. No governance signal means no effect on B_agm beyond what B_mkt already captures.

### 3.3 Epistemic vs. aleatoric separation

The governance effect f(D1) is drawn once per scenario per belief draw and held fixed across Monte Carlo vote samples. This reflects the distinction:

- **Epistemic uncertainty** (drawn once): "How effective is this particular governance reform?" This is a property of the reform itself and does not change between individual voter decisions.
- **Aleatoric uncertainty** (drawn M_V times): "Given the true effectiveness, what fraction of shareholders actually vote against?" This is irreducible noise from aggregating millions of individual voting decisions.

In the tree evaluation, 50 vote samples (aleatoric) are drawn per governance-effect draw (epistemic), and the Monte Carlo average approximates:

```
E[U | f] = (1/M_V) * sum_{m=1}^{M_V} U(expit(alpha_V + B_agm(f) + sigma_V * epsilon_m))
```

---

## 4. Design Justification

### 4.1 Why logit-normal?

The logit-normal distribution is the natural choice for modelling a proportion (vote fraction) that arises from aggregating many individual binary decisions. Four properties make it ideal here:

1. **Bounded support.** V in (0, 1) by construction. A Normal model on the raw vote percentage would assign positive probability to votes below 0% or above 100%.

2. **Additive effects on the logit scale.** The belief state B_t, ASA mobilisation effects gamma_A and gamma_AH, and governance effects gamma_D * f are all additive on the logit scale. This is natural for latent-variable models where multiple factors contribute to a probability. The logit link linearises interaction effects.

3. **Consistency with the Stan identification.** The belief model anchors lambda_rem = 1, meaning the belief state B_t lives on the logit scale of remuneration opposition. Using a logit-normal vote model means the game tree operates on exactly the same scale as the upstream Bayesian estimation, requiring no scale transformations.

4. **Flexible shape.** Unlike the Beta distribution, the logit-normal can be parameterised directly through regression-style predictors. The logit-normal also handles extreme probabilities (near 0 or 1) through the long tails on the logit scale, producing appropriate concentration near the boundaries on the probability scale.

### 4.2 Why logit-normal plus structural floor rather than a mixture model?

An alternative to the structural floor (Section 1.4) would be a mixture model:

```
V ~ pi * TruncatedLogitNormal(mu_V, sigma_V, lower=0.25) + (1-pi) * LogitNormal(mu_V, sigma_V)
```

where pi is the probability of the "crisis regime" in which the vote cannot fall below 25%. This is more elegant in theory but introduces an additional parameter (pi) that must be estimated, and creates discontinuities in the likelihood surface that complicate MCMC. The structural floor achieves the same practical effect (near-zero probability of V < 0.25 in crisis scenarios) with a simpler implementation that does not alter the parameter estimation pipeline.

### 4.3 Why separate gamma_A, gamma_AH, and gamma_D?

The model distinguishes three effects for distinct reasons:

**gamma_A vs gamma_D: different causal mechanisms.** ASA mobilisation operates through information transmission: retail shareholders who would not otherwise attend to the vote learn about governance concerns and are prompted to vote against. Governance actions operate through belief revision: shareholders update their assessment of the severity of the underlying problem. These mechanisms are empirically separable (ASA can recommend against with or without board action, and vice versa).

**gamma_A vs gamma_AH: different regimes.** The ASA effect operates differently in crisis vs non-crisis contexts:

- In non-crisis cases (h=0), the ASA recommendation is often the *primary* driver of whether the 25% threshold is crossed. Without ASA mobilisation, most non-crisis companies see single-digit opposition. The ASA recommendation shifts the distribution from a median of approximately 10-15% to a median of approximately 35-45%.
- In crisis cases (h=1), the vote exceeds 25% regardless of ASA involvement (the structural floor encodes this). The ASA recommendation amplifies the *magnitude* of opposition, shifting the median from approximately 35-40% to approximately 50-55%, and increasing the probability of an overwhelming (50%+) outcome.

**Data anchoring:** In the panel data, the gap between ASA-against and ASA-not-against median opposition is approximately 25-30 pp in non-crisis cases but approximately 15-20 pp in crisis cases (where the baseline is already elevated). On the logit scale, however, the crisis-case shift is larger because the logit function stretches differences near the extremes. A 40% to 55% shift (logit -0.41 to 0.20) is 0.61 logit units, while a 12% to 40% shift (logit -2.0 to -0.41) is 1.59 logit units. The interaction term gamma_AH captures the additional logit-scale shift in crises.

### 4.4 Why not estimate gamma_D from the cross-company panel?

The cross-company panel provides descriptive statistics on governance effects (Section 3.1), but gamma_D is estimated within the Stan state-space model rather than from the panel. The reasons:

1. **Endogeneity.** In the cross-company data, boards facing worse crises take more aggressive action. The raw correlation between D3 and high votes overstates the causal effect because it confounds the board's response with crisis severity. The Stan model partially addresses this by conditioning on the latent belief state, though the conditioning is imperfect (see Section 3.1 data anchoring).

2. **Scale consistency.** The Stan model produces gamma_D on exactly the logit scale needed for the vote model, jointly estimated with B_t, alpha_V, and sigma_V. A separately estimated coefficient would break the joint posterior structure.

3. **Qualitative constraint only.** The cross-company data establishes the non-monotonic sign pattern (D1 positive, D3 likely negative) and provides a plausibility check on magnitudes. The directional information enters through the bounds on f(D1), not through gamma_D itself.

### 4.5 Why Uniform for governance effects?

The Uniform distributions for f(D1) reflect maximum ignorance about magnitude, conditional on direction:

- **Direction is partially known.** The empirical evidence (Section 3.1) establishes that reviews reduce protest. For CEO exits, the direction is ambiguous, hence the wider bounds U(-1, 0.5) that allow both signs.

- **Magnitude is genuinely uncertain.** The effectiveness of a governance review depends on scope, independence, market credibility, and timing. The Star Casino review had dramatically different impact from the Qantas review. A Uniform over a wide interval captures this without imposing a preferred magnitude.

- **Interaction with gamma_D.** The overall vote effect is gamma_D * f, where gamma_D is informed by the posterior. The Uniform on f means the magnitude uncertainty is resolved at runtime. Each simulation draw commits to a specific effectiveness level, which then interacts with the posterior estimate of vote sensitivity to governance signals.

### 4.6 Why five checkpoints?

The checkpoint structure reflects information asymmetry between market participants and Qantas management:

| Checkpoint | Information state | Key asymmetry |
|---|---|---|
| C_pre | Pre-crisis baseline | None: common beliefs |
| C0 | CEO resignation announced (05-Sep) | Management knows private ASA engagement (gamma_E); market does not |
| C1 | Review announced (10-Oct) | Common knowledge of review; management still has private gamma_E signal |
| C2 | ASA goes public (18-Oct) | gamma_A + gamma_AH become common knowledge; information gap narrows |
| C3 | AGM observed (03-Nov) | Common knowledge of 82.9% vote; posterior sharpened via SMC |

---

## 5. Board Overconfidence Bias on the Vote

When the Board is the focal actor, cognitive biases distort its perception of the vote distribution. These are calibrated from the governance overconfidence literature.

### 5.1 Overestimation of governance effectiveness

The Board overestimates how much its actions reduce protest by a factor beta ~ U(0.25, 1.0) (production midpoint beta = 0.625):

```
f_bias(D1 = review) ~ U(0.63, 1.0)       vs. unbiased U(0, 1)
f_bias(D1 = CEO exit) ~ U(-0.62, 0.5)    vs. unbiased U(-1, 0.5)
```

The biased Board believes a review is at least 63% effective (floor raised from 0) and that CEO exit backlash is at most 62% of the true worst case (floor raised from -1.0 to -0.62). The upper bound for D3 remains at 0.5: the Board's optimism affects the downside, not the upside.

**Data anchoring:** The beta range U(0.25, 1.0) is calibrated from three sources:

- Twardawski and Kind (2023): boards overestimate the value of M&A decisions by 20-40%, suggesting a floor of approximately 0.6-0.8 for governance effectiveness overestimation.
- Coffeng et al. (2021): approximately 20% of boards select the objectively best option in structured decisions, implying that 80% are overconfident about the quality of their chosen action.
- The lower bound of 0.25 allows for boards that are only mildly overconfident; the upper bound of 1.0 allows for boards that are perfectly calibrated (no bias). The U(0.25, 1.0) distribution places more mass on moderate-to-high bias than on low bias, with a mean of 0.625.

### 5.2 Overprecision on vote uncertainty

The Board perceives kappa ~ U(2, 5) times more precision than warranted:

```
sigma_V_hat = sigma_V / sqrt(kappa)
```

Production default: kappa = 3.5, giving sigma_V_hat = 0.53 * sigma_V. The Board's subjective confidence intervals are roughly half the width of the true intervals, leading it to underestimate the probability of extreme vote outcomes.

**Data anchoring:** The kappa range is calibrated from:

- Boundy-Singer et al. (2022): metacognitive miscalibration studies show that organisational leaders' subjective 90% confidence intervals contain the true value only 40-60% of the time, implying overprecision factors of 2-5x on variance.
- The U(2, 5) range spans "moderate overconfidence" (confidence intervals twice as narrow as they should be) to "severe overconfidence" (intervals that are less than half the true width).

### 5.3 Bias propagation

Both biases are applied consistently:

- In the focal actor's own expected utility calculation (biased f and biased sigma_V flow into the vote model)
- In rollout simulations for predictive distributions (opponent models see the biased Board acting on its biased beliefs)

This produces self-consistent decision-making under cognitive bias: the Board genuinely believes its optimistic assessment, and opponents model the Board as believing it.

---

## 6. How the Vote Feeds Into Utility Functions

The vote outcome V (and derived indicators) enters all three actors' utility functions, creating the strategic tension that drives the game. These are unchanged from v1.

### 6.1 Board utility

The Board is penalised for high opposition through four channels:

```
u_B includes: -w_vote * max(V - 0.25, 0)^2
              -w_over * I[overwhelming]
              -w_spill * V * I[strike]
              -w_second * I[strike AND CEO_present]
```

| Component | Weight | Mechanism | Data anchoring |
|---|---|---|---|
| Quadratic vote penalty | w_vote = 2.0 | Convex cost: marginal damage increases with opposition | Convexity reflects that a 30% vote is mildly embarrassing while an 80% vote is catastrophic. The 2.0 scaling is normalised relative to other utility components |
| Overwhelming penalty | w_over = 3.0 | Fixed cost at V >= 50%: governance crisis threshold | The 50% threshold is the point at which media coverage shifts from "governance concern" to "governance crisis" and board spill becomes a realistic prospect |
| Spill risk | w_spill = 2.5 | Proportional to V once strike threshold crossed | Scaling with V reflects that a 26% strike and a 49% strike carry different spill probabilities |
| Second-strike spill | w_second = 8.0 | CEO still present after first strike leads to near-certain spill next year | The large weight (8.0) reflects the Corporations Act two-strikes rule: a second consecutive 25%+ vote triggers a spill resolution. The CEO's continued presence makes a second strike near-certain |

### 6.2 ASA utility

ASA benefits from high opposition through complementary channels:

```
u_A includes: +w_vote_A * V
              +w_over_A * I[overwhelming]
              +w_rep_A * max(V - 0.25, 0)
              +w_align_A * I[rec_strike AND strike]
```

| Component | Weight | Mechanism | Data anchoring |
|---|---|---|---|
| Vote reward | w_vote_A = 2.0 | Linear benefit from opposition | Linear rather than convex because the ASA benefits from every percentage point of demonstrated shareholder concern |
| Overwhelming bonus | w_over_A = 2.0 | Threshold reward for governance crisis signal | The 50% threshold is a media-attention trigger that amplifies the ASA's institutional visibility |
| Reputational gain | w_rep_A = 1.0 | Bonus for high-profile outcome above 25% | The 25% threshold is the statutory trigger that validates the ASA's recommendation |
| Market alignment | w_align_A = 1.5 | Credibility bonus when ASA recommendation matches market action | Anchored to the panel observation that in 100% of h=1 cases, the market votes a first strike. ASA's institutional credibility depends on leading, not lagging, this consensus |

### 6.3 CEO utility

The vote creates non-monetary penalties that scale with loss aversion (lambda_D = 2.25):

```
D_raw includes: D_agm * I[V > 0.25] + D_disgrace * I[overwhelming]
```

| Component | Raw value | Loss-aversion-scaled | Data anchoring |
|---|---|---|---|
| AGM humiliation | D_agm = 30 | lambda_D * 30 = 67.5 | Scaled relative to the CEO's annual compensation utility. A first strike produces public humiliation, media scrutiny, and questions about the CEO's mandate |
| Public disgrace | D_disgrace = 30 | lambda_D * 30 = 67.5 | An overwhelming vote (50%+) transforms a governance concern into a personal crisis for the CEO, with career-long reputational consequences |

**Loss aversion calibration:** lambda_D = 2.25 follows Tversky and Kahneman (1992). Under reference-dependent CRRA, the CEO evaluates these penalties relative to their expected status as a powerful executive. Losses loom 2.25x larger than equivalent gains, creating a strong incentive to resign before the AGM if a high vote is anticipated.

---

## 7. Monte Carlo Integration

### 7.1 Runtime procedure

At the V node in the game tree, the TreeEvaluator performs Monte Carlo integration:

1. **Draw governance effect once** (epistemic): f ~ U(l, u) based on the D1 action and any overconfidence bias.
2. **Draw M_V = 50 vote samples** (aleatoric): For each sample m:
   - Construct B_agm from the belief draw and action history (including gamma_A, gamma_AH if applicable)
   - Draw logit(V_m) ~ N(alpha_V + B_agm, sigma_V)
   - Apply expit to get V_m in (0, 1)
   - If headline_incident = 1: apply V_m = max(V_m, V_floor) where V_floor ~ Beta(50, 150) drawn once per belief draw
   - Derive strike and overwhelming indicators
3. **Average downstream utility** across all 50 samples:

```
U_i^(V)(h) = (1/50) * sum_{m=1}^{50} U_i^(next)(h union {V_m, strike_m, overwhelming_m})
```

**Note on the structural floor draw:** V_floor is drawn once per belief draw (epistemic), not once per vote sample. The floor represents a structural property of the crisis scenario (the minimum level of opposition that the incident generates), not a per-vote random variable.

### 7.2 Computational budget

Each vote node evaluation requires 50 forward recursions through the remaining game tree. Across the full solver:

- **Per belief draw:** 3 initial actions x 50 vote samples = 150 tree evaluations at the vote node
- **Per checkpoint:** N belief draws x 150 = typically 75,000 vote-node evaluations
- **Per predictive distribution call:** K = 200 opponent samples x R = 20 rollouts x 50 vote samples = 200,000 vote draws

The logit-normal sampling is computationally trivial (one Normal draw + one expit transform + one max operation per sample). The cost is dominated by the downstream tree recursion.

---

## 8. Numerical Implementation Details

### 8.1 Numerically stable expit

The inverse-logit function uses a branch-stable implementation:

```python
def _expit(x):
    if x >= 0:
        z = exp(-x)
        return 1 / (1 + z)
    else:
        z = exp(x)
        return z / (1 + z)
```

For large positive x, exp(-x) approaches 0 and the result approaches 1. For large negative x, exp(x) approaches 0 and the result approaches 0. Neither branch computes exp(|x|) for large |x|, avoiding float64 overflow.

### 8.2 Sigma floor

The logit-scale noise has a floor: sigma_V = max(sigma_V, 1e-6). This prevents degenerate zero-variance draws from the posterior from producing division-by-zero or deterministic vote outcomes.

### 8.3 Vote threshold precision

The thresholds (0.25 and 0.50) are loaded from Excel, not hard-coded. This allows sensitivity analysis over alternative threshold definitions without code changes.

### 8.4 Structural floor implementation

The crisis floor is implemented as a post-hoc max operation rather than a truncated distribution, to keep the parameter estimation pipeline (which does not include the floor) cleanly separated from the runtime simulation (which does). This means the posterior draws of alpha_V, B_mkt, gamma_A, gamma_AH, gamma_D, and sigma_V are estimated without the floor constraint, and the floor is applied only at the Monte Carlo integration stage.

This separation is intentional: the Stan model estimates parameters from historical data that includes both crisis and non-crisis outcomes, while the floor encodes domain knowledge about crisis-specific dynamics that is not captured in the Stan likelihood.

---

## 9. Validation

### 9.1 Empirical calibration checks

At checkpoint C0 (the primary analysis point), the vote model must satisfy three calibration targets:

**Target 1: Qantas outcome.** The observed 82.9% vote should fall within the central 90% of the predictive distribution at C0 with ASA recommending against, D1 review, and headline_incident = 1. The model was not fitted to the Qantas vote (Qantas is excluded from the cross-company panel).

**Target 2: First-strike rate in crisis scenarios.** Across Monte Carlo draws at C0 with h=1 and ASA recommending against, the fraction of draws producing V >= 0.25 should exceed 0.99. This is enforced jointly by the logit-scale mean (which places the median vote well above 25%) and the structural floor (which catches extreme lower-tail draws).

**Target 3: Overwhelming-vote rate in crisis + ASA scenarios.** The fraction of draws producing V >= 0.50 should be approximately 0.40-0.60, matching the panel observation that roughly half of h=1 ASA-against cases produce overwhelming votes (7 of 14 exceed 50%).

**Target 4: ASA differentiation.** The difference in median V between ASA-against and ASA-not-against scenarios (holding h and D1 constant) should be at least 15 pp in non-crisis cases and at least 10 pp in crisis cases (where the baseline is already high).

### 9.2 Test coverage

The vote model has 11 dedicated unit tests (expanded from 8 in v1):

| Test | Validates |
|---|---|
| `test_vote_model_basic` | Output range V in (0, 1) and boolean indicators |
| `test_vote_increases_with_strike_recommendation` | gamma_A > 0 shifts vote upward |
| `test_vote_headline_interaction` | gamma_AH provides additional shift when h=1 (new) |
| `test_vote_non_monotonic_governance_effect` | D1 reduces and D3 can increase protest |
| `test_governance_effect_d3_allows_mitigation` | D3 f can be positive, up to 0.5 (new) |
| `test_governance_effect_respects_bias` | Biased D1 bounds shift correctly |
| `test_governance_effect_d3_bias` | Biased D3 bounds shift correctly (updated for new bounds) |
| `test_d0_unaffected_by_bias` | D0 always returns 0 regardless of bias |
| `test_sigma_scale_reduces_vote_variance` | Overprecision reduces vote spread |
| `test_sigma_scale_1_is_identity` | No-bias case is identity |
| `test_crisis_floor_prevents_sub_threshold` | V_final >= V_floor when h=1 (new) |

---

## 10. Summary of Changes from v1

| Component | v1 | v2 | Rationale |
|---|---|---|---|
| ASA effect | Single gamma_A | gamma_A + gamma_AH interaction | ASA effect differs substantially between crisis and non-crisis; pooled estimate understated crisis impact |
| gamma_A estimation | Levels OLS with prior_year_pct + inverse-variance pooling of two channels | First-differences OLS, single vote channel | Removes bad control (prior_year_pct is downstream of prior ASA recommendations) and removes non-independence from pooling correlated estimates |
| Crisis vote floor | None | V_floor ~ Beta(50, 150) applied when h=1 | 15/15 crisis cases produced first strikes; logit-normal alone assigns 3-14% probability to V < 0.25 in realistic crisis parameterisations |
| D3 governance effect | f ~ U(-1, 0) | f ~ U(-1, 0.5) | Strict amplification claim rested on 4 confounded observations; revised bounds allow partial mitigation while maintaining prior toward amplification |
| C3 resampling | Simple importance resampling | SMC with ESS diagnostic | Importance resampling may degenerate when the observed 82.9% vote is in the prior's extreme tail; SMC maintains particle diversity |
| Unit tests | 8 tests | 11 tests | Three new tests for interaction term, relaxed D3 bounds, and crisis floor |

---

## 11. References

**Statistical methodology:**
- Gelman, A., Jakulin, A., Pittau, M. G., & Su, Y.-S. (2008). A weakly informative default prior distribution for logistic and other regression models. *Annals of Applied Statistics*, 2(4), 1360-1383.
- Aitchison, J. & Shen, S. M. (1980). Logistic-normal distributions. *Biometrika*, 67(2), 261-272.
- Angrist, J. D. & Pischke, J.-S. (2009). *Mostly Harmless Econometrics*. Princeton University Press. Ch. 3 (bad controls).

**Overconfidence calibration:**
- Twardawski, T. & Kind, A. (2023). Board overconfidence in mergers and acquisitions. *Journal of Business Research*.
- Coffeng, T. et al. (2021). Board decision-making quality. *Corporate Governance*.
- Boundy-Singer, Z. et al. (2022). Metacognitive miscalibration in organizational leaders. *Organizational Behavior and Human Decision Processes*.

**Australian governance:**
- Corporations Act 2001 (Cth) ss.250U-250V.
- Ertimur, Y., Ferri, F., & Stubben, S. (2011). Board of directors' responsiveness to shareholders. *Journal of Corporate Finance*.
- Tversky, A. & Kahneman, D. (1992). Advances in prospect theory. *Journal of Risk and Uncertainty*, 5(4), 297-323.

**Data sources:**
- `data/ranked_voting_recommendations.csv`: 36 ASX company-years (2021-2023)
- `data/voting-recommendations.csv`: Extended cross-company panel
- `data/checkpoints/belief_C*.npz`: Posterior draws (500 per checkpoint)
- `data/governance_spec.xlsx`: Vote thresholds, utility weights, game tree parameters
