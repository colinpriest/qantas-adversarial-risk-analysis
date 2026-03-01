# Bayesian Definitions

This document catalogues every Bayesian model, formula, and distributional assumption used in the Qantas Adversarial Risk Assessment (ARA) pipeline.

---

## 1. Market Model for Abnormal Returns

**File:** `compute_abnormal_returns.py`

**Formula (OLS, not Bayesian in estimation but produces a key input):**

```
r_QAN,t = alpha + beta * r_MKT,t + epsilon_t
```

where `epsilon_t ~ N(0, sigma^2)`.  Abnormal return is the residual:

```
AR_t = r_QAN,t - (alpha_hat + beta_hat * r_MKT,t)
```

- **Purpose:** Isolate Qantas-specific return shocks from broad market movements. The monthly aggregation of these abnormal returns becomes an observable signal in the belief model.
- **Outcome:** `AR_t` — the portion of Qantas's daily log-return unexplained by the ASX 200 index. Positive values indicate idiosyncratic outperformance; negative values indicate idiosyncratic underperformance.
- **Conditions (inputs):** Daily log-returns for Qantas (`r_QAN`) and the ASX 200 (`r_MKT`), over the estimation window Oct 2020 – Dec 2023.
- **Data:** `data/qantas_share_price_data.json` — daily closing prices for Qantas and the ASX 200 index.
- **Parameter reasoning:** Standard single-factor market model. Alpha and beta are estimated by OLS from the full estimation window (static fit by default), which is the conventional approach in event-study methodology.

---

## 2. Media Measurement Model (Stan)

**File:** `models/media_better.stan`, fitted by `fit_media_better_stan.py`

This is a state-space model that separates latent media *coverage* (how much the topic appears) from latent media *intensity* (how strong or negative the coverage is) using sparse monthly observations.

### Latent States

**Coverage** — logistic trend with floor:

```
C_t = 0.02 + 0.96 * logit^{-1}(aC + bC * t_scaled)
```

The floor (0.02) and ceiling (0.98) prevent coverage from reaching exactly zero or one.

**Intensity** — AR(1) process on the log scale:

```
logM_1     = mu_logM + (sigma_logM / sqrt(1 - phi^2)) * z_1
logM_t     = mu_logM + phi * (logM_{t-1} - mu_logM) + sigma_logM * z_t
```

where `z_t ~ N(0, 1)` are standard normal innovations (non-centred parameterisation).

### Observation Model

```
log(y_i + eps) ~ N(logC_{t_i} + logM_{t_i}, sigmaY)
```

where `y_i` is the observed media intensity at time index `t_i`, and `eps` is a small constant to handle zero observations.

### Priors

| Parameter | Prior | Reasoning |
|-----------|-------|-----------|
| `aC` | `N(0, 1)` | Weakly informative; coverage trend intercept centred at 50% on logistic scale |
| `bC` | `N(0, 1)` | Weakly informative; allows both increasing and decreasing coverage trends |
| `mu_logM` | `N(mean(log(y)), 0.8)` | **Data-anchored.** Centred on the empirical mean of observed log-intensities. The 0.8 SD allows moderate departure from the observed average |
| `sigma_logM` | `half-N(0, 0.35)` | Weakly informative half-normal. Constrains month-to-month intensity volatility to plausible ranges; 0.35 chosen to allow substantial but not explosive variation on the log scale |
| `phi_raw` | `N(0, 0.8)` | Raw persistence parameter, mapped to `phi = 0.98 * tanh(phi_raw)`. The N(0, 0.8) prior places most mass on moderate persistence values; the tanh mapping constrains phi to (-0.98, 0.98) ensuring stationarity |
| `sigmaY` | `half-N(0, 0.8)` | Observation noise on the log scale. Allows for substantial measurement error in media intensity data, reflecting the inherent noisiness of monthly media counts |

### Outcome

- `C_t` — posterior distribution of media *coverage* at each month (probability that the topic is actively covered)
- `logM_t` — posterior distribution of log media *intensity* (underlying strength of coverage, conditional on it existing)

### Data

`data/monthly_media_variables.xlsx` — sparse monthly observations of media intensity variables for Qantas governance-related coverage. The sparsity (many months with no observation) motivates the state-space approach: the AR(1) latent process interpolates through gaps.

### Downstream Use

Log-differences of the posterior intensity draws become the media shock series:

```
shock_t = logM_t - logM_{t-1}
```

(computed in `compute_media_shocks.py`). These shocks enter the belief model as an exogenous input.

---

## 3. Belief Dynamics Model (Stan)

**File:** `models/belief_model.stan`, fitted by `fit_belief_model_stan.py`

This is the core state-space model. It tracks a latent shareholder *belief state* `B_t` over time, identified through three observation channels: abnormal returns, remuneration-report votes, and chair-election votes.

### Anchoring Constraint

`lambda_rem = 1` (fixed, not estimated). This anchors the belief state to the logit scale of remuneration-report opposition. Without this constraint the model would be unidentified (the scale of `B_t` is arbitrary in a latent-variable model). With it, a belief state of `B_t = 0` means the remuneration-report opposition is at baseline `alpha_rem`, and each unit increase in `B_t` shifts the logit of opposition by 1.

### State Evolution

```
rho     = 0.95 * tanh(rho_raw)

B_1     = sigma_B0 * zB_1 + beta * shock_1
B_t     = rho * B_{t-1} + beta * shock_t + sigma_B * zB_t    (t >= 2)
```

where `shock_t` is the (z-scored) media shock from the media model, and `zB_t ~ N(0, 1)` are standard normal innovations (non-centred parameterisation).

### Observation Likelihoods

**Abnormal returns** (z-scored):

```
abret_t ~ N(lambda_r * B_t, 1.0)
```

The unit variance comes from the z-scoring; `lambda_r` captures how strongly the belief state drives abnormal returns.

**Remuneration-report votes** (logit-normal, anchored):

```
logit(y_rem_i) ~ N(alpha_rem + 1 * B_{t_i}, sigma_rem)
```

The loading is fixed at 1 (the anchoring constraint). `alpha_rem` is the baseline opposition level on the logit scale.

**Chair-election votes** (logit-normal, free loading):

```
logit(y_chair_i) ~ N(alpha_chair + lambda_chair * B_{t_i}, sigma_chair)
```

`lambda_chair` is estimated, allowing the chair-election to respond differently to shareholder distrust than the remuneration vote.

### Priors

| Parameter | Prior | Reasoning |
|-----------|-------|-----------|
| `rho_raw` | `N(0, 0.4)` | Mapped to `rho = 0.95 * tanh(rho_raw)`. The 0.4 SD concentrates mass around moderate persistence; the 0.95 ceiling prevents exact unit-root behaviour |
| `beta` | `N(0, 0.7)` | Shock-to-belief sensitivity. Weakly informative; permits both positive and negative media-shock effects. The 0.7 SD is wide enough to be uninformative about sign or magnitude |
| `sigma_B` | `half-t(4, 0, 0.25)` | State evolution noise. The Student-t(4) prior has heavier tails than a half-normal, accommodating occasional large innovations while keeping the median small (0.25 scale). This reflects an expectation that beliefs usually evolve smoothly but can occasionally jump |
| `sigma_B0` | `half-t(4, 0, 0.35)` | Initial-state noise. Slightly wider than `sigma_B` to accommodate greater uncertainty about the starting point of the belief process |
| `lambda_r` | `N(0, 0.5)` | Abnormal-return loading. Weakly informative; the 0.5 SD is appropriate given that both `B_t` and `abret_t` are on standardised scales |
| `alpha_rem` | `N(logit(0.10), 1.0)` | **Domain-informed.** Baseline remuneration-report opposition is centred at 10% (logit ≈ -2.2), reflecting that most companies see single-digit opposition in normal times. The 1.0 SD on the logit scale translates to substantial uncertainty about the baseline rate |
| `sigma_rem` | `half-t(4, 0, 0.6)` | Remuneration-vote observation noise. The 0.6 scale permits moderate variability in how accurately the vote reflects underlying beliefs |
| `alpha_chair` | `N(logit(0.02), 1.5)` | **Domain-informed.** Baseline chair opposition is centred at 2%, reflecting that chair votes rarely attract significant opposition. The wider SD (1.5) reflects greater uncertainty about this baseline |
| `lambda_chair` | `N(0, 0.7)` | Chair-vote belief loading. Weakly informative; allows the chair-election to be more or less sensitive to distrust than the remuneration vote |
| `sigma_chair` | `half-t(4, 0, 0.8)` | Chair-vote observation noise. Wider than `sigma_rem` because chair votes are rarer and noisier data points |

### Outcome

- `B_t` — posterior distribution of latent shareholder distrust at each month, in logit-units of remuneration-report opposition
- All loading and noise parameters — posterior distributions enabling forward simulation and measurement updates

### Data

- `data/abret_monthly.csv` — monthly aggregated abnormal returns (from Step 1)
- `data/media_shock_draws.npz` — posterior media shock draws (from Steps 4–5)
- `data/agm-votes.csv` — historical AGM remuneration and chair vote outcomes for Qantas

### Conditions (inputs)

The model is conditioned on:
- The full monthly time series of (z-scored) media shocks
- The full monthly time series of (z-scored) abnormal returns
- Sparse AGM vote observations (typically one per year) as point-in-time measurements

---

## 4. Data-Driven Shock Priors for gamma_A

**File:** `fit_shock_priors.py`

These priors quantify the expected impact of ASA (Australian Shareholders' Association) public mobilisation on shareholder voting outcomes. They are fitted from a cross-company panel of voting recommendations and outcomes.

### 4a. Vote Channel (OLS on logit scale)

**Formula:**

```
logit(rem_against_pct_i) = b_0 + b_1 * asa_against_i + b_2 * prior_year_pct_i
                           + b_3 * log_mkt_cap_i + gics_FE + epsilon_i
```

**Prior derived:**

```
gamma_A^{vote} ~ N(b_1_hat, sqrt(SE_1^2 + (0.35 * sigma_resid)^2))
```

- **Purpose:** Estimate the additive effect of an ASA "against" recommendation on the logit of remuneration-report opposition, controlling for firm size, prior-year opposition, and industry.
- **Outcome:** `b_1_hat` — the estimated logit-scale shift in remuneration opposition when ASA recommends voting against. This becomes the mean of the prior for `gamma_A`.
- **Conditions:** `asa_against` (binary: did ASA recommend against?), `prior_year_pct` (previous year's opposition rate), `log_mkt_cap` (log market capitalisation), and GICS industry fixed effects.
- **Data:** `data/voting-recommendations.csv` — cross-company panel of ASA voting recommendations and corresponding AGM outcomes. As-of filtering ensures only data available *before* the relevant checkpoint date is used, preventing look-ahead bias.
- **Parameter reasoning:** The extra variance term `(0.35 * sigma_resid)^2` inflates the standard error to account for cross-company heterogeneity that OLS standard errors understate. The 0.35 multiplier is a conservative choice — enough to prevent overconfidence in the point estimate without rendering the prior uninformative.

### 4b. Strike Channel (Bayesian Logistic Regression)

**Formula:**

```
first_strike_i ~ Bernoulli(logistic(X_i * beta))

beta ~ N(0, 2.5^2 * I)
```

where `X_i` includes intercept, `asa_against`, `prior_year_pct`, `log_mkt_cap`, and industry dummies.

Fitted using MAP (Newton's method on the penalised log-likelihood) with Laplace approximation for the posterior covariance.

**Prior derived:**

```
gamma_A^{strike} ~ N(beta_1_MAP, sqrt(SE_{Laplace,1}^2 + 0.10^2))
```

- **Purpose:** Estimate the effect of ASA mobilisation on the probability of a first strike (>25% vote against remuneration), using a binary outcome model. The Bayesian prior `N(0, 2.5^2)` on coefficients regularises the estimate when the data exhibits quasi-separation (e.g., zero cells in the ASA × strike cross-tabulation).
- **Outcome:** `beta_1_MAP` — the MAP estimate of the log-odds shift in first-strike probability due to ASA opposition.
- **Conditions:** Same as the vote channel.
- **Data:** Same cross-company panel, with the outcome variable being the binary first-strike indicator.
- **Parameter reasoning:** The `N(0, 2.5)` prior on logistic regression coefficients follows Gelman et al.'s recommendation for weakly informative priors that prevent separation-induced infinite estimates while remaining largely uninformative for plausible effect sizes. The extra dispersion term (0.10) is smaller than in the vote channel because the Laplace SE already captures most of the uncertainty; the 0.10 floor prevents the strike-channel prior from being unrealistically precise.

### 4c. Combined Prior (Inverse-Variance Weighting)

**Formula:**

```
w_vote    = 1 / sigma_vote^2
w_strike  = 1 / sigma_strike^2

gamma_A^{combined} ~ N(
    mu = (w_vote * mu_vote + w_strike * mu_strike) / (w_vote + w_strike),
    sigma = 1 / sqrt(w_vote + w_strike)
)
```

- **Purpose:** Blend the vote-channel and strike-channel estimates into a single prior, weighting each by the precision of its estimate. This gives more weight to whichever channel has tighter uncertainty.
- **Outcome:** A single Normal prior for `gamma_A` on the logit/log-odds scale, usable directly in the checkpoint update.
- **Parameter reasoning:** Inverse-variance weighting is the optimal combination of two independent Normal estimates of the same quantity. The two channels measure the same underlying effect (ASA mobilisation impact) through different outcomes, so pooling them reduces variance while preserving the information from both.

### 4d. Belief-Scale Mapping (Optional)

**Formula:**

```
gamma_A^{belief} = gamma_A^{vote} / lambda_rem
```

implemented via Monte Carlo propagation over the posterior draws of `lambda_rem` from the Stan belief model.

- **Purpose:** Transform the vote-logit-scale effect into the belief-state scale, for direct use as an additive shock in `checkpoint_update.py`.
- **Outcome:** A Normal approximation `N(mu_belief, sd_belief)` of the belief-scale ASA mobilisation shock.
- **Parameter reasoning:** Because `lambda_rem = 1` (anchored), this mapping is near-identity. It is included for generality in case the anchoring constraint is relaxed in future model variants.

---

## 5. Checkpoint Belief Distributions

**File:** `checkpoint_update.py`

This step constructs belief distributions at four critical moments in the Qantas 2023 governance crisis, starting from the Stan belief model posterior and adding forward shocks.

### Shock Distributions

**gamma_A — public ASA mobilisation:**

```
gamma_A ~ N(mu_A, sigma_A),  truncated below at 0
```

where `(mu_A, sigma_A)` are loaded from the data-driven priors (Section 4). The truncation at zero encodes the assumption that public mobilisation cannot *reduce* shareholder pressure.

- **Data:** Parameters from `fit_shock_priors.py`.
- **Reasoning for truncation:** ASA mobilisation is a directional action — it campaigns for a vote against. It is not plausible that announcing a public campaign would reduce opposition below what it would otherwise have been.

**gamma_E — private ASA engagement (asymmetric information):**

```
kappa  ~ Beta(2, 2)
gamma_E = kappa * gamma_A
```

- **Purpose:** Model the information shock that Qantas management receives from private ASA engagement *before* the public mobilisation. Management learns something about ASA's intent but does not observe the full public impact.
- **Outcome:** `gamma_E` — the management-private belief shift, which is a fraction of the eventual public shock.
- **Reasoning:** `Beta(2, 2)` is a symmetric, unimodal distribution on [0, 1] with mean 0.5 and moderate variance. This encodes the view that private engagement reveals *some* but not *all* of the eventual public-campaign impact, with genuine uncertainty about how much. The symmetry around 0.5 reflects agnosticism about whether private engagement reveals more or less than half the eventual impact.

**gamma_review — independent review announcement:**

```
gamma_review ~ N(-0.2, 0.2)
```

- **Purpose:** Capture the belief impact of Qantas announcing an independent governance review (10 October 2023).
- **Outcome:** Additive shift to belief state on the logit scale. Negative mean indicates that announcing a review is expected to slightly *reduce* shareholder distrust.
- **Reasoning:** Not parameterised from data. The mean of -0.2 reflects a judgement that announcing a review provides a modest credibility signal that partially addresses shareholder concerns. The SD of 0.2 expresses genuine uncertainty about whether the announcement is interpreted as a meaningful concession or as a deflection tactic. Both values are on the logit scale of remuneration opposition, where 0.2 corresponds to a modest shift.

### Checkpoint Construction

Four checkpoints represent the information structure at different dates:

| Checkpoint | Date | Market Belief | Management Belief |
|---|---|---|---|
| C0 | 2023-10-01 | `B_sep` | `B_sep + gamma_E` |
| C1 | 2023-10-10 | `B_sep + gamma_review` | `B_sep + gamma_E + gamma_review` |
| C2 | 2023-10-18 | `B_sep + gamma_review + gamma_A` | `B_sep + gamma_E + gamma_review + gamma_A` |
| C3 | 2023-11-03 | Measurement-updated (see below) | Measurement-updated (aligned) |

where `B_sep` is the Stan posterior belief state at September 2023 (the last month of the estimation window).

The key asymmetric-information feature: management observes `gamma_E` at C0 (from private engagement in September 2023), while the market does not. At C2, the public mobilisation `gamma_A` becomes common knowledge.

### AGM Measurement Update (C3)

**Formula:**

```
y_obs   = 0.829  (82.9% voted against the remuneration report)
y_logit = logit(0.829)

log w_i = log N(y_logit | alpha_rem_i + B_{C2,i}, sigma_rem_i)

w_i     = softmax(log w_i)

B_{C3}  ~ Categorical(w_1, ..., w_S)  applied to B_{C2} draws
```

This is a likelihood-weighted importance resampling step.

- **Purpose:** Update the belief distribution at C3 to incorporate the observed AGM outcome. Before C3, beliefs are based on the model and shock priors. After observing the actual 82.9% vote, draws that are more consistent with this outcome receive higher weight.
- **Outcome:** A resampled set of belief draws that reflect both the prior (model + shocks) and the observed AGM vote. This is a Bayesian posterior update implemented via importance sampling.
- **Conditions:** The likelihood function uses the remuneration-vote measurement equation from the Stan model (Section 3), with `alpha_rem` and `sigma_rem` draws from the Stan posterior. The anchoring constraint (`lambda_rem = 1`) means the likelihood directly connects `B_t` to the logit of the vote.
- **Data:** The observed 82.9% against-vote at the Qantas 2023 AGM.

---

## 6. Simulation-Layer Probabilistic Models

**File:** `qantas-simulation.py`

The simulation layer uses the belief posteriors from the checkpoint model to evaluate governance decisions. While not fitted via MCMC, it contains several generative probabilistic models that propagate uncertainty.

### 6a. Belief Path Evolution (Forward Simulation)

**Formula:**

```
B^{mkt}_{t} = rho_mkt * B^{mkt}_{t-1} + delta_mkt + beta_X * X_t + epsilon^{mkt}_t
B^{mgmt}_{t} = rho_mgmt * B^{mgmt}_{t-1} + delta_mgmt + beta_X * X_t + epsilon^{mgmt}_t
```

where `epsilon_t ~ N(0, sigma_B)` and `delta` is the governance-package-specific belief shift.

- **Purpose:** Project forward the belief paths for market and management under each governance package, using separate AR(1) dynamics for each actor.
- **Outcome:** Simulated monthly belief paths from the checkpoint date to the AGM.
- **Parameter reasoning:** Default `rho = 0.85` and `sigma_B = 0.35` are calibrated to produce plausible persistence and volatility on the belief scale. In checkpoint-mode runs, initial beliefs are drawn from the checkpoint posteriors (Section 5), so these parameters govern only the forward evolution.

### 6b. Vote Outcome Model

**Formula:**

```
p_{rem}   = logistic(alpha_rem + kappa_rem * B^{mkt}_{agm} + ASA_shift)
p_{dir}   = logistic(alpha_dir + kappa_dir * B^{mkt}_{agm})

y_{rem}   ~ N(p_{rem}, sd(p_{rem}))    clipped to [0, 1]
y_{dir}   ~ N(p_{dir}, sd(p_{dir}))    clipped to [0, 1]
```

where the noise standard deviation is `sd(p) = 0.06 + 0.06 * (0.5 - |p - 0.5|)`, giving more noise near p = 0.5 and less at extremes.

- **Purpose:** Generate noisy vote outcomes from latent vote probabilities, reflecting that actual vote percentages are uncertain even given the true underlying probability.
- **Outcome:** Simulated remuneration and chair vote-against percentages at the AGM.
- **Parameter reasoning:** `alpha_rem = -0.35`, `kappa_rem = 1.0` — chosen so that at neutral beliefs (`B = 0`), remuneration opposition is approximately `logistic(-0.35) ≈ 41%`, reflecting the elevated opposition context at Qantas. `alpha_dir = -0.90`, `kappa_dir = 1.10` — chair opposition has a lower baseline (~29%) and slightly steeper response to beliefs. The heteroscedastic noise function with base SD of 0.06 reflects that real vote outcomes vary by a few percentage points around their expected values, with more randomness when the outcome is close to contested.

### 6c. CEO Transition Model

**Formula:**

```
P(transition) = logistic(1.8 + 0.6 * B^{mgmt}_{agm} + 1.5 * V)

P(sacked | transition) = logistic(-2.4 + 0.6 * B^{mgmt}_{agm} + 1.8 * V)
```

CEO mode: 0 = stay, 1 = resign (transition and not sacked), 2 = sacked (transition and sacked).

- **Purpose:** Model the endogenous CEO departure decision as a function of management's perceived shareholder pressure (`B_mgmt`) and board deference (`V`, where higher V = more independent board).
- **Outcome:** A categorical draw for each simulation: stay, resign, or be sacked.
- **Parameter reasoning:** Not estimated from data. The intercepts and slopes are structured so that (a) CEO transition probability increases with shareholder distrust and board independence, (b) conditional on transition, resignation is more likely when management perceives the pressure (high `B_mgmt`) but sacking requires both high pressure and an independent board (high `V`). The specific coefficients (1.8, 0.6, 1.5, -2.4, 1.8) were calibrated to produce plausible transition probabilities across the range of belief states observed in the Qantas case.

### 6d. Market Reaction Model (Cumulative Abnormal Return)

**Formula:**

```
mu_CAR = eta_0 + eta_B * B^{mkt}_{1} + eta_{CEO_mode}

CAR ~ N(mu_CAR, sigma_car)
```

where `eta_{CEO_mode}` is `+0.012` for resignation and `-0.006` for sacking.

- **Purpose:** Generate the short-term market reaction to the governance decision announcement.
- **Outcome:** Simulated one-month cumulative abnormal return.
- **Parameter reasoning:** `eta_0 = 0.002` (slight positive base), `eta_B = -0.012` (higher distrust leads to worse market reaction to any announcement), `sigma_car = 0.012` (daily CAR noise aggregated to monthly). `eta_ceo_resign = +0.012` reflects that markets typically react positively to a CEO departure under pressure (interpreted as accountability). `eta_ceo_sacked = -0.006` reflects that forced removal signals deeper problems. These are calibrated from event-study magnitudes in comparable corporate governance events.

### 6e. Institutional Exit Pressure Model

**Formula:**

```
exit_{12m} ~ N(mu_exit + kappa_exit * B^{mkt}_{agm}, sigma_exit)
```

- **Purpose:** Model the 12-month cumulative institutional selling pressure as a function of shareholder distrust at the AGM.
- **Outcome:** Net institutional flow (negative = selling pressure).
- **Parameter reasoning:** `mu_exit = 0.0` (neutral base), `kappa_exit = -0.60` (higher distrust drives net selling), `sigma_exit = 0.40` (substantial idiosyncratic variation in institutional behaviour). These reflect the observation that institutional investors respond to governance concerns with gradual portfolio rebalancing, but individual fund decisions are highly heterogeneous.

### 6f. Management Overconfidence Bias

**Formula:**

```
B^{mgmt}_{0} = B^{mkt}_{0} - bias
bias ~ N(0.90, 0.12)
```

(Used only in fallback mode when checkpoint posteriors are not available.)

- **Purpose:** Model the systematic gap between management's perception of shareholder sentiment and the market's actual sentiment. Management underestimates distrust.
- **Outcome:** Management's initial belief state is shifted downward (less distrust perceived) relative to the market's belief.
- **Parameter reasoning:** The mean bias of 0.90 logit units is substantial — it implies management perceives roughly half the opposition probability that the market does. This is motivated by the CEO overconfidence literature and the specific Qantas context where management repeatedly underestimated governance backlash. The 0.12 SD allows moderate uncertainty about the degree of overconfidence.

### 6g. Board Optimism Shift (Stackelberg Layer)

**Formula:**

```
B^{board}_{agm} = B^{mkt}_{agm} + board_optimism_shift
```

where `board_optimism_shift = -0.8` (default).

- **Purpose:** In the adversarial Stackelberg simulation, the board evaluates strike and revolt probabilities through beliefs that are systematically more optimistic (lower distrust) than the market's actual beliefs.
- **Outcome:** The board's perceived strike and revolt probabilities are lower than the market-truth probabilities, creating a wedge between what the board expects and what actually happens.
- **Parameter reasoning:** The -0.8 shift (in logit units) represents a judgement that the Qantas board, like many boards, systematically underestimates the probability of adverse shareholder action. This is distinct from management overconfidence — it applies to the board's collective assessment. The magnitude is calibrated so that a board facing a true ~60% opposition probability perceives it as ~40%.

### 6h. Spill Probability (Stackelberg Layer)

**Formula:**

```
spill | strike2 ~ Bernoulli(p_spill_given_strike2)
```

where `p_spill_given_strike2 = 0.35`.

- **Purpose:** In the second-strike scenario, model the probability that a board spill motion is triggered conditional on a second strike occurring.
- **Outcome:** Binary spill event for each simulation where a second strike occurs.
- **Parameter reasoning:** Under Australia's "two strikes" rule, a second consecutive >25% vote against the remuneration report triggers a resolution on whether to spill the board. The 0.35 probability reflects that while a spill resolution is automatic, its passage is uncertain — institutional investors may vote for the second strike as a signal without supporting a full board spill. The value is informed by the historical passage rate of spill motions in the ASX context.

---

## Summary Table

| # | Model | Estimated From Data? | Scale | Key Output |
|---|-------|---------------------|-------|------------|
| 1 | Market model (OLS) | Yes — share price data | Daily log-returns | Abnormal returns |
| 2 | Media measurement (Stan MCMC) | Yes — monthly media observations | Log-intensity | Media coverage and intensity posteriors |
| 3 | Belief dynamics (Stan MCMC) | Yes — abnormal returns, votes, media shocks | Logit (rem opposition) | Latent belief state posteriors |
| 4a | Vote-channel shock prior (OLS) | Yes — cross-company voting panel | Logit | gamma_A prior (vote) |
| 4b | Strike-channel shock prior (Bayes logit MAP) | Yes — cross-company voting panel | Log-odds | gamma_A prior (strike) |
| 4c | Combined shock prior | Derived from 4a + 4b | Logit | gamma_A prior (blended) |
| 5 | Checkpoint beliefs | Partially — base from Stan; shocks from priors | Logit | Belief distributions at 4 dates |
| 5 (C3) | AGM measurement update | Yes — observed 82.9% vote | Logit | Posterior belief given vote |
| 6a | Forward belief paths | No — calibrated | Logit | Simulated belief trajectories |
| 6b | Vote outcome | No — calibrated | Probability | Simulated AGM vote percentages |
| 6c | CEO transition | No — calibrated | Probability | CEO stay/resign/sacked |
| 6d | Market reaction (CAR) | No — calibrated from event studies | Log-return | Short-term market response |
| 6e | Exit pressure | No — calibrated | Institutional flow | 12-month selling pressure |
| 6f | Management overconfidence | No — informed by literature | Logit shift | Management belief bias |
| 6g | Board optimism | No — calibrated | Logit shift | Board-perceived risk |
| 6h | Spill probability | No — informed by ASX history | Probability | Spill event given second strike |
