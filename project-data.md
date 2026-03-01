# Project Data

This document catalogues every data file that informs the Bayesian priors or conditions the results in the Qantas ARA pipeline. Files are grouped into three tiers:

1. **Primary source data** — external data collected or curated for this project
2. **Pipeline-generated intermediate data** — outputs of one step that become inputs to a later step
3. **Reference and background documents** — qualitative research that informed parameter choices and model design

---

## Primary Source Data

### data/qantas_share_price_data.json

**Contents:** Daily price data for Qantas Airways (QAN.AX) and the ASX 200 index (^AXJO), spanning 4 January 2010 to 2 October 2025 (3,983 trading days). Fetched from market data sources.

**Structure:** JSON with four top-level keys:

| Key | Description |
|-----|-------------|
| `metadata` | Ticker identifiers, exchange, currency, index name, fetch timestamp |
| `statistics` | Summary stats: date range, total days, price range, total returns for both QAN and ASX 200 |
| `significant_drops` | List of largest single-day percentage declines |
| `data` | Array of 3,983 daily records |

**Data dictionary (per record in `data`):**

| Field | Type | Description |
|-------|------|-------------|
| `date` | string | Trading date (YYYY-MM-DD) |
| `open` | float | QAN opening price (AUD) |
| `high` | float | QAN intraday high (AUD) |
| `low` | float | QAN intraday low (AUD) |
| `close` | float | QAN closing price (AUD) |
| `volume` | int | QAN shares traded |
| `daily_change` | float | Absolute change from previous close (AUD) |
| `daily_change_percent` | float | Percentage change from previous close |
| `index_close` | float | ASX 200 closing level |
| `index_open` | float | ASX 200 opening level |
| `index_daily_change_percent` | float | ASX 200 percentage change |
| `ma_7` | float | 7-day moving average (present in later records only) |
| `ma_30` | float | 30-day moving average (present in later records only) |
| `ma_90` | float | 90-day moving average (present in later records only) |

**Relevance:** This is the sole source for computing abnormal returns. The OLS market model regresses QAN log-returns on ASX 200 log-returns, and the residuals (abnormal returns) become a monthly observable in the belief dynamics model.

**Used in:** `compute_abnormal_returns.py` (Step 1). The estimation window uses Oct 2020 – Dec 2023 by default.

---

### data/monthly_media_variables.xlsx

**Contents:** Sparse monthly observations of media coverage variables for Qantas governance-related events. 50 rows spanning January 2011 to August 2025, but with observations only in months where notable media events occurred (many months have no entry at all).

**Data dictionary:**

| Field | Type | Description |
|-------|------|-------------|
| `year` | int | Calendar year |
| `month` | int | Calendar month (1–12) |
| `year_month` | string | YYYY-MM identifier |
| `media_event_count` | int | Number of distinct governance-related media events in the month |
| `media_damage_intensity` | float | Aggregate damage-intensity score across events (sum of per-event log-transformed severity scores) |
| `media_sentiment_mean` | float | Mean sentiment score across events (-1 to +1; negative = critical/damaging) |
| `media_response_quality` | float | Mean quality-of-corporate-response score (1 = poor/no response, 5 = strong/proactive) |
| `media_concentration_index` | float | Herfindahl-style concentration of events across topic categories (1.0 = all one topic, lower = diverse) |

**Relevance:** These are the raw observations for the media measurement model. The sparsity of observations (50 out of ~175 possible months) is the primary reason a state-space model is needed — the AR(1) latent process interpolates through unobserved months.

**Used in:** `build_media_monthly_complete.py` (Step 2), which constructs a complete monthly grid and generates observation masks; then `prep_stan_media_data.py` (Step 3), which extracts the `media_damage_intensity` values as `y_obs` for the Stan media model.

---

### data/agm-votes.csv

**Contents:** Qantas AGM voting outcomes for remuneration-report and chair-election resolutions, 2020–2023. 4 rows.

**Data dictionary:**

| Field | Type | Description |
|-------|------|-------------|
| `year_month` | string | YYYY-MM of the AGM |
| `vote_against_rem_pct` | float | Proportion of votes cast against the remuneration report (0–1) |
| `vote_against_chair_pct` | float | Proportion of votes cast against the chair's re-election (0–1); blank if chair not up for re-election |

**Records:**

| year_month | vote_against_rem_pct | vote_against_chair_pct |
|------------|---------------------|----------------------|
| 2020-10 | 0.0893 | — |
| 2021-11 | 0.0999 | — |
| 2022-11 | 0.0938 | 0.0204 |
| 2023-11 | 0.8293 | — |

**Relevance:** These vote outcomes are the point-in-time measurements in the belief dynamics Stan model. The 2020–2022 votes condition the posterior during MCMC fitting. The 2023-11 vote (82.93% against remuneration) is held out and used as the likelihood in the importance-weighted measurement update at checkpoint C3.

**Used in:** `build_historical_belief_table.py` (Step 6) merges these into the monthly grid. `prep_stan_belief_data.py` (Step 7) extracts `rem_obs` and `chair_obs` with their time indices for the Stan model. The 2023 vote is used in `checkpoint_update.py` (Step 10) for the C3 measurement update.

---

### data/voting-recommendations.csv

**Contents:** Cross-company panel of ASX-listed companies that received remuneration-report votes in the period 2021–2023, along with ASA voting recommendations and contextual variables. 36 rows covering 29 distinct companies.

**Data dictionary:**

| Field | Type | Description |
|-------|------|-------------|
| `company` | string | Company name |
| `asx_code` | string | ASX ticker symbol |
| `year` | int | AGM year |
| `month` | int | AGM month |
| `rem_against_pct` | float | Proportion voting against remuneration report (0–1) |
| `first_strike` | int | 1 if this vote constituted a first strike (>25% against), 0 otherwise |
| `asa_against` | int | 1 if ASA recommended voting against the remuneration report, 0 otherwise |
| `asa_proxy_against` | int | 1 if ASA's proxy adviser also recommended against, 0 otherwise |
| `proxy_adv_against` | int | 1 if independent proxy advisers (ISS/Glass Lewis) recommended against |
| `multi_target` | int | 1 if the company was targeted by multiple shareholder advocacy groups |
| `prior_year_pct` | float | Previous year's vote-against percentage (lagged outcome) |
| `log_mkt_cap` | float | Natural log of market capitalisation at AGM time |
| `headline_incident` | int | 1 if the company was associated with a major public incident in the prior year |
| `gics` | string | GICS industry classification |

**Relevance:** This panel is the training data for the data-driven shock priors (`gamma_A`). The `asa_against` column is the treatment variable — it captures whether ASA publicly recommended voting against the remuneration report. The outcome variables are `rem_against_pct` (vote channel, continuous) and `first_strike` (strike channel, binary). The controls (`prior_year_pct`, `log_mkt_cap`, `gics`) absorb confounding variation so that the `asa_against` coefficient isolates the marginal effect of ASA mobilisation.

**Used in:** `fit_shock_priors.py` (Step 9) fits the OLS vote-channel model and the Bayesian logistic strike-channel model. `build_historical_belief_table.py` (Step 6) also uses this data to construct market-wide ASA activity indicators for each month.

---

### agm-pdfs/QAN-2020-AGM-Results.pdf

**Contents:** Official ASX market announcement from Qantas (lodged 23 October 2020) reporting the results of all resolutions at the 2020 Annual General Meeting, held virtually on 23 October 2020.

**Relevance:** Source document for verifying the 2020-10 remuneration vote figure (8.93% against) recorded in `agm-votes.csv`. Authorised by Andrew Finch, General Counsel & Company Secretary.

**Used in:** Manual verification of `data/agm-votes.csv`.

---

### agm-pdfs/QAN-2021-AGM-Results.pdf

**Contents:** Official ASX market announcement from Qantas (lodged 5 November 2021) reporting the results of all resolutions at the 2021 Annual General Meeting, held virtually on 5 November 2021.

**Relevance:** Source document for verifying the 2021-11 remuneration vote figure (9.99% against) recorded in `agm-votes.csv`.

**Used in:** Manual verification of `data/agm-votes.csv`.

---

### agm-pdfs/QAN-2022-AGM-Results.pdf

**Contents:** Official ASX market announcement from Qantas (lodged 4 November 2022) reporting the results of all resolutions at the 2022 Annual General Meeting, held on 4 November 2022.

**Relevance:** Source document for verifying the 2022-11 remuneration vote (9.38% against) and chair re-election vote (2.04% against) in `agm-votes.csv`. The 2022 chair vote is the only chair-election observation available to the Stan model.

**Used in:** Manual verification of `data/agm-votes.csv`.

---

## Pipeline-Generated Intermediate Data

These files are outputs of earlier pipeline steps that become inputs to later steps. They carry posterior distributions, summary statistics, or reformatted data forward through the pipeline.

### data/media_monthly_complete.xlsx

**Contents:** Complete monthly grid for the media model, spanning October 2020 to August 2025 (59 months). Every month has a row regardless of whether media data was observed.

**Data dictionary:**

| Field | Type | Description |
|-------|------|-------------|
| `t` | int | Month index (1–59) |
| `t_scaled` | float | Time index scaled to [-1, 1] for the logistic coverage trend |
| `year_month` | string | YYYY-MM |
| `year` | int | Calendar year |
| `month` | int | Calendar month |
| `media_observed` | int | 1 if any media data exists for this month, 0 otherwise |
| `event_count_observed` | int | 1 if event count is observed, 0 otherwise |
| `intensity_observed` | int | 1 if damage intensity is observed, 0 otherwise |
| `sentiment_observed` | int | 1 if sentiment is observed, 0 otherwise |
| `response_observed` | int | 1 if response quality is observed, 0 otherwise |
| `concentration_observed` | int | 1 if concentration index is observed, 0 otherwise |
| `media_event_count` | float | Observed event count (NaN if unobserved) |
| `media_damage_intensity` | float | Observed damage intensity (NaN if unobserved) |
| `media_sentiment_mean` | float | Observed mean sentiment (NaN if unobserved) |
| `media_response_quality` | float | Observed response quality (NaN if unobserved) |
| `media_concentration_index` | float | Observed concentration index (NaN if unobserved) |

**Produced by:** `build_media_monthly_complete.py` (Step 2).

**Used in:** `prep_stan_media_data.py` (Step 3).

---

### data/stan_media_data.json

**Contents:** Data formatted for the Stan media measurement model. Extracts the observed intensity values and their time indices from the complete monthly grid.

**Data dictionary:**

| Field | Type | Description |
|-------|------|-------------|
| `T` | int | Total number of months in the grid (59) |
| `eps` | float | Small constant added before taking log (1e-6) |
| `t_scaled` | float[] | Scaled time index for each month, length T |
| `N_y` | int | Number of months with observed intensity values (46) |
| `y_idx` | int[] | 1-based month indices where intensity was observed, length N_y |
| `y_obs` | float[] | Observed `media_damage_intensity` values, length N_y |

**Produced by:** `prep_stan_media_data.py` (Step 3).

**Used in:** `fit_media_better_stan.py` (Step 4) passes this directly to the Stan sampler.

---

### data/media_better_draws.npz

**Contents:** Posterior draws from the Stan media measurement model. 8,000 draws (4 chains x 2,000 post-warmup) for each latent time series.

**Arrays:**

| Key | Shape | Description |
|-----|-------|-------------|
| `year_month` | (59,) | Month labels |
| `C_draws` | (8000, 59) | Posterior draws of media coverage C_t at each month |
| `M_draws` | (8000, 59) | Posterior draws of media intensity M_t (exponentiated from logM_t) at each month |

**Produced by:** `fit_media_better_stan.py` (Step 4).

**Used in:** `compute_media_shocks.py` (Step 5) computes log-differences of `M_draws`.

---

### data/media_better_summary.csv

**Contents:** Posterior summary statistics for media coverage and intensity, one row per month.

**Data dictionary:**

| Field | Type | Description |
|-------|------|-------------|
| `t` | int | Month index (1–59) |
| `year_month` | string | YYYY-MM |
| `C_mean` | float | Posterior mean of coverage C_t |
| `C_p05` | float | 5th percentile of C_t |
| `C_p95` | float | 95th percentile of C_t |
| `M_mean` | float | Posterior mean of intensity M_t |
| `M_p05` | float | 5th percentile of M_t |
| `M_p95` | float | 95th percentile of M_t |

**Produced by:** `fit_media_better_stan.py` (Step 4).

**Used in:** Diagnostic and reporting purposes.

---

### data/media_better_diagnostics.txt

**Contents:** CmdStanPy MCMC diagnostic output for the media model. Reports on tree depth, divergences, E-BFMI, effective sample size, and split R-hat. The fit reported 2 of 8,000 transitions (0.03%) with divergences; all other diagnostics satisfactory.

**Produced by:** `fit_media_better_stan.py` (Step 4).

**Used in:** Model validation and reporting.

---

### data/media_shock_draws.npz

**Contents:** Log-difference media shocks derived from the media model posterior.

**Arrays:**

| Key | Shape | Description |
|-----|-------|-------------|
| `year_month` | (59,) | Month labels |
| `shock_draws` | (8000, 59) | `logM_t - logM_{t-1}` for each draw; first month set to 0 |
| `logM_draws` | (8000, 59) | Raw log-intensity draws (carried forward for reference) |

**Produced by:** `compute_media_shocks.py` (Step 5).

**Used in:** `build_historical_belief_table.py` (Step 6) computes summary statistics (mean, p05, p95) of the shock draws for each month. `prep_stan_belief_data.py` (Step 7) uses the posterior mean shocks as the `shock` input to the belief model (after z-scoring).

---

### data/abret_daily.csv

**Contents:** Daily abnormal returns for Qantas over the estimation window. ~800 rows (Oct 2020 – Dec 2023).

**Data dictionary:**

| Field | Type | Description |
|-------|------|-------------|
| `date` | string | Trading date (YYYY-MM-DD) |
| `close` | float | QAN closing price (AUD) |
| `index_close` | float | ASX 200 closing level |
| `r_qan` | float | QAN daily log-return |
| `r_mkt` | float | ASX 200 daily log-return |
| `abret` | float | Daily abnormal return (r_qan minus fitted market model) |

**Produced by:** `compute_abnormal_returns.py` (Step 1).

**Used in:** Source for monthly aggregation. Not directly used by the Stan model.

---

### data/abret_monthly.csv

**Contents:** Monthly aggregation of daily abnormal returns. ~39 rows (Oct 2020 – Dec 2023).

**Data dictionary:**

| Field | Type | Description |
|-------|------|-------------|
| `month` | string | YYYY-MM |
| `abret_sum` | float | Sum of daily abnormal returns in the month |
| `abret_mean` | float | Mean daily abnormal return |
| `abret_vol` | float | Standard deviation of daily abnormal returns within the month |
| `qan_ret_sum` | float | Sum of QAN daily log-returns |
| `mkt_ret_sum` | float | Sum of ASX 200 daily log-returns |
| `n_days` | int | Number of trading days in the month |
| `neg_tail_freq` | float | Fraction of days with abnormal return below -2 * daily SD (left-tail frequency) |
| `month_start` | date | First calendar day of the month |
| `month_end` | date | Last calendar day of the month |

**Produced by:** `compute_abnormal_returns.py` (Step 1).

**Used in:** `build_historical_belief_table.py` (Step 6) merges `abret_sum` and `abret_vol` into the historical belief table. `prep_stan_belief_data.py` (Step 7) z-scores `abret_sum` to create the `abret` input for the Stan belief model.

---

### data/historical_belief_table.csv

**Contents:** The convergence table that merges all upstream data sources into a single monthly panel. 39 rows (Oct 2020 – Dec 2023), 42 columns. This is the single source of truth for the belief model's inputs and the checkpoint update's flags.

**Data dictionary (selected key columns):**

| Field | Type | Description |
|-------|------|-------------|
| `t` | int | Month index (1–39) |
| `year_month` | string | YYYY-MM |
| `media_shock_mean` | float | Posterior mean of media shock for this month |
| `media_shock_p05` | float | 5th percentile of media shock |
| `media_shock_p95` | float | 95th percentile of media shock |
| `abret_sum` | float | Monthly sum of daily abnormal returns |
| `abret_vol` | float | Monthly abnormal return volatility |
| `vote_against_rem_pct` | float | Remuneration vote-against proportion (non-null for 4 AGM months only) |
| `vote_against_chair_pct` | float | Chair vote-against proportion (non-null for 1 AGM month only) |
| `asa_engagement_private` | int | 1 if ASA private engagement with Qantas was active in this month |
| `asa_public_mobilisation` | int | 1 if ASA public mobilisation (voting intention published) was active |

**Market-wide ASA activity columns (from voting-recommendations panel):**

| Field | Type | Description |
|-------|------|-------------|
| `mkt_asa_against_count` | float | Count of ASA "against" recommendations across all companies this month |
| `mkt_asa_against_any` | float | 1 if any ASA "against" recommendation occurred this month |
| `mkt_asa_against_w_sum` | float | Market-cap-weighted sum of ASA "against" actions |
| `mkt_asa_proxy_against_count` | float | Count of ASA proxy "against" recommendations |
| `mkt_proxy_adv_against_count` | float | Count of independent proxy adviser "against" recommendations |
| `mkt_multi_target_count` | float | Count of multi-target situations |
| `mkt_headline_incident_count` | float | Count of headline incidents |
| `mkt_first_strike_count` | float | Count of first strikes this month |
| `mkt_rem_against_pct_mean` | float | Mean vote-against percentage across all companies with AGMs this month |
| `mkt_rem_against_pct_mean_given_asa_against` | float | Mean vote-against percentage for ASA-targeted companies only |

**Qantas-specific columns (from voting-recommendations panel, populated for 2023-11 only):**

| Field | Type | Description |
|-------|------|-------------|
| `qan_asa_against` | float | ASA "against" for Qantas |
| `qan_asa_proxy_against` | float | ASA proxy "against" for Qantas |
| `qan_proxy_adv_against` | float | Independent proxy adviser "against" for Qantas |
| `qan_multi_target` | float | Qantas was multi-targeted |
| `qan_headline_incident` | float | Qantas had headline incident |
| `qan_first_strike` | float | Qantas received first strike |
| `qan_rem_against_pct` | float | Qantas remuneration vote-against (0.829) |
| `qan_prior_year_pct` | float | Qantas prior-year vote-against |
| `qan_log_mkt_cap` | float | Qantas log market cap |
| `qan_company` | string | "Qantas" |
| `qan_gics` | string | "Industrials" |

**Produced by:** `build_historical_belief_table.py` (Step 6).

**Used in:** `prep_stan_belief_data.py` (Step 7) extracts the shock and abnormal return series plus vote observations. `checkpoint_update.py` (Step 10) reads the `asa_engagement_private` and `asa_public_mobilisation` flags to determine which shocks are active at each checkpoint.

---

### data/stan_belief_data.json

**Contents:** Data formatted for the Stan belief dynamics model. All continuous series are z-scored; vote observations are extracted with 1-based time indices.

**Data dictionary:**

| Field | Type | Description |
|-------|------|-------------|
| `T` | int | Number of months (36; the belief model uses Oct 2020 – Sep 2023) |
| `shock` | float[] | Z-scored media shock series (posterior means), length T |
| `abret` | float[] | Z-scored monthly abnormal return sums, length T |
| `shock_mean` | float | Mean used for z-scoring shocks |
| `shock_sd` | float | SD used for z-scoring shocks |
| `abret_mean` | float | Mean used for z-scoring abnormal returns (~0) |
| `abret_sd` | float | SD used for z-scoring abnormal returns |
| `N_rem` | int | Number of remuneration vote observations (3: 2020, 2021, 2022) |
| `rem_idx` | int[] | 1-based month indices of remuneration votes |
| `rem_obs` | float[] | Remuneration vote-against proportions (0.0893, 0.0999, 0.0938) |
| `N_chair` | int | Number of chair vote observations (1: 2022) |
| `chair_idx` | int[] | 1-based month indices of chair votes |
| `chair_obs` | float[] | Chair vote-against proportions (0.0204) |

Note: The 2023 AGM vote is excluded from Stan fitting — it is held out for the checkpoint C3 measurement update.

**Produced by:** `prep_stan_belief_data.py` (Step 7).

**Used in:** `fit_belief_model_stan.py` (Step 8) passes this directly to the Stan sampler.

---

### data/belief_state_draws.npz

**Contents:** Posterior draws from the Stan belief dynamics model. 8,000 draws for the latent belief state and all model parameters.

**Arrays:**

| Key | Shape | Description |
|-----|-------|-------------|
| `year_month` | (36,) | Month labels (Oct 2020 – Sep 2023) |
| `B_draws` | (8000, 36) | Posterior draws of belief state B_t at each month |
| `alpha_rem` | (8000,) | Posterior draws of remuneration-vote baseline (logit scale) |
| `lambda_rem` | (8000,) | Remuneration-vote loading — fixed at 1.0 (anchoring constraint) |
| `sigma_rem` | (8000,) | Posterior draws of remuneration-vote observation noise |
| `beta` | (8000,) | Posterior draws of media-shock-to-belief sensitivity |
| `sigma_B` | (8000,) | Posterior draws of state evolution noise |
| `sigma_B0` | (8000,) | Posterior draws of initial-state noise |
| `rho` | (8000,) | Posterior draws of belief persistence (AR(1) coefficient) |

**Produced by:** `fit_belief_model_stan.py` (Step 8).

**Used in:** `fit_shock_priors.py` (Step 9) optionally uses `lambda_rem` draws for belief-scale mapping. `checkpoint_update.py` (Step 10) uses `B_draws` (specifically the September 2023 column) as the starting point, and `alpha_rem` + `sigma_rem` for the C3 measurement update likelihood.

---

### data/belief_state_summary.csv

**Contents:** Posterior summary statistics for the belief state, one row per month.

**Data dictionary:**

| Field | Type | Description |
|-------|------|-------------|
| `t` | int | Month index (1–36) |
| `year_month` | string | YYYY-MM |
| `B_mean` | float | Posterior mean of B_t |
| `B_p05` | float | 5th percentile of B_t |
| `B_p95` | float | 95th percentile of B_t |

**Produced by:** `fit_belief_model_stan.py` (Step 8).

**Used in:** Diagnostic and reporting purposes.

---

### data/belief_state_diagnostics.txt

**Contents:** CmdStanPy MCMC diagnostic output for the belief model. Reports 31 of 8,000 transitions (0.39%) with divergences; all other diagnostics (tree depth, E-BFMI, ESS, R-hat) satisfactory.

**Produced by:** `fit_belief_model_stan.py` (Step 8).

**Used in:** Model validation and reporting.

---

### data/priors/shock_priors_C0_2023-10-01.json through C3_2023-11-03.json

**Contents:** Data-driven Normal priors for the ASA public mobilisation shock `gamma_A`, fitted from the voting-recommendations panel with as-of filtering to each checkpoint date. Four files (C0–C3), plus an optional `shock_priors_asof_2023-10-01.json`.

**Structure (each file):** JSON object with up to four keys, each containing a `NormalPrior`:

| Key | Description |
|-----|-------------|
| `gamma_A_vote_logit` | Prior from OLS on logit(rem_against_pct) |
| `gamma_A_strike_logodds` | Prior from Bayesian logistic regression on first_strike |
| `gamma_A_combined` | Inverse-variance blend of vote and strike channels |
| `gamma_A_belief_from_vote` | Vote-logit prior mapped to belief scale via lambda_rem draws (optional) |

**Fields per prior entry:**

| Field | Type | Description |
|-------|------|-------------|
| `dist` | string | Always "normal" |
| `mu` | float | Prior mean |
| `sigma` | float | Prior standard deviation |
| `n` | int | Number of observations used in fitting |
| `model` | string | Description of the estimation model |
| `asof` | string | As-of date (YYYY-MM-DD) — data after this date is excluded |
| `max_included_month` | string | Last YYYY-MM included in the estimation |
| `notes` | string | Additional context |

**Example values (C0/C2, which are identical because as-of filtering gives the same data up to Sep 2023):**
- `gamma_A_vote_logit`: mu = 2.149, sigma = 0.702, n = 23
- `gamma_A_strike_logodds`: mu = 3.051, sigma = 1.384, n = 23
- `gamma_A_combined`: mu = 2.334, sigma = 0.626, n = 23

**Produced by:** `fit_shock_priors.py` (Step 9).

**Used in:** `checkpoint_update.py` (Step 10) loads the prior (by default `gamma_A_vote_logit` from the C2 file) to draw the `gamma_A` shock applied at checkpoint C2.

---

### data/checkpoints/belief_C0_2023-10-01.npz through belief_C3_2023-11-03.npz

**Contents:** Belief distributions at four critical moments in the Qantas 2023 governance crisis. Each file contains market and management belief draws plus the shock draws that produced them.

**Arrays (per file):**

| Key | Shape | Description |
|-----|-------|-------------|
| `checkpoint` | scalar | Checkpoint identifier (e.g., "C0_2023-10-01") |
| `seed` | scalar | RNG seed used (123) |
| `B_mkt` | (8000,) | Market belief distribution at this checkpoint |
| `B_mgmt` | (8000,) | Management belief distribution at this checkpoint |
| `gamma_A` | (8000,) | Draws of the public mobilisation shock |
| `gamma_E` | (8000,) | Draws of the private engagement shock |
| `gamma_review` | (8000,) | Draws of the review announcement shock |
| `agm_rem_against` | scalar | Observed AGM vote (0.829) |
| `gamma_A_prior` | dict | Metadata about the prior source used |

C3 differs from C0–C2 in that the belief draws have been importance-resampled to incorporate the observed 82.9% vote.

**Produced by:** `checkpoint_update.py` (Step 10).

**Used in:** `qantas-simulation.py` (Step 11) loads these as initial belief distributions for evaluating governance packages.

---

### data/sim_summary_by_checkpoint.csv

**Contents:** Output of the non-adversarial governance decision simulation. One row per (checkpoint, package) combination. Covers 4 checkpoints x 4 packages (though some packages are infeasible at later checkpoints due to monotonicity constraints).

**Data dictionary (selected columns):**

| Field | Type | Description |
|-------|------|-------------|
| `Checkpoint` | string | C0, C1, C2, or C3 |
| `Package` | string | D0 Minimal, D1 Review-first, D2 Accountability-lite, or D3 CEO transition |
| `Feasible` | bool | Whether the package is feasible given the sequential state |
| `E[L_market]` | float | Expected market loss |
| `E[L_board]` | float | Expected board loss |
| `E[L_ceo]` | float | Expected CEO loss |
| `L_market 5%` / `L_market 95%` | float | Loss quantiles |
| `E[impl_cost]` through `E[ceo_agency]` | float | Loss decomposition components |
| `P(strike1)` through `P(spill)` | float | Event probabilities |
| `Mean CAR (m1)` | float | Mean cumulative abnormal return |
| `P(CEO stay)` / `P(CEO resign)` / `P(CEO sacked)` | float | CEO mode probabilities |
| `Mean B_mkt_agm` / `Mean B_mgmt_agm` | float | Mean belief states at AGM |
| `market_optimal` / `board_optimal` / `ceo_preferred` | bool | Whether this package minimises the respective actor's loss |

**Produced by:** `qantas-simulation.py` (Step 11).

**Used in:** Final analysis output.

---

### data/sim_summary_by_checkpoint_adversarial.csv

**Contents:** Output of the adversarial Stackelberg simulation. One row per (checkpoint, package, ASA action, CEO action) combination, producing a much larger table than the non-adversarial version.

**Data dictionary (additional columns beyond the non-adversarial version):**

| Field | Type | Description |
|-------|------|-------------|
| `D` | string | Governance package |
| `A` | string | ASA action (DoNothing, RecommendStrike, Low, Medium, High) |
| `CEO_action` | string | CEO strategic action (Stay, Resign) |
| `asa_shift` | float | ASA vote-logit shift applied |
| `asa_cost` | float | Cost to ASA of the action |
| `E[U_ASA]` | float | Expected ASA utility |

**Produced by:** `qantas-simulation.py` (Step 11) in adversarial mode.

**Used in:** Final analysis output.

---

## Reference and Background Documents

These documents do not enter the pipeline as machine-readable data, but informed the design of models, the choice of parameter values, and the structure of the decision problem.

### asa_background/process-prior-to-AGM.md

**Contents:** Research note on the ASA's engagement process with ASX-listed companies prior to AGMs. Synthesises public ASA documentation, ASIC reviews, and law firm guidance to establish the typical timeline and nature of ASA engagement.

**Key findings used in the project:**
- ASA Voting Intentions are typically published ~14 days before the AGM
- Private engagement between ASA and the company begins weeks to months before the public recommendation
- For Qantas's 3 November 2023 AGM, engagement likely started in August–September 2023, with the public "Against" recommendation appearing around 18 October 2023
- ASIC found that in 65/80 cases, proxy advisers had engaged with the company before issuing an "against" recommendation

**Relevance:** This research directly informed the checkpoint timing structure:
- C0 (1 Oct) is set *after* the inferred start of private engagement, justifying the `gamma_E` (private engagement shock) being applied to management beliefs at C0
- C2 (18 Oct) aligns with the public reporting of ASA's "Against" recommendation, justifying `gamma_A` (public mobilisation shock) becoming common knowledge at C2
- The `kappa ~ Beta(2, 2)` scaling of `gamma_E = kappa * gamma_A` reflects the finding that private engagement reveals partial but incomplete information about the eventual public campaign intensity

**Used in:** Design of `checkpoint_update.py` checkpoint structure and shock timing.

---

### ceo-background/ceo-overconfidence.md

**Contents:** Literature review and case-study evidence on CEO overconfidence, applied to Alan Joyce's tenure at Qantas. Covers:

- **Empirical evidence of Joyce's overconfidence:** Share sale timing (June 2023, three days after ACCC evidence), pattern of doubling down (2011 fleet grounding, 2020 illegal outsourcing, ghost flights response), Saar governance review finding "excessive deference toward Joyce," and compensation as a self-importance proxy ($21.4M FY23, ~$125M cumulative)
- **Academic literature:** Malmendier & Tate (2005) Longholder measure, Hayward & Hambrick (1997) hubris and acquisition premiums, Svenson (1981) better-than-average effect, CEO overconfidence meta-analysis (2023 Journal of Management), CEO narcissism and board deference
- **Quantitative findings:** Overconfident CEOs show higher investment-cash flow sensitivity, pay larger acquisition premiums, and generate wider variance of outcomes. Board vigilance moderates the relationship (weaker boards amplify hubris effects)

**Relevance:** This research informed:
- The **management overconfidence bias** parameter (`mgmt_bias = 0.90, mgmt_bias_sigma = 0.12`) — the mean of 0.90 logit units reflects the substantial gap between Joyce's perceived and actual shareholder sentiment, calibrated against the literature finding that overconfident CEOs systematically overestimate their position
- The **board deference** parameter (`V = 0.30`) — informed by the Saar review's finding of "excessive deference" and the Hayward & Hambrick finding that weak boards amplify CEO hubris
- The **board optimism shift** (`board_optimism_shift = -0.8`) — the board's own tendency to underestimate governance risk, informed by the McKinsey finding that boards overestimate CEO performance relative to direct reports

**Used in:** Parameter calibration in `qantas-simulation.py` (`ToyParams` defaults).

---

### ceo-background/resigned-vs-sacked.md

**Contents:** Analysis of the financial and reputational consequences of CEO resignation versus termination, applied to Alan Joyce's case and supported by academic research. Covers:

- **Joyce's actual departure:** Accelerated "retirement" to 5 September 2023; preserved "good leaver" status; initial $21.4M FY23 pay reduced by $9.26M via clawback to ~$14.4M; still retained pro-rata LTIP vesting on 2022–2025 plans; received final $3.8M LTIP in September 2025
- **Stanford Push-out Score research (Gow, Larcker, Tayan 2017):** 23% of CEO departures clearly voluntary, 29% clearly involuntary, 48% ambiguous; mutual incentives to disguise involuntary exits
- **Evaluative stigma research:** Forced-departure CEOs experience demotion, reduced board appointments, lower pay, and longer job search
- **Face-saving dynamics:** Euphemisms ("personal reasons," "mutual agreement") serve face-saving function for both parties

**Relevance:** This research informed:
- The **CEO transition model** structure (logistic probability of transition, then conditional logistic for resign vs. sacked) — reflecting the real-world distinction between "good leaver" and "bad leaver" outcomes
- The **market reaction** parameters (`eta_ceo_resign = +0.012, eta_ceo_sacked = -0.006`) — markets react positively to resignation (accountability signal) but negatively to sacking (deeper-problems signal)
- The **management agency cost** parameters (`w_ceo_resign_mgmt = 0.8, w_ceo_sacked_mgmt = 2.4`) — sacking carries ~3x the personal cost of resignation, reflecting the financial and reputational penalty differential documented in the literature
- The CEO transition intercepts and slopes were structured so that resignation becomes more likely under high pressure + independent board, while sacking additionally requires strong board independence

**Used in:** Parameter calibration in `qantas-simulation.py` (CEO transition model and loss function).

---

### ceo-background/2nd-strike.md

**Contents:** Comprehensive analysis of Australia's "two strikes" rule on remuneration voting, including detailed data on second strikes and board spill outcomes from 2020–2024. Covers:

- **Second strike data (ASX 200/300, 2020–2024):** Year-by-year counts. No ASX 200 second strikes in 2023. 6 ASX 200 second strikes in 2024 (including Dicker Data and Lovisa with 4th consecutive strikes). Board spill resolutions averaged under 7% support in 2024 and 4.5% in 2025
- **Legal framework:** Sections 250U–250W of the Corporations Act. 25% threshold for a strike. Spill resolution requires simple majority. Spill meeting within 90 days. CEO is exempt from spill
- **Historical track record:** Since the rule's 2011 introduction, no incumbent directors have lost their seat at a spill meeting. Spill resolution passage rate ~15% for small companies, effectively zero for large-cap
- **Quantification of consequences:** NED fees, search costs, D&O insurance impacts, share price volatility, institutional knowledge loss
- **Behavioural paradox:** Institutional investors deliberately avoid triggering second strikes because they fear spill consequences more than they dislike pay outcomes, creating a natural ceiling on spill probability

**Relevance:** This research informed:
- The **spill probability** parameter (`p_spill_given_strike2 = 0.35`) — reflecting that while spill resolutions are automatic upon second strike, passage is uncertain and historically very rare at large-cap companies. The 0.35 is deliberately higher than the historical base rate (~0%) because the Qantas situation was unprecedented in intensity
- The **second-strike threshold** (`theta_strike2 = 0.25`) — directly from the statutory 25% threshold
- The **loss weights** for strike and spill events (`w_strike = 4.0, w_strike2 = 8.0, w_spill = 15.0`) — calibrated to reflect the escalating severity documented in the research: first strike is a signal, second strike triggers legal process, spill is catastrophic but has never succeeded
- The overall structure of the sequential decision problem, where strike_count is tracked across checkpoints and the second-strike/spill machinery only activates when strike_count >= 1

**Used in:** Parameter calibration in `qantas-simulation.py` (strike/spill model and loss weights). Design of the `DecisionState` sequential state structure.

---

### timing-tree.md

**Contents:** Decision tree specification for the sequential governance problem, defining the order of moves across checkpoints C0–C4, the action sets available to each actor (Board, ASA, CEO) at each node, and the state transitions.

**Key structure:**
- C0: Initialisation (beliefs loaded, no actions)
- C1: Board chooses from {D0 Do nothing, D1 Commission review, D2 Sack CEO}
- C2: ASA chooses from {DoNothing, RecommendStrike}
- C3a: Stochastic AGM outcome (vote realised)
- C3b: CEO may pre-empt (Resign or Stay), conditional on not already removed
- C4: Board responds to review findings (D0 or D2), if CEO still present

**Relevance:** Defines the game tree that the simulation implements. The checkpoint dates, actor move ordering, and state transition rules in `qantas-simulation.py` directly follow this specification.

**Used in:** Design of `qantas-simulation.py` sequential simulation logic.

---

## Summary: Data Flow Through the Pipeline

```
PRIMARY SOURCE DATA
├─ qantas_share_price_data.json ──► compute_abnormal_returns.py
│                                       ├─► abret_daily.csv
│                                       └─► abret_monthly.csv ──────────────────┐
│                                                                                │
├─ monthly_media_variables.xlsx ──► build_media_monthly_complete.py              │
│                                       └─► media_monthly_complete.xlsx          │
│                                             └─► prep_stan_media_data.py        │
│                                                   └─► stan_media_data.json     │
│                                                         └─► fit_media_better_stan.py
│                                                               ├─► media_better_draws.npz
│                                                               └─► media_better_summary.csv
│                                                                     │
│                                                         compute_media_shocks.py
│                                                               └─► media_shock_draws.npz ──┐
│                                                                                            │
├─ agm-votes.csv ───────────────────────────────────────────────────────────────┐│
├─ voting-recommendations.csv ──────────────────────────────────────────────┐   ││
│                                                                           │   ││
│                                           build_historical_belief_table.py◄───┘│
│                                                 └─► historical_belief_table.csv│
│                                                       └─► prep_stan_belief_data.py
│                                                             └─► stan_belief_data.json
│                                                                   └─► fit_belief_model_stan.py
│                                                                         └─► belief_state_draws.npz
│                                                                               │
│                                     fit_shock_priors.py ◄─────────────────────┤
│                                       └─► data/priors/shock_priors_*.json     │
│                                                   │                           │
│                                     checkpoint_update.py ◄────────────────────┘
│                                       └─► data/checkpoints/belief_C*.npz
│                                                   │
│                                     qantas-simulation.py
│                                       ├─► sim_summary_by_checkpoint.csv
│                                       └─► sim_summary_by_checkpoint_adversarial.csv
│
REFERENCE DOCUMENTS (inform parameter calibration)
├─ asa_background/process-prior-to-AGM.md ──► checkpoint timing, gamma_E design
├─ ceo-background/ceo-overconfidence.md ────► mgmt_bias, board deference, board_optimism_shift
├─ ceo-background/resigned-vs-sacked.md ───► CEO transition model, market reaction, agency costs
├─ ceo-background/2nd-strike.md ────────────► spill probability, strike thresholds, loss weights
└─ timing-tree.md ──────────────────────────► game tree structure, sequential decision logic
```
