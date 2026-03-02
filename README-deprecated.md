# Qantas Adversarial Risk Assessment (ARA) Pipeline

An adversarial risk assessment model for Qantas governance decisions, combining financial market data analysis, Bayesian media sentiment modelling, shareholder voting behaviour, data-driven shock priors, belief checkpoint updating, and governance decision simulation.

## Prerequisites

- Python 3.8+
- [CmdStan](https://mc-stan.org/users/interfaces/cmdstan) (required for Bayesian model fitting)
- On Windows: [RTools 4.0](https://cran.r-project.org/bin/windows/Rtools/rtools40.html) installed at `C:\rtools40` for C++ compilation

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
Qantas/
├── data/
│   ├── qantas_share_price_data.json   # Raw daily share price + ASX200 index data
│   ├── monthly_media_variables.xlsx   # Sparse monthly media observations
│   ├── agm-votes.csv                  # AGM voting records
│   ├── voting-recommendations.csv     # Cross-company shareholder voting recommendations panel
│   ├── priors/                        # Data-driven shock priors (from fit_shock_priors.py)
│   └── checkpoints/                   # Belief checkpoint distributions (from checkpoint_update.py)
├── models/
│   ├── media_better.stan              # Bayesian media measurement model (Stan)
│   └── belief_model.stan              # Bayesian belief dynamics model (Stan, anchored scale)
├── agm-pdfs/                          # Source AGM results PDFs (QAN 2020–2022)
├── asa_background/                    # ASA engagement background documentation
├── plans/                             # Project planning documents
├── compute_abnormal_returns.py        # Step 1
├── build_media_monthly_complete.py    # Step 2
├── prep_stan_media_data.py            # Step 3
├── fit_media_better_stan.py           # Step 4
├── compute_media_shocks.py            # Step 5
├── build_historical_belief_table.py   # Step 6
├── prep_stan_belief_data.py           # Step 7
├── fit_belief_model_stan.py           # Step 8
├── fit_shock_priors.py                # Step 9
├── checkpoint_update.py               # Step 10
├── qantas-simulation.py              # Step 11
└── requirements.txt
```

## Pipeline Execution Order

The pipeline has two independent branches that converge at Step 6, then continues sequentially through belief estimation, shock prior fitting, checkpoint construction, and simulation. Steps 1 and 2 can run in parallel since they have no shared dependencies.

```
Step 1: compute_abnormal_returns.py ─────────────────────────────────┐
                                                                     │
Step 2: build_media_monthly_complete.py                              │
   │                                                                 │
Step 3: prep_stan_media_data.py                                      │
   │                                                                 │
Step 4: fit_media_better_stan.py                                     │
   │                                                                 │
Step 5: compute_media_shocks.py                                      │
   │                                                                 │
Step 6: build_historical_belief_table.py  ◄──────────────────────────┘
   │
Step 7: prep_stan_belief_data.py
   │
Step 8: fit_belief_model_stan.py
   │
   ├──► Step 9: fit_shock_priors.py
   │       │
   └──► Step 10: checkpoint_update.py ◄── (uses priors from Step 9)
          │
        Step 11: qantas-simulation.py
```

---

## Script Descriptions

### Step 1 — `compute_abnormal_returns.py`

Calculates market-adjusted abnormal returns from daily Qantas share price data.

- Computes daily log returns for Qantas and the ASX200 index
- Fits a market model via OLS: `r_qan = alpha + beta * r_mkt + epsilon`
- Derives abnormal returns: `AR_t = r_qan_t - (alpha + beta * r_mkt_t)`
- Aggregates daily abnormal returns to monthly summaries (sum, mean, volatility, negative-tail frequency)

**Input:** `data/qantas_share_price_data.json`
**Outputs:** `data/abret_daily.csv`, `data/abret_monthly.csv`

```bash
python compute_abnormal_returns.py
# or with explicit arguments:
python compute_abnormal_returns.py --json data/qantas_share_price_data.json --start 2020-10-01 --end 2023-09-30 --outdir data
```

---

### Step 2 — `build_media_monthly_complete.py`

Creates a complete monthly grid from sparse media observations, making missingness explicit.

- Builds a full month-by-month grid for the specified date window
- Left-joins observed media variables (event count, damage intensity, sentiment, response quality, concentration index)
- Unobserved months remain as NaN rather than being filled with zeros
- Generates per-field observation masks (e.g. `intensity_observed`, `event_count_observed`)
- Adds a scaled time index `t_scaled` in [-1, 1] for trend modelling

**Input:** `data/monthly_media_variables.xlsx`
**Output:** `data/media_monthly_complete.xlsx`

```bash
python build_media_monthly_complete.py --xlsx data/monthly_media_variables.xlsx --start 2020-10 --end 2023-09 --out data/media_monthly_complete.xlsx
```

---

### Step 3 — `prep_stan_media_data.py`

Transforms the complete media grid into the JSON format required by the Stan media measurement model.

- Extracts observed intensity values and their 1-based time indices (for Stan's array indexing)
- Includes the scaled time vector for the coverage trend
- Optionally includes event count data (`--include_counts`)
- Validates that no intensity values are negative

**Input:** `data/media_monthly_complete.xlsx` (from Step 2)
**Output:** `data/stan_media_data.json`

```bash
python prep_stan_media_data.py --xlsx data/media_monthly_complete.xlsx --out data/stan_media_data.json
```

---

### Step 4 — `fit_media_better_stan.py`

Fits the Bayesian media measurement model using MCMC sampling via CmdStanPy.

- Compiles `models/media_better.stan` (first run only; reuses cached executable afterwards)
- Runs 4 MCMC chains (2000 warmup + 2000 sampling iterations each)
- The model estimates two latent time series:
  - **C_t** — media coverage/concentration (logistic trend with floor)
  - **M_t** — true media damage intensity (AR(1) on log scale)
- Extracts full posterior draws and computes summaries (mean, 5th/95th percentiles)

**Inputs:** `data/stan_media_data.json` (from Step 3), `models/media_better.stan`
**Outputs:** `data/media_better_summary.csv`, `data/media_better_draws.npz`, `data/media_better_diagnostics.txt`

```bash
python fit_media_better_stan.py --data data/stan_media_data.json --stan models/media_better.stan --outdir data
```

> Note: The first run compiles the Stan model, which takes 1-2 minutes. Subsequent runs reuse the compiled executable.

---

### Step 5 — `compute_media_shocks.py`

Converts posterior media intensity draws into month-to-month log-difference shocks.

- Loads the full posterior draws for M_t
- Computes `shock_t = log(M_t) - log(M_{t-1})` for each MCMC draw
- The first month's shock is set to zero (no prior reference point)

**Input:** `data/media_better_draws.npz` (from Step 4)
**Output:** `data/media_shock_draws.npz` (contains `shock_draws`, `logM_draws`, `year_month`)

```bash
python compute_media_shocks.py
```

---

### Step 6 — `build_historical_belief_table.py`

Merges all data sources into a single unified monthly table — the convergence point of the two upstream branches.

- Creates a base monthly grid with time index and media shock statistics (mean, 5th/95th percentiles across posterior draws)
- Left-joins monthly abnormal returns (sum and volatility)
- Left-joins AGM voting data (percentage votes against remuneration report, percentage votes against chair)
- Adds binary ASA engagement indicators for specific months (private engagement: Sep–Nov 2023; public mobilisation: Oct–Nov 2023)
- Left-joins cross-company voting recommendation features from `voting-recommendations.csv`:
  - **Market-wide aggregates** — monthly counts and size-weighted sums of ASA recommendations, proxy adviser opposition, multi-targeting, headline incidents, first strikes, and mean remuneration opposition
  - **Qantas-specific flags** — binary indicators and continuous metrics for Qantas (prefixed `qan_`)

**Inputs:**
- `data/abret_monthly.csv` (from Step 1)
- `data/media_shock_draws.npz` (from Step 5)
- `data/agm-votes.csv`
- `data/voting-recommendations.csv`

**Output:** `data/historical_belief_table.csv`

```bash
python build_historical_belief_table.py --start 2020-10 --end 2023-12
# or with explicit paths:
python build_historical_belief_table.py --abret_csv data/abret_monthly.csv --shock_npz data/media_shock_draws.npz --agm_csv data/agm-votes.csv --vote_csv data/voting-recommendations.csv --out data/historical_belief_table.csv
```

---

### Step 7 — `prep_stan_belief_data.py`

Prepares the historical belief table for the Stan belief dynamics model.

- Z-score standardises media shocks and abnormal returns (records the normalisation constants for later back-transformation)
- Extracts AGM voting observations with 1-based indices for Stan:
  - Remuneration report opposition (`rem_idx`, `rem_obs`)
  - Chair opposition (`chair_idx`, `chair_obs`)

**Input:** `data/historical_belief_table.csv` (from Step 6)
**Output:** `data/stan_belief_data.json`

```bash
python prep_stan_belief_data.py
```

---

### Step 8 — `fit_belief_model_stan.py`

Fits the Bayesian belief state-space model using MCMC sampling.

- Compiles `models/belief_model.stan` (first run only)
- Runs 4 MCMC chains (2000 warmup + 2000 sampling iterations each)
- Uses an **anchored scale** where `lambda_rem` is fixed to 1.0, so the latent belief state B_t is denominated in remuneration-vote logit units
- Estimates a latent belief state **B_t** that evolves over time, driven by media shocks and abnormal returns, with AGM vote shares as noisy observations
- Extracts full posterior draws for B_t and additional parameters needed by downstream checkpoint updating: `alpha_rem`, `lambda_rem` (constant 1.0), `sigma_rem`, `beta`, `sigma_B`, `sigma_B0`, `rho`

**Inputs:** `data/stan_belief_data.json` (from Step 7), `models/belief_model.stan`
**Outputs:** `data/belief_state_summary.csv`, `data/belief_state_draws.npz`, `data/belief_state_diagnostics.txt`

```bash
python fit_belief_model_stan.py --data data/stan_belief_data.json --stan models/belief_model.stan --outdir data
```

---

### Step 9 — `fit_shock_priors.py`

Fits data-driven priors for `gamma_A` (the ASA public mobilisation shock magnitude) from the cross-company voting recommendations panel.

Two estimation channels:
- **Vote channel** — OLS regression of `logit(rem_against_pct)` on `asa_against` with controls (prior-year opposition, log market cap, GICS industry fixed effects). Produces a prior on the logit-scale treatment effect.
- **Strike channel** — Bayesian logistic regression (MAP + Laplace approximation) of `first_strike` on `asa_against` with the same controls. Uses `N(0, 2.5^2)` prior on coefficients to handle quasi-separation. Produces a prior on the log-odds-scale effect.
- **Combined prior** — inverse-variance weighted blend of the vote and strike channel estimates.
- Optionally maps the vote-logit effect onto the belief scale by dividing by `lambda_rem` posterior draws from Step 8.

As-of filtering ensures that only data available before each checkpoint date is used, preventing look-ahead bias.

**Inputs:**
- `data/voting-recommendations.csv`
- `data/belief_state_draws.npz` (from Step 8, optional — for belief-scale mapping via `--belief_npz`)

**Output:** `data/priors/shock_priors_*.json`

```bash
# Single as-of date:
python fit_shock_priors.py --asof 2023-10-01

# All four checkpoint dates at once:
python fit_shock_priors.py --write_all_checkpoints

# With belief-scale mapping:
python fit_shock_priors.py --write_all_checkpoints --belief_npz data/belief_state_draws.npz
```

---

### Step 10 — `checkpoint_update.py`

Builds belief distributions at four checkpoints in the Qantas 2023 crisis timeline, incorporating event-specific shocks and a Bayesian measurement update at the AGM.

The four checkpoints are:

| Checkpoint | Date | Event |
|------------|------|-------|
| C0 | 2023-10-01 | Pre-mobilisation — market does not see ASA campaign; management aware of private engagement |
| C1 | 2023-10-10 | Review announcement — governance review shock applied to both market and management beliefs |
| C2 | 2023-10-18 | Public mobilisation — ASA campaign becomes visible to the market |
| C3 | 2023-11-03 | AGM — remuneration vote observed (82.9% against); importance-weighted resampling updates beliefs |

Each checkpoint produces separate **market** and **management** belief distributions, reflecting their asymmetric information sets:
- `gamma_E` (private engagement shock) drawn as `kappa * gamma_A` where `kappa ~ Beta(2, 2)` — management sees this from Sep 2023, the market does not
- `gamma_A` (public mobilisation shock) drawn from data-driven prior (Step 9) or CLI fallback
- `gamma_review` (review announcement shock) drawn from `N(-0.2, 0.2)` by default
- C3 applies importance-weighted resampling using the observed remuneration vote and the Stan posterior for `alpha_rem` and `sigma_rem` (with `lambda_rem = 1` enforced by an anchored-model assertion)

**Inputs:**
- `data/belief_state_draws.npz` (from Step 8)
- `data/historical_belief_table.csv` (from Step 6)
- `data/priors/shock_priors_C2_2023-10-18.json` (from Step 9, default; configurable)

**Outputs:** `data/checkpoints/belief_C0_2023-10-01.npz` through `belief_C3_2023-11-03.npz`

Each `.npz` contains: `B_mkt`, `B_mgmt`, `gamma_A`, `gamma_E`, `gamma_review`, `agm_rem_against`, `gamma_A_prior` metadata.

```bash
python checkpoint_update.py

# With explicit paths:
python checkpoint_update.py --npz data/belief_state_draws.npz --table data/historical_belief_table.csv --outdir data/checkpoints

# Use CLI-specified shock parameters instead of data-driven priors:
python checkpoint_update.py --no_use_gamma_A_prior_json --gamma_A_mean 0.8 --gamma_A_sd 0.4
```

---

### Step 11 — `qantas-simulation.py`

Simulates governance decision outcomes using the ARA framework, driven by checkpoint belief distributions from Step 10.

Evaluates four governance packages against a loss function:

| Package | Description | Components |
|---------|-------------|------------|
| D0 | Minimal response | No substantive action |
| D1 | Review-first | Independent review + timeline specificity |
| D2 | Accountability-lite | Remuneration clawback + high transparency |
| D3 | CEO transition | Independent review + high transparency + CEO change |

The simulator iterates over all four checkpoints (C0–C3), loading the market belief draws (`B_mkt`) from each checkpoint `.npz` as the initial belief distribution. For each checkpoint and package it:
1. Resamples initial beliefs from the checkpoint posterior (or falls back to a parametric prior if no draws are provided)
2. Evolves the belief forward over a checkpoint-specific horizon (1–2 months to AGM) under the governance action's feature impacts
3. Computes AGM opposition probabilities (remuneration strike > 25%, board revolt > 30%) — or uses observed outcomes if provided
4. Samples market reaction (cumulative abnormal return around announcement)
5. Estimates exit pressure from institutional shareholders
6. Calculates a weighted loss combining implementation cost, share price impact, strike/revolt risk, and exit pressure

Key parameter defaults aligned with the anchored belief model: `beta_X = 0.0` (checkpoint posteriors already incorporate information to date), `kappa_rem = 1.0` (consistent with `lambda_rem = 1` anchoring).

At C3 (post-AGM), strike/revolt weights are zeroed since these outcomes are already realised and no longer stochastic.

**Inputs:** `data/checkpoints/belief_C0_2023-10-01.npz` through `belief_C3_2023-11-03.npz` (from Step 10)
**Outputs:** Console report per checkpoint + `data/sim_summary_by_checkpoint.csv`

```bash
python qantas-simulation.py
```

---

## Full Pipeline — Quick Reference

```bash
# Step 1: Abnormal returns from share price data
python compute_abnormal_returns.py

# Step 2: Complete monthly media grid from sparse observations
python build_media_monthly_complete.py --xlsx data/monthly_media_variables.xlsx --start 2020-10 --end 2023-09 --out data/media_monthly_complete.xlsx

# Step 3: Format media data for Stan
python prep_stan_media_data.py --xlsx data/media_monthly_complete.xlsx --out data/stan_media_data.json

# Step 4: Fit Bayesian media model (MCMC — slow on first run)
python fit_media_better_stan.py --data data/stan_media_data.json --stan models/media_better.stan --outdir data

# Step 5: Compute media shocks from posterior draws
python compute_media_shocks.py

# Step 6: Merge all data into historical belief table
python build_historical_belief_table.py --start 2020-10 --end 2023-12

# Step 7: Prepare belief data for Stan
python prep_stan_belief_data.py

# Step 8: Fit Bayesian belief model (MCMC — slow on first run)
python fit_belief_model_stan.py --data data/stan_belief_data.json --stan models/belief_model.stan --outdir data

# Step 9: Fit data-driven shock priors from voting panel
python fit_shock_priors.py --write_all_checkpoints --belief_npz data/belief_state_draws.npz

# Step 10: Build belief checkpoint distributions
python checkpoint_update.py

# Step 11: Run governance decision simulation
python qantas-simulation.py
```

## Raw Data Files

| File | Description |
|------|-------------|
| `data/qantas_share_price_data.json` | Daily Qantas close price, ASX200 index close, and volume |
| `data/monthly_media_variables.xlsx` | Monthly media metrics: event count, damage intensity, sentiment, response quality, concentration index |
| `data/agm-votes.csv` | AGM voting results (% against remuneration report, % against chair) |
| `data/voting-recommendations.csv` | Cross-company shareholder voting recommendations panel with ASA targeting, proxy adviser positions, market cap, GICS codes, and strike history |

## Stan Models

| Model | File | Purpose |
|-------|------|---------|
| Media measurement | `models/media_better.stan` | Estimates latent media coverage (logistic trend) and intensity (log-scale AR(1)) from sparse observations |
| Belief dynamics | `models/belief_model.stan` | Anchored-scale state-space model for latent shareholder belief (in rem-vote logit units), driven by media shocks and abnormal returns, observed through AGM votes. `lambda_rem` is fixed to 1 for identifiability |

## Reference Materials

| Directory | Contents |
|-----------|----------|
| `agm-pdfs/` | Source AGM results PDFs (Qantas 2020–2022) |
| `asa_background/` | ASA engagement process documentation |
| `ceo-background/` | CEO overconfidence and departure analysis |
| `plans/` | Project planning and versioned design documents |
