# Qantas Adversarial Risk Analysis (ARA) — V2 Engine

A three-player adversarial risk analysis engine for Qantas governance decisions. The system combines Bayesian belief estimation (Stan) with game-theoretic tree recursion (Python) to compute optimal governance policies under strategic uncertainty.

**Actors:** Board, ASA (Australian Shareholders Association), CEO

## Architecture

The system is organized into two layers:

- **Layer A — Bayesian estimation (Stan):** Existing Stan models produce belief checkpoint `.npz` files containing posterior draws for market/management beliefs and model parameters. See [deprecated pipeline](README-deprecated.md) for the upstream estimation steps.
- **Layer B — Adversarial tree engine (Python):** The `engine/` package consumes checkpoints and Excel data contracts to evaluate governance decisions via ARA recursion.

## Project Structure

```
Qantas/
├── engine/                            # Core ARA engine package
│   ├── state.py                       # DecisionState, BeliefBundle, ParameterSampler
│   ├── chance_models.py               # VoteModel, ReviewModel (endogenous chance nodes)
│   ├── utilities.py                   # utility_board(), utility_asa(), utility_ceo()
│   ├── modes.py                       # ModeConfig: Board/ASA L1/L2 analysis modes
│   ├── predictive.py                  # PredictiveDistribution — ARA opponent modelling
│   ├── tree.py                        # TreeEvaluator — node-indexed recursion
│   └── solver.py                      # Solver — full orchestration across draws & actions
│
├── run/                               # CLI entry points
│   ├── run_unified_ARA.py             # Unified game tree (all strategic modes)
│   ├── game_tree.py                   # Shared tree builder and utilities
│   ├── sensitivity.py                 # Grid sensitivity analysis
│   └── apply_estimated_weights.py     # Apply quantification outputs to governance_spec
│
├── board_utility_quantification.py    # Board utility parameter estimation pipeline
├── utility-quantification/            # Quantification supporting files
│   ├── cache/                         # LLM response cache (SHA-256 keyed)
│   ├── ara_board_utility_experiment_spec.md  # Formal specification
│   └── what-was-built.md             # Implementation notes
│
├── data/
│   ├── governance_spec.xlsx           # Game tree structure, actions, utilities (8 sheets)
│   ├── opponent_priors.xlsx           # Prior distributions for opponent parameters
│   └── checkpoints/                   # Belief checkpoint .npz files (Cpre, C0–C3)
│
├── tests/
│   ├── test_engine.py                 # 114 tests across 14 test classes
│   └── create_test_data.py            # Generate synthetic checkpoint data
│
├── outputs/                           # Results from run scripts (CSV, HTML dashboards)
├── models/                            # Stan models (belief_model.stan, media_better.stan)
├── deprecated/                        # Legacy V1 pipeline scripts
└── requirements.txt
```

## Prerequisites

- Python 3.8+
- Dependencies: `pip install -r requirements.txt` (numpy, pandas, openpyxl, cmdstanpy)

## Quick Start

```bash
# Build all 4 strategic modes in one run, produces interactive HTML with mode selector
python -m run.run_unified_ARA --n_draws 500

# Run tests
python -m pytest tests/test_engine.py -v
```

## Run Scripts

### `run_unified_ARA.py` — Unified Game Tree

Builds all four strategic modes (All Stochastic, Board Strategic, ASA Strategic, CEO Strategic) in a single run. Produces one interactive HTML dashboard with a mode selector dropdown to switch between views in the browser — no need to re-run the script.

All three actor EU streams (Board, CEO, ASA) are propagated through every node. Board decisions always use argmax-count from posterior weight draws for stochastic probabilities. Generates PNG tree diagrams (stochastic mode) and interactive HTML dashboard with multi-actor utility decomposition.


| Argument             | Default                                    | Description                                           |
| -------------------- | ------------------------------------------ | ----------------------------------------------------- |
| `--n_draws`          | `500`                                      | Number of posterior weight draws                      |
| `--posterior-draws`  | `outputs/stan_posterior_draws.npz`          | Path to posterior draws file                          |
| `--param-estimates`  | `outputs/parameter_estimates.csv`           | Path to parameter estimates (for vote weights)        |
| `--no-laplacian`     | off                                        | Disable Laplacian smoothing on stochastic decision probs |
| `--seed`             | `42`                                       | Random seed                                           |
| `--output`           | (none)                                     | Output CSV path                                       |


### `sensitivity.py` — Parameter Sensitivity

Sweeps a grid over utility weights (81 combinations by default) and tracks how the optimal action shifts.


| Argument       | Default                           | Description                          |
| -------------- | --------------------------------- | ------------------------------------ |
| `--focal`      | `Board`                           | Focal actor (Board or ASA)           |
| `--checkpoint` | `C0`                              | Belief checkpoint                    |
| `--n_draws`    | `20`                              | Draws per grid point                 |
| `--K`          | `50`                              | Opponent samples (reduced for speed) |
| `--R_rollouts` | `10`                              | Rollouts (reduced for speed)         |
| `--output`     | `outputs/sensitivity_results.csv` | Output path                          |


### `board_utility_quantification.py` — Board Utility Parameter Estimation

A supporting calibration pipeline that estimates the Board utility function weights using LLM stakeholder simulation (gpt-4o-mini). The engine (`run_unified_ARA.py`) uses the resulting parameter values from `governance_spec.xlsx`.

The pipeline uses a two-stage estimation strategy:
- **Stage 4A (Softmax MLE):** Estimates 8 action-varying parameters from choice probabilities
- **Stage 4B (Factor Rating OLS):** Estimates 5 scenario-level parameters from LLM factor importance ratings

```bash
# Full pipeline (~$0.60 at gpt-4o-mini prices)
python board_utility_quantification.py --all

# Re-run estimation only (uses cached LLM responses)
python board_utility_quantification.py --stage 4,5,6

# Apply estimates to governance_spec.xlsx
python -m run.apply_estimated_weights outputs/parameter_estimates.csv

# Preview without writing
python -m run.apply_estimated_weights outputs/parameter_estimates.csv --dry-run
```

| Argument       | Default           | Description                          |
| -------------- | ----------------- | ------------------------------------ |
| `--stage`      | `all`             | Comma-separated stages (1-6) or 'all' |
| `--model`      | `gpt-4o-mini`    | LLM model for elicitation            |
| `--n_reps`     | `40`              | Repetitions per scenario             |
| `--n_starts`   | `10`              | L-BFGS-B starting points             |
| `--bootstrap_B`| `500`             | Bootstrap samples for SEs            |
| `--output_dir` | `outputs/`        | Output directory                     |

Output: `outputs/board_utility_dashboard.html` (self-contained interactive dashboard with 12 tabs). See [docs/board-utility-quantification.md](docs/board-utility-quantification.md) for full documentation.


## Game Tree

The analysis starts on **31-Aug-2023** (ACCC legal action against Qantas). The first branch point is the CEO's resignation decision on **05-Sep-2023**, which splits the tree into two scenarios:

```
D0_ceo  [CEO: resign or stay?]  (05-Sep-2023)
 │
 ├─ CEO_resign (actual: what happened)
 │   └─► D1  [Board: governance reform]
 │        ├─ D0_minimal  /  D1_review        ← no D3 (CEO already gone)
 │        └─► A2 → V → M_agm → D_rev → R → M_rev → Terminal
 │                                ↑ no Drev_sack_ceo     ↑ no D4
 │
 └─ CEO_stay (counterfactual)
     └─► D1  [Board: governance reform]
          ├─ D0_minimal  /  D1_review  /  D3_ceo_transition
          │
          └─► A2  [ASA: strike recommendation]
               ├─ A2_no_strike  /  A2_rec_strike
               │
               └─► V  [Nature: shareholder vote]
                    │   logit(V) ~ N(alpha + B_agm, sigma)
                    │
                    └─► M_agm  [Market reaction]
                         │
                         └─► D_rev  [Board: review / CEO removal]
                              ├─ Drev_no_action  /  Drev_commission_review  /  Drev_sack_ceo
                              │
                              └─► R  [Nature: review findings]
                                   │   R ~ Bernoulli(pi_R)  if commissioned
                                   │
                                   └─► M_rev  [Market reaction]
                                        │
                                        └─► D4  [CEO: respond]  (only if CEO present)
                                             ├─ D4_stay  /  D4_resign  /  D4_negotiate_exit
                                             │
                                             └─► Terminal  →  compute utility
```

**Feasibility rules:** CEO removal at D3 or D_rev automatically eliminates D4 options. D3_ceo_transition requires `CEO_present` (infeasible if CEO already resigned). Review commission is gated by `review_not_commissioned`. CEO actions require `CEO_present`.

## Data Contracts

### `governance_spec.xlsx` (8 sheets)


| Sheet                  | Contents                                                                           |
| ---------------------- | ---------------------------------------------------------------------------------- |
| `node_order`           | Node sequence, types (decision/chance/terminal), and owners                        |
| `action_sets`          | Actions per node with feasibility codes                                            |
| `vote_thresholds`      | Strike (25%) and overwhelming (50%) thresholds                                     |
| `utilities_board`      | Board utility weight parameters                                                    |
| `utilities_asa`        | ASA utility weight parameters                                                      |
| `utilities_ceo`        | CEO utility weight parameters                                                      |
| `policy_parameters`    | Fixed Level-1 policy parameters                                                    |
| `board_overconfidence` | Board cognitive bias: governance effect bounds + sigma_scale (Saar/Joyce evidence) |


### `opponent_priors.xlsx`

Prior distributions (normal, lognormal, beta, uniform, gamma) for each perspective-target actor pair used in ARA opponent modelling:

- Board's beliefs about ASA and CEO utility parameters
- ASA's beliefs about Board and CEO utility parameters

### Belief Checkpoints (`data/checkpoints/`)


| Checkpoint | Date       | Event                                   |
| ---------- | ---------- | --------------------------------------- |
| Cpre       | 2023-08-31 | ACCC legal action (analysis start date) |
| C0         | 2023-10-01 | Pre-mobilisation baseline               |
| C1         | 2023-10-10 | Governance review announced             |
| C2         | 2023-10-18 | ASA public mobilisation                 |
| C3         | 2023-11-03 | Pre-AGM peak distrust                   |


Each `.npz` contains 500 posterior draws: `B_mkt`, `B_mgmt`, `alpha_vote`, `gamma_A`, `gamma_D`, `sigma_vote`, `review_param_1`, `review_param_2`.

## Engine Modules

### `state.py` — Game State & Data Loading

- `**DecisionState`** — Tracks `CEO_present`, `review_commissioned`, `review_completed`, `CEO_removed`, `CEO_resigned_early`. Enforces feasibility rules from `governance_spec.xlsx`. Provides `feasible_actions()`, `apply()`, `next_node()`, `for_scenario()`. D0_ceo actions (`CEO_resign`, `CEO_stay`) are handled by `apply()`; `for_scenario()` delegates to it.
- `**BeliefBundle**` — Loads checkpoint `.npz` files. `get_draw(i)` returns all parameters for draw *i*.
- `**ParameterSampler*`* — Samples opponent utility parameters from priors in `opponent_priors.xlsx`.

### `chance_models.py` — Stochastic Outcomes

- `**VoteModel**` — Vote percentage via logit-normal: `logit(V) ~ N(alpha + B_mkt + gamma_A * strike + gamma_D * reform, sigma)`. ASA strike amplifies opposition; governance reform dampens it. Board overconfidence bias scales sigma down (`sigma_scale < 1`) to model overprecision.
- `**ReviewModel**` — Adverse finding via `Bernoulli(expit(review_param_1 + adjustments))`, conditional on review being commissioned.

### `utilities.py` — Actor Utility Functions


| Actor | Objective                                                   | Key terms                                                                                                                                                       |
| ----- | ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Board | Minimize opposition & disruption                            | Vote penalty (quadratic above 25%), CEO loss cost, reform implementation cost                                                                                   |
| ASA   | Maximize accountability outcomes                            | Vote reward (linear), CEO removal bonus, adverse review bonus, mobilisation cost                                                                                |
| CEO   | Maximize CRRA wealth utility, minimize non-monetary penalty | U = W^(1-γ)/(1-γ) − D; wealth by departure mode (W_resign, W_stay_sacked, W_stay_kept); additive penalty D (sacking, AGM humiliation, disgrace, adverse review) |


### `modes.py` — Analysis Configurations


| Mode                    | Focal | Opponents             | Level |
| ----------------------- | ----- | --------------------- | ----- |
| `MODE_BOARD`            | Board | ASA=ARA, CEO=ARA      | 1     |
| `MODE_ASA`              | ASA   | Board=ARA, CEO=ARA    | 1     |
| `MODE_BOARD_L2`         | Board | ASA=ARA, CEO=ARA      | 2     |
| `MODE_ASA_L2`           | ASA   | Board=ARA, CEO=ARA    | 2     |
| `MODE_ASA_POLICY_BOARD` | ASA   | Board=Policy, CEO=ARA | 1     |


### `predictive.py` — ARA Opponent Modelling

Computes predictive distributions over opponent actions. For each of *K* parameter samples from the focal actor's priors about the opponent: sample opponent utility parameters, evaluate *R* stochastic rollouts per feasible action, identify opponent's best response. Returns empirical distribution over best responses.

Level-2 modelling recurses: opponents model the focal actor strategically (with level decrement to prevent infinite recursion).

### `tree.py` — Game Tree Recursion

Node-indexed value computation:

- **Terminal:** compute utility for target actor
- **Chance (V, R):** Monte Carlo integration over sampled outcomes
- **Focal decision:** maximize over feasible actions
- **Opponent decision:** expectation weighted by predictive distribution

### `solver.py` — Orchestrator

`Solver.solve()` iterates over belief draws and feasible initial actions, delegates to `TreeEvaluator`, and returns a `SolveResult` with expected utilities, optimal action, and outcome statistics (Pr_strike, Pr_CEO_removed, mean_vote_percent, etc.). Accepts a `scenario` parameter (`"ceo_stayed"` or `"ceo_resigned"`) to condition the tree. `solve_scenarios()` runs both scenarios, computes the D0_ceo predictive distribution (Pr(CEO_resign) via ARA), and attaches predicted scenario probabilities to each result. `predict_d0_ceo()` returns the focal actor's ARA-predicted distribution over CEO's resign/stay decision.

## Output Format

```
checkpoint | scenario     | Pr_scenario | focal | mode       | action            | EU    | is_optimal
C0         | ceo_stayed   | 0.27        | Board | Board Mode | D0_minimal        | -0.42 | False
C0         | ceo_stayed   | 0.27        | Board | Board Mode | D1_review         | -0.38 | True
C0         | ceo_stayed   | 0.27        | Board | Board Mode | D3_ceo_transition | -0.50 | False
C0         | ceo_resigned | 0.73        | Board | Board Mode | D0_minimal        | -0.35 | False
C0         | ceo_resigned | 0.73        | Board | Board Mode | D1_review         | -0.30 | True
```

`Pr_scenario` is the ARA-predicted probability of the CEO's D0_ceo action from the focal actor's perspective.

Outcome statistics per action include: `Pr_strike`, `Pr_overwhelming`, `Pr_CEO_removed`, `Pr_review_adverse`, `mean_vote_percent`, `sd_vote_percent`.

## Key Design Decisions

- **D0_ceo as a genuine decision node** — CEO resignation (05-Sep-2023) is modelled as a strategic decision, not an exogenous parameter. The Board/ASA uses ARA opponent modelling to predict Pr(CEO_resign), while feasibility rules automatically prune downstream nodes when `CEO_present=False`
- **Level-1 and Level-2 opponent modelling** with recursion guard (level decrement prevents infinite loops)
- **Rollout-based Psi computation** (configurable K opponent samples, R rollouts)
- **All parameters externalized to Excel** — no magic numbers in code
- **Focal symmetry verified:** swapping Board/ASA flips max/expectation at decision nodes
- **CEO removal automatically eliminates D4 nodes** via feasibility rules

## Tests

114 tests across 14 classes covering data loading, feasibility rules, chance models, utilities, mode configurations, predictive distributions, tree evaluation, solver integration, spec validation, edge cases, overconfidence bias, D0_ceo decision node, scenario conditioning, and scenario utilities.

```bash
python -m pytest tests/test_engine.py -v
```

