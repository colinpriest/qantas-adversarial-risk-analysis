# Qantas Adversarial Risk Analysis (ARA) Engine — V2

Three-player adversarial risk analysis of Qantas governance decisions under strategic uncertainty. Combines Bayesian belief estimation (Stan checkpoints) with game-theoretic tree recursion to compute optimal policies for Board, ASA (Australian Shareholders Association), and CEO.

## Tech Stack

| Layer | Technology | Version | Purpose |
|-------|------------|---------|---------|
| Runtime | Python | 3.8+ | Scripting and numerical computing |
| Numerics | NumPy, Pandas | 1.20+, 1.3+ | Array operations and data manipulation |
| Bayesian | CmdStanPy | 1.0+ | Interface to Stan posterior samples |
| Data I/O | openpyxl | 3.0+ | Excel-based data contracts |
| Testing | pytest | (dev) | Comprehensive test suite (114 tests) |

**Why this stack:** Numerical game theory with uncertain beliefs requires explicit Bayesian reasoning (Stan), pandas for complex state transitions, and Excel for non-technical stakeholder accessibility.

## Quick Start

### Prerequisites
```bash
# Python 3.8+
python --version

# Install dependencies
pip install -r requirements.txt
```

### Development
```bash
# Build all 4 strategic modes, produces interactive HTML with mode selector
python -m run.run_unified_ARA --n_draws 500

# Without Laplacian smoothing
python -m run.run_unified_ARA --n_draws 500 --no-laplacian
```

### Testing
```bash
# Run all 114 tests
python -m pytest tests/test_engine.py -v

# Run specific test class
python -m pytest tests/test_engine.py::TestDataLoading -v

# Run with coverage
python -m pytest tests/test_engine.py --cov=engine --cov-report=html
```

## Project Structure

```
Qantas/
├── engine/                          # Core ARA engine (7 modules)
│   ├── __init__.py                  # Package marker
│   ├── state.py                     # DecisionState, BeliefBundle, ParameterSampler
│   ├── chance_models.py             # VoteModel, ReviewModel (endogenous stochasticity)
│   ├── utilities.py                 # utility_board(), utility_asa(), utility_ceo()
│   ├── modes.py                     # ModeConfig: focal actor configurations
│   ├── predictive.py                # PredictiveDistribution — ARA opponent modelling
│   ├── tree.py                      # TreeEvaluator — node-indexed value recursion
│   └── solver.py                    # Solver — full orchestration & parallelization
│
├── run/                             # CLI entry points
│   ├── __init__.py
│   ├── run_unified_ARA.py           # Unified game tree (all strategic modes)
│   ├── game_tree.py                 # Shared tree builder and utilities
│   ├── sensitivity.py               # Grid sensitivity over utility weights
│   └── visualise_tree.py            # Tree visualization (development utility)
│
├── tests/                           # Test suite (2 files, 114 tests)
│   ├── __init__.py
│   ├── test_engine.py               # Comprehensive engine tests across 14 classes
│   └── create_test_data.py          # Synthetic checkpoint generation
│
├── data/                            # Data contracts & checkpoints
│   ├── governance_spec.xlsx         # Game tree structure, actions, utilities (8 sheets)
│   ├── opponent_priors.xlsx         # Prior distributions for opponent parameters
│   └── checkpoints/                 # Belief checkpoint .npz files (Cpre, C0–C3)
│
├── outputs/                         # Results from run scripts (CSV)
├── docs/                            # Formal specifications and diagrams
├── models/                          # Stan models (belief_model.stan, media_better.stan)
├── deprecated/                      # Legacy V1 pipeline scripts
├── [background]/                    # Research materials (agm-pdfs, board-background, etc.)
├── README.md                        # User-facing documentation
├── requirements.txt                 # Python dependencies
└── CLAUDE.md                        # This file
```

## Architecture Overview

The engine operates in two tightly coupled layers:

**Layer A — Belief Estimation (External Stan):** Upstream Bayesian models produce posterior draws (500 per checkpoint) for market beliefs (`B_mkt`), management beliefs (`B_mgmt`), voting model parameters (`alpha_vote`, `gamma_A`, `gamma_D`, `sigma_vote`), and review findings parameters. These are serialized in `.npz` checkpoint files.

**Layer B — Adversarial Tree Recursion (Python):** The engine consumes checkpoints and Excel data contracts to execute a unified game tree recursion. The tree has 10 decision/chance/terminal nodes and supports:
- **D0_ceo** (CEO resign/stay) — pre-game branching point
- **D1** (Board governance reform) — minimal, review, or CEO transition
- **A2** (ASA strike recommendation) — no strike or recommend strike
- **V** (Shareholder vote) — logit-normal with belief-dependent parameters
- **D_rev** (Board review decision) — no action, commission, or sack CEO
- **R** (Review findings) — Bernoulli adverse outcome
- **D4** (CEO response) — stay, resign, or negotiate exit
- **Terminal** — compute utility for all actors

Node recursion pattern:
- **Terminal nodes:** Compute utility for target actor
- **Chance nodes (V, R):** Monte Carlo integration over sampled outcomes
- **Focal decision nodes:** Maximize expected utility over feasible actions
- **Opponent decision nodes:** Weight by predictive distribution (ARA best-response belief)

### Key Modules

| Module | Location | Purpose |
|--------|----------|---------|
| **state.py** | engine/ | Game state tracking (CEO_present, review status); feasibility rule enforcement; belief bundle loading; prior sampling |
| **chance_models.py** | engine/ | Vote percentage via logit-normal; review findings via Bernoulli; overconfidence bias scaling |
| **utilities.py** | engine/ | Actor-specific utility functions: Board minimizes opposition & disruption; ASA maximizes accountability; CEO maximizes CRRA wealth utility |
| **modes.py** | engine/ | Focal actor configurations (Board, ASA, Board-L2, ASA-L2); switches opponent roles and modelling depth |
| **predictive.py** | engine/ | ARA opponent modelling: samples opponent parameters from focal actor's priors, evaluates stochastic rollouts, returns empirical best-response distribution |
| **tree.py** | engine/ | Node-indexed game tree recursion with memoization; terminal utility computation; chance node integration; focal decision maximization |
| **solver.py** | engine/ | Orchestrates full pipeline: loads beliefs, iterates belief draws, evaluates actions via tree recursion, runs both scenarios, returns optimal action & EU |

## Development Guidelines

### Code Style
- **File naming:** snake_case (e.g., `chance_models.py`, `run_unified_ARA.py`)
- **Function naming:** snake_case with verb prefix (e.g., `feasible_actions()`, `load_governance_spec()`)
- **Class naming:** PascalCase (e.g., `DecisionState`, `BeliefBundle`, `TreeEvaluator`)
- **Variable naming:** snake_case for normal variables; `SCREAMING_SNAKE_CASE` for module-level constants
- **Boolean variables:** Prefix with `is_`, `has_`, `can_` (e.g., `CEO_present`, `review_commissioned`)

### Import Order
1. `__future__` imports (e.g., `from __future__ import annotations`)
2. Standard library (e.g., `logging`, `json`, `copy`)
3. Third-party (e.g., `numpy`, `pandas`, `openpyxl`)
4. Engine modules (e.g., `from engine.state import DecisionState`)
5. Type hints only imported at end if needed

### Data Contracts
All parameters are **externalized to Excel** (no magic numbers in code):

- **governance_spec.xlsx** — Game tree structure (node order, types, owners), feasibility rules, action sets, utility weights, board overconfidence parameters
- **opponent_priors.xlsx** — Prior distributions for opponent utility parameters (normal, lognormal, beta, uniform, gamma)
- **Belief checkpoints** (`.npz`) — Posterior draws from upstream Stan models

### Patterns & Conventions

**State Transitions:**
```python
# Immutable copies via dataclass copy()
new_state = current_state.apply(node_name, action)
# Feasibility rules automatically filtered
feasible = state.feasible_actions(node_name)
```

**Belief Indexing:**
```python
# BeliefBundle provides indexed access to posterior draws
belief_bundle = BeliefBundle("path/to/checkpoint.npz")
draw_i = belief_bundle.get_draw(i)  # Returns dict of parameters for draw i
```

**Tree Recursion:**
```python
# TreeEvaluator.evaluate(node_name, state, belief_draw) → float
# Automatically dispatches based on node type (decision/chance/terminal)
value = tree.evaluate("D1", current_state, belief_draw_i)
```

**Opponent Modelling:**
```python
# PredictiveDistribution computes empirical best-response distribution
# Over K opponent parameter samples, R stochastic rollouts per action
pred = PredictiveDistribution(beliefs, param_sampler, chance_models, K=200, R=20)
distribution = pred.compute(node_name, state, focal_beliefs_draw)
# Returns dict: {action: Pr(best_response), ...}
```

## Available Commands

| Command | Description |
|---------|-------------|
| `python -m run.run_unified_ARA --n_draws 500` | Build all 4 modes, interactive HTML with mode selector |
| `python -m run.run_unified_ARA --n_draws 500 --no-laplacian` | Same without Laplacian smoothing |
| `python -m pytest tests/test_engine.py -v` | Run all tests |
| `python -m pytest tests/test_engine.py::TestClass -v` | Run specific test class |

## Testing Strategy

**Test Coverage:** 114 tests across 14 classes covering:
- Data loading and validation (governance_spec, opponent_priors, checkpoints)
- Feasibility rules and state transitions
- Chance models (vote percentage, review findings)
- Utility function calculations
- Mode configurations and focal actor switching
- Predictive distributions (ARA opponent modelling)
- Tree evaluation (decision/chance/terminal nodes)
- Solver integration (end-to-end)
- Spec validation checklist
- Edge cases and overconfidence bias

**Test Data:** Synthetic checkpoints generated by `create_test_data.py` for reproducible testing without external dependencies.

**Run tests:**
```bash
python -m pytest tests/test_engine.py -v                    # All tests
python -m pytest tests/test_engine.py::TestSolver -v        # Specific class
python -m pytest tests/test_engine.py -k sensitivity -v     # Pattern matching
python -m pytest tests/test_engine.py --tb=short -v         # Short traceback
```

## Environment Variables

Not typically required for local development. Production deployments may specify:

| Variable | Purpose | Example |
|----------|---------|---------|
| `PYTHONPATH` | Project root for module imports | `D:/adversarial risk management/Qantas` |
| `CHECKPOINT_DIR` | Override checkpoint directory | `data/checkpoints` |
| `STAN_CACHE_DIR` | CmdStanPy build cache | `.cmdstan` |

## Output Format

All run scripts produce CSV files with columns:

```
checkpoint, scenario, Pr_scenario, focal, mode, action, EU, is_optimal,
Pr_strike, Pr_overwhelming, Pr_CEO_removed, Pr_review_adverse,
mean_vote_percent, sd_vote_percent
```

- **checkpoint:** Belief checkpoint (Cpre, C0–C3)
- **scenario:** CEO pre-game decision (ceo_stayed or ceo_resigned)
- **Pr_scenario:** ARA-predicted probability of scenario from focal actor's perspective
- **focal:** Focal actor (Board or ASA)
- **mode:** Mode configuration (Board Mode, ASA Mode, Board L2, etc.)
- **action:** Optimal first action (D0_minimal, D1_review, D3_ceo_transition, A2_rec_strike, etc.)
- **EU:** Expected utility of optimal action (averaged over belief draws)
- **is_optimal:** Boolean indicating whether action is optimal
- **Outcome statistics:** Pr_strike, Pr_overwhelming, Pr_CEO_removed, Pr_review_adverse, mean/sd vote%

## Key Design Decisions

1. **D0_ceo as genuine decision node** — CEO resignation (05-Sep-2023) is modelled as a strategic choice, not exogenous. Focal actor uses ARA to predict Pr(CEO_resign).

2. **Two-layer architecture** — Bayesian belief estimation (Stan) separated from game-theoretic recursion (Python) for modularity and interpretability.

3. **Excel data contracts** — No magic numbers. All parameters (utilities, priors, feasibility rules) externalized to Excel for stakeholder oversight.

4. **Level-1 and Level-2 opponent modelling** — Recursion guard (level decrement) prevents infinite loops; Level-2 allows analysis of strategic sophistication.

5. **Scenario branching at D0_ceo** — Both pre-game scenarios evaluated; solver computes Pr(CEO_resign) via ARA and attaches to results.

6. **Immutable game state** — DecisionState is immutable; apply() returns new copy to enable tree recursion without side effects.

7. **Feasibility rules as lookup table** — Rules stored in governance_spec.xlsx; evaluated dynamically as state changes.

## References

- **README.md** — User-facing documentation with full API reference
- **specs-for-version-2.md** — Detailed specifications (game tree, parameters, methodology)
- **docs/algebraic.md** — Formal algebraic notation for utilities and models
- **[Actor backgrounds]** — Research materials (agm-pdfs/, board-background/, asa_background/, ceo-background/)


## Skill Usage Guide

When working on tasks involving these technologies, invoke the corresponding skill:

| Skill | Invoke When |
|-------|-------------|
| pandas | Manages Pandas DataFrames for complex state transitions and data manipulation |
| openpyxl | Handles Excel data contracts (governance_spec.xlsx, opponent_priors.xlsx) |
| cmdstanpy | Interfaces with Stan posterior samples and Bayesian belief checkpoints |
| numpy | Handles NumPy array operations and numerical computations |
| python | Manages Python 3.8+ scripting, numerical computing, and module organization |
| stan | Manages Stan Bayesian models for belief estimation (belief_model.stan, media_better.stan) |
| pytest | Configures pytest test suite with 114 tests and coverage reporting |
