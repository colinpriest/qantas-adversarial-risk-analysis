## What was built

**board_utility_quantification.py** — ~4,300-line Python script implementing the full 6-stage pipeline from the spec, with a two-stage parameter estimation strategy and self-contained interactive HTML dashboard.

### 12 Sections


| Section | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| SEC 0   | Imports, constants. Anchored params (W_CAR=15.0, LAMBDA_LA=2.25), price table, param mappings. Two-stage parameter classification: FIXED (w1-w4, w9 — scenario-level, estimated via Stage 4B factor regression) and ESTIMABLE (w_removal, w8s, w_remove_ceo_overwhelming, w8r, w_inaction, w12, w13, w15 — action-varying, estimated via Stage 4A softmax MLE). Collinear groups collapsed: w7+w8→w_removal, w10+w11+w14→w_inaction. FACTOR_PARAM_MAP links scenario-level params to LLM factor ratings.                                                                                      |
| SEC 1   | Pydantic schemas: `ActionCode`, `ElicitationResponse`, `FactorRating`, `TokenUsage`, `RunCostSummary`                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| SEC 2   | Text sanitization: `sanitise_text()` with NFC normalization, smart quote replacement, unidecode fallback                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| SEC 3   | Caching: SHA256-keyed JSON cache. Key uses `hashlib.sha256` (not Python `hash()`) for deterministic cross-process stability. System prompt excluded from cache key (factor order is randomised but results are stored canonically). Hit/miss tracking with tqdm postfix display.                                                                                                                                                                                                                                                                                        |
| SEC 4   | LLM client: instructor + retry logic (rate limit: exp backoff 1-60s, server errors: 2-120s)                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| SEC 5   | **~95 scenarios** across 4 tiers. Tier 1 (~~40 identification): vote grid V∈{0.10..0.83}, CEO-resigned variants at 7 vote levels, w_inaction contrast pairs. Tier 2 (20 joint): multi-penalty combinations. Tier 3 (~~34 behavioural): loss aversion pairs, self-assessment bias (5 vote levels × 2 origins), Ikea effect (5 vote levels × 2 appointment types), optimism bias, non-linearity. Tier 4 (1 historical calibration).                                                                                                                                       |
| SEC 6   | Elicitation: ThreadPoolExecutor(10), `hashlib.sha256`-seeded factor ordering (deterministic), token limit handling, 40 reps/scenario default                                                                                                                                                                                                                                                                                                                                                                                                                            |
| SEC 7   | Preprocessing: Aggregate across seeds, filter ≥7/10 success, Stage 2 skip logic compares scenario ID sets                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| SEC 8   | **Stage 4A — Softmax MLE**: `decompose_utility_board()` mirroring engine exactly, phi matrix, L-BFGS-B with analytical gradient, lambda profiled over grid, Hessian SE + bootstrap SE (B=500), w10/w11/w14 collinearity collapse, Wald p-values                                                                                                                                                                                                                                                                                                                         |
| SEC 8B  | **Stage 4B — Factor rating regression**: OLS of mean factor ratings on scenario-level phi for w1-w4, w9. Special handling: w2 uses raw V with quadratic conversion. Bridge scaling via w_removal. Returns R², p-value, SE per parameter.                                                                                                                                                                                                                                                                                                                                |
| SEC 9   | Diagnostics: Loss aversion, non-linearity (AIC model comparison), optimism bias, self-assessment bias (5+ scenarios per group, t-test), Ikea effect (5+ scenarios per group, t-test), factor order effects (per-factor position regression).                                                                                                                                                                                                                                                                                                                            |
| SEC 10  | Validation: Per-scenario KL divergence, historical prediction (Tier 4 out-of-sample), factor regression validation. Interaction effects analysis: residual scatter, KL by node, strike×CEO interaction, Mann-Whitney fit tests, worst-fitting scenarios.                                                                                                                                                                                                                                                                                                                |
| SEC 11  | Dashboard: Self-contained HTML with embedded Plotly.js (~4.5MB), **12 tabs**, atomic writes, auto-refresh. Includes: Overview, Cost & Usage, Scenario Battery, Elicitation Results, Elicited Probabilities, Parameter Estimates (forest plot with 95% CIs, Wald p-values), Covariance (heatmap + numeric grid with high-correlation warnings), Behavioural Diagnostics (expanded detail for all 6 tests), Interaction Effects (scatter plots, box plots, heterogeneity tests), Validation, Linearity Diagnostics (Q-Q plot, scale-location, phi basis table), Raw Data. |
| SEC 12  | CLI: `--stage 1,2,3` selective execution, progressive dashboard updates, error handling                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |


### Two-Stage Estimation Design

The softmax choice model `P(a|s;w,λ) = exp(λ·EU(a)) / Σ exp(λ·EU(a'))` can only identify parameters whose phi varies across **actions** within a scenario. Five parameters (w1, w2, w3, w4, w9) have phi that depends only on scenario-level features (vote %, strike, overwhelming, CEO early departure), so they cancel in softmax and have zero gradient.

**Stage 4A** (softmax MLE): Estimates the 8 action-varying parameters + lambda (profiled).

**Stage 4B** (factor rating OLS): Estimates the 5 scenario-level parameters by regressing LLM factor ratings on scenario phi. Uses FACTOR_PARAM_MAP to link each parameter to the most informative factor(s). Special cases: w2 regresses on V directly (not (V-0.25)²) and converts via `w2 = gamma / (2*(V_ref - 0.25))`.

All 13 parameters are now estimated from data (none fixed at arbitrary spec defaults).

### Key design changes (from initial build)

1. **Deterministic caching**: Replaced Python `hash()` (process-randomised) with `hashlib.sha256` for factor order seeding. System prompt excluded from cache key.
2. **Expanded scenario battery**: Added 7 CEO-resigned scenarios for w1 identification (was 1, now 8). Added 16 extra scenarios for self-assessment bias and Ikea effect diagnostics (5 vote levels × 2 conditions each).
3. **w2 regression fix**: Uses V directly as regressor (not (V-0.25)² which compresses near zero), with quadratic conversion.
4. **Forest plot 95% CIs**: Error bars use ×1.96 for 95% confidence intervals. Spec defaults removed from plot to avoid scale compression.
5. **Wald p-values**: Added for all softmax MLE parameters using Abramowitz & Stegun normal CDF approximation.
6. **Correlation diagnostics**: Heatmap annotated with values + numeric grid table. Auto-detected warnings for |r|>0.8 with structural explanations (e.g., w8s↔w_inaction near-complementarity).
7. **Expanded behavioural diagnostics**: Self-assessment and Ikea effect now have 5+ observations per group for valid t-tests. Factor order effects show per-factor detail table.
8. **Interaction effects tab**: Replaced placeholder with residual scatter, KL-by-node box plots, strike×CEO interaction table, Mann-Whitney heterogeneity tests, worst-fitting scenarios.
9. **Linearity diagnostics tab**: New tab with residual-vs-vote scatter, Q-Q plot, scale-location heteroscedasticity check, phi basis function summary table.
10. **n_reps default**: Increased from 10 to 40 for better precision (29% SE reduction per doubling).

### Key outputs

- `outputs/scenarios.csv` — ~95 scenarios with state vectors and prompts
- `outputs/board_utility_dashboard.html` — self-contained interactive dashboard (12 tabs)
- `outputs/parameter_estimates.csv` — all 13 estimated weights with SEs and methods
- `outputs/covariance_matrix.csv` — full parameter covariance matrix
- `outputs/scenario_fit.csv` — per-scenario KL divergence and residuals
- `outputs/behavioural_diagnostics.csv` — all 6 diagnostic test results
- `outputs/validation_results.json` — within-sample fit and historical prediction

### Usage

```bash
python board_utility_quantification.py --stage 1          # scenarios only (no API key)
python board_utility_quantification.py --stage 1,2 --n_reps 2  # smoke test
python board_utility_quantification.py --all --n_reps 40  # full pipeline (default)
python board_utility_quantification.py --stage 4,5,6      # re-run estimation (uses cached elicitation)

python board_utility_quantification.py --all --n_draws 50
python board_utility_quantification.py --model gpt-5-mini --all

python -m run.apply_estimated_weights outputs/parameter_estimates.csv

```

