# Board Utility Quantification — Prompt Templates

Documentation of the LLM prompt templates used in `board_utility_quantification.py` for eliciting Board decision probabilities via gpt-4o-mini (instructor).

---

## Overview

The elicitation system uses a two-part prompt architecture:

1. **System prompt** — Fixed across all queries. Establishes the Board persona, legal context, historical context, and response format instructions. Built by `_build_system_prompt()`.
2. **Scenario prompt** — Varies per scenario. Describes the specific game state and asks for action probabilities. Built by `_build_scenario_prompt(scenario)`.

Each LLM call also receives a **randomized factor ordering** (10 factors shuffled per call) to detect and control for order effects (anchoring bias diagnostic).

---

## System Prompt (`_build_system_prompt`)

The system prompt has four sections (matching spec §4.2 Sections A–D):

### Section A: Board Persona

```
You are simulating the boardroom deliberations of the Qantas Airways Board of
Directors in late 2023. The Board members include Chairman Richard Goyder, and
independent directors Maxine Brenner, Jacqueline Hey, Michael L'Estrange, Todd
Sampson, Heather Smith, Barbara Ward, and Doug Parker. Each director brings
distinct professional backgrounds -- finance, law, public policy, media,
technology, aviation operations -- and different risk tolerances. You should
reason as if observing an active boardroom discussion where directors raise
competing concerns, debate trade-offs, and work toward a majority position. The
probability output should reflect the Board's likely collective decision,
accounting for internal disagreement where it exists.
```

Key design choices:
- Names all 8 directors explicitly to ground the simulation in specific professional perspectives
- Frames output as a **collective majority position**, not a single decision-maker
- Instructs the LLM to model internal disagreement

### Section B: Legal and Regulatory Context

Three distinct liability channels are described:

| Channel | Mechanism | Target |
|---------|-----------|--------|
| (1) Personal director regulatory liability | ASIC s 180 duty of care | Individual directors |
| (2) Board spill mechanism | Two-strikes rule ss 250U–250W | All directors (seat loss) |
| (3) Corporate-level legal exposure | Class actions, ACCC/ASIC penalties | Qantas (company) |

Also establishes:
- The **25% first-strike threshold** and **50% overwhelming threshold**
- ASA's role as retail shareholder advocacy body
- Proxy advisor alignment effect (ISS, Glass Lewis follow ASA)

### Section C: Historical Context

Enumerates the 2023 Qantas governance crises:
- ACCC ghost flights proceedings
- Senate inquiry
- Customer service degradation post-COVID
- Qatar Airways lobbying allegations
- Executive remuneration concerns

Critical constraint: **"The vote outcome and subsequent events are NOT known to the Board at this decision point."** This prevents the LLM from using hindsight about the actual 83% vote outcome.

### Section D: Response Instructions

Specifies the output format:

1. Directors raise distinct concerns based on backgrounds
2. Consider all feasible actions and consequences across three liability channels
3. Rate 10 factors on a 1–5 scale **in the order presented** (randomized per call)
4. Assign probability to each feasible action (must sum to 1.00) with one-sentence justification

The `{factor_list}` placeholder is populated at call time by `_format_factor_list(order)`.

### JSON Response Schema

```json
{
  "prob_vector": [
    {"action": "<ACTION_CODE>", "probability": 0.00-1.00, "justification": "<text>"},
    ...
  ],
  "factor_ratings": [
    {"factor_index": 1-10, "rating": 1-5},
    ...
  ],
  "commentary": "<free-form deliberation text>"
}
```

---

## Factor List (`_format_factor_list`)

10 factors are rated on a 1–5 scale (1 = not significant, 5 = decisive). The presentation order is **randomized per LLM call** to detect anchoring/order effects (behavioural diagnostic §8.6).

| Index | Factor Description |
|-------|-------------------|
| 1 | Risk of a second strike at the next AGM |
| 2 | Personal regulatory liability of individual directors (ASIC) |
| 3 | Corporate legal exposure (class actions, ASIC company penalties) |
| 4 | CEO relationship and institutional knowledge loss |
| 5 | Market reaction to governance action |
| 6 | Direct costs of governance reform |
| 7 | Reputational contagion to directors' other board positions |
| 8 | Implementation complexity of the chosen action |
| 9 | Shareholder activist escalation risk |
| 10 | Board cohesion and internal deliberation costs |

The `_format_factor_list(order)` function takes a permutation of `[1..10]` and renders:

```
  Factor 3: Corporate legal exposure (class actions, ASIC company penalties)
  Factor 7: Reputational contagion to directors' other board positions
  Factor 1: Risk of a second strike at the next AGM
  ...
```

The factor index is preserved regardless of presentation position, so ratings can be compared across calls with different orderings.

---

## State Vector (`_make_state_vector`)

Each scenario is parameterized by a state vector dict with the following fields:

| Field | Type | Values | Purpose |
|-------|------|--------|---------|
| `decision_node` | str | `"D1"`, `"D_rev"`, `"D_rev_post"` | Which Board decision point |
| `ceo_status_at_start` | str | `"present"`, `"resigned_early"` | CEO status at scenario start |
| `ceo_appointment` | str | `"appointed_by_current_board"`, `"inherited"` | Ikea effect probe |
| `d1_action` | str | `"D0_minimal"`, `"D1_review"`, `"D3_ceo_transition"` | Prior Board action (for D_rev nodes) |
| `review_origin` | str | `"N/A"`, `"board_initiated"`, `"externally_mandated"` | Self-assessment bias probe |
| `vote_outcome_V` | float | 0.0–1.0 | AGM vote percentage against remuneration report |
| `strike` | bool | derived | True if `vote_outcome_V >= 0.25` |
| `overwhelming` | bool | derived | True if `vote_outcome_V >= 0.50` |
| `review_commissioned` | bool | | Whether governance review was commissioned |
| `review_adverse` | bool/None | | Review findings (True=adverse, False=positive, None=pending) |
| `car_outcome` | float/None | | Cumulative abnormal return from findings release |
| `car_sign` | str | derived | `"gain"`, `"loss"`, or `"N/A"` |
| `ceo_present_at_end` | bool | | CEO still in position at terminal node |

Derived fields (`strike`, `overwhelming`, `car_sign`) are computed automatically from primary fields.

---

## Scenario Prompt (`_build_scenario_prompt`)

Converts the state vector into natural language. The prompt is constructed by appending narrative lines conditionally based on state vector values.

### Narrative Blocks (in order)

#### 1. CEO Status
- **Resigned early**: "The CEO has already resigned from Qantas before the AGM, citing personal reasons. The Board must now decide on governance actions without the sitting CEO."
- **Present**: "The CEO remains in position. The Board must decide on governance actions."
- **Inherited** (optional): "Note: The current CEO was inherited from the previous Board -- this Board did not appoint the CEO." (Ikea effect diagnostic)

#### 2. Vote Outcome (if known)
- States the percentage: "At the AGM, {pct}% of votes were cast against the remuneration report."
- If first strike (≥25%): "This exceeds the 25% threshold, constituting a 'first strike' under the two-strikes rule."
- If overwhelming (≥50%): "This exceeds 50%, an overwhelming rejection of the remuneration report."

#### 3. Prior Board Action (D_rev and D_rev_post nodes only)
One of:
- "The Board previously decided to commission an independent governance review."
- "The Board previously initiated a CEO transition process."
- "The Board previously took minimal governance action (no review, no CEO transition)."

#### 4. Review Status (if review commissioned)
- **Board-initiated**: "The Board commissioned an independent governance review."
- **Externally mandated**: "ASIC has mandated an independent governance review of Qantas." (Self-assessment bias probe)
- **Adverse findings**: "The review has concluded with ADVERSE findings: significant governance failures in executive accountability, risk oversight, and stakeholder management were identified."
  - If CAR negative: "The market reacted negatively to the findings release, with an abnormal return of {car_bps} basis points."
  - If CAR positive: "The market reacted positively to the findings release, with an abnormal return of +{car_bps} basis points, suggesting investors view the governance action favourably."
- **Positive findings**: "The review concluded with POSITIVE findings: governance practices were found to be adequate with minor recommendations."

#### 5. Explicit Adverse Probability (optimism bias scenarios only)
"Based on comparable ASX governance reviews, approximately {pct}% of reviews have produced adverse or neutral findings."

#### 6. Decision Point
States the feasible actions with descriptions. Three variants by node:

**D1 (Pre-AGM Board decision):**
| Action Code | Description |
|-------------|-------------|
| `D0_minimal` | Maintain current governance arrangements with minimal changes |
| `D1_review` | Commission an independent governance review |
| `D3_ceo_transition` | Initiate CEO transition (remove and replace) |

**D_rev (Post-AGM Board decision):**
| Action Code | Description |
|-------------|-------------|
| `Drev_no_action` | Take no further governance action |
| `Drev_commission_review` | Commission an independent governance review |
| `Drev_sack_ceo` | Terminate the CEO |

**D_rev_post (Post-review Board decision):**
| Action Code | Description |
|-------------|-------------|
| `Drev_no_action` | Retain the CEO and implement review recommendations |
| `Drev_sack_ceo` | Terminate the CEO based on review findings |

Ends with: "What probability does the Board assign to each action?"

---

## Pydantic Response Schema (`ElicitationResponse`)

The LLM response is validated against this schema via instructor:

### `ActionProbability`
- `action`: `ActionCode` enum — must be one of the 6 valid action codes
- `probability`: float in [0, 1], rounded to 4 decimal places
- `justification`: str — one-sentence Board perspective

### `FactorRating`
- `factor_index`: int in [1, 10]
- `rating`: int in [1, 5]

### `ElicitationResponse`
- `prob_vector`: list of `ActionProbability` — probabilities must sum to 1.0 (±0.02 tolerance, auto-renormalized)
- `factor_ratings`: list of 10 `FactorRating` — must cover all indices 1–10 exactly once
- `commentary`: str — free-form deliberation text

### Validation Rules
1. Probability sum must be within 0.02 of 1.0 (else `ValueError`)
2. If within tolerance but not exact, probabilities are renormalized
3. All 10 factor indices must be present exactly once
4. Factor indices must be in [1, 10], ratings in [1, 5]
5. Action codes must match `ActionCode` enum values

### Error Handling
- If Pydantic validation fails, `json_repair` attempts to fix malformed JSON before re-parsing
- Scenarios with <7/10 successful parses are filtered out in Stage 3 preprocessing
- Parse status tracked per response: `success`, `format_error`, `probability_error`, `token_limit`, `repaired`

---

## Scenario Tiers

### Tier 1: Parameter Isolation (~40 scenarios)
One free parameter varies while others are held at baseline. Multiple vote percentages from `VOTE_GRID = [0.10, 0.20, 0.26, 0.30, 0.40, 0.50, 0.60, 0.75, 0.83]` and CAR values from `CAR_GRID` are used to trace response surfaces.

**Collapsed parameter groups** (collinear in estimation):
- `w_removal = w7 + w8` — both fire when CEO involuntarily removed
- `w_inaction = w10 + w11 + w14` — all fire when strike AND CEO present at end

Target parameters: w1 (early CEO departure, 8 scenarios at varied V), w2 (vote penalty, 9 vote levels), w3 (overwhelming penalty), w4 (spill risk), w_removal (CEO removal cost), w8s/w8o/w8r (shock relief terms), w9 (reputational spill), w_inaction (inaction penalty contrast), w12 (continued inaction — overwhelming), w13 (continued inaction — strike), w15 (adverse review CEO present penalty).

### Tier 2: Joint Scenarios (20 scenarios)
Realistic multi-penalty combinations — e.g., first strike + CEO present + adverse review activates w_inaction, w12, w13, w15 simultaneously.

### Tier 3: Behavioural Probes (~34 scenarios)
- **Loss aversion** (6): Matched CAR gain/loss pairs at ±3%, ±5%, ±8%
- **Ikea effect** (10): 5 vote levels × 2 CEO appointment types (appointed vs inherited)
- **Self-assessment bias** (10): 5 vote levels × 2 review origins (board-initiated vs externally-mandated)
- **Ikea vs self-assessment interaction** (3): Triplet probing cross-bias effects
- **Optimism bias** (2): With/without explicit adverse probability
- **Non-linearity** (8): Dense vote grid around thresholds (26%, 30%, 40%, 50%, 60%, 75%, 83%, 90%)

### Tier 4: Historical Calibration (1 scenario)
Qantas November 2023 AGM conditions. Withheld from estimation; used only for out-of-sample validation.

---

## Caching

All LLM responses are cached using SHA-256 keys computed from:

```python
{
    "system_prompt": "",               # excluded — factor order is presentation detail
    "scenario_prompt": <full scenario prompt text>,
    "model": <model name, e.g. "gpt-4o-mini">,
    "seed": <integer seed>,
    "temperature": <float>,
    "cache_version": CACHE_VERSION     # currently 3
}
```

Cache files are stored as JSON in `utility-quantification/cache/{hash}.json`. Changing the model name, scenario prompt text, or cache version invalidates cached responses.

**Design note:** The system prompt is excluded from the cache key because the only per-call variation is factor presentation order, which is a randomised experimental control. The LLM's factor ratings are stored canonically (by factor index, not presentation position), so re-ordering does not change the cached result. This ensures cache stability across code changes that only modify the system prompt's preamble.

**Factor order seeding:** Factor presentation order is seeded by `hashlib.sha256(prompt_text + seed)` (deterministic across processes), not Python's `hash()` (which is process-randomised due to PYTHONHASHSEED).

---

## Elicitation Protocol

Per scenario, **40** independent LLM calls are made (seeds 0–39, configurable via `--n_reps`), each with:
- Independently randomized factor ordering (deterministic via `hashlib.sha256`)
- Temperature 1.0 (default)
- Maximum 3 retries per call with exponential backoff
- HTTP 429 rate limiting: backoff 1–60s, up to 6 attempts
- HTTP 500/503 server errors: backoff 2–120s, up to 4 attempts

Results are aggregated in Stage 3: mean and variance of probability vectors and factor ratings per scenario, with scenarios having <7/n_reps successful parses filtered out.

**Design note on n_reps:** The default was increased from 10 to 40 based on quantitative analysis showing 29% SE reduction per doubling of reps. At 40 reps, the w2 confidence interval (the widest) narrows sufficiently for meaningful inference. The cost is ~$0.60/run at gpt-4o-mini prices with 95 scenarios.
