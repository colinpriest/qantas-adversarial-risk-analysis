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

#### 1. CEO Status and Crisis Framing

- **Resigned early**: "The CEO has already resigned from Qantas before the AGM, citing personal reasons. The Board must now decide on governance actions without the sitting CEO."
- **Present**: Triggers the full **cognitive bias framing** (see [Cognitive Bias Context Logic](#cognitive-bias-context-logic) below). The prompt includes three layered bias mechanisms before the scenario-specific state.
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

## Cognitive Bias Context Logic

The scenario prompt exploits three well-documented cognitive biases to elicit realistic Board action probabilities. When the CEO is present, the prompt constructs layered psychological pressure that mirrors the actual information environment facing corporate directors during a severe governance crisis.

### Design Rationale

Empirical observation: without bias framing, the LLM assigns ~25-35% probability to CEO transition at D1. With all three biases active, probabilities rise to 90%+. This matches the empirical base rate — 100% of comparable ASX 100 ESG crises resulted in CEO departure.

The biases are not "tricks" — they represent genuine information that real Board directors would have when deliberating. The prompt makes this information explicit rather than relying on the LLM's implicit training distribution.

### Bias 1: Anchoring (Peer Base Rate)

**Trigger:** `ceo_status_at_start != "resigned_early"` (CEO is present)

**Mechanism:** States the base rate of CEO departure in comparable crises as an absolute figure ("100% of comparable cases"). This anchors the LLM's probability estimate near the stated base rate.

**Prompt text (excerpt):**
> PEER BENCHMARK: Of all ASX 100 companies that experienced a severe ESG crisis over the past decade [...] every single one resulted in CEO departure. In 100% of comparable cases, the CEO did not survive the crisis.

**Source justification:** AMP (fee-for-no-service, CEO resigned), Crown Resorts (money laundering, CEO resigned), Rio Tinto (Juukan Gorge, CEO terminated), Westpac (AUSTRAC breaches, CEO resigned), NAB (Royal Commission, CEO resigned). All five are verifiable ASX 100 CEO departures following severe ESG crises in the 2018-2023 period.

### Bias 2: Bandwagon Effect (Named Peer Companies)

**Trigger:** Same as Bias 1 (CEO present)

**Mechanism:** Names five specific peer companies by name with their crisis type. This creates social proof — "everyone else did it" — leveraging the bandwagon/conformity effect documented in corporate governance literature (Westphal & Zajac, 1997).

**Prompt text (excerpt):**
> [...] including AMP (fee-for-no-service), Crown Resorts (money laundering failures), Rio Tinto (Juukan Gorge), Westpac (AUSTRAC anti-money laundering breaches), and NAB (Royal Commission misconduct) [...]

**Design choice:** Companies are listed by name (not anonymised) because named exemplars produce stronger anchoring than abstract statistics (Kahneman & Tversky, 1974). Each company's crisis type is parenthetically specified to establish comparability with the Qantas ESG crisis.

### Bias 3: Loss Aversion (Regulatory/Legal Consequences of Inaction)

**Trigger:** Same as Bias 1 (CEO present)

**Mechanism:** Frames inaction as an active choice with negative personal consequences for directors. This exploits loss aversion — the tendency to weigh potential losses more heavily than equivalent gains (Kahneman & Tversky, 1979). Directors are told that doing nothing increases their personal legal exposure.

**Prompt text (excerpt):**
> REGULATORY RISK: Directors who fail to act decisively in a severe ESG crisis face heightened personal exposure. ASIC has increased enforcement of directors' duties (s180-s184 Corporations Act), and courts have held that failure to address known governance failures can constitute a breach of the duty of care and diligence. Boards that retained underperforming CEOs through comparable crises (e.g. AMP, Crown) subsequently faced class action lawsuits, APRA enforcement actions, and proxy adviser campaigns targeting individual directors for removal. Inaction is not a neutral choice — it signals acceptance of the status quo to regulators, institutional investors, and the market, and materially increases the Board's legal liability.

**Legal basis:** ASIC's enforcement of s180 (duty of care and diligence) has intensified post-Royal Commission. The Federal Court in *ASIC v Healey* [2011] established that directors cannot rely on management representations when warning signs are present. Class action risk is real — AMP directors faced personal liability claims following their fee-for-no-service scandal.

### Bias 4: Counter-Bias Deactivation (Naming Retention Biases)

**Trigger:** Same as Bias 1 (CEO present)

**Mechanism:** Explicitly names the cognitive biases that favour CEO retention, so the LLM recognizes and discounts them rather than unconsciously reproducing them. This is a meta-cognitive intervention — the prompt tells the simulated Board "you are likely experiencing these biases; here is what they look like."

**Biases named and deactivated:**

| Bias | Deactivation Language | Psychological Mechanism |
|------|----------------------|------------------------|
| Escalation of commitment | "This is not about whether appointing the CEO was right; the question is which leadership structure best protects Qantas from 2023 onward." | Separates past decision from forward-looking optimisation, allowing directors to change course without admitting error |
| Status quo bias | "Boards systematically overweight transition risks [...] and underweight the compounding costs of retaining a CEO whose credibility has been severely damaged." | Makes the asymmetric risk weighting explicit so the LLM corrects for it |
| Groupthink | "Directors should ask: if we were appointing a CEO today with these facts on the table, would we appoint this person as the crisis-recovery CEO?" | Forces a fresh-appointment counterfactual that breaks the "we back the CEO" group norm |
| Over-optimism | "In every comparable ASX 100 case, this optimism proved unfounded and delayed inevitable action." | Uses peer evidence to show that optimism about recovery under current CEO is empirically unjustified |
| Hyperbolic discounting | "Prioritising short-term stability [...] over the longer-term reputational, regulatory, and class-action costs that compound with each month of inaction." | Names the temporal discounting directly and frames future costs as present-value compounding |

**Design rationale:** Research on cognitive debiasing (Fischhoff, 1982; Larrick, 2004) shows that naming a bias reduces its influence on judgement. The prompt does not argue the biases are wrong — it simply makes them visible, shifting the LLM's simulated deliberation from System 1 (automatic pattern-matching favouring status quo) to System 2 (deliberate evaluation of forward-looking costs and benefits).

**Fresh Appointment Test (in CEO Retention Risk Assessment):**

At 1+ shocks, the prompt additionally poses a concrete counterfactual:

> "If this CEO were not already in position and the Board were appointing a crisis-recovery CEO today, would a candidate with this CEO's track record on the current ESG failures be selected? If the answer is no, then retention is being driven by sunk-cost reasoning, not forward-looking value maximisation."

This technique is drawn from decision architecture literature (Kahneman, Lovallo & Sibony, 2011) — requiring directors to evaluate the current CEO as if they were a new candidate forces a reference-class comparison that bypasses sunk-cost and commitment biases.

**Forward-Looking Frame (in CEO Retention Risk Assessment):**

At exactly 1 shock, the prompt reframes the decision:

> "The relevant question is not whether the Board's past support for the CEO was justified. The question is: given the ACCC action, Senate inquiry, customer trust collapse, and public mood today, which leadership structure best protects Qantas over the next five years?"

This allows directors to preserve their self-regard for past decisions while acknowledging changed circumstances — reducing choice-supportive bias by decoupling the dismissal decision from implicit self-criticism.

**Urgency Framing (in Prior Inaction Consequences):**

When the Board previously took minimal action and shocks have occurred at D_rev/D_rev_post, the prompt adds immediacy to counter hyperbolic discounting:

> "These risks are not distant hypotheticals — the ACCC proceedings are active, the Senate inquiry is ongoing, enterprise bargaining is approaching, and government contract decisions are imminent. Imagine the headlines and analyst calls if the Board announces a leadership reset now versus in six months after further regulatory and reputational damage."

This uses the availability heuristic (vivid "day-after" counterfactuals) and ties costs to specific imminent events rather than abstract future probabilities.

### Bias Interaction Model

The action-promoting and counter-bias mechanisms work as **two complementary layers**:

```
P(CEO_transition) = f(action_biases + counter_bias_deactivation + scenario_state)

  LAYER 1: ACTION-PROMOTING               LAYER 2: RETENTION-BIAS DEACTIVATION
  ┌───────────────────────────┐            ┌─────────────────────────────────────┐
  │ ANCHORING   100% base rate│            │ Name escalation of commitment      │
  │ BANDWAGON   5 named peers │            │ Name status quo bias               │
  │ LOSS AVERSION legal risk  │            │ Name groupthink → fresh appt test  │
  │ URGENCY     imminent events│           │ Name over-optimism → peer evidence │
  └────────────┬──────────────┘            │ Name hyperbolic discounting        │
               │                           │ Forward-looking reframe            │
               │                           └──────────┬──────────────────────────┘
               └──────────┬───────────────────────────┘
                          ▼
                ┌───────────────────────┐
                │  CEO present at D1:   │
                │  P(transition) ≈ 95%+ │
                └───────────────────────┘
```

Without any biases (CEO resigned branch), the prompt simply states "The CEO has already resigned" — no anchoring, no peer comparison, no regulatory threat. This creates a natural control group.

### Escalating Context Blocks (D_rev and D_rev_post nodes)

Beyond the D1-level biases, the prompt includes two additional context-dependent bias amplifiers that activate at later decision nodes:

#### Prior Inaction Consequences (D_rev/D_rev_post after D0_minimal)

When the Board previously took minimal action (`d1_action == "D0_minimal"`) and governance shocks have occurred, the prompt adds a **compounding consequences** block:

| Shock | Consequence Language |
|-------|---------------------|
| First strike | "Continued inaction now makes a second strike near-certain, which would trigger a full board spill — all directors would lose their seats." |
| First strike | "ASIC is likely to assess director culpability under s 180 based on the cumulative pattern of Board inaction." |
| First strike | "Shareholder class action exposure increases significantly because plaintiffs can now demonstrate a pattern of Board inaction across multiple decision points." |
| Overwhelming (50%+) | "Continued Board inaction at this stage is inconsistent with peer ASX100 governance responses [...] Proxy advisors will flag the Board's persistent inaction." |

These consequences are **only shown when the Board previously chose D0_minimal** — if the Board already commissioned a review or transitioned the CEO, this block is omitted. This ensures the bias pressure is proportional to the degree of prior inaction.

#### CEO Retention Risk Assessment (all nodes where CEO is present)

A tiered risk assessment block adapts to the cumulative shock state:

| Shocks | Severity | Framing |
|--------|----------|---------|
| 2+ shocks | Highest | "places this situation in the highest-severity category [...] boards that retained the CEO after multiple governance shocks subsequently lost director seats, faced personal regulatory proceedings" |
| 1 shock | Significant | "retention after this shock carries escalating risks: potential second strike, regulatory scrutiny of Board inaction, and shareholder class action exposure" |
| 0 shocks | Baseline | "CEO removal at this stage would carry full transition costs [...] without the governance-failure justification" (i.e., **argues against** removal when no shocks have occurred) |

The zero-shock case is critical: it provides a **counter-bias** that reduces the probability of CEO transition when the governance situation does not warrant it, preventing the anchoring/bandwagon biases from producing unrealistically high transition probabilities at low vote levels.

### Summary: Bias Architecture by Decision Node

| Node | Action Biases | Counter-Bias Deactivation | Expected Effect |
|------|---------------|--------------------------|-----------------|
| D1, CEO present, no shocks | Anchoring + Bandwagon + Loss Aversion | All 5 named biases + 0-shock counter-balance | P(transition) moderate (~70-85%) |
| D1, CEO present, after strike | Anchoring + Bandwagon + Loss Aversion | All 5 named biases + forward-looking frame | P(transition) high (~90-97%) |
| D1, CEO present, after overwhelming | Anchoring + Bandwagon + Loss Aversion | All 5 named biases + forward-looking frame | P(transition) very high (~95-99%) |
| D_rev, after D0_minimal + strike | All D1 biases + Prior Inaction + Urgency | All 5 named biases + fresh appointment test | P(sack) very high |
| D_rev, after D0_minimal + overwhelming | All D1 biases + Prior Inaction + Urgency | All 5 named biases + fresh appointment test | P(sack) near-certain |
| D_rev_post, adverse review + CEO present | All D1 biases + highest-severity | All 5 named biases + fresh appointment test | P(sack) near-certain |
| Any node, CEO resigned | None (clean control) | None | Probabilities reflect pure governance considerations |

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
3. Factor ratings are deduplicated by index (first occurrence kept) — the LLM sometimes returns 11-12 ratings with duplicate indices when prompts are long (cognitive bias framing increases token count). After deduplication, all 10 indices 1-10 must be present.
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

Target parameters: w1 (early CEO departure, 8 scenarios at varied V), w2 (vote penalty, 9 vote levels), w3 (overwhelming penalty), w4 (spill risk), w_removal (CEO removal cost), w8s/w_remove_ceo_overwhelming/w8r (shock relief terms), w9 (reputational spill), w_inaction (inaction penalty contrast), w12 (continued inaction — overwhelming), w13 (continued inaction — strike), w15 (adverse review CEO present penalty).

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
