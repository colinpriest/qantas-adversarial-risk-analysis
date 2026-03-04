# Board Utility Function — Detailed Technical Reference

This document provides a comprehensive specification of the Board's utility function in the Qantas Adversarial Risk Analysis (ARA) engine. It covers every Board decision point, the mathematical formulae underlying utility calculation, justification for the structure and parameter values, and the data sources from which parameters are derived.

***

## Table of Contents

1.  [Overview](#1-overview)
2.  [Board Decision Points](#2-board-decision-points)
3.  [Board Utility Function Formula](#3-board-utility-function-formula)
4.  [Parameter Reference Table](#4-parameter-reference-table)
5.  [Detailed Parameter Specifications](#5-detailed-parameter-specifications)
6.  [Board Overconfidence Bias](#6-board-overconfidence-bias)
7.  [Chance Node Models Relevant to Board Utility](#7-chance-node-models-relevant-to-board-utility)
8.  [How Board Utility Is Computed in the Engine](#8-how-board-utility-is-computed-in-the-engine)
9.  [Opponent Priors on Board Parameters](#9-opponent-priors-on-board-parameters)
10. [Summary of Data Sources](#10-summary-of-data-sources)

***

## 1. Overview

The Board's objective in the ARA engine is to **minimise opposition and disruption** while managing the implementation costs of governance reform. The Board faces a crisis scenario (Qantas AGM, November 2023) in which shareholder dissatisfaction, ASA mobilisation, and CEO conduct failures create strategic pressure.

The Board utility function is an additive, multi-component penalty function. It begins at zero and accumulates negative terms (costs, penalties) based on game outcomes, with one positive channel (review CAR when a governance review produces a favourable market reaction). The function is evaluated at the Terminal node after all decisions and chance outcomes have been resolved.

**Source files:**

-   Implementation: `engine/utilities.py` — `utility_board()`
-   Parameters: `data/governance_spec.xlsx`, sheet `utilities_board`
-   Opponent priors: `data/opponent_priors.xlsx`, sheet `priors`
-   Overconfidence: `data/governance_spec.xlsx`, sheet `board_overconfidence`
-   Algebraic specification: `docs/algebraic.md`, Section 5.2

***

## 2. Board Decision Points

The Board owns three decision nodes in the game tree (plus conditional post-review variants). At each, the Board selects from a set of feasible actions, and the engine evaluates the expected utility of each action via recursive tree evaluation.

### 2.1 D1 — Initial Governance Reform Package

| Property             | Value                      |
|----------------------|----------------------------|
| **Position in tree** | Second node (after D0_ceo) |
| **Owner**            | Board                      |
| **Node type**        | Decision                   |
| **When reached**     | Always (unconditional)     |

**Decision:** What governance reform package should the Board announce before the AGM?

**Actions:**

| Action                        | Code                | Feasibility   | Description                                                                                                                                                                                                         |
|-------------------------------|---------------------|---------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Minimal action (status quo)   | `D0_minimal`        | Always        | Take no proactive governance action. Maintain current management and governance structures.                                                                                                                         |
| Commission independent review | `D1_review`         | Always        | Announce an independent governance review with a published timeline and terms of reference. Signals accountability without immediate personnel changes.                                                             |
| CEO transition                | `D3_ceo_transition` | `CEO_present` | Announce the CEO's removal or managed departure. The most aggressive governance action, directly addressing shareholder concerns about executive accountability. Only feasible if the CEO has not already resigned. |

**Strategic context:** D1 is the Board's first-mover action. It occurs before the ASA decides its strike recommendation (A2), before the shareholder vote (V), and before any post-AGM events. The Board's choice at D1 influences the vote outcome through the governance effect channel in the vote model: announcing a review reduces the expected protest vote (positive governance effect), while CEO transition has an ambiguous effect (shareholders may view it as decisive action or as confirmation of severe problems).

### 2.2 D_rev — Post-AGM Board Response

| Property             | Value                                      |
|----------------------|--------------------------------------------|
| **Position in tree** | Seventh node (after D4, before R)          |
| **Owner**            | Board                                      |
| **Node type**        | Decision                                   |
| **When reached**     | Always (but actions are state-conditional) |

**Decision:** What action should the Board take after observing the AGM vote outcome and the CEO's initial response?

**Actions:**

| Action            | Code                     | Feasibility               | Description                                                                                                      |
|-------------------|--------------------------|---------------------------|------------------------------------------------------------------------------------------------------------------|
| No action         | `Drev_no_action`         | Always                    | Board takes no further action post-AGM.                                                                          |
| Commission review | `Drev_commission_review` | `review_not_commissioned` | Commission an independent governance review (if not already commissioned at D1).                                 |
| Sack CEO          | `Drev_sack_ceo`          | `CEO_present`             | Board removes the CEO. Only feasible if CEO is still in position (has not resigned at D4 or been removed at D1). |

**Strategic context:** D_rev occurs after the vote (V) and after the CEO has had the opportunity to respond (D4). The Board observes the vote percentage — including whether a first strike (≥25%) or overwhelming vote (≥50%) occurred — and the CEO's choice (stay, resign, or negotiate). This information shapes the Board's response: a first strike with the CEO still present creates acute pressure to act, while CEO self-removal reduces the urgency.

### 2.3 D_rev_post_review — Post-Review Board Response (Conditional)

| Property             | Value                                                                          |
|----------------------|--------------------------------------------------------------------------------|
| **Position in tree** | Eleventh node (after D4_post_review, before Terminal)                          |
| **Owner**            | Board                                                                          |
| **Node type**        | Decision                                                                       |
| **When reached**     | Only if `post_review_round = true` (review adverse AND CEO still present at R) |

**Decision:** What action should the Board take after adverse review findings when the CEO is still present?

**Actions:**

| Action            | Code                     | Feasibility                                     | Description                                                                      |
|-------------------|--------------------------|-------------------------------------------------|----------------------------------------------------------------------------------|
| No action         | `Drev_no_action`         | `post_review_round`                             | Board takes no further action after adverse review.                              |
| Sack CEO          | `Drev_sack_ceo`          | `post_review_round_and_CEO_present`             | Board removes CEO after adverse review findings.                                 |
| Commission review | `Drev_commission_review` | `post_review_round_and_review_not_commissioned` | Commission another review (only if first review was triggered at D_rev, not D1). |

**Strategic context:** This node is reached only on adverse review paths where the CEO has survived to this point. The adverse_review_ceo_present_penalty (§5.15) creates a strong incentive for the Board to remove the CEO at this node, since retaining the CEO after documented governance failures exposes the Board to regulatory liability, shareholder revolt at the next AGM, and class-action risk.

***

## 3. Board Utility Function Formula

The Board utility is computed at the Terminal node over the complete game outcome. Formally:

```
u_B(Z) = 0
         − w_early   · 1[CEO_resigned_early]                                     (1)
         − w_vote    · (V − 0.25)²₊                                              (2)
         − w_over    · 1[overwhelming]                                            (3)
         − w_spill   · V · 1[strike]                                              (4)
         + w_CAR     · CAR · 1[review_commissioned]                               (5)
         − w_cost    · C_direct · 1[review_commissioned]                          (6)
         − w_impl    · (1[d1 = D3] + 1[d_rev = sack] + 1[d_rev' = sack])        (7)
         − w_loss    · 1[CEO_removed ∧ ¬CEO_resigned_early]                      (8)
         − w_rep     · 1[overwhelming]                                            (9)
         − w_spill2  · 1[strike ∧ CEO_present_at_end]                            (10)
         − w_reg     · 1[strike ∧ CEO_present_at_end]                            (11)
         − w_d1_liab · 1[overwhelming ∧ d1 = D0_minimal]                         (12)
         − w_legal1  · 1[strike ∧ d1 = D0_minimal]                              (13)
         − w_legal2  · 1[strike ∧ CEO_present_at_end]                            (14)
         − w_adverse · 1[review_commissioned ∧ review_adverse ∧ CEO_present_at_end] (15)
```

where:

-   `(x)₊ = max(x, 0)` — the positive part function
-   `CEO_present_at_end = ¬CEO_removed ∧ ¬CEO_resigned_early`
-   `V` is the continuous vote fraction (e.g., 0.83 for 83%)
-   `CAR` is the cumulative abnormal return from the review findings release (can be positive or negative)
-   `C_direct` is the direct cost of the review in decimal CAR (positive value, subtracted)
-   All indicator functions `1[·]` are {0, 1}-valued

### 3.1 Structural Rationale

The utility function is designed as a **penalty-based objective** rather than a reward-maximising function. This reflects the Board's institutional position: in a crisis, the Board's goal is damage minimisation rather than opportunity exploitation. The additive structure captures the key insight that multiple sources of damage accumulate independently — a high protest vote, an adverse review, and CEO retention each impose separate costs that do not substitute for one another.

The function is **linear in most indicators** (with the important exception of the quadratic vote penalty, term 2). This reflects the assumption that the Board evaluates outcomes categorically: either a strike occurred or it did not, either the CEO was removed or not. The quadratic vote penalty provides the critical exception — the Board cares not just about whether a strike occurs but about *how severe* the vote is beyond the 25% threshold. This convexity captures the escalating reputational damage from increasingly adverse vote outcomes.

***

## 4. Parameter Reference Table

All parameter values are stored in `data/governance_spec.xlsx`, sheet `utilities_board`. No magic numbers appear in code.

| \# | Parameter Name                       | Symbol    | Value | Term | Data Basis                        |
|----|--------------------------------------|-----------|-------|------|-----------------------------------|
| 1  | `early_ceo_departure_cost`           | w_early   | 0.5   | (1)  | Subjective                        |
| 2  | `vote_penalty_weight`                | w_vote    | 2.0   | (2)  | Subjective (calibrated)           |
| 3  | `overwhelming_penalty_weight`        | w_over    | 3.0   | (3)  | Subjective                        |
| 4  | `spill_risk_weight`                  | w_spill   | 2.5   | (4)  | Subjective (empirically anchored) |
| 5  | `review_car_weight`                  | w_CAR     | 15.0  | (5)  | Data-derived                      |
| 6  | `review_direct_cost_weight`          | w_cost    | 15.0  | (6)  | Data-derived                      |
| 7  | `implementation_cost_sack`           | w_impl    | 0.3   | (7)  | Subjective                        |
| 8  | `ceo_loss_cost`                      | w_loss    | 1.5   | (8)  | Subjective                        |
| 9  | `reputational_spill_weight`          | w_rep     | 1.0   | (9)  | Subjective                        |
| 10 | `second_strike_spill_penalty`        | w_spill2  | 8.0   | (10) | Empirically anchored              |
| 11 | `board_regulatory_liability`         | w_reg     | 5.0   | (11) | Subjective (legally grounded)     |
| 12 | `board_d1_liability`                 | w_d1_liab | 4.0   | (12) | Subjective (legally grounded)     |
| 13 | `qantas_legal_d1_penalty`            | w_legal1  | 3.0   | (13) | Subjective (legally grounded)     |
| 14 | `qantas_legal_d_rev_penalty`         | w_legal2  | 2.0   | (14) | Subjective (legally grounded)     |
| 15 | `adverse_review_ceo_present_penalty` | w_adverse | 5.0   | (15) | Data-informed                     |

**Classification key:**

-   **Data-derived**: Parameter value is directly computed from observable data or a calibrated statistical model.
-   **Empirically anchored**: Parameter value is informed by observable data but requires subjective scaling judgement.
-   **Subjective (legally grounded)**: Parameter reflects legal or regulatory consequences documented in statute or case law, but the *magnitude* is a modelling judgement.
-   **Subjective (calibrated)**: Parameter value is chosen to produce behaviourally plausible results and tested through sensitivity analysis.
-   **Subjective**: Parameter value is a modelling judgement without direct empirical calibration.

***

## 5. Detailed Parameter Specifications

### 5.1 Early CEO Departure Cost — `early_ceo_departure_cost` = 0.5

**Term (1):** `u -= 0.5 · 1[CEO_resigned_early]`

**What it captures:** The cost to the Board when the CEO resigns before the game tree begins (the D0_ceo = CEO_resign scenario, corresponding to the historical event of 5 September 2023). Even though early resignation removes the CEO controversy from the AGM, it imposes transition costs: leadership vacuum during crisis, market uncertainty about succession, disruption to ongoing ACCC settlement negotiations.

**Justification for value:** Set at 0.5 — significantly lower than the full `ceo_loss_cost` (1.5) because voluntary pre-crisis departure is far less disruptive than mid-crisis removal. The CEO controls the timing and narrative ("taking responsibility"), reducing market shock and allowing an orderly succession process. The Qantas Board had a succession candidate (Vanessa Hudson) identified, making the transition smoother than a contested removal.

**Basis:** Subjective. The 0.5 value reflects the judgement that early voluntary departure imposes approximately one-third the disruption cost of involuntary removal. This ratio is consistent with CEO turnover literature showing that forced departures produce 2–4x the stock price impact of voluntary departures for large-cap firms.

**Implication:** A low early departure cost makes the "ceo_resigned" scenario relatively benign for the Board. The Board's expected utility is considerably higher when the CEO departs voluntarily before the AGM than when the CEO stays and the Board must navigate the full governance crisis.

***

### 5.2 Vote Penalty Weight — `vote_penalty_weight` = 2.0

**Term (2):** `u -= 2.0 · (V − 0.25)²`

**What it captures:** The **quadratic** escalation of reputational damage as the protest vote exceeds the first-strike threshold of 25%. The quadratic functional form means that incremental votes beyond 25% impose increasing marginal cost — a 35% vote is worse than 30% by more than 30% is worse than 25%.

**Justification for structure:** A first strike on the remuneration report is a public signal of shareholder dissatisfaction that triggers media coverage, proxy advisor scrutiny, and institutional investor engagement. The severity of these consequences scales nonlinearly: a bare first strike (26%) attracts modest attention, while a vote near 50% triggers a qualitatively different regulatory and reputational regime. The quadratic form `(V − 0.25)²` captures this escalation while remaining analytically tractable.

**Justification for value:** At 2.0, the quadratic penalty produces the following scale of costs:

| Vote % | Excess above 25% | Quadratic penalty | Total contribution |
|--------|------------------|-------------------|--------------------|
| 30%    | 0.05             | 0.0025            | −0.005             |
| 40%    | 0.15             | 0.0225            | −0.045             |
| 50%    | 0.25             | 0.0625            | −0.125             |
| 60%    | 0.35             | 0.1225            | −0.245             |
| 83%    | 0.58             | 0.3364            | −0.673             |

The quadratic term alone contributes moderately; its primary function is to differentiate between outcomes within the "first strike achieved" region. The bulk of the vote-related disutility comes from the `spill_risk_weight` (linear in V) and the indicator-based penalties (terms 10–14).

**Basis:** Subjective (calibrated). The weight was set to ensure that the total vote-related disutility (sum of terms 2, 3, 4, 9, 10–14) produces a meaningful ordering of outcomes: overwhelming \> standard strike \> near-miss. Sensitivity analysis confirms that results are robust to perturbations of ±50% in this weight.

***

### 5.3 Overwhelming Penalty Weight — `overwhelming_penalty_weight` = 3.0

**Term (3):** `u -= 3.0 · 1[overwhelming]`

**What it captures:** A discrete penalty for an overwhelming protest vote (≥50% against the remuneration report). An overwhelming vote is qualitatively distinct from a standard first strike: it signals majority shareholder dissatisfaction, attracts national media coverage, and places the Board under immediate pressure to act.

**Justification for value:** The threshold for "overwhelming" is set at 50% (from `vote_thresholds` sheet). The 3.0 penalty reflects the Board's assessment that crossing the 50% line imposes:

-   Heightened media scrutiny and sustained negative press coverage
-   Automatic engagement demands from institutional investors (proxy advisors flag the company for "urgent governance action")
-   Increased probability of director not being re-elected at the next AGM

The penalty is separate from and additive with the `reputational_spill_weight` (term 9), which captures the reputational contagion effect. Together, an overwhelming vote contributes at least 4.0 in disutility (3.0 + 1.0) before any other consequences.

**Basis:** Subjective. The 3.0 value reflects the modelling judgement that crossing the 50% threshold is approximately 3x more salient to the Board than a generic unit of governance disruption.

***

### 5.4 Spill Risk Weight — `spill_risk_weight` = 2.5

**Term (4):** `u -= 2.5 · V · 1[strike]`

**What it captures:** A **linear** penalty in the vote percentage, conditional on a first strike occurring. This captures the escalating risk of a board spill (under the Corporations Act 2001, ss 250U–250W "two-strikes rule") and the general severity of shareholder opposition.

**Justification for structure:** The linear form `V · 1[strike]` means that the penalty scales directly with the vote outcome once the 25% threshold is crossed. A vote of 83% (as actually occurred for Qantas) imposes `2.5 × 0.83 = 2.075` in disutility, while a bare first strike at 26% imposes only `2.5 × 0.26 = 0.65`. This captures the empirical reality that a higher protest vote signals greater probability of:

-   A second strike at the next AGM
-   A successful spill resolution (requires \>50% at the *next* AGM after a second strike)
-   Loss of institutional investor confidence

**Justification for value:** The 2.5 coefficient was calibrated against the empirical data on second-strike outcomes from 2020–2024 (`background/ceo/2nd-strike.md`):

-   No ASX 200 company that received a first strike in 2023 received a second strike in 2024
-   Between 2011 and 2024, no incumbent director has lost their board seat at a spill meeting
-   Spill resolutions average under 7% support (2024) and 4.5% support (2025)

Despite the historically low base rate of successful spills, the coefficient is set at 2.5 (rather than near zero) because the *prospect* of a spill — even if unlikely to succeed — creates acute governance pressure: the Board must prepare for a spill meeting, divert attention from operations, and manage institutional investor relations during the intervening year.

**Basis:** Subjective (empirically anchored). The value reflects a balance between the very low base rate of actual spill outcomes and the substantial reputational and operational cost of the spill *process*.

***

### 5.5 Review CAR Weight — `review_car_weight` = 15.0

**Term (5):** `u += 15.0 · CAR · 1[review_commissioned]`

**What it captures:** The market-valuation impact of the governance review findings, measured as cumulative abnormal return (CAR) during the findings release window. A positive CAR (market relief) increases Board utility; a negative CAR (adverse market reaction) decreases it.

**Justification for structure:** The review CAR enters Board utility **linearly** because the Board experiences market reactions directly through their fiduciary responsibility to shareholders and through their personal shareholdings, options, and reputation. The `review_car_weight` translates the decimal CAR (e.g., −0.05 for a 5% drop) into utility-scale units.

**Justification for value:** The weight of 15.0 was calibrated from the ASX governance review case study data (`background/board/governance-review-case-studies.md`), which documents findings-release CARs for six companies:

| Company            | Period  | Findings Window CAR |
|--------------------|---------|---------------------|
| CBA                | 2017–18 | +1.75%              |
| Westpac            | 2019–20 | −3.00%              |
| Rio Tinto          | 2020    | −2.65%              |
| Star Entertainment | 2021–22 | −13.95%             |
| BOQ                | 2022–23 | −5.70%              |
| Qantas             | 2023–24 | +0.85%              |

These CARs range from −13.95% to +1.75%, with a distribution centred around −5% (the location parameter of the review model). At w_CAR = 15.0:

| CAR outcome      | Utility contribution |
|------------------|----------------------|
| Star (−13.95%)   | −2.09                |
| BOQ (−5.70%)     | −0.86                |
| Westpac (−3.00%) | −0.45                |
| CBA (+1.75%)     | +0.26                |
| Qantas (+0.85%)  | +0.13                |

These magnitudes are commensurate with the other utility components (e.g., `ceo_loss_cost` = 1.5, `second_strike_spill_penalty` = 8.0), ensuring that the review outcome is a material consideration in Board decision-making but does not dominate all other factors.

**Basis:** Data-derived. The weight is calibrated so that the empirical range of observed CARs (±14%) maps to a utility range of approximately ±2.1, which places review risk in the middle tier of Board concerns (above implementation costs, below second-strike risk).

**Note on the t-distribution:** The review CAR is drawn from a Student-t hierarchy (see §7.2), which produces heavy tails. At ν=3 degrees of freedom, extreme draws (e.g., CAR = −20%) are not negligible. The high w_CAR = 15.0 means these tail events have a substantial impact on expected utility, which is the intended design: the possibility of a catastrophic review outcome (like Star's −13.95%) should materially deter the Board from commissioning a review unless the expected benefit justifies the tail risk.

***

### 5.6 Review Direct Cost Weight — `review_direct_cost_weight` = 15.0

**Term (6):** `u -= 15.0 · C_direct · 1[review_commissioned]`

**What it captures:** The direct, observable costs of commissioning an external governance review: independent reviewer fees, management distraction, and internal resource consumption.

**Data source:** `background/board/direct-costs-governance-review.md` — a detailed research paper estimating three cost components for an ASX-listed company with market capitalisation of AUD 10 billion (Qantas 2023–24):

| Component                                               | Central Estimate (AUD) | Decimal CAR  |
|---------------------------------------------------------|------------------------|--------------|
| A: Independent reviewer fees                            | 3,000,000              | −0.00030     |
| B: Management distraction (cognitive bandwidth method)  | 2,400,000              | −0.00040     |
| C: Internal resource consumption + remediation overhead | 2,600,000              | −0.00026     |
| **Total**                                               | **9,600,000**          | **−0.00096** |

The direct cost follows a Gamma distribution:

```
C_direct ~ Gamma(α = 4.55, rate β = 4741)
```

Properties: mean ≈ 9.6 bps, SD ≈ 4.5 bps, mode ≈ 7.5 bps, 90% CI: [3.1, 18.5] bps.

**Justification for value:** The weight is set at 15.0 (matching the review CAR weight) so that both the findings-release market reaction and the direct costs of the review are expressed on the same utility scale. At w_cost = 15.0 and expected C_direct = 0.00096:

```
Expected direct cost contribution = 15.0 × 0.00096 ≈ 0.014
```

This is economically small relative to other utility components — the direct cost of a review contributes approximately 0.014 units of disutility, compared to 1.5 for CEO loss or 8.0 for second-strike risk. This is consistent with the research finding that direct review costs are "small relative to the factors on the benefit side" (direct-costs-governance-review.md, §5.2).

**Basis:** Data-derived. Both the Gamma distribution parameters and the weight are calibrated from the research document's cost estimates.

***

### 5.7 Implementation Cost — `implementation_cost_sack` = 0.3

**Term (7):** Applied once for each sacking action:

-   `u -= 0.3` if `d1 = D3_ceo_transition`
-   `u -= 0.3` if `d_rev = Drev_sack_ceo`
-   `u -= 0.3` if `d_rev_post_review = Drev_sack_ceo`

**What it captures:** The administrative and transitional cost of implementing a CEO removal: board deliberation time, legal counsel for termination, succession process management, and short-term operational disruption during the leadership transition.

**Justification for value:** Set at 0.3 — deliberately low relative to other costs because the implementation mechanics of CEO removal at an ASX-listed company are well-established. The Board has access to legal counsel, succession planning frameworks, and HR support. The 0.3 value represents the incremental cost of executing the decision, not the broader consequences of CEO departure (which are captured by `ceo_loss_cost`, term 8).

**Basis:** Subjective. The 0.3 value was reduced from 1.0 (in the V2 commit) after recognising that the original value double-counted disruption effects already captured by `ceo_loss_cost`. The current value ensures that implementation cost is a minor factor in the Board's decision — the Board should not retain a problematic CEO solely because of administrative inconvenience.

**Implication:** The low implementation cost means that the cost of sacking the CEO at D_rev or D_rev_post_review is dominated by the `ceo_loss_cost` (1.5) rather than the act of implementing the decision (0.3). This is intentional: the Board's hesitation to remove the CEO should stem from genuine disruption concerns, not procedural overhead.

***

### 5.8 CEO Loss Cost — `ceo_loss_cost` = 1.5

**Term (8):** `u -= 1.5 · 1[CEO_removed ∧ ¬CEO_resigned_early]`

**What it captures:** The broader disruption cost to the Board from losing the CEO through involuntary or mid-game removal (sacking at D1/D_rev/D_rev_post_review, or CEO resignation/negotiation at D4/D4_post_review). This includes:

-   Loss of institutional knowledge and CEO-specific relationships
-   Market uncertainty during the transition period
-   Potential for adverse media coverage of the removal itself
-   Risk that the successor performs worse in the short term

**Exclusion:** Not applied when the CEO resigned early (pre-game, D0_ceo = CEO_resign), because that scenario has its own cost (`early_ceo_departure_cost` = 0.5). The early resignation is less disruptive because it is voluntary, controlled, and occurs before the AGM crisis intensifies.

**Justification for value:** The 1.5 value (positive, representing a cost) was corrected from an erroneous −1.5 (negative) that had been present in the Excel data contract. The sign error was critical: with `ceo_loss_cost = −1.5`, the Board's utility *increased* when the CEO was removed, inverting the intended incentive structure. The corrected value of +1.5 ensures that CEO removal imposes a genuine cost on the Board, creating the intended tension between accountability (removing a problematic CEO) and stability (retaining institutional knowledge).

**Basis:** Subjective. The 1.5 value is a modelling judgement that places CEO loss cost in the "moderate" tier — substantial enough to create meaningful reluctance to remove the CEO, but not so large that it prevents removal even when multiple other penalty terms (second-strike, regulatory liability) favour it.

**Historical note:** This was the subject of a critical bug fix (2026-03-03). The sign error caused the Board to prefer CEO removal in all scenarios, which contradicted the observed behaviour of Australian boards (who typically exhaust alternatives before removing a CEO).

***

### 5.9 Reputational Spill Weight — `reputational_spill_weight` = 1.0

**Term (9):** `u -= 1.0 · 1[overwhelming]`

**What it captures:** The reputational contagion effect of an overwhelming vote. When more than 50% of shareholders vote against the remuneration report, the event becomes a "governance crisis" that spills over into:

-   The company's broader brand and customer reputation
-   Other directors' external board positions (directors of "problem companies" receive fewer board invitations)
-   The company's ability to attract and retain talent at all levels

**Justification for value:** Set at 1.0 — a moderate additional penalty on top of the `overwhelming_penalty_weight` (3.0). Together, an overwhelming vote imposes 4.0 units of direct indicator-based disutility. The reputational spill is smaller than the direct overwhelming penalty because reputational contagion operates over a longer time horizon and is partially absorbed by the company's broader franchise value.

**Basis:** Subjective.

***

### 5.10 Second-Strike Spill Penalty — `second_strike_spill_penalty` = 8.0

**Term (10):** `u -= 8.0 · 1[strike ∧ CEO_present_at_end]`

**What it captures:** The existential risk to Board members from the Corporations Act 2001 two-strikes rule (ss 250U–250W). When a first strike occurs and the CEO remains in position at the end of the game, the Board faces:

1.  **Near-certain second strike at the next AGM.** With the underlying governance problems unaddressed (the CEO who triggered the crisis is still in place), shareholders have strong incentives to deliver a second consecutive strike.
2.  **Conditional spill resolution.** A second strike triggers a mandatory vote on a board spill resolution at the same AGM. Although spill resolutions have historically attracted very low support (averaging under 7% in 2024, 4.5% in 2025), the prospect of a spill is existentially threatening to individual directors.
3.  **Year of governance limbo.** Between the first strike and the next AGM, the Board operates under sustained scrutiny from institutional investors, proxy advisors, and regulators, diverting attention from operations.

**Justification for value:** The 8.0 value is the **largest single penalty** in the Board utility function. This reflects the asymmetry between the probability and the consequence of a board spill:

-   The *probability* of a successful spill is very low (no incumbent ASX director has ever lost their seat at a spill meeting since the rule was introduced in 2011)
-   But the *consequence* of a spill is total: all directors (except the managing director) vacate their offices and must stand for re-election
-   The *process* of facing a spill resolution — even if it fails — imposes massive personal and reputational costs on directors

The 8.0 value is set so that the second-strike penalty dominates most other considerations. When a first strike has occurred and the CEO is still present, the sum of `second_strike_spill_penalty` (8.0), `board_regulatory_liability` (5.0), and `qantas_legal_d_rev_penalty` (2.0) totals 15.0 — creating overwhelming pressure for the Board to either remove the CEO or face catastrophic consequences.

**Data sources:**

-   `background/ceo/2nd-strike.md`: Historical second-strike data (2020–2024), showing 0 second strikes in the ASX 200 in 2023, and spill resolution support averaging under 7%
-   Corporations Act 2001, ss 250U–250W: Legal framework for two-strikes rule
-   The value is empirically anchored to the severity of the spill mechanism but the specific magnitude (8.0) is a subjective scaling judgement

**Basis:** Empirically anchored. The functional form (strike ∧ CEO_present_at_end) is grounded in the legal mechanism; the magnitude is a modelling judgement.

***

### 5.11 Board Regulatory Liability — `board_regulatory_liability` = 5.0

**Term (11):** `u -= 5.0 · 1[strike ∧ CEO_present_at_end]`

**What it captures:** The personal regulatory risk to individual Board members from ASIC enforcement action when the Board demonstrably fails to respond to a governance crisis. If a first strike occurs (shareholders have formally signalled dissatisfaction) and the CEO remains in position (the Board took no action), individual directors face:

-   **ASIC director banning orders** under the Corporations Act 2001
-   **Personal fines** for breach of directors' duties (s 180 — duty of care and diligence)
-   **Class action exposure** as named respondents (not just the company)

**Justification for value:** Set at 5.0 — the second-largest indicator penalty after the second-strike risk. The high value reflects that regulatory liability is *personal* to individual directors: unlike corporate penalties (which the company absorbs), ASIC banning orders and personal fines affect directors' careers, other board positions, and professional standing.

**Justification for trigger:** The joint condition (strike ∧ CEO_present_at_end) captures the specific scenario that creates regulatory exposure: the Board had a clear signal of failure (the strike) and chose not to act (CEO still present). This is the pattern that ASIC has historically targeted in enforcement actions — demonstrated Board inaction in the face of known problems.

**Basis:** Subjective (legally grounded). The regulatory mechanism is documented in the Corporations Act; the magnitude is a modelling judgement based on the severity of personal consequences to directors.

***

### 5.12 Board D1 Liability — `board_d1_liability` = 4.0

**Term (12):** `u -= 4.0 · 1[overwhelming ∧ d1 = D0_minimal]`

**What it captures:** Additional regulatory and reputational liability when the Board took no action at D1 (minimal/status quo) despite the crisis being severe enough to produce an overwhelming vote (≥50%). This captures the specific failure mode where:

-   The Board was aware of the governance crisis before the AGM
-   The Board chose not to announce any reform package (no review, no CEO transition)
-   The vote outcome confirmed that a majority of shareholders considered the Board's inaction unacceptable

**Justification for trigger:** The overwhelming threshold (50%) is used rather than first-strike (25%) because the claim of Board negligence is stronger when a *majority* of shareholders rejected the status quo. A bare first strike (26%) could be attributed to proxy advisor influence or activist minority; an overwhelming vote is unambiguous.

**Justification for value:** Set at 4.0. This is lower than the `board_regulatory_liability` (5.0) because D1 inaction is a sin of omission (failing to act proactively) rather than a sin of commission (retaining the CEO after explicit shareholder rejection). The 4.0 value creates a meaningful incentive for the Board to take *some* action at D1, even if the optimal choice is uncertain.

**Basis:** Subjective (legally grounded).

***

### 5.13 Qantas Legal D1 Penalty — `qantas_legal_d1_penalty` = 3.0

**Term (13):** `u -= 3.0 · 1[strike ∧ d1 = D0_minimal]`

**What it captures:** Qantas-specific legal exposure (class actions, ACCC/ASIC company-level penalties) that becomes more severe when the Board's pre-AGM inaction is demonstrated. If the Board took no action at D1 and a first strike subsequently occurred, this provides evidence of corporate-level governance failure that strengthens plaintiff claims in:

-   **Shareholder class actions** alleging failure to disclose governance risks
-   **ACCC enforcement** (the Board's inaction suggests indifference to the underlying conduct issues)
-   **ASIC company penalties** for breach of continuous disclosure obligations

**Justification for value:** Set at 3.0, lower than the personal regulatory liability (5.0) because these are *company-level* consequences rather than personal director liabilities. The company's insurance and indemnification arrangements partially absorb these costs. The 3.0 value ensures that Qantas's legal exposure from D1 inaction is a meaningful but not dominant consideration.

**Basis:** Subjective (legally grounded).

***

### 5.14 Qantas Legal D_rev Penalty — `qantas_legal_d_rev_penalty` = 2.0

**Term (14):** `u -= 2.0 · 1[strike ∧ CEO_present_at_end]`

**What it captures:** Additional Qantas-specific legal exposure from the Board's failure to remove the CEO after a first strike. This overlaps with but is distinct from the regulatory liability (term 11): it captures the *company's* legal exposure rather than directors' personal exposure.

**Justification for value:** Set at 2.0, the lowest of the legal/regulatory penalty terms. This reflects that D_rev inaction (keeping the CEO after a strike) is more readily defensible than D1 inaction: the Board can argue that it was assessing the situation, waiting for review findings, or giving the CEO an opportunity to respond. The legal exposure is real but more attenuated.

**Basis:** Subjective (legally grounded).

***

### 5.15 Adverse Review + CEO Present Penalty — `adverse_review_ceo_present_penalty` = 5.0

**Term (15):** `u -= 5.0 · 1[review_commissioned ∧ review_adverse ∧ CEO_present_at_end]`

**What it captures:** The severe governance liability from retaining the CEO after the Board's own governance review has produced adverse (negative or neutral) findings. This is the "smoking gun" scenario: the Board commissioned an independent review, the review found problems, and the Board failed to act on the findings.

**Justification for structure:** This penalty captures three interlocking consequences:

1.  **Regulatory scrutiny.** A board that commissions a review and then ignores adverse findings is in a worse position than one that never commissioned a review at all. The review creates a documented record of known problems.
2.  **Shareholder revolt at next AGM.** Published adverse findings with the CEO still in place provide a clear narrative for activist shareholders: "The Board knew about the problems and did nothing."
3.  **Class-action exposure.** Adverse review findings, if published, establish that the Board was aware of governance failures — strengthening causation arguments in shareholder class actions.

**Data source:** The probability of adverse findings is modelled as `p_adverse ~ Beta(10, 5)`, mean = 2/3, derived from the Dirichlet(5, 5, 5) distribution for non-regulatory reviews grouping negative and neutral outcomes together (`background/board/external-governance-review-Bayesian-distribution.md`, §4.2). Of the three non-regulatory ASX reviews in the dataset (2013–2023), one was negative (Rio Tinto), one was neutral (Mineral Resources), and one was positive (BOQ). Grouping negative and neutral as "adverse" gives a mean probability of approximately 67%.

**Justification for value:** Set at 5.0, matching the `board_regulatory_liability`. This reflects the judgement that retaining the CEO after adverse review findings creates liability of comparable severity to retaining the CEO after a first strike — both represent documented governance failure with known consequences.

**Implication:** The adverse review penalty is the primary mechanism that ensures the Board prefers to sack the CEO after adverse review findings. Combined with the `ceo_loss_cost` (1.5 to remove, 5.0 penalty to retain), the net incentive strongly favours removal. The only scenario where retaining the CEO is safe is when both conditions fail: no first strike AND a positive review.

**Basis:** Data-informed. The review outcome distribution is derived from the ASX case study panel; the penalty magnitude is a subjective scaling judgement.

***

## 6. Board Overconfidence Bias

When the Board is the focal actor, the engine applies cognitive biases that distort the Board's *perception* of the game — not the actual outcomes. These biases affect how the Board evaluates its options but not what actually happens. Biases are loaded from `data/governance_spec.xlsx`, sheet `board_overconfidence`.

### 6.1 Overestimation of Governance Effectiveness

The Board overestimates the effectiveness of its governance actions on reducing the shareholder protest vote.

**Mechanism:** The governance effect f(D1) is drawn from a Uniform distribution with biased bounds:

| D1 Action         | Unbiased Bounds | Biased Bounds | Effect                                                                      |
|-------------------|-----------------|---------------|-----------------------------------------------------------------------------|
| D1_review         | U(0, 1)         | U(0.63, 1.0)  | Board thinks review is at least 63% effective (vs true minimum of 0%)       |
| D3_ceo_transition | U(−1, 0.5)      | U(−0.62, 0.5) | Board underestimates backlash from CEO exit (floor raised from −1 to −0.62) |

**Calibration:** From the Board overconfidence literature review (`background/board/board-level-overconfidence.md`):

-   Mean bias factor: β \~ U(0.25, 1.0). Production midpoint β = 0.625.
-   The bias shifts the floor upward: `d1_floor = β × d1_ceiling_unbiased = 0.625 × 1.0 ≈ 0.63`
-   Sources: Twardawski & Kind 2023, Brahma et al. 2023 (M&A board overconfidence), Coffeng et al. 2021 (only 20% of boards choose the best option while reporting high satisfaction)

### 6.2 Overprecision on Vote Uncertainty

The Board underestimates the variance of the vote outcome, perceiving shareholder sentiment as more predictable than it is.

**Mechanism:** The sigma_vote parameter in the Board's EU calculation is scaled down:

```
σ̂_V = σ_V × κ_σ,    κ_σ = 1/√κ
```

| Parameter                | Value                     | Meaning                                               |
|--------------------------|---------------------------|-------------------------------------------------------|
| κ (overprecision factor) | 3.5 (midpoint of U(2, 5)) | Board perceives 3.5× more precision                   |
| κ_σ (sigma_scale)        | 0.53                      | Board perceives vote uncertainty as 53% of true level |

**Calibration:** From CFO forecast survey data spanning 22 years and 28,000+ forecasts:

-   Executives provide 80% confidence intervals that capture only 33–36% of realised outcomes
-   This implies subjective standard deviation is approximately 1/3 of objective (κ ≈ 9 for individuals)
-   Board-level κ is set at U(2, 5), accounting for some correction through board deliberation processes
-   Production midpoint κ = 3.5, giving κ_σ = 0.53
-   Sources: Boundy-Singer et al. 2022, Guggenmos 2021 (metacognitive miscalibration)

### 6.3 Overestimation of Review Quality

The Board overestimates the governance quality that the review will find, perceiving findings as more favourable.

**Mechanism:** Two channels:

1.  **CAR location shift:** μ_f_biased = μ_f + 0.03 (Board perceives CAR \~3pp more favourable)
2.  **Adverse probability reduction:** β_biased = 5 × (1 + 10 × 0.03) = 6.5, so p_adverse \~ Beta(10, 6.5) with mean ≈ 0.606 (vs 0.667 unbiased)

| Parameter       | Unbiased | Biased | Effect                                               |
|-----------------|----------|--------|------------------------------------------------------|
| review_car_bias | 0.0      | 0.03   | Board perceives review CAR \~3pp higher              |
| E[p_adverse]    | 0.667    | 0.606  | Board underestimates probability of adverse findings |
| E[CAR]          | −5.0%    | −2.0%  | Board expects milder market reaction                 |

**Calibration:** From case studies — with Board overconfidence factor β = 0.625, the Board perceives approximately 3pp improvement in the findings window CAR.

### 6.4 Bias Parameters (Production Defaults)

| Parameter         | Value | Source Sheet         |
|-------------------|-------|----------------------|
| `d1_floor`        | 0.63  | board_overconfidence |
| `d1_ceiling`      | 1.00  | board_overconfidence |
| `d3_floor`        | −0.62 | board_overconfidence |
| `d3_ceiling`      | 0.50  | board_overconfidence |
| `sigma_scale`     | 0.53  | board_overconfidence |
| `review_car_bias` | 0.03  | board_overconfidence |

### 6.5 Propagation

Biases are applied consistently in both:

-   The focal actor's own EU calculation (tree value computation)
-   Rollout simulations within predictive distributions (opponent modelling)

This ensures self-consistent decision-making: the Board's perception of its options and its prediction of opponent behaviour are both distorted by the same cognitive biases.

***

## 7. Chance Node Models Relevant to Board Utility

Two chance nodes produce stochastic outcomes that directly enter the Board utility function.

### 7.1 Vote Model (V)

The shareholder vote outcome V ∈ (0, 1) is generated by a logit-normal model:

```
B_agm = B_mkt^(i) + γ_A^(i) · 1[A2 = rec_strike] + γ_AH^(i) · 1[A2 = rec_strike] · 1[headline] + γ_D^(i) · f(D1)

logit(V) ~ N(α_V^(i) + B_agm, σ_V^(i))

V_final = max(V_logit_normal, V_floor)   if headline_incident
```

**Board-relevant features:**

-   The governance effect f(D1) determines how the Board's D1 choice affects the vote
-   f(D0_minimal) = 0 (no action, baseline)
-   f(D1_review) \~ U(0, 1): review REDUCES protest; E[f] = 0.5
-   f(D3_ceo_transition) \~ U(−1, 0.5): ambiguous effect; E[f] = −0.25
-   γ_D is typically negative (from posterior), so positive f reduces B_agm → reduces vote

The vote outcome feeds into utility terms (2), (3), (4), (9), (10), (11), (12), (13), (14) — all the vote-dependent and strike-dependent components.

### 7.2 Review Model (R)

If review is commissioned, the review produces:

**Outcome rating (adverse vs positive):**

```
p_adverse ~ Beta(10, 5),    mean = 2/3
adverse ~ Bernoulli(p_adverse)
```

**Market reaction (CAR):**

```
μ_f ~ t(4, −0.05, 0.03)
σ_f ~ |N(0, 0.10)|
CAR ~ t(3, μ_f, σ_f)
```

**Direct cost:**

```
C_direct ~ Gamma(4.55, rate = 4741),    mean ≈ 9.6 bps
```

The review outcome feeds into utility terms (5), (6), and (15).

**Distribution rationale:** The Student-t(3) for CAR produces genuinely heavy tails, accommodating extreme outcomes like Star's −13.95% findings-window CAR. The use of t(4) for μ_f (rather than Cauchy/t(1)) ensures that the mean of the location parameter is well-defined (E[μ_f] = −0.05), preventing single extreme draws from dominating Monte Carlo averages. See `docs/algebraic.md`, §3.2 for the detailed justification.

***

## 8. How Board Utility Is Computed in the Engine

### 8.1 Tree Recursion

When the Board is the focal actor, the engine evaluates the game tree recursively (`engine/tree.py`):

1.  **D1 (Board, focal):** Enumerate all feasible D1 actions {D0_minimal, D1_review, D3_ceo_transition}. For each, recurse through the tree and compute expected utility. Select the action that maximises EU.
2.  **A2 (ASA, opponent):** Compute the ARA predictive distribution p_B(A2 \| h) over ASA's action. Weight downstream utilities by this distribution.
3.  **V (Nature, chance):** Monte Carlo integration over vote outcomes (default M_V = 50 samples). Each sample produces a vote percentage, strike indicator, and overwhelming indicator.
4.  **D4 (CEO, opponent):** Compute ARA predictive distribution p_B(D4 \| h). Weight downstream utilities.
5.  **D_rev (Board, focal):** Enumerate feasible D_rev actions. Select the action that maximises EU.
6.  **R (Nature, chance):** Monte Carlo integration over review outcomes (default M_R = 20 samples). Each sample produces a CAR, adverse indicator, and direct cost.
7.  **Post-review round** (conditional): If adverse ∧ CEO_present, recurse through D4_post_review and D_rev_post_review.
8.  **Terminal:** Compute `utility_board(outcome, params)` using the complete game outcome.

### 8.2 Parameter Loading

Board utility weights are loaded from `data/governance_spec.xlsx`:

```python
params = load_utility_weights("data/governance_spec.xlsx", "Board")
# Returns dict: {"vote_penalty_weight": 2.0, "review_car_weight": 15.0, ...}
```

### 8.3 Solver Integration

The solver (`engine/solver.py`) orchestrates the full computation:

1.  Load beliefs (BeliefBundle from checkpoint .npz)
2.  For each candidate D1 action:
    -   For each belief draw i = 1, ..., N:
        -   Evaluate the tree recursion starting from the node after D1
    -   Compute EU(action) = mean over belief draws
3.  Select d1\* = argmax EU(action)

***

## 9. Opponent Priors on Board Parameters

When the Board is an *opponent* (i.e., ASA or CEO is the focal actor), the focal actor models the Board's utility using sampled parameters from `data/opponent_priors.xlsx`. These priors encode the focal actor's uncertainty about the Board's true preferences.

### 9.1 ASA's Priors on Board Utility Parameters

| Parameter                            | Distribution         | Param1 (μ or shape) | Param2 (σ or scale) | Interpretation                                            |
|--------------------------------------|----------------------|---------------------|---------------------|-----------------------------------------------------------|
| `vote_penalty_weight`                | Normal(2.0, 0.5)     | 2.0                 | 0.5                 | ASA believes Board's vote sensitivity is \~2.0 ± 0.5      |
| `ceo_loss_cost`                      | Normal(1.5, 0.5)     | 1.5                 | 0.5                 | ASA believes Board's CEO departure cost is \~1.5 ± 0.5    |
| `spill_risk_weight`                  | Normal(2.5, 0.5)     | 2.5                 | 0.5                 | ASA believes Board's spill sensitivity is \~2.5 ± 0.5     |
| `implementation_cost_review`         | LogNormal(−1.2, 0.3) | −1.2                | 0.3                 | ASA believes review implementation cost has median ≈ 0.30 |
| `second_strike_spill_penalty`        | Normal(8.0, 2.0)     | 8.0                 | 2.0                 | ASA believes Board's spill fear is \~8.0 ± 2.0            |
| `board_regulatory_liability`         | Normal(5.0, 1.5)     | 5.0                 | 1.5                 | ASA believes Board's regulatory exposure is \~5.0 ± 1.5   |
| `board_d1_liability`                 | Normal(4.0, 1.0)     | 4.0                 | 1.0                 | ASA believes Board's D1 liability is \~4.0 ± 1.0          |
| `qantas_legal_d1_penalty`            | Normal(3.0, 1.0)     | 3.0                 | 1.0                 | ASA believes Qantas legal exposure is \~3.0 ± 1.0         |
| `qantas_legal_d_rev_penalty`         | Normal(2.0, 0.8)     | 2.0                 | 0.8                 | ASA believes D_rev legal exposure is \~2.0 ± 0.8          |
| `adverse_review_ceo_present_penalty` | Normal(5.0, 2.0)     | 5.0                 | 2.0                 | ASA believes adverse-review penalty is \~5.0 ± 2.0        |

### 9.2 CEO's Priors on Board Utility Parameters

| Parameter                            | Distribution      | Param1 | Param2 | Interpretation                                          |
|--------------------------------------|-------------------|--------|--------|---------------------------------------------------------|
| `vote_penalty_weight`                | Normal(2.0, 0.5)  | 2.0    | 0.5    | CEO believes Board's vote sensitivity is \~2.0 ± 0.5    |
| `ceo_loss_cost`                      | Normal(1.5, 0.5)  | 1.5    | 0.5    | CEO believes departure cost is \~1.5 ± 0.5              |
| `spill_risk_weight`                  | Normal(2.5, 0.5)  | 2.5    | 0.5    | CEO believes Board's spill sensitivity is \~2.5 ± 0.5   |
| `review_car_weight`                  | Normal(15.0, 3.0) | 15.0   | 3.0    | CEO believes Board's CAR sensitivity is \~15.0 ± 3.0    |
| `review_direct_cost_weight`          | Normal(15.0, 3.0) | 15.0   | 3.0    | CEO believes Board's cost sensitivity is \~15.0 ± 3.0   |
| `implementation_cost_sack`           | Normal(0.3, 0.1)  | 0.3    | 0.1    | CEO believes sacking implementation cost is \~0.3 ± 0.1 |
| `early_ceo_departure_cost`           | Normal(0.5, 0.2)  | 0.5    | 0.2    | CEO believes early departure cost is \~0.5 ± 0.2        |
| `second_strike_spill_penalty`        | Normal(8.0, 2.0)  | 8.0    | 2.0    | CEO believes Board's spill fear is \~8.0 ± 2.0          |
| `board_regulatory_liability`         | Normal(5.0, 1.5)  | 5.0    | 1.5    | CEO believes Board regulatory risk is \~5.0 ± 1.5       |
| `board_d1_liability`                 | Normal(4.0, 1.0)  | 4.0    | 1.0    | CEO believes Board D1 liability is \~4.0 ± 1.0          |
| `qantas_legal_d1_penalty`            | Normal(3.0, 1.0)  | 3.0    | 1.0    | CEO believes Qantas D1 legal risk is \~3.0 ± 1.0        |
| `qantas_legal_d_rev_penalty`         | Normal(2.0, 0.8)  | 2.0    | 0.8    | CEO believes D_rev legal risk is \~2.0 ± 0.8            |
| `adverse_review_ceo_present_penalty` | Normal(5.0, 2.0)  | 5.0    | 2.0    | CEO believes adverse-review penalty is \~5.0 ± 2.0      |

**Note:** The ASA and CEO priors on Board parameters are centred at the Board's actual parameter values, reflecting the assumption that opponents have approximately correct (but uncertain) beliefs about the Board's preferences. The standard deviations encode the degree of uncertainty about the Board's true preferences.

***

## 10. Summary of Data Sources

| Source Document                                  | Location                                                               | Parameters Informed                                                              |
|--------------------------------------------------|------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| ASX governance review case studies (2014–2023)   | `background/board/governance-review-case-studies.md`                   | `review_car_weight` (15.0), CAR distribution parameters                          |
| Direct costs of governance review                | `background/board/direct-costs-governance-review.md`                   | `review_direct_cost_weight` (15.0), Gamma(4.55, 4741) parameters                 |
| External governance review Bayesian distribution | `background/board/external-governance-review-Bayesian-distribution.md` | `adverse_review_ceo_present_penalty` data basis, Beta(10, 5) adverse probability |
| Board-level overconfidence literature review     | `background/board/board-level-overconfidence.md`                       | All overconfidence bias parameters (d1_floor, sigma_scale, etc.)                 |
| Second-strike voting data (2020–2024)            | `background/ceo/2nd-strike.md`                                         | `second_strike_spill_penalty` (8.0), `spill_risk_weight` (2.5)                   |
| Shareholder vote model v2 specification          | `background/shareholders/shareholder-vote-v2.md`                       | Vote model structure, governance effect specification                            |
| External governance review analysis              | `background/board/external-governance-review-analysis.md`              | Review outcome classification, regulatory vs non-regulatory patterns             |
| Corporations Act 2001, ss 250U–250W              | External (Commonwealth legislation)                                    | `second_strike_spill_penalty`, `board_regulatory_liability`                      |
| Tversky & Kahneman 1992                          | External (academic literature)                                         | Overconfidence framework basis                                                   |
| CFO forecast survey data (28,000+ forecasts)     | Referenced in overconfidence literature                                | `sigma_scale` (overprecision parameter)                                          |
| Twardawski & Kind 2023, Brahma et al. 2023       | Referenced in overconfidence literature                                | Mean bias (overestimation) parameters                                            |
| Coffeng et al. 2021                              | Referenced in overconfidence literature                                | Board decision quality calibration                                               |
| Governance specification                         | `data/governance_spec.xlsx`                                            | All 15 parameter values (utilities_board sheet)                                  |
| Opponent priors                                  | `data/opponent_priors.xlsx`                                            | ASA and CEO prior distributions on Board parameters                              |

### Classification of Parameter Basis

| Basis                             | Parameters                                                                                                                          |
|-----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| **Data-derived**                  | `review_car_weight`, `review_direct_cost_weight`                                                                                    |
| **Empirically anchored**          | `spill_risk_weight`, `second_strike_spill_penalty`                                                                                  |
| **Subjective (legally grounded)** | `board_regulatory_liability`, `board_d1_liability`, `qantas_legal_d1_penalty`, `qantas_legal_d_rev_penalty`                         |
| **Data-informed**                 | `adverse_review_ceo_present_penalty`                                                                                                |
| **Subjective (calibrated)**       | `vote_penalty_weight`                                                                                                               |
| **Subjective**                    | `early_ceo_departure_cost`, `overwhelming_penalty_weight`, `implementation_cost_sack`, `ceo_loss_cost`, `reputational_spill_weight` |
