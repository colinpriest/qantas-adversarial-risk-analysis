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

**Strategic context:** This node is reached only on negative review paths where the CEO has survived to this point. The `negative_review_finding_penalty` (term 12, §5.12) and the `inaction_ceo_present_penalty` (term 3, §5.3) together create a strong incentive for the Board to remove the CEO at this node, since retaining the CEO after documented governance failures exposes the Board to regulatory liability, shareholder revolt at the next AGM, and class-action risk.

***

## 3. Board Utility Function Formula

The Board utility is computed at the Terminal node over the complete game outcome. The function has three structural layers: unconditional inaction penalties, vote-dependent penalties, and retained outcome-specific components.

```
u_B(Z) = 0
         ── 1. INACTION COMPONENTS (unconditional) ──
         − w_inact_base   · 1[board_inactive]                                 (1)
         − w_inact_no_rev · 1[¬review_commissioned]                           (2)
         − w_inact_ceo    · 1[ceo_present_at_end]                             (3)
         − w_inact_no_sack · 1[¬removed_involuntary]                          (4)

         ── 2. VOTE PENALTIES (linear in normalized excess) ──
         − w_v_strike · max(0, (V − 0.25) / 0.75)                            (5)
         − w_v_over   · max(0, (V − 0.50) / 0.50)                            (6)

         ── 3. RETAINED COMPONENTS ──
         − w_pass     · 1[CEO_resigned_early]                                 (7)
         + w_CAR⁺     · max(CAR, 0) · 1[review_commissioned]                 (8a)
         − w_CAR⁻     · max(−CAR, 0) · 1[review_commissioned]               (8b)
         − w_cost     · C_direct · 1[review_commissioned]                     (9)
         − w_impl     · (1[d1 = D3] + 1[d_rev = sack] + 1[d_rev' = sack])   (10)
         − max(0, w_loss − w_loss_over · 1[overwhelming])
                       · 1[removed_involuntary]                               (11)
         − w_rev_neg  · 1[review_commissioned ∧ outcome = negative]          (12)
         − w_rev_bal  · 1[review_commissioned ∧ outcome = balanced]          (13)
         − w_rev_post · 1[removed_involuntary ∧ ¬review_commissioned]        (14)
```

where:

-   `board_inactive` = Board took only minimal action: `d1 = D0_minimal` AND `d_rev ∉ {sack, review}` AND `d_rev' ≠ sack`
-   `ceo_present_at_end = ¬CEO_removed ∧ ¬CEO_resigned_early`
-   `removed_involuntary = CEO_removed ∧ ¬CEO_resigned_early`
-   `(x)₊ = max(x, 0)` — the positive part function
-   Vote penalties are **linear** in normalized excess (not quadratic)
-   Review CAR uses **loss aversion**: `w_CAR⁺ = w_CAR / ((1 + λ) / 2)`, `w_CAR⁻ = λ · w_CAR⁺`
-   Review outcome is **trinary**: negative, balanced, or positive (not binary adverse/positive)
-   All indicator functions `1[·]` are {0, 1}-valued

### 3.1 Structural Rationale

The utility function is organized in three layers reflecting distinct governance mechanisms:

**Layer 1 — Inaction penalties** fire unconditionally regardless of vote level. They capture the Board's baseline governance obligations: act on known problems, commission oversight, ensure executive accountability. These create a non-zero cost floor for passive governance strategies even when the shareholder vote is mild.

**Layer 2 — Vote penalties** scale linearly with vote excess above thresholds (25% strike, 50% overwhelming), normalized to [0, 1] range. The linear form (replacing the earlier quadratic specification) captures proportional escalation of reputational damage as the protest vote increases.

**Layer 3 — Retained components** capture outcome-specific costs: CEO departure disruption, review findings impact, implementation costs, and due diligence requirements. The review CAR term uses Kahneman-Tversky loss aversion (lambda = 2.25) to reflect the asymmetric impact of negative vs positive market reactions.

***

## 4. Parameter Reference Table

All parameter values are stored in `data/governance_spec.xlsx`, sheet `utilities_board`. No magic numbers appear in code.

| #  | Parameter Name                      | Symbol        | Value | Term  | Layer    |
|----|-------------------------------------|---------------|-------|-------|----------|
| 1  | `inaction_base_penalty`             | w_inact_base  | 3.0   | (1)   | Inaction |
| 2  | `inaction_no_review_penalty`        | w_inact_no_rev| 2.0   | (2)   | Inaction |
| 3  | `inaction_ceo_present_penalty`      | w_inact_ceo   | 5.0   | (3)   | Inaction |
| 4  | `inaction_no_sack_penalty`          | w_inact_no_sack| 3.0  | (4)   | Inaction |
| 5  | `vote_strike_penalty`               | w_v_strike    | 2.0   | (5)   | Vote     |
| 6  | `vote_overwhelming_penalty`         | w_v_over      | 3.0   | (6)   | Vote     |
| 7  | `board_passivity_after_departure`   | w_pass        | 0.5   | (7)   | Retained |
| 8  | `review_car_weight`                 | w_CAR         | 15.0  | (8)   | Retained |
| 8L | `review_car_loss_aversion`          | lambda_la     | 2.25  | (8)   | Retained |
| 9  | `review_direct_cost_weight`         | w_cost        | 15.0  | (9)   | Retained |
| 10 | `implementation_cost_sack`          | w_impl        | 1.0   | (10)  | Retained |
| 11 | `ceo_loss_cost`                     | w_loss        | 1.5   | (11)  | Retained |
| 11s| `ceo_loss_shock_overwhelming`       | w_loss_over   | 0.5   | (11)  | Retained |
| 12 | `negative_review_finding_penalty`   | w_rev_neg     | 5.0   | (12)  | Retained |
| 13 | `balanced_review_finding_penalty`   | w_rev_bal     | 2.5   | (13)  | Retained |
| 14 | `review_after_removal_penalty`      | w_rev_post    | 3.0   | (14)  | Retained |

***

## 5. Detailed Parameter Specifications

### 5.1 Inaction Base Penalty — `inaction_base_penalty` = 3.0

**Term (1):** `u -= 3.0 · 1[board_inactive]`

**What it captures:** The baseline governance cost when the Board takes only minimal action throughout the entire game: `d1 = D0_minimal` AND no sacking or review at D_rev or D_rev_post_review. This penalises complete passivity in the face of a known governance crisis.

**Indicator definition:** `board_inactive` is true when `d1 = D0_minimal` AND `d_rev ∉ {sack, review}` AND `d_rev_post_review ≠ sack`. Any decisive action at any decision point clears this indicator.

**Justification for value:** Set at 3.0. The penalty creates a meaningful cost floor for passive strategies even when the shareholder vote is mild. A Board that takes no action despite the known governance crisis faces reputational costs, institutional investor engagement, and proxy advisor criticism regardless of the AGM outcome.

**Basis:** Estimated via ordinal probit regression on expert governance assessments (board utility quantification pipeline, Stage 4A softmax MLE).

***

### 5.2 Inaction No-Review Penalty — `inaction_no_review_penalty` = 2.0

**Term (2):** `u -= 2.0 · 1[¬review_commissioned]`

**What it captures:** The governance cost of failing to commission an independent review. In a crisis context, the absence of an independent review signals that the Board is not engaging with the underlying governance problems. This is distinct from the base inaction penalty: a Board that takes CEO transition action at D1 without commissioning a review still incurs this penalty.

**Justification for value:** Set at 2.0 — lower than the base inaction penalty (3.0) because failing to commission a review is a less egregious omission than complete passivity. The Board may have legitimate reasons for not commissioning a review (e.g., the CEO has already departed), but the absence of independent oversight is still a governance weakness.

**Basis:** Estimated via ordinal probit regression on expert governance assessments.

***

### 5.3 Inaction CEO-Present Penalty — `inaction_ceo_present_penalty` = 5.0

**Term (3):** `u -= 5.0 · 1[ceo_present_at_end]`

**What it captures:** The governance cost when the CEO responsible for the crisis remains in position at the terminal node. This is the largest inaction penalty, reflecting the Board's fundamental obligation to ensure executive accountability. Retaining the CEO exposes the Board to:

-   Near-certain second strike at the next AGM (under the Corporations Act 2001 two-strikes rule)
-   Regulatory scrutiny from ASIC for failing to address demonstrated governance failures
-   Sustained negative media coverage and institutional investor engagement

**Justification for value:** Set at 5.0 — the highest inaction component. The value reflects that CEO retention is the single most visible indicator of Board inaction to external stakeholders. When the CEO who triggered the crisis remains in place, all other governance actions (reviews, reforms) are perceived as inadequate by shareholders and regulators.

**Basis:** Estimated via ordinal probit regression on expert governance assessments.

***

### 5.4 Inaction No-Sack Penalty — `inaction_no_sack_penalty` = 3.0

**Term (4):** `u -= 3.0 · 1[¬removed_involuntary]`

**What it captures:** The governance cost when the Board does not explicitly remove the CEO through involuntary departure (sacking at D1, D_rev, or D_rev_post_review). This fires even when the CEO departs voluntarily (resign or negotiate at D4), because the Board's failure to exercise its removal authority signals institutional weakness.

**Distinction from term (3):** Term (3) fires when the CEO is present at the end; term (4) fires when the Board did not *actively* remove the CEO. Both fire when the CEO stays; only term (4) fires when the CEO resigns voluntarily (the Board did not sack, but the CEO departed). Neither fires when the Board sacks the CEO.

**Justification for value:** Set at 3.0, matching the base inaction penalty. The value reflects the importance the Board places on demonstrating decisive authority, distinct from simply achieving CEO departure through any channel.

**Basis:** Estimated via ordinal probit regression on expert governance assessments.

***

### 5.5 Vote Strike Penalty — `vote_strike_penalty` = 2.0

**Term (5):** `u -= 2.0 · max(0, (V − 0.25) / 0.75)`

**What it captures:** The **linear** escalation of reputational damage as the protest vote exceeds the first-strike threshold of 25%, normalized to the [0, 1] range. The normalization `(V − 0.25) / 0.75` maps the excess vote to a unit interval: a bare first strike (26%) produces a penalty of approximately 0.013, while an 83% vote produces a penalty of 1.55.

**Justification for structure:** The linear form (replacing the earlier quadratic specification) captures proportional escalation of reputational damage. Each additional percentage point of protest vote above 25% imposes a constant marginal cost. The normalization ensures that the penalty is interpretable: a weight of 2.0 means that the maximum possible vote-strike penalty (at V = 100%) is 2.0 units.

| Vote % | Normalized excess | Penalty contribution |
|--------|-------------------|----------------------|
| 30%    | 0.067             | −0.133               |
| 40%    | 0.200             | −0.400               |
| 50%    | 0.333             | −0.667               |
| 60%    | 0.467             | −0.933               |
| 83%    | 0.773             | −1.547               |

**Basis:** Estimated via ordinal probit regression on expert governance assessments.

***

### 5.6 Vote Overwhelming Penalty — `vote_overwhelming_penalty` = 3.0

**Term (6):** `u -= 3.0 · max(0, (V − 0.50) / 0.50)`

**What it captures:** Additional linear penalty for vote excess above the 50% overwhelming threshold, normalized to [0, 1]. An overwhelming vote is qualitatively distinct from a standard first strike: it signals majority shareholder dissatisfaction, attracts national media coverage, and places the Board under immediate pressure to act. The two-strikes rule (Corporations Act 2001, ss 250U-250W) makes an overwhelming first strike particularly threatening.

**Justification for value:** Set at 3.0 — higher than the strike penalty (2.0) because crossing the 50% line triggers qualitatively different consequences: automatic engagement demands from institutional investors, heightened regulatory scrutiny, and increased director re-election risk. The normalization means that at V = 83%, the overwhelming penalty contributes `3.0 × 0.66 = 1.98` in disutility.

**Basis:** Estimated via ordinal probit regression on expert governance assessments.

***

### 5.7 Board Passivity After Departure — `board_passivity_after_departure` = 0.5

**Term (7):** `u -= 0.5 · 1[CEO_resigned_early]`

**What it captures:** The cost to the Board when the CEO resigns before the game tree begins (the D0_ceo = CEO_resign scenario). Even though early resignation removes the CEO controversy from the AGM, it imposes transition costs: leadership vacuum during crisis, market uncertainty about succession, disruption to ongoing operations. The penalty captures the Board's failure to manage the departure proactively.

**Justification for value:** Set at 0.5 — significantly lower than the full `ceo_loss_cost` (1.5) because voluntary pre-crisis departure is far less disruptive than mid-crisis involuntary removal. The CEO controls the timing and narrative, reducing market shock and allowing orderly succession.

**Basis:** Subjective. The 0.5 value reflects the judgement that early voluntary departure imposes approximately one-third the disruption cost of involuntary removal, consistent with CEO turnover literature showing forced departures produce 2-4x the stock price impact of voluntary departures.

***

### 5.8 Review CAR Weight — `review_car_weight` = 15.0 (with loss aversion lambda = 2.25)

**Terms (8a, 8b):**
```
u += w_CAR_pos · max(CAR, 0) · 1[review_commissioned]
u -= w_CAR_neg · max(−CAR, 0) · 1[review_commissioned]
```

where `w_CAR_pos = w_CAR / ((1 + lambda) / 2)` and `w_CAR_neg = lambda · w_CAR_pos`.

**What it captures:** The market-valuation impact of governance review findings, measured as cumulative abnormal return (CAR) during the findings release window. The asymmetric treatment reflects Kahneman-Tversky loss aversion: negative CARs impose disproportionately larger utility costs than positive CARs of equal magnitude provide benefit.

**Loss aversion decomposition:** With `w_CAR = 15.0` and `lambda = 2.25`:
-   `w_CAR_pos = 15.0 / 1.625 = 9.23` — weight on positive CAR
-   `w_CAR_neg = 2.25 × 9.23 = 20.77` — weight on negative CAR

| CAR outcome      | Utility contribution |
|------------------|----------------------|
| Star (−13.95%)   | −2.90                |
| BOQ (−5.70%)     | −1.18                |
| Westpac (−3.00%) | −0.62                |
| CBA (+1.75%)     | +0.16                |
| Qantas (+0.85%)  | +0.08                |

**Data source:** ASX governance review case study data (`background/board/governance-review-case-studies.md`), documenting findings-release CARs for six companies ranging from −13.95% to +1.75%.

**Basis:** Data-derived. The anchor weight is calibrated from empirical CARs; the loss aversion parameter (2.25) is from Tversky and Kahneman (1992).

***

### 5.9 Review Direct Cost Weight — `review_direct_cost_weight` = 15.0

**Term (9):** `u -= 15.0 · C_direct · 1[review_commissioned]`

**What it captures:** The direct costs of commissioning an external governance review: independent reviewer fees, management distraction, and internal resource consumption.

**Data source:** `background/board/direct-costs-governance-review.md` estimates total direct costs of approximately AUD 9.6 million (0.00096 decimal CAR) for an ASX-listed company with AUD 10 billion market capitalisation. The direct cost follows `C_direct ~ Gamma(4.55, rate = 4741)`, mean = 9.6 bps.

**Justification for value:** At w_cost = 15.0 and expected C_direct = 0.00096, the expected contribution is `15.0 × 0.00096 = 0.014` — economically small relative to other utility components. The weight matches the review CAR weight so that market reactions and direct costs are on the same utility scale.

**Basis:** Data-derived.

***

### 5.10 Implementation Cost — `implementation_cost_sack` = 1.0

**Term (10):** Applied once for each sacking action:

-   `u -= 1.0` if `d1 = D3_ceo_transition`
-   `u -= 1.0` if `d_rev = Drev_sack_ceo`
-   `u -= 1.0` if `d_rev_post_review = Drev_sack_ceo`

**What it captures:** The administrative and transitional cost of implementing a CEO removal: board deliberation time, legal counsel for termination, succession process management, and short-term operational disruption during the leadership transition.

**Justification for value:** Set at 1.0 — the incremental cost of executing the removal decision, distinct from the broader consequences of CEO departure captured by `ceo_loss_cost` (term 11).

**Basis:** Subjective.

***

### 5.11 CEO Loss Cost — `ceo_loss_cost` = 1.5 (with shock attenuation)

**Term (11):** `u -= max(0, w_loss − w_loss_over · 1[overwhelming]) · 1[removed_involuntary]`

**What it captures:** The broader disruption cost from involuntary CEO removal (sacking at D1/D_rev/D_rev_post_review, or CEO resignation/negotiation at D4/D4_post_review). The base cost is attenuated when an overwhelming vote has occurred, reflecting the reduced market shock when CEO removal follows a strong shareholder mandate.

**Shock attenuation mechanism:** The `ceo_loss_shock_overwhelming` parameter (0.5) reduces the effective CEO loss cost when the vote exceeds 50%. With an overwhelming vote, the effective cost is `max(0, 1.5 − 0.5) = 1.0`. Without the overwhelming vote, the full cost of 1.5 applies. The rationale is that an overwhelming shareholder mandate legitimises CEO removal, reducing the negative market reaction and institutional investor concern.

**Exclusion:** Not applied when the CEO resigned early (pre-game, D0_ceo = CEO_resign), because that scenario is captured by `board_passivity_after_departure` (term 7).

**Historical note:** The sign of this parameter was the subject of a critical bug fix (2026-03-03). The original value of −1.5 in the Excel data contract caused CEO removal to *increase* Board utility.

**Basis:** Subjective. The 1.5 base value places CEO loss in the "moderate" tier — creating meaningful reluctance to remove the CEO, but not preventing removal when inaction penalties dominate.

***

### 5.12 Negative Review Finding Penalty — `negative_review_finding_penalty` = 5.0

**Term (12):** `u -= 5.0 · 1[review_commissioned ∧ outcome = negative]`

**What it captures:** The governance liability from a negative review finding. When the Board's own governance review produces a negative outcome, the Board faces regulatory scrutiny, shareholder revolt risk at the next AGM, and class-action exposure. The review creates a documented record of known governance failures.

**Key change from prior specification:** This penalty fires regardless of whether the CEO is still present at the terminal node. Review findings reflect on Board governance quality independent of CEO status — a negative finding is damaging whether or not the Board subsequently removed the CEO.

**Justification for value:** Set at 5.0, reflecting the severity of documented governance failure. A negative review finding establishes a factual basis for regulatory enforcement (ASIC), shareholder class actions, and proxy advisor downgrade.

**Basis:** Estimated via ordinal probit regression on expert governance assessments.

***

### 5.13 Balanced Review Finding Penalty — `balanced_review_finding_penalty` = 2.5

**Term (13):** `u -= 2.5 · 1[review_commissioned ∧ outcome = balanced]`

**What it captures:** The moderate governance cost from a balanced review finding. A balanced outcome acknowledges some governance concerns while also recognising mitigating factors — the typical "mistakes were made but steps are being taken" conclusion of Board-initiated crisis reviews. While less damaging than a negative finding, a balanced review still imposes costs: it confirms that problems existed (useful to plaintiffs), generates media coverage, and requires the Board to demonstrate follow-through.

**Distinction from negative penalty:** The balanced penalty (2.5) is exactly half the negative penalty (5.0), reflecting the reduced severity of "mixed" findings relative to clearly adverse ones. A positive review finding (no governance concerns identified) incurs no penalty.

**Basis:** Estimated via ordinal probit regression on expert governance assessments.

***

### 5.14 Review After Removal Penalty — `review_after_removal_penalty` = 3.0

**Term (14):** `u -= 3.0 · 1[removed_involuntary ∧ ¬review_commissioned]`

**What it captures:** The due diligence cost when the Board removes the CEO involuntarily without having commissioned a governance review. Removing the CEO without independent review evidence exposes the Board to claims of:

-   Arbitrary or politically motivated removal (no documented basis)
-   Failure to address systemic governance issues (removing the individual without diagnosing the system)
-   Unfair dismissal claims from the departing CEO (no independent findings to justify removal)

**Justification for structure:** This is an interaction term that separates the global no-review penalty (term 2, which is relatively small) from the context-specific incentive to commission a review before sacking the CEO. The combination of terms (2) and (14) means that failing to commission a review is mildly costly in general (2.0) but substantially more costly when the Board has also removed the CEO (2.0 + 3.0 = 5.0).

**Justification for value:** Set at 3.0 — large enough to create a strong incentive for the Board to commission a review before or alongside CEO removal, but not so large as to prevent removal in urgent circumstances.

**Basis:** Estimated via ordinal probit regression on expert governance assessments.

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
2.  **Outcome probability shift:** The Dirichlet positive-outcome concentration is inflated: α_pos_biased = 1 × (1 + 10 × 0.03) = 1.3, giving Dirichlet(38, 160, 1.3) with a slight tilt toward positive outcomes compared to the unbiased Dirichlet(38, 160, 1).

| Parameter       | Unbiased              | Biased                 | Effect                                                    |
|-----------------|-----------------------|------------------------|-----------------------------------------------------------|
| review_car_bias | 0.0                   | 0.03                   | Board perceives review CAR \~3pp higher                   |
| Outcome dist    | Dirichlet(38, 160, 1) | Dirichlet(38, 160, 1.3)| Board slightly overestimates probability of positive findings |
| E[CAR]          | −5.0%                 | −2.0%                  | Board expects milder market reaction                      |

**Calibration:** From case studies — with Board overconfidence factor beta = 0.625, the Board perceives approximately 3pp improvement in the findings window CAR. The Dirichlet shift is modest because the positive outcome concentration (1.3 vs 1.0) has minimal impact on the dominant balanced category (160).

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

The vote outcome feeds into utility terms (5) and (6) — the linear vote penalties that scale with excess above the strike (25%) and overwhelming (50%) thresholds.

### 7.2 Review Model (R)

If review is commissioned, the review produces:

**Outcome rating (trinary: negative, balanced, positive):**

```
(p_neg, p_bal, p_pos) ~ Dirichlet(38, 160, 1)
outcome ~ Categorical(p_neg, p_bal, p_pos)
```

Expected outcome probabilities: E = (0.191, 0.804, 0.005) — balanced dominates at approximately 80%, negative approximately 19%, positive less than 1%. This reflects the empirical pattern of Board-initiated crisis reviews, which typically produce "mistakes were made" conclusions without conceding liability. The `post_review_round` (activating D4_post_review and D_rev_post_review nodes) is triggered only for "negative" outcomes. Source: `background/board/external-review-distributions.md`.

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

The review outcome feeds into utility terms (8a), (8b), (9), (12), and (13).

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

| Parameter                            | Distribution         | Param1 (μ or shape) | Param2 (σ or scale) | Interpretation                                                  |
|--------------------------------------|----------------------|---------------------|---------------------|-----------------------------------------------------------------|
| `inaction_base_penalty`              | Normal(3.0, 1.0)     | 3.0                 | 1.0                 | ASA believes Board's base inaction penalty is \~3.0 ± 1.0       |
| `inaction_no_review_penalty`         | Normal(2.0, 0.5)     | 2.0                 | 0.5                 | ASA believes Board's no-review penalty is \~2.0 ± 0.5           |
| `inaction_ceo_present_penalty`       | Normal(5.0, 1.5)     | 5.0                 | 1.5                 | ASA believes Board's CEO-present penalty is \~5.0 ± 1.5         |
| `inaction_no_sack_penalty`           | Normal(3.0, 1.0)     | 3.0                 | 1.0                 | ASA believes Board's no-sack penalty is \~3.0 ± 1.0             |
| `vote_strike_penalty`               | Normal(2.0, 0.5)     | 2.0                 | 0.5                 | ASA believes Board's strike sensitivity is \~2.0 ± 0.5          |
| `vote_overwhelming_penalty`          | Normal(3.0, 1.0)     | 3.0                 | 1.0                 | ASA believes Board's overwhelming sensitivity is \~3.0 ± 1.0    |
| `ceo_loss_cost`                      | Normal(1.5, 0.5)     | 1.5                 | 0.5                 | ASA believes Board's CEO departure cost is \~1.5 ± 0.5          |
| `ceo_loss_shock_overwhelming`        | Normal(0.5, 0.2)     | 0.5                 | 0.2                 | ASA believes overwhelming shock relief is \~0.5 ± 0.2           |
| `implementation_cost_sack`           | Normal(1.0, 0.3)     | 1.0                 | 0.3                 | ASA believes sacking implementation cost is \~1.0 ± 0.3         |
| `negative_review_finding_penalty`    | Normal(5.0, 2.0)     | 5.0                 | 2.0                 | ASA believes negative-review penalty is \~5.0 ± 2.0             |
| `balanced_review_finding_penalty`    | Normal(2.5, 1.0)     | 2.5                 | 1.0                 | ASA believes balanced-review penalty is \~2.5 ± 1.0             |
| `review_after_removal_penalty`       | Normal(3.0, 1.0)     | 3.0                 | 1.0                 | ASA believes review-after-removal penalty is \~3.0 ± 1.0        |

### 9.2 CEO's Priors on Board Utility Parameters

| Parameter                            | Distribution      | Param1 | Param2 | Interpretation                                                    |
|--------------------------------------|-------------------|--------|--------|-------------------------------------------------------------------|
| `inaction_base_penalty`              | Normal(3.0, 1.0)  | 3.0    | 1.0    | CEO believes Board's base inaction penalty is \~3.0 ± 1.0         |
| `inaction_no_review_penalty`         | Normal(2.0, 0.5)  | 2.0    | 0.5    | CEO believes Board's no-review penalty is \~2.0 ± 0.5             |
| `inaction_ceo_present_penalty`       | Normal(5.0, 1.5)  | 5.0    | 1.5    | CEO believes Board's CEO-present penalty is \~5.0 ± 1.5           |
| `inaction_no_sack_penalty`           | Normal(3.0, 1.0)  | 3.0    | 1.0    | CEO believes Board's no-sack penalty is \~3.0 ± 1.0               |
| `vote_strike_penalty`               | Normal(2.0, 0.5)  | 2.0    | 0.5    | CEO believes Board's strike sensitivity is \~2.0 ± 0.5            |
| `vote_overwhelming_penalty`          | Normal(3.0, 1.0)  | 3.0    | 1.0    | CEO believes Board's overwhelming sensitivity is \~3.0 ± 1.0      |
| `board_passivity_after_departure`    | Normal(0.5, 0.2)  | 0.5    | 0.2    | CEO believes early departure cost is \~0.5 ± 0.2                  |
| `review_car_weight`                  | Normal(15.0, 3.0) | 15.0   | 3.0    | CEO believes Board's CAR sensitivity is \~15.0 ± 3.0              |
| `review_direct_cost_weight`          | Normal(15.0, 3.0) | 15.0   | 3.0    | CEO believes Board's cost sensitivity is \~15.0 ± 3.0             |
| `implementation_cost_sack`           | Normal(1.0, 0.3)  | 1.0    | 0.3    | CEO believes sacking implementation cost is \~1.0 ± 0.3           |
| `ceo_loss_cost`                      | Normal(1.5, 0.5)  | 1.5    | 0.5    | CEO believes departure cost is \~1.5 ± 0.5                        |
| `ceo_loss_shock_overwhelming`        | Normal(0.5, 0.2)  | 0.5    | 0.2    | CEO believes overwhelming shock relief is \~0.5 ± 0.2             |
| `negative_review_finding_penalty`    | Normal(5.0, 2.0)  | 5.0    | 2.0    | CEO believes negative-review penalty is \~5.0 ± 2.0               |
| `balanced_review_finding_penalty`    | Normal(2.5, 1.0)  | 2.5    | 1.0    | CEO believes balanced-review penalty is \~2.5 ± 1.0               |
| `review_after_removal_penalty`       | Normal(3.0, 1.0)  | 3.0    | 1.0    | CEO believes review-after-removal penalty is \~3.0 ± 1.0          |

**Note:** The ASA and CEO priors on Board parameters are centred at the Board's actual parameter values, reflecting the assumption that opponents have approximately correct (but uncertain) beliefs about the Board's preferences. The standard deviations encode the degree of uncertainty about the Board's true preferences.

***

## 10. Summary of Data Sources

| Source Document                                  | Location                                                               | Parameters Informed                                                              |
|--------------------------------------------------|------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| ASX governance review case studies (2014-2023)   | `background/board/governance-review-case-studies.md`                   | `review_car_weight` (15.0), CAR distribution parameters                          |
| Direct costs of governance review                | `background/board/direct-costs-governance-review.md`                   | `review_direct_cost_weight` (15.0), Gamma(4.55, 4741) parameters                |
| External review outcome distributions            | `background/board/external-review-distributions.md`                    | Dirichlet(38, 160, 1) trinary outcome probabilities                              |
| Board-level overconfidence literature review     | `background/board/board-level-overconfidence.md`                       | All overconfidence bias parameters (d1_floor, sigma_scale, etc.)                 |
| Shareholder vote model v2 specification          | `background/shareholders/shareholder-vote-v2.md`                       | Vote model structure, governance effect specification                            |
| External governance review analysis              | `background/board/external-governance-review-analysis.md`              | Review outcome classification, regulatory vs non-regulatory patterns             |
| Corporations Act 2001, ss 250U-250W              | External (Commonwealth legislation)                                    | Inaction penalties (legal grounding for governance obligations)                   |
| Tversky & Kahneman 1992                          | External (academic literature)                                         | `review_car_loss_aversion` (2.25), overconfidence framework                      |
| CFO forecast survey data (28,000+ forecasts)     | Referenced in overconfidence literature                                 | `sigma_scale` (overprecision parameter)                                          |
| Twardawski & Kind 2023, Brahma et al. 2023       | Referenced in overconfidence literature                                 | Mean bias (overestimation) parameters                                            |
| Coffeng et al. 2021                              | Referenced in overconfidence literature                                 | Board decision quality calibration                                               |
| Board utility quantification pipeline            | `board_utility_quantification.py`                                      | Estimated weights for inaction, vote, and review parameters                      |
| Governance specification                         | `data/governance_spec.xlsx`                                            | All 16 parameter values (utilities_board sheet)                                  |
| Opponent priors                                  | `data/opponent_priors.xlsx`                                            | ASA and CEO prior distributions on Board parameters                              |

### Classification of Parameter Basis

| Basis                             | Parameters                                                                                                                                                                                                 |
|-----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Data-derived**                  | `review_car_weight`, `review_direct_cost_weight`                                                                                                                                                           |
| **Estimated (ordinal probit)**    | `inaction_base_penalty`, `inaction_no_review_penalty`, `inaction_ceo_present_penalty`, `inaction_no_sack_penalty`, `vote_strike_penalty`, `vote_overwhelming_penalty`, `negative_review_finding_penalty`, `balanced_review_finding_penalty`, `review_after_removal_penalty` |
| **Subjective**                    | `board_passivity_after_departure`, `implementation_cost_sack`, `ceo_loss_cost`, `ceo_loss_shock_overwhelming`                                                                                              |
