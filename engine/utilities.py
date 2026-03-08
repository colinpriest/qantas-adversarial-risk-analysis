"""
Utility functions for Board, ASA, and CEO.

Each utility function takes a terminal outcome and actor-specific parameters,
returning a scalar utility value. Parameters are loaded from governance_spec.xlsx
or sampled from opponent priors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TerminalOutcome:
    """Complete description of a terminal game outcome."""
    # Actions taken
    d1_action: str = "D0_minimal"
    a2_action: str = "A2_no_strike"
    d_rev_action: str = "Drev_no_action"
    d4_action: str = "D4_stay"
    d4_post_review_action: str = "D4_stay"
    d_rev_post_review_action: str = "Drev_no_action"

    # Chance outcomes
    vote_percent: float = 0.0
    strike_indicator: bool = False
    overwhelming_indicator: bool = False
    review_adverse: bool = False       # Derived: review_car < 0
    review_car: float = 0.0           # Abnormal return from findings release
    review_direct_cost: float = 0.0   # Direct cost of review (decimal CAR, positive)

    # Final state
    CEO_removed: bool = False
    CEO_resigned_early: bool = False  # CEO resigned before game tree (pre-game scenario)
    review_commissioned: bool = False


def utility_board(outcome: TerminalOutcome, params: dict[str, float]) -> float:
    """
    Board utility function.

    Structure (matches quantification pipeline decomposition):

    1. INACTION COMPONENTS (4 additive, unconditional — fire regardless of vote):
       - inaction_base_penalty:       Board took minimal action at all points
       - inaction_no_review_penalty:  No governance review commissioned
       - inaction_ceo_present_penalty: CEO still present at terminal
       - inaction_no_sack_penalty:    Board didn't explicitly remove CEO

    2. VOTE PENALTIES (scenario-level, linear in vote excess):
       - vote_strike_penalty:       w × max(0, (V-0.25)/0.75)
       - vote_overwhelming_penalty: w × max(0, (V-0.50)/0.50)

    3. RETAINED:
       - early_ceo_departure_cost (w1)
       - implementation_cost_sack + ceo_loss_cost (w_removal)
       - ceo_loss_shock_overwhelming (w_remove_ceo_overwhelming)
       - adverse_review_ceo_present_penalty (w15)
       - review_car_weight, review_direct_cost_weight (anchored)
    """
    u = 0.0

    # ── Derived indicators ──
    ceo_present_at_end = not outcome.CEO_removed and not outcome.CEO_resigned_early
    removed_involuntary = outcome.CEO_removed and not outcome.CEO_resigned_early

    board_inactive = (outcome.d1_action == "D0_minimal")
    if outcome.d_rev_action in ("Drev_sack_ceo", "Drev_commission_review"):
        board_inactive = False
    if outcome.d_rev_post_review_action == "Drev_sack_ceo":
        board_inactive = False

    # ── 1. INACTION COMPONENTS (unconditional — fire regardless of vote level) ──
    if board_inactive:
        u -= params.get("inaction_base_penalty", 3.0)
    if not outcome.review_commissioned:
        u -= params.get("inaction_no_review_penalty", 2.0)
    if ceo_present_at_end:
        u -= params.get("inaction_ceo_present_penalty", 5.0)
    if not removed_involuntary:
        u -= params.get("inaction_no_sack_penalty", 3.0)

    # ── 2. VOTE PENALTIES (scenario-level, linear in vote excess) ──
    vote_pct = outcome.vote_percent
    if outcome.strike_indicator:
        w_strike = params.get("vote_strike_penalty", 2.0)
        u -= w_strike * max(0.0, (vote_pct - 0.25) / 0.75)
    if outcome.overwhelming_indicator:
        w_overwhelming = params.get("vote_overwhelming_penalty", 3.0)
        u -= w_overwhelming * max(0.0, (vote_pct - 0.50) / 0.50)

    # ── 3. RETAINED COMPONENTS ──

    # Early CEO resignation: reduced disruption cost (w1, graduated by response)
    if outcome.CEO_resigned_early:
        u -= params.get("early_ceo_departure_cost", 0.5)

    # Review findings CAR impact with loss aversion (anchored):
    if outcome.review_commissioned:
        w_car_anchor = params.get("review_car_weight", 15.0)
        lambda_la = params.get("review_car_loss_aversion", 2.25)
        w_car_pos = w_car_anchor / ((1.0 + lambda_la) / 2.0)
        w_car_neg = lambda_la * w_car_pos
        u += w_car_pos * max(outcome.review_car, 0.0)
        u -= w_car_neg * max(-outcome.review_car, 0.0)

    # Review direct cost (anchored):
    if outcome.review_commissioned:
        u -= params.get("review_direct_cost_weight", 15.0) * outcome.review_direct_cost

    # CEO removal cost (w_removal = implementation_cost_sack + ceo_loss_cost)
    if outcome.d1_action == "D3_ceo_transition":
        u -= params.get("implementation_cost_sack", 1.0)
    if outcome.d_rev_action == "Drev_sack_ceo":
        u -= params.get("implementation_cost_sack", 1.0)
    if outcome.d_rev_post_review_action == "Drev_sack_ceo":
        u -= params.get("implementation_cost_sack", 1.0)

    # CEO loss cost with shock attenuation (w_remove_ceo_overwhelming for overwhelming relief)
    if removed_involuntary:
        base_ceo_loss = params.get("ceo_loss_cost", 1.5)
        shock_relief = 0.0
        if outcome.overwhelming_indicator:
            shock_relief += params.get("ceo_loss_shock_overwhelming", 0.5)
        u -= max(0.0, base_ceo_loss - shock_relief)

    # Adverse review + CEO present penalty (w15)
    if (outcome.review_commissioned and outcome.review_adverse
            and ceo_present_at_end):
        u -= params.get("adverse_review_ceo_present_penalty", 5.0)

    return u


def utility_asa(outcome: TerminalOutcome, params: dict[str, float]) -> float:
    """
    ASA utility function.

    The ASA wants to maximise:
    - High opposition vote (signal of shareholder dissatisfaction)
    - CEO removal (governance change)
    - Adverse review findings (vindication)
    While managing mobilisation costs.
    """
    u = 0.0

    # Early CEO resignation: partial vindication reward
    if outcome.CEO_resigned_early:
        u += params.get("early_ceo_departure_reward", 2.0)

    # Vote reward: linear in vote percent
    u += params.get("vote_reward_weight", 2.0) * outcome.vote_percent

    # Overwhelming vote bonus
    if outcome.overwhelming_indicator:
        u += params.get("overwhelming_reward_weight", 2.0)

    # CEO removal reward (not applied for early resignation — handled above)
    if outcome.CEO_removed and not outcome.CEO_resigned_early:
        u += params.get("ceo_removal_reward", 3.0)

    # Review findings CAR impact: ASA benefits from negative CAR
    # (adverse findings vindicate governance concerns).
    if outcome.review_commissioned:
        u -= params.get("review_car_weight", 15.0) * outcome.review_car

    # Mobilisation cost (only if strike recommended)
    if outcome.a2_action == "A2_rec_strike":
        u -= params.get("mobilisation_cost", 1.0)

    # Reputational gain from high-profile campaign
    if outcome.vote_percent > 0.25:
        u += params.get("reputational_gain_weight", 1.0) * (outcome.vote_percent - 0.25)

    # Market alignment bonus: ASA's credibility and influence as a governance
    # advocate is enhanced when its recommendation aligns with mainstream
    # institutional investor behaviour. Empirically, in 100% of headline-
    # incident cases (ranked_voting_recommendations.csv, headline_incident=1,
    # Qantas excluded), the market votes a first strike. ASA gains standing
    # when it leads the consensus — recommending strike and being validated
    # by the actual vote outcome. Deviation (staying silent while the market
    # acts, or recommending when the market does not follow) erodes credibility.
    if outcome.a2_action == "A2_rec_strike" and outcome.strike_indicator:
        u += params.get("market_alignment_bonus", 1.5)

    return u


def utility_ceo(outcome: TerminalOutcome, params: dict[str, float]) -> float:
    """
    CEO utility function — reference-dependent CRRA with loss aversion.

    U_total = U_money(W) − λ_D · D

    Monetary utility uses Kahneman–Tversky loss aversion around a reference
    point W_ref (pre-crisis expected compensation):

      W ≥ W_ref:  U_money = CRRA(W)                                   [standard]
      W <  W_ref: U_money = λ·CRRA(W) − (λ−1)·CRRA(W_ref)   [losses amplified by λ]

    where CRRA(W) = W^(1−γ)/(1−γ) (γ ≠ 1) or ln(W) (γ = 1), and λ ≈ 2.25
    per Tversky & Kahneman (1992) cumulative prospect theory estimates.

    Non-monetary penalties (D) are also scaled by loss aversion (λ_D):
    an executive with a large ego evaluates reputational losses and career
    destruction relative to their expected status as a powerful CEO, making
    sacking, public disgrace, and AGM humiliation feel disproportionately
    worse.  λ_D defaults to λ (same coefficient) but can be set independently.

    Resign path: W = W_resign (partial bonus retained), D = D_resign
                 (moderate stigma — framed as "taking responsibility").
    Stay path:   W depends on departure mode (sacked/negotiated/kept).
                 W values are calibrated for ACCC-era pay erosion
                 (frozen LTIs, reserved clawbacks, legal costs).
                 D starts at D_stay (baseline crisis cost) plus additive
                 conditional penalties from game outcomes.

    Departure-mode non-monetary penalties (additive to D_stay):
      D_sacked       = 100  Board fires CEO (maximum reputational destruction)
      D_resign_late  =  60  CEO voluntarily resigns mid-game at D4 (controls
                            narrative but post-AGM, too late for graceful exit)
      D_negotiate    =  45  CEO negotiates exit at D4 (face-saving terms,
                            close to pre-game D_resign=40 but mid-crisis)
    """
    gamma = max(0.5, min(3.0, params.get("gamma", 1.5)))
    loss_aversion = params.get("loss_aversion", 2.25)
    W_ref = params.get("W_ref", 16.0)

    # Loss aversion on non-monetary penalties: defaults to same λ
    loss_aversion_D = params.get("loss_aversion_D", loss_aversion)

    # --- Resign path (D0_ceo = CEO_resign) ---
    if outcome.CEO_resigned_early:
        W = params.get("W_resign", 8.0)
        D_raw = params.get("D_resign", 40.0)
    else:
        # --- Stay path: W depends on departure mode ---
        # W values reflect ACCC-era pay erosion: Board had flagged clawback
        # of up to A$14.4M, LTIs frozen, STI under review.
        # Use last non-stay D4 action (post-review overrides if CEO acted there)
        effective_d4 = outcome.d4_action
        if outcome.d4_post_review_action in ("D4_resign", "D4_negotiate_exit"):
            effective_d4 = outcome.d4_post_review_action

        if not outcome.CEO_removed:
            W = params.get("W_stay_kept", 7.0)
        elif effective_d4 == "D4_negotiate_exit":
            W = (params.get("W_stay_sacked", 1.5) +
                 params.get("W_stay_kept", 7.0)) / 2.0
        elif effective_d4 == "D4_resign":
            W = params.get("W_stay_sacked", 1.5) * 1.3
        else:
            W = params.get("W_stay_sacked", 1.5)

        # --- Stay path: D starts at baseline crisis cost ---
        D_raw = params.get("D_stay", 25.0)
        if outcome.CEO_removed:
            # Departure-mode-dependent non-monetary penalty:
            #   Sacked by Board:         D_sacked (100) — maximum reputational damage
            #   Voluntary resign at D4:  D_resign_late (60) — post-AGM, controls narrative
            #   Negotiated exit at D4:   D_negotiate (45) — face-saving terms
            if effective_d4 == "D4_negotiate_exit":
                D_raw += params.get("D_negotiate", 45.0)
            elif effective_d4 == "D4_resign":
                D_raw += params.get("D_resign_late", 60.0)
            else:
                D_raw += params.get("D_sacked", 100.0)
        if outcome.vote_percent > 0.25:
            D_raw += params.get("D_agm", 30.0)
        if outcome.overwhelming_indicator:
            D_raw += params.get("D_disgrace", 30.0)
        if outcome.review_commissioned and outcome.review_adverse:
            D_raw += params.get("D_adverse_review", 10.0)

    # Reference-dependent CRRA with loss aversion on W
    W = max(W, 0.01)
    W_ref = max(W_ref, 0.01)

    if abs(gamma - 1.0) < 1e-6:
        crra_W = np.log(W)
        crra_ref = np.log(W_ref)
    else:
        crra_W = (W ** (1.0 - gamma)) / (1.0 - gamma)
        crra_ref = (W_ref ** (1.0 - gamma)) / (1.0 - gamma)

    if W >= W_ref:
        U_money = crra_W
    else:
        U_money = loss_aversion * crra_W - (loss_aversion - 1.0) * crra_ref

    # Loss aversion on non-monetary penalties: the CEO evaluates
    # reputational/career destruction relative to expected status as a
    # powerful, high-profile executive.  Being fired feels λ_D times worse
    # than the raw D value.
    return U_money - loss_aversion_D * D_raw


# Dispatch map for convenience
UTILITY_FUNCTIONS = {
    "Board": utility_board,
    "ASA": utility_asa,
    "CEO": utility_ceo,
}


def compute_utility(
    actor: str,
    outcome: TerminalOutcome,
    params: dict[str, float],
) -> float:
    """Compute utility for any actor."""
    fn = UTILITY_FUNCTIONS.get(actor)
    if fn is None:
        raise ValueError(f"Unknown actor: {actor}")
    return fn(outcome, params)
