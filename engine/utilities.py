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

    The Board wants to minimise:
    - High opposition votes (reputational damage, spill risk)
    - Adverse review findings
    - CEO disruption costs
    While managing implementation costs of governance reforms.
    """
    u = 0.0

    # Early CEO resignation: reduced disruption cost (voluntary, pre-game)
    if outcome.CEO_resigned_early:
        u -= params.get("early_ceo_departure_cost", 0.5)

    # Vote penalty: quadratic in vote percent above first strike
    vote_pct = outcome.vote_percent
    if vote_pct > 0.25:
        u -= params.get("vote_penalty_weight", 2.0) * (vote_pct - 0.25) ** 2
    if outcome.overwhelming_indicator:
        u -= params.get("overwhelming_penalty_weight", 3.0)
    if outcome.strike_indicator:
        u -= params.get("spill_risk_weight", 2.5) * vote_pct

    # Review findings CAR impact: continuous abnormal return from
    # the findings release (Student-t model). Negative CAR hurts Board
    # (shareholder value destroyed), positive CAR helps (governance validated).
    if outcome.review_commissioned:
        u += params.get("review_car_weight", 15.0) * outcome.review_car

    # Review direct cost: stochastic Gamma(4.55, 4741) in decimal CAR,
    # covering reviewer fees, management distraction, internal resources.
    # Calibrated from board-background/direct-costs-governance-review.md.
    # Mean ≈ 9.6 bps, expressed as positive decimal CAR (subtracted here).
    if outcome.review_commissioned:
        u -= params.get("review_direct_cost_weight", 15.0) * outcome.review_direct_cost

    # Implementation costs (CEO transition only — review cost now stochastic above)
    if outcome.d1_action == "D3_ceo_transition":
        u -= params.get("implementation_cost_sack", 1.0)

    if outcome.d_rev_action == "Drev_sack_ceo":
        u -= params.get("implementation_cost_sack", 1.0)

    # CEO loss cost (disruption from losing CEO — not applied for early
    # resignation, which has its own cost above)
    if outcome.CEO_removed and not outcome.CEO_resigned_early:
        u -= params.get("ceo_loss_cost", 1.5)

    # Reputational spill cost
    if outcome.overwhelming_indicator:
        u -= params.get("reputational_spill_weight", 1.0)

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

    return u


def utility_ceo(outcome: TerminalOutcome, params: dict[str, float]) -> float:
    """
    CEO utility function — CRRA over wealth + additive non-monetary penalty.

    U_total = U_money(W) − D

    where U_money(W) = W^(1−γ)/(1−γ)  (CRRA, γ ≠ 1)  or  ln(W)  (γ = 1).

    Resign path: W = W_resign (partial bonus retained), D = D_resign (moderate stigma).
    Stay path:   W depends on departure mode (sacked/negotiated/kept),
                 D is additive from game outcomes (AGM, disgrace, sacking, review).
    """
    gamma = max(0.5, min(3.0, params.get("gamma", 1.5)))

    # --- Resign path (D0_ceo = CEO_resign) ---
    if outcome.CEO_resigned_early:
        W = params.get("W_resign", 8.0)
        D = params.get("D_resign", 50.0)
    else:
        # --- Stay path: W depends on departure mode ---
        if not outcome.CEO_removed:
            W = params.get("W_stay_kept", 12.0)
        elif outcome.d4_action == "D4_negotiate_exit":
            # Negotiated exit: between sacked and kept
            W = (params.get("W_stay_sacked", 3.0) +
                 params.get("W_stay_kept", 12.0)) / 2.0
        elif outcome.d4_action == "D4_resign":
            # Late resignation under pressure: slightly better than sacked
            W = params.get("W_stay_sacked", 3.0) * 1.3
        else:
            # Forced removal (sacked by board)
            W = params.get("W_stay_sacked", 3.0)

        # --- Stay path: D is additive from outcome components ---
        D = 0.0
        if outcome.CEO_removed:
            D += params.get("D_sacked", 100.0)
        if outcome.vote_percent > 0.25:
            D += params.get("D_agm", 30.0)
        if outcome.overwhelming_indicator:
            D += params.get("D_disgrace", 30.0)
        if outcome.review_commissioned and outcome.review_adverse:
            D += params.get("D_adverse_review", 10.0)

    # CRRA monetary utility
    W = max(W, 0.01)
    if abs(gamma - 1.0) < 1e-6:
        U_money = np.log(W)
    else:
        U_money = (W ** (1.0 - gamma)) / (1.0 - gamma)

    return U_money - D


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
