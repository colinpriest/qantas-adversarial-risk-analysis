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
    review_outcome: str = "none"       # "none", "negative", "balanced", "positive"
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
       - board_passivity_after_departure (w1)
       - implementation_cost_sack + ceo_loss_cost (w_removal)
       - ceo_loss_shock_overwhelming (w_remove_ceo_overwhelming)
       - negative_review_finding_penalty (w_review_negative)
       - balanced_review_finding_penalty (w_review_balanced)
       - review_after_removal_penalty (w_review_post_removal)
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

    # Board passivity after CEO departure (w1, graduated by response strength).
    # Penalty for failing to respond when the CEO resigned early.
    # Zero when Board responds decisively (response_strength = 1.0).
    if outcome.CEO_resigned_early:
        u -= params.get("board_passivity_after_departure", 0.5)

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

    # Review finding penalties (w_review_negative, w_review_balanced)
    # Independent indicator-based: negative and balanced each have own penalty.
    # Fires regardless of CEO status — review findings reflect on Board governance.
    if outcome.review_commissioned:
        if outcome.review_outcome == "negative":
            u -= params.get("negative_review_finding_penalty", 5.0)
        elif outcome.review_outcome == "balanced":
            u -= params.get("balanced_review_finding_penalty", 2.5)

    # Due diligence: no review after involuntary CEO removal (w_review_post_removal)
    # Board removed the CEO but did not commission a governance review to address
    # systemic issues or justify the removal decision.
    if removed_involuntary and not outcome.review_commissioned:
        u -= params.get("review_after_removal_penalty", 3.0)

    return u


def utility_asa(outcome: TerminalOutcome, params: dict[str, float]) -> float:
    """
    ASA utility function — 7-dimensional weighted assessment.

    Source: background/asa/asa_bayesian_params.md

    Dimensions (Likert 1–5, weights sum to 1.0):
      FW  (0.10) Financial Welfare — share price, legal exposure
      PPL (0.30) Pay/Performance Linkage — remuneration outcomes, clawback
      TD  (0.10) Transparency/Disclosure — governance disclosure quality
      EGR (0.15) ESG/Governance Risk — regulatory, labour, ESG signals
      BA  (0.20) Board Accountability — consequences imposed on management
      OL  (0.10) Organizational Legitimacy — ASA member trust, credibility
      PF  (0.05) Procedural Fairness — AGM process, share trading norms

    Structure:
    1. Base dimension scores determined by path to A2 (CEO resign/stay × D1 action).
    2. Post-A2 outcome adjustments shift relevant dimension scores.
    3. Weighted sum across all dimensions.
    4. Mobilisation cost deducted if strike recommended.
    """
    # ── 1. Path-dependent base dimension scores ──
    scores = _asa_base_scores(outcome, params)

    # ── 2. Post-A2 outcome adjustments ──

    # Vote: higher vote strengthens accountability signal and ASA standing
    if outcome.strike_indicator:
        scores["BA"] += params.get("strike_ba_shift", 1.5)
        scores["OL"] += params.get("strike_ol_shift", 1.0)
    if outcome.overwhelming_indicator:
        scores["BA"] += params.get("overwhelming_ba_shift", 1.0)
        scores["OL"] += params.get("overwhelming_ol_shift", 0.5)

    # CEO removal post-A2: accountability achieved through board/CEO action
    if outcome.CEO_removed and not outcome.CEO_resigned_early:
        scores["BA"] += params.get("ceo_removal_ba_shift", 1.0)
        scores["FW"] += params.get("ceo_removal_fw_shift", 0.5)

    # Review findings: negative = vindication of governance concerns
    if outcome.review_commissioned:
        if outcome.review_outcome == "negative":
            scores["TD"] += params.get("negative_review_td_shift", 1.0)
            scores["EGR"] += params.get("negative_review_egr_shift", 0.5)
        elif outcome.review_outcome == "balanced":
            scores["TD"] += params.get("balanced_review_td_shift", 0.3)

    # Market alignment: ASA credibility enhanced when recommendation validated
    if outcome.a2_action == "A2_rec_strike" and outcome.strike_indicator:
        scores["OL"] += params.get("market_alignment_ol_shift", 1.0)
        scores["PF"] += params.get("market_alignment_pf_shift", 0.5)

    # Clip all scores to [1, 5] Likert bounds
    for dim in scores:
        scores[dim] = max(1.0, min(5.0, scores[dim]))

    # ── 3. Weighted sum ──
    u = sum(ASA_DIMENSION_WEIGHTS[dim] * scores[dim] for dim in ASA_DIMENSION_WEIGHTS)

    # ── 4. Mobilisation cost ──
    if outcome.a2_action == "A2_rec_strike":
        u -= params.get("mobilisation_cost", 0.3)

    return u


# ── ASA 7-dimensional utility constants ──────────────────────────────────

ASA_DIMENSION_WEIGHTS = {
    "FW": 0.10, "PPL": 0.30, "TD": 0.10,
    "EGR": 0.15, "BA": 0.20, "OL": 0.10, "PF": 0.05,
}

# Base dimension scores per A2 node path (from asa_bayesian_params.md Section 2).
# Key: (CEO_resigned_early, d1_action) → {dimension: μ}
ASA_BASE_SCORE_TABLE = {
    # A2-1: CEO resigns → Do nothing (weighted mean 1.84)
    (True, "D0_minimal"): {
        "FW": 2.1, "PPL": 1.3, "TD": 1.8, "EGR": 1.5,
        "BA": 2.3, "OL": 2.0, "PF": 3.0,
    },
    # A2-2: CEO resigns → Commission review (weighted mean 2.09)
    (True, "D1_review"): {
        "FW": 2.2, "PPL": 1.3, "TD": 2.2, "EGR": 1.9,
        "BA": 2.9, "OL": 2.4, "PF": 3.0,
    },
    # A2-3: CEO stays → Do nothing (weighted mean 1.43)
    (False, "D0_minimal"): {
        "FW": 1.7, "PPL": 1.2, "TD": 1.5, "EGR": 1.3,
        "BA": 1.2, "OL": 1.4, "PF": 2.9,
    },
    # A2-4: CEO stays → Commission review (weighted mean 1.67)
    (False, "D1_review"): {
        "FW": 1.9, "PPL": 1.2, "TD": 2.0, "EGR": 1.6,
        "BA": 1.9, "OL": 1.7, "PF": 2.9,
    },
    # A2-5: CEO stays → Board forces exit (weighted mean 2.27)
    (False, "D3_ceo_transition"): {
        "FW": 2.5, "PPL": 1.6, "TD": 2.3, "EGR": 2.2,
        "BA": 3.3, "OL": 2.7, "PF": 3.0,
    },
}

# Default fallback (A2-3: worst case — CEO stays, Board does nothing)
_ASA_DEFAULT_SCORES = ASA_BASE_SCORE_TABLE[(False, "D0_minimal")]


def _asa_base_scores(outcome: TerminalOutcome, params: dict[str, float]) -> dict[str, float]:
    """Look up base dimension scores for the path to A2."""
    key = (outcome.CEO_resigned_early, outcome.d1_action)
    base = ASA_BASE_SCORE_TABLE.get(key, _ASA_DEFAULT_SCORES)
    return dict(base)  # mutable copy


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
        if outcome.review_commissioned and outcome.review_outcome == "negative":
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


# ── Decomposed utility (for terminal node tooltips) ──────────────────

def _decompose_board(outcome: TerminalOutcome, params: dict[str, float]) -> dict[str, float]:
    """Return Board utility broken into named components."""
    components: dict[str, float] = {}

    ceo_present_at_end = not outcome.CEO_removed and not outcome.CEO_resigned_early
    removed_involuntary = outcome.CEO_removed and not outcome.CEO_resigned_early
    board_inactive = (outcome.d1_action == "D0_minimal")
    if outcome.d_rev_action in ("Drev_sack_ceo", "Drev_commission_review"):
        board_inactive = False
    if outcome.d_rev_post_review_action == "Drev_sack_ceo":
        board_inactive = False

    # Inaction components
    if board_inactive:
        components["inaction_base"] = -params.get("inaction_base_penalty", 3.0)
    if not outcome.review_commissioned:
        components["no_review"] = -params.get("inaction_no_review_penalty", 2.0)
    if ceo_present_at_end:
        components["ceo_present"] = -params.get("inaction_ceo_present_penalty", 5.0)
    if not removed_involuntary:
        components["no_sack"] = -params.get("inaction_no_sack_penalty", 3.0)

    # Vote penalties
    vote_pct = outcome.vote_percent
    if outcome.strike_indicator:
        w_strike = params.get("vote_strike_penalty", 2.0)
        components["vote_strike"] = -w_strike * max(0.0, (vote_pct - 0.25) / 0.75)
    if outcome.overwhelming_indicator:
        w_ow = params.get("vote_overwhelming_penalty", 3.0)
        components["vote_overwhelming"] = -w_ow * max(0.0, (vote_pct - 0.50) / 0.50)

    # Retained
    if outcome.CEO_resigned_early:
        components["board_passivity_after_departure"] = -params.get("board_passivity_after_departure", 0.5)

    if outcome.review_commissioned:
        w_car_anchor = params.get("review_car_weight", 15.0)
        lambda_la = params.get("review_car_loss_aversion", 2.25)
        w_car_pos = w_car_anchor / ((1.0 + lambda_la) / 2.0)
        w_car_neg = lambda_la * w_car_pos
        car_val = w_car_pos * max(outcome.review_car, 0.0) - w_car_neg * max(-outcome.review_car, 0.0)
        if abs(car_val) > 1e-9:
            components["review_car"] = car_val
        direct_cost = -params.get("review_direct_cost_weight", 15.0) * outcome.review_direct_cost
        if abs(direct_cost) > 1e-9:
            components["review_direct_cost"] = direct_cost

    impl_cost = 0.0
    if outcome.d1_action == "D3_ceo_transition":
        impl_cost -= params.get("implementation_cost_sack", 1.0)
    if outcome.d_rev_action == "Drev_sack_ceo":
        impl_cost -= params.get("implementation_cost_sack", 1.0)
    if outcome.d_rev_post_review_action == "Drev_sack_ceo":
        impl_cost -= params.get("implementation_cost_sack", 1.0)
    if abs(impl_cost) > 1e-9:
        components["implementation_cost"] = impl_cost

    if removed_involuntary:
        base_ceo_loss = params.get("ceo_loss_cost", 1.5)
        shock_relief = 0.0
        if outcome.overwhelming_indicator:
            shock_relief += params.get("ceo_loss_shock_overwhelming", 0.5)
        net = -max(0.0, base_ceo_loss - shock_relief)
        if abs(net) > 1e-9:
            components["ceo_loss_cost"] = net

    if outcome.review_commissioned:
        if outcome.review_outcome == "negative":
            components["negative_review_finding"] = -params.get("negative_review_finding_penalty", 5.0)
        elif outcome.review_outcome == "balanced":
            components["balanced_review_finding"] = -params.get("balanced_review_finding_penalty", 2.5)

    if removed_involuntary and not outcome.review_commissioned:
        components["review_after_removal"] = -params.get("review_after_removal_penalty", 3.0)

    return components


def _decompose_asa(outcome: TerminalOutcome, params: dict[str, float]) -> dict[str, float]:
    """Return ASA utility broken into named components (7-dimensional)."""
    components: dict[str, float] = {}

    # Base dimension scores
    scores = _asa_base_scores(outcome, params)
    base_u = sum(ASA_DIMENSION_WEIGHTS[d] * scores[d] for d in ASA_DIMENSION_WEIGHTS)
    components["base_weighted"] = round(base_u, 4)

    # Post-A2 adjustments (tracked as shift contributions)
    adj_scores = dict.fromkeys(ASA_DIMENSION_WEIGHTS, 0.0)

    if outcome.strike_indicator:
        adj_scores["BA"] += params.get("strike_ba_shift", 1.5)
        adj_scores["OL"] += params.get("strike_ol_shift", 1.0)
    if outcome.overwhelming_indicator:
        adj_scores["BA"] += params.get("overwhelming_ba_shift", 1.0)
        adj_scores["OL"] += params.get("overwhelming_ol_shift", 0.5)
    if outcome.CEO_removed and not outcome.CEO_resigned_early:
        adj_scores["BA"] += params.get("ceo_removal_ba_shift", 1.0)
        adj_scores["FW"] += params.get("ceo_removal_fw_shift", 0.5)
    if outcome.review_commissioned:
        if outcome.review_outcome == "negative":
            adj_scores["TD"] += params.get("negative_review_td_shift", 1.0)
            adj_scores["EGR"] += params.get("negative_review_egr_shift", 0.5)
        elif outcome.review_outcome == "balanced":
            adj_scores["TD"] += params.get("balanced_review_td_shift", 0.3)
    if outcome.a2_action == "A2_rec_strike" and outcome.strike_indicator:
        adj_scores["OL"] += params.get("market_alignment_ol_shift", 1.0)
        adj_scores["PF"] += params.get("market_alignment_pf_shift", 0.5)

    # Compute clipped adjustment contribution
    for dim in ASA_DIMENSION_WEIGHTS:
        raw = scores[dim] + adj_scores[dim]
        clipped = max(1.0, min(5.0, raw))
        adj_contribution = ASA_DIMENSION_WEIGHTS[dim] * (clipped - scores[dim])
        if abs(adj_contribution) > 1e-9:
            components[f"adj_{dim}"] = round(adj_contribution, 4)

    if outcome.a2_action == "A2_rec_strike":
        components["mobilisation_cost"] = -params.get("mobilisation_cost", 0.3)

    return components


def _decompose_ceo(outcome: TerminalOutcome, params: dict[str, float]) -> dict[str, float]:
    """Return CEO utility broken into named components (summary level)."""
    components: dict[str, float] = {}

    if outcome.CEO_resigned_early:
        components["W_resign"] = params.get("W_resign", 8.0)
        components["D_resign"] = -params.get("D_resign", 40.0)
    else:
        effective_d4 = outcome.d4_action
        if outcome.d4_post_review_action in ("D4_resign", "D4_negotiate_exit"):
            effective_d4 = outcome.d4_post_review_action

        if not outcome.CEO_removed:
            components["W_stay_kept"] = params.get("W_stay_kept", 7.0)
        elif effective_d4 == "D4_negotiate_exit":
            components["W_negotiate"] = (params.get("W_stay_sacked", 1.5) + params.get("W_stay_kept", 7.0)) / 2.0
        elif effective_d4 == "D4_resign":
            components["W_resign_late"] = params.get("W_stay_sacked", 1.5) * 1.3
        else:
            components["W_sacked"] = params.get("W_stay_sacked", 1.5)

        D_raw = params.get("D_stay", 25.0)
        components["D_stay_base"] = -D_raw
        if outcome.CEO_removed:
            if effective_d4 == "D4_negotiate_exit":
                components["D_negotiate"] = -params.get("D_negotiate", 45.0)
            elif effective_d4 == "D4_resign":
                components["D_resign_late"] = -params.get("D_resign_late", 60.0)
            else:
                components["D_sacked"] = -params.get("D_sacked", 100.0)
        if outcome.vote_percent > 0.25:
            components["D_agm"] = -params.get("D_agm", 30.0)
        if outcome.overwhelming_indicator:
            components["D_disgrace"] = -params.get("D_disgrace", 30.0)
        if outcome.review_commissioned and outcome.review_outcome == "negative":
            components["D_adverse_review"] = -params.get("D_adverse_review", 10.0)

    return components


_DECOMPOSE_FUNCTIONS = {
    "Board": _decompose_board,
    "ASA": _decompose_asa,
    "CEO": _decompose_ceo,
}


def decompose_utility(
    actor: str,
    outcome: TerminalOutcome,
    params: dict[str, float],
) -> dict[str, float]:
    """Return named utility components for any actor at a terminal outcome."""
    fn = _DECOMPOSE_FUNCTIONS.get(actor)
    if fn is None:
        return {}
    return fn(outcome, params)
