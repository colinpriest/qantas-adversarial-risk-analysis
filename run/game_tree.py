"""
Shared game tree computation for the unified ARA run script.

Uses the same tree structure, probabilities, and utility decomposition
as board_utility_quantification.py.  All three actor EU streams (Board,
CEO, ASA) are propagated through every node, enabling any single actor
to be switched to strategic (deterministic) mode.

Default mode: all actors stochastic with Laplacian smoothing.
Strategic mode: one actor's decision nodes become 100% deterministic
(argmax over that actor's EU), no Laplacian.

Tree node order:
    D0_ceo -> D1 -> A2 -> V -> D4 -> D_rev -> [R if review] -> D4_post -> D_rev_post -> Terminal

Board EU = w_draws @ phi + anchored  (posterior-weighted, varies per draw)
CEO EU  = utility_ceo(TerminalOutcome, params)  (scalar per terminal)
ASA EU  = utility_asa(TerminalOutcome, params)  (scalar per terminal)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from engine.utilities import (
    TerminalOutcome, utility_ceo, utility_asa,
    decompose_utility, _decompose_ceo, _decompose_asa,
)
from run.visualise_tree import VizNode

logger = logging.getLogger(__name__)


# ── Anchored constants (from board_utility_quantification.py) ─────────
W_CAR_ANCHOR = 15.0
W_COST_ANCHOR = 15.0
LAMBDA_LA_DEFAULT = 2.25


# ── Parameter names (must match w_draws column order) ─────────────────
ESTIMABLE_PARAM_NAMES = [
    "w_inaction_base", "w_inaction_no_review", "w_inaction_delay",
    "w_passivity", "w_removal", "w_remove_ceo_overwhelming",
    "w_review_negative", "w_review_balanced",
    "w_review_post_removal", "w_ceo_accountability",
]

VOTE_PARAM_NAMES = ["w_strike", "w_overwhelming"]

# Actor names for EU propagation
EU_ACTORS = ("Board", "CEO", "ASA")


# ── Non-Board probabilities ──────────────────────────────────────────
# A2 probabilities loaded from ASA utility quantification pipeline output
# (asa_utility_quantification.py -> outputs/asa/asa_a2_calibration.json).
# These originate from LLM direct probability elicitation, calibrated via
# a logistic model with monotonicity and signaling gap constraints.
# Other actor probabilities from board_utility_quantification.py.


def _load_a2_probs_from_calibration() -> dict:
    """Load A2 action probabilities from the ASA calibration pipeline output.

    Returns dict keyed by path (e.g. 'stayed_D0_minimal') with
    sub-dicts {'A2_no_strike': float, 'A2_rec_strike': float}.
    """
    cal_path = Path(__file__).parent.parent / "outputs" / "asa" / "asa_a2_calibration.json"
    if cal_path.exists():
        import json
        with open(cal_path, "r", encoding="utf-8") as f:
            cal = json.load(f)
        logger.info(f"Loaded A2 calibrated probs from {cal_path}")
        return cal["tree_probs"]
    else:
        logger.warning(
            f"A2 calibration file not found: {cal_path}. "
            "Run asa_utility_quantification.py first. "
            "Using fallback probs (test/development only)."
        )
        # Fallback for tests/development — these match the calibrated output
        # at the time of last pipeline run but should NOT be used in production.
        return {
            "resigned_D0_minimal":        {"A2_no_strike": 0.036, "A2_rec_strike": 0.964},
            "resigned_D1_review":         {"A2_no_strike": 0.090, "A2_rec_strike": 0.910},
            "stayed_D0_minimal":          {"A2_no_strike": 0.014, "A2_rec_strike": 0.986},
            "stayed_D1_review":           {"A2_no_strike": 0.036, "A2_rec_strike": 0.964},
            "stayed_D3_ceo_transition":   {"A2_no_strike": 0.113, "A2_rec_strike": 0.887},
        }


TREE_DEFAULT_PROBS = {
    "D0_ceo": {"CEO_resign": 0.962, "CEO_stay": 0.038},
    "A2": _load_a2_probs_from_calibration(),
    "V": {
        "A2_no_strike":  {"no_strike": 0.55, "first_strike": 0.30, "overwhelming": 0.15},
        "A2_rec_strike": {"no_strike": 0.15, "first_strike": 0.40, "overwhelming": 0.45},
    },
    "D4": {
        "no_strike":    {"D4_stay": 0.95, "D4_resign": 0.03, "D4_negotiate_exit": 0.02},
        "first_strike": {"D4_stay": 0.10, "D4_resign": 0.30, "D4_negotiate_exit": 0.60},
        "overwhelming": {"D4_stay": 0.02, "D4_resign": 0.26, "D4_negotiate_exit": 0.72},
    },
    "R": {"negative": 0.191, "balanced": 0.804, "positive": 0.005},
    "D4_post": {
        "no_strike":    {"D4_stay": 0.05, "D4_resign": 0.40, "D4_negotiate_exit": 0.55},
        "first_strike": {"D4_stay": 0.02, "D4_resign": 0.35, "D4_negotiate_exit": 0.63},
        "overwhelming": {"D4_stay": 0.01, "D4_resign": 0.30, "D4_negotiate_exit": 0.69},
    },
}

VOTE_REPRESENTATIVES = {
    "no_strike": 0.15,
    "first_strike": 0.35,
    "overwhelming": 0.70,
}

REVIEW_REPRESENTATIVES = {
    "negative": {"review_car": -0.05, "review_direct_cost": 0.00096},
    "balanced": {"review_car": -0.01, "review_direct_cost": 0.00096},
    "positive": {"review_car": 0.03, "review_direct_cost": 0.00096},
}


# ── Board utility: phi-based decomposition ────────────────────────────

def decompose_utility_board(
    vote_percent: float,
    strike: bool,
    overwhelming: bool,
    d1_action: str,
    d_rev_action: str,
    d_rev_post_action: str,
    CEO_removed: bool,
    CEO_resigned_early: bool,
    review_commissioned: bool,
    review_outcome: str,
    review_car: float,
    review_direct_cost: float,
) -> dict[str, float]:
    """Decompose Board utility into per-parameter basis function values (phi).

    Returns dict mapping parameter name -> phi value.
    EU = sum_k w_k * phi_k + anchored_contribution.
    """
    removed_involuntary = float(CEO_removed and not CEO_resigned_early)
    board_inactive = (d1_action == "D0_minimal")
    if d_rev_action in ("Drev_sack_ceo", "Drev_commission_review"):
        board_inactive = False
    if d_rev_post_action == "Drev_sack_ceo":
        board_inactive = False

    _D1_STRENGTH = {"D0_minimal": 0.0, "D1_review": 0.5, "D3_ceo_transition": 1.0}
    _DREV_STRENGTH = {"Drev_no_action": 0.0, "Drev_commission_review": 0.5, "Drev_sack_ceo": 1.0}
    response_strength = max(
        _D1_STRENGTH.get(d1_action, 0.0),
        _DREV_STRENGTH.get(d_rev_action, 0.0),
        _DREV_STRENGTH.get(d_rev_post_action, 0.0),
    )

    return {
        "w_inaction_base": -float(board_inactive),
        "w_inaction_no_review": -float(not review_commissioned and not removed_involuntary),
        "w_inaction_delay": -float(
            d1_action == "D0_minimal"
            and (d_rev_action in ("Drev_commission_review", "Drev_sack_ceo")
                 or d_rev_post_action in ("Drev_commission_review", "Drev_sack_ceo"))
        ),
        "w_passivity": -float(CEO_resigned_early) * (1.0 - response_strength),
        "w_removal": -removed_involuntary,
        "w_remove_ceo_overwhelming": removed_involuntary * float(overwhelming),
        "w_review_negative": -float(review_commissioned and review_outcome == "negative"),
        "w_review_balanced": -float(review_commissioned and review_outcome == "balanced"),
        "w_review_post_removal": -float(removed_involuntary and not review_commissioned),
        "w_ceo_accountability": float(removed_involuntary) * float(review_commissioned),
    }


def compute_anchored_contribution(
    vote_percent: float,
    strike: bool,
    overwhelming: bool,
    review_commissioned: bool,
    review_car: float,
    review_direct_cost: float,
    vote_weights: Optional[dict[str, float]] = None,
) -> float:
    """Non-estimable (anchored) contribution to Board utility.

    Includes review CAR + direct cost + vote penalties.
    """
    contrib = 0.0
    if review_commissioned:
        w_car_pos = W_CAR_ANCHOR / ((1 + LAMBDA_LA_DEFAULT) / 2)
        w_car_neg = LAMBDA_LA_DEFAULT * w_car_pos
        contrib += w_car_pos * max(review_car, 0.0) - w_car_neg * max(-review_car, 0.0)
        contrib -= W_COST_ANCHOR * review_direct_cost

    # Vote penalties (estimated, added as point-estimate anchored terms)
    if vote_weights:
        if strike:
            w_s = vote_weights.get("w_strike", 0.8)
            contrib -= w_s * max(0.0, (vote_percent - 0.25) / 0.75)
        if overwhelming:
            w_o = vote_weights.get("w_overwhelming", 0.8087)
            contrib -= w_o * max(0.0, (vote_percent - 0.50) / 0.50)

    return contrib


# ── Tree state helpers ────────────────────────────────────────────────

def tree_apply_action(ts: dict, node: str, action: str) -> dict:
    """Apply action at node, returning a new state dict."""
    ns = dict(ts)
    if node == "D0_ceo":
        if action == "CEO_resign":
            ns["ceo_present"] = False
            ns["CEO_resigned_early"] = True
            ns["CEO_removed"] = True
    elif node == "D1":
        ns["d1_action"] = action
        if action == "D3_ceo_transition":
            ns["ceo_present"] = False
            ns["CEO_removed"] = True
        if action == "D1_review":
            ns["review_commissioned"] = True
    elif node == "A2":
        ns["a2_action"] = action
    elif node == "V":
        vpct = VOTE_REPRESENTATIVES.get(action, 0.15)
        ns["vote_percent"] = vpct
        ns["strike"] = action in ("first_strike", "overwhelming")
        ns["overwhelming"] = action == "overwhelming"
    elif node in ("D4", "D4_post"):
        if action in ("D4_resign", "D4_negotiate_exit"):
            ns["ceo_present"] = False
            ns["CEO_removed"] = True
    elif node in ("D_rev", "D_rev_post"):
        if node == "D_rev":
            ns["d_rev_action"] = action
        else:
            ns["d_rev_post_action"] = action
        if action == "Drev_commission_review":
            ns["review_commissioned"] = True
        elif action == "Drev_sack_ceo":
            ns["ceo_present"] = False
            ns["CEO_removed"] = True
    elif node == "R":
        ns["review_outcome"] = action
        rep = REVIEW_REPRESENTATIVES.get(action, REVIEW_REPRESENTATIVES["balanced"])
        ns["review_car"] = rep["review_car"]
        ns["review_direct_cost"] = rep["review_direct_cost"]
    return ns


def tree_feasible_actions(node: str, ts: dict) -> list[str]:
    """Return feasible actions for a node given current tree state."""
    cp = ts.get("ceo_present", True)
    if node == "D0_ceo":
        return ["CEO_resign", "CEO_stay"]
    elif node == "D1":
        acts = ["D0_minimal", "D1_review"]
        if cp:
            acts.append("D3_ceo_transition")
        return acts
    elif node == "A2":
        return ["A2_no_strike", "A2_rec_strike"]
    elif node == "V":
        return ["no_strike", "first_strike", "overwhelming"]
    elif node in ("D4", "D4_post"):
        if not cp:
            return []
        return ["D4_stay", "D4_resign", "D4_negotiate_exit"]
    elif node == "D_rev":
        acts = ["Drev_no_action"]
        if not ts.get("review_commissioned", False):
            acts.append("Drev_commission_review")
        if cp:
            acts.append("Drev_sack_ceo")
        return acts
    elif node == "R":
        return ["negative", "balanced", "positive"]
    elif node == "D_rev_post":
        acts = ["Drev_no_action"]
        if cp:
            acts.append("Drev_sack_ceo")
        return acts
    return []


def tree_node_type(node: str) -> tuple[str, str]:
    """Return (type, owner) for a node name."""
    types = {
        "D0_ceo": ("decision", "CEO"),
        "D1": ("decision", "Board"),
        "A2": ("decision", "ASA"),
        "V": ("chance", "Nature"),
        "D4": ("decision", "CEO"),
        "D_rev": ("decision", "Board"),
        "R": ("chance", "Nature"),
        "D4_post": ("decision", "CEO"),
        "D_rev_post": ("decision", "Board"),
        "Terminal": ("terminal", "Nature"),
    }
    return types.get(node, ("terminal", "Nature"))


def tree_next_node(node: str) -> str:
    """Return the next node in the game tree sequence."""
    return {
        "D0_ceo": "D1",
        "D1": "A2",
        "A2": "V",
        "V": "D4",
        "D4": "D_rev",
        "D_rev": "Terminal",
        "R": "D4_post",
        "D4_post": "D_rev_post",
        "D_rev_post": "Terminal",
    }.get(node, "Terminal")


def _tree_get_vote_key(ts: dict) -> str:
    """Get vote outcome key for D4 probability lookup."""
    if ts.get("overwhelming"):
        return "overwhelming"
    elif ts.get("strike"):
        return "first_strike"
    return "no_strike"


def tree_get_probs(node: str, ts: dict, probs: dict) -> dict[str, float]:
    """Get action/outcome probabilities for non-strategic decision/chance nodes."""
    if node == "D0_ceo":
        return probs["D0_ceo"]
    elif node == "A2":
        d1a = ts.get("d1_action", "D0_minimal")
        ceo_resigned = ts.get("CEO_resigned_early", False)
        prefix = "resigned" if ceo_resigned else "stayed"
        composite_key = f"{prefix}_{d1a}"
        return probs["A2"].get(composite_key, probs["A2"]["stayed_D0_minimal"])
    elif node == "V":
        a2a = ts.get("a2_action", "A2_rec_strike")
        return probs["V"].get(a2a, probs["V"]["A2_rec_strike"])
    elif node == "D4":
        vk = _tree_get_vote_key(ts)
        return probs["D4"].get(vk, probs["D4"]["first_strike"])
    elif node == "R":
        return probs["R"]
    elif node == "D4_post":
        vk = _tree_get_vote_key(ts)
        return probs["D4_post"].get(vk, probs["D4_post"]["first_strike"])
    return {}


# ── Tree state -> TerminalOutcome conversion ──────────────────────────

def _ts_to_terminal_outcome(ts: dict) -> TerminalOutcome:
    """Convert tree state dict to a TerminalOutcome for CEO/ASA utility."""
    # Determine effective D4 actions
    d4_action = "D4_stay"
    d4_post_action = "D4_stay"
    # These get set when D4/D4_post actions are applied
    # We need to track them through the tree state
    d4_action = ts.get("d4_action", "D4_stay")
    d4_post_action = ts.get("d4_post_action", "D4_stay")

    return TerminalOutcome(
        d1_action=ts.get("d1_action", "D0_minimal"),
        a2_action=ts.get("a2_action", "A2_no_strike"),
        d_rev_action=ts.get("d_rev_action", "Drev_no_action"),
        d4_action=d4_action,
        d4_post_review_action=d4_post_action,
        d_rev_post_review_action=ts.get("d_rev_post_action", "Drev_no_action"),
        vote_percent=ts.get("vote_percent", 0.0),
        strike_indicator=ts.get("strike", False),
        overwhelming_indicator=ts.get("overwhelming", False),
        review_outcome=ts.get("review_outcome", "none") or "none",
        review_car=ts.get("review_car", 0.0),
        review_direct_cost=ts.get("review_direct_cost", 0.00096),
        CEO_removed=ts.get("CEO_removed", False),
        CEO_resigned_early=ts.get("CEO_resigned_early", False),
        review_commissioned=ts.get("review_commissioned", False),
    )


def _ts_to_board_decompose_args(ts: dict) -> dict:
    """Convert tree state to kwargs for decompose_utility_board()."""
    return {
        "vote_percent": ts.get("vote_percent", 0.0),
        "strike": ts.get("strike", False),
        "overwhelming": ts.get("overwhelming", False),
        "d1_action": ts.get("d1_action", "D0_minimal"),
        "d_rev_action": ts.get("d_rev_action", "Drev_no_action"),
        "d_rev_post_action": ts.get("d_rev_post_action", "Drev_no_action"),
        "CEO_removed": ts.get("CEO_removed", False),
        "CEO_resigned_early": ts.get("CEO_resigned_early", False),
        "review_commissioned": ts.get("review_commissioned", False),
        "review_outcome": ts.get("review_outcome") or "none",
        "review_car": ts.get("review_car", 0.0),
        "review_direct_cost": ts.get("review_direct_cost", 0.00096),
    }


def _ts_to_anchored_args(ts: dict) -> dict:
    """Convert tree state to kwargs for compute_anchored_contribution().

    In the recursive tree, R outcomes are expanded as separate branches.
    The loss-averse CAR anchored contribution is therefore excluded to
    avoid double-counting.
    """
    return {
        "vote_percent": ts.get("vote_percent", 0.0),
        "strike": ts.get("strike", False),
        "overwhelming": ts.get("overwhelming", False),
        "review_commissioned": False,
        "review_car": 0.0,
        "review_direct_cost": 0.0,
    }


# ── Extended tree_apply_action that tracks D4 actions ─────────────────

def _tree_apply_action_ext(ts: dict, node: str, action: str) -> dict:
    """Apply action at node, tracking D4/D4_post actions in state."""
    ns = tree_apply_action(ts, node, action)
    if node == "D4":
        ns["d4_action"] = action
    elif node == "D4_post":
        ns["d4_post_action"] = action
    return ns


# ── Recursive tree builder ────────────────────────────────────────────

# Type alias: per-actor EU draws
EUDict = dict  # {"Board": np.ndarray, "CEO": np.ndarray, "ASA": np.ndarray}


@dataclass
class TreeConfig:
    """Configuration for tree computation."""
    w_draws: np.ndarray              # (n_posterior, K) Board weight draws
    vote_weights: dict[str, float]   # {w_strike: float, w_overwhelming: float}
    ceo_params: dict[str, float]     # CEO utility parameters
    asa_params: dict[str, float]     # ASA utility parameters
    probs: dict                      # stochastic actor probabilities
    strategic_actor: Optional[str] = None  # None = all stochastic; "Board"/"ASA"/"CEO"
    param_names: list[str] = field(default_factory=lambda: list(ESTIMABLE_PARAM_NAMES))
    laplacian: bool = True           # Laplacian smoothing on stochastic decision probs


def _route_child(node_name: str, action: str, new_ts: dict) -> str:
    """Determine the next node for a child, handling special routing."""
    # D_rev commission review -> R for findings
    if node_name == "D_rev" and action == "Drev_commission_review":
        return "R"
    # D_rev with review already commissioned at D1: route to R
    if node_name == "D_rev" and new_ts.get("review_commissioned", False):
        return "R"
    # R with negative outcome and CEO present -> D4_post
    if node_name == "R" and action == "negative" and new_ts.get("ceo_present"):
        return "D4_post"
    # R with balanced/positive, or negative with CEO absent -> Terminal
    if node_name == "R":
        return "Terminal"
    return tree_next_node(node_name)


def build_game_tree(
    node_name: str,
    ts: dict,
    cfg: TreeConfig,
    node_id: str = "root",
) -> tuple[VizNode, EUDict]:
    """Recursively build the game tree, returning (VizNode, eu_dict).

    eu_dict maps actor name -> (n_posterior,) array of EU values.
    All three actor EU streams are propagated through every node.
    VizNode.eu always shows Board EU (the posterior-weighted primary metric).
    """
    ntype, owner = tree_node_type(node_name)
    n = cfg.w_draws.shape[0]

    # ── Terminal node ──
    if ntype == "terminal":
        return _build_terminal(ts, cfg, node_id, n)

    feasible = tree_feasible_actions(node_name, ts)

    # Pass-through: D4/D4_post with CEO absent
    if not feasible:
        next_node = tree_next_node(node_name)
        return build_game_tree(next_node, ts, cfg, node_id)

    # ── Strategic decision: 100% to EU-maximizing action ──
    if ntype == "decision" and owner == cfg.strategic_actor:
        return _build_strategic_decision(
            node_name, ts, cfg, node_id, feasible, ntype, owner, n)

    # ── Board stochastic decision: argmax-count with Laplacian ──
    if ntype == "decision" and owner == "Board":
        return _build_board_stochastic(
            node_name, ts, cfg, node_id, feasible, ntype, owner, n)

    # ── All other nodes: fixed probs with Dirichlet ──
    return _build_stochastic_node(
        node_name, ts, cfg, node_id, feasible, ntype, owner, n)


def _build_terminal(
    ts: dict, cfg: TreeConfig, node_id: str, n: int,
) -> tuple[VizNode, EUDict]:
    """Build a terminal node with all three actor utilities."""
    # Board EU via phi decomposition (varies per draw)
    phi_dict = decompose_utility_board(**_ts_to_board_decompose_args(ts))
    anchored_val = compute_anchored_contribution(
        **_ts_to_anchored_args(ts), vote_weights=cfg.vote_weights,
    )
    phi_vec = np.array([phi_dict.get(p, 0.0) for p in cfg.param_names])
    board_eu_draws = cfg.w_draws @ phi_vec + anchored_val
    board_eu = float(np.mean(board_eu_draws))

    # Board utility components: phi * mean(w) for each parameter
    w_means = np.mean(cfg.w_draws, axis=0)
    board_components = {}
    for k, pname in enumerate(cfg.param_names):
        phi_val = phi_dict.get(pname, 0.0)
        if abs(phi_val) > 1e-9:
            board_components[pname] = round(float(phi_val * w_means[k]), 4)
    if abs(anchored_val) > 1e-9:
        board_components["anchored"] = round(float(anchored_val), 4)

    # CEO utility (scalar)
    outcome = _ts_to_terminal_outcome(ts)
    ceo_eu = float(utility_ceo(outcome, cfg.ceo_params))
    ceo_components = _decompose_ceo(outcome, cfg.ceo_params)

    # ASA utility (scalar)
    asa_eu = float(utility_asa(outcome, cfg.asa_params))
    asa_components = _decompose_asa(outcome, cfg.asa_params)

    # Terminal decomposition for mouseover display
    terminal_decomp = {
        "Board_EU": round(board_eu, 4),
        **{f"Board__{k}": v for k, v in board_components.items()},
        "CEO_EU": round(ceo_eu, 4),
        **{f"CEO__{k}": round(float(v), 4) for k, v in ceo_components.items()},
        "ASA_EU": round(asa_eu, 4),
        **{f"ASA__{k}": round(float(v), 4) for k, v in asa_components.items()},
    }

    node = VizNode(
        id=node_id,
        node_name="Terminal",
        node_type="terminal",
        owner="Nature",
        eu=board_eu,
        eu_board=board_eu,
        eu_asa=asa_eu,
        eu_ceo=ceo_eu,
        terminal_decomposition=terminal_decomp,
    )
    eu_dict = {
        "Board": board_eu_draws,
        "CEO": np.full(n, ceo_eu),
        "ASA": np.full(n, asa_eu),
    }
    return node, eu_dict


def _build_strategic_decision(
    node_name: str,
    ts: dict,
    cfg: TreeConfig,
    node_id: str,
    feasible: list[str],
    ntype: str,
    owner: str,
    n: int,
) -> tuple[VizNode, EUDict]:
    """Build a strategic (deterministic) decision node.

    100% allocated to the action that maximizes the strategic actor's
    mean EU.  No Laplacian smoothing.
    """
    child_data = {}  # action -> (VizNode, EUDict)

    for action in feasible:
        new_ts = _tree_apply_action_ext(ts, node_name, action)
        child_id = node_id + "__" + action.lower()
        next_node = _route_child(node_name, action, new_ts)
        child_viz, child_eus = build_game_tree(next_node, new_ts, cfg, child_id)
        child_data[action] = (child_viz, child_eus)

    # Pick the action with highest mean EU for the strategic actor
    mean_eus = {a: float(np.mean(child_data[a][1][owner])) for a in feasible}
    best_action = max(feasible, key=lambda a: mean_eus[a])

    # 100% to best action, 0% to all others
    children = []
    for action in feasible:
        prob = 1.0 if action == best_action else 0.0
        children.append((action, prob, child_data[action][0]))

    # All EU streams take the best action's values (deterministic)
    node_eus = {actor: child_data[best_action][1][actor].copy()
                for actor in EU_ACTORS}

    node = VizNode(
        id=node_id,
        node_name=node_name,
        node_type=ntype,
        owner=owner,
        eu=float(np.mean(node_eus["Board"])),
        eu_board=float(np.mean(node_eus["Board"])),
        eu_asa=float(np.mean(node_eus["ASA"])),
        eu_ceo=float(np.mean(node_eus["CEO"])),
        children=children,
    )
    return node, node_eus


def _build_board_stochastic(
    node_name: str,
    ts: dict,
    cfg: TreeConfig,
    node_id: str,
    feasible: list[str],
    ntype: str,
    owner: str,
    n: int,
) -> tuple[VizNode, EUDict]:
    """Build a Board decision node in stochastic mode.

    Uses argmax-count over Board posterior weight draws for action
    probabilities, with Laplacian smoothing.  All three EU streams
    are propagated as probability-weighted sums.
    """
    child_data = {}  # action -> (VizNode, EUDict)

    for action in feasible:
        new_ts = _tree_apply_action_ext(ts, node_name, action)
        child_id = node_id + "__" + action.lower()
        next_node = _route_child(node_name, action, new_ts)
        child_viz, child_eus = build_game_tree(next_node, new_ts, cfg, child_id)
        child_data[action] = (child_viz, child_eus)

    # Argmax-count from Board EU draws
    board_eu_mat = np.column_stack([child_data[a][1]["Board"] for a in feasible])
    K = len(feasible)
    if K > 1:
        best_idx = np.argmax(board_eu_mat, axis=1)
        alpha = 1.0 if cfg.laplacian else 0.0
        action_probs = {}
        for j, a in enumerate(feasible):
            count = float(np.sum(best_idx == j))
            action_probs[a] = (count + alpha) / (n + K * alpha)
    else:
        action_probs = {feasible[0]: 1.0}

    # Per-draw Dirichlet for epistemic uncertainty propagation
    prob_values = [action_probs.get(a, 1e-6) for a in feasible]
    prob_sum = sum(prob_values)
    CONC_SUM = 20.0
    dir_alpha = np.array([p / prob_sum * CONC_SUM for p in prob_values])
    dir_alpha = np.maximum(dir_alpha, 0.5)
    node_seed = hash(node_id) % (2**31)
    rng_node = np.random.default_rng(node_seed)
    per_draw_probs = rng_node.dirichlet(dir_alpha, size=n)

    # Propagate all three EU streams
    node_eus = {}
    for actor in EU_ACTORS:
        eu_mat = np.column_stack([child_data[a][1][actor] for a in feasible])
        node_eus[actor] = np.sum(per_draw_probs * eu_mat, axis=1)

    children = []
    for action in feasible:
        children.append((action, round(action_probs[action], 4),
                         child_data[action][0]))

    node = VizNode(
        id=node_id,
        node_name=node_name,
        node_type=ntype,
        owner=owner,
        eu=float(np.mean(node_eus["Board"])),
        eu_board=float(np.mean(node_eus["Board"])),
        eu_asa=float(np.mean(node_eus["ASA"])),
        eu_ceo=float(np.mean(node_eus["CEO"])),
        children=children,
    )
    return node, node_eus


def _build_stochastic_node(
    node_name: str,
    ts: dict,
    cfg: TreeConfig,
    node_id: str,
    feasible: list[str],
    ntype: str,
    owner: str,
    n: int,
) -> tuple[VizNode, EUDict]:
    """Build a non-Board decision or chance node in stochastic mode.

    Uses fixed probabilities from TREE_DEFAULT_PROBS with per-draw
    Dirichlet sampling for epistemic uncertainty propagation.
    """
    mean_action_probs = tree_get_probs(node_name, ts, cfg.probs)

    # Per-draw Dirichlet probabilities
    prob_values = [mean_action_probs.get(a, 1e-6) for a in feasible]
    prob_sum = sum(prob_values)

    if node_name == "R" and len(feasible) == 3:
        alpha = np.array([38.0, 160.0, 1.0])
    else:
        CONC_SUM = 20.0
        alpha = np.array([p / prob_sum * CONC_SUM for p in prob_values])
        alpha = np.maximum(alpha, 0.5)

    node_seed = hash(node_id) % (2**31)
    rng_node = np.random.default_rng(node_seed)
    per_draw_probs = rng_node.dirichlet(alpha, size=n)

    node_eus = {actor: np.zeros(n) for actor in EU_ACTORS}
    children = []
    child_eu_by_action = {}  # action -> {actor: array}

    for j, action in enumerate(feasible):
        p_mean = mean_action_probs.get(action, 0.0)
        new_ts = _tree_apply_action_ext(ts, node_name, action)
        child_id = node_id + "__" + action.lower()
        next_node = _route_child(node_name, action, new_ts)
        child_viz, child_eus = build_game_tree(next_node, new_ts, cfg, child_id)
        children.append((action, round(p_mean, 4), child_viz))
        child_eu_by_action[action] = child_eus
        for actor in EU_ACTORS:
            node_eus[actor] += per_draw_probs[:, j] * child_eus[actor]

    # ── A2 diagnostic: per-draw EU comparison across all actors ──
    if node_name == "A2" and len(feasible) >= 2:
        _log_a2_diagnostic(node_id, feasible, child_eu_by_action,
                           mean_action_probs, n)

    node = VizNode(
        id=node_id,
        node_name=node_name,
        node_type=ntype,
        owner=owner,
        eu=float(np.mean(node_eus["Board"])),
        eu_board=float(np.mean(node_eus["Board"])),
        eu_asa=float(np.mean(node_eus["ASA"])),
        eu_ceo=float(np.mean(node_eus["CEO"])),
        children=children,
    )
    return node, node_eus


def _log_a2_diagnostic(
    node_id: str,
    feasible: list[str],
    child_eu_by_action: dict,
    mean_action_probs: dict,
    n: int,
) -> None:
    """Log per-draw diagnostic for A2 nodes: who wins on each draw?"""
    header = f"\n{'='*80}\nA2 DIAGNOSTIC: {node_id}\n{'='*80}"
    lines = [header]
    lines.append(f"Fixed priors: {', '.join(f'{a}={mean_action_probs.get(a,0):.3f}' for a in feasible)}")
    lines.append(f"Posterior draws: n={n}")
    lines.append("")

    for actor in EU_ACTORS:
        eu_arrays = {a: child_eu_by_action[a][actor] for a in feasible}
        means = {a: float(np.mean(eu_arrays[a])) for a in feasible}
        sds = {a: float(np.std(eu_arrays[a])) for a in feasible}
        medians = {a: float(np.median(eu_arrays[a])) for a in feasible}

        # Per-draw argmax count
        eu_mat = np.column_stack([eu_arrays[a] for a in feasible])
        best_idx = np.argmax(eu_mat, axis=1)
        argmax_counts = {a: int(np.sum(best_idx == j)) for j, a in enumerate(feasible)}
        argmax_pcts = {a: argmax_counts[a] / n * 100 for a in feasible}

        lines.append(f"  {actor}:")
        lines.append(f"    {'Action':<25s} {'Mean':>10s} {'SD':>10s} {'Median':>10s} {'ArgmaxPct':>10s}")
        for a in feasible:
            lines.append(f"    {a:<25s} {means[a]:>+10.4f} {sds[a]:>10.4f} {medians[a]:>+10.4f} {argmax_pcts[a]:>9.1f}%")

        # Difference analysis (first vs second action)
        if len(feasible) == 2:
            a0, a1 = feasible
            diff = eu_arrays[a0] - eu_arrays[a1]
            lines.append(f"    Δ({a0} - {a1}): mean={float(np.mean(diff)):+.4f}, "
                         f"median={float(np.median(diff)):+.4f}, "
                         f"sd={float(np.std(diff)):.4f}")
            pct_pos = float(np.mean(diff > 0)) * 100
            lines.append(f"    Pr({a0} > {a1}) = {pct_pos:.1f}%  |  "
                         f"Pr({a1} > {a0}) = {100 - pct_pos:.1f}%")

            # Quintile breakdown: when a0 wins vs loses, how big is the margin?
            wins = diff[diff > 0]
            losses = diff[diff < 0]
            if len(wins) > 0:
                lines.append(f"    When {a0} wins ({len(wins)} draws): "
                             f"mean margin={float(np.mean(wins)):+.4f}, "
                             f"p5={float(np.percentile(wins, 5)):+.4f}, "
                             f"p95={float(np.percentile(wins, 95)):+.4f}")
            if len(losses) > 0:
                lines.append(f"    When {a1} wins ({len(losses)} draws): "
                             f"mean margin={float(np.mean(np.abs(losses))):+.4f}, "
                             f"p5={float(np.percentile(np.abs(losses), 5)):+.4f}, "
                             f"p95={float(np.percentile(np.abs(losses), 95)):+.4f}")
        lines.append("")

    logger.info("\n".join(lines))


# ── Loading helpers ───────────────────────────────────────────────────

def load_posterior_draws(npz_path: str | Path) -> np.ndarray:
    """Load posterior weight draws from stan_posterior_draws.npz.

    Returns w_draws: (n_posterior, K) array of Board weight draws.
    """
    data = np.load(npz_path)
    w_draws = data["w_draws"]
    logger.info(f"Loaded {w_draws.shape[0]} posterior draws, {w_draws.shape[1]} parameters")
    return w_draws


def load_vote_weights(csv_path: str | Path) -> dict[str, float]:
    """Load vote penalty point estimates from parameter_estimates.csv."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    est = dict(zip(df["parameter"], df["estimate"]))
    return {
        "w_strike": est.get("w_strike", 0.8),
        "w_overwhelming": est.get("w_overwhelming", 0.8087),
    }


def load_actor_params_from_spec(
    governance_spec_path: str | Path,
) -> tuple[dict[str, float], dict[str, float]]:
    """Load CEO and ASA utility parameters from governance_spec.xlsx.

    Returns (ceo_params, asa_params).
    """
    import pandas as pd

    ceo_params = {}
    asa_params = {}

    try:
        # CEO parameters from utilities_ceo sheet
        df_ceo = pd.read_excel(governance_spec_path, sheet_name="utilities_ceo")
        for _, row in df_ceo.iterrows():
            name = str(row.get("parameter_name", row.get("parameter", ""))).strip()
            val = row.get("value", None)
            if name and val is not None and not np.isnan(float(val)):
                ceo_params[name] = float(val)
    except Exception as e:
        logger.warning(f"Could not load CEO params from spec: {e}")

    try:
        # ASA parameters from utilities_asa sheet
        df_asa = pd.read_excel(governance_spec_path, sheet_name="utilities_asa")
        for _, row in df_asa.iterrows():
            name = str(row.get("parameter_name", row.get("parameter", ""))).strip()
            val = row.get("value", None)
            if name and val is not None and not np.isnan(float(val)):
                asa_params[name] = float(val)
    except Exception as e:
        logger.warning(f"Could not load ASA params from spec: {e}")

    return ceo_params, asa_params


def make_initial_state() -> dict:
    """Create the initial tree state at D0_ceo."""
    return {
        "ceo_present": True,
        "CEO_resigned_early": False,
        "CEO_removed": False,
        "review_commissioned": False,
        "review_outcome": "none",
        "vote_percent": 0.0,
        "strike": False,
        "overwhelming": False,
        "d1_action": "D0_minimal",
        "a2_action": "A2_no_strike",
        "d_rev_action": "Drev_no_action",
        "d_rev_post_action": "Drev_no_action",
        "d4_action": "D4_stay",
        "d4_post_action": "D4_stay",
        "review_car": 0.0,
        "review_direct_cost": 0.00096,
    }
