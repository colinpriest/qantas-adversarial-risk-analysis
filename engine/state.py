"""
Core data structures: DecisionState, BeliefBundle, ParameterSampler.

DecisionState tracks the game state (CEO present, review status, etc.)
and enforces feasibility rules from governance_spec.xlsx.

BeliefBundle loads and indexes checkpoint posterior draws.

ParameterSampler draws opponent utility parameters from prior distributions.
"""

from __future__ import annotations

import json
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Feasibility rule evaluators
# ---------------------------------------------------------------------------

FEASIBILITY_RULES = {
    "always": lambda s: True,
    "CEO_present": lambda s: s.CEO_present,
    "CEO_not_removed": lambda s: not s.CEO_removed,
    "review_not_commissioned": lambda s: not s.review_commissioned,
    "review_commissioned": lambda s: s.review_commissioned,
    "review_completed": lambda s: s.review_completed,
    "not_reviewed_yet": lambda s: not s.review_commissioned and not s.review_completed,
    # Phase 6: conditional post-review round (review_adverse AND CEO was present at review)
    "post_review_round": lambda s: s.post_review_round,
    "post_review_round_and_CEO_present": lambda s: s.post_review_round and s.CEO_present,
    "post_review_round_and_review_not_commissioned": lambda s: s.post_review_round and not s.review_commissioned,
}


# ---------------------------------------------------------------------------
# DecisionState
# ---------------------------------------------------------------------------

@dataclass
class DecisionState:
    """Tracks mutable game state through the decision tree."""

    CEO_present: bool = True
    review_commissioned: bool = False
    review_completed: bool = False
    CEO_removed: bool = False
    CEO_resigned_early: bool = False  # CEO resigned before game tree (pre-game scenario)
    review_adverse: bool = False      # Review findings were adverse (CAR < 0)
    post_review_round: bool = False   # Phase 6 active: review adverse AND CEO present at review
    headline_incident: bool = True    # Headline governance incident (activates gamma_AH + crisis floor)
    checkpoint_id: str = "C0"

    # Loaded from governance_spec - maps (node_name, action_name) -> feasibility_code
    _action_rules: dict = field(default_factory=dict, repr=False)

    # Node order metadata
    _node_order: list = field(default_factory=list, repr=False)
    _node_owners: dict = field(default_factory=dict, repr=False)
    _node_types: dict = field(default_factory=dict, repr=False)

    def feasible_actions(self, node_name: str) -> list[str]:
        """Return list of feasible action names for the given node."""
        actions = []
        for (node, action), code in self._action_rules.items():
            if node != node_name:
                continue
            rule = FEASIBILITY_RULES.get(code)
            if rule is None:
                raise ValueError(f"Unknown feasibility code: {code!r}")
            if rule(self):
                actions.append(action)
        return actions

    def apply(self, node_name: str, action: str) -> DecisionState:
        """Return a new DecisionState with the action applied."""
        new = copy.copy(self)
        new._action_rules = self._action_rules
        new._node_order = self._node_order
        new._node_owners = self._node_owners
        new._node_types = self._node_types

        # Apply state transitions based on action
        if action == "D3_ceo_transition":
            new.CEO_present = False
            new.CEO_removed = True
        elif action == "D1_review":
            new.review_commissioned = True
        elif action == "D0_minimal":
            pass  # Status quo

        elif action == "Drev_commission_review":
            new.review_commissioned = True
        elif action == "Drev_sack_ceo":
            new.CEO_present = False
            new.CEO_removed = True

        elif action == "D4_resign":
            new.CEO_present = False
            new.CEO_removed = True
        elif action == "D4_negotiate_exit":
            new.CEO_present = False
            new.CEO_removed = True

        # D0_ceo actions
        elif action == "CEO_resign":
            new.CEO_present = False
            new.CEO_removed = True
            new.CEO_resigned_early = True
        elif action == "CEO_stay":
            pass  # CEO stays — default state

        # Review outcome transitions (set by R chance node)
        elif action == "adverse":
            new.review_adverse = True
            new.review_completed = True
            if new.CEO_present:
                new.post_review_round = True
        elif action == "no_adverse":
            new.review_adverse = False
            new.review_completed = True

        return new

    def next_node(self, current_node: str) -> Optional[str]:
        """Return the next node in the tree order, or None if terminal."""
        try:
            idx = self._node_order.index(current_node)
        except ValueError:
            return None
        if idx + 1 < len(self._node_order):
            return self._node_order[idx + 1]
        return None

    def node_owner(self, node_name: str) -> str:
        return self._node_owners.get(node_name, "Nature")

    def node_type(self, node_name: str) -> str:
        return self._node_types.get(node_name, "terminal")

    def is_terminal(self, node_name: str) -> bool:
        return self.node_type(node_name) == "terminal"

    def for_scenario(self, scenario: str) -> DecisionState:
        """Configure state for a pre-game CEO resignation scenario.

        Delegates to apply("D0_ceo", action) — D0_ceo is a real decision node
        in the game tree owned by the CEO.

        Args:
            scenario: "ceo_resigned" or "ceo_stayed".
                - ceo_resigned: applies CEO_resign at D0_ceo.
                - ceo_stayed: applies CEO_stay at D0_ceo (no-op).
        """
        action_map = {
            "ceo_resigned": "CEO_resign",
            "ceo_stayed": "CEO_stay",
        }
        if scenario not in action_map:
            raise ValueError(f"Unknown scenario: {scenario!r}")
        return self.apply("D0_ceo", action_map[scenario])

    def to_init_dict(self) -> dict:
        """Serialize initialization metadata for cross-process transfer."""
        return {
            "action_rules": dict(self._action_rules),
            "node_order": list(self._node_order),
            "node_owners": dict(self._node_owners),
            "node_types": dict(self._node_types),
        }

    @classmethod
    def from_init_dict(cls, data: dict, checkpoint_id: str = "C0") -> "DecisionState":
        """Reconstruct from pre-loaded data without file I/O."""
        state = cls(checkpoint_id=checkpoint_id)
        state._action_rules = data["action_rules"]
        state._node_order = data["node_order"]
        state._node_owners = data["node_owners"]
        state._node_types = data["node_types"]
        return state

    @classmethod
    def from_governance_spec(cls, path: str | Path, checkpoint_id: str = "C0") -> DecisionState:
        """Load DecisionState from governance_spec.xlsx."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Governance spec not found: {path}")

        # Load node order
        node_order_df = pd.read_excel(path, sheet_name="node_order")
        _validate_node_order(node_order_df)
        node_order_df = node_order_df.sort_values("order_index")

        node_order = node_order_df["node_name"].tolist()
        node_owners = dict(zip(node_order_df["node_name"], node_order_df["owner"]))
        node_types = dict(zip(node_order_df["node_name"], node_order_df["node_type"]))

        # Load action sets
        action_sets_df = pd.read_excel(path, sheet_name="action_sets")
        _validate_action_sets(action_sets_df, node_order_df)
        action_rules = {}
        for _, row in action_sets_df.iterrows():
            action_rules[(row["node_name"], row["action_name"])] = row["feasibility_code"]

        state = cls(checkpoint_id=checkpoint_id)
        state._action_rules = action_rules
        state._node_order = node_order
        state._node_owners = node_owners
        state._node_types = node_types
        return state


def _validate_node_order(df: pd.DataFrame) -> None:
    """Validate node_order sheet."""
    required_cols = {"order_index", "node_name", "node_type", "owner"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"node_order missing columns: {missing}")

    if df["order_index"].duplicated().any():
        raise ValueError("order_index must be unique")

    if not df["order_index"].is_monotonic_increasing:
        df_sorted = df.sort_values("order_index")
        if not df_sorted["order_index"].is_monotonic_increasing:
            raise ValueError("order_index must be strictly increasing")

    valid_types = {"decision", "chance", "terminal"}
    invalid = set(df["node_type"]) - valid_types
    if invalid:
        raise ValueError(f"Invalid node_type values: {invalid}")

    terminal_count = (df["node_name"] == "Terminal").sum()
    if terminal_count != 1:
        raise ValueError(f"Terminal must appear exactly once, found {terminal_count}")

    # V and R must have owner = Nature
    for node in ["V", "R"]:
        if node in df["node_name"].values:
            owner = df.loc[df["node_name"] == node, "owner"].iloc[0]
            if owner != "Nature":
                raise ValueError(f"Node {node} must have owner=Nature, got {owner}")


def _validate_action_sets(action_df: pd.DataFrame, node_order_df: pd.DataFrame) -> None:
    """Validate action_sets sheet."""
    required_cols = {"node_name", "action_name", "feasibility_code"}
    missing = required_cols - set(action_df.columns)
    if missing:
        raise ValueError(f"action_sets missing columns: {missing}")

    decision_nodes = set(node_order_df.loc[
        node_order_df["node_type"] == "decision", "node_name"
    ])
    action_nodes = set(action_df["node_name"])
    invalid_nodes = action_nodes - decision_nodes
    if invalid_nodes:
        raise ValueError(f"action_sets references non-decision nodes: {invalid_nodes}")

    valid_codes = set(FEASIBILITY_RULES.keys())
    invalid_codes = set(action_df["feasibility_code"]) - valid_codes
    if invalid_codes:
        raise ValueError(f"Invalid feasibility codes: {invalid_codes}")


# ---------------------------------------------------------------------------
# Vote thresholds loader
# ---------------------------------------------------------------------------

def load_vote_thresholds(path: str | Path) -> dict[str, float]:
    """Load vote thresholds from governance_spec.xlsx."""
    df = pd.read_excel(path, sheet_name="vote_thresholds")
    thresholds = dict(zip(df["threshold_name"], df["value"]))

    required = {"first_strike", "overwhelming"}
    missing = required - set(thresholds.keys())
    if missing:
        raise ValueError(f"Missing vote thresholds: {missing}")

    if not (0 < thresholds["first_strike"] < thresholds["overwhelming"] < 1):
        raise ValueError("Must have 0 < first_strike < overwhelming < 1")

    return thresholds


# ---------------------------------------------------------------------------
# Utility weights loader
# ---------------------------------------------------------------------------

def load_utility_weights(path: str | Path, actor: str) -> dict[str, float]:
    """Load utility weights for a given actor from governance_spec.xlsx."""
    sheet_map = {
        "Board": "utilities_board",
        "ASA": "utilities_asa",
        "CEO": "utilities_ceo",
    }
    if actor not in sheet_map:
        raise ValueError(f"Unknown actor: {actor}")

    df = pd.read_excel(path, sheet_name=sheet_map[actor])
    return dict(zip(df["parameter_name"], df["value"]))


# ---------------------------------------------------------------------------
# Policy parameters loader
# ---------------------------------------------------------------------------

def load_policy_parameters(path: str | Path) -> dict[tuple[str, str, str], float]:
    """Load Level-1 fixed policy parameters.

    Returns dict keyed by (actor, node_name, parameter_name) -> value.
    """
    df = pd.read_excel(path, sheet_name="policy_parameters")
    params = {}
    for _, row in df.iterrows():
        key = (row["actor"], row["node_name"], row["parameter_name"])
        params[key] = row["value"]
    return params


# ---------------------------------------------------------------------------
# Board overconfidence bias loader
# ---------------------------------------------------------------------------

def load_board_overconfidence(path: str | Path) -> dict[str, float]:
    """Load Board overconfidence bias parameters from governance_spec.xlsx.

    Returns dict with keys: d1_floor, d1_ceiling, d3_floor, d3_ceiling,
    sigma_scale, review_car_bias. These define the Uniform distribution
    bounds for the Board's biased perception of governance action effectiveness
    on shareholder protest, the sigma_vote scaling factor for overprecision,
    and the review CAR location bias for overestimation of governance quality.

    Calibration (board-background/literature-review-Board-overconfidence.pdf):
      Mean bias:     μ̂ = (1+β)μ*, β ~ U(0.25, 1.0). Midpoint β=0.625.
      Variance bias: σ̂² = σ*²/κ, κ ~ U(2, 5). sigma_scale = 1/√κ, midpoint 0.53.
      Review bias:   μ_f_biased = μ_f + review_car_bias. Board perceives
                     review findings CAR ~3pp more favourable. Midpoint 0.03.
    Sources: Twardawski & Kind 2023, Brahma et al. 2023, Coffeng et al. 2021,
    Boundy-Singer et al. 2022, Ertimur et al. 2011, Fan & Radhakrishnan 2020.
    """
    df = pd.read_excel(path, sheet_name="board_overconfidence")
    params = dict(zip(df["parameter_name"], df["value"]))

    required = {"d1_floor", "d1_ceiling", "d3_floor", "d3_ceiling",
                "sigma_scale", "review_car_bias"}
    missing = required - set(params.keys())
    if missing:
        raise ValueError(f"board_overconfidence missing parameters: {missing}")

    if not (0 <= params["d1_floor"] < params["d1_ceiling"] <= 1):
        raise ValueError(
            f"d1 bounds must satisfy 0 <= floor < ceiling <= 1, "
            f"got [{params['d1_floor']}, {params['d1_ceiling']}]"
        )
    if not (-1 <= params["d3_floor"] < params["d3_ceiling"] <= 1):
        raise ValueError(
            f"d3 bounds must satisfy -1 <= floor < ceiling <= 1, "
            f"got [{params['d3_floor']}, {params['d3_ceiling']}]"
        )
    if not (0 < params["sigma_scale"] <= 1):
        raise ValueError(
            f"sigma_scale must satisfy 0 < sigma_scale <= 1, "
            f"got {params['sigma_scale']}"
        )
    if not (params["review_car_bias"] >= 0):
        raise ValueError(
            f"review_car_bias must be >= 0, "
            f"got {params['review_car_bias']}"
        )

    return params


# ---------------------------------------------------------------------------
# BeliefBundle
# ---------------------------------------------------------------------------

class BeliefBundle:
    """Loads and indexes checkpoint posterior draws from .npz files."""

    def __init__(self, path: str | Path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        data = np.load(path, allow_pickle=True)
        self.B_mkt = data["B_mkt"].astype(np.float64)
        self.B_mgmt = data["B_mgmt"].astype(np.float64)
        self.N = len(self.B_mkt)

        # Load vote model parameters (may or may not exist in all checkpoints)
        self.alpha_vote = data["alpha_vote"] if "alpha_vote" in data else np.zeros(self.N)
        self.gamma_A = data["gamma_A"] if "gamma_A" in data else np.zeros(self.N)
        self.gamma_AH = data["gamma_AH"] if "gamma_AH" in data else np.zeros(self.N)
        self.gamma_D = data["gamma_D"] if "gamma_D" in data else np.zeros(self.N)
        self.sigma_vote = data["sigma_vote"] if "sigma_vote" in data else np.ones(self.N) * 0.5

        # Review model parameters
        self.review_param_1 = data["review_param_1"] if "review_param_1" in data else np.zeros(self.N)
        self.review_param_2 = data["review_param_2"] if "review_param_2" in data else np.ones(self.N) * 0.3

        # Metadata
        if "metadata_json" in data:
            meta_raw = data["metadata_json"]
            if isinstance(meta_raw, np.ndarray):
                meta_raw = str(meta_raw)
            self.metadata = json.loads(meta_raw) if isinstance(meta_raw, str) else {}
        else:
            self.metadata = {}

        # Validate lengths
        arrays = [self.B_mkt, self.B_mgmt, self.alpha_vote,
                  self.gamma_A, self.gamma_AH, self.gamma_D, self.sigma_vote,
                  self.review_param_1, self.review_param_2]
        for arr in arrays:
            if len(arr) != self.N:
                raise ValueError(
                    f"Array length mismatch: expected {self.N}, got {len(arr)}"
                )

    def get_draw(self, i: int) -> dict:
        """Return all parameters for draw index i."""
        if i < 0 or i >= self.N:
            raise IndexError(f"Draw index {i} out of range [0, {self.N})")
        return {
            "B_mkt": float(self.B_mkt[i]),
            "B_mgmt": float(self.B_mgmt[i]),
            "alpha_vote": float(self.alpha_vote[i]),
            "gamma_A": float(self.gamma_A[i]),
            "gamma_AH": float(self.gamma_AH[i]),
            "gamma_D": float(self.gamma_D[i]),
            "sigma_vote": float(self.sigma_vote[i]),
            "review_param_1": float(self.review_param_1[i]),
            "review_param_2": float(self.review_param_2[i]),
        }

    def to_dict(self) -> dict:
        """Serialize to dict for cross-process transfer (no file I/O on deserialize)."""
        return {
            "B_mkt": self.B_mkt,
            "B_mgmt": self.B_mgmt,
            "alpha_vote": self.alpha_vote,
            "gamma_A": self.gamma_A,
            "gamma_AH": self.gamma_AH,
            "gamma_D": self.gamma_D,
            "sigma_vote": self.sigma_vote,
            "review_param_1": self.review_param_1,
            "review_param_2": self.review_param_2,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BeliefBundle":
        """Reconstruct from dict without file I/O."""
        obj = object.__new__(cls)
        obj.B_mkt = data["B_mkt"]
        obj.B_mgmt = data["B_mgmt"]
        obj.N = len(obj.B_mkt)
        obj.alpha_vote = data["alpha_vote"]
        obj.gamma_A = data["gamma_A"]
        obj.gamma_AH = data["gamma_AH"]
        obj.gamma_D = data["gamma_D"]
        obj.sigma_vote = data["sigma_vote"]
        obj.review_param_1 = data["review_param_1"]
        obj.review_param_2 = data["review_param_2"]
        obj.metadata = data.get("metadata", {})
        return obj


# ---------------------------------------------------------------------------
# ParameterSampler
# ---------------------------------------------------------------------------

class ParameterSampler:
    """Samples opponent utility parameters from prior distributions."""

    DISTRIBUTION_SAMPLERS = {
        "normal": lambda rng, p1, p2, _: rng.normal(p1, p2),
        "lognormal": lambda rng, p1, p2, _: rng.lognormal(p1, p2),
        "beta": lambda rng, p1, p2, _: rng.beta(p1, p2),
        "uniform": lambda rng, p1, p2, _: rng.uniform(p1, p2),
        "gamma": lambda rng, p1, p2, _: rng.gamma(p1, p2),
    }

    def __init__(self, path: str | Path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Opponent priors not found: {path}")

        df = pd.read_excel(path, sheet_name="priors")
        self._validate(df)

        # Index: (perspective_actor, target_actor, parameter_name) -> (dist, p1, p2, p3)
        self._priors: dict[tuple[str, str, str], tuple] = {}
        for _, row in df.iterrows():
            key = (row["perspective_actor"], row["target_actor"], row["parameter_name"])
            p3 = row.get("param3", None)
            if pd.isna(p3):
                p3 = None
            self._priors[key] = (row["distribution"], row["param1"], row["param2"], p3)

    def _validate(self, df: pd.DataFrame) -> None:
        required_cols = {"perspective_actor", "target_actor", "parameter_name",
                         "distribution", "param1", "param2"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"opponent_priors missing columns: {missing}")

        valid_dists = set(self.DISTRIBUTION_SAMPLERS.keys())
        invalid = set(df["distribution"]) - valid_dists
        if invalid:
            raise ValueError(f"Invalid distributions: {invalid}")

        # Check required perspectives exist
        perspectives = set(zip(df["perspective_actor"], df["target_actor"]))
        required_perspectives = {
            ("Board", "ASA"), ("Board", "CEO"),
            ("ASA", "Board"), ("ASA", "CEO"),
        }
        missing_persp = required_perspectives - perspectives
        if missing_persp:
            raise ValueError(f"Missing required perspectives: {missing_persp}")

    def sample_parameters(
        self,
        perspective_actor: str,
        target_actor: str,
        rng: np.random.Generator,
    ) -> dict[str, float]:
        """Sample all parameters for target_actor from perspective_actor's priors."""
        params = {}
        for (persp, tgt, pname), (dist, p1, p2, p3) in self._priors.items():
            if persp == perspective_actor and tgt == target_actor:
                sampler = self.DISTRIBUTION_SAMPLERS[dist]
                params[pname] = float(sampler(rng, p1, p2, p3))
        return params

    def get_parameter_names(self, perspective_actor: str, target_actor: str) -> list[str]:
        """Return list of parameter names available for this perspective."""
        return [
            pname for (persp, tgt, pname) in self._priors
            if persp == perspective_actor and tgt == target_actor
        ]

    def to_dict(self) -> dict:
        """Serialize to dict for cross-process transfer."""
        return {"priors": self._priors}

    @classmethod
    def from_dict(cls, data: dict) -> "ParameterSampler":
        """Reconstruct from dict without file I/O."""
        obj = object.__new__(cls)
        obj._priors = data["priors"]
        return obj
