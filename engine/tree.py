"""
Tree evaluator: node-indexed recursive value computation.

value(node_name, history, state, draw_i, focal_actor, mode)

Switches on node type:
- Terminal: return utility for focal actor
- Decision node owned by focal: return max over feasible actions
- Decision node owned by opponent: get predictive distribution, return weighted sum
- Chance node: sample/integrate accordingly
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from engine.state import DecisionState, BeliefBundle, ParameterSampler
from engine.modes import ModeConfig
from engine.utilities import TerminalOutcome, compute_utility
from engine.chance_models import ChanceModels, OverconfidenceBias
from engine.predictive import PredictiveDistribution


class TreeEvaluator:
    """
    Evaluates the game tree via node-indexed recursion.
    """

    def __init__(
        self,
        beliefs: BeliefBundle,
        chance_models: ChanceModels,
        predictive: PredictiveDistribution,
        utility_weights: dict[str, dict[str, float]],
        n_vote_samples: int = 50,
        n_review_samples: int = 20,
        overconfidence_bias: Optional[OverconfidenceBias] = None,
    ):
        """
        Args:
            beliefs: Checkpoint posterior draws.
            chance_models: Vote and review models.
            predictive: Predictive distribution engine.
            utility_weights: Dict mapping actor -> utility weights.
            n_vote_samples: Monte Carlo samples for vote integration.
            n_review_samples: Monte Carlo samples for review integration.
            overconfidence_bias: Optional bias on governance effect estimates.
                When set, the focal actor's EU calculation uses biased
                governance effects (overestimation/overprecision), while
                rollout simulations use unbiased effects.
        """
        self.beliefs = beliefs
        self.chance_models = chance_models
        self.predictive = predictive
        self.utility_weights = utility_weights
        self.n_vote_samples = n_vote_samples
        self.n_review_samples = n_review_samples
        self.overconfidence_bias = overconfidence_bias

    def value(
        self,
        node_name: str,
        history: dict,
        state: DecisionState,
        draw_i: int,
        mode: ModeConfig,
        rng: np.random.Generator,
        utility_target: Optional[str] = None,
    ) -> float:
        """
        Compute the value of the game tree at the given node.

        Args:
            node_name: Current node in the tree.
            history: Dict of actions/outcomes taken so far.
            state: Current game state.
            draw_i: Belief draw index.
            mode: Analysis mode (focal actor + opponent modelling).
            rng: Random number generator.
            utility_target: Actor whose utility to evaluate (defaults to focal).

        Returns:
            Expected utility value at this node.
        """
        if utility_target is None:
            utility_target = mode.focal_actor

        # Terminal node
        if state.is_terminal(node_name) or node_name is None:
            outcome = PredictiveDistribution._build_outcome(history, state)
            return compute_utility(
                utility_target,
                outcome,
                self.utility_weights.get(utility_target, {}),
            )

        node_type = state.node_type(node_name)
        node_owner = state.node_owner(node_name)

        if node_type == "terminal":
            outcome = PredictiveDistribution._build_outcome(history, state)
            return compute_utility(
                utility_target,
                outcome,
                self.utility_weights.get(utility_target, {}),
            )

        elif node_type == "chance":
            return self._chance_value(
                node_name, history, state, draw_i, mode, rng, utility_target
            )

        elif node_type == "decision":
            if mode.is_focal(node_owner):
                # Focal actor: maximise over feasible actions
                return self._focal_value(
                    node_name, history, state, draw_i, mode, rng, utility_target
                )
            else:
                # Opponent: use predictive distribution
                return self._opponent_value(
                    node_name, history, state, draw_i, mode, rng, utility_target
                )

        raise ValueError(f"Unknown node type: {node_type} for node {node_name}")

    def _focal_value(
        self,
        node_name: str,
        history: dict,
        state: DecisionState,
        draw_i: int,
        mode: ModeConfig,
        rng: np.random.Generator,
        utility_target: str,
    ) -> float:
        """Focal actor maximises over feasible actions."""
        feasible = state.feasible_actions(node_name)
        if not feasible:
            # Skip this node
            next_node = state.next_node(node_name)
            return self.value(next_node, history, state, draw_i, mode, rng, utility_target)

        best_value = -np.inf
        for action in feasible:
            h = dict(history)
            h[node_name] = action
            s = state.apply(node_name, action)
            next_node = s.next_node(node_name)

            v = self.value(next_node, h, s, draw_i, mode, rng, utility_target)
            if v > best_value:
                best_value = v

        return best_value

    def _opponent_value(
        self,
        node_name: str,
        history: dict,
        state: DecisionState,
        draw_i: int,
        mode: ModeConfig,
        rng: np.random.Generator,
        utility_target: str,
    ) -> float:
        """Opponent value: compute predictive distribution and take expectation."""
        feasible = state.feasible_actions(node_name)
        if not feasible:
            next_node = state.next_node(node_name)
            return self.value(next_node, history, state, draw_i, mode, rng, utility_target)

        owner = state.node_owner(node_name)
        model_type = mode.get_opponent_model_type(owner)

        if model_type == "Policy":
            # Use fixed policy (deterministic)
            action = self.predictive._fixed_policy(
                owner, node_name, history, state, feasible, rng=rng
            )
            h = dict(history)
            h[node_name] = action
            s = state.apply(node_name, action)
            next_node = s.next_node(node_name)
            return self.value(next_node, h, s, draw_i, mode, rng, utility_target)

        # ARA: compute predictive distribution
        pred_dist = self.predictive.predict(
            node_name, history, state, draw_i,
            focal_actor=mode.focal_actor,
            mode=mode,
            level=mode.level,
            rng=rng,
        )

        # Take expectation over opponent's actions
        expected_value = 0.0
        for action, prob in pred_dist.items():
            if prob <= 0:
                continue
            h = dict(history)
            h[node_name] = action
            s = state.apply(node_name, action)
            next_node = s.next_node(node_name)

            v = self.value(next_node, h, s, draw_i, mode, rng, utility_target)
            expected_value += prob * v

        return expected_value

    def _chance_value(
        self,
        node_name: str,
        history: dict,
        state: DecisionState,
        draw_i: int,
        mode: ModeConfig,
        rng: np.random.Generator,
        utility_target: str,
    ) -> float:
        """Integrate over chance outcomes via Monte Carlo."""
        if node_name == "V":
            return self._vote_value(
                history, state, draw_i, mode, rng, utility_target
            )
        elif node_name == "R":
            return self._review_value(
                history, state, draw_i, mode, rng, utility_target
            )
        elif node_name in ("M_agm", "M_rev"):
            # Market reaction nodes: pass through to next node
            h = dict(history)
            h[node_name] = "market_reaction"
            next_node = state.next_node(node_name)
            return self.value(next_node, h, state, draw_i, mode, rng, utility_target)
        else:
            # Unknown chance node: skip
            next_node = state.next_node(node_name)
            return self.value(next_node, history, state, draw_i, mode, rng, utility_target)

    def _vote_value(
        self,
        history: dict,
        state: DecisionState,
        draw_i: int,
        mode: ModeConfig,
        rng: np.random.Generator,
        utility_target: str,
    ) -> float:
        """Monte Carlo integration over vote outcomes."""
        # Draw governance effect ONCE per scenario (epistemic uncertainty).
        # This is held fixed across all MC vote samples within this draw.
        # If overconfidence_bias is set, the focal actor's EU uses biased
        # governance effects (e.g., Board overestimates review effectiveness)
        # AND a scaled-down sigma_vote (Board underestimates vote uncertainty).
        d1_action = history.get("D1", "D0_minimal")
        gov_effect = self.chance_models.vote._governance_effect(
            d1_action, rng, bias=self.overconfidence_bias
        )

        # Extract sigma_scale from bias (overprecision on vote uncertainty)
        bias_sigma_scale = None
        if self.overconfidence_bias is not None:
            bias_sigma_scale = self.overconfidence_bias.sigma_scale

        total = 0.0
        for _ in range(self.n_vote_samples):
            s_rng = np.random.default_rng(rng.integers(0, 2**32))
            vote_out = self.chance_models.sample_vote(
                draw_i, self.beliefs, history, state, s_rng,
                governance_effect=gov_effect,
                sigma_scale=bias_sigma_scale,
            )

            h = dict(history)
            h["V"] = "vote"
            h["V_percent"] = vote_out.vote_percent
            h["V_strike"] = vote_out.strike_indicator
            h["V_overwhelming"] = vote_out.overwhelming_indicator

            next_node = state.next_node("V")
            total += self.value(next_node, h, state, draw_i, mode, s_rng, utility_target)

        return total / self.n_vote_samples

    def _review_value(
        self,
        history: dict,
        state: DecisionState,
        draw_i: int,
        mode: ModeConfig,
        rng: np.random.Generator,
        utility_target: str,
    ) -> float:
        """Monte Carlo integration over review outcomes.

        If overconfidence_bias is set, the focal actor's EU calculation uses
        a biased review CAR location (shifted upward → Board believes findings
        will produce a more favourable market reaction).

        The direct cost of commissioning the review is drawn ONCE per scenario
        (epistemic uncertainty about fees + distraction + internal resources)
        and held fixed across MC review samples, following the same pattern
        as governance_effect in _vote_value.
        """
        if not state.review_commissioned:
            # No review: deterministic pass-through
            h = dict(history)
            h["R"] = "review"
            h["R_adverse"] = False
            h["R_car"] = 0.0
            h["R_direct_cost"] = 0.0
            s_copy = state.apply("R", "no_adverse")
            s_copy.review_completed = True
            next_node = state.next_node("R")
            return self.value(next_node, h, s_copy, draw_i, mode, rng, utility_target)

        # Draw direct cost ONCE per scenario (epistemic uncertainty).
        # Gamma(4.55, 4741) in decimal CAR — fees, distraction, internal resources.
        review_direct_cost = self.chance_models.sample_review_direct_cost(rng)

        total = 0.0
        for _ in range(self.n_review_samples):
            s_rng = np.random.default_rng(rng.integers(0, 2**32))
            review_out = self.chance_models.sample_review(
                draw_i, self.beliefs, history, state, s_rng,
                bias=self.overconfidence_bias,
            )

            h = dict(history)
            h["R"] = "review"
            h["R_adverse"] = review_out.review_adverse
            h["R_car"] = review_out.review_car
            h["R_direct_cost"] = review_direct_cost

            s = state.apply("R", "adverse" if review_out.review_adverse else "no_adverse")
            s.review_completed = True

            next_node = state.next_node("R")
            total += self.value(next_node, h, s, draw_i, mode, s_rng, utility_target)

        return total / self.n_review_samples

    def optimal_action(
        self,
        node_name: str,
        history: dict,
        state: DecisionState,
        draw_i: int,
        mode: ModeConfig,
        rng: np.random.Generator,
    ) -> tuple[str, float]:
        """
        Find the optimal action for the focal actor at node_name.

        Returns (best_action, best_value).
        """
        feasible = state.feasible_actions(node_name)
        if not feasible:
            raise ValueError(f"No feasible actions at {node_name}")

        best_action = None
        best_value = -np.inf

        for action in feasible:
            h = dict(history)
            h[node_name] = action
            s = state.apply(node_name, action)
            next_node = s.next_node(node_name)

            v = self.value(next_node, h, s, draw_i, mode, rng)
            if v > best_value:
                best_value = v
                best_action = action

        return best_action, best_value
