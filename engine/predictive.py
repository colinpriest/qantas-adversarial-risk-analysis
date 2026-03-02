"""
Predictive distribution engine.

Computes p_i(X | h) — the focal actor's predictive distribution over an
opponent's action at a decision node. This is the core ARA mechanism.

Algorithm:
1. Identify owner of node.
2. Sample K opponent parameter draws Theta_j^(k).
3. For each k:
   - For each feasible action x:
     - Compute opponent expected utility Psi_j(x; h, Theta_j^(k))
   - Determine best response x*(k)
4. Return empirical distribution over x*(k).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from engine.state import DecisionState, BeliefBundle, ParameterSampler
from engine.modes import ModeConfig
from engine.utilities import TerminalOutcome, compute_utility
from engine.chance_models import ChanceModels, OverconfidenceBias


class PredictiveDistribution:
    """
    Computes predictive distributions over opponent actions at decision nodes.
    """

    def __init__(
        self,
        beliefs: BeliefBundle,
        param_sampler: ParameterSampler,
        chance_models: ChanceModels,
        policy_params: dict,
        K: int = 200,
        R_rollouts: int = 20,
        overconfidence_bias: Optional[OverconfidenceBias] = None,
    ):
        self.beliefs = beliefs
        self.param_sampler = param_sampler
        self.chance_models = chance_models
        self.policy_params = policy_params
        self.K = K
        self.R_rollouts = R_rollouts
        self.overconfidence_bias = overconfidence_bias

    def predict(
        self,
        node_name: str,
        history: dict,
        state: DecisionState,
        draw_i: int,
        focal_actor: str,
        mode: ModeConfig,
        level: int,
        rng: np.random.Generator,
    ) -> dict[str, float]:
        """
        Compute predictive distribution over opponent's action at node_name.

        Returns dict mapping action_name -> probability.
        """
        owner = state.node_owner(node_name)
        feasible = state.feasible_actions(node_name)

        if not feasible:
            return {}

        if len(feasible) == 1:
            return {feasible[0]: 1.0}

        counts = {a: 0 for a in feasible}

        for k in range(self.K):
            k_rng = np.random.default_rng(rng.integers(0, 2**32))

            # Sample opponent parameters from focal's beliefs
            theta_j = self.param_sampler.sample_parameters(
                perspective_actor=focal_actor,
                target_actor=owner,
                rng=k_rng,
            )

            # Find best response for opponent
            best_action = None
            best_value = -np.inf

            for action in feasible:
                psi = self._compute_psi_rollout(
                    owner=owner,
                    node_name=node_name,
                    action=action,
                    history=history,
                    state=state,
                    draw_i=draw_i,
                    theta_j=theta_j,
                    focal_actor=focal_actor,
                    mode=mode,
                    level=level,
                    rng=k_rng,
                )
                if psi > best_value:
                    best_value = psi
                    best_action = action

            if best_action is not None:
                counts[best_action] += 1

        # Normalise to distribution
        total = sum(counts.values())
        if total == 0:
            # Uniform fallback
            p = 1.0 / len(feasible)
            return {a: p for a in feasible}

        return {a: c / total for a, c in counts.items()}

    def _compute_psi_rollout(
        self,
        owner: str,
        node_name: str,
        action: str,
        history: dict,
        state: DecisionState,
        draw_i: int,
        theta_j: dict[str, float],
        focal_actor: str,
        mode: ModeConfig,
        level: int,
        rng: np.random.Generator,
    ) -> float:
        """
        Compute Psi_j(x; h, Theta_j) via stochastic rollouts.

        Simulates the game forward from the forced action, using fixed policies
        for other actors, and returns average utility for the owner.
        """
        total_utility = 0.0

        for r in range(self.R_rollouts):
            r_rng = np.random.default_rng(rng.integers(0, 2**32))

            # Apply the forced action
            h = dict(history)
            h[node_name] = action
            s = state.apply(node_name, action)

            # Simulate forward through remaining nodes
            current_node = s.next_node(node_name)
            outcome = self._simulate_forward(
                current_node=current_node,
                history=h,
                state=s,
                draw_i=draw_i,
                owner=owner,
                focal_actor=focal_actor,
                mode=mode,
                level=level,
                rng=r_rng,
            )

            total_utility += compute_utility(owner, outcome, theta_j)

        return total_utility / self.R_rollouts

    def _simulate_forward(
        self,
        current_node: Optional[str],
        history: dict,
        state: DecisionState,
        draw_i: int,
        owner: str,
        focal_actor: str,
        mode: ModeConfig,
        level: int,
        rng: np.random.Generator,
    ) -> TerminalOutcome:
        """Simulate forward from current_node to terminal, return outcome."""
        h = dict(history)
        s = state

        while current_node is not None and not s.is_terminal(current_node):
            node_type = s.node_type(current_node)
            node_owner = s.node_owner(current_node)

            if node_type == "chance":
                # Sample chance outcome using focal actor's (possibly biased)
                # model of Nature. ARA Level-1: the focal actor uses its own
                # beliefs about chance nodes everywhere, including rollouts.
                if current_node == "V":
                    # Draw governance effect with bias (if any)
                    d1_action = h.get("D1", "D0_minimal")
                    gov_effect = self.chance_models.vote._governance_effect(
                        d1_action, rng, bias=self.overconfidence_bias
                    )
                    # Extract sigma_scale from bias (overprecision)
                    bias_sigma_scale = None
                    if self.overconfidence_bias is not None:
                        bias_sigma_scale = self.overconfidence_bias.sigma_scale
                    vote_out = self.chance_models.sample_vote(
                        draw_i, self.beliefs, h, s, rng,
                        governance_effect=gov_effect,
                        sigma_scale=bias_sigma_scale,
                    )
                    h["V"] = "vote"
                    h["V_percent"] = vote_out.vote_percent
                    h["V_strike"] = vote_out.strike_indicator
                    h["V_overwhelming"] = vote_out.overwhelming_indicator
                elif current_node == "R":
                    review_out = self.chance_models.sample_review(
                        draw_i, self.beliefs, h, s, rng,
                        bias=self.overconfidence_bias,
                    )
                    h["R"] = "review"
                    h["R_adverse"] = review_out.review_adverse
                    h["R_car"] = review_out.review_car
                    h["R_direct_cost"] = (
                        self.chance_models.sample_review_direct_cost(rng)
                        if s.review_commissioned else 0.0
                    )
                    if review_out.review_adverse:
                        s = s.apply("R", "adverse")
                    s.review_completed = True
                elif current_node in ("M_agm", "M_rev"):
                    # Market reaction nodes - pass through
                    h[current_node] = "market_reaction"

            elif node_type == "decision":
                # Use fixed policy or predictive distribution
                feasible = s.feasible_actions(current_node)
                if not feasible:
                    current_node = s.next_node(current_node)
                    continue

                if node_owner == owner:
                    # Owner picks best action (myopic for rollouts)
                    action = self._owner_policy(
                        node_owner, current_node, h, s, feasible, rng=rng
                    )
                elif (level >= 2 and
                      mode.get_strategic_counterpart(node_owner) is not None):
                    # Level-2: model this actor strategically
                    pred = self.predict(
                        current_node, h, s, draw_i,
                        focal_actor=owner,
                        mode=mode,
                        level=level - 1,
                        rng=rng,
                    )
                    action = self._sample_from_dist(pred, rng)
                else:
                    # Fixed policy
                    action = self._fixed_policy(
                        node_owner, current_node, h, s, feasible, rng=rng
                    )

                h[current_node] = action
                s = s.apply(current_node, action)

            current_node = s.next_node(current_node)

        return self._build_outcome(h, s)

    def _fixed_policy(
        self,
        actor: str,
        node_name: str,
        history: dict,
        state: DecisionState,
        feasible: list[str],
        rng: np.random.Generator = None,
    ) -> str:
        """Apply fixed policy for Level-1 non-strategic actors."""
        if not feasible:
            raise ValueError(f"No feasible actions for {actor} at {node_name}")

        if actor == "Board" and node_name == "D_rev":
            vote_pct = history.get("V_percent", 0.0)
            review_thresh = self.policy_params.get(
                ("Board", "D_rev", "review_vote_threshold"), 0.25
            )
            sack_thresh = self.policy_params.get(
                ("Board", "D_rev", "sack_vote_threshold"), 0.50
            )
            if vote_pct >= sack_thresh and "Drev_sack_ceo" in feasible:
                return "Drev_sack_ceo"
            elif vote_pct >= review_thresh and "Drev_commission_review" in feasible:
                return "Drev_commission_review"
            return "Drev_no_action"

        elif actor == "CEO" and node_name == "D0_ceo":
            # D0_ceo: CEO's pre-game resignation decision.
            # Default fixed policy: CEO stays (conservative).
            return "CEO_stay" if "CEO_stay" in feasible else feasible[0]

        elif actor == "CEO" and node_name == "D4":
            vote_pct = history.get("V_percent", 0.0)
            resign_thresh = self.policy_params.get(
                ("CEO", "D4", "resign_vote_threshold"), 0.40
            )
            if vote_pct >= resign_thresh and "D4_resign" in feasible:
                return "D4_resign"
            return "D4_stay" if "D4_stay" in feasible else feasible[0]

        elif actor == "ASA" and node_name == "A2":
            # ASA recommendation conditioned on Board's D1 action.
            # Empirical data from data/ranked_voting_recommendations.csv:
            #   Rank 0 (D0): n=24, k=22 Against  (point est 91.7%)
            #   Rank 1 (D1): n=8,  k=5  Against  (point est 62.5%)
            #   Rank 2 (D3): n=4,  k=4  Against  (point est 100.0%)
            #
            # Small samples → use Beta-Binomial posterior to propagate
            # parameter uncertainty. Prior: Beta(1, 1) (uniform/Laplace).
            # Posterior: Beta(k+1, n-k+1).
            #   Rank 0: Beta(23, 3) — posterior mean 0.885, tight
            #   Rank 1: Beta(6, 4)  — posterior mean 0.600, wide
            #   Rank 2: Beta(5, 1)  — posterior mean 0.833, wide
            d1_action = history.get("D1", "D0_minimal")

            if rng is None:
                # No rng: fall back to posterior means
                p_map = {"D3_ceo_transition": 5/6, "D1_review": 6/10}
                return "A2_rec_strike" if "A2_rec_strike" in feasible else feasible[0]

            # Sample p_strike from Beta posterior, then flip
            if d1_action == "D3_ceo_transition":
                # Rank 2: Beta(5, 1) — n=4, k=4. Posterior mean 0.833.
                p_strike = rng.beta(5, 1)
            elif d1_action == "D1_review":
                # Rank 1: Beta(6, 4) — n=8, k=5. Posterior mean 0.600.
                p_strike = rng.beta(6, 4)
            else:
                # Rank 0: Beta(23, 3) — n=24, k=22. Posterior mean 0.885.
                p_strike = rng.beta(23, 3)

            if rng.random() >= p_strike:
                return "A2_no_strike" if "A2_no_strike" in feasible else feasible[0]
            return "A2_rec_strike" if "A2_rec_strike" in feasible else feasible[0]

        elif actor == "Board" and node_name == "D1":
            return "D0_minimal" if "D0_minimal" in feasible else feasible[0]

        # Default: first feasible action
        return feasible[0]

    def _owner_policy(
        self,
        actor: str,
        node_name: str,
        history: dict,
        state: DecisionState,
        feasible: list[str],
        rng: np.random.Generator = None,
    ) -> str:
        """Myopic policy for the owner inside rollouts (picks 'reasonable' action)."""
        return self._fixed_policy(actor, node_name, history, state, feasible, rng=rng)

    @staticmethod
    def _sample_from_dist(
        dist: dict[str, float],
        rng: np.random.Generator,
    ) -> str:
        """Sample an action from a probability distribution."""
        actions = list(dist.keys())
        probs = np.array([dist[a] for a in actions])
        probs = probs / probs.sum()  # Ensure normalisation
        return actions[rng.choice(len(actions), p=probs)]

    @staticmethod
    def _build_outcome(history: dict, state: DecisionState) -> TerminalOutcome:
        """Build a TerminalOutcome from history and final state."""
        return TerminalOutcome(
            d1_action=history.get("D1", "D0_minimal"),
            a2_action=history.get("A2", "A2_no_strike"),
            d_rev_action=history.get("D_rev", "Drev_no_action"),
            d4_action=history.get("D4", "D4_stay"),
            vote_percent=history.get("V_percent", 0.0),
            strike_indicator=history.get("V_strike", False),
            overwhelming_indicator=history.get("V_overwhelming", False),
            review_adverse=history.get("R_adverse", False),
            review_car=history.get("R_car", 0.0),
            review_direct_cost=history.get("R_direct_cost", 0.0),
            CEO_removed=state.CEO_removed,
            CEO_resigned_early=state.CEO_resigned_early,
            review_commissioned=state.review_commissioned,
        )
