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
        K_nested: Optional[int] = None,
        R_nested: Optional[int] = None,
        no_prior_actors: Optional[set[str]] = None,
    ):
        self.beliefs = beliefs
        self.param_sampler = param_sampler
        self.chance_models = chance_models
        self.policy_params = policy_params
        self.K = K
        self.R_rollouts = R_rollouts
        self.overconfidence_bias = overconfidence_bias
        # Actors whose predictive distributions should NOT receive Laplace
        # smoothing.  Default (empty set): all actors get a Dirichlet(1,...,1)
        # prior so no feasible action has exactly zero probability.
        self.no_prior_actors: set[str] = no_prior_actors or set()
        # Lighter-weight settings for recursive Level-2 predictions to avoid
        # quadratic explosion (K * R inside every nested rollout).
        # Default: downscale to something modest if not explicitly provided.
        self.K_nested = (
            K_nested if K_nested is not None
            else max(5, min(10, K // 2))
        )
        self.R_nested = (
            R_nested if R_nested is not None
            else max(2, min(5, R_rollouts))
        )

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
        use_nested: bool = False,
    ) -> dict[str, float]:
        """
        Compute predictive distribution over opponent's action at node_name.

        Returns dict mapping action_name -> probability.
        """
        # Use reduced K/R when this call is part of a recursive Level-2 rollout.
        K_eff = self.K_nested if use_nested else self.K
        R_eff = self.R_nested if use_nested else self.R_rollouts
        owner = state.node_owner(node_name)
        feasible = state.feasible_actions(node_name)

        if not feasible:
            return {}

        if len(feasible) == 1:
            return {feasible[0]: 1.0}

        counts = {a: 0 for a in feasible}

        for k in range(K_eff):
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
                    R_rollouts=R_eff,
                )
                if psi > best_value:
                    best_value = psi
                    best_action = action

            if best_action is not None:
                counts[best_action] += 1

        # Laplace smoothing: Dirichlet(1,...,1) prior over opponent's mixed
        # strategy ensures no feasible action has exactly zero probability.
        # This is the default; --no-{actor}-prior disables it per actor.
        use_prior = owner not in self.no_prior_actors
        pseudocount = 1 if use_prior else 0

        # Normalise to distribution
        total = sum(counts.values()) + pseudocount * len(feasible)
        if total == 0:
            # Uniform fallback (only reachable when prior disabled and all
            # rollouts returned None best actions)
            p = 1.0 / len(feasible)
            return {a: p for a in feasible}

        return {a: (c + pseudocount) / total for a, c in counts.items()}

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
        R_rollouts: Optional[int] = None,
    ) -> float:
        """
        Compute Psi_j(x; h, Theta_j) via stochastic rollouts.

        Simulates the game forward from the forced action, using fixed policies
        for other actors, and returns average utility for the owner.
        """
        total_utility = 0.0
        R = R_rollouts or self.R_rollouts

        for r in range(R):
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

        return total_utility / R

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
                    # Crisis floor (one draw per rollout — epistemic)
                    crisis_floor = None
                    if s.headline_incident:
                        crisis_floor = float(rng.beta(50, 150))
                    vote_out = self.chance_models.sample_vote(
                        draw_i, self.beliefs, h, s, rng,
                        governance_effect=gov_effect,
                        sigma_scale=bias_sigma_scale,
                        crisis_floor=crisis_floor,
                    )
                    h["V"] = "vote"
                    h["V_percent"] = vote_out.vote_percent
                    h["V_strike"] = vote_out.strike_indicator
                    h["V_overwhelming"] = vote_out.overwhelming_indicator
                elif current_node == "R":
                    # Draw p_adverse once per rollout (epistemic)
                    p_adverse = self.chance_models.review.draw_adverse_probability(
                        rng, bias=self.overconfidence_bias
                    )
                    review_out = self.chance_models.sample_review(
                        draw_i, self.beliefs, h, s, rng,
                        bias=self.overconfidence_bias,
                        p_adverse=p_adverse,
                    )
                    h["R"] = "review"
                    h["R_adverse"] = review_out.review_adverse
                    h["R_car"] = review_out.review_car
                    h["R_direct_cost"] = (
                        self.chance_models.sample_review_direct_cost(rng)
                        if s.review_commissioned else 0.0
                    )
                    s = s.apply("R", "adverse" if review_out.review_adverse else "no_adverse")
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
                        use_nested=True,
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

        if actor == "Board" and node_name in ("D_rev", "D_rev_post_review"):
            vote_pct = history.get("V_percent", 0.0)
            review_thresh = self.policy_params.get(
                ("Board", node_name, "review_vote_threshold"), 0.25
            )
            sack_thresh = self.policy_params.get(
                ("Board", node_name, "sack_vote_threshold"), 0.25
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

        elif actor == "CEO" and node_name in ("D4", "D4_post_review"):
            vote_pct = history.get("V_percent", 0.0)
            resign_thresh = self.policy_params.get(
                ("CEO", node_name, "resign_vote_threshold"), 0.40
            )
            if vote_pct >= resign_thresh and "D4_resign" in feasible:
                return "D4_resign"
            return "D4_stay" if "D4_stay" in feasible else feasible[0]

        elif actor == "ASA" and node_name == "A2":
            # ASA recommendation — informative Beta prior from
            # background/asa/asa-informative-prior.md.
            #
            # Source: ranked_voting_recommendations.csv, headline_incident=1 only.
            # Qantas (QAN, 2023) excluded. Pooled observed rate: 14/15 = 0.933.
            #
            # Key finding: ASA recommendation is near-automatic given a headline
            # incident. Board action (review, CEO replacement) plausibly affects the
            # shareholder vote, but NOT the ASA recommendation — the data provide no
            # statistical evidence for a board-action effect at the recommendation
            # stage. The three Beta distributions are nearly identical; separation is
            # a modelling convention, not a data-driven estimate.
            #
            # Monotonic decreasing by board accountability level (convention):
            #   Board Action 0 — Do nothing          : Beta(46, 4)  mean=0.920
            #   Board Action 1 — Review or CEO resigns: Beta(44, 4)  mean=0.917
            #   Board Action 2 — Sack CEO             : Beta(43, 4)  mean=0.914
            #
            # All 90% CIs are entirely above 0.84.
            #
            # At A2 the knowable board actions are:
            #   D3_ceo_transition → "Sack CEO"
            #   D1_review         → "Review"    (Board Action 1)
            #   CEO_resigned_early→ "CEO resigns" (Board Action 1)
            #   D0_minimal        → "Do nothing" (Board Action 0)
            d1_action = history.get("D1", "D0_minimal")
            ceo_resigned_early = history.get("D0_ceo") == "CEO_resign"

            if rng is None:
                # No rng: MAP — all means > 0.90, always recommend strike
                return "A2_rec_strike" if "A2_rec_strike" in feasible else feasible[0]

            # Two-stage Beta-Binomial: sample p_strike from informative prior
            if d1_action == "D3_ceo_transition":
                # Board Action 2 — Sack CEO: Beta(43, 4) mean=0.914
                p_strike = rng.beta(43, 4)
            elif d1_action == "D1_review" or ceo_resigned_early:
                # Board Action 1 — Review or CEO resigns: Beta(44, 4) mean=0.917
                p_strike = rng.beta(44, 4)
            else:
                # Board Action 0 — Do nothing: Beta(46, 4) mean=0.920
                p_strike = rng.beta(46, 4)

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
            d4_post_review_action=history.get("D4_post_review", "D4_stay"),
            d_rev_post_review_action=history.get("D_rev_post_review", "Drev_no_action"),
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
