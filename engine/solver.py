"""
Solver: orchestrates the full ARA computation.

solve(focal_actor, checkpoint_id, mode_config)

Steps:
1. Load BeliefBundle.
2. For each candidate initial action:
   - For each belief draw i:
     - Compute value via tree recursion.
   - Average over i.
3. Return EU per action, optimal action, outcome distributions, diagnostics.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from engine.state import (
    DecisionState, BeliefBundle, ParameterSampler,
    load_vote_thresholds, load_utility_weights, load_policy_parameters,
    load_board_overconfidence,
)
from engine.modes import ModeConfig, AVAILABLE_MODES
from engine.chance_models import ChanceModels, OverconfidenceBias
from engine.predictive import PredictiveDistribution
from engine.tree import TreeEvaluator
from engine.utilities import TerminalOutcome, compute_utility

logger = logging.getLogger(__name__)

# Sentinel for "use the bias loaded from governance_spec.xlsx"
_USE_SPEC_DEFAULT = object()


@dataclass
class SolveResult:
    """Results from solving the ARA game."""
    focal_actor: str
    checkpoint_id: str
    mode_name: str
    scenario: str = "ceo_stayed"  # "ceo_resigned" or "ceo_stayed"

    # D0_ceo predictive: predicted probability of this scenario
    scenario_prob: float = 0.0
    # Full D0_ceo predictive distribution: {action -> prob}
    d0_ceo_predictive: dict[str, float] = field(default_factory=dict)

    # EU per initial action: {action_name: expected_utility}
    EU_per_action: dict[str, float] = field(default_factory=dict)
    optimal_action: str = ""
    optimal_EU: float = 0.0

    # Per-draw details for diagnostics
    draw_values: dict[str, list[float]] = field(default_factory=dict)

    # Outcome distribution summaries
    outcome_stats: dict[str, dict] = field(default_factory=dict)

    # Opponent predictive distributions at key nodes (for diagnostics)
    predictive_dists: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)

    # Utility decomposition per action
    utility_decomposition: dict[str, dict[str, float]] = field(default_factory=dict)

    # Checkpoint belief statistics
    belief_stats: dict[str, float] = field(default_factory=dict)

    # Overconfidence bias label (for diagnostics)
    overconfidence_bias_label: str = ""

    # Timing
    elapsed_seconds: float = 0.0
    n_draws_used: int = 0

    def summary_df(self) -> pd.DataFrame:
        """Return a summary DataFrame."""
        rows = []
        for action, eu in self.EU_per_action.items():
            row = {
                "checkpoint": self.checkpoint_id,
                "scenario": self.scenario,
                "Pr_scenario": self.scenario_prob,
                "focal": self.focal_actor,
                "mode": self.mode_name,
                "action": action,
                "Expected_Utility": eu,
                "is_optimal": action == self.optimal_action,
            }
            # Add outcome stats if available
            if action in self.outcome_stats:
                for k, v in self.outcome_stats[action].items():
                    row[k] = v
            rows.append(row)
        return pd.DataFrame(rows)

    def print_diagnostics(self) -> None:
        """Print detailed diagnostic breakdown."""
        W = 70
        print("\n" + "=" * W)
        print("DIAGNOSTIC REPORT")
        print("=" * W)

        # 0. Scenario and D0_ceo predicted probability
        if self.scenario:
            print(f"\n--- Scenario ---")
            print(f"  {self.scenario} (Pr = {self.scenario_prob:.1%})")
        if self.d0_ceo_predictive:
            print(f"\n--- D0_ceo predictive distribution ---")
            for action, prob in self.d0_ceo_predictive.items():
                print(f"  {action}: {prob:.1%}")

        # 0b. Overconfidence bias
        if self.overconfidence_bias_label:
            print(f"\n--- Overconfidence bias ---")
            print(f"  {self.overconfidence_bias_label}")

        # 1. Checkpoint belief stats
        if self.belief_stats:
            print(f"\n--- Checkpoint {self.checkpoint_id} belief state ---")
            print(f"  B_mkt  mean={self.belief_stats['B_mkt_mean']:+.3f}  "
                  f"sd={self.belief_stats['B_mkt_sd']:.3f}  "
                  f"[{self.belief_stats['B_mkt_p05']:+.3f}, "
                  f"{self.belief_stats['B_mkt_p95']:+.3f}]")
            print(f"  B_mgmt mean={self.belief_stats['B_mgmt_mean']:+.3f}  "
                  f"sd={self.belief_stats['B_mgmt_sd']:.3f}")

        # 2. Per-action breakdown
        print(f"\n--- Expected utility decomposition (N={self.n_draws_used} draws) ---")
        for action in self.EU_per_action:
            eu = self.EU_per_action[action]
            tag = " <-- OPTIMAL" if action == self.optimal_action else ""
            print(f"\n  [{action}] Expected_Utility = {eu:+.4f}{tag}")

            if action in self.outcome_stats:
                s = self.outcome_stats[action]
                print(f"    Vote:   mean={s['mean_vote_percent']:.1%}  "
                      f"sd={s['sd_vote_percent']:.1%}")
                print(f"    Strike (>25%): {s['Pr_strike']:.0%}   "
                      f"Overwhelming (>50%): {s['Pr_overwhelming']:.0%}")
                print(f"    CEO removed:   {s['Pr_CEO_removed']:.0%}   "
                      f"Review adverse: {s['Pr_review_adverse']:.0%}")
                print(f"    Review CAR: mean={s['mean_review_car']:+.2%}  "
                      f"sd={s['sd_review_car']:.2%}")
                print(f"    Review direct cost: mean={s['mean_review_direct_cost']:.4%}  "
                      f"sd={s['sd_review_direct_cost']:.4%}")

            if action in self.utility_decomposition:
                d = self.utility_decomposition[action]
                parts = [f"{k}={v:+.3f}" for k, v in d.items()]
                print(f"    Utility components: {', '.join(parts)}")

            if action in self.draw_values:
                vals = self.draw_values[action]
                print(f"    Per-draw values: mean={np.mean(vals):+.4f}  "
                      f"sd={np.std(vals):.4f}  "
                      f"range=[{np.min(vals):+.4f}, {np.max(vals):+.4f}]")

        # 3. Opponent predictive distributions
        if self.predictive_dists:
            print(f"\n--- Opponent predictive distributions (sample) ---")
            for d1_action, node_dists in self.predictive_dists.items():
                print(f"\n  Given D1={d1_action}:")
                for node, dist in node_dists.items():
                    parts = [f"{a}: {p:.0%}" for a, p in dist.items()]
                    print(f"    {node}: {', '.join(parts)}")

        # 4. Timing
        print(f"\n--- Timing ---")
        print(f"  Total: {self.elapsed_seconds:.1f}s  "
              f"({self.elapsed_seconds/max(self.n_draws_used,1):.1f}s per draw)")
        print()


class Solver:
    """
    Main solver for the adversarial risk analysis game.
    """

    def __init__(
        self,
        governance_spec_path: str | Path,
        opponent_priors_path: str | Path,
        checkpoint_dir: str | Path,
        K: int = 200,
        R_rollouts: int = 20,
        n_vote_samples: int = 50,
        n_review_samples: int = 20,
        seed: int = 42,
    ):
        self.governance_spec_path = Path(governance_spec_path)
        self.opponent_priors_path = Path(opponent_priors_path)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.K = K
        self.R_rollouts = R_rollouts
        self.n_vote_samples = n_vote_samples
        self.n_review_samples = n_review_samples
        self.seed = seed

        # Load configuration
        self.vote_thresholds = load_vote_thresholds(self.governance_spec_path)
        self.utility_weights = {
            actor: load_utility_weights(self.governance_spec_path, actor)
            for actor in ["Board", "ASA", "CEO"]
        }
        self.policy_params = load_policy_parameters(self.governance_spec_path)
        self.param_sampler = ParameterSampler(self.opponent_priors_path)

        # Board overconfidence bias (default from governance_spec)
        bias_params = load_board_overconfidence(self.governance_spec_path)
        self.overconfidence_bias = OverconfidenceBias(**bias_params)

    def solve(
        self,
        focal_actor: str,
        checkpoint_id: str,
        mode: Optional[ModeConfig] = None,
        n_draws: Optional[int] = None,
        initial_node: str = "D1",
        overconfidence_bias=_USE_SPEC_DEFAULT,
        scenario: str = "ceo_stayed",
    ) -> SolveResult:
        """
        Solve the ARA game for a given focal actor and checkpoint.

        Args:
            focal_actor: "Board" or "ASA"
            checkpoint_id: Checkpoint identifier (e.g. "C0")
            mode: Mode configuration. If None, uses default for focal_actor.
            n_draws: Number of belief draws to use. If None, uses all available.
            overconfidence_bias: Cognitive bias on governance effects.
                Default: uses bias loaded from governance_spec.xlsx (hubris).
                Pass None for unbiased counterfactual analysis.
                Pass an OverconfidenceBias instance for a custom profile.
            initial_node: Root decision node (default "D1").
            scenario: Pre-game CEO resignation scenario.
                "ceo_stayed": CEO present (default, backward-compatible).
                "ceo_resigned": CEO resigned on 05-Sep-2023 before the game tree.
                    Automatically prunes D3_ceo_transition, Drev_sack_ceo, and D4.

        Returns:
            SolveResult with EU per action, optimal action, and diagnostics.
        """
        t0 = time.time()

        # Resolve overconfidence bias
        if overconfidence_bias is _USE_SPEC_DEFAULT:
            effective_bias = self.overconfidence_bias
        else:
            effective_bias = overconfidence_bias

        # Default mode
        if mode is None:
            mode = AVAILABLE_MODES.get(focal_actor.lower())
            if mode is None:
                raise ValueError(f"No default mode for focal actor: {focal_actor}")

        # Load checkpoint beliefs
        checkpoint_path = self._find_checkpoint(checkpoint_id)
        beliefs = BeliefBundle(checkpoint_path)

        # Setup engine components
        chance_models = ChanceModels(self.vote_thresholds)
        predictive = PredictiveDistribution(
            beliefs=beliefs,
            param_sampler=self.param_sampler,
            chance_models=chance_models,
            policy_params=self.policy_params,
            K=self.K,
            R_rollouts=self.R_rollouts,
            overconfidence_bias=effective_bias,
        )
        tree = TreeEvaluator(
            beliefs=beliefs,
            chance_models=chance_models,
            predictive=predictive,
            utility_weights=self.utility_weights,
            n_vote_samples=self.n_vote_samples,
            n_review_samples=self.n_review_samples,
            overconfidence_bias=effective_bias,
        )

        # Setup initial state and apply pre-game scenario
        base_state = DecisionState.from_governance_spec(
            self.governance_spec_path,
            checkpoint_id=checkpoint_id,
        )
        base_state = base_state.for_scenario(scenario)

        # Determine draws to use
        N = beliefs.N
        if n_draws is not None:
            N = min(n_draws, N)

        # Get feasible initial actions
        initial_actions = base_state.feasible_actions(initial_node)
        if not initial_actions:
            raise ValueError(f"No feasible actions at {initial_node}")

        logger.info(
            f"Solving {mode.name}: focal={focal_actor}, checkpoint={checkpoint_id}, "
            f"scenario={scenario}, N={N}, actions={initial_actions}"
        )

        # Evaluate each initial action across belief draws
        result = SolveResult(
            focal_actor=focal_actor,
            checkpoint_id=checkpoint_id,
            mode_name=mode.name,
            scenario=scenario,
            n_draws_used=N,
        )

        # Record overconfidence bias in result
        if effective_bias is not None:
            result.overconfidence_bias_label = (
                f"D1=[{effective_bias.d1_floor:.1f},{effective_bias.d1_ceiling:.1f}], "
                f"D3=[{effective_bias.d3_floor:.1f},{effective_bias.d3_ceiling:.1f}], "
                f"sigma_scale={effective_bias.sigma_scale:.2f}, "
                f"review_car_bias={effective_bias.review_car_bias:.4f}"
            )
        else:
            result.overconfidence_bias_label = "None (unbiased counterfactual)"

        # Collect checkpoint belief stats
        result.belief_stats = {
            "B_mkt_mean": float(np.mean(beliefs.B_mkt[:N])),
            "B_mkt_sd": float(np.std(beliefs.B_mkt[:N])),
            "B_mkt_p05": float(np.percentile(beliefs.B_mkt[:N], 5)),
            "B_mkt_p95": float(np.percentile(beliefs.B_mkt[:N], 95)),
            "B_mgmt_mean": float(np.mean(beliefs.B_mgmt[:N])),
            "B_mgmt_sd": float(np.std(beliefs.B_mgmt[:N])),
            "alpha_vote_mean": float(np.mean(beliefs.alpha_vote[:N])),
            "gamma_A_mean": float(np.mean(beliefs.gamma_A[:N])),
            "gamma_D_mean": float(np.mean(beliefs.gamma_D[:N])),
            "sigma_vote_mean": float(np.mean(beliefs.sigma_vote[:N])),
        }

        n_actions = len(initial_actions)
        total_evals = n_actions * N
        pbar = tqdm(
            total=total_evals,
            desc=f"Solving ({focal_actor}, {checkpoint_id})",
            unit="draw",
        )

        for action_idx, action in enumerate(initial_actions):
            pbar.set_postfix(
                action=f"{action} [{action_idx+1}/{n_actions}]",
                refresh=True,
            )
            values = []
            outcome_accum = {
                "vote_percent": [],
                "strike_prob": 0,
                "overwhelming_prob": 0,
                "CEO_removed_prob": 0,
                "review_adverse_prob": 0,
                "review_car": [],
                "review_direct_cost": [],
            }

            for i in range(N):
                rng = np.random.default_rng(self.seed + i)

                # Apply initial action
                h = {initial_node: action}
                s = base_state.apply(initial_node, action)
                next_node = s.next_node(initial_node)

                # Evaluate tree from next node
                v = tree.value(next_node, h, s, i, mode, rng)
                values.append(v)

                # Collect outcome statistics via a single rollout
                out_rng = np.random.default_rng(self.seed + i + N)
                outcome = predictive._simulate_forward(
                    current_node=next_node,
                    history=h,
                    state=s,
                    draw_i=i,
                    owner=focal_actor,
                    focal_actor=focal_actor,
                    mode=mode,
                    level=mode.level,
                    rng=out_rng,
                )
                outcome_accum["vote_percent"].append(outcome.vote_percent)
                outcome_accum["strike_prob"] += int(outcome.strike_indicator)
                outcome_accum["overwhelming_prob"] += int(outcome.overwhelming_indicator)
                outcome_accum["CEO_removed_prob"] += int(outcome.CEO_removed)
                outcome_accum["review_adverse_prob"] += int(outcome.review_adverse)
                outcome_accum["review_car"].append(outcome.review_car)
                outcome_accum["review_direct_cost"].append(outcome.review_direct_cost)

                pbar.update(1)

            eu = float(np.mean(values))
            result.EU_per_action[action] = eu
            result.draw_values[action] = values

            # Summarise outcome statistics
            result.outcome_stats[action] = {
                "Pr_strike": outcome_accum["strike_prob"] / N,
                "Pr_overwhelming": outcome_accum["overwhelming_prob"] / N,
                "Pr_CEO_removed": outcome_accum["CEO_removed_prob"] / N,
                "Pr_review_adverse": outcome_accum["review_adverse_prob"] / N,
                "mean_vote_percent": float(np.mean(outcome_accum["vote_percent"])),
                "sd_vote_percent": float(np.std(outcome_accum["vote_percent"])),
                "mean_review_car": float(np.mean(outcome_accum["review_car"])),
                "sd_review_car": float(np.std(outcome_accum["review_car"])),
                "mean_review_direct_cost": float(np.mean(outcome_accum["review_direct_cost"])),
                "sd_review_direct_cost": float(np.std(outcome_accum["review_direct_cost"])),
            }

            # Print via tqdm.write so it doesn't collide with the bar
            tqdm.write(
                f"  Action {action_idx+1}/{n_actions} '{action}': "
                f"EU={eu:.4f}"
            )

        pbar.close()

        # Collect diagnostics: utility decomposition and predictive dists
        for action in initial_actions:
            # Utility decomposition: run one representative outcome
            stats = result.outcome_stats[action]
            w = self.utility_weights.get(focal_actor, {})
            decomp = {}

            vote_pct = stats["mean_vote_percent"]
            if vote_pct > 0.25:
                decomp["vote_penalty"] = -w.get("vote_penalty_weight", 0) * (vote_pct - 0.25) ** 2
                decomp["spill_risk"] = -w.get("spill_risk_weight", 0) * vote_pct
            else:
                decomp["vote_penalty"] = 0.0
                decomp["spill_risk"] = 0.0

            if stats["Pr_overwhelming"] > 0:
                decomp["overwhelming_penalty"] = -w.get("overwhelming_penalty_weight", 0) * stats["Pr_overwhelming"]
            else:
                decomp["overwhelming_penalty"] = 0.0

            decomp["review_car_impact"] = w.get("review_car_weight", 0) * stats["mean_review_car"]
            decomp["review_direct_cost"] = -w.get("review_direct_cost_weight", 0) * stats["mean_review_direct_cost"]

            if action == "D3_ceo_transition":
                decomp["implementation_cost_sack"] = -w.get("implementation_cost_sack", 0)
                decomp["ceo_loss"] = -w.get("ceo_loss_cost", 0)

            if stats["Pr_CEO_removed"] > 0 and action != "D3_ceo_transition":
                decomp["ceo_loss"] = -w.get("ceo_loss_cost", 0) * stats["Pr_CEO_removed"]

            result.utility_decomposition[action] = decomp

            # Sample predictive distribution at opponent nodes (first draw only)
            h = {initial_node: action}
            s = base_state.apply(initial_node, action)
            diag_rng = np.random.default_rng(self.seed + 999)

            node_dists = {}
            for node in base_state._node_order:
                if (base_state.node_type(node) == "decision"
                        and not mode.is_focal(base_state.node_owner(node))):
                    feasible = s.feasible_actions(node)
                    if len(feasible) > 1:
                        dist = predictive.predict(
                            node, h, s, 0, focal_actor, mode,
                            mode.level, diag_rng,
                        )
                        node_dists[f"{node} ({base_state.node_owner(node)})"] = dist

            if node_dists:
                result.predictive_dists[action] = node_dists

        # Find optimal
        best_action = max(result.EU_per_action, key=result.EU_per_action.get)
        result.optimal_action = best_action
        result.optimal_EU = result.EU_per_action[best_action]
        result.elapsed_seconds = time.time() - t0

        logger.info(
            f"Solved in {result.elapsed_seconds:.1f}s: "
            f"optimal={best_action} (EU={result.optimal_EU:.4f})"
        )

        return result

    def _find_checkpoint(self, checkpoint_id: str) -> Path:
        """Find checkpoint .npz file by ID."""
        pattern = f"belief_{checkpoint_id}_*.npz"
        matches = list(self.checkpoint_dir.glob(pattern))
        if not matches:
            # Also try exact match
            exact = self.checkpoint_dir / f"{checkpoint_id}.npz"
            if exact.exists():
                return exact
            raise FileNotFoundError(
                f"No checkpoint found for {checkpoint_id} in {self.checkpoint_dir}"
            )
        # Return most recent
        return sorted(matches)[-1]

    def predict_d0_ceo(
        self,
        focal_actor: str,
        checkpoint_id: str,
        mode: Optional[ModeConfig] = None,
        n_draws: Optional[int] = None,
        overconfidence_bias=_USE_SPEC_DEFAULT,
    ) -> dict[str, float]:
        """Compute the focal actor's predictive distribution over D0_ceo.

        Uses the ARA predictive distribution engine to model the CEO's
        resign/stay decision from the focal actor's perspective.

        Returns dict mapping D0_ceo action -> predicted probability,
        e.g. {"CEO_resign": 0.73, "CEO_stay": 0.27}.
        """
        # Resolve overconfidence bias
        if overconfidence_bias is _USE_SPEC_DEFAULT:
            effective_bias = self.overconfidence_bias
        else:
            effective_bias = overconfidence_bias

        # Default mode
        if mode is None:
            mode = AVAILABLE_MODES.get(focal_actor.lower())

        # Load checkpoint
        checkpoint_path = self._find_checkpoint(checkpoint_id)
        beliefs = BeliefBundle(checkpoint_path)

        # Setup engine components
        chance_models = ChanceModels(self.vote_thresholds)
        predictive = PredictiveDistribution(
            beliefs=beliefs,
            param_sampler=self.param_sampler,
            chance_models=chance_models,
            policy_params=self.policy_params,
            K=self.K,
            R_rollouts=self.R_rollouts,
            overconfidence_bias=effective_bias,
        )

        # Base state (before D0_ceo)
        base_state = DecisionState.from_governance_spec(
            self.governance_spec_path,
            checkpoint_id=checkpoint_id,
        )

        N = beliefs.N
        if n_draws is not None:
            N = min(n_draws, N)

        # Average predictive distribution across belief draws
        accum = {}
        for i in range(N):
            rng = np.random.default_rng(self.seed + i)
            pred = predictive.predict(
                "D0_ceo", {}, base_state, i, focal_actor, mode,
                mode.level, rng,
            )
            for action, prob in pred.items():
                accum[action] = accum.get(action, 0.0) + prob

        return {a: p / N for a, p in accum.items()}

    def solve_scenarios(
        self,
        focal_actor: str,
        checkpoint_id: str,
        mode: Optional[ModeConfig] = None,
        n_draws: Optional[int] = None,
        overconfidence_bias=_USE_SPEC_DEFAULT,
    ) -> dict[str, SolveResult]:
        """Solve for both CEO resignation scenarios and return results.

        Computes the D0_ceo predictive distribution (CEO's predicted
        resign/stay probability from the focal actor's perspective),
        then solves the game tree conditional on each scenario.

        Returns dict mapping scenario name -> SolveResult.
        Each SolveResult includes scenario_prob and d0_ceo_predictive.
        """
        # Compute D0_ceo predictive distribution
        d0_pred = self.predict_d0_ceo(
            focal_actor=focal_actor,
            checkpoint_id=checkpoint_id,
            mode=mode,
            n_draws=n_draws,
            overconfidence_bias=overconfidence_bias,
        )

        # Map scenario names to D0_ceo actions
        scenario_action_map = {
            "ceo_stayed": "CEO_stay",
            "ceo_resigned": "CEO_resign",
        }

        logger.info(
            f"D0_ceo predictive: "
            + ", ".join(f"{a}={p:.1%}" for a, p in d0_pred.items())
        )

        results = {}
        for scenario in ["ceo_stayed", "ceo_resigned"]:
            result = self.solve(
                focal_actor=focal_actor,
                checkpoint_id=checkpoint_id,
                mode=mode,
                n_draws=n_draws,
                overconfidence_bias=overconfidence_bias,
                scenario=scenario,
            )
            # Attach D0_ceo predictive info
            d0_action = scenario_action_map[scenario]
            result.scenario_prob = d0_pred.get(d0_action, 0.0)
            result.d0_ceo_predictive = d0_pred
            results[scenario] = result

        return results

    def solve_all_checkpoints(
        self,
        focal_actor: str,
        checkpoint_ids: Optional[list[str]] = None,
        mode: Optional[ModeConfig] = None,
        n_draws: Optional[int] = None,
        overconfidence_bias=_USE_SPEC_DEFAULT,
        scenario: str = "ceo_stayed",
    ) -> pd.DataFrame:
        """
        Solve for multiple checkpoints and return combined summary.
        """
        if checkpoint_ids is None:
            checkpoint_ids = ["C0", "C1", "C2", "C3"]

        results = []
        for cid in checkpoint_ids:
            try:
                result = self.solve(
                    focal_actor=focal_actor,
                    checkpoint_id=cid,
                    mode=mode,
                    n_draws=n_draws,
                    overconfidence_bias=overconfidence_bias,
                    scenario=scenario,
                )
                results.append(result.summary_df())
            except FileNotFoundError:
                logger.warning(f"Checkpoint {cid} not found, skipping")

        if not results:
            return pd.DataFrame()
        return pd.concat(results, ignore_index=True)
