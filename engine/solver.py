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
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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


# ---------------------------------------------------------------------------
# Worker process state — initialized once per worker via ProcessPoolExecutor
# initializer, eliminating per-task file I/O (Excel + .npz reads).
# Both "solve" (full K/R) and "predict" (K_d0/R_d0) engines are built
# in a single init so the pool can be reused across predict_d0_ceo + solve.
# ---------------------------------------------------------------------------
_worker_engine: dict = {}


def _init_worker_engine(
    belief_data: dict,
    sampler_data: dict,
    state_init_data: dict,
    vote_thresholds: dict,
    utility_weights: dict,
    policy_params: dict,
    overconfidence_bias_dict: dict | None,
    K: int,
    R_rollouts: int,
    n_vote_samples: int,
    n_review_samples: int,
    K_d0_ceo: int,
    R_d0_ceo: int,
    no_prior_actors: set[str] | None = None,
) -> None:
    """One-time per-worker initialization — builds both solve and predict engines."""
    global _worker_engine

    # Constrain intra-process threading so many processes can run concurrently.
    # If the user already set these, respect their choices.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    beliefs = BeliefBundle.from_dict(belief_data)
    param_sampler = ParameterSampler.from_dict(sampler_data)
    effective_bias = OverconfidenceBias(**overconfidence_bias_dict) if overconfidence_bias_dict else None
    chance_models = ChanceModels(vote_thresholds)

    effective_no_prior = no_prior_actors or set()

    # Full engine for solve() — K opponent samples, R rollouts per action
    predictive_solve = PredictiveDistribution(
        beliefs=beliefs,
        param_sampler=param_sampler,
        chance_models=chance_models,
        policy_params=policy_params,
        K=K,
        R_rollouts=R_rollouts,
        overconfidence_bias=effective_bias,
        no_prior_actors=effective_no_prior,
    )
    tree = TreeEvaluator(
        beliefs=beliefs,
        chance_models=chance_models,
        predictive=predictive_solve,
        utility_weights=utility_weights,
        n_vote_samples=n_vote_samples,
        n_review_samples=n_review_samples,
        overconfidence_bias=effective_bias,
    )

    # Lightweight engine for predict_d0_ceo() — reduced K/R for tractability
    predictive_d0 = PredictiveDistribution(
        beliefs=beliefs,
        param_sampler=param_sampler,
        chance_models=chance_models,
        policy_params=policy_params,
        K=K_d0_ceo,
        R_rollouts=R_d0_ceo,
        overconfidence_bias=effective_bias,
        no_prior_actors=effective_no_prior,
    )

    _worker_engine = {
        "beliefs": beliefs,
        "predictive_solve": predictive_solve,
        "predictive_d0": predictive_d0,
        "tree": tree,
        "state_init_data": state_init_data,
    }


def _warmup() -> bool:
    """No-op task used to force worker process spawning during pool warmup."""
    return True


def _evaluate_single_draw(
    draw_i: int,
    action: str,
    initial_node: str,
    scenario: str,
    focal_actor: str,
    mode_dict: dict,
    seed: int,
    checkpoint_id: str,
) -> dict:
    """Evaluate a single belief draw using pre-initialized worker engine.

    Engine components (beliefs, tree, predictive) are constructed once per
    worker process by _init_worker_engine — no file I/O per task.
    """
    e = _worker_engine
    mode = ModeConfig(**mode_dict)

    beliefs = e["beliefs"]
    tree = e["tree"]
    predictive = e["predictive_solve"]

    base_state = DecisionState.from_init_dict(e["state_init_data"], checkpoint_id=checkpoint_id)
    base_state = base_state.for_scenario(scenario)

    N = beliefs.N
    rng = np.random.default_rng(seed + draw_i)

    # Apply initial action
    h = {initial_node: action}
    s = base_state.apply(initial_node, action)
    next_node = s.next_node(initial_node)

    # Evaluate tree
    v = tree.value(next_node, h, s, draw_i, mode, rng)

    # Collect outcome statistics via a single rollout
    out_rng = np.random.default_rng(seed + draw_i + N)
    outcome = predictive._simulate_forward(
        current_node=next_node,
        history=h,
        state=s,
        draw_i=draw_i,
        owner=focal_actor,
        focal_actor=focal_actor,
        mode=mode,
        level=mode.level,
        rng=out_rng,
    )

    return {
        "action": action,
        "draw_i": draw_i,
        "value": v,
        "vote_percent": outcome.vote_percent,
        "strike": int(outcome.strike_indicator),
        "overwhelming": int(outcome.overwhelming_indicator),
        "CEO_removed": int(outcome.CEO_removed),
        "review_outcome": outcome.review_outcome,
        "review_car": outcome.review_car,
        "review_direct_cost": outcome.review_direct_cost,
    }


def _predict_single_draw(
    draw_i: int,
    focal_actor: str,
    mode_dict: dict,
    seed: int,
) -> dict[str, float]:
    """Compute one draw's predictive distribution over D0_ceo.

    Uses pre-initialized worker engine — no file I/O per task.
    """
    e = _worker_engine
    mode = ModeConfig(**mode_dict)
    predictive = e["predictive_d0"]

    base_state = DecisionState.from_init_dict(e["state_init_data"])

    rng = np.random.default_rng(seed + draw_i)
    return predictive.predict(
        "D0_ceo", {}, base_state, draw_i, focal_actor, mode,
        mode.level, rng,
    )


def _predict_diagnostic_draw(
    action: str,
    node: str,
    initial_node: str,
    draw_i: int,
    focal_actor: str,
    mode_dict: dict,
    seed: int,
    scenario: str,
    checkpoint_id: str,
) -> dict:
    """Compute one diagnostic predictive distribution in a worker process.

    Reconstructs history and state from initial_action + scenario, then
    calls predictive.predict() at the target opponent node.
    """
    e = _worker_engine
    mode = ModeConfig(**mode_dict)
    predictive = e["predictive_solve"]

    base_state = DecisionState.from_init_dict(
        e["state_init_data"], checkpoint_id=checkpoint_id,
    )
    base_state = base_state.for_scenario(scenario)

    h = {initial_node: action}
    s = base_state.apply(initial_node, action)

    rng = np.random.default_rng(seed + 999 + draw_i)
    dist = predictive.predict(
        node, h, s, draw_i, focal_actor, mode,
        mode.level, rng,
    )
    return {
        "action": action,
        "node": node,
        "draw_i": draw_i,
        "dist": dist,
    }


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

    # Utility decomposition per action (focal actor components)
    utility_decomposition: dict[str, dict[str, float]] = field(default_factory=dict)

    # ASA utility decomposition per D1 action (for A2 node display)
    asa_utility_decomposition: dict[str, dict[str, float]] = field(default_factory=dict)

    # CEO utility decomposition per D1 action (for D4 node display)
    ceo_utility_decomposition: dict[str, dict[str, float]] = field(default_factory=dict)

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

    def display_summary(self) -> str:
        """Return formatted summary string with proper decimal precision."""
        df = self.summary_df()
        formatters = {}
        for col in df.columns:
            if col.startswith("Pr_"):
                formatters[col] = "{:.2f}".format
            elif col == "Expected_Utility":
                formatters[col] = "{:+.4f}".format
            elif col.startswith("mean_") or col.startswith("sd_"):
                formatters[col] = "{:.4f}".format
        return df.to_string(index=False, formatters=formatters)

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
                      f"Review: neg={s['Pr_review_negative']:.0%} "
                      f"bal={s['Pr_review_balanced']:.0%} "
                      f"pos={s['Pr_review_positive']:.0%}")
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
        n_workers: int | None = None,
        K_d0_ceo: int = 50,
        R_d0_ceo: int = 10,
        no_prior_actors: set[str] | None = None,
    ):
        self.governance_spec_path = Path(governance_spec_path)
        self.opponent_priors_path = Path(opponent_priors_path)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.K = K
        self.R_rollouts = R_rollouts
        self.n_vote_samples = n_vote_samples
        self.n_review_samples = n_review_samples
        self.seed = seed
        self.n_workers = n_workers if n_workers is not None else max((os.cpu_count() or 2) - 1, 1)
        self.K_d0_ceo = K_d0_ceo
        self.R_d0_ceo = R_d0_ceo
        # Actors whose predictive distributions should NOT receive Laplace
        # smoothing (Dirichlet(1,...,1) prior).  Default: all actors get prior.
        self.no_prior_actors: set[str] = no_prior_actors or set()

        # Bayesian prior for D0_ceo prediction: Beta(alpha, beta).
        # Jeffreys prior Beta(0.5, 0.5) updated with 12 Australian observations
        # of no-remorse CEOs facing moral-reputational crises — all 12 resigned.
        # Posterior: Beta(0.5+12, 0.5+0) = Beta(12.5, 0.5).
        # Posterior mean = 12.5/13 ≈ 0.962. See ESG-and-CEO-turnover.md.
        self.ceo_departure_prior_alpha = 12.5
        self.ceo_departure_prior_beta = 0.5

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

        # Persistent process pool — created lazily, reused across
        # predict_d0_ceo + solve calls for the same checkpoint.
        self._pool: ProcessPoolExecutor | None = None
        self._pool_checkpoint_id: str | None = None

    def _ensure_pool(
        self,
        checkpoint_id: str,
        beliefs: BeliefBundle,
        base_state: DecisionState,
        bias_dict: dict | None,
    ) -> ProcessPoolExecutor:
        """Get or create a persistent process pool for the given checkpoint.

        Creates the pool on first call, reuses it for subsequent calls with the
        same checkpoint. Workers are initialized with both solve and predict
        engine configurations so the pool serves predict_d0_ceo + solve.
        """
        if self._pool is not None and self._pool_checkpoint_id == checkpoint_id:
            return self._pool

        # Shut down old pool if checkpoint changed
        if self._pool is not None:
            logger.info("Shutting down previous worker pool (checkpoint changed)...")
            self._pool.shutdown(wait=False)
            self._pool = None

        logger.info(f"Starting {self.n_workers} worker processes...")
        t0 = time.time()

        init_args = (
            beliefs.to_dict(),
            self.param_sampler.to_dict(),
            base_state.to_init_dict(),
            self.vote_thresholds,
            self.utility_weights,
            self.policy_params,
            bias_dict,
            self.K,
            self.R_rollouts,
            self.n_vote_samples,
            self.n_review_samples,
            self.K_d0_ceo,
            self.R_d0_ceo,
            self.no_prior_actors,
        )

        self._pool = ProcessPoolExecutor(
            max_workers=self.n_workers,
            initializer=_init_worker_engine,
            initargs=init_args,
        )

        # Force all workers to spawn and initialize now (not lazily on first task).
        # This makes the startup time visible to the user upfront rather than
        # appearing as a mysterious delay after the progress bar is shown.
        warmup_futures = [self._pool.submit(_warmup) for _ in range(self.n_workers)]
        for f in warmup_futures:
            f.result()

        dt = time.time() - t0
        logger.info(f"Workers ready in {dt:.1f}s")
        self._pool_checkpoint_id = checkpoint_id
        return self._pool

    def shutdown_pool(self) -> None:
        """Shut down the persistent process pool."""
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
            self._pool_checkpoint_id = None

    def __del__(self) -> None:
        self.shutdown_pool()

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
            no_prior_actors=self.no_prior_actors,
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
            "gamma_AH_mean": float(np.mean(beliefs.gamma_AH[:N])),
            "gamma_D_mean": float(np.mean(beliefs.gamma_D[:N])),
            "sigma_vote_mean": float(np.mean(beliefs.sigma_vote[:N])),
        }

        n_actions = len(initial_actions)
        total_evals = n_actions * N

        # Serialise mode and bias for worker processes
        mode_dict = {
            "name": mode.name,
            "focal_actor": mode.focal_actor,
            "opponent_models": mode.opponent_models,
            "level": mode.level,
            "strategic_counterparts": mode.strategic_counterparts,
        }
        bias_dict = None
        if effective_bias is not None:
            bias_dict = {
                "d1_floor": effective_bias.d1_floor,
                "d1_ceiling": effective_bias.d1_ceiling,
                "d3_floor": effective_bias.d3_floor,
                "d3_ceiling": effective_bias.d3_ceiling,
                "sigma_scale": effective_bias.sigma_scale,
                "review_car_bias": effective_bias.review_car_bias,
            }

        use_parallel = self.n_workers > 1 and total_evals > 1

        if use_parallel:
            self._solve_parallel(
                result, initial_actions, N, beliefs,
                base_state, mode_dict, bias_dict,
                focal_actor, checkpoint_id, initial_node, scenario,
            )
        else:
            self._solve_sequential(
                result, initial_actions, N,
                tree, predictive, base_state, mode,
                focal_actor, initial_node, checkpoint_id,
            )

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

        # Sample predictive distributions at opponent nodes, averaged over
        # multiple belief draws for a representative estimate (not just draw_i=0).
        # Pre-scan to count predict() calls so tqdm can show accurate progress.
        n_diag = min(5, N)
        diag_tasks: list[tuple[str, str]] = []  # (action, node)
        for action in initial_actions:
            s_a = base_state.apply(initial_node, action)
            for node in base_state._node_order:
                if (base_state.node_type(node) == "decision"
                        and not mode.is_focal(base_state.node_owner(node))):
                    if len(s_a.feasible_actions(node)) > 1:
                        diag_tasks.append((action, node))

        total_predict_calls = len(diag_tasks) * n_diag
        action_node_dists: dict[str, dict] = {a: {} for a in initial_actions}

        use_parallel_diag = self.n_workers > 1 and total_predict_calls > 1

        from tqdm import tqdm as _tqdm
        if use_parallel_diag:
            executor = self._ensure_pool(
                checkpoint_id, beliefs, base_state, bias_dict,
            )
            futures = {}
            for action, node in diag_tasks:
                for di in range(n_diag):
                    future = executor.submit(
                        _predict_diagnostic_draw,
                        action, node, initial_node, di,
                        focal_actor, mode_dict, self.seed,
                        scenario, checkpoint_id,
                    )
                    futures[future] = (action, node, di)

            # Accumulators: (action, node) -> {response_action: sum_of_probs}
            agg_map: dict[tuple[str, str], dict[str, float]] = {}

            with _tqdm(
                total=total_predict_calls,
                desc=f"Diagnostics ({scenario}, {focal_actor}) "
                     f"[{self.n_workers} workers]",
                unit="predict",
                smoothing=0,
            ) as pbar:
                for future in as_completed(futures):
                    r = future.result()
                    key = (r["action"], r["node"])
                    if key not in agg_map:
                        agg_map[key] = {}
                    for a, p in r["dist"].items():
                        agg_map[key][a] = agg_map[key].get(a, 0.0) + p / n_diag
                    pbar.update(1)

            for action, node in diag_tasks:
                owner = base_state.node_owner(node)
                agg = agg_map.get((action, node), {})
                action_node_dists[action][f"{node} ({owner})"] = agg
        else:
            with _tqdm(
                total=total_predict_calls,
                desc=f"Diagnostics ({scenario}, {focal_actor})",
                unit="predict",
                smoothing=0,
            ) as pbar:
                for action, node in diag_tasks:
                    owner = base_state.node_owner(node)
                    h = {initial_node: action}
                    s = base_state.apply(initial_node, action)
                    agg: dict[str, float] = {}
                    for di in range(n_diag):
                        diag_rng = np.random.default_rng(self.seed + 999 + di)
                        d_i = predictive.predict(
                            node, h, s, di, focal_actor, mode,
                            mode.level, diag_rng,
                        )
                        for a, p in d_i.items():
                            agg[a] = agg.get(a, 0.0) + p / n_diag
                        pbar.set_postfix(action=action, node=node,
                                         draw=f"{di+1}/{n_diag}")
                        pbar.update(1)
                    action_node_dists[action][f"{node} ({owner})"] = agg

        w_asa = self.utility_weights.get("ASA", {})
        for action in initial_actions:
            if action_node_dists[action]:
                result.predictive_dists[action] = action_node_dists[action]

            # ASA utility decomposition (for A2 node display) — uses ASA weights,
            # opposite signs from Board where appropriate
            stats_a = result.outcome_stats[action]
            mean_vp = stats_a["mean_vote_percent"]
            asa_d: dict[str, float] = {}
            asa_d["vote_reward"] = w_asa.get("vote_reward_weight", 2.0) * mean_vp
            if stats_a["Pr_overwhelming"] > 0:
                asa_d["overwhelming_reward"] = (
                    w_asa.get("overwhelming_reward_weight", 2.0) * stats_a["Pr_overwhelming"]
                )
            if stats_a["Pr_CEO_removed"] > 0:
                asa_d["ceo_removal_reward"] = (
                    w_asa.get("ceo_removal_reward", 3.0) * stats_a["Pr_CEO_removed"]
                )
            # ASA benefits from NEGATIVE review CAR (adverse findings vindicate governance)
            mean_car = stats_a.get("mean_review_car", 0.0)
            asa_d["review_car_impact"] = -w_asa.get("review_car_weight", 15.0) * mean_car
            # Mobilisation cost — weighted by Pr(rec_strike) from predictive distribution
            a2_pred = action_node_dists[action].get("A2 (ASA)", {})
            pr_rec = a2_pred.get("A2_rec_strike", 0.0)
            asa_d["mobilisation_cost"] = -w_asa.get("mobilisation_cost", 0.3) * pr_rec
            if mean_vp > 0.25:
                asa_d["reputational_gain"] = (
                    w_asa.get("reputational_gain_weight", 1.0) * (mean_vp - 0.25)
                )
            asa_d["market_alignment_bonus"] = (
                w_asa.get("market_alignment_bonus", 1.5) * pr_rec * stats_a["Pr_strike"]
            )
            result.asa_utility_decomposition[action] = asa_d

        # CEO utility decomposition (for D4 node display) — uses CEO weights,
        # breaks down reference-dependent CRRA utility into monetary and
        # non-monetary (D) penalty components.
        w_ceo = self.utility_weights.get("CEO", {})
        ceo_gamma = max(0.5, min(3.0, w_ceo.get("gamma", 1.5)))
        ceo_la = w_ceo.get("loss_aversion", 2.25)
        ceo_la_D = w_ceo.get("loss_aversion_D", ceo_la)
        W_ref = max(w_ceo.get("W_ref", 16.0), 0.01)

        for action in initial_actions:
            stats_a = result.outcome_stats[action]
            ceo_d: dict[str, float] = {}

            pr_removed = stats_a["Pr_CEO_removed"]
            pr_strike = stats_a["Pr_strike"]
            pr_ovw = stats_a["Pr_overwhelming"]
            pr_negative = stats_a.get("Pr_review_negative", 0.0)

            # D_stay baseline (crisis cost for choosing to stay)
            D_stay = w_ceo.get("D_stay", 25.0)
            ceo_d["D_stay_baseline"] = -ceo_la_D * D_stay

            # D_agm: first-strike penalty (vote > 25%)
            if pr_strike > 0:
                ceo_d["D_agm_first_strike"] = (
                    -ceo_la_D * w_ceo.get("D_agm", 30.0) * pr_strike
                )

            # D_disgrace: overwhelming vote penalty
            if pr_ovw > 0:
                ceo_d["D_disgrace_overwhelming"] = (
                    -ceo_la_D * w_ceo.get("D_disgrace", 30.0) * pr_ovw
                )

            # D_departure: weighted by D4 predictive distribution
            if pr_removed > 0:
                D_sacked = w_ceo.get("D_sacked", 100.0)
                D_resign_late = w_ceo.get("D_resign_late", 60.0)
                D_negotiate = w_ceo.get("D_negotiate", 45.0)
                d4_pred = action_node_dists[action].get("D4 (CEO)", {})
                pr_d4_stay = d4_pred.get("D4_stay", 0.0)
                pr_d4_resign = d4_pred.get("D4_resign", 0.0)
                pr_d4_negotiate = d4_pred.get("D4_negotiate_exit", 0.0)
                d4_total = pr_d4_stay + pr_d4_resign + pr_d4_negotiate
                if d4_total > 0:
                    weighted_D = (
                        pr_d4_stay * D_sacked
                        + pr_d4_resign * D_resign_late
                        + pr_d4_negotiate * D_negotiate
                    ) / d4_total
                else:
                    weighted_D = D_sacked  # fallback: sacked
                ceo_d["D_departure_penalty"] = -ceo_la_D * weighted_D * pr_removed

            # D_adverse_review (triggered by negative review outcome)
            if pr_negative > 0:
                ceo_d["D_adverse_review"] = (
                    -ceo_la_D * w_ceo.get("D_adverse_review", 10.0) * pr_negative
                )

            # Monetary component: E[U_money] ≈ Pr(kept)*CRRA(W_kept) + Pr(removed)*CRRA(W_removed)
            W_kept = max(w_ceo.get("W_stay_kept", 7.0), 0.01)
            W_sacked = max(w_ceo.get("W_stay_sacked", 0.5), 0.01)
            # Approximate W_removed using D4 predictive for departure mode
            if pr_removed > 0 and d4_total > 0:
                W_negotiate = (W_sacked + W_kept) / 2.0
                W_resign_late = W_sacked * 1.3
                W_removed = (
                    pr_d4_stay * W_sacked
                    + pr_d4_resign * W_resign_late
                    + pr_d4_negotiate * W_negotiate
                ) / d4_total
            else:
                W_removed = W_sacked

            def _crra_money(W_val):
                W_val = max(W_val, 0.01)
                if abs(ceo_gamma - 1.0) < 1e-6:
                    crra_w = np.log(W_val)
                    crra_r = np.log(W_ref)
                else:
                    crra_w = (W_val ** (1.0 - ceo_gamma)) / (1.0 - ceo_gamma)
                    crra_r = (W_ref ** (1.0 - ceo_gamma)) / (1.0 - ceo_gamma)
                if W_val >= W_ref:
                    return crra_w
                return ceo_la * crra_w - (ceo_la - 1.0) * crra_r

            U_kept = _crra_money(W_kept)
            U_removed = _crra_money(W_removed)
            ceo_d["U_money_wealth"] = (1 - pr_removed) * U_kept + pr_removed * U_removed

            result.ceo_utility_decomposition[action] = ceo_d

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

    def _solve_sequential(
        self,
        result: SolveResult,
        initial_actions: list[str],
        N: int,
        tree: TreeEvaluator,
        predictive: PredictiveDistribution,
        base_state: DecisionState,
        mode: ModeConfig,
        focal_actor: str,
        initial_node: str,
        checkpoint_id: str,
    ) -> None:
        """Run the belief-draw loop sequentially (n_workers=1)."""
        n_actions = len(initial_actions)
        total_evals = n_actions * N
        pbar = tqdm(
            total=total_evals,
            desc=f"Solving ({focal_actor}, {checkpoint_id})",
            unit="draw",
            smoothing=0,
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
                "review_negative_prob": 0,
                "review_balanced_prob": 0,
                "review_positive_prob": 0,
                "review_car": [],
                "review_direct_cost": [],
            }

            for i in range(N):
                rng = np.random.default_rng(self.seed + i)

                h = {initial_node: action}
                s = base_state.apply(initial_node, action)
                next_node = s.next_node(initial_node)

                v = tree.value(next_node, h, s, i, mode, rng)
                values.append(v)

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
                outcome_accum["review_negative_prob"] += int(outcome.review_outcome == "negative")
                outcome_accum["review_balanced_prob"] += int(outcome.review_outcome == "balanced")
                outcome_accum["review_positive_prob"] += int(outcome.review_outcome == "positive")
                outcome_accum["review_car"].append(outcome.review_car)
                outcome_accum["review_direct_cost"].append(outcome.review_direct_cost)

                pbar.update(1)

            eu = float(np.mean(values))
            result.EU_per_action[action] = eu
            result.draw_values[action] = values

            result.outcome_stats[action] = {
                "Pr_strike": outcome_accum["strike_prob"] / N,
                "Pr_overwhelming": outcome_accum["overwhelming_prob"] / N,
                "Pr_CEO_removed": outcome_accum["CEO_removed_prob"] / N,
                "Pr_review_negative": outcome_accum["review_negative_prob"] / N,
                "Pr_review_balanced": outcome_accum["review_balanced_prob"] / N,
                "Pr_review_positive": outcome_accum["review_positive_prob"] / N,
                "mean_vote_percent": float(np.mean(outcome_accum["vote_percent"])),
                "sd_vote_percent": float(np.std(outcome_accum["vote_percent"])),
                "mean_review_car": float(np.mean(outcome_accum["review_car"])),
                "sd_review_car": float(np.std(outcome_accum["review_car"])),
                "mean_review_direct_cost": float(np.mean(outcome_accum["review_direct_cost"])),
                "sd_review_direct_cost": float(np.std(outcome_accum["review_direct_cost"])),
            }

            tqdm.write(
                f"  Action {action_idx+1}/{n_actions} '{action}': "
                f"EU={eu:.4f}"
            )

        pbar.close()

    def _solve_parallel(
        self,
        result: SolveResult,
        initial_actions: list[str],
        N: int,
        beliefs: BeliefBundle,
        base_state: DecisionState,
        mode_dict: dict,
        bias_dict: dict | None,
        focal_actor: str,
        checkpoint_id: str,
        initial_node: str,
        scenario: str,
    ) -> None:
        """Run the belief-draw loop in parallel across worker processes.

        Uses the persistent pool from _ensure_pool() — workers are already
        initialized from a previous call (or created now if this is the first).
        """
        n_actions = len(initial_actions)
        total_evals = n_actions * N

        # Pre-allocate accumulators per action
        action_values: dict[str, list] = {a: [] for a in initial_actions}
        action_outcomes: dict[str, dict] = {
            a: {
                "vote_percent": [],
                "strike_prob": 0,
                "overwhelming_prob": 0,
                "CEO_removed_prob": 0,
                "review_negative_prob": 0,
                "review_balanced_prob": 0,
                "review_positive_prob": 0,
                "review_car": [],
                "review_direct_cost": [],
            }
            for a in initial_actions
        }

        executor = self._ensure_pool(checkpoint_id, beliefs, base_state, bias_dict)

        pbar = tqdm(
            total=total_evals,
            desc=f"Solving ({focal_actor}, {checkpoint_id}) [{self.n_workers} workers]",
            unit="draw",
            smoothing=0,
        )

        futures = {}
        for action in initial_actions:
            for i in range(N):
                future = executor.submit(
                    _evaluate_single_draw,
                    i, action, initial_node, scenario,
                    focal_actor, mode_dict, self.seed, checkpoint_id,
                )
                futures[future] = (action, i)

        for future in as_completed(futures):
            r = future.result()
            action = r["action"]
            action_values[action].append(r["value"])
            acc = action_outcomes[action]
            acc["vote_percent"].append(r["vote_percent"])
            acc["strike_prob"] += r["strike"]
            acc["overwhelming_prob"] += r["overwhelming"]
            acc["CEO_removed_prob"] += r["CEO_removed"]
            r_outcome = r["review_outcome"]
            acc["review_negative_prob"] += int(r_outcome == "negative")
            acc["review_balanced_prob"] += int(r_outcome == "balanced")
            acc["review_positive_prob"] += int(r_outcome == "positive")
            acc["review_car"].append(r["review_car"])
            acc["review_direct_cost"].append(r["review_direct_cost"])
            pbar.update(1)

        pbar.close()

        # Aggregate results per action
        for action_idx, action in enumerate(initial_actions):
            values = action_values[action]
            eu = float(np.mean(values))
            result.EU_per_action[action] = eu
            result.draw_values[action] = values

            acc = action_outcomes[action]
            result.outcome_stats[action] = {
                "Pr_strike": acc["strike_prob"] / N,
                "Pr_overwhelming": acc["overwhelming_prob"] / N,
                "Pr_CEO_removed": acc["CEO_removed_prob"] / N,
                "Pr_review_negative": acc["review_negative_prob"] / N,
                "Pr_review_balanced": acc["review_balanced_prob"] / N,
                "Pr_review_positive": acc["review_positive_prob"] / N,
                "mean_vote_percent": float(np.mean(acc["vote_percent"])),
                "sd_vote_percent": float(np.std(acc["vote_percent"])),
                "mean_review_car": float(np.mean(acc["review_car"])),
                "sd_review_car": float(np.std(acc["review_car"])),
                "mean_review_direct_cost": float(np.mean(acc["review_direct_cost"])),
                "sd_review_direct_cost": float(np.std(acc["review_direct_cost"])),
            }

            tqdm.write(
                f"  Action {action_idx+1}/{n_actions} '{action}': "
                f"EU={eu:.4f}"
            )

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

        Uses Level-2 ARA: the CEO strategically models the Board's likely
        D1 action via a nested predict() call, rather than assuming the
        Board uses a fixed policy (which defaults to D0_minimal and makes
        resignation appear irrational).

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

        # Construct Level-2 mode for D0_ceo prediction.
        # Key: "Board" in strategic_counterparts lets the CEO's rollouts
        # trigger predict() at Board decision nodes instead of fixed policy.
        d0_mode = ModeConfig(
            name=f"{mode.name} (D0_ceo L2)",
            focal_actor=focal_actor,
            opponent_models={"ASA": "ARA", "CEO": "ARA", "Board": "ARA"},
            level=2,
            strategic_counterparts={"Board": "CEO", "ASA": "Board", "CEO": "Board"},
        )

        # Load checkpoint
        checkpoint_path = self._find_checkpoint(checkpoint_id)
        beliefs = BeliefBundle(checkpoint_path)

        # Setup engine components with reduced K/R for tractability.
        # The nested Board prediction at D1 reuses the same K/R.
        chance_models = ChanceModels(self.vote_thresholds)
        predictive = PredictiveDistribution(
            beliefs=beliefs,
            param_sampler=self.param_sampler,
            chance_models=chance_models,
            policy_params=self.policy_params,
            K=self.K_d0_ceo,
            R_rollouts=self.R_d0_ceo,
            overconfidence_bias=effective_bias,
            no_prior_actors=self.no_prior_actors,
        )

        # Base state (before D0_ceo)
        base_state = DecisionState.from_governance_spec(
            self.governance_spec_path,
            checkpoint_id=checkpoint_id,
        )

        N = beliefs.N
        if n_draws is not None:
            N = min(n_draws, N)

        logger.info(
            f"D0_ceo Level-2 prediction: K={self.K_d0_ceo}, R={self.R_d0_ceo}, "
            f"N={N}, focal={focal_actor}, "
            f"prior=Beta({self.ceo_departure_prior_alpha}, {self.ceo_departure_prior_beta})"
        )

        # Average predictive distribution across belief draws
        accum = {}

        if self.n_workers > 1 and N > 1:
            # Serialise for workers
            d0_mode_dict = {
                "name": d0_mode.name,
                "focal_actor": d0_mode.focal_actor,
                "opponent_models": d0_mode.opponent_models,
                "level": d0_mode.level,
                "strategic_counterparts": d0_mode.strategic_counterparts,
            }
            bias_dict = None
            if effective_bias is not None:
                bias_dict = {
                    "d1_floor": effective_bias.d1_floor,
                    "d1_ceiling": effective_bias.d1_ceiling,
                    "d3_floor": effective_bias.d3_floor,
                    "d3_ceiling": effective_bias.d3_ceiling,
                    "sigma_scale": effective_bias.sigma_scale,
                    "review_car_bias": effective_bias.review_car_bias,
                }

            # Use persistent pool (creates on first call, reuses after)
            executor = self._ensure_pool(checkpoint_id, beliefs, base_state, bias_dict)

            futures = {}
            pbar = tqdm(total=N, desc="D0_ceo L2 predict", unit="draw", smoothing=0)
            for i in range(N):
                futures[executor.submit(
                    _predict_single_draw,
                    i, focal_actor, d0_mode_dict, self.seed,
                )] = i
            for future in as_completed(futures):
                pred = future.result()
                for action, prob in pred.items():
                    accum[action] = accum.get(action, 0.0) + prob
                pbar.update(1)
            pbar.close()
        else:
            for i in tqdm(range(N), desc="D0_ceo L2 predict", unit="draw", smoothing=0):
                rng = np.random.default_rng(self.seed + i)
                pred = predictive.predict(
                    "D0_ceo", {}, base_state, i, focal_actor, d0_mode,
                    d0_mode.level, rng,
                )
                for action, prob in pred.items():
                    accum[action] = accum.get(action, 0.0) + prob

        # Bayesian updating: combine ARA evidence with Beta prior.
        # The accum sums are soft pseudo-counts (sum to N).
        # Adding Beta(alpha, beta) prior pseudo-counts gives a posterior
        # that weights the empirical departure rate (from ASX 100 crisis data)
        # against the game-theoretic ARA prediction.
        # See ceo-background/ESG-and-CEO-turnover.md for prior derivation.
        alpha = self.ceo_departure_prior_alpha  # 12.0 — departure pseudo-count
        beta = self.ceo_departure_prior_beta    # 1.5 — survival pseudo-count

        n_resign = accum.get("CEO_resign", 0.0)
        n_stay = accum.get("CEO_stay", 0.0)

        alpha_post = alpha + n_resign
        beta_post = beta + n_stay
        total_post = alpha_post + beta_post

        ara_only = {a: p / N for a, p in accum.items()} if N > 0 else accum
        posterior = {
            "CEO_resign": alpha_post / total_post,
            "CEO_stay": beta_post / total_post,
        }

        logger.info(
            f"D0_ceo Bayesian update: "
            f"prior=Beta({alpha:.1f},{beta:.1f}) [mean={alpha/(alpha+beta):.1%}], "
            f"ARA evidence (N={N}): resign={ara_only.get('CEO_resign', 0):.1%}, "
            f"posterior: resign={posterior['CEO_resign']:.1%}"
        )

        return posterior

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
