"""
Sensitivity analysis engine.

Grid over key parameters and re-run the solver to identify
how optimal policies shift under different assumptions.

Usage:
    python -m run.sensitivity --focal Board --checkpoint C0 --n_draws 20
"""

import argparse
import copy
import itertools
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine.solver import Solver
from engine.modes import AVAILABLE_MODES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# Default sensitivity grid
DEFAULT_GRID = {
    "vote_penalty_weight": [1.0, 2.0, 3.0],
    "mobilisation_cost": [0.5, 1.0, 2.0],
    "ceo_loss_cost": [0.5, 1.5, 3.0],
    "spill_risk_weight": [1.0, 2.5, 4.0],
}


def _run_single_sensitivity_combo(
    governance_spec_path: str,
    opponent_priors_path: str,
    checkpoint_dir: str,
    K: int,
    R_rollouts: int,
    seed: int,
    focal_actor: str,
    checkpoint_id: str,
    n_draws: int,
    overrides: dict[str, float],
) -> dict:
    """Run one sensitivity combo in a worker process."""
    solver = Solver(
        governance_spec_path=governance_spec_path,
        opponent_priors_path=opponent_priors_path,
        checkpoint_dir=checkpoint_dir,
        K=K,
        R_rollouts=R_rollouts,
        seed=seed,
        n_workers=1,  # No nested parallelism
    )

    # Apply overrides
    for pname, pval in overrides.items():
        for actor_weights in solver.utility_weights.values():
            if pname in actor_weights:
                actor_weights[pname] = pval

    mode = AVAILABLE_MODES.get(focal_actor.lower())
    result = solver.solve(
        focal_actor=focal_actor,
        checkpoint_id=checkpoint_id,
        mode=mode,
        n_draws=n_draws,
    )
    row = {
        "checkpoint": checkpoint_id,
        "focal": focal_actor,
        "optimal_action": result.optimal_action,
        "optimal_EU": result.optimal_EU,
    }
    row.update(overrides)
    for action, eu in result.EU_per_action.items():
        row[f"EU_{action}"] = eu
    return row


def run_sensitivity(
    solver: Solver,
    focal_actor: str,
    checkpoint_id: str,
    param_grid: dict[str, list[float]],
    n_draws: int = 20,
    n_workers: int | None = None,
) -> pd.DataFrame:
    """
    Run sensitivity analysis over a parameter grid.

    For each combination in the grid, overrides the corresponding utility weight
    and re-runs the solver.
    """
    if n_workers is None:
        n_workers = solver.n_workers

    mode = AVAILABLE_MODES.get(focal_actor.lower())

    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))

    results = []
    total = len(combinations)

    if n_workers > 1 and total > 1:
        logger.info(f"Running {total} sensitivity combos across {n_workers} workers")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {}
            for idx, combo in enumerate(combinations):
                overrides = dict(zip(param_names, combo))
                future = executor.submit(
                    _run_single_sensitivity_combo,
                    str(solver.governance_spec_path),
                    str(solver.opponent_priors_path),
                    str(solver.checkpoint_dir),
                    solver.K,
                    solver.R_rollouts,
                    solver.seed,
                    focal_actor,
                    checkpoint_id,
                    n_draws,
                    overrides,
                )
                futures[future] = idx

            for future in as_completed(futures):
                idx = futures[future]
                results.append(future.result())
                logger.info(f"Sensitivity combo {idx + 1}/{total} complete")
    else:
        for idx, combo in enumerate(combinations):
            logger.info(f"Sensitivity run {idx + 1}/{total}")

            # Override utility weights
            overrides = dict(zip(param_names, combo))
            original_weights = {}
            for pname, pval in overrides.items():
                for actor_weights in solver.utility_weights.values():
                    if pname in actor_weights:
                        original_weights[(id(actor_weights), pname)] = actor_weights[pname]
                        actor_weights[pname] = pval

            try:
                result = solver.solve(
                    focal_actor=focal_actor,
                    checkpoint_id=checkpoint_id,
                    mode=mode,
                    n_draws=n_draws,
                )
                row = {
                    "checkpoint": checkpoint_id,
                    "focal": focal_actor,
                    "optimal_action": result.optimal_action,
                    "optimal_EU": result.optimal_EU,
                }
                row.update(overrides)

                # Add all action EUs
                for action, eu in result.EU_per_action.items():
                    row[f"EU_{action}"] = eu

                results.append(row)

            finally:
                # Restore original weights
                for (actor_id, pname), orig_val in original_weights.items():
                    for actor_weights in solver.utility_weights.values():
                        if id(actor_weights) == actor_id and pname in actor_weights:
                            actor_weights[pname] = orig_val

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Sensitivity analysis")
    parser.add_argument("--focal", type=str, default="Board",
                        choices=["Board", "ASA"])
    parser.add_argument("--checkpoint", type=str, default="C0")
    parser.add_argument("--n_draws", type=int, default=20)
    parser.add_argument("--K", type=int, default=50,
                        help="Opponent parameter samples (reduced for speed)")
    parser.add_argument("--R_rollouts", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_workers", type=int, default=None,
                        help="Number of parallel worker processes (default: cpu_count - 1)")
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    data_dir = PROJECT_ROOT / "data"
    solver = Solver(
        governance_spec_path=data_dir / "governance_spec.xlsx",
        opponent_priors_path=data_dir / "opponent_priors.xlsx",
        checkpoint_dir=data_dir / "checkpoints",
        K=args.K,
        R_rollouts=args.R_rollouts,
        seed=args.seed,
        n_workers=args.n_workers,
    )

    df = run_sensitivity(
        solver=solver,
        focal_actor=args.focal,
        checkpoint_id=args.checkpoint,
        param_grid=DEFAULT_GRID,
        n_draws=args.n_draws,
        n_workers=solver.n_workers,
    )

    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS RESULTS")
    print("=" * 70)
    print(df.to_string(index=False))
    print()

    # Policy shift analysis
    print("\nOPTIMAL POLICY DISTRIBUTION:")
    print(df["optimal_action"].value_counts().to_string())
    print()

    output_path = args.output or str(PROJECT_ROOT / "outputs" / "sensitivity_results.csv")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
