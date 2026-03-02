"""
Run the ARA engine in ASA-focal mode.

Usage:
    python -m run.run_asa_mode --checkpoint C0 --n_draws 100
    python -m run.run_asa_mode --all_checkpoints --n_draws 50
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine.solver import Solver
from engine.modes import MODE_ASA, MODE_ASA_L2, MODE_ASA_POLICY_BOARD
from run.visualise_tree import build_viz_tree, render_tree

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run ARA in ASA-focal mode")
    parser.add_argument("--checkpoint", type=str, default="C0")
    parser.add_argument("--all_checkpoints", action="store_true")
    parser.add_argument("--n_draws", type=int, default=100)
    parser.add_argument("--K", type=int, default=200)
    parser.add_argument("--R_rollouts", type=int, default=20)
    parser.add_argument("--level", type=int, default=1, choices=[1, 2])
    parser.add_argument("--board_policy", action="store_true",
                        help="Use fixed policy for Board instead of ARA")
    parser.add_argument("--scenario", type=str, default="both",
                        choices=["ceo_stayed", "ceo_resigned", "both"],
                        help="Pre-game CEO resignation scenario (default: both)")
    parser.add_argument("--seed", type=int, default=42)
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
    )

    if args.board_policy:
        mode = MODE_ASA_POLICY_BOARD
    elif args.level == 2:
        mode = MODE_ASA_L2
    else:
        mode = MODE_ASA

    # Resolve scenarios to run
    scenarios = (["ceo_stayed", "ceo_resigned"] if args.scenario == "both"
                 else [args.scenario])

    if args.all_checkpoints:
        dfs = []
        for scenario in scenarios:
            df_s = solver.solve_all_checkpoints(
                focal_actor="ASA",
                mode=mode,
                n_draws=args.n_draws,
                scenario=scenario,
            )
            dfs.append(df_s)
        df = __import__("pandas").concat(dfs, ignore_index=True)
        print("\n" + "=" * 70)
        print("ASA-FOCAL ARA RESULTS")
        print("=" * 70)
        print(df.to_string(index=False))
        print()
    else:
        results = {}
        dfs = []

        if args.scenario == "both":
            # Use solve_scenarios() which computes D0_ceo predictive
            results = solver.solve_scenarios(
                focal_actor="ASA",
                checkpoint_id=args.checkpoint,
                mode=mode,
                n_draws=args.n_draws,
            )
            # Print D0_ceo predictive distribution
            first_result = next(iter(results.values()))
            print(f"\n{'=' * 70}")
            print("D0_ceo PREDICTED DISTRIBUTION (ASA's model of CEO)")
            print("=" * 70)
            for action, prob in first_result.d0_ceo_predictive.items():
                print(f"  {action}: {prob:.1%}")

            for scenario, result in results.items():
                dfs.append(result.summary_df())
                print(f"\n{'=' * 70}")
                print(f"ASA-FOCAL ARA RESULTS — scenario: {scenario} "
                      f"(Pr = {result.scenario_prob:.1%})")
                print("=" * 70)
                print(result.summary_df().to_string(index=False))
                result.print_diagnostics()
        else:
            for scenario in scenarios:
                result = solver.solve(
                    focal_actor="ASA",
                    checkpoint_id=args.checkpoint,
                    mode=mode,
                    n_draws=args.n_draws,
                    scenario=scenario,
                )
                results[scenario] = result
                dfs.append(result.summary_df())

                print(f"\n{'=' * 70}")
                print(f"ASA-FOCAL ARA RESULTS — scenario: {scenario}")
                print("=" * 70)
                print(result.summary_df().to_string(index=False))
                result.print_diagnostics()

        df = __import__("pandas").concat(dfs, ignore_index=True)

    output_path = args.output or str(PROJECT_ROOT / "outputs" / "asa_mode_results.csv")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")

    # Generate tree diagrams (single-checkpoint, per-scenario)
    if not args.all_checkpoints:
        logger.info("Generating tree diagrams…")
        out_dir = PROJECT_ROOT / "outputs"
        for scenario, result in results.items():
            root = build_viz_tree(solver, result, "ASA", args.checkpoint,
                                  mode, bias=None, n_mc=200, scenario=scenario)
            tag = f"ASA_{args.checkpoint}_{scenario}"
            render_tree(root,
                        title=f"Game Tree — Probabilities  (ASA, {args.checkpoint}, {scenario})",
                        diagram_mode="prob",
                        output_path=str(out_dir / f"tree_prob_{tag}"))
            render_tree(root,
                        title=f"Game Tree — Expected Utility  (ASA, {args.checkpoint}, {scenario})",
                        diagram_mode="eu",
                        output_path=str(out_dir / f"tree_eu_{tag}"))


if __name__ == "__main__":
    main()
