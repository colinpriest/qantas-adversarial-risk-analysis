"""
Run the ARA engine in Board-focal mode.

Usage:
    python -m run.run_board_mode --checkpoint C0 --n_draws 100
    python -m run.run_board_mode --all_checkpoints --n_draws 50
    python -m run.run_board_mode --checkpoint C0 --no-bias-board --n_draws 100
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine.solver import Solver
from engine.modes import MODE_BOARD, MODE_BOARD_L2, MODE_BOARD_POLICY_ASA
from run.visualise_tree import build_unified_tree, render_tree
from run.interactive_tree import render_interactive_tree

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run ARA in Board-focal mode")
    parser.add_argument("--checkpoint", type=str, default="C0",
                        help="Checkpoint ID (e.g. C0, C1, C2, C3)")
    parser.add_argument("--all_checkpoints", action="store_true",
                        help="Run all checkpoints")
    parser.add_argument("--n_draws", type=int, default=100,
                        help="Number of belief draws to use")
    parser.add_argument("--K", type=int, default=200,
                        help="Number of opponent parameter samples")
    parser.add_argument("--R_rollouts", type=int, default=50,
                        help="Number of rollouts per Psi evaluation")
    parser.add_argument("--n_vote_samples", type=int, default=50,
                        help="Monte Carlo samples for vote integration (lower for speed)")
    parser.add_argument("--n_review_samples", type=int, default=20,
                        help="Monte Carlo samples for review integration (lower for speed)")
    parser.add_argument("--level", type=int, default=2, choices=[1, 2],
                        help="Opponent modelling level")
    parser.add_argument("--asa_policy", action="store_true",
                        help="Use empirical fixed policy for ASA instead of ARA")
    parser.add_argument("--no-bias-board", action="store_true",
                        help="Disable Board overconfidence bias (counterfactual: accurate self-assessment)")
    parser.add_argument("--scenario", type=str, default="both",
                        choices=["ceo_stayed", "ceo_resigned", "both"],
                        help="Pre-game CEO resignation scenario (default: both)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_workers", type=int, default=None,
                        help="Number of parallel worker processes (default: cpu_count - 1)")
    parser.add_argument("--K_d0", type=int, default=50,
                        help="Opponent samples for D0_ceo Level-2 prediction (default: 50; "
                             "reduce to 10–20 for faster runs)")
    parser.add_argument("--R_d0", type=int, default=10,
                        help="Rollouts per action for D0_ceo prediction (default: 10; "
                             "reduce to 3–5 for faster runs)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path")

    args = parser.parse_args()

    data_dir = PROJECT_ROOT / "data"
    solver = Solver(
        governance_spec_path=data_dir / "governance_spec.xlsx",
        opponent_priors_path=data_dir / "opponent_priors.xlsx",
        checkpoint_dir=data_dir / "checkpoints",
        K=args.K,
        R_rollouts=args.R_rollouts,
        n_vote_samples=args.n_vote_samples,
        n_review_samples=args.n_review_samples,
        seed=args.seed,
        n_workers=args.n_workers,
        K_d0_ceo=args.K_d0,
        R_d0_ceo=args.R_d0,
    )

    if args.asa_policy:
        mode = MODE_BOARD_POLICY_ASA
    elif args.level == 2:
        mode = MODE_BOARD_L2
    else:
        mode = MODE_BOARD

    # Overconfidence bias: Board is biased by default (from governance_spec).
    # --no-bias-board disables this for counterfactual analysis.
    bias_kwarg = {}
    if args.no_bias_board:
        logger.info("Overconfidence bias: DISABLED (counterfactual unbiased analysis)")
        bias_kwarg = {"overconfidence_bias": None}
    else:
        b = solver.overconfidence_bias
        logger.info(
            f"Overconfidence bias: from governance_spec "
            f"(D1=[{b.d1_floor:.1f},{b.d1_ceiling:.1f}], "
            f"D3=[{b.d3_floor:.1f},{b.d3_ceiling:.1f}], "
            f"sigma_scale={b.sigma_scale:.2f}, "
            f"review_car_bias={b.review_car_bias:.4f})"
        )

    # Resolve scenarios to run
    scenarios = (["ceo_stayed", "ceo_resigned"] if args.scenario == "both"
                 else [args.scenario])

    if args.all_checkpoints:
        dfs = []
        for scenario in scenarios:
            df_s = solver.solve_all_checkpoints(
                focal_actor="Board",
                mode=mode,
                n_draws=args.n_draws,
                scenario=scenario,
                **bias_kwarg,
            )
            dfs.append(df_s)
        df = __import__("pandas").concat(dfs, ignore_index=True)
        print("\n" + "=" * 70)
        print("BOARD-FOCAL ARA RESULTS")
        print("=" * 70)
        print(df.to_string(index=False, formatters={
            c: "{:.2f}".format for c in df.columns if c.startswith("Pr_")
        }))
        print()
    else:
        results = {}
        dfs = []

        if args.scenario == "both":
            # Use solve_scenarios() which computes D0_ceo predictive
            results = solver.solve_scenarios(
                focal_actor="Board",
                checkpoint_id=args.checkpoint,
                mode=mode,
                n_draws=args.n_draws,
                **bias_kwarg,
            )
            # Print D0_ceo predictive distribution
            first_result = next(iter(results.values()))
            print(f"\n{'=' * 70}")
            print("D0_ceo PREDICTED DISTRIBUTION (Board's model of CEO)")
            print("=" * 70)
            for action, prob in first_result.d0_ceo_predictive.items():
                print(f"  {action}: {prob:.1%}")

            for scenario, result in results.items():
                dfs.append(result.summary_df())
                print(f"\n{'=' * 70}")
                print(f"BOARD-FOCAL ARA RESULTS — scenario: {scenario} "
                      f"(Pr = {result.scenario_prob:.1%})")
                print("=" * 70)
                print(result.display_summary())
                result.print_diagnostics()
        else:
            for scenario in scenarios:
                result = solver.solve(
                    focal_actor="Board",
                    checkpoint_id=args.checkpoint,
                    mode=mode,
                    n_draws=args.n_draws,
                    scenario=scenario,
                    **bias_kwarg,
                )
                results[scenario] = result
                dfs.append(result.summary_df())

                print(f"\n{'=' * 70}")
                print(f"BOARD-FOCAL ARA RESULTS — scenario: {scenario}")
                print("=" * 70)
                print(result.display_summary())
                result.print_diagnostics()

        df = __import__("pandas").concat(dfs, ignore_index=True)

    # Save
    output_path = args.output or str(PROJECT_ROOT / "outputs" / "board_mode_results.csv")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")

    # Generate unified tree diagram (single tree from D0_ceo)
    if not args.all_checkpoints:
        bias = None if args.no_bias_board else solver.overconfidence_bias
        logger.info("Generating tree diagrams…")
        out_dir = PROJECT_ROOT / "outputs"

        # D0_ceo probabilities from Bayesian prior
        alpha = solver.ceo_departure_prior_alpha
        beta_ = solver.ceo_departure_prior_beta
        d0_probs = {
            "CEO_resign": alpha / (alpha + beta_),
            "CEO_stay": beta_ / (alpha + beta_),
        }

        root = build_unified_tree(
            solver, results, "Board", args.checkpoint,
            mode, bias, n_mc=200, d0_probs=d0_probs,
        )
        tag = f"Board_{args.checkpoint}"
        render_tree(root,
                    title=f"Game Tree — Probabilities  (Board, {args.checkpoint})",
                    diagram_mode="prob",
                    output_path=str(out_dir / f"tree_prob_{tag}"))
        render_tree(root,
                    title=f"Game Tree — Expected Utility  (Board, {args.checkpoint})",
                    diagram_mode="eu",
                    output_path=str(out_dir / f"tree_eu_{tag}"))

        actual_path = str(PROJECT_ROOT / "data" / "actual_outcomes.json")
        html_path = render_interactive_tree(
            root=root,
            results=results,
            focal="Board",
            checkpoint_id=args.checkpoint,
            actual_outcomes_path=actual_path,
            output_dir=out_dir,
            d0_probs=d0_probs,
        )
        logger.info(f"Interactive HTML saved to {html_path}")


if __name__ == "__main__":
    main()
