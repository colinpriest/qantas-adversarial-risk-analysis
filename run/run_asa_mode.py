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
from run.visualise_tree import build_unified_tree, render_tree

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run ARA in ASA-focal mode")
    parser.add_argument("--checkpoint", type=str, default="C0")
    parser.add_argument("--all_checkpoints", action="store_true")
    parser.add_argument("--n_draws", type=int, default=100)
    parser.add_argument("--K", type=int, default=200)
    parser.add_argument("--R_rollouts", type=int, default=50)
    parser.add_argument("--level", type=int, default=1, choices=[1, 2])
    parser.add_argument("--board_policy", action="store_true",
                        help="Use fixed policy for Board instead of ARA")
    parser.add_argument("--scenario", type=str, default="both",
                        choices=["ceo_stayed", "ceo_resigned", "both"],
                        help="Pre-game CEO resignation scenario (default: both)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_workers", type=int, default=None,
                        help="Number of parallel worker processes (default: cpu_count - 1)")
    parser.add_argument("--K_d0", type=int, default=50,
                        help="Opponent samples for D0_ceo Level-2 prediction (default: 50)")
    parser.add_argument("--R_d0", type=int, default=10,
                        help="Rollouts per action for D0_ceo prediction (default: 10)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no-board-prior", action="store_true",
                        help="Disable Laplace smoothing on Board's predictive distribution")
    parser.add_argument("--no-ceo-prior", action="store_true",
                        help="Disable Laplace smoothing on CEO's predictive distribution")
    parser.add_argument("--no-asa-prior", action="store_true",
                        help="Disable Laplace smoothing on ASA's predictive distribution")

    args = parser.parse_args()

    no_prior_actors = set()
    if args.no_board_prior:
        no_prior_actors.add("Board")
    if args.no_ceo_prior:
        no_prior_actors.add("CEO")
    if args.no_asa_prior:
        no_prior_actors.add("ASA")
    if no_prior_actors:
        logger.info(f"Laplace smoothing DISABLED for: {', '.join(sorted(no_prior_actors))}")

    data_dir = PROJECT_ROOT / "data"
    solver = Solver(
        governance_spec_path=data_dir / "governance_spec.xlsx",
        opponent_priors_path=data_dir / "opponent_priors.xlsx",
        checkpoint_dir=data_dir / "checkpoints",
        K=args.K,
        R_rollouts=args.R_rollouts,
        seed=args.seed,
        n_workers=args.n_workers,
        K_d0_ceo=args.K_d0,
        R_d0_ceo=args.R_d0,
        no_prior_actors=no_prior_actors,
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
                print(result.display_summary())
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
                print(result.display_summary())
                result.print_diagnostics()

        df = __import__("pandas").concat(dfs, ignore_index=True)

    output_path = args.output or str(PROJECT_ROOT / "outputs" / "asa_mode_results.csv")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")

    # Generate unified tree diagram (single tree from D0_ceo)
    if not args.all_checkpoints:
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
            solver, results, "ASA", args.checkpoint,
            mode, bias=None, n_mc=200, d0_probs=d0_probs,
        )
        tag = f"ASA_{args.checkpoint}"
        render_tree(root,
                    title=f"Game Tree — Probabilities  (ASA, {args.checkpoint})",
                    diagram_mode="prob",
                    output_path=str(out_dir / f"tree_prob_{tag}"))
        render_tree(root,
                    title=f"Game Tree — Expected Utility  (ASA, {args.checkpoint})",
                    diagram_mode="eu",
                    output_path=str(out_dir / f"tree_eu_{tag}"))


if __name__ == "__main__":
    main()
