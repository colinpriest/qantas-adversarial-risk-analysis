"""
Run the game tree in Board-focal mode.

Board decisions are strategic (argmax over posterior weight draws).
All other actors use fixed probabilities from board_utility_quantification.py.

Usage:
    python -m run.run_board_mode --n_draws 500
    python -m run.run_board_mode --n_draws 500 --no-laplacian
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from run.game_tree import (
    TreeConfig, TREE_DEFAULT_PROBS, ESTIMABLE_PARAM_NAMES,
    build_game_tree, make_initial_state,
    load_posterior_draws, load_vote_weights, load_actor_params_from_spec,
)
from run.visualise_tree import render_tree
from run.interactive_tree import render_interactive_tree

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run game tree in Board-focal mode")
    parser.add_argument("--n_draws", type=int, default=500,
                        help="Number of posterior draws to use (default: 500)")
    parser.add_argument("--posterior-draws", type=str, default=None,
                        help="Path to stan_posterior_draws.npz (default: outputs/)")
    parser.add_argument("--param-estimates", type=str, default=None,
                        help="Path to parameter_estimates.csv (for vote weights)")
    parser.add_argument("--no-laplacian", action="store_true",
                        help="Disable Laplacian smoothing on Board decision probs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path")

    args = parser.parse_args()

    # ── Load posterior draws ──
    draws_path = args.posterior_draws or str(PROJECT_ROOT / "outputs" / "stan_posterior_draws.npz")
    w_draws = load_posterior_draws(draws_path)

    # Subsample if needed
    if w_draws.shape[0] > args.n_draws:
        rng = __import__("numpy").random.default_rng(args.seed)
        idx = rng.choice(w_draws.shape[0], size=args.n_draws, replace=False)
        w_draws = w_draws[idx]
        logger.info(f"Subsampled to {args.n_draws} posterior draws")

    # ── Load vote penalty weights ──
    est_path = args.param_estimates or str(PROJECT_ROOT / "outputs" / "parameter_estimates.csv")
    vote_weights = load_vote_weights(est_path)
    logger.info(f"Vote weights: w_strike={vote_weights['w_strike']:.4f}, "
                f"w_overwhelming={vote_weights['w_overwhelming']:.4f}")

    # ── Load CEO and ASA parameters ──
    data_dir = PROJECT_ROOT / "data"
    ceo_params, asa_params = load_actor_params_from_spec(data_dir / "governance_spec.xlsx")
    logger.info(f"Loaded {len(ceo_params)} CEO params, {len(asa_params)} ASA params")

    # ── Build tree config ──
    cfg = TreeConfig(
        w_draws=w_draws,
        vote_weights=vote_weights,
        ceo_params=ceo_params,
        asa_params=asa_params,
        probs=dict(TREE_DEFAULT_PROBS),
        focal_actor="Board",
        param_names=list(ESTIMABLE_PARAM_NAMES),
        laplacian=not args.no_laplacian,
    )

    # ── Build the full game tree ──
    logger.info("Building Board-focal game tree...")
    initial_state = make_initial_state()
    root, eu_draws = build_game_tree("D0_ceo", initial_state, cfg)

    logger.info(f"Root EU = {root.eu:+.4f} (mean over {w_draws.shape[0]} draws)")

    # ── Print summary ──
    print(f"\n{'=' * 70}")
    print("BOARD-FOCAL GAME TREE RESULTS")
    print("=" * 70)
    _print_tree_summary(root, depth=0)

    # ── Generate tree diagrams ──
    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = "Board_posterior"
    logger.info("Rendering tree diagrams...")
    render_tree(root,
                title="Game Tree — Probabilities  (Board-focal, posterior)",
                diagram_mode="prob",
                output_path=str(out_dir / f"tree_prob_{tag}"))
    render_tree(root,
                title="Game Tree — Expected Utility  (Board-focal, posterior)",
                diagram_mode="eu",
                output_path=str(out_dir / f"tree_eu_{tag}"))

    # ── Generate interactive HTML ──
    actual_path = str(PROJECT_ROOT / "data" / "actual_outcomes.json")
    # render_interactive_tree expects a results dict; we pass empty since
    # we embed all data directly in VizNode.terminal_decomposition
    html_path = render_interactive_tree(
        root=root,
        results={},
        focal="Board",
        checkpoint_id="posterior",
        actual_outcomes_path=actual_path,
        output_dir=out_dir,
    )
    logger.info(f"Interactive HTML saved to {html_path}")


def _print_tree_summary(node, depth=0, max_depth=3):
    """Print a compact tree summary."""
    indent = "  " * depth
    if node.node_type == "terminal":
        print(f"{indent}Terminal EU={node.eu:+.4f}")
        return
    print(f"{indent}{node.node_name} [{node.owner}] EU={node.eu:+.4f}")
    if depth < max_depth:
        for label, prob, child in node.children:
            if prob > 0.001:
                print(f"{indent}  -> {label} (p={prob:.3f})")
                _print_tree_summary(child, depth + 1, max_depth)


if __name__ == "__main__":
    main()
