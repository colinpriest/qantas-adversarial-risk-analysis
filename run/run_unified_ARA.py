"""
Unified ARA game tree — builds all four strategic modes in a single run.

Produces one interactive HTML dashboard with a mode selector to switch
between Stochastic (default), Board Strategic, ASA Strategic, and CEO
Strategic views.  All four trees share the same utility parameters;
only the decision probabilities change.

Usage:
    python -m run.run_unified_ARA --n_draws 500
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

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

# Mode keys in display order (first is the default / commentary source)
MODE_KEYS = ["stochastic", "board", "asa", "ceo"]
MODE_LABELS = {
    "stochastic": "All Stochastic",
    "board": "Board Strategic",
    "asa": "ASA Strategic",
    "ceo": "CEO Strategic",
}
MODE_STRATEGIC = {
    "stochastic": None,
    "board": "Board",
    "asa": "ASA",
    "ceo": "CEO",
}


def main():
    parser = argparse.ArgumentParser(
        description="Unified ARA game tree — all strategic modes")
    parser.add_argument("--n_draws", type=int, default=500,
                        help="Number of posterior draws to use (default: 500)")
    parser.add_argument("--posterior-draws", type=str, default=None,
                        help="Path to stan_posterior_draws.npz")
    parser.add_argument("--param-estimates", type=str, default=None,
                        help="Path to parameter_estimates.csv (for vote weights)")
    parser.add_argument("--no-laplacian", action="store_true",
                        help="Disable Laplacian smoothing on stochastic decision probs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path")
    parser.add_argument("--no-commentary", action="store_true",
                        help="Skip LLM commentary generation (faster for diagnostics)")

    args = parser.parse_args()

    # ── Load data ──
    draws_path = args.posterior_draws or str(
        PROJECT_ROOT / "outputs" / "stan_posterior_draws.npz")
    w_draws = load_posterior_draws(draws_path)

    if w_draws.shape[0] > args.n_draws:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(w_draws.shape[0], size=args.n_draws, replace=False)
        w_draws = w_draws[idx]
        logger.info(f"Subsampled to {args.n_draws} posterior draws")

    est_path = args.param_estimates or str(
        PROJECT_ROOT / "outputs" / "parameter_estimates.csv")
    vote_weights = load_vote_weights(est_path)
    logger.info(f"Vote weights: w_strike={vote_weights['w_strike']:.4f}, "
                f"w_overwhelming={vote_weights['w_overwhelming']:.4f}")

    data_dir = PROJECT_ROOT / "data"
    ceo_params, asa_params = load_actor_params_from_spec(
        data_dir / "governance_spec.xlsx")
    logger.info(f"Loaded {len(ceo_params)} CEO params, {len(asa_params)} ASA params")

    # ── Build all 4 trees ──
    trees = {}   # mode_key -> VizNode
    for mode_key in MODE_KEYS:
        strategic = MODE_STRATEGIC[mode_key]
        label = MODE_LABELS[mode_key]
        logger.info(f"Building tree: {label}...")

        cfg = TreeConfig(
            w_draws=w_draws,
            vote_weights=vote_weights,
            ceo_params=ceo_params,
            asa_params=asa_params,
            probs=dict(TREE_DEFAULT_PROBS),
            strategic_actor=strategic,
            param_names=list(ESTIMABLE_PARAM_NAMES),
            laplacian=not args.no_laplacian,
        )
        initial_state = make_initial_state()
        root, eu_dict = build_game_tree("D0_ceo", initial_state, cfg)
        trees[mode_key] = root

        board_eu = float(np.mean(eu_dict["Board"]))
        ceo_eu = float(np.mean(eu_dict["CEO"]))
        asa_eu = float(np.mean(eu_dict["ASA"]))
        logger.info(f"  {label}: Board={board_eu:+.4f}, CEO={ceo_eu:+.4f}, ASA={asa_eu:+.4f}")

    # ── Print summary for default (stochastic) mode ──
    print(f"\n{'=' * 70}")
    print("UNIFIED ARA GAME TREE RESULTS")
    print("=" * 70)
    _print_tree_summary(trees["stochastic"], depth=0)

    # ── Generate PNG tree diagrams for default mode ──
    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Rendering PNG tree diagrams (stochastic mode)...")
    render_tree(trees["stochastic"],
                title="Game Tree -- Probabilities  (All Stochastic)",
                diagram_mode="prob",
                output_path=str(out_dir / "tree_prob_unified"))
    render_tree(trees["stochastic"],
                title="Game Tree -- Expected Utility  (All Stochastic)",
                diagram_mode="eu",
                output_path=str(out_dir / "tree_eu_unified"))

    # ── Generate interactive HTML with all 4 modes ──
    actual_path = str(PROJECT_ROOT / "data" / "actual_outcomes.json")
    html_path = render_interactive_tree(
        root=trees,         # dict of mode_key -> VizNode
        results={},
        focal="Board",
        checkpoint_id="unified",
        actual_outcomes_path=actual_path,
        output_dir=out_dir,
        skip_commentary=getattr(args, 'no_commentary', False),
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
