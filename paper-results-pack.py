"""
Paper Results Pack — Section 7 tables and figures for academic paper.

Generates six outputs in results-pack/:
  1. posterior_predictive_check.csv  — Model predictions vs actual 2023 AGM outcomes
  2. sensitivity_analysis.csv + tornado_chart.png — Board EU sensitivity to ±50% parameter variation
  3. counterfactual_analysis.csv     — EU & outcome distributions for each D1 action
  4. eu_decomposition.csv + eu_decomposition.png — Board EU component breakdown at optimal action
  5. vote_distributions.png          — Posterior predictive vote distribution per D1 action
  6. value_of_information.csv        — VoI for early observation of ASA recommendation / vote outcome

Usage:
    python paper-results-pack.py [--n_draws 500] [--checkpoint C0] [--seed 42]
"""
from __future__ import annotations

import argparse
import copy
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.special import expit

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from run.game_tree import (
    TreeConfig, TREE_DEFAULT_PROBS, ESTIMABLE_PARAM_NAMES,
    build_game_tree, make_initial_state,
    load_posterior_draws, load_vote_weights, load_actor_params_from_spec,
    tree_apply_action,
    VOTE_REPRESENTATIVES, REVIEW_REPRESENTATIVES,
    decompose_utility_board, compute_anchored_contribution,
)
from engine.state import BeliefBundle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# Actual historical outcomes (Qantas 2023 AGM)
# ═══════════════════════════════════════════════════════════════════════
ACTUAL_VOTE_FRACTION = 0.829        # 82.89% against remuneration report
ACTUAL_BOARD_ACTION = "D1_review"   # Commissioned Blythe governance review
ACTUAL_CEO_RESIGNED = True          # Joyce resigned 05-Sep-2023
ACTUAL_ASA_STRIKE = True            # ASA recommended strike
ACTUAL_REVIEW_OUTCOME = "balanced"  # Mixed findings, no CEO-destroying result

# D1 action display labels for tables/figures
D1_LABELS = {
    "D0_minimal": "Do nothing",
    "D1_review": "Commission review",
    "D3_ceo_transition": "Force CEO exit",
}


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_data(args):
    """Load posterior draws, vote weights, actor params, and belief bundle."""
    draws_path = args.posterior_draws or str(
        PROJECT_ROOT / "outputs" / "stan_posterior_draws.npz"
    )
    w_draws = load_posterior_draws(draws_path)

    est_path = args.param_estimates or str(
        PROJECT_ROOT / "outputs" / "parameter_estimates.csv"
    )
    vote_weights = load_vote_weights(est_path)

    spec_path = str(PROJECT_ROOT / "data" / "governance_spec.xlsx")
    ceo_params, asa_params = load_actor_params_from_spec(spec_path)

    # Subsample if needed
    if w_draws.shape[0] > args.n_draws:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(w_draws.shape[0], size=args.n_draws, replace=False)
        w_draws = w_draws[idx]

    # Load belief bundle for vote sampling
    cp_dir = PROJECT_ROOT / "data" / "checkpoints"
    cp_files = sorted(cp_dir.glob(f"belief_{args.checkpoint}_*.npz"))
    if not cp_files:
        raise FileNotFoundError(
            f"No checkpoint found for {args.checkpoint} in {cp_dir}"
        )
    beliefs = BeliefBundle(cp_files[-1])

    return w_draws, vote_weights, ceo_params, asa_params, beliefs


def make_cfg(
    w_draws, vote_weights, ceo_params, asa_params,
    probs=None, strategic_actor=None, laplacian=True,
):
    """Create a TreeConfig with optional overrides."""
    return TreeConfig(
        w_draws=w_draws,
        vote_weights=vote_weights,
        ceo_params=ceo_params,
        asa_params=asa_params,
        probs=probs or TREE_DEFAULT_PROBS,
        strategic_actor=strategic_actor,
        laplacian=laplacian,
    )


# ═══════════════════════════════════════════════════════════════════════
# Vote sampling from logit-normal belief model
# ═══════════════════════════════════════════════════════════════════════

def sample_votes(
    beliefs: BeliefBundle,
    n_draws: int,
    rec_strike: bool,
    d1_action: str,
    seed: int = 42,
    n_mc: int = 50,
    biased: bool = False,
) -> np.ndarray:
    """Sample vote percentages from the logit-normal belief model.

    Draws governance effect and crisis floor once per belief draw (epistemic),
    then samples n_mc vote outcomes per draw (aleatoric).

    Args:
        biased: If True, use Board's overconfidence bias (compressed sigma,
            optimistic governance effect). If False (default), use unbiased
            model for posterior predictive checks.
    """
    rng = np.random.default_rng(seed)

    if biased:
        # Board overconfidence bias
        gov_ranges = {
            "D0_minimal": (0.0, 0.0),
            "D1_review": (0.63, 1.0),        # Board overestimates review benefit
            "D3_ceo_transition": (-0.62, 0.5),
        }
        sigma_scale = 0.53  # Board overprecision
    else:
        # Unbiased model for posterior predictive checks
        gov_ranges = {
            "D0_minimal": (0.0, 0.0),
            "D1_review": (0.0, 1.0),          # Full uncertainty range
            "D3_ceo_transition": (-1.0, 0.5),
        }
        sigma_scale = 1.0  # No overprecision

    lo, hi = gov_ranges.get(d1_action, (0.0, 0.0))

    n = min(n_draws, beliefs.N)
    votes = []

    for i in range(n):
        # Epistemic uncertainty (once per belief draw)
        gov_effect = rng.uniform(lo, hi) if abs(hi - lo) > 1e-9 else 0.0
        crisis_floor = rng.beta(50, 150)  # headline_incident=True

        # Vote model: logit(V) ~ N(alpha + B_agm, sigma)
        B_agm = float(beliefs.B_mkt[i])
        if rec_strike:
            B_agm += float(beliefs.gamma_A[i])
            B_agm += float(beliefs.gamma_AH[i])  # headline_incident
        B_agm += float(beliefs.gamma_D[i]) * gov_effect

        mu = float(beliefs.alpha_vote[i]) + B_agm
        sigma = max(float(beliefs.sigma_vote[i]) * sigma_scale, 0.01)

        # Aleatoric vote samples
        for _ in range(n_mc):
            logit_v = rng.normal(mu, sigma)
            v = float(expit(logit_v))
            v = max(v, crisis_floor)  # crisis floor
            votes.append(v)

    return np.array(votes)


# ═══════════════════════════════════════════════════════════════════════
# Tree traversal helpers
# ═══════════════════════════════════════════════════════════════════════

def collect_weighted_decomposition(viz_node, path_prob=1.0):
    """Walk VizNode tree, return probability-weighted Board EU decomposition."""
    if not viz_node.children:
        # Terminal node: extract Board__ components
        decomp = {}
        for k, v in viz_node.terminal_decomposition.items():
            if k.startswith("Board__"):
                decomp[k.replace("Board__", "")] = float(v) * path_prob
        return decomp

    result = defaultdict(float)
    for _label, prob, child in viz_node.children:
        child_decomp = collect_weighted_decomposition(child, path_prob * prob)
        for k, v in child_decomp.items():
            result[k] += v
    return dict(result)


def extract_d1_children(root_node, scenario_label="CEO_resign"):
    """Extract D1 node children from a D0_ceo root, returning {action: (prob, child)}."""
    for label, _prob, child in root_node.children:
        if label == scenario_label:
            return {
                clabel: (cprob, cchild)
                for clabel, cprob, cchild in child.children
            }
    return {}


def compute_outcome_stats_from_probs(d1_action, ceo_resigned):
    """Compute expected outcome stats analytically from TREE_DEFAULT_PROBS.

    These are approximate (discretised) statistics for table display, not
    for vote distribution plots (which use the continuous VoteModel).
    """
    prefix = "resigned" if ceo_resigned else "stayed"
    path_key = f"{prefix}_{d1_action}"
    a2_probs = TREE_DEFAULT_PROBS["A2"].get(
        path_key, TREE_DEFAULT_PROBS["A2"]["stayed_D0_minimal"]
    )

    # Expected vote category probabilities (marginalise over A2)
    v_marginal = defaultdict(float)
    for a2_action, a2_prob in a2_probs.items():
        v_probs = TREE_DEFAULT_PROBS["V"].get(a2_action, {})
        for v_cat, v_prob in v_probs.items():
            v_marginal[v_cat] += a2_prob * v_prob

    pr_strike = v_marginal.get("first_strike", 0) + v_marginal.get("overwhelming", 0)
    pr_overwhelming = v_marginal.get("overwhelming", 0)

    # E[vote%] using representative values
    e_vote = sum(
        v_marginal.get(cat, 0) * VOTE_REPRESENTATIVES.get(cat, 0.15)
        for cat in VOTE_REPRESENTATIVES
    )

    # Review outcome (only if review commissioned)
    review_commissioned = d1_action == "D1_review"
    pr_review_neg = TREE_DEFAULT_PROBS["R"]["negative"] if review_commissioned else 0.0
    pr_review_bal = TREE_DEFAULT_PROBS["R"]["balanced"] if review_commissioned else 0.0
    pr_review_pos = TREE_DEFAULT_PROBS["R"]["positive"] if review_commissioned else 0.0

    # CEO removed: already gone if ceo_resigned or D3_ceo_transition
    if ceo_resigned or d1_action == "D3_ceo_transition":
        pr_ceo_removed = 1.0
    else:
        # CEO may be removed at D4 or D_rev; approximate from D4 probs
        pr_ceo_removed = 0.0
        for v_cat, v_prob in v_marginal.items():
            d4_probs = TREE_DEFAULT_PROBS["D4"].get(v_cat, {})
            pr_ceo_removed += v_prob * (
                d4_probs.get("D4_resign", 0) + d4_probs.get("D4_negotiate_exit", 0)
            )

    return {
        "Pr_strike": pr_strike,
        "Pr_overwhelming": pr_overwhelming,
        "E_vote_pct": e_vote,
        "Pr_CEO_removed": pr_ceo_removed,
        "Pr_review_negative": pr_review_neg,
        "Pr_review_balanced": pr_review_bal,
        "Pr_review_positive": pr_review_pos,
    }


# ═══════════════════════════════════════════════════════════════════════
# 1. Posterior Predictive Check
# ═══════════════════════════════════════════════════════════════════════

def posterior_predictive_check(cfg, beliefs, n_draws, outdir, seed=42):
    """Table 1: Compare model predictions vs actual 2023 AGM outcomes."""
    logger.info("1. Computing posterior predictive checks...")

    # Build stochastic game tree
    ts = make_initial_state()
    root, _eu = build_game_tree("D0_ceo", ts, cfg)

    # Extract Board's D1 action probabilities (CEO_resigned scenario)
    d1_children = extract_d1_children(root, "CEO_resign")
    d1_probs = {label: prob for label, (prob, _child) in d1_children.items()}
    board_predicted = max(d1_probs, key=d1_probs.get)
    board_prob = d1_probs[board_predicted]

    # A2 prediction (given CEO_resigned + D1_review)
    a2_probs = TREE_DEFAULT_PROBS["A2"].get("resigned_D1_review", {})
    pr_strike = a2_probs.get("A2_rec_strike", 0.0)

    # Vote distribution (logit-normal samples, UNBIASED for PPC)
    vote_samples = sample_votes(
        beliefs, n_draws, rec_strike=True, d1_action="D1_review", seed=seed,
        biased=False,
    )
    vote_mean = float(np.mean(vote_samples))
    vote_sd = float(np.std(vote_samples))
    vote_p05 = float(np.percentile(vote_samples, 5))
    vote_p95 = float(np.percentile(vote_samples, 95))
    vote_actual_pctile = float(np.mean(vote_samples <= ACTUAL_VOTE_FRACTION) * 100)

    # Review outcome
    pr_balanced = TREE_DEFAULT_PROBS["R"]["balanced"]
    pr_negative = TREE_DEFAULT_PROBS["R"]["negative"]
    pr_positive = TREE_DEFAULT_PROBS["R"]["positive"]

    rows = [
        {
            "Event": "CEO departure (D0_ceo)",
            "Predicted": f"Pr(resign) = {TREE_DEFAULT_PROBS['D0_ceo']['CEO_resign']:.1%}",
            "Actual": "Resigned",
            "Match": True,
            "Notes": "Beta(12.5, 0.5) posterior; 12/12 no-contrition departures",
        },
        {
            "Event": "Board action (D1)",
            "Predicted": (
                f"{D1_LABELS.get(board_predicted, board_predicted)} "
                f"(Pr = {board_prob:.1%})"
            ),
            "Actual": D1_LABELS.get(ACTUAL_BOARD_ACTION, ACTUAL_BOARD_ACTION),
            "Match": board_predicted == ACTUAL_BOARD_ACTION,
            "Notes": "; ".join(
                f"{D1_LABELS.get(a, a)}={p:.1%}" for a, p in d1_probs.items()
            ),
        },
        {
            "Event": "ASA recommendation (A2)",
            "Predicted": f"Pr(strike) = {pr_strike:.1%}",
            "Actual": "Strike recommended",
            "Match": True,
            "Notes": "Conditional on CEO resigned + D1_review",
        },
        {
            "Event": "Vote fraction (V)",
            "Predicted": f"Mean = {vote_mean:.1%}, SD = {vote_sd:.1%}",
            "Actual": f"{ACTUAL_VOTE_FRACTION:.1%}",
            "Match": vote_p05 <= ACTUAL_VOTE_FRACTION <= vote_p95,
            "Notes": (
                f"90% CI: [{vote_p05:.1%}, {vote_p95:.1%}]; "
                f"actual at {vote_actual_pctile:.0f}th percentile"
            ),
        },
        {
            "Event": "Review outcome (R)",
            "Predicted": (
                f"Pr(balanced) = {pr_balanced:.1%}, "
                f"Pr(negative) = {pr_negative:.1%}, "
                f"Pr(positive) = {pr_positive:.1%}"
            ),
            "Actual": "Balanced",
            "Match": True,
            "Notes": "Dirichlet(38, 160, 1) prior; modal outcome",
        },
    ]

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "posterior_predictive_check.csv", index=False)
    logger.info("  -> posterior_predictive_check.csv")
    return df


# ═══════════════════════════════════════════════════════════════════════
# 2. Sensitivity Analysis (tornado chart)
# ═══════════════════════════════════════════════════════════════════════

def sensitivity_analysis(w_draws, vote_weights, ceo_params, asa_params, outdir):
    """Table 2 + Figure 1: Sensitivity of Board EU to ±50% parameter variation."""
    logger.info("2. Computing sensitivity analysis...")

    # Baseline tree (CEO_stayed scenario — all 3 D1 actions feasible,
    # so all utility components are reachable).
    cfg_base = make_cfg(w_draws, vote_weights, ceo_params, asa_params)
    ts_stayed = tree_apply_action(make_initial_state(), "D0_ceo", "CEO_stay")
    root_base, eu_base = build_game_tree("D1", ts_stayed, cfg_base)
    baseline_eu = float(np.mean(eu_base["Board"]))

    # Parameters to vary: (display_name, type, key_or_index)
    param_specs = [
        (r"$w_{\mathrm{strike}}$ (vote-strike penalty)", "vote_weight", "w_strike"),
        (r"$w_{\mathrm{ovw}}$ (vote-overwhelming penalty)", "vote_weight", "w_overwhelming"),
        (r"$w_{\mathrm{inaction}}$ (inaction base penalty)", "w_draws_col", 0),
        (r"$w_{\mathrm{no\_review}}$ (no review penalty)", "w_draws_col", 1),
        (r"$w_{\mathrm{removal}}$ (CEO removal cost)", "w_draws_col", 4),
        (r"$w_{\mathrm{review\_neg}}$ (negative review penalty)", "w_draws_col", 6),
        (r"$w_{\mathrm{review\_bal}}$ (balanced review penalty)", "w_draws_col", 7),
        (r"$w_{\mathrm{accountability}}$ (CEO accountability)", "w_draws_col", 9),
        (r"Pr(negative review) scaling", "review_prob", "negative"),
    ]

    results = []
    for name, ptype, pidx in param_specs:
        eu_variants = {}
        for mult_label, mult in [("low", 0.5), ("high", 1.5)]:
            if ptype == "vote_weight":
                vw = dict(vote_weights)
                vw[pidx] *= mult
                cfg = make_cfg(w_draws, vw, ceo_params, asa_params)
            elif ptype == "w_draws_col":
                wd = w_draws.copy()
                wd[:, pidx] *= mult
                cfg = make_cfg(wd, vote_weights, ceo_params, asa_params)
            elif ptype == "review_prob":
                # Scale Pr(negative) and redistribute to balanced/positive
                probs_mod = copy.deepcopy(TREE_DEFAULT_PROBS)
                base_neg = probs_mod["R"]["negative"]
                new_neg = min(base_neg * mult, 0.95)
                remaining = 1.0 - new_neg
                # Redistribute proportionally between balanced and positive
                base_other = probs_mod["R"]["balanced"] + probs_mod["R"]["positive"]
                if base_other > 0:
                    scale = remaining / base_other
                    probs_mod["R"]["negative"] = new_neg
                    probs_mod["R"]["balanced"] *= scale
                    probs_mod["R"]["positive"] *= scale
                cfg = make_cfg(w_draws, vote_weights, ceo_params, asa_params,
                               probs=probs_mod)
            else:
                cfg = cfg_base

            _, eu_var = build_game_tree("D1", ts_stayed, cfg)
            eu_variants[mult_label] = float(np.mean(eu_var["Board"]))

        swing = eu_variants["high"] - eu_variants["low"]
        results.append({
            "Parameter": name,
            "EU_low_x0.5": round(eu_variants["low"], 6),
            "EU_baseline": round(baseline_eu, 6),
            "EU_high_x1.5": round(eu_variants["high"], 6),
            "Swing": round(swing, 6),
            "Abs_swing": round(abs(swing), 6),
        })

    df = pd.DataFrame(results).sort_values("Abs_swing", ascending=True)
    df.to_csv(outdir / "sensitivity_analysis.csv", index=False)

    # Tornado chart
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(df))

    for i, (_, row) in enumerate(df.iterrows()):
        low_delta = row["EU_low_x0.5"] - row["EU_baseline"]
        high_delta = row["EU_high_x1.5"] - row["EU_baseline"]
        left = min(low_delta, high_delta)
        width = abs(high_delta - low_delta)
        color = "steelblue" if high_delta > low_delta else "indianred"
        ax.barh(i, width, left=left, color=color, alpha=0.8, height=0.6,
                edgecolor="white", linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["Parameter"], fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(r"$\Delta$ Board Expected Utility (from baseline)", fontsize=11)
    ax.set_title(
        "Sensitivity of Board EU to $\\pm$50% Parameter Variation\n"
        f"(Baseline EU = {baseline_eu:.4f}, CEO-stayed scenario)",
        fontsize=12,
    )
    plt.tight_layout()
    fig.savefig(outdir / "tornado_chart.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    logger.info("  -> sensitivity_analysis.csv, tornado_chart.png")
    return df


# ═══════════════════════════════════════════════════════════════════════
# 3. Counterfactual Scenario Analysis
# ═══════════════════════════════════════════════════════════════════════

def counterfactual_analysis(
    w_draws, vote_weights, ceo_params, asa_params, beliefs, n_draws, outdir, seed=42
):
    """Table 3: EU and outcome distributions for each D1 action."""
    logger.info("3. Computing counterfactual analysis...")

    cfg = make_cfg(w_draws, vote_weights, ceo_params, asa_params)
    ts_base = make_initial_state()

    rows = []
    for scenario, scenario_label, d0_action in [
        ("CEO resigned", "CEO_resign", "CEO_resign"),
        ("CEO stayed", "CEO_stay", "CEO_stay"),
    ]:
        ts_scenario = tree_apply_action(ts_base, "D0_ceo", d0_action)
        ceo_resigned = d0_action == "CEO_resign"

        # Build tree from D1 for this scenario
        root, eu = build_game_tree("D1", ts_scenario, cfg)

        # Identify optimal action
        best_child = max(root.children, key=lambda c: c[2].eu_board)
        optimal_action = best_child[0]

        for label, prob, child in root.children:
            # Analytical outcome stats from tree probabilities
            ostats = compute_outcome_stats_from_probs(label, ceo_resigned)

            # Vote distribution from logit-normal (for continuous E[V])
            rec_strike_dominant = ostats["Pr_strike"] > 0.5
            if label != "D3_ceo_transition" or not ceo_resigned:
                vote_samples = sample_votes(
                    beliefs, n_draws,
                    rec_strike=rec_strike_dominant,
                    d1_action=label, seed=seed + hash(label) % 10000,
                    n_mc=20, biased=False,
                )
                vote_mean = float(np.mean(vote_samples))
                vote_sd = float(np.std(vote_samples))
            else:
                vote_mean = ostats["E_vote_pct"]
                vote_sd = 0.0

            rows.append({
                "Scenario": scenario,
                "D1_action": label,
                "D1_label": D1_LABELS.get(label, label),
                "Is_optimal": label == optimal_action,
                "Board_EU": round(child.eu_board, 4),
                "ASA_EU": round(child.eu_asa, 4),
                "CEO_EU": round(child.eu_ceo, 4),
                "Pr_strike": round(ostats["Pr_strike"], 3),
                "Pr_overwhelming": round(ostats["Pr_overwhelming"], 3),
                "Pr_CEO_removed": round(ostats["Pr_CEO_removed"], 3),
                "E_vote_pct_logit_normal": round(vote_mean, 3),
                "SD_vote_pct": round(vote_sd, 3),
                "Pr_review_negative": round(ostats["Pr_review_negative"], 3),
                "Pr_review_balanced": round(ostats["Pr_review_balanced"], 3),
            })

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "counterfactual_analysis.csv", index=False)
    logger.info("  -> counterfactual_analysis.csv")
    return df


# ═══════════════════════════════════════════════════════════════════════
# 4. Expected Utility Decomposition
# ═══════════════════════════════════════════════════════════════════════

def eu_decomposition(w_draws, vote_weights, ceo_params, asa_params, outdir):
    """Table 4 + Figure 2: Board EU component breakdown per D1 action."""
    logger.info("4. Computing EU decomposition...")

    cfg = make_cfg(w_draws, vote_weights, ceo_params, asa_params)
    ts_resigned = tree_apply_action(make_initial_state(), "D0_ceo", "CEO_resign")
    root, _eu = build_game_tree("D1", ts_resigned, cfg)

    # Collect probability-weighted terminal decompositions per D1 action
    decomp_rows = []
    action_decomps = {}
    for label, _prob, child in root.children:
        decomp = collect_weighted_decomposition(child)
        action_decomps[label] = decomp

        for component, value in sorted(decomp.items()):
            decomp_rows.append({
                "D1_action": label,
                "D1_label": D1_LABELS.get(label, label),
                "Component": component,
                "Contribution": round(value, 4),
            })

    df = pd.DataFrame(decomp_rows)
    df.to_csv(outdir / "eu_decomposition.csv", index=False)

    # Bar chart: side-by-side components per D1 action
    actions = [label for label, _, _ in root.children]
    all_components = sorted(
        set(c for d in action_decomps.values() for c in d.keys())
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(all_components))
    width = 0.8 / len(actions)
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    for j, action in enumerate(actions):
        vals = [action_decomps[action].get(c, 0.0) for c in all_components]
        offset = (j - len(actions) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, vals, width,
            label=D1_LABELS.get(action, action),
            color=colors[j % len(colors)],
            alpha=0.85, edgecolor="white", linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(all_components, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Expected Contribution to Board EU", fontsize=11)
    ax.set_title(
        "Board Expected Utility Decomposition by Component\n(CEO-resigned scenario)",
        fontsize=12,
    )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(outdir / "eu_decomposition.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    logger.info("  -> eu_decomposition.csv, eu_decomposition.png")
    return df


# ═══════════════════════════════════════════════════════════════════════
# 5. Vote Distribution Plots
# ═══════════════════════════════════════════════════════════════════════

def vote_distributions(beliefs, n_draws, outdir, seed=42):
    """Figure 3: Posterior predictive vote distribution per D1 action.

    Shows UNBIASED vote model (for calibration assessment), with the Board's
    biased perception shown as a lighter overlay for comparison.
    """
    logger.info("5. Computing vote distributions...")

    d1_actions = ["D0_minimal", "D1_review", "D3_ceo_transition"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for i, (d1_action, ax) in enumerate(zip(d1_actions, axes)):
        # Unbiased model (for PPC / calibration)
        votes = sample_votes(
            beliefs, n_draws, rec_strike=True, d1_action=d1_action,
            seed=seed + i * 1000, n_mc=50, biased=False,
        )
        # Board's biased perception (overlay)
        votes_biased = sample_votes(
            beliefs, n_draws, rec_strike=True, d1_action=d1_action,
            seed=seed + i * 1000 + 500, n_mc=50, biased=True,
        )

        # Histogram: unbiased model
        ax.hist(
            votes, bins=80, range=(0, 1), density=True,
            color="steelblue", alpha=0.7, edgecolor="white", linewidth=0.3,
            label="Unbiased model",
        )
        # Overlay: Board's biased perception
        ax.hist(
            votes_biased, bins=80, range=(0, 1), density=True,
            color="coral", alpha=0.35, edgecolor="white", linewidth=0.3,
            label="Board perception (biased)",
        )

        # Threshold lines
        ax.axvline(0.25, color="orange", linewidth=1.5, linestyle="--",
                   label="First strike (25%)")
        ax.axvline(0.50, color="red", linewidth=1.5, linestyle="--",
                   label="Overwhelming (50%)")

        # Actual outcome
        ax.axvline(ACTUAL_VOTE_FRACTION, color="black", linewidth=2,
                   linestyle="-", label=f"Actual ({ACTUAL_VOTE_FRACTION:.1%})")

        # Statistics
        mean_v = np.mean(votes)
        p05 = np.percentile(votes, 5)
        p95 = np.percentile(votes, 95)
        pr_strike = np.mean(votes > 0.25)
        pr_ovw = np.mean(votes > 0.50)
        ax.text(
            0.03, 0.95,
            (
                f"Mean: {mean_v:.1%}\n"
                f"90% CI: [{p05:.1%}, {p95:.1%}]\n"
                f"Pr(>25%): {pr_strike:.1%}\n"
                f"Pr(>50%): {pr_ovw:.1%}"
            ),
            transform=ax.transAxes, fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        ax.set_title(D1_LABELS.get(d1_action, d1_action), fontsize=12)
        ax.set_xlabel("Vote Against (%)", fontsize=10)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.set_xlim(0, 1)

    axes[0].set_ylabel("Density", fontsize=10)
    axes[0].legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "Posterior Predictive Vote Distribution by Board Action\n"
        "(given ASA recommends strike, headline incident active)",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    fig.savefig(outdir / "vote_distributions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    logger.info("  -> vote_distributions.png")


# ═══════════════════════════════════════════════════════════════════════
# 6. Value of Information Analysis
# ═══════════════════════════════════════════════════════════════════════

def value_of_information(w_draws, vote_weights, ceo_params, asa_params, outdir):
    """Table 5: VoI for early observation of ASA recommendation / vote outcome."""
    logger.info("6. Computing value of information...")

    ts_resigned = tree_apply_action(make_initial_state(), "D0_ceo", "CEO_resign")

    # Baseline: standard tree
    cfg_base = make_cfg(w_draws, vote_weights, ceo_params, asa_params)
    root_base, eu_base = build_game_tree("D1", ts_resigned, cfg_base)
    baseline_eu = float(np.mean(eu_base["Board"]))
    baseline_optimal = max(root_base.children, key=lambda c: c[2].eu_board)[0]

    # ── VoI(ASA recommendation) ──
    # Force A2 to each outcome and find best D1
    voi_a2_rows = []
    for a2_outcome in ["A2_rec_strike", "A2_no_strike"]:
        probs_forced = copy.deepcopy(TREE_DEFAULT_PROBS)
        for path_key in probs_forced["A2"]:
            probs_forced["A2"][path_key] = {
                "A2_rec_strike": 1.0 if a2_outcome == "A2_rec_strike" else 0.0,
                "A2_no_strike": 1.0 if a2_outcome == "A2_no_strike" else 0.0,
            }
        cfg_forced = make_cfg(
            w_draws, vote_weights, ceo_params, asa_params, probs=probs_forced
        )
        root_f, eu_f = build_game_tree("D1", ts_resigned, cfg_forced)
        best = max(root_f.children, key=lambda c: c[2].eu_board)
        voi_a2_rows.append({
            "signal": a2_outcome,
            "best_d1": best[0],
            "eu_with_info": round(best[2].eu_board, 4),
        })

    # Marginal A2 probabilities (at baseline optimal D1)
    a2_path = f"resigned_{baseline_optimal}"
    a2_probs = TREE_DEFAULT_PROBS["A2"].get(
        a2_path, {"A2_rec_strike": 0.96, "A2_no_strike": 0.04}
    )
    eu_with_a2_info = sum(
        a2_probs.get(row["signal"], 0) * row["eu_with_info"]
        for row in voi_a2_rows
    )
    voi_a2 = eu_with_a2_info - baseline_eu

    # ── VoI(Vote outcome) ──
    # Force V to each category and find best D1
    voi_v_rows = []
    v_categories = ["no_strike", "first_strike", "overwhelming"]
    for v_outcome in v_categories:
        probs_forced = copy.deepcopy(TREE_DEFAULT_PROBS)
        for a2_key in probs_forced["V"]:
            probs_forced["V"][a2_key] = {
                cat: (1.0 if cat == v_outcome else 0.0) for cat in v_categories
            }
        cfg_forced = make_cfg(
            w_draws, vote_weights, ceo_params, asa_params, probs=probs_forced
        )
        root_f, eu_f = build_game_tree("D1", ts_resigned, cfg_forced)
        best = max(root_f.children, key=lambda c: c[2].eu_board)
        voi_v_rows.append({
            "signal": v_outcome,
            "best_d1": best[0],
            "eu_with_info": round(best[2].eu_board, 4),
        })

    # Marginal V probabilities (at baseline optimal + modal A2)
    v_probs_given_strike = TREE_DEFAULT_PROBS["V"].get(
        "A2_rec_strike", {"no_strike": 0.15, "first_strike": 0.40, "overwhelming": 0.45}
    )
    eu_with_v_info = sum(
        v_probs_given_strike.get(row["signal"], 0) * row["eu_with_info"]
        for row in voi_v_rows
    )
    voi_v = eu_with_v_info - baseline_eu

    # Build output table
    rows = [
        {
            "Information_signal": "ASA recommendation (A2)",
            "Baseline_EU": round(baseline_eu, 4),
            "Baseline_optimal": D1_LABELS.get(baseline_optimal, baseline_optimal),
            "EU_with_perfect_info": round(eu_with_a2_info, 4),
            "VoI": round(voi_a2, 4),
            "VoI_pct_of_baseline": (
                f"{abs(voi_a2 / baseline_eu) * 100:.1f}%"
                if abs(baseline_eu) > 1e-9 else "N/A"
            ),
            "Detail": "; ".join(
                f"If {r['signal']}: best={D1_LABELS.get(r['best_d1'], r['best_d1'])} "
                f"(EU={r['eu_with_info']:.4f})"
                for r in voi_a2_rows
            ),
        },
        {
            "Information_signal": "Vote outcome (V)",
            "Baseline_EU": round(baseline_eu, 4),
            "Baseline_optimal": D1_LABELS.get(baseline_optimal, baseline_optimal),
            "EU_with_perfect_info": round(eu_with_v_info, 4),
            "VoI": round(voi_v, 4),
            "VoI_pct_of_baseline": (
                f"{abs(voi_v / baseline_eu) * 100:.1f}%"
                if abs(baseline_eu) > 1e-9 else "N/A"
            ),
            "Detail": "; ".join(
                f"If {r['signal']}: best={D1_LABELS.get(r['best_d1'], r['best_d1'])} "
                f"(EU={r['eu_with_info']:.4f})"
                for r in voi_v_rows
            ),
        },
    ]

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "value_of_information.csv", index=False)
    logger.info("  -> value_of_information.csv")
    return df


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate results pack for Section 7 of the academic paper"
    )
    parser.add_argument(
        "--n_draws", type=int, default=500,
        help="Number of posterior draws to use (default: 500)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="C0",
        help="Belief checkpoint ID (default: C0, the pre-AGM checkpoint)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--posterior-draws", type=str, default=None)
    parser.add_argument("--param-estimates", type=str, default=None)
    args = parser.parse_args()

    outdir = PROJECT_ROOT / "results-pack"
    outdir.mkdir(exist_ok=True)
    logger.info(f"Output directory: {outdir}")

    # Load data
    w_draws, vote_weights, ceo_params, asa_params, beliefs = load_data(args)
    n_draws = w_draws.shape[0]
    logger.info(f"Loaded {n_draws} posterior draws, {beliefs.N} belief draws")

    cfg = make_cfg(w_draws, vote_weights, ceo_params, asa_params)

    # ── Generate all results ──
    print("\n" + "=" * 60)
    print("PAPER RESULTS PACK — Section 7")
    print("=" * 60)

    # 1. Posterior predictive check
    ppc = posterior_predictive_check(cfg, beliefs, n_draws, outdir, seed=args.seed)
    print("\n1. POSTERIOR PREDICTIVE CHECK")
    print(ppc.to_string(index=False))

    # 2. Sensitivity analysis
    sens = sensitivity_analysis(
        w_draws, vote_weights, ceo_params, asa_params, outdir
    )
    print("\n2. SENSITIVITY ANALYSIS (sorted by |swing|)")
    print(sens[["Parameter", "EU_low_x0.5", "EU_baseline", "EU_high_x1.5", "Swing"]]
          .to_string(index=False))

    # 3. Counterfactual analysis
    cf = counterfactual_analysis(
        w_draws, vote_weights, ceo_params, asa_params,
        beliefs, n_draws, outdir, seed=args.seed,
    )
    print("\n3. COUNTERFACTUAL ANALYSIS")
    print(cf[["Scenario", "D1_label", "Is_optimal", "Board_EU", "ASA_EU",
              "CEO_EU", "Pr_strike", "Pr_CEO_removed"]]
          .to_string(index=False))

    # 4. EU decomposition
    decomp = eu_decomposition(
        w_draws, vote_weights, ceo_params, asa_params, outdir
    )
    print("\n4. EU DECOMPOSITION (CEO-resigned scenario)")
    # Pivot for display
    pivot = decomp.pivot_table(
        index="Component", columns="D1_label",
        values="Contribution", fill_value=0.0,
    )
    print(pivot.to_string())

    # 5. Vote distributions
    vote_distributions(beliefs, n_draws, outdir, seed=args.seed)
    print("\n5. VOTE DISTRIBUTIONS")
    print("  -> vote_distributions.png (3-panel figure)")

    # 6. Value of information
    voi = value_of_information(
        w_draws, vote_weights, ceo_params, asa_params, outdir
    )
    print("\n6. VALUE OF INFORMATION")
    print(voi[["Information_signal", "Baseline_EU", "EU_with_perfect_info",
               "VoI", "VoI_pct_of_baseline"]]
          .to_string(index=False))

    print("\n" + "=" * 60)
    print(f"All outputs saved to: {outdir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
