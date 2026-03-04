"""
Visualise the ARA game tree with annotations from solver results.

Generates two left-to-right tree diagrams from a SINGLE unified tree
starting at D0_ceo (CEO resign/stay decision):
  1. Probability tree  — edge labels show split probabilities
  2. Expected utility tree — node labels show EU values

The D0_ceo root branches into CEO_resign and CEO_stay, each with the
full sub-game rooted at D1.  Edge probabilities at D0_ceo come from
the Bayesian prior Beta(12, 1.5) by default (mean = 88.9% resign),
or from the full Level-2 ARA prediction with --compute-d0.

Strategy:
  - Run the solver for BOTH scenarios to get EU per initial action
  - Build the visual tree by walking game structure with:
    * Fixed policies for decision-node probabilities (instant)
    * Cheap MC sampling for chance-node probabilities
    * Direct utility computation at terminals (instant)
  - Use solver results to annotate the D1 root with accurate EU values

Node shapes:   box = decision,  ellipse = chance,  diamond = terminal
Node colours:  Board = blue,  ASA = green,  CEO = red,  Nature = grey

Usage:
    python -m run.visualise_tree --checkpoint C0 --n_draws 5
    python -m run.visualise_tree --checkpoint C3 --focal ASA --n_draws 10
    python -m run.visualise_tree --compute-d0 --n_draws 5
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Graphviz binary on Windows
_GV_BIN = Path(r"C:\Program Files\Graphviz\bin")
if _GV_BIN.exists():
    os.environ["PATH"] = str(_GV_BIN) + os.pathsep + os.environ.get("PATH", "")

import graphviz

from engine.solver import Solver, SolveResult
from engine.state import DecisionState, BeliefBundle
from engine.modes import (
    AVAILABLE_MODES, ModeConfig,
    MODE_BOARD_POLICY_ASA, MODE_ASA_POLICY_BOARD,
)
from engine.chance_models import ChanceModels, OverconfidenceBias
from engine.predictive import PredictiveDistribution
from engine.utilities import TerminalOutcome, compute_utility

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Visual constants ────────────────────────────────────────────────
OWNER_COLOURS = {
    "Board": "#4A90D9", "ASA": "#50C878",
    "CEO": "#E85D5D", "Nature": "#AAAAAA",
}
TYPE_SHAPES = {"decision": "box", "chance": "ellipse", "terminal": "diamond"}

NICE_NODE = {
    "D0_ceo": "D0_ceo\nCEO\nResign?",
    "D1": "D1\nBoard\nGovernance",
    "A2": "A2\nASA\nStrike Rec.",
    "V": "V\nShareholder\nVote",
    "M_agm": "M_agm\nMarket (AGM)",
    "D4": "D4\nCEO\nResponse",
    "D_rev": "D_rev\nBoard\nReview Resp.",
    "R": "R\nReview\nFindings",
    "M_rev": "M_rev\nMarket (Rev.)",
    "D4_post_review": "D4'\nCEO\nPost-Review",
    "D_rev_post_review": "D_rev'\nBoard\nPost-Review",
    "Terminal": "Terminal",
}
NICE_EDGE = {
    "CEO_resign": "CEO resigns\n(05-Sep-2023)",
    "CEO_stay": "CEO stays\n(counterfactual)",
    "D0_minimal": "Do nothing",
    "D1_review": "Commission review",
    "D3_ceo_transition": "Force CEO exit",
    "A2_no_strike": "No strike",
    "A2_rec_strike": "Rec. strike",
    "Drev_no_action": "No action",
    "Drev_commission_review": "Commission review",
    "Drev_sack_ceo": "Sack CEO",
    "D4_stay": "Stay",
    "D4_resign": "Resign",
    "D4_negotiate_exit": "Negotiate exit",
}


# ── Data structures ─────────────────────────────────────────────────

@dataclass
class VizNode:
    id: str
    node_name: str
    node_type: str          # decision / chance / terminal
    owner: str
    eu: float = 0.0
    children: list = field(default_factory=list)   # (label, prob, VizNode)


# ── Cheap MC helpers for chance-node probabilities ──────────────────

def _sample_vote_probs(
    beliefs, history, state, chance, bias, n, seed,
):
    """Return {pr_strike, pr_no, avg_v_strike, avg_v_no}."""
    sc, nc = 0, 0
    vs, vn = 0.0, 0.0
    for i in range(n):
        rng = np.random.default_rng(seed + i + 5000)
        d1 = history.get("D1", "D0_minimal")
        ge = chance.vote._governance_effect(d1, rng, bias=bias)
        ss = bias.sigma_scale if bias else None
        vo = chance.sample_vote(i % beliefs.N, beliefs, history, state, rng,
                                governance_effect=ge, sigma_scale=ss)
        if vo.strike_indicator:
            sc += 1; vs += vo.vote_percent
        else:
            nc += 1; vn += vo.vote_percent
    pr_s = sc / max(n, 1)
    return dict(pr_strike=pr_s, pr_no=1 - pr_s,
                avg_v_strike=vs / sc if sc else 0.35,
                avg_v_no=vn / nc if nc else 0.15)


def _sample_review_stats(beliefs, history, state, chance, bias, n, seed):
    """Return P(adverse), mean CAR for adverse branch, mean CAR for clean branch."""
    adv_count = 0
    car_adv_sum = 0.0
    car_clean_sum = 0.0
    clean_count = 0
    for i in range(n):
        out = chance.sample_review(
            i % beliefs.N, beliefs, history, state,
            np.random.default_rng(seed + i + 8000), bias=bias,
        )
        if out.review_adverse:
            adv_count += 1
            car_adv_sum += out.review_car
        else:
            clean_count += 1
            car_clean_sum += out.review_car
    p_adv = adv_count / max(n, 1)
    mean_car_adv = car_adv_sum / max(adv_count, 1)
    mean_car_clean = car_clean_sum / max(clean_count, 1)
    return dict(p_adv=p_adv, mean_car_adv=mean_car_adv,
                mean_car_clean=mean_car_clean)


def _terminal_eu(history, state, focal, weights):
    outcome = PredictiveDistribution._build_outcome(history, state)
    return compute_utility(focal, outcome, weights)


# ── Subtree builder (uses fixed policies — no ARA recursion) ────────

def _sample_policy_probs(pred, owner, node_name, history, state, feasible,
                         n, seed):
    """Sample a fixed policy n times to get an empirical distribution."""
    counts = {a: 0 for a in feasible}
    for i in range(n):
        rng = np.random.default_rng(seed + i + 3000)
        a = pred._fixed_policy(owner, node_name, history, state, feasible,
                               rng=rng)
        counts[a] += 1
    total = sum(counts.values())
    return {a: c / total for a, c in counts.items()} if total else {
        a: 1.0 / len(feasible) for a in feasible}


def _sample_ara_probs(pred, node_name, history, state, focal, mode, n, seed):
    """Compute path-conditioned ARA predictive distribution.

    For post-chance opponent nodes (D4, D_rev, etc.), the ARA predictive
    depends on the vote result which is in history. Averages over n
    belief draws for a representative estimate.
    """
    agg = {}
    for i in range(min(n, 5)):
        rng = np.random.default_rng(seed + i + 5000)
        d_i = pred.predict(
            node_name, history, state, i, focal, mode,
            mode.level, rng,
        )
        for a, p in d_i.items():
            agg[a] = agg.get(a, 0.0) + p
    total = sum(agg.values())
    if total == 0:
        feasible = state.feasible_actions(node_name)
        return {a: 1.0 / len(feasible) for a in feasible}
    return {a: v / total for a, v in agg.items()}


def _build_scenario_subtree(
    solver: Solver,
    result: SolveResult,
    focal: str,
    checkpoint_id: str,
    mode: ModeConfig,
    bias: Optional[OverconfidenceBias],
    n_mc: int,
    scenario: str,
    ctr: list[int],
) -> VizNode:
    """Build the subtree from D1 onward for a single scenario."""
    cp_path = solver._find_checkpoint(checkpoint_id)
    beliefs = BeliefBundle(cp_path)
    chance = ChanceModels(solver.vote_thresholds)
    base_state = DecisionState.from_governance_spec(
        solver.governance_spec_path, checkpoint_id=checkpoint_id)
    base_state = base_state.for_scenario(scenario)
    weights = solver.utility_weights.get(focal, {})
    seed = solver.seed

    # Light-weight predictive engine — used for _fixed_policy() sampling
    pred = PredictiveDistribution(
        beliefs=beliefs, param_sampler=solver.param_sampler,
        chance_models=chance, policy_params=solver.policy_params,
        K=20, R_rollouts=5, overconfidence_bias=bias,
    )

    # Index solver predictive dists
    _solver_preds = {}
    for d1_act, node_dists in result.predictive_dists.items():
        for node_label, dist in node_dists.items():
            node_key = node_label.split(" ")[0]
            _solver_preds[(d1_act, node_key)] = dist

    def _nid():
        ctr[0] += 1
        return f"n{ctr[0]}"

    def _walk(node_name, history, state, depth):
        ntype = state.node_type(node_name) if node_name else "terminal"
        owner = state.node_owner(node_name) if node_name else "Nature"

        if node_name is None or ntype == "terminal" or depth > 12:
            eu = _terminal_eu(history, state, focal, weights)
            return VizNode(_nid(), "Terminal", "terminal", "Nature", eu=eu)

        nd = VizNode(_nid(), node_name, ntype, owner)

        if ntype == "decision":
            feasible = state.feasible_actions(node_name)
            if not feasible:
                # No feasible actions (e.g. D4 when CEO already resigned/sacked)
                # Skip this node entirely and proceed to next
                return _walk(state.next_node(node_name), history, state, depth)

            if node_name == "D1" and depth == 0:
                # Root of subtree: use solver EU values directly
                best = result.optimal_action
                for a in feasible:
                    h = dict(history); h[node_name] = a
                    s = state.apply(node_name, a)
                    child = _walk(s.next_node(node_name), h, s, depth + 1)
                    p = 1.0 if a == best else 0.0
                    nd.children.append((a, p, child))
                nd.eu = result.optimal_EU

            elif not mode.is_focal(owner):
                d1_act = history.get("D1", "D0_minimal")
                # Pre-chance nodes (D0_ceo, A2): use solver predictive dists
                # (unconditional on vote/review outcomes — valid before V).
                # Post-chance nodes (D4, D_rev, etc.): use path-conditioned
                # ARA predictive — the correct answer depends on the vote
                # result which is in the history at this point.
                _pre_chance_nodes = {"D0_ceo", "A2"}
                solver_dist = None
                if node_name in _pre_chance_nodes:
                    solver_dist = _solver_preds.get((d1_act, node_name))
                if solver_dist:
                    probs = solver_dist
                else:
                    # Path-conditioned ARA predictive: K=30, R=5 for speed
                    probs = _sample_ara_probs(
                        pred, node_name, history, state,
                        focal, mode, n_mc, seed)
                for a in feasible:
                    h = dict(history); h[node_name] = a
                    s = state.apply(node_name, a)
                    child = _walk(s.next_node(node_name), h, s, depth + 1)
                    nd.children.append((a, probs.get(a, 0.0), child))

            else:
                # Focal decision node: build children first, then assign
                # 100% probability to the action with the highest EU.
                child_data = []
                for a in feasible:
                    h = dict(history); h[node_name] = a
                    s = state.apply(node_name, a)
                    child = _walk(s.next_node(node_name), h, s, depth + 1)
                    child_data.append((a, child))

                # Find action with best EU among children
                best_action = max(child_data, key=lambda x: x[1].eu)[0]
                for a, child in child_data:
                    p = 1.0 if a == best_action else 0.0
                    nd.children.append((a, p, child))

        elif ntype == "chance":
            if node_name in ("M_agm", "M_rev"):
                h = dict(history); h[node_name] = "market_reaction"
                child = _walk(state.next_node(node_name), h, state, depth + 1)
                nd.children.append(("pass-through", 1.0, child))

            elif node_name == "V":
                vp = _sample_vote_probs(
                    beliefs, history, state, chance, bias, n_mc, seed)
                for lbl, pr, vpct, strike, ovw in [
                    ("No strike (<25%)", vp["pr_no"],
                     vp["avg_v_no"], False, False),
                    ("Strike (>=25%)", vp["pr_strike"],
                     vp["avg_v_strike"], True, False),
                ]:
                    h = dict(history)
                    h["V"] = "vote"; h["V_percent"] = vpct
                    h["V_strike"] = strike; h["V_overwhelming"] = ovw
                    child = _walk(
                        state.next_node("V"), h, state, depth + 1)
                    nd.children.append((lbl, pr, child))

            elif node_name == "R":
                if not state.review_commissioned:
                    h = dict(history)
                    h["R"] = "review"; h["R_adverse"] = False
                    h["R_car"] = 0.0
                    sc = state.apply("R", "no_adverse")
                    child = _walk(state.next_node("R"), h, sc, depth + 1)
                    nd.children.append(("No review", 1.0, child))
                else:
                    rs = _sample_review_stats(
                        beliefs, history, state, chance, bias, n_mc, seed)
                    for lbl, pr, adv, car in [
                        ("No adverse", 1 - rs["p_adv"], False,
                         rs["mean_car_clean"]),
                        ("Adverse finding", rs["p_adv"], True,
                         rs["mean_car_adv"]),
                    ]:
                        h = dict(history)
                        h["R"] = "review"; h["R_adverse"] = adv
                        h["R_car"] = car
                        s = state.apply(
                            "R", "adverse" if adv else "no_adverse")
                        child = _walk(
                            state.next_node("R"), h, s, depth + 1)
                        nd.children.append((lbl, pr, child))

        return nd

    root = _walk("D1", {}, base_state, 0)

    # Back-propagate EU from terminals
    def _bp(nd):
        if not nd.children:
            return nd.eu
        # Recurse into all children first so their EUs are correct
        for _, p, ch in nd.children:
            _bp(ch)

        # Focal decision nodes (except D1 root): reassign probabilities
        # based on now-correct child EUs.  _walk() couldn't do this because
        # child EUs were still 0.0 at tree-construction time.
        if (nd.node_type == "decision" and mode.is_focal(nd.owner)
                and nd.node_name != "D1"):
            best_idx = max(range(len(nd.children)),
                           key=lambda i: nd.children[i][2].eu)
            nd.children = [
                (label, 1.0 if i == best_idx else 0.0, child)
                for i, (label, _, child) in enumerate(nd.children)
            ]
            nd.eu = nd.children[best_idx][2].eu
        else:
            total = sum(p * ch.eu for _, p, ch in nd.children)
            if nd.node_name == "D1":
                pass  # Keep solver EU at root
            elif nd.eu == 0.0:
                nd.eu = total

        return nd.eu
    _bp(root)

    # Override D1-branch EU from solver result
    for action, _, child in root.children:
        if action in result.EU_per_action:
            child.eu = result.EU_per_action[action]

    return root


def build_unified_tree(
    solver: Solver,
    results: dict[str, SolveResult],
    focal: str,
    checkpoint_id: str,
    mode: ModeConfig,
    bias: Optional[OverconfidenceBias],
    n_mc: int = 200,
    d0_probs: Optional[dict[str, float]] = None,
) -> VizNode:
    """Build a unified tree starting at D0_ceo with both scenario subtrees."""
    ctr = [0]

    def _nid():
        ctr[0] += 1
        return f"n{ctr[0]}"

    # D0_ceo root
    root = VizNode(_nid(), "D0_ceo", "decision", "CEO")

    # D0_ceo edge probabilities
    if d0_probs is None:
        alpha = solver.ceo_departure_prior_alpha
        beta = solver.ceo_departure_prior_beta
        d0_probs = {
            "CEO_resign": alpha / (alpha + beta),
            "CEO_stay": beta / (alpha + beta),
        }

    # Build subtree for each scenario
    scenario_map = {
        "CEO_resign": "ceo_resigned",
        "CEO_stay": "ceo_stayed",
    }

    for action in ["CEO_resign", "CEO_stay"]:
        scenario = scenario_map[action]
        result = results.get(scenario)
        if result is None:
            continue

        subtree = _build_scenario_subtree(
            solver, result, focal, checkpoint_id, mode, bias,
            n_mc, scenario, ctr,
        )

        prob = d0_probs.get(action, 0.5)
        root.children.append((action, prob, subtree))

    # D0_ceo EU = weighted sum of subtree EUs
    root.eu = sum(p * ch.eu for _, p, ch in root.children)

    return root


# ── Graphviz rendering ──────────────────────────────────────────────

def render_tree(root, title, diagram_mode, output_path, fmt="png"):
    g = graphviz.Digraph(format=fmt, engine="dot")
    g.attr(rankdir="LR", label=title, labelloc="t", fontsize="18",
           fontname="Helvetica", bgcolor="#FAFAFA", dpi="150",
           nodesep="0.5", ranksep="1.0", margin="0.3")
    g.attr("node", fontname="Helvetica", fontsize="10",
           style="filled", penwidth="1.5")
    g.attr("edge", fontname="Helvetica", fontsize="9")

    def _add(nd):
        colour = OWNER_COLOURS.get(nd.owner, "#CCC")
        shape = TYPE_SHAPES.get(nd.node_type, "ellipse")
        nice = NICE_NODE.get(nd.node_name, nd.node_name)
        if diagram_mode == "eu":
            label = f"{nice}\nEU={nd.eu:+.3f}"
        else:
            label = nice

        g.node(nd.id, label=label, shape=shape,
               fillcolor=colour, fontcolor="white")

        for action, prob, child in nd.children:
            _add(child)
            alabel = NICE_EDGE.get(action, action)
            if diagram_mode == "prob":
                elabel = f"{alabel}\np={prob:.2f}"
            else:
                elabel = f"{alabel}\nEU={child.eu:+.3f}"
            pw = str(max(0.5, prob * 4.0))
            ecol = "#333" if prob > 0.01 else "#CCC"
            g.edge(nd.id, child.id, label=elabel,
                   penwidth=pw, color=ecol, fontcolor="#333")

    _add(root)
    g.render(output_path, cleanup=True)
    logger.info(f"Saved: {output_path}.{fmt}")


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Visualise the ARA game tree")
    p.add_argument("--checkpoint", default="C0")
    p.add_argument("--focal", default="Board", choices=["Board", "ASA"])
    p.add_argument("--n_draws", type=int, default=5,
                   help="Belief draws for solver (default 5; keep small)")
    p.add_argument("--n_mc", type=int, default=200,
                   help="MC samples for chance-node probabilities")
    p.add_argument("--no-bias", action="store_true")
    p.add_argument("--compute-d0", action="store_true",
                   help="Compute D0_ceo probabilities via full Level-2 ARA "
                        "(slow). Default: use Bayesian prior Beta(12, 1.5).")
    p.add_argument("--format", default="png", choices=["png", "svg", "pdf"])
    p.add_argument("--interactive", action="store_true",
                   help="Generate interactive HTML tree (D3.js)")
    p.add_argument("--actual-outcomes", type=str, default=None,
                   help="Path to actual outcomes JSON config (for red lines in interactive mode)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    data = PROJECT_ROOT / "data"
    out = PROJECT_ROOT / "outputs"
    out.mkdir(exist_ok=True)

    # Use Policy modes for the solver — much faster than ARA
    if args.focal == "Board":
        mode = MODE_BOARD_POLICY_ASA
    else:
        mode = MODE_ASA_POLICY_BOARD

    solver = Solver(
        governance_spec_path=data / "governance_spec.xlsx",
        opponent_priors_path=data / "opponent_priors.xlsx",
        checkpoint_dir=data / "checkpoints",
        K=30, R_rollouts=5, seed=args.seed,
    )
    bias = None if args.no_bias else solver.overconfidence_bias

    # Solve both scenarios
    results = {}
    for scenario in ["ceo_stayed", "ceo_resigned"]:
        logger.info(f"Solving: {args.focal}, {args.checkpoint}, scenario={scenario}, n={args.n_draws}")
        result = solver.solve(
            focal_actor=args.focal, checkpoint_id=args.checkpoint,
            mode=mode, n_draws=args.n_draws, overconfidence_bias=bias,
            scenario=scenario,
        )
        results[scenario] = result
        result.print_diagnostics()

    # D0_ceo probabilities
    d0_probs = None
    if args.compute_d0:
        logger.info("Computing D0_ceo via Level-2 ARA (this takes a while)...")
        d0_probs = solver.predict_d0_ceo(
            focal_actor=args.focal,
            checkpoint_id=args.checkpoint,
            n_draws=args.n_draws,
            overconfidence_bias=bias,
        )
        logger.info(f"D0_ceo ARA prediction: {d0_probs}")
    else:
        alpha = solver.ceo_departure_prior_alpha
        beta = solver.ceo_departure_prior_beta
        d0_probs = {
            "CEO_resign": alpha / (alpha + beta),
            "CEO_stay": beta / (alpha + beta),
        }
        logger.info(
            f"D0_ceo from Bayesian prior Beta({alpha},{beta}): "
            f"resign={d0_probs['CEO_resign']:.1%}, stay={d0_probs['CEO_stay']:.1%}"
        )

    # Build unified tree
    logger.info("Building unified tree from D0_ceo...")
    root = build_unified_tree(
        solver, results, args.focal, args.checkpoint, mode, bias,
        n_mc=args.n_mc, d0_probs=d0_probs,
    )

    tag = f"{args.focal}_{args.checkpoint}"

    # Render
    render_tree(root,
                title=f"Game Tree — Probabilities  ({args.focal}, {args.checkpoint})",
                diagram_mode="prob",
                output_path=str(out / f"tree_prob_{tag}"), fmt=args.format)
    render_tree(root,
                title=f"Game Tree — Expected Utility  ({args.focal}, {args.checkpoint})",
                diagram_mode="eu",
                output_path=str(out / f"tree_eu_{tag}"), fmt=args.format)

    print(f"\nDone. Diagrams in {out}/")
    print(f"  tree_prob_{tag}.{args.format}")
    print(f"  tree_eu_{tag}.{args.format}")

    # Interactive HTML
    if args.interactive:
        from run.interactive_tree import render_interactive_tree
        actual_path = args.actual_outcomes or str(data / "actual_outcomes.json")
        html_path = render_interactive_tree(
            root=root,
            results=results,
            focal=args.focal,
            checkpoint_id=args.checkpoint,
            actual_outcomes_path=actual_path,
            output_dir=out,
            d0_probs=d0_probs,
        )
        print(f"  {html_path.name}")

    print(f"\nD0_ceo: resign={d0_probs['CEO_resign']:.1%}, stay={d0_probs['CEO_stay']:.1%}")


if __name__ == "__main__":
    main()
