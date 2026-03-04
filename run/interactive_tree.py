"""
Interactive HTML tree visualisation for the ARA game tree.

Generates a self-contained HTML file with an interactive D3.js tree that
supports expand/collapse, actual-outcome highlighting (red lines),
edge mouseover tooltips with LLM commentary, and node info popups
with utility decomposition and opponent predictive distributions.

Usage (via visualise_tree.py):
    python -m run.visualise_tree --checkpoint C0 --interactive
    python -m run.visualise_tree --checkpoint C0 --interactive --actual-outcomes data/actual_outcomes.json
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from run.visualise_tree import (
    VizNode, OWNER_COLOURS, NICE_NODE, NICE_EDGE,
)
from engine.solver import SolveResult

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Actual outcomes loading ───────────────────────────────────────────

def load_actual_outcomes(config_path: Optional[Path]) -> dict:
    """Load actual historical outcomes from a JSON config file.

    Returns empty dict if path is None, file missing, or JSON invalid.
    """
    if config_path is None:
        return {}
    p = Path(config_path)
    if not p.exists():
        logger.warning(f"Actual outcomes file not found: {p}")
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Strip private keys (comments, details)
        return {k: v for k, v in data.items() if not k.startswith("_")}
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Could not load actual outcomes: {e}")
        return {}


def _is_actual_edge(node_name: str, edge_label: str, actual: dict) -> bool:
    """Check whether an edge matches the actual historical outcome."""
    actual_action = actual.get(node_name)
    if actual_action is None:
        return False

    # Direct match (decision nodes)
    if edge_label == actual_action:
        return True

    # Vote node: fuzzy matching
    if node_name == "V":
        if "strike" in actual_action.lower():
            # Actual was a strike — match the strike edge
            if "Strike" in edge_label and "No" not in edge_label:
                return True
        elif "no" in actual_action.lower():
            if "No strike" in edge_label:
                return True
        return False

    # Review node: fuzzy matching
    if node_name == "R":
        if actual_action == "adverse" and "Adverse" in edge_label:
            return True
        if actual_action == "no_adverse" and "No adverse" in edge_label:
            return True
        return False

    # Pass-through nodes always match
    if node_name in ("M_agm", "M_rev") and edge_label == "pass-through":
        return True

    return False


# ── Tree serialization ────────────────────────────────────────────────

def viznode_to_dict(
    node: VizNode,
    results: dict,
    focal: str,
    actual_outcomes: dict,
    commentary: dict,
    scenario_context: Optional[str] = None,
    d1_action_context: Optional[str] = None,
    on_actual_path: bool = True,
) -> dict:
    """Recursively serialize VizNode tree to a nested dict for D3.js.

    Parameters
    ----------
    node : VizNode
    results : dict[str, SolveResult]  scenario_key -> SolveResult
    focal : str  focal actor name
    actual_outcomes : dict  from load_actual_outcomes()
    commentary : dict  "{node_id}__{label}" -> commentary text
    scenario_context : str  which scenario subtree we are in
    d1_action_context : str  which D1 action branch we are in
    on_actual_path : bool  whether this node is on the actual outcome path
        (True for root; propagated only when parent edge is actual)
    """
    d = {
        "id": node.id,
        "name": node.node_name,
        "type": node.node_type,
        "owner": node.owner,
        "eu": round(node.eu, 4),
        "nice_label": NICE_NODE.get(node.node_name, node.node_name),
        "colour": OWNER_COLOURS.get(node.owner, "#CCC"),
        "scenario": scenario_context,
        "utility_decomposition": {},
        "focal_utility_decomposition": {},
        "predictive_dist": {},
        "outcome_stats": {},
        "node_commentary": commentary.get(f"{node.id}__node", ""),
        "children": [],
    }

    # Enrich node with data from SolveResult
    _enrich_node_data(d, results, scenario_context, d1_action_context)

    for label, prob, child in node.children:
        # Track scenario and D1 action context as we descend
        sc = scenario_context
        d1 = d1_action_context
        if node.node_name == "D0_ceo":
            sc = {"CEO_resign": "ceo_resigned", "CEO_stay": "ceo_stayed"}.get(label, sc)
        if node.node_name == "D1":
            d1 = label

        # Edge is actual only if THIS node is on the actual path AND
        # the edge label matches the actual outcome for this node
        edge_is_actual = (
            on_actual_path and _is_actual_edge(node.node_name, label, actual_outcomes)
        )

        child_dict = viznode_to_dict(
            child, results, focal, actual_outcomes, commentary,
            scenario_context=sc, d1_action_context=d1,
            on_actual_path=edge_is_actual,
        )
        edge = {
            "label": label,
            "nice_label": NICE_EDGE.get(label, label),
            "prob": round(prob, 4),
            "is_actual": edge_is_actual,
            "child_eu": round(child.eu, 4),
            "commentary": commentary.get(f"{node.id}__{label}", ""),
            "child": child_dict,
        }
        d["children"].append(edge)

    return d


def _enrich_node_data(
    node_dict: dict,
    results: dict,
    scenario: Optional[str],
    d1_action: Optional[str],
) -> None:
    """Attach utility decomposition, predictive dists, outcome stats from SolveResult."""
    if not scenario or scenario not in results:
        return
    result = results[scenario]

    # Utility decomposition keyed by D1 action.
    # Opponent nodes show opponent utility (driving their choice) + focal utility (impact).
    # A2 (ASA) shows ASA + Board decompositions.
    # D4, D4_post_review (CEO) show CEO + Board decompositions.
    # All other nodes show focal actor components only.
    if d1_action:
        asa_decomp = getattr(result, "asa_utility_decomposition", {})
        ceo_decomp = getattr(result, "ceo_utility_decomposition", {})
        node_name = node_dict.get("name")

        if node_name == "A2" and d1_action in asa_decomp:
            node_dict["utility_decomposition"] = {
                k: round(v, 4) for k, v in asa_decomp[d1_action].items()
            }
            # Also attach focal (Board) decomposition for side-by-side display
            if d1_action in result.utility_decomposition:
                node_dict["focal_utility_decomposition"] = {
                    k: round(v, 4)
                    for k, v in result.utility_decomposition[d1_action].items()
                }
        elif node_name in ("D4", "D4_post_review") and d1_action in ceo_decomp:
            node_dict["utility_decomposition"] = {
                k: round(v, 4) for k, v in ceo_decomp[d1_action].items()
            }
            # Also attach focal (Board) decomposition for side-by-side display
            if d1_action in result.utility_decomposition:
                node_dict["focal_utility_decomposition"] = {
                    k: round(v, 4)
                    for k, v in result.utility_decomposition[d1_action].items()
                }
        elif d1_action in result.utility_decomposition:
            node_dict["utility_decomposition"] = {
                k: round(v, 4) for k, v in result.utility_decomposition[d1_action].items()
            }

    # Outcome stats keyed by D1 action
    if d1_action and d1_action in result.outcome_stats:
        stats = result.outcome_stats[d1_action]
        node_dict["outcome_stats"] = {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in stats.items()
        }

    # Predictive distributions keyed by D1 action
    if d1_action and d1_action in result.predictive_dists:
        node_dict["predictive_dist"] = {
            node_label: {a: round(p, 4) for a, p in dist.items()}
            for node_label, dist in result.predictive_dists[d1_action].items()
        }


# ── LLM commentary generation ────────────────────────────────────────

# Canonical node play-order — used to build temporal context for each prompt.
_NODE_SEQUENCE = [
    "D0_ceo", "D1", "A2", "V", "M_agm",
    "D4", "D_rev", "R", "M_rev",
    "D4_post_review", "D_rev_post_review", "Terminal",
]

def _load_api_key() -> Optional[str]:
    """Load OPENAI_API_KEY from .env file."""
    try:
        from dotenv import load_dotenv
        import os
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        key = os.environ.get("OPENAI_API_KEY", "")
        return key if key else None
    except ImportError:
        logger.warning("python-dotenv not installed; cannot load .env")
        return None


def _get_openai_client(api_key: Optional[str]):
    """Create an instructor-patched OpenAI client.

    Returns None if api_key is missing, libraries not installed, or key is invalid.
    """
    if not api_key:
        return None
    try:
        import instructor
        from openai import OpenAI
        raw_client = OpenAI(api_key=api_key)
        client = instructor.from_openai(raw_client)
        return client
    except ImportError:
        logger.warning("instructor/openai not installed; commentary disabled")
        return None
    except Exception as e:
        logger.warning(f"OpenAI client init failed: {e}; commentary disabled")
        return None


PLACEHOLDER_COMMENTARY = "Commentary not available without API key."

# gpt-4o-mini pricing (per 1M tokens, as of 2025)
_GPT4O_MINI_INPUT_COST_PER_M = 0.15   # $0.15 per 1M input tokens
_GPT4O_MINI_OUTPUT_COST_PER_M = 0.60  # $0.60 per 1M output tokens


def generate_commentary(
    tree_dict: dict,
    focal: str,
    checkpoint_id: str,
    api_key: Optional[str] = None,
) -> dict:
    """Generate LLM commentary for edges and nodes, embedded at build time.

    Uses up to 20 concurrent threads for API calls.

    Returns dict: "{node_id}__{label}" -> commentary text (for edges)
                  "{node_id}__node" -> commentary text (for nodes)

    If API key is missing or invalid, all values are placeholder strings.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    client = _get_openai_client(api_key)

    # Collect all edges and nodes that need commentary
    items = []
    _collect_commentary_items(tree_dict, items, focal=focal)

    logger.info(f"Commentary: {len(items)} items to annotate")

    if client is None:
        return {key: PLACEHOLDER_COMMENTARY for key, _, _ in items}

    # Import Pydantic model here (only needed when client is available)
    try:
        from pydantic import BaseModel, Field
    except ImportError:
        logger.warning("pydantic not installed; commentary disabled")
        return {key: PLACEHOLDER_COMMENTARY for key, _, _ in items}

    class GameTreeCommentary(BaseModel):
        explanation: str = Field(
            ..., description="1-2 sentence plain-English explanation")
        justification: str = Field(
            ..., description="Brief strategic insight about why this matters")

    system_prompt = _build_system_prompt(focal, checkpoint_id)

    # Thread-safe token tracking
    token_lock = threading.Lock()
    token_stats = {"input": 0, "output": 0, "calls": 0, "errors": 0}

    import re as _re
    import sys as _sys
    from tqdm import tqdm

    def _call_llm(key, item_type, context):
        """Single LLM call for one commentary item (up to 3 retries via instructor)."""
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                response_model=GameTreeCommentary,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context},
                ],
                max_retries=3,
            )
            # Extract token usage from the raw response
            usage = resp._raw_response.usage if hasattr(resp, '_raw_response') and resp._raw_response else None
            with token_lock:
                token_stats["calls"] += 1
                if usage:
                    token_stats["input"] += usage.prompt_tokens
                    token_stats["output"] += usage.completion_tokens
            return key, f"{resp.explanation}\n\n{resp.justification}"
        except Exception as e:
            with token_lock:
                token_stats["errors"] += 1
            err_str = str(e)
            if "field required" in err_str.lower() or "validation error" in err_str.lower():
                missing = _re.findall(r"(\w+)\n\s+Field required", err_str)
                field_str = ", ".join(missing) if missing else "unknown field"
                msg = f"Commentary schema error [{key}]: missing '{field_str}' after 3 retries"
            else:
                msg = f"Commentary failed [{key}]: {err_str.splitlines()[0][:100]}"
            tqdm.write(f"  WARNING: {msg}", file=_sys.stderr)
            return key, PLACEHOLDER_COMMENTARY

    # Run all calls concurrently with up to 20 threads
    commentary = {}
    max_workers = min(20, len(items))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_call_llm, key, item_type, context): key
            for key, item_type, context in items
        }
        with tqdm(total=len(futures), desc="LLM commentary", unit="item", smoothing=0) as pbar:
            for future in as_completed(futures):
                key, text = future.result()
                commentary[key] = text
                pbar.update(1)

    # Report token usage and cost
    input_cost = token_stats["input"] * _GPT4O_MINI_INPUT_COST_PER_M / 1_000_000
    output_cost = token_stats["output"] * _GPT4O_MINI_OUTPUT_COST_PER_M / 1_000_000
    total_cost = input_cost + output_cost
    logger.info(
        f"Commentary complete: {token_stats['calls']} calls, "
        f"{token_stats['errors']} errors, "
        f"{token_stats['input']:,} input tokens, "
        f"{token_stats['output']:,} output tokens, "
        f"total cost: ${total_cost:.4f}"
    )

    return commentary


def _collect_commentary_items(
    node_dict: dict,
    items: list,
    resolved: list | None = None,
    focal: str = "Board",
) -> None:
    """Walk tree dict and collect (key, type, context_prompt) tuples.

    Parameters
    ----------
    node_dict : dict  serialised VizNode
    items : list  accumulator of (key, type, prompt) tuples
    resolved : list[str]  "Node(action)" strings for ancestors already played,
        used to enforce temporal realism in each prompt.
    """
    if resolved is None:
        resolved = []

    node_id = node_dict["id"]
    node_name = node_dict["name"]
    nice = node_dict["nice_label"]

    # Build temporal-context string for this node
    resolved_str = ", ".join(resolved) if resolved else "none"
    try:
        idx = _NODE_SEQUENCE.index(node_name)
        remaining = _NODE_SEQUENCE[idx + 1:]
    except ValueError:
        remaining = []
    remaining_str = " → ".join(remaining) if remaining else "none (terminal)"

    temporal_ctx = (
        f"Temporal position: nodes already resolved = [{resolved_str}]; "
        f"nodes not yet played = [{remaining_str}]. "
        f"Do NOT reference outcomes from nodes that have not yet been played."
    )

    # Build utility decomposition summary for the prompt
    decomp = node_dict.get("utility_decomposition", {})
    decomp_str = ""
    if decomp:
        parts = [f"{k.replace('_', ' ')}: {v:+.3f}" for k, v in decomp.items()]
        decomp_str = (
            f"Utility components (from {node_dict['owner']}'s perspective): "
            f"{'; '.join(parts)}. "
        )

    # Build key outcome stats summary
    stats = node_dict.get("outcome_stats", {})
    stats_str = ""
    if stats:
        stat_parts = []
        for k in ("Pr_strike", "Pr_CEO_removed", "Pr_overwhelming", "Pr_review_adverse",
                  "mean_vote_percent"):
            if k in stats:
                v = stats[k]
                stat_parts.append(f"{k.replace('_', ' ')}={v:.1%}"
                                   if isinstance(v, float) else f"{k}={v}")
        stats_str = f"Outcome statistics: {', '.join(stat_parts)}. "

    # Scenario context: tell LLM what has already happened at D0_ceo
    scenario = node_dict.get("scenario")
    if scenario == "ceo_resigned":
        scenario_str = (
            "SCENARIO: The CEO has ALREADY RESIGNED before this node. "
            "CEO removal is therefore NOT a relevant concern at this point — "
            "the CEO is gone. Do not mention CEO removal as a future risk."
        )
    elif scenario == "ceo_stayed":
        scenario_str = (
            "SCENARIO: The CEO has chosen to STAY (not resign). "
            "CEO removal remains a live strategic concern."
        )
    else:
        scenario_str = ""

    # Perspective framing: opponent nodes must be described from the owner's POV
    owner = node_dict["owner"]
    if owner not in (focal, "Nature"):
        perspective_str = (
            f"CRITICAL: This node is owned by {owner}, NOT the focal actor ({focal}). "
            f"Write the commentary entirely from {owner}'s perspective — "
            f"explain {owner}'s own utility function and strategic incentives, "
            f"NOT how this affects {focal}."
        )
    else:
        perspective_str = f"Write from the focal actor ({focal})'s perspective."

    # For focal decision nodes, identify the optimal action to ground the LLM
    optimal_action_str = ""
    if node_dict.get("type") == "decision" and owner == focal:
        children = node_dict.get("children", [])
        if children:
            optimal_edge = max(children, key=lambda e: e["child_eu"])
            opt_label = optimal_edge["nice_label"].replace("\n", " ")
            optimal_action_str = (
                f"OPTIMAL ACTION CHOSEN: '{opt_label}' "
                f"(leads to EU {optimal_edge['child_eu']:+.4f}). "
                f"Your commentary MUST explain why THIS specific action — '{opt_label}' — "
                f"maximises {focal}'s expected utility. Do NOT describe other actions as if chosen. "
            )

    # Node commentary
    items.append((
        f"{node_id}__node",
        "node",
        f"Describe the game tree node '{node_name}' ({nice}). "
        f"Owner: {owner}. Type: {node_dict.get('type', '')}. "
        f"EU: {node_dict['eu']:+.4f}. "
        f"{scenario_str} "
        f"{optimal_action_str}"
        f"{decomp_str}{stats_str}"
        f"{perspective_str} "
        f"{temporal_ctx}",
    ))

    # Edge commentary
    for edge in node_dict.get("children", []):
        label = edge["label"]
        nice_label = edge["nice_label"].replace("\n", " ")
        # For opponent nodes, frame edge from the owner's perspective
        if owner not in (focal, "Nature"):
            edge_perspective = (
                f"This action is chosen by {owner} (an opponent of {focal}). "
                f"Explain why {owner} would choose this based on {owner}'s own utility "
                f"and incentives — do NOT frame it from {focal}'s perspective."
            )
        else:
            edge_perspective = f"Explain from {focal}'s perspective."
        items.append((
            f"{node_id}__{label}",
            "edge",
            f"At node '{node_name}' ({owner}), the action/outcome "
            f"'{nice_label}' has probability {edge['prob']:.1%} and "
            f"leads to child EU {edge['child_eu']:+.4f}. "
            f"{scenario_str} "
            f"{edge_perspective} "
            f"Justify the probability and utility in 1-2 sentences. "
            f"{temporal_ctx}",
        ))
        # Recurse into child with updated resolved path
        child_resolved = resolved + [f"{node_name}({nice_label})"]
        _collect_commentary_items(edge["child"], items, child_resolved, focal=focal)


def _build_system_prompt(focal: str, checkpoint_id: str) -> str:
    return (
        "You are an expert in adversarial risk analysis and Australian corporate governance. "
        "You are annotating a game tree for an AGM governance crisis involving a major airline.\n\n"
        "The game involves three players with DISTINCT utility functions:\n"
        "- Board: minimises shareholder opposition (vote penalty, spill risk) and disruption "
        "(CEO loss cost, review costs). Penalised by high votes, CEO removal, and adverse reviews.\n"
        "- ASA (Australian Shareholders Association): maximises accountability. GAINS from high "
        "opposition votes (vote_reward), CEO removal (ceo_removal_reward), and adverse review "
        "findings (review_car_impact is NEGATIVE of Board's — adverse = vindication for ASA). "
        "Pays mobilisation_cost only if it recommends a strike.\n"
        "- CEO: maximises reference-dependent CRRA wealth utility minus career/reputational "
        "penalties. Strongly prefers to avoid sacking (D_sacked penalty is large).\n\n"
        f"Focal actor for this run: {focal}. Belief checkpoint: {checkpoint_id}.\n\n"
        "CRITICAL PERSPECTIVE RULE: Each prompt will tell you which actor OWNS the node. "
        "You MUST write commentary from that owner's perspective — their utility function, "
        "their incentives, their trade-offs. Do NOT describe an opponent's node from the "
        f"focal actor ({focal})'s perspective. The prompt will make the required perspective "
        "explicit with a CRITICAL label.\n\n"
        "CRITICAL TEMPORAL CONSTRAINT: The game tree is sequential. Each annotation must be "
        "written from the perspective of an actor AT THAT POINT IN THE GAME — they cannot see "
        "downstream outcomes that have not yet occurred. Each prompt tells you which nodes have "
        "already resolved and which are still in the future. You must NEVER cite actual vote "
        "percentages, actual CEO decisions, or any other real historical outcome for a node that "
        "is downstream of the node being annotated. You may reference EXPECTED or PROBABILISTIC "
        "outcomes of future nodes (e.g. 'expected strike probability ~X%') but never their "
        "realised values.\n\n"
        "Provide concise commentary explaining the strategic and probabilistic reasoning "
        "behind each action's probability and expected utility. Focus on game-theoretic "
        "incentives, belief parameters, and payoff structure. "
        "Keep explanations to 1-2 sentences and justifications to 1-2 sentences."
    )


# ── HTML template ─────────────────────────────────────────────────────

def render_html(
    tree_json: str,
    focal: str,
    checkpoint_id: str,
    output_path: Path,
) -> None:
    """Write a self-contained interactive HTML file."""
    html = _HTML_TEMPLATE.replace("__TREE_DATA__", tree_json)
    html = html.replace("__FOCAL__", focal)
    html = html.replace("__CHECKPOINT__", checkpoint_id)
    html = html.replace("__TITLE__", f"ARA Game Tree \u2014 {focal}, {checkpoint_id}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"Saved interactive HTML: {output_path}")


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; background: #FAFAFA; overflow: hidden; }

/* ── Controls bar ───────────────────────────────────── */
#controls {
  position: fixed; top: 0; left: 0; right: 0; z-index: 300;
  display: flex; align-items: center; gap: 10px;
  padding: 8px 16px; background: #fff; border-bottom: 1px solid #ddd;
  box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
#controls button {
  padding: 5px 14px; border: 1px solid #bbb; border-radius: 4px;
  background: #fff; cursor: pointer; font-size: 12px; font-family: inherit;
  transition: background 0.15s;
}
#controls button:hover { background: #f0f0f0; }
#controls button.active { background: #4A90D9; color: #fff; border-color: #4A90D9; }
#info-bar { margin-left: auto; color: #777; font-size: 12px; }

/* ── Legend ──────────────────────────────────────────── */
#legend {
  position: fixed; bottom: 12px; left: 16px; z-index: 300;
  background: #fff; border: 1px solid #ddd; border-radius: 6px;
  padding: 10px 14px; font-size: 11px; line-height: 1.7;
  box-shadow: 0 1px 6px rgba(0,0,0,0.08);
}
.legend-item { display: flex; align-items: center; gap: 6px; }
.legend-swatch {
  width: 14px; height: 14px; border-radius: 3px; flex-shrink: 0;
}
.legend-line {
  width: 24px; height: 0; border-top: 3px solid; flex-shrink: 0;
}

/* ── SVG tree ───────────────────────────────────────── */
#tree-container { position: absolute; top: 40px; left: 0; right: 0; bottom: 0; }
svg { width: 100%; height: 100%; }

.node-group { cursor: pointer; }
.node-shape { stroke-width: 1.5px; }
.node-label { fill: #fff; font-size: 10px; text-anchor: middle; dominant-baseline: central; pointer-events: none; }
.node-eu-label { fill: #fff; font-size: 8px; text-anchor: middle; font-style: italic; pointer-events: none; }
.expand-icon {
  font-size: 11px; font-weight: 600; cursor: pointer;
  text-anchor: middle; dominant-baseline: central;
  pointer-events: none;
}
.expand-badge {
  font-size: 10px; font-weight: 600; fill: #fff;
  text-anchor: middle; dominant-baseline: central;
  pointer-events: none;
}
.expand-badge-bg {
  pointer-events: none;
}
.info-icon {
  fill: #fff; stroke: #fff; stroke-width: 0.5; font-size: 11px; cursor: pointer;
  text-anchor: middle; dominant-baseline: central;
}

.link { fill: none; stroke: #999; stroke-width: 1.5; stroke-opacity: 0.6; }
.link.actual { stroke: #E85D5D; stroke-width: 3; stroke-opacity: 1; }
.link-label { font-size: 9px; fill: #555; pointer-events: none; }
.link-hover-target { fill: none; stroke: transparent; stroke-width: 14; cursor: pointer; }

/* ── Tooltip (edge mouseover) ───────────────────────── */
#tooltip {
  position: absolute; padding: 10px 14px;
  background: rgba(20,20,30,0.92); color: #eee; border-radius: 6px;
  font-size: 11px; line-height: 1.6; max-width: 380px;
  pointer-events: none; z-index: 400; display: none;
  box-shadow: 0 3px 12px rgba(0,0,0,0.3);
}
#tooltip .tt-title { font-weight: 600; font-size: 12px; color: #fff; margin-bottom: 4px; }
#tooltip .tt-sep { border-top: 1px solid rgba(255,255,255,0.15); margin: 6px 0; }
#tooltip .tt-row { display: flex; justify-content: space-between; gap: 12px; }
#tooltip .tt-label { color: #aaa; }
#tooltip .tt-val { font-weight: 500; }
#tooltip .tt-commentary { margin-top: 6px; color: #ccc; font-style: italic; white-space: pre-line; }

/* ── Node popup (mouseover) ─────────────────────────── */
#overlay {
  position: fixed; inset: 0; background: rgba(0,0,0,0.25); z-index: 450; display: none;
}
#node-popup {
  position: fixed;
  background: #fff; border-radius: 8px; padding: 24px 28px;
  box-shadow: 0 6px 30px rgba(0,0,0,0.22); max-width: 540px; width: 90%;
  max-height: 82vh; overflow-y: auto; z-index: 500; display: none;
  font-size: 13px; line-height: 1.6;
}
#node-popup .popup-close {
  position: absolute; top: 10px; right: 14px; cursor: pointer;
  font-size: 20px; color: #999; font-weight: 300; line-height: 1;
}
#node-popup .popup-close:hover { color: #333; }
#node-popup h3 { margin-bottom: 10px; font-size: 16px; }
#node-popup .badge {
  display: inline-block; padding: 2px 8px; border-radius: 3px;
  color: #fff; font-size: 11px; margin-right: 6px;
}
#node-popup table { width: 100%; border-collapse: collapse; margin: 10px 0; }
#node-popup th { text-align: left; font-size: 11px; color: #888; padding: 4px 8px; border-bottom: 2px solid #eee; }
#node-popup td { padding: 4px 8px; border-bottom: 1px solid #f0f0f0; font-size: 12px; }
#node-popup td:last-child { text-align: right; font-variant-numeric: tabular-nums; }
#node-popup .section-title { font-weight: 600; font-size: 13px; margin-top: 14px; margin-bottom: 4px; color: #444; }
#node-popup .commentary-text { color: #666; font-style: italic; margin-top: 8px; white-space: pre-line; }
</style>
</head>
<body>

<!-- Controls -->
<div id="controls">
  <button id="btn-prob" class="active" onclick="setView('prob')">Probability View</button>
  <button id="btn-eu" onclick="setView('eu')">Expected Utility View</button>
  <button id="btn-actual" class="active" onclick="toggleActual()">Actual Path</button>
  <button onclick="expandAll()">Expand All</button>
  <button onclick="collapseAll()">Collapse All</button>
  <button onclick="fitToScreen()">Fit to Screen</button>
  <span id="info-bar">Focal: __FOCAL__ &nbsp;|&nbsp; Checkpoint: __CHECKPOINT__</span>
</div>

<!-- Legend -->
<div id="legend">
  <div class="legend-item"><span class="legend-swatch" style="background:#4A90D9"></span> Board (decision)</div>
  <div class="legend-item"><span class="legend-swatch" style="background:#50C878"></span> ASA (decision)</div>
  <div class="legend-item"><span class="legend-swatch" style="background:#E85D5D"></span> CEO (decision)</div>
  <div class="legend-item"><span class="legend-swatch" style="background:#AAAAAA; border-radius:50%"></span> Nature (chance)</div>
  <div class="legend-item"><span class="legend-swatch" style="background:#888; transform:rotate(45deg)"></span> Terminal</div>
  <div class="legend-item"><span class="legend-line" style="border-color:#E85D5D"></span> Actual outcome</div>
  <div class="legend-item"><span class="legend-line" style="border-color:#999"></span> Model edge</div>
</div>

<!-- Tooltip -->
<div id="tooltip"></div>

<!-- Node popup overlay -->
<div id="overlay" onclick="closePopup()"></div>
<div id="node-popup"><span class="popup-close" onclick="closePopup()">&times;</span><div id="popup-content"></div></div>

<!-- Tree container -->
<div id="tree-container"><svg id="tree-svg"></svg></div>

<script>
// ── Data ──
const TREE_DATA = __TREE_DATA__;
const FOCAL_ACTOR = "__FOCAL__";
let VIEW_MODE = "prob";
let SHOW_ACTUAL = true;

// ── Owner colours ──
const COLOURS = {"Board":"#4A90D9","ASA":"#50C878","CEO":"#E85D5D","Nature":"#AAAAAA"};

// ── SVG setup ──
const container = document.getElementById("tree-container");
const svg = d3.select("#tree-svg");
const width = container.clientWidth;
const height = container.clientHeight;

const g = svg.append("g");

// Zoom
const zoom = d3.zoom()
    .scaleExtent([0.15, 3])
    .on("zoom", (e) => g.attr("transform", e.transform));
svg.call(zoom);

// Initial transform
const margin = {top: 40, left: 140};
svg.call(zoom.transform, d3.zoomIdentity.translate(margin.left, height / 2).scale(0.85));

// ── Convert nested data to d3.hierarchy ──
function childrenAccessor(d) {
    if (!d.children || d.children.length === 0) return null;
    return d.children.map(e => {
        const child = {...e.child};
        child._edge = {
            label: e.label,
            nice_label: e.nice_label,
            prob: e.prob,
            is_actual: e.is_actual,
            child_eu: e.child_eu,
            commentary: e.commentary,
        };
        return child;
    });
}

const root = d3.hierarchy(TREE_DATA, childrenAccessor);

// Store original children for expand/collapse (must traverse ALL nodes)
function walkAll(d, fn) {
    fn(d);
    const kids = d.children || d._collapsed;
    if (kids) kids.forEach(c => walkAll(c, fn));
}
walkAll(root, d => { d._allChildren = d.children; });

// Collapse branching nodes deeper than level 3 initially.
// Skip pass-through nodes (single child) — they just add clicks.
walkAll(root, d => {
    if (d.depth >= 3 && d.children && d.children.length > 1) {
        d._collapsed = d.children;
        d.children = null;
    }
});

// Auto-expand nodes on the actual outcome path (red lines always visible)
function expandActualPath(d) {
    const kids = d.children || d._collapsed;
    if (!kids) return;
    for (const child of kids) {
        if (child.data._edge && child.data._edge.is_actual) {
            if (d._collapsed) { d.children = d._collapsed; d._collapsed = null; }
            expandActualPath(child);
        }
    }
}
if (SHOW_ACTUAL) expandActualPath(root);

// ── Tree layout ──
const nodeW = 54, nodeH = 32;
const treeLayout = d3.tree().nodeSize([nodeH + 14, 260]);

function update(source) {
    const duration = 400;
    treeLayout(root);

    // ── Links ──
    const links = root.links();
    const linkSel = g.selectAll(".link-group").data(links, d => d.target.data.id);

    const linkEnter = linkSel.enter().append("g").attr("class", "link-group");

    // Visible link path
    linkEnter.append("path")
        .attr("class", d => "link" + (SHOW_ACTUAL && d.target.data._edge && d.target.data._edge.is_actual ? " actual" : ""))
        .attr("d", d => linkPath(source, source));

    // Invisible wider path for hover target
    linkEnter.append("path")
        .attr("class", "link-hover-target")
        .attr("d", d => linkPath(source, source))
        .on("mouseover", (ev, d) => showTooltip(ev, d))
        .on("mousemove", (ev) => moveTooltip(ev))
        .on("mouseout", () => hideTooltip());

    // Link labels
    linkEnter.append("text")
        .attr("class", "link-label")
        .attr("dy", -4);

    const linkMerge = linkEnter.merge(linkSel);

    linkMerge.select(".link")
        .transition().duration(duration)
        .attr("d", d => linkPath(d.source, d.target))
        .attr("class", d => "link" + (SHOW_ACTUAL && d.target.data._edge && d.target.data._edge.is_actual ? " actual" : ""))
        .attr("stroke-width", d => {
            const edge = d.target.data._edge;
            if (SHOW_ACTUAL && edge && edge.is_actual) return 3;
            if (edge && edge.prob < 0.01) return 0.5;
            return Math.max(1, (edge ? edge.prob : 0.5) * 4);
        })
        .attr("stroke-opacity", d => {
            const edge = d.target.data._edge;
            if (SHOW_ACTUAL && edge && edge.is_actual) return 1;
            if (edge && edge.prob < 0.01) return 0.25;
            return 0.6;
        });

    linkMerge.select(".link-hover-target")
        .transition().duration(duration)
        .attr("d", d => linkPath(d.source, d.target));

    linkMerge.select(".link-label")
        .transition().duration(duration)
        .attr("x", d => (d.source.y + d.target.y) / 2)
        .attr("y", d => (d.source.x + d.target.x) / 2)
        .text(d => edgeLabelText(d));

    linkSel.exit()
        .transition().duration(duration).style("opacity", 0).remove();

    // ── Nodes ──
    const nodes = root.descendants();
    const nodeSel = g.selectAll(".node-group").data(nodes, d => d.data.id);

    const nodeEnter = nodeSel.enter().append("g")
        .attr("class", "node-group")
        .attr("transform", `translate(${source.y0 || 0},${source.x0 || 0})`);

    // Draw node shapes — click to expand/collapse, mouseover for info popup
    nodeEnter.each(function(d) {
        const el = d3.select(this);
        const col = COLOURS[d.data.owner] || "#888";
        let shape;
        if (d.data.type === "terminal") {
            // Diamond
            shape = el.append("polygon")
                .attr("class", "node-shape")
                .attr("points", `0,${-nodeH/2} ${nodeW/2},0 0,${nodeH/2} ${-nodeW/2},0`)
                .attr("fill", col).attr("stroke", d3.color(col).darker(0.4));
        } else if (d.data.type === "chance") {
            // Ellipse
            shape = el.append("ellipse")
                .attr("class", "node-shape")
                .attr("rx", nodeW/2).attr("ry", nodeH/2)
                .attr("fill", col).attr("stroke", d3.color(col).darker(0.4));
        } else {
            // Rectangle (decision)
            shape = el.append("rect")
                .attr("class", "node-shape")
                .attr("x", -nodeW/2).attr("y", -nodeH/2)
                .attr("width", nodeW).attr("height", nodeH)
                .attr("rx", 4).attr("ry", 4)
                .attr("fill", col).attr("stroke", d3.color(col).darker(0.4));
        }
        // Click on node shape: expand/collapse children
        shape.on("click", (ev, d2) => { ev.stopPropagation(); toggle(d); });
        // Mouseover on node shape: show info popup; mouseout: schedule hide
        shape.on("mouseover", (ev) => { showNodePopup(d); })
             .on("mouseout", () => { scheduleHidePopup(); });
    });

    // Node label (name) — pointer-events none so clicks/hovers pass to shape
    nodeEnter.append("text")
        .attr("class", "node-label")
        .attr("dy", d => (VIEW_MODE === "eu" && d.data.type !== "terminal") ? -4 : 0);

    // EU sublabel
    nodeEnter.append("text")
        .attr("class", "node-eu-label")
        .attr("dy", 8);

    // Expand badge background (orange circle when collapsed)
    nodeEnter.append("circle")
        .attr("class", "expand-badge-bg")
        .attr("cx", nodeW/2 + 2)
        .attr("cy", 0)
        .attr("r", 9);

    // Expand badge text (child count when collapsed)
    nodeEnter.append("text")
        .attr("class", "expand-badge")
        .attr("x", nodeW/2 + 2)
        .attr("dy", 1);

    const nodeMerge = nodeEnter.merge(nodeSel);

    nodeMerge.transition().duration(duration)
        .attr("transform", d => `translate(${d.y},${d.x})`);

    nodeMerge.select(".node-label")
        .text(d => {
            const n = d.data.name;
            return n === "Terminal" ? "T" : n;
        })
        .attr("dy", d => (VIEW_MODE === "eu" && d.data.type !== "terminal") ? -4 : 0);

    nodeMerge.select(".node-eu-label")
        .text(d => VIEW_MODE === "eu" ? `EU=${d.data.eu >= 0 ? "+" : ""}${d.data.eu.toFixed(2)}` : "")
        .attr("visibility", d => VIEW_MODE === "eu" ? "visible" : "hidden");

    // Count hidden descendants for collapsed badge
    function countDescendants(node) {
        let n = 0;
        const kids = node._collapsed || node.children;
        if (kids) kids.forEach(c => { n += 1 + countDescendants(c); });
        return n;
    }

    nodeMerge.select(".expand-badge-bg")
        .attr("fill", d => (d._collapsed && d._collapsed.length > 0) ? "#E8853D" : "none")
        .attr("stroke", d => (d._collapsed && d._collapsed.length > 0) ? "#C96E2A" : "none")
        .attr("stroke-width", 1.5);

    nodeMerge.select(".expand-badge")
        .text(d => {
            if (!d._collapsed || d._collapsed.length === 0) return "";
            const n = countDescendants(d);
            return "+" + n;
        });

    nodeSel.exit()
        .transition().duration(duration)
        .attr("transform", `translate(${source.y},${source.x})`)
        .style("opacity", 0).remove();

    // Stash positions for transitions
    nodes.forEach(d => { d.x0 = d.x; d.y0 = d.y; });
}

// ── Link path generator ──
function linkPath(s, t) {
    return `M${s.y},${s.x}C${(s.y+t.y)/2},${s.x} ${(s.y+t.y)/2},${t.x} ${t.y},${t.x}`;
}

// ── Edge label text ──
function edgeLabelText(d) {
    const edge = d.target.data._edge;
    if (!edge) return "";
    const nice = edge.nice_label.replace(/\n/g, " ");
    if (VIEW_MODE === "prob") {
        return `${nice}  p=${(edge.prob * 100).toFixed(1)}%`;
    } else {
        const eu = edge.child_eu;
        return `${nice}  EU=${eu >= 0 ? "+" : ""}${eu.toFixed(3)}`;
    }
}

// ── Toggle expand/collapse ──
function toggle(d) {
    if (d.children) {
        d._collapsed = d.children;
        d.children = null;
    } else if (d._collapsed) {
        d.children = d._collapsed;
        d._collapsed = null;
    }
    update(d);
}

function expandAll() {
    walkAll(root, d => {
        if (d._collapsed) { d.children = d._collapsed; d._collapsed = null; }
    });
    update(root);
}

function collapseAll() {
    walkAll(root, d => {
        if (d.depth >= 1 && d.children && d.children.length > 1) {
            d._collapsed = d.children; d.children = null;
        }
    });
    update(root);
}

function fitToScreen() {
    // Compute bounding box of all visible nodes
    const nodes = root.descendants();
    if (nodes.length === 0) return;
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    nodes.forEach(d => {
        // In the LR layout: d.y = horizontal position, d.x = vertical position
        minX = Math.min(minX, d.x - nodeH);
        maxX = Math.max(maxX, d.x + nodeH);
        minY = Math.min(minY, d.y - nodeW);
        maxY = Math.max(maxY, d.y + nodeW + 80); // extra space for edge labels
    });
    const treeW = maxY - minY;
    const treeH = maxX - minX;
    const pad = 60;
    const availW = container.clientWidth - pad * 2;
    const availH = container.clientHeight - pad * 2;
    const scale = Math.min(availW / treeW, availH / treeH, 1.5);
    const tx = pad - minY * scale + (availW - treeW * scale) / 2;
    const ty = pad - minX * scale + (availH - treeH * scale) / 2;
    svg.transition().duration(500)
        .call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
}

// ── View toggle ──
function setView(mode) {
    VIEW_MODE = mode;
    document.getElementById("btn-prob").classList.toggle("active", mode === "prob");
    document.getElementById("btn-eu").classList.toggle("active", mode === "eu");
    update(root);
}

// ── Actual path toggle ──
function toggleActual() {
    SHOW_ACTUAL = !SHOW_ACTUAL;
    document.getElementById("btn-actual").classList.toggle("active", SHOW_ACTUAL);
    update(root);
}

// ── Tooltip (edge mouseover) ──
const tooltip = document.getElementById("tooltip");

function showTooltip(ev, d) {
    const edge = d.target.data._edge;
    if (!edge) return;
    const nd = d.source.data;

    let html = `<div class="tt-title">${edge.nice_label.replace(/\n/g, " ")}</div>`;
    html += `<div class="tt-row"><span class="tt-label">Probability:</span><span class="tt-val">${(edge.prob*100).toFixed(1)}%</span></div>`;
    html += `<div class="tt-row"><span class="tt-label">Expected Utility:</span><span class="tt-val">${edge.child_eu >= 0 ? "+" : ""}${edge.child_eu.toFixed(4)}</span></div>`;

    // Utility decomposition from the source node (if D1 root)
    const decomp = nd.utility_decomposition;
    if (decomp && Object.keys(decomp).length > 0) {
        html += `<div class="tt-sep"></div>`;
        html += `<div class="tt-label" style="margin-bottom:2px">Utility Components:</div>`;
        for (const [k, v] of Object.entries(decomp)) {
            const name = k.replace(/_/g, " ");
            html += `<div class="tt-row"><span class="tt-label">${name}</span><span class="tt-val">${v >= 0 ? "+" : ""}${v.toFixed(4)}</span></div>`;
        }
    }

    // Commentary
    if (edge.commentary) {
        html += `<div class="tt-sep"></div>`;
        html += `<div class="tt-commentary">${edge.commentary}</div>`;
    }

    tooltip.innerHTML = html;
    tooltip.style.display = "block";
    moveTooltip(ev);
}

function moveTooltip(ev) {
    const x = ev.pageX + 14;
    const y = ev.pageY - 10;
    tooltip.style.left = x + "px";
    tooltip.style.top = y + "px";
}

function hideTooltip() {
    tooltip.style.display = "none";
}

// ── Node popup (mouseover) ──
let _popupHideTimer = null;

function showNodePopup(d) {
    // Cancel any pending hide
    if (_popupHideTimer) { clearTimeout(_popupHideTimer); _popupHideTimer = null; }

    const nd = d.data;
    const el = document.getElementById("popup-content");
    let html = "";

    // Header
    const col = COLOURS[nd.owner] || "#888";
    html += `<h3>${nd.nice_label.replace(/\n/g, " &mdash; ")}</h3>`;
    html += `<span class="badge" style="background:${col}">${nd.owner}</span>`;
    html += `<span class="badge" style="background:#888">${nd.type}</span>`;
    html += `<span style="font-size:12px; color:#666; margin-left:8px">EU = ${nd.eu >= 0 ? "+" : ""}${nd.eu.toFixed(4)}</span>`;

    // Edge probabilities (children)
    if (nd.children && nd.children.length > 0) {
        html += `<div class="section-title">Actions / Splits</div>`;
        html += `<table><tr><th>Action</th><th>Probability</th><th>EU</th></tr>`;
        for (const e of nd.children) {
            const actual = (SHOW_ACTUAL && e.is_actual) ? ' style="background:#FFF0F0; font-weight:600"' : '';
            html += `<tr${actual}>`;
            html += `<td>${e.nice_label.replace(/\n/g, " ")}${SHOW_ACTUAL && e.is_actual ? " \u2B50" : ""}</td>`;
            html += `<td>${(e.prob*100).toFixed(1)}%</td>`;
            html += `<td>${e.child_eu >= 0 ? "+" : ""}${e.child_eu.toFixed(4)}</td>`;
            html += `</tr>`;
        }
        html += `</table>`;
    }

    // Utility decomposition (owner) + optional focal decomposition
    const decomp = nd.utility_decomposition;
    const focalDecomp = nd.focal_utility_decomposition;
    const hasFocalDecomp = focalDecomp && Object.keys(focalDecomp).length > 0;
    if ((decomp && Object.keys(decomp).length > 0) || hasFocalDecomp) {
        html += `<div class="section-title">Utility Decomposition</div>`;
        if (decomp && Object.keys(decomp).length > 0) {
            if (hasFocalDecomp) {
                html += `<div style="font-size:11px;color:#555;margin:3px 0 2px;font-weight:600">${nd.owner}</div>`;
            }
            html += `<table><tr><th>Component</th><th>Value</th></tr>`;
            for (const [k, v] of Object.entries(decomp)) {
                html += `<tr><td>${k.replace(/_/g, " ")}</td><td>${v >= 0 ? "+" : ""}${v.toFixed(4)}</td></tr>`;
            }
            html += `</table>`;
        }
        if (hasFocalDecomp) {
            html += `<div style="font-size:11px;color:#555;margin:6px 0 2px;font-weight:600">${FOCAL_ACTOR} (focal)</div>`;
            html += `<table><tr><th>Component</th><th>Value</th></tr>`;
            for (const [k, v] of Object.entries(focalDecomp)) {
                html += `<tr><td>${k.replace(/_/g, " ")}</td><td>${v >= 0 ? "+" : ""}${v.toFixed(4)}</td></tr>`;
            }
            html += `</table>`;
        }
    }

    // Outcome stats
    const stats = nd.outcome_stats;
    if (stats && Object.keys(stats).length > 0) {
        html += `<div class="section-title">Outcome Statistics</div>`;
        html += `<table><tr><th>Metric</th><th>Value</th></tr>`;
        for (const [k, v] of Object.entries(stats)) {
            const label = k.replace(/_/g, " ");
            const val = typeof v === "number"
                ? (k.startsWith("Pr_") || k.includes("percent") ? (v*100).toFixed(1) + "%" : v.toFixed(4))
                : v;
            html += `<tr><td>${label}</td><td>${val}</td></tr>`;
        }
        html += `</table>`;
    }

    // Predictive distributions
    const pred = nd.predictive_dist;
    if (pred && Object.keys(pred).length > 0) {
        html += `<div class="section-title">Opponent Predictive Distributions</div>`;
        for (const [node_label, dist] of Object.entries(pred)) {
            html += `<div style="font-size:12px; color:#555; margin-top:6px; font-weight:500">${node_label}</div>`;
            html += `<table><tr><th>Action</th><th>Pr(best response)</th></tr>`;
            for (const [a, p] of Object.entries(dist)) {
                html += `<tr><td>${a}</td><td>${(p*100).toFixed(1)}%</td></tr>`;
            }
            html += `</table>`;
        }
    }

    // Node commentary
    if (nd.node_commentary) {
        html += `<div class="section-title">Commentary</div>`;
        html += `<div class="commentary-text">${nd.node_commentary}</div>`;
    }

    el.innerHTML = html;

    // Position the popup near the node using its SVG coordinates
    const popup = document.getElementById("node-popup");
    const svgEl = document.getElementById("tree-svg");
    const pt = svgEl.createSVGPoint();
    pt.x = d.y + nodeW/2 + 20;
    pt.y = d.x;
    const ctm = g.node().getCTM();
    const screenPt = pt.matrixTransform(ctm);

    // Clamp to viewport
    const pw = 440, maxH = window.innerHeight * 0.7;
    let left = screenPt.x + 10;
    let top = screenPt.y - 40;
    if (left + pw > window.innerWidth) left = screenPt.x - pw - 30;
    if (top < 50) top = 50;
    if (top + maxH > window.innerHeight - 10) top = window.innerHeight - maxH - 10;

    popup.style.left = left + "px";
    popup.style.top = top + "px";
    popup.style.width = pw + "px";
    popup.style.maxHeight = maxH + "px";
    popup.style.display = "block";
    document.getElementById("overlay").style.display = "none";
}

function scheduleHidePopup() {
    _popupHideTimer = setTimeout(() => {
        document.getElementById("node-popup").style.display = "none";
    }, 200);
}

function closePopup() {
    if (_popupHideTimer) { clearTimeout(_popupHideTimer); _popupHideTimer = null; }
    document.getElementById("overlay").style.display = "none";
    document.getElementById("node-popup").style.display = "none";
}

// Keep popup open when mouse enters it
document.getElementById("node-popup").addEventListener("mouseenter", () => {
    if (_popupHideTimer) { clearTimeout(_popupHideTimer); _popupHideTimer = null; }
});
document.getElementById("node-popup").addEventListener("mouseleave", () => {
    scheduleHidePopup();
});

// Close popup on Escape
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closePopup(); });

// ── Initial render ──
update(root);
// Auto-fit after first layout so entire tree is visible
setTimeout(fitToScreen, 100);
</script>
</body>
</html>
"""


# ── Top-level entry point ─────────────────────────────────────────────

def render_interactive_tree(
    root: VizNode,
    results: dict,
    focal: str,
    checkpoint_id: str,
    actual_outcomes_path=None,
    output_dir=None,
    d0_probs=None,
) -> Path:
    """Generate a self-contained interactive HTML tree visualisation.

    Parameters
    ----------
    root : VizNode  unified tree from build_unified_tree()
    results : dict[str, SolveResult]  scenario -> SolveResult
    focal : str  focal actor ("Board" or "ASA")
    checkpoint_id : str  e.g. "C0"
    actual_outcomes_path : str or Path  path to actual_outcomes.json
    output_dir : Path  output directory (default: PROJECT_ROOT / "outputs")
    d0_probs : dict  D0_ceo probabilities (unused here, reserved)

    Returns
    -------
    Path  path to the generated HTML file
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "outputs"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load actual outcomes
    actual = load_actual_outcomes(actual_outcomes_path)
    if actual:
        logger.info(f"Loaded actual outcomes: {list(actual.keys())}")
    else:
        logger.info("No actual outcomes loaded; red lines disabled")

    # Load API key and generate commentary
    api_key = _load_api_key()
    logger.info("Generating LLM commentary..." if api_key else "No API key; commentary will use placeholders")

    # First pass: serialize tree without commentary (needed for commentary prompts)
    tree_dict_raw = viznode_to_dict(root, results, focal, actual, {})

    # Generate commentary
    commentary = generate_commentary(tree_dict_raw, focal, checkpoint_id, api_key)

    # Second pass: serialize tree WITH commentary
    tree_dict = viznode_to_dict(root, results, focal, actual, commentary)

    # Render HTML
    tree_json = json.dumps(tree_dict, indent=None)
    output_path = output_dir / f"tree_interactive_{focal}_{checkpoint_id}.html"
    render_html(tree_json, focal, checkpoint_id, output_path)

    return output_path
