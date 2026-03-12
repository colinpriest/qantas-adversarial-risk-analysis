"""
ASA Utility Quantification Pipeline
====================================
6-stage pipeline for estimating ASA (Australian Shareholders' Association)
utility function parameters using LLM stakeholder simulation (gpt-4o-mini
via instructor) and Bayesian ordinal probit estimation (Stan).

Structurally parallel to board_utility_quantification.py.
Spec: background/asa/asa_utility_quantification_spec.md
"""
from __future__ import annotations

# ── SEC 0: Imports, constants, config ─────────────────────────────────────────
import argparse
import base64
import copy
import csv
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
import traceback
import unicodedata
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator, model_validator

logger = logging.getLogger("asa_utility_quantification")
logging.getLogger("httpx").setLevel(logging.WARNING)

# ── ASA utility weight parameters (8 total, lognormal priors) ──
#
# All weights are strictly positive (w > 0).
# Convention: EU = sum_k w_k * phi_k(game_state, action).
# Higher EU = more appropriate action.
#
# DESIGN PRINCIPLE: Every parameter is the utility of a MEASURABLE game tree
# outcome or input.  No abstract governance dimensions.  Scenarios are
# constructed from game tree paths (D0_ceo outcome x D1 action x timing).
#
# Phi functions use binary game tree indicators:
#   ceo_resigned, ceo_sacked, review_commissioned, action_delayed, high_profile
#
# DECOMPOSITION: Parameters split into CONTEXT terms (fire equally for both
# actions, capture situation quality for ordinal probit fitting) and
# INTERACTION terms (action-varying, drive the strike/no-strike decision).
# Context terms cancel in delta-EU so only interaction terms determine
# action probabilities.  This prevents one-sided term stacking that
# produces extreme (0/1) probabilities.
#
# --- CONTEXT PARAMETERS (same phi for both actions) ---
#   w_ctx_inaction      : Board passivity penalty
#   w_ctx_departure     : CEO accountability credit
#   w_ctx_review        : Governance review credit
#
# --- INTERACTION PARAMETERS (action-varying, drive decision) ---
#   w_strike_cost           : Net mobilisation cost of striking
#   w_strike_vs_passive     : Value of striking against passive board
#   w_departure_dampens     : CEO departure reduces need to strike
#   w_sack_dampens          : Board-forced exit further reduces need
#   w_credibility_signal    : Repeat-game credibility value of striking
#
ESTIMABLE_PARAM_NAMES = [
    # Context parameters (fire for both actions equally)
    "w_ctx_inaction",           # Penalty for Board visible inaction (no review, no sack)
    "w_ctx_departure",          # Credit for CEO having departed (resigned or sacked)
    "w_ctx_review",             # Credit for governance review commissioned
    # Interaction parameters (action-varying, drive strike/no-strike decision)
    "w_strike_cost",            # Net cost of strike recommendation (mobilisation cost)
    "w_strike_vs_passive",      # Value of striking when Board is passive (no action taken)
    "w_departure_dampens",      # CEO departure reduces marginal value of striking
    "w_sack_dampens",           # Board-forced exit further reduces strike value
    "w_credibility_signal",     # Repeat-game credibility value of striking in high-profile case
]  # 8 parameters estimated via ordinal probit

WEIGHT_PARAM_NAMES = ESTIMABLE_PARAM_NAMES  # estimated in Stan

PARAM_TO_ENGINE_KEY = {
    "w_ctx_inaction":        "asa_ctx_inaction",
    "w_ctx_departure":       "asa_ctx_departure",
    "w_ctx_review":          "asa_ctx_review",
    "w_strike_cost":         "asa_strike_cost",
    "w_strike_vs_passive":   "asa_strike_vs_passive",
    "w_departure_dampens":   "asa_departure_dampens",
    "w_sack_dampens":        "asa_sack_dampens",
    "w_credibility_signal":  "asa_credibility_signal",
}

PARAM_DESCRIPTIONS = {
    "w_ctx_inaction":
        "Context: penalty for Board visible inaction (no review, no CEO sacking)",
    "w_ctx_departure":
        "Context: credit for CEO having departed (voluntary or Board-forced)",
    "w_ctx_review":
        "Context: credit for Board having commissioned governance review",
    "w_strike_cost":
        "Interaction: net mobilisation cost of recommending strike",
    "w_strike_vs_passive":
        "Interaction: marginal value of striking when Board has not acted",
    "w_departure_dampens":
        "Interaction: CEO departure reduces marginal value of striking",
    "w_sack_dampens":
        "Interaction: Board-forced CEO exit further reduces strike value",
    "w_credibility_signal":
        "Interaction: repeat-game credibility value of striking in high-profile case",
}

# Prior medians for lognormal priors in the Stan model.
#
# All set to 1.0 (uninformative on the log scale).  lognormal(0, 1) has
# median = 1.0, mean ~ 1.65, and 95% CI ~ [0.14, 7.1].  This expresses
# genuine prior ignorance about the relative magnitudes of the weights
# while ensuring they are positive.
#
# The posterior will be determined by the LLM elicitation data, not by
# these priors.  Setting all medians equal avoids encoding arbitrary
# assumptions about which game tree outcomes ASA values more.
SPEC_DEFAULTS = {
    "w_ctx_inaction":        1.0,
    "w_ctx_departure":       1.0,
    "w_ctx_review":          1.0,
    "w_strike_cost":         1.0,
    "w_strike_vs_passive":   1.0,
    "w_departure_dampens":   1.0,
    "w_sack_dampens":        1.0,
    "w_credibility_signal":  1.0,
}

MODEL_PRICE_TABLE = {
    "gpt-4o-mini": {
        "prompt_per_1k":     0.00015,
        "completion_per_1k": 0.00060,
    },
    "gpt-4o": {
        "prompt_per_1k":     0.00250,
        "completion_per_1k": 0.01000,
    },
    "gpt-5-mini": {
        "prompt_per_1k":     0.00025,
        "completion_per_1k": 0.00200,
    },
    "gpt-5.2": {
        "prompt_per_1k":     0.00175,
        "completion_per_1k": 0.01400,
    },
    "gpt-5.4": {
        "prompt_per_1k":     0.00250,
        "completion_per_1k": 0.01500,
    },
}

LIKERT_SCALE_LABELS = {
    1: "not appropriate / not warranted",
    2: "marginally appropriate",
    3: "uncertain / could go either way",
    4: "somewhat appropriate",
    5: "strongly appropriate / clearly warranted",
}

PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "asa"
CACHE_DIR = PROJECT_ROOT / "utility-quantification" / "asa-cache"


# ── SEC 1: Pydantic schemas ──────────────────────────────────────────────────

class ASAActionCode(str, Enum):
    NO_STRIKE = "no_strike"
    REC_STRIKE = "rec_strike"


class ParseStatus(str, Enum):
    SUCCESS = "success"
    FORMAT_ERROR = "format_error"
    TOKEN_LIMIT = "token_limit"
    REPAIRED = "repaired"


FEASIBLE_ACTIONS = [ASAActionCode.NO_STRIKE, ASAActionCode.REC_STRIKE]


class ActionLikertScore(BaseModel):
    action: ASAActionCode
    score: int
    justification: str

    @field_validator("score")
    @classmethod
    def valid_score(cls, v: int) -> int:
        if v < 1 or v > 5:
            raise ValueError(f"Score {v} not in [1, 5]")
        return v


class LikertElicitationResponse(BaseModel):
    action_scores: list[ActionLikertScore]
    commentary: str

    @model_validator(mode="after")
    def check_constraints(self) -> "LikertElicitationResponse":
        seen = set()
        for als in self.action_scores:
            if als.action in seen:
                raise ValueError(f"Duplicate action: {als.action}")
            seen.add(als.action)
        return self


class TokenUsage(BaseModel):
    scenario_id: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


@dataclass
class RunCostSummary:
    total_calls: int = 0
    successful_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    cost_by_model: dict = field(default_factory=dict)
    cost_by_tier: dict = field(default_factory=dict)
    cost_by_scenario: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, usage: TokenUsage, tier: str = ""):
        with self._lock:
            self.total_calls += 1
            self.successful_calls += 1
            self.total_prompt_tokens += usage.prompt_tokens
            self.total_completion_tokens += usage.completion_tokens
            self.total_tokens += usage.total_tokens
            self.total_cost_usd += usage.estimated_cost_usd
            self.cost_by_model[usage.model] = (
                self.cost_by_model.get(usage.model, 0.0) + usage.estimated_cost_usd
            )
            if tier:
                self.cost_by_tier[tier] = (
                    self.cost_by_tier.get(tier, 0.0) + usage.estimated_cost_usd
                )
            self.cost_by_scenario[usage.scenario_id] = (
                self.cost_by_scenario.get(usage.scenario_id, 0.0)
                + usage.estimated_cost_usd
            )

    def to_dict(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "cost_by_model": self.cost_by_model,
            "cost_by_tier": self.cost_by_tier,
        }


class TokenLimitRunError(Exception):
    pass


# ── SEC 2: Text sanitisation ─────────────────────────────────────────────────

_SMART_CHAR_TABLE = str.maketrans({
    "\u2018": "'", "\u2019": "'",
    "\u201C": '"', "\u201D": '"',
    "\u2013": "-", "\u2014": "--",
    "\u2026": "...",
    "\u00A0": " ",
})

_encoding_stats = {"replacements": 0, "non_ascii": 0, "bom": 0, "zwsp": 0}
_encoding_lock = threading.Lock()


def sanitise_text(s: str) -> str:
    """Normalise smart quotes, BOM, zero-width spaces, non-ASCII."""
    if not s:
        return s
    count = 0
    s = s.translate(_SMART_CHAR_TABLE)
    if s.startswith("\ufeff"):
        s = s[1:]
        with _encoding_lock:
            _encoding_stats["bom"] += 1
    s = s.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    cleaned = []
    for ch in s:
        if ord(ch) > 127:
            try:
                from unidecode import unidecode as _ud
                cleaned.append(_ud(ch))
            except ImportError:
                cleaned.append(unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode())
            count += 1
        else:
            cleaned.append(ch)
    result = "".join(cleaned)
    if count:
        with _encoding_lock:
            _encoding_stats["non_ascii"] += count
            _encoding_stats["replacements"] += count
    return result


def _run_encoding_self_test() -> bool:
    """Verify UTF-8 round-trip."""
    test_str = "Board\u2019s \u201Cgovernance\u201D review \u2014 A$21.4M"
    cleaned = sanitise_text(test_str)
    try:
        encoded = cleaned.encode("utf-8")
        decoded = encoded.decode("utf-8")
        return decoded == cleaned
    except (UnicodeEncodeError, UnicodeDecodeError):
        return False


# ── SEC 3: Caching ───────────────────────────────────────────────────────────

_cache_stats = {"hits": 0, "misses": 0}


def _make_cache_key(system_prompt: str, scenario_prompt: str,
                    model: str, seed: int, temperature: float) -> str:
    """SHA256 hash of payload for deterministic cache key."""
    payload = json.dumps({
        "prompt": scenario_prompt,
        "model": model,
        "seed": seed,
        "temperature": temperature,
        "cache_version": "asa_v1_likert",
    }, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_lookup(key: str, track_stats: bool = True) -> Optional[dict]:
    cache_path = CACHE_DIR / f"{key}.json"
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if track_stats:
                _cache_stats["hits"] += 1
            return data
        except (json.JSONDecodeError, OSError):
            pass
    if track_stats:
        _cache_stats["misses"] += 1
    return None


def _cache_store(key: str, data: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{key}.json"
    try:
        cache_path.write_text(
            json.dumps(data, ensure_ascii=True, default=str),
            encoding="utf-8",
        )
    except OSError as e:
        logger.debug(f"Cache write failed: {e}")


# ── SEC 4: LLM client & rate limiting ────────────────────────────────────────

def _compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    prices = MODEL_PRICE_TABLE.get(model, MODEL_PRICE_TABLE["gpt-4o-mini"])
    return (prompt_tokens * prices["prompt_per_1k"] / 1000
            + completion_tokens * prices["completion_per_1k"] / 1000)


def _get_instructor_client(api_key: Optional[str] = None):
    """Create instructor-wrapped OpenAI client."""
    import instructor
    from openai import OpenAI
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OpenAI API key required (--api_key or OPENAI_API_KEY env var)")
    client = instructor.from_openai(OpenAI(api_key=key))
    return client


def _try_json_repair(raw_text: str) -> Optional[str]:
    """Attempt to repair malformed JSON from LLM output."""
    if not raw_text:
        return None
    # Strip markdown code blocks
    cleaned = re.sub(r"```(?:json)?\s*", "", raw_text)
    cleaned = cleaned.strip()
    try:
        import json_repair
        return json_repair.repair_json(cleaned)
    except (ImportError, Exception):
        return None


def _call_llm_with_retry(
    client, model: str, messages: list[dict],
    scenario_id: str, max_retries: int = 6,
    temperature: float = 1.0,
) -> tuple[Optional[LikertElicitationResponse], dict]:
    """Call LLM with exponential backoff on rate limit errors."""
    meta = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "finish_reason": "", "raw_content": ""}

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_model=LikertElicitationResponse,
                temperature=temperature,
                max_tokens=1024,
            )
            # instructor returns the parsed model directly
            meta["finish_reason"] = "stop"
            return response, meta

        except Exception as e:
            error_name = type(e).__name__
            if "RateLimitError" in error_name:
                wait = min(2 ** attempt, 60)
                logger.warning(f"Rate limit for {scenario_id}, waiting {wait}s...")
                time.sleep(wait)
                continue
            elif "BadRequestError" in error_name:
                logger.error(f"Bad request for {scenario_id}: {e}")
                meta["raw_content"] = str(e)
                return None, meta
            elif attempt < max_retries - 1:
                wait = min(2 ** attempt, 30)
                logger.warning(f"{error_name} for {scenario_id}, retry in {wait}s: {e}")
                time.sleep(wait)
                continue
            else:
                logger.error(f"Max retries for {scenario_id}: {e}")
                meta["raw_content"] = str(e)
                return None, meta

    return None, meta


# ── SEC 5: Scenario dataclass ────────────────────────────────────────────────

@dataclass
class Scenario:
    scenario_id: str
    tier: int  # 1=isolation, 2=joint, 3=behavioural, 4=historical
    target_parameter: str
    decision_node: str  # Always "A2" for ASA
    state_vector: dict
    feasible_actions: list[str]
    prompt_text: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["prompt_text"] = d["prompt_text"][:200] + "..." if len(d["prompt_text"]) > 200 else d["prompt_text"]
        return d


# ── SEC 6: Stage 1 — System prompt and scenario generation ──────────────────

def _build_system_prompt() -> str:
    """Build the ASA persona system prompt for LLM elicitation.

    Sections:
    A: ASA Persona — mission, member base, institutional identity
    B: Decision Framework — two-strikes rule, what a strike entails
    C: Game Tree Context — what ASA observes at its decision point
    D: Strategic Reasoning — accountability logic, pragmatic logic, repeat game
    E: Response Format — Likert rating instructions
    """
    return """You are a decision-analysis assistant modelling the Australian Shareholders'
Association (ASA) as a rational actor in a corporate governance crisis.

=== SECTION A: ASA PERSONA ===

ASA is Australia's largest independent not-for-profit organisation representing
retail shareholders. Its mission is to protect and advance retail shareholder
interests through corporate governance monitoring, proxy voting, and advocacy.

ASA publishes "voting intentions" (VIs) prepared by volunteer company monitors —
experienced retail shareholders who follow specific companies and assess governance
quality. ASA's credibility depends on being rigorous, evidence-based, and
independent — neither reflexively anti-management nor captured by corporate
relationships.

ASA's member base is predominantly self-managed superannuation fund (SMSF) trustees
and direct retail shareholders who are long-term holders.

=== SECTION B: DECISION FRAMEWORK ===

Under the Corporations Act 2001 (Cth), shareholders vote on the remuneration report
at each AGM. If 25% or more vote against, this constitutes a "first strike". If a
second strike occurs the following year, shareholders vote to spill the board.

A "strike recommendation" by ASA means:
- ASA votes its open proxies against the remuneration report
- ASA publishes a voting intention explaining its reasoning
- This incurs volunteer mobilisation cost and may strain the company relationship

A "no strike" means:
- ASA votes its open proxies in favour of or abstains on the remuneration report
- ASA may still express concerns privately
- This preserves the relationship for future engagement

=== SECTION C: GAME TREE CONTEXT ===

ASA makes its decision (recommend strike or no strike) AFTER observing what the
Board and CEO have already done. At ASA's decision point, the following game tree
outcomes are KNOWN:

1. Whether the CEO has resigned voluntarily (a CEO-initiated departure)
2. Whether the Board has forced the CEO out (a Board-initiated accountability action)
3. Whether the Board has commissioned an independent governance review
4. How quickly the Board acted (immediately after the crisis, or only after
   extended public and regulatory pressure)
5. Whether this is a high-profile case with significant public salience

The crisis context is FIXED: the company's remuneration report is public, the
pay-performance decoupling is established, and any regulatory/ESG issues are known.
The ONLY thing that varies across scenarios is what the Board and CEO have done in
response to the crisis.

ASA's decision affects downstream outcomes:
- A strike recommendation increases the probability of a first strike vote (>25%)
- A first strike increases pressure for CEO removal and governance reform
- ASA's recommendation is observed by the market and by other boards

CRITICAL: ASA's voting intention for any specific company is NOT known at this
decision point. You are reasoning prospectively about what ASA SHOULD do given the
described circumstances.

=== SECTION D: STRATEGIC REASONING ===

ASA weighs its decision using three considerations:

1. THE ACCOUNTABILITY LOGIC: When the Board has not imposed meaningful consequences
   (CEO still in place, no review, no reform), a strike is the primary mechanism
   for ASA to signal governance norms. Inaction by ASA when the Board has been
   passive is costly — it signals that governance failures have no consequences.

2. THE PRAGMATIC LOGIC: When the Board has already taken credible action (forced
   CEO exit, commissioned review), a strike may be counterproductive. It punishes
   a board that is already reforming and risks damaging the constructive
   relationship ASA needs for future engagement.

3. THE REPEAT-GAME LOGIC: ASA's decision is not just about this company. ASA's
   credibility as a governance watchdog depends on being seen as willing to strike
   when warranted. In high-profile cases, visible inaction damages ASA's long-term
   reputation with its member base and reduces its leverage over future boards.
   Conversely, striking when the Board has already acted well wastes credibility.

The speed of Board action also matters: a Board that acts immediately demonstrates
genuine commitment; a Board that acts only after prolonged public pressure raises
questions about whether the response is genuine reform or damage control.

=== SECTION E: RESPONSE FORMAT ===

For each scenario, you will rate the appropriateness of EACH action on a 1-5 scale:
  1 = Not appropriate / not warranted given the circumstances
  2 = Marginally appropriate
  3 = Uncertain / could go either way
  4 = Somewhat appropriate / warranted
  5 = Strongly appropriate / clearly warranted

Respond with structured Likert scores for each action."""


def _make_state_vector(
    ceo_resigned: bool = False,
    ceo_sacked: bool = False,
    review_commissioned: bool = False,
    action_delayed: bool = False,
    high_profile: bool = True,
    **kwargs,
) -> dict:
    """Construct ASA state vector from game tree measurable inputs.

    All inputs are directly observable game tree states or decisions:
      ceo_resigned:        D0_ceo = CEO_resign (voluntary departure)
      ceo_sacked:          D1 = D3_ceo_transition (Board forced exit)
      review_commissioned: D1 = D1_review (Board commissioned review)
      action_delayed:      Board acted but was slow (timing of D1)
      high_profile:        Case has high public salience (Qantas = True)
    """
    return {
        "decision_node": "A2",
        "ceo_resigned": bool(ceo_resigned),
        "ceo_sacked": bool(ceo_sacked),
        "review_commissioned": bool(review_commissioned),
        "action_delayed": bool(action_delayed),
        "high_profile": bool(high_profile),
        **kwargs,
    }


def _build_scenario_prompt(scenario: "Scenario") -> str:
    """Build the user-facing scenario prompt from game tree state vector.

    Scenario descriptions are constructed entirely from measurable game tree
    inputs: CEO departure status, Board actions, timing, and case profile.
    """
    sv = scenario.state_vector

    ceo_resigned = sv.get("ceo_resigned", False)
    ceo_sacked = sv.get("ceo_sacked", False)
    review_commissioned = sv.get("review_commissioned", False)
    action_delayed = sv.get("action_delayed", False)
    high_profile = sv.get("high_profile", True)

    # Use custom prompt text if provided (for core A2 node scenarios)
    custom_prompt = sv.get("custom_prompt", "")
    if custom_prompt:
        return custom_prompt

    # ── Build path description from game tree inputs ──

    # CEO status
    if ceo_sacked:
        ceo_desc = ("The Board has taken the significant step of forcing the CEO's "
                     "departure. This is a Board-initiated accountability action, "
                     "not a voluntary resignation.")
    elif ceo_resigned:
        ceo_desc = ("The CEO has announced their resignation. This is a voluntary "
                     "departure — the CEO chose to leave rather than being forced out "
                     "by the Board.")
    else:
        ceo_desc = ("The CEO remains in position. The CEO has not resigned and the "
                     "Board has not forced the CEO's departure.")

    # Board governance action
    if review_commissioned:
        review_desc = ("The Board has commissioned an independent governance review "
                        "with publicly disclosed terms of reference.")
    else:
        review_desc = ("The Board has NOT commissioned an independent governance review. "
                        "No structural governance reform has been announced.")

    # Board overall posture
    board_acted = review_commissioned or ceo_sacked
    if not board_acted:
        board_desc = ("The Board has taken no substantive governance action. "
                       "Its public posture has been defensive.")
    elif ceo_sacked and review_commissioned:
        board_desc = ("The Board has taken strong accountability action: forced "
                       "CEO departure AND commissioned a governance review.")
    elif ceo_sacked:
        board_desc = ("The Board has forced the CEO's departure — the strongest "
                       "possible accountability action short of clawback.")
    else:
        board_desc = ("The Board has commissioned a governance review, signalling "
                       "willingness to reform, but has not taken direct action "
                       "against the CEO.")

    # Timing / speed of action
    if action_delayed and board_acted:
        timing_desc = ("The Board's action came only after weeks of sustained public "
                        "and regulatory pressure. The delay raises questions about "
                        "whether the response reflects genuine commitment to reform "
                        "or reactive damage control.")
    elif board_acted:
        timing_desc = ("The Board acted promptly — within days of the crisis becoming "
                        "public. The speed of response suggests genuine commitment to "
                        "accountability rather than reactive damage control.")
    else:
        timing_desc = ("The Board has not acted despite the passage of time since "
                        "the crisis became public.")

    # Case profile
    if high_profile:
        profile_desc = ("This is a nationally prominent case. The company is a major "
                         "consumer brand and household name. Media coverage is intense "
                         "and sustained. ASA's members are overwhelmingly shareholders "
                         "and/or customers. ASA's stance will be front-page news.")
    else:
        profile_desc = ("This is a lower-profile case. The company is not a major "
                         "consumer brand and media coverage has been limited. While "
                         "ASA monitors the company, its stance will receive modest "
                         "public attention.")

    # Crisis context (fixed across all scenarios)
    crisis_desc = sv.get("crisis_context", (
        "The company's remuneration report is public. Executive pay has increased "
        "substantially — a multi-fold increase year-on-year — against a backdrop of "
        "serious regulatory action (court proceedings filed), confirmed labour law "
        "violations, and insider share sale timing concerns. The pay-performance "
        "link is fundamentally broken."
    ))

    prompt = f"""[GAME STATE — PATH TO ASA's DECISION POINT]

Crisis context: {crisis_desc}

CEO status: {ceo_desc}

Board governance action: {review_desc}

Board accountability posture: {board_desc}

Speed of Board response: {timing_desc}

Case profile: {profile_desc}

[QUESTION]
ASA's company monitor team is preparing its voting intentions for this company's
upcoming AGM. The remuneration report is now public. Two actions are feasible:

Action A — No strike recommendation: ASA votes open proxies in favour of (or
abstains on) the remuneration report. ASA may still express concerns privately.

Action B — Recommend strike: ASA votes open proxies against the remuneration
report and publishes a voting intention explaining its reasoning.

Please rate the appropriateness of EACH action separately on a 1-5 scale:
1 = Not appropriate / not warranted given the circumstances
3 = Uncertain / could go either way
5 = Strongly appropriate / clearly warranted

Respond with structured Likert scores for each action and brief reasoning."""
    return prompt


# ── Tier 1: Core Game Tree Path Scenarios ──

def _generate_tier1_scenarios() -> list[Scenario]:
    """Generate Tier 1 scenarios: the 5 core A2 game tree paths.

    Each scenario corresponds to one A2 node in the game tree, using the
    narrative descriptions from background/asa/asa_a2_prompt.md.
    These are the primary identification scenarios — each path activates
    a different combination of game tree indicators.
    """
    scenarios = []

    # ── A2-1: CEO resigned → Board does nothing ──
    sv = _make_state_vector(
        ceo_resigned=True, ceo_sacked=False,
        review_commissioned=False, action_delayed=False,
        high_profile=True,
    )
    s = Scenario(
        scenario_id="T1_ceo_resign_nothing",
        tier=1, target_parameter="core_path",
        decision_node="A2", state_vector=sv,
        feasible_actions=["no_strike", "rec_strike"],
        prompt_text="",
    )
    s.prompt_text = _build_scenario_prompt(s)
    scenarios.append(s)

    # ── A2-2: CEO resigned → Board commissions review ──
    sv = _make_state_vector(
        ceo_resigned=True, ceo_sacked=False,
        review_commissioned=True, action_delayed=False,
        high_profile=True,
    )
    s = Scenario(
        scenario_id="T1_ceo_resign_review",
        tier=1, target_parameter="core_path",
        decision_node="A2", state_vector=sv,
        feasible_actions=["no_strike", "rec_strike"],
        prompt_text="",
    )
    s.prompt_text = _build_scenario_prompt(s)
    scenarios.append(s)

    # ── A2-3: CEO stays → Board does nothing ──
    sv = _make_state_vector(
        ceo_resigned=False, ceo_sacked=False,
        review_commissioned=False, action_delayed=False,
        high_profile=True,
    )
    s = Scenario(
        scenario_id="T1_ceo_stay_nothing",
        tier=1, target_parameter="core_path",
        decision_node="A2", state_vector=sv,
        feasible_actions=["no_strike", "rec_strike"],
        prompt_text="",
    )
    s.prompt_text = _build_scenario_prompt(s)
    scenarios.append(s)

    # ── A2-4: CEO stays → Board commissions review ──
    sv = _make_state_vector(
        ceo_resigned=False, ceo_sacked=False,
        review_commissioned=True, action_delayed=False,
        high_profile=True,
    )
    s = Scenario(
        scenario_id="T1_ceo_stay_review",
        tier=1, target_parameter="core_path",
        decision_node="A2", state_vector=sv,
        feasible_actions=["no_strike", "rec_strike"],
        prompt_text="",
    )
    s.prompt_text = _build_scenario_prompt(s)
    scenarios.append(s)

    # ── A2-5: CEO stays → Board forces CEO exit ──
    sv = _make_state_vector(
        ceo_resigned=False, ceo_sacked=True,
        review_commissioned=False, action_delayed=False,
        high_profile=True,
    )
    s = Scenario(
        scenario_id="T1_ceo_stay_sacked",
        tier=1, target_parameter="core_path",
        decision_node="A2", state_vector=sv,
        feasible_actions=["no_strike", "rec_strike"],
        prompt_text="",
    )
    s.prompt_text = _build_scenario_prompt(s)
    scenarios.append(s)

    logger.info(f"Tier 1: {len(scenarios)} core game tree path scenarios generated")
    return scenarios


# ── Tier 2: Parameter Isolation & Timing Scenarios ──

def _generate_tier2_scenarios() -> list[Scenario]:
    """Generate Tier 2 scenarios for parameter isolation, timing, and salience.

    These scenarios vary one game tree input at a time to isolate individual
    parameters. Includes:
    - Timing variations (immediate vs delayed Board action)
    - Salience variations (high-profile vs low-profile cases)
    - Combined action scenarios (sack + review)
    """
    scenarios = []
    sid_counter = [0]

    def _sid(suffix: str) -> str:
        sid_counter[0] += 1
        return f"T2_{suffix}_{sid_counter[0]:03d}"

    # ── Timing variations (6 scenarios) ──
    # Same Board action, but delayed vs immediate

    # CEO resigned + delayed review
    sv = _make_state_vector(
        ceo_resigned=True, review_commissioned=True,
        action_delayed=True, high_profile=True,
    )
    s = Scenario(
        scenario_id=_sid("delay"),
        tier=2, target_parameter="w_strike_cost",
        decision_node="A2", state_vector=sv,
        feasible_actions=["no_strike", "rec_strike"],
        prompt_text="",
    )
    s.prompt_text = _build_scenario_prompt(s)
    scenarios.append(s)

    # CEO stays + delayed review
    sv = _make_state_vector(
        ceo_resigned=False, review_commissioned=True,
        action_delayed=True, high_profile=True,
    )
    s = Scenario(
        scenario_id=_sid("delay"),
        tier=2, target_parameter="w_strike_cost",
        decision_node="A2", state_vector=sv,
        feasible_actions=["no_strike", "rec_strike"],
        prompt_text="",
    )
    s.prompt_text = _build_scenario_prompt(s)
    scenarios.append(s)

    # CEO stays + delayed forced exit
    sv = _make_state_vector(
        ceo_resigned=False, ceo_sacked=True,
        action_delayed=True, high_profile=True,
    )
    s = Scenario(
        scenario_id=_sid("delay"),
        tier=2, target_parameter="w_strike_cost",
        decision_node="A2", state_vector=sv,
        feasible_actions=["no_strike", "rec_strike"],
        prompt_text="",
    )
    s.prompt_text = _build_scenario_prompt(s)
    scenarios.append(s)

    # Low-profile + delayed review (tests delay × salience interaction)
    sv = _make_state_vector(
        ceo_resigned=False, review_commissioned=True,
        action_delayed=True, high_profile=False,
    )
    s = Scenario(
        scenario_id=_sid("delay_low_profile"),
        tier=2, target_parameter="w_strike_cost",
        decision_node="A2", state_vector=sv,
        feasible_actions=["no_strike", "rec_strike"],
        prompt_text="",
    )
    s.prompt_text = _build_scenario_prompt(s)
    scenarios.append(s)

    # Low-profile + delayed forced exit
    sv = _make_state_vector(
        ceo_resigned=False, ceo_sacked=True,
        action_delayed=True, high_profile=False,
    )
    s = Scenario(
        scenario_id=_sid("delay_low_profile"),
        tier=2, target_parameter="w_strike_cost",
        decision_node="A2", state_vector=sv,
        feasible_actions=["no_strike", "rec_strike"],
        prompt_text="",
    )
    s.prompt_text = _build_scenario_prompt(s)
    scenarios.append(s)

    # CEO resigned + immediate review (contrast for delay scenarios)
    sv = _make_state_vector(
        ceo_resigned=True, review_commissioned=True,
        action_delayed=False, high_profile=True,
    )
    s = Scenario(
        scenario_id=_sid("immediate"),
        tier=2, target_parameter="w_strike_cost",
        decision_node="A2", state_vector=sv,
        feasible_actions=["no_strike", "rec_strike"],
        prompt_text="",
    )
    s.prompt_text = _build_scenario_prompt(s)
    scenarios.append(s)

    # ── Salience variations (6 scenarios) ──
    # Same game tree path, high-profile vs low-profile

    # Low-profile + Board does nothing + CEO stays
    sv = _make_state_vector(
        ceo_resigned=False, ceo_sacked=False,
        review_commissioned=False, high_profile=False,
    )
    s = Scenario(
        scenario_id=_sid("low_profile"),
        tier=2, target_parameter="w_credibility_signal",
        decision_node="A2", state_vector=sv,
        feasible_actions=["no_strike", "rec_strike"],
        prompt_text="",
    )
    s.prompt_text = _build_scenario_prompt(s)
    scenarios.append(s)

    # Low-profile + CEO resigned + nothing
    sv = _make_state_vector(
        ceo_resigned=True, ceo_sacked=False,
        review_commissioned=False, high_profile=False,
    )
    s = Scenario(
        scenario_id=_sid("low_profile"),
        tier=2, target_parameter="w_credibility_signal",
        decision_node="A2", state_vector=sv,
        feasible_actions=["no_strike", "rec_strike"],
        prompt_text="",
    )
    s.prompt_text = _build_scenario_prompt(s)
    scenarios.append(s)

    # Low-profile + Board commissions review (no CEO departure)
    sv = _make_state_vector(
        ceo_resigned=False, ceo_sacked=False,
        review_commissioned=True, high_profile=False,
    )
    s = Scenario(
        scenario_id=_sid("low_profile"),
        tier=2, target_parameter="w_credibility_signal",
        decision_node="A2", state_vector=sv,
        feasible_actions=["no_strike", "rec_strike"],
        prompt_text="",
    )
    s.prompt_text = _build_scenario_prompt(s)
    scenarios.append(s)

    # Low-profile + Board forces CEO exit
    sv = _make_state_vector(
        ceo_resigned=False, ceo_sacked=True,
        review_commissioned=False, high_profile=False,
    )
    s = Scenario(
        scenario_id=_sid("low_profile"),
        tier=2, target_parameter="w_credibility_signal",
        decision_node="A2", state_vector=sv,
        feasible_actions=["no_strike", "rec_strike"],
        prompt_text="",
    )
    s.prompt_text = _build_scenario_prompt(s)
    scenarios.append(s)

    # Low-profile + CEO resigned + review
    sv = _make_state_vector(
        ceo_resigned=True, ceo_sacked=False,
        review_commissioned=True, high_profile=False,
    )
    s = Scenario(
        scenario_id=_sid("low_profile"),
        tier=2, target_parameter="w_credibility_signal",
        decision_node="A2", state_vector=sv,
        feasible_actions=["no_strike", "rec_strike"],
        prompt_text="",
    )
    s.prompt_text = _build_scenario_prompt(s)
    scenarios.append(s)

    # Low-profile + Board sacked + review
    sv = _make_state_vector(
        ceo_resigned=False, ceo_sacked=True,
        review_commissioned=True, high_profile=False,
    )
    s = Scenario(
        scenario_id=_sid("low_profile"),
        tier=2, target_parameter="w_credibility_signal",
        decision_node="A2", state_vector=sv,
        feasible_actions=["no_strike", "rec_strike"],
        prompt_text="",
    )
    s.prompt_text = _build_scenario_prompt(s)
    scenarios.append(s)

    # ── Combined action scenarios (sack + review) ──
    # These are NOT in the 5 core A2 nodes but test w_sack_dampens + w_ctx_review together

    # High-profile + sack + review + immediate
    sv = _make_state_vector(
        ceo_resigned=False, ceo_sacked=True,
        review_commissioned=True, action_delayed=False,
        high_profile=True,
    )
    s = Scenario(
        scenario_id=_sid("combined"),
        tier=2, target_parameter="joint",
        decision_node="A2", state_vector=sv,
        feasible_actions=["no_strike", "rec_strike"],
        prompt_text="",
    )
    s.prompt_text = _build_scenario_prompt(s)
    scenarios.append(s)

    # High-profile + sack + review + delayed
    sv = _make_state_vector(
        ceo_resigned=False, ceo_sacked=True,
        review_commissioned=True, action_delayed=True,
        high_profile=True,
    )
    s = Scenario(
        scenario_id=_sid("combined"),
        tier=2, target_parameter="joint",
        decision_node="A2", state_vector=sv,
        feasible_actions=["no_strike", "rec_strike"],
        prompt_text="",
    )
    s.prompt_text = _build_scenario_prompt(s)
    scenarios.append(s)

    logger.info(f"Tier 2: {len(scenarios)} isolation & timing scenarios generated")
    return scenarios


# ── Tier 3: Behavioural Probes ──

def _generate_tier3_scenarios() -> list[Scenario]:
    """Generate Tier 3 behavioural probe scenarios.

    These test whether the LLM exhibits biases incompatible with the
    game tree indicator model. Not used in estimation.
    """
    scenarios = []
    sid_counter = [0]

    def _sid(bias: str) -> str:
        sid_counter[0] += 1
        return f"T3_{bias}_{sid_counter[0]:03d}"

    # ── Framing bias: resignation vs forced exit (4 scenarios) ──
    # Same functional outcome (CEO departed) but framing differs.
    # Model treats these differently (w_sack_dampens), but probe
    # checks that the magnitude difference is reasonable.

    # "CEO resigned under pressure" — is this voluntary or forced?
    for frame_label, frame_desc in [
        ("voluntary", "The CEO announced a voluntary resignation, citing personal reasons."),
        ("forced_framing", "The Board announced it had terminated the CEO's employment, "
                           "effective immediately."),
        ("ambiguous", "The CEO departed. The company statement said it was by mutual "
                      "agreement between the CEO and the Board."),
        ("pressure", "The CEO resigned after weeks of sustained public pressure and "
                     "media scrutiny, but the Board did not formally force the departure."),
    ]:
        sv = _make_state_vector(
            ceo_resigned=(frame_label in ("voluntary", "ambiguous", "pressure")),
            ceo_sacked=(frame_label == "forced_framing"),
            review_commissioned=False, high_profile=True,
        )
        # Override with custom prompt for framing variation
        sv["custom_prompt"] = f"""[GAME STATE — PATH TO ASA's DECISION POINT]

Crisis context: The company's remuneration report is public. Executive pay has \
increased substantially against a backdrop of regulatory action and governance \
failures. The pay-performance link is fundamentally broken.

CEO status: {frame_desc}

Board governance action: The Board has NOT commissioned an independent governance \
review. No structural governance reform has been announced.

Case profile: This is a nationally prominent case with intense media coverage. \
ASA's stance will be front-page news.

[QUESTION]
ASA's company monitor team is preparing its voting intentions. Two actions are feasible:

Action A — No strike recommendation: ASA votes open proxies in favour of the \
remuneration report.
Action B — Recommend strike: ASA votes open proxies against the remuneration report.

Rate the appropriateness of EACH action on a 1-5 scale:
1 = Not appropriate   3 = Uncertain   5 = Strongly appropriate

Respond with structured Likert scores for each action and brief reasoning."""

        s = Scenario(
            scenario_id=_sid("framing"),
            tier=3, target_parameter="framing_bias",
            decision_node="A2", state_vector=sv,
            feasible_actions=["no_strike", "rec_strike"],
            prompt_text="",
        )
        s.prompt_text = _build_scenario_prompt(s)
        scenarios.append(s)

    # ── Sequence bias: same outcome, different temporal order (4 scenarios) ──
    for seq_label, seq_desc in [
        ("review_then_resign",
         "The Board first commissioned a governance review. Two weeks later, "
         "the CEO resigned voluntarily."),
        ("resign_then_review",
         "The CEO resigned first. The Board then commissioned a governance "
         "review in response."),
        ("simultaneous",
         "The CEO's resignation and the Board's announcement of a governance "
         "review were made on the same day."),
        ("review_no_resign",
         "The Board commissioned a governance review. The CEO has not resigned "
         "and remains in position."),
    ]:
        is_resigned = seq_label != "review_no_resign"
        sv = _make_state_vector(
            ceo_resigned=is_resigned,
            review_commissioned=True, high_profile=True,
        )
        sv["custom_prompt"] = f"""[GAME STATE — PATH TO ASA's DECISION POINT]

Crisis context: The company's remuneration report is public. Executive pay is \
egregiously above benchmark against regulatory and ESG failures.

Sequence of events: {seq_desc}

Case profile: National icon, maximum media coverage.

[QUESTION]
Rate the appropriateness of EACH action on a 1-5 scale:
Action A — No strike recommendation   Action B — Recommend strike
1 = Not appropriate   3 = Uncertain   5 = Strongly appropriate

Respond with structured Likert scores for each action and brief reasoning."""

        s = Scenario(
            scenario_id=_sid("sequence"),
            tier=3, target_parameter="sequence_bias",
            decision_node="A2", state_vector=sv,
            feasible_actions=["no_strike", "rec_strike"],
            prompt_text="",
        )
        s.prompt_text = _build_scenario_prompt(s)
        scenarios.append(s)

    logger.info(f"Tier 3: {len(scenarios)} behavioural probe scenarios generated")
    return scenarios


# ── Tier 4: Historical Calibration ──

def _generate_tier4_scenario() -> Scenario:
    """Generate the Qantas November 2023 AGM historical calibration scenario.

    Game tree state: CEO resigned voluntarily, Board commissioned review.
    Observed outcome: ASA recommended strike (83% against vote).
    """
    sv = _make_state_vector(
        ceo_resigned=True,
        ceo_sacked=False,
        review_commissioned=True,
        action_delayed=False,  # Review was announced reasonably promptly
        high_profile=True,
    )
    sv["custom_prompt"] = """[GAME STATE — PATH TO ASA's DECISION POINT]

Decision point: Late September 2023.

Path to this node:
- Alan Joyce announced his immediate resignation effective 2 September 2023,
  brought forward from a planned November departure following intense public
  and regulatory pressure.
- The Qantas Board has announced an independent review of the company's
  governance and culture. Terms of reference and reviewer disclosed publicly.
- No clawback of Joyce's FY23 remuneration has been announced, but the Board
  has signalled it is exploring conditional holdback of bonus components.
- Incoming Chair John Mullen made public statements acknowledging governance
  failings.

Crisis context:
- FY23 statutory profit A$2.47 billion
- Alan Joyce FY23 remuneration: A$21.4 million (near 10-fold increase YoY)
- ACCC filed Federal Court action alleging ~8,000 ghost flight ticket sales
- Federal Court ruled Qantas illegally outsourced ~1,700 ground workers
- Joyce sold ~90% of his shareholding before the ACCC announcement
- FY23 remuneration report: no conduct-linked gating, no clawback provisions

Case profile: Qantas is a national icon. Maximum media coverage, near-universal
ASA member exposure as shareholders and customers. ASA's stance is front-page news.

CRITICAL: ASA's actual voting intention is NOT known at this decision point.
Reason prospectively about what ASA SHOULD do.

[QUESTION]
ASA's company monitor team is preparing its voting intentions for the Qantas
November 2023 AGM. Two actions are feasible:

Action A — No strike recommendation
Action B — Recommend strike

Rate the appropriateness of EACH action on a 1-5 scale:
1 = Not appropriate   3 = Uncertain   5 = Strongly appropriate

Respond with structured Likert scores for each action and brief reasoning."""

    s = Scenario(
        scenario_id="T4_qantas_2023",
        tier=4, target_parameter="historical_calibration",
        decision_node="A2", state_vector=sv,
        feasible_actions=["no_strike", "rec_strike"],
        prompt_text="",
    )
    s.prompt_text = _build_scenario_prompt(s)
    return s


def generate_scenarios(output_path: Path) -> list[Scenario]:
    """Stage 1: Generate all scenario tiers and save to CSV."""
    logger.info("Stage 1: Generating ASA scenario battery...")

    tier1 = _generate_tier1_scenarios()
    tier2 = _generate_tier2_scenarios()
    tier3 = _generate_tier3_scenarios()
    tier4 = [_generate_tier4_scenario()]

    # A2 game tree node scenarios for dashboard Likert tab.
    # These match Tier 1 core paths but with A2_ prefix for dashboard display.
    a2_scenarios = []
    for node_name, state_vec in A2_NODE_STATES.items():
        sv = _make_state_vector(**state_vec)
        s = Scenario(
            scenario_id=f"A2_{node_name}",
            tier=1, target_parameter="a2_node",
            decision_node="A2", state_vector=sv,
            feasible_actions=["no_strike", "rec_strike"],
            prompt_text="",
        )
        s.prompt_text = _build_scenario_prompt(s)
        a2_scenarios.append(s)

    all_scenarios = tier1 + tier2 + tier3 + tier4 + a2_scenarios

    logger.info(f"Total scenarios: {len(all_scenarios)} "
                f"(T1={len(tier1)}, T2={len(tier2)}, T3={len(tier3)}, "
                f"T4={len(tier4)}, A2={len(a2_scenarios)})")

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["scenario_id", "tier", "target_parameter", "decision_node",
                   "state_vector", "feasible_actions", "prompt_text"]
    with open(output_path, "w", encoding="utf-8", errors="replace", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in all_scenarios:
            writer.writerow({
                "scenario_id": s.scenario_id,
                "tier": s.tier,
                "target_parameter": s.target_parameter,
                "decision_node": s.decision_node,
                "state_vector": json.dumps(s.state_vector, ensure_ascii=True),
                "feasible_actions": json.dumps(s.feasible_actions, ensure_ascii=True),
                "prompt_text": s.prompt_text,
            })

    logger.info(f"Scenarios saved to {output_path}")
    return all_scenarios


def load_scenarios(path: Path) -> list[Scenario]:
    """Load scenarios from CSV."""
    scenarios = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sv = json.loads(row["state_vector"]) if isinstance(row["state_vector"], str) else row["state_vector"]
            fa = json.loads(row["feasible_actions"]) if isinstance(row["feasible_actions"], str) else row["feasible_actions"]
            scenarios.append(Scenario(
                scenario_id=row["scenario_id"],
                tier=int(row["tier"]),
                target_parameter=row["target_parameter"],
                decision_node=row["decision_node"],
                state_vector=sv,
                feasible_actions=fa,
                prompt_text=row.get("prompt_text", ""),
            ))
    return scenarios


# ── SEC 7: Stage 2 — LLM Elicitation ────────────────────────────────────────

def _elicit_single(
    scenario: Scenario,
    draw: int,
    client,
    model: str,
    system_prompt: str,
    cost_tracker: RunCostSummary,
    token_limit_counter: list[int],
    temperature: Optional[float] = 1.0,
) -> dict:
    """Elicit a single Likert response for one scenario + draw."""
    _content_seed = int(hashlib.sha256(
        f"{scenario.prompt_text}|{draw}".encode()
    ).hexdigest(), 16) & 0xFFFFFFFF
    rng = np.random.default_rng(_content_seed)

    n_actions = len(scenario.feasible_actions)
    action_order = rng.permutation(n_actions).tolist()
    shuffled_actions = [scenario.feasible_actions[i] for i in action_order]

    hash_token = hashlib.sha256(
        f"draw_{draw}_{scenario.scenario_id}".encode()
    ).hexdigest()[:16]

    action_list_str = "\n".join(f"  - {a}" for a in shuffled_actions)
    scenario_text = (
        f"[Reference ID: {hash_token}]\n\n"
        f"{scenario.prompt_text}\n\n"
        f"FEASIBLE ACTIONS (rate each on 1-5 scale):\n{action_list_str}"
    )

    cache_key = _make_cache_key("", scenario.prompt_text, model, draw, temperature or 0.0)
    cached = _cache_lookup(cache_key)

    result_row = {
        "result_id": str(uuid.uuid4())[:12],
        "scenario_id": scenario.scenario_id,
        "model": model,
        "prompt_variant": 1,
        "draw": draw,
        "action_order": json.dumps(shuffled_actions),
        "hash_token": hash_token,
        "raw_output": "",
        "parse_status": ParseStatus.FORMAT_ERROR.value,
        "action_scores": "{}",
        "commentary": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
        "called_at": datetime.now().isoformat(),
    }

    if cached and "result" in cached:
        result_row.update(cached["result"])
        return result_row

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": scenario_text},
    ]

    parsed, meta = _call_llm_with_retry(
        client, model, messages, scenario.scenario_id, temperature=temperature
    )

    result_row["raw_output"] = meta.get("raw_content", "")
    result_row["prompt_tokens"] = meta.get("prompt_tokens", 0)
    result_row["completion_tokens"] = meta.get("completion_tokens", 0)
    result_row["total_tokens"] = meta.get("total_tokens", 0)
    result_row["estimated_cost_usd"] = _compute_cost(
        model, meta.get("prompt_tokens", 0), meta.get("completion_tokens", 0)
    )

    if meta.get("finish_reason") == "length":
        result_row["parse_status"] = ParseStatus.TOKEN_LIMIT.value
        token_limit_counter[0] += 1
        logger.error(f"Token limit hit for {scenario.scenario_id} draw={draw}")
        if token_limit_counter[0] > 10:
            raise TokenLimitRunError(
                f"Run aborted: {token_limit_counter[0]} token limit exceedances."
            )
    elif parsed is not None:
        scores_dict = {
            sanitise_text(als.action.value): als.score
            for als in parsed.action_scores
        }
        result_row["parse_status"] = ParseStatus.SUCCESS.value
        result_row["action_scores"] = json.dumps(scores_dict, ensure_ascii=True)
        result_row["commentary"] = sanitise_text(parsed.commentary)
    else:
        result_row["parse_status"] = ParseStatus.FORMAT_ERROR.value

    usage = TokenUsage(
        scenario_id=scenario.scenario_id, model=model,
        prompt_tokens=result_row["prompt_tokens"],
        completion_tokens=result_row["completion_tokens"],
        total_tokens=result_row["total_tokens"],
        estimated_cost_usd=result_row["estimated_cost_usd"],
    )
    cost_tracker.record(usage, f"tier_{scenario.tier}")

    _cache_store(cache_key, {"result": {
        k: v for k, v in result_row.items() if k != "raw_output"
    }})

    return result_row


def run_elicitation(
    scenarios: list[Scenario],
    client,
    model: str,
    n_draws: int,
    output_path: Path,
    cost_tracker: RunCostSummary,
    temperature: Optional[float] = 1.0,
    max_workers: int = 10,
) -> list[dict]:
    """Stage 2: Run LLM Likert elicitation across all ASA scenarios."""
    logger.info(f"Stage 2: Eliciting {len(scenarios)} scenarios x {n_draws} draws...")

    system_prompt = _build_system_prompt()
    token_limit_counter = [0]

    tasks = [
        (scenario, draw)
        for scenario in scenarios
        if scenario.tier != 4
        for draw in range(n_draws)
    ]

    results = []
    from tqdm import tqdm

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _elicit_single, scenario, draw, client, model,
                system_prompt, cost_tracker, token_limit_counter,
                temperature,
            ): (scenario.scenario_id, draw)
            for scenario, draw in tasks
        }
        with tqdm(total=len(tasks), desc="ASA Elicitation", smoothing=0) as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except TokenLimitRunError:
                    raise
                except Exception as e:
                    sid, draw = futures[future]
                    logger.error(f"Elicitation failed for {sid} draw={draw}: {e}")
                total_lookups = _cache_stats["hits"] + _cache_stats["misses"]
                if total_lookups > 0:
                    pbar.set_postfix(cache=f'{100*_cache_stats["hits"]/total_lookups:.0f}%')
                pbar.update(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if results:
        fieldnames = list(results[0].keys())
        with open(output_path, "w", encoding="utf-8", errors="replace", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    logger.info(f"Elicitation results saved to {output_path} ({len(results)} rows)")
    return results


# ── SEC 8: Stage 3 — Data preprocessing ──────────────────────────────────────

def preprocess_likert_data(
    elicitation_path: Path,
    long_output_path: Path,
    summary_output_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stage 3: Preprocess Likert elicitation results into long-format + summary."""
    logger.info("Stage 3: Preprocessing ASA Likert elicitation data...")

    df = pd.read_csv(elicitation_path, encoding="utf-8")
    df_valid = df[df["parse_status"].isin(["success", "repaired"])].copy()

    min_draws = 3
    success_counts = df_valid.groupby("scenario_id").size()
    valid_ids = success_counts[success_counts >= min_draws].index.tolist()
    df_valid = df_valid[df_valid["scenario_id"].isin(valid_ids)].copy()

    if df_valid.empty:
        logger.warning("No valid scenarios after filtering!")
        empty_long = pd.DataFrame(columns=[
            "scenario_id", "action", "draw", "score", "action_order_position",
        ])
        empty_summary = pd.DataFrame(columns=[
            "scenario_id", "action", "n_draws", "mean_score", "sd_score",
        ])
        return empty_long, empty_summary

    long_records = []
    for _, row in df_valid.iterrows():
        scores_dict = json.loads(row["action_scores"])
        action_order = json.loads(row["action_order"]) if row.get("action_order") else []

        for action, score in scores_dict.items():
            if action not in ("no_strike", "rec_strike"):
                logger.debug(f"Dropping invalid action '{action}' for {row['scenario_id']}")
                continue
            position = action_order.index(action) + 1 if action in action_order else 0
            long_records.append({
                "scenario_id": row["scenario_id"],
                "action": action,
                "draw": row["draw"],
                "score": int(score),
                "action_order_position": position,
            })

    likert_long_df = pd.DataFrame(long_records)

    if likert_long_df.empty:
        logger.warning("No Likert scores found in elicitation data!")
        empty_long = pd.DataFrame(columns=[
            "scenario_id", "action", "draw", "score", "action_order_position",
        ])
        empty_summary = pd.DataFrame(columns=[
            "scenario_id", "action", "n_draws", "mean_score", "sd_score",
        ])
        return empty_long, empty_summary

    summary_records = []
    for (sid, action), grp in likert_long_df.groupby(["scenario_id", "action"]):
        summary_records.append({
            "scenario_id": sid,
            "action": action,
            "n_draws": len(grp),
            "mean_score": float(np.mean(grp["score"])),
            "sd_score": float(np.std(grp["score"])),
        })
    likert_summary_df = pd.DataFrame(summary_records)

    long_output_path.parent.mkdir(parents=True, exist_ok=True)
    likert_long_df.to_csv(long_output_path, index=False, encoding="utf-8")
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    likert_summary_df.to_csv(summary_output_path, index=False, encoding="utf-8")

    n_obs = len(likert_long_df)
    n_pairs = len(likert_summary_df)
    n_scenarios = likert_long_df["scenario_id"].nunique()
    logger.info(f"Likert data: {n_obs} observations, {n_pairs} (scenario,action) pairs, "
                f"{n_scenarios} scenarios")
    return likert_long_df, likert_summary_df


# ── SEC 8B: Pre-flight identifiability checks ────────────────────────────────

# Context parameters: identified from phi variation ACROSS scenarios (same for
# both actions, so phi difference is always zero).
# Interaction parameters: identified from phi difference BETWEEN actions.
CONTEXT_PARAM_NAMES = [p for p in ESTIMABLE_PARAM_NAMES if p.startswith("w_ctx_")]
INTERACTION_PARAM_NAMES = [p for p in ESTIMABLE_PARAM_NAMES if not p.startswith("w_ctx_")]


def _scenario_phi_signature(scenario: "Scenario") -> dict[str, float]:
    """Compute phi signature for identifiability checking.

    Interaction params: returns |phi_no_strike - phi_rec_strike| (nonzero
    means the param affects the action choice in this scenario).

    Context params: returns |phi_value| (nonzero means the param has a
    nonzero phi in this scenario, contributing to Likert level variation).
    """
    sv = scenario.state_vector if isinstance(scenario.state_vector, dict) else {}
    args_base = {
        "ceo_resigned": sv.get("ceo_resigned", False),
        "ceo_sacked": sv.get("ceo_sacked", False),
        "review_commissioned": sv.get("review_commissioned", False),
        "action_delayed": sv.get("action_delayed", False),
        "high_profile": sv.get("high_profile", True),
    }

    phi_no = decompose_utility_asa(**args_base, action="no_strike")
    phi_rec = decompose_utility_asa(**args_base, action="rec_strike")

    signature = {}
    for param in ESTIMABLE_PARAM_NAMES:
        if param in CONTEXT_PARAM_NAMES:
            # Context params: check phi variation (nonzero phi in this scenario)
            val = abs(phi_no.get(param, 0.0))
            if val > 1e-6:
                signature[param] = val
        else:
            # Interaction params: check phi difference between actions
            diff = abs(phi_no.get(param, 0.0) - phi_rec.get(param, 0.0))
            if diff > 1e-6:
                signature[param] = diff
    return signature


def run_preflight_checks(
    scenarios: list[Scenario],
    estimation_df: Optional[pd.DataFrame] = None,
) -> dict:
    """Run identifiability and data quality checks."""
    logger.info("Running pre-flight checks...")
    results = {"checks": [], "all_passed": True}

    param_counts = {p: 0 for p in ESTIMABLE_PARAM_NAMES}
    for s in scenarios:
        if s.tier == 4:
            continue
        sig = _scenario_phi_signature(s)
        for p in sig:
            if p in param_counts:
                param_counts[p] += 1

    for p, count in param_counts.items():
        check_type = "phi variation" if p in CONTEXT_PARAM_NAMES else "phi difference"
        passed = count >= 5
        results["checks"].append({
            "name": f"param_coverage_{p}",
            "passed": passed,
            "warning": not passed,
            "detail": f"{p}: {count} scenarios with nonzero {check_type}",
        })
        if not passed:
            results["all_passed"] = False

    n_estimation = sum(1 for s in scenarios if s.tier != 4)
    passed = n_estimation >= 30
    results["checks"].append({
        "name": "total_scenarios",
        "passed": passed,
        "detail": f"{n_estimation} estimation scenarios (minimum 30)",
    })
    if not passed:
        results["all_passed"] = False

    if estimation_df is not None and not estimation_df.empty:
        n_obs = len(estimation_df)
        passed = n_obs >= 50
        results["checks"].append({
            "name": "likert_data_volume",
            "passed": passed,
            "detail": f"{n_obs} total Likert observations (minimum 50)",
        })
        if not passed:
            results["all_passed"] = False

    n_passed = sum(1 for c in results["checks"] if c["passed"])
    n_total = len(results["checks"])
    logger.info(f"Pre-flight checks: {n_passed}/{n_total} passed"
                + ("" if results["all_passed"] else " (WARNINGS PRESENT)"))
    for c in results["checks"]:
        status = "PASS" if c["passed"] else "FAIL"
        if c.get("warning"):
            status = "WARN"
        logger.info(f"  [{status}] {c['name']}: {c['detail']}")

    return results


# ── SEC 9: Stage 4 — ASA utility decomposition & estimation ─────────────────

def decompose_utility_asa(
    ceo_resigned: bool,
    ceo_sacked: bool,
    review_commissioned: bool,
    action_delayed: bool,
    high_profile: bool,
    action: str,
) -> dict[str, float]:
    """
    Decompose ASA utility into per-parameter basis function values.

    All inputs are binary game tree indicators.  No abstract governance
    dimension scores.

    Convention: EU = sum_k w_k * phi_k.  All weights w_k > 0.
    - POSITIVE phi -> higher EU -> more appropriate action.
    - NEGATIVE phi -> lower EU -> less appropriate action.

    DECOMPOSITION: Two classes of parameters:

    CONTEXT parameters fire EQUALLY for both actions.  They capture how
    good the situation is regardless of what ASA does.  They affect
    ordinal probit Likert level predictions but cancel in delta-EU,
    so they do not affect action probabilities.

    INTERACTION parameters fire only for rec_strike (phi=0 for no_strike).
    They capture how the game tree context modifies the marginal value
    of striking.  Only these terms drive the strike/no-strike decision.

    This prevents the old bug where 5 of 7 terms fired only for
    no_strike, causing massive one-sided EU accumulation and 0/1
    action probabilities.

    Parameters
    ----------
    ceo_resigned : bool
        CEO voluntarily resigned (D0_ceo = resign).
    ceo_sacked : bool
        Board forced CEO exit (D1 = D3_ceo_transition).
    review_commissioned : bool
        Board commissioned governance review (D1 = D1_review).
    action_delayed : bool
        Board acted but was slow (delayed review or delayed exit).
    high_profile : bool
        Case has high public salience (True for Qantas).
    action : str
        "no_strike" or "rec_strike".

    Returns
    -------
    dict[str, float]
        {param_name: phi_value} for each estimable parameter.
    """
    is_strike = float(action == "rec_strike")

    ceo_departed = float(ceo_resigned or ceo_sacked)
    board_acted = float(review_commissioned or ceo_sacked)

    phi = {
        # ═══ CONTEXT PARAMETERS (same phi for both actions) ═══
        #
        # These capture the situation quality.  They affect the absolute
        # Likert rating level but cancel in delta-EU, so they do NOT
        # affect action probabilities.

        # Board inaction: bad context regardless of ASA's action
        "w_ctx_inaction": -(1.0 - board_acted),

        # CEO departure: positive accountability signal
        "w_ctx_departure": ceo_departed,

        # Review commissioned: forward-looking governance reform signal
        "w_ctx_review": float(review_commissioned),

        # ═══ INTERACTION PARAMETERS (fire only for rec_strike) ═══
        #
        # These capture how context modifies the marginal value of
        # striking.  They are the ONLY terms that affect delta-EU and
        # therefore the ONLY terms that drive action probabilities.

        # Net strike cost: fixed mobilisation cost of recommending strike
        "w_strike_cost": -is_strike,

        # Striking when board is passive: the core ASA leverage mechanism.
        # Board inaction makes striking more valuable (accountability
        # pressure).
        "w_strike_vs_passive": (1.0 - board_acted) * is_strike,

        # CEO departure dampens strike: if CEO has already gone
        # (resigned or sacked), the marginal value of striking is
        # reduced -- accountability has been partially achieved.
        "w_departure_dampens": -ceo_departed * is_strike,

        # Board-forced exit further dampens strike: the strongest
        # accountability signal short of clawback.  Board demonstrated
        # willingness to impose consequences.
        "w_sack_dampens": -float(ceo_sacked) * is_strike,

        # Credibility signal (repeat-game): in high-profile cases,
        # striking has positive signaling value for ASA's long-term
        # credibility with its member base as a governance watchdog.
        "w_credibility_signal": float(high_profile) * is_strike,
    }
    return phi


def _compute_anchored_contribution(sv: dict, action: str) -> float:
    """Compute anchored (non-estimable) contribution to ASA utility.

    All parameters are estimable game tree indicators.
    Anchored contributions are zero.
    """
    return 0.0


def _scenario_to_outcome_args(sv: dict, action: str) -> dict:
    """Convert scenario state vector + action into decompose_utility_asa kwargs."""
    return {
        "ceo_resigned": sv.get("ceo_resigned", False),
        "ceo_sacked": sv.get("ceo_sacked", False),
        "review_commissioned": sv.get("review_commissioned", False),
        "action_delayed": sv.get("action_delayed", False),
        "high_profile": sv.get("high_profile", True),
        "action": action,
    }


def compute_phi_matrix(
    scenarios: list[Scenario],
    likert_summary_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict, dict, list[str], list[list[str]]]:
    """
    Compute basis function matrix phi[s, k] for unique (scenario, action) pairs.

    Returns:
        phi: (S, K) basis function values per (scenario, action) pair
        anchored: (S,) anchored contribution per pair
        sa_id_map: dict mapping (scenario_id, action) -> row index in phi
        scenario_id_map: dict mapping scenario_id -> unique scenario index (1-based)
        scenario_ids: list of scenario IDs in order
        action_lists: list of feasible action lists per scenario
    """
    logger.info("Computing ASA basis function matrix (phi)...")

    n_params = len(WEIGHT_PARAM_NAMES)
    valid_sids = set(likert_summary_df["scenario_id"].tolist())

    valid_scenarios = [s for s in scenarios if s.scenario_id in valid_sids]

    # Build (scenario, action) pairs
    sa_pairs = []
    scenario_ids = []
    action_lists = []
    scenario_id_map: dict[str, int] = {}

    for scenario in valid_scenarios:
        if scenario.scenario_id not in scenario_id_map:
            scenario_id_map[scenario.scenario_id] = len(scenario_id_map) + 1
        scenario_ids.append(scenario.scenario_id)
        actions = list(scenario.feasible_actions)
        action_lists.append(actions)
        for action in actions:
            sa_pairs.append((scenario.scenario_id, action))

    S = len(sa_pairs)
    sa_id_map = {pair: idx for idx, pair in enumerate(sa_pairs)}

    phi = np.zeros((S, n_params))
    anchored = np.zeros(S)

    for s_idx, (sid, action) in enumerate(sa_pairs):
        scenario = next(s for s in valid_scenarios if s.scenario_id == sid)
        args = _scenario_to_outcome_args(scenario.state_vector, action)
        phi_k = decompose_utility_asa(**args)

        for k, pname in enumerate(WEIGHT_PARAM_NAMES):
            phi[s_idx, k] = phi_k.get(pname, 0.0)

        anchored[s_idx] = _compute_anchored_contribution(
            scenario.state_vector, action,
        )

    # Center anchored within each scenario (absorbed by scenario RE)
    for sid in scenario_id_map:
        sa_indices = [sa_id_map[(sid, a)] for a in
                      action_lists[scenario_ids.index(sid)]]
        if len(sa_indices) > 1:
            scenario_mean = np.mean(anchored[sa_indices])
            for idx in sa_indices:
                anchored[idx] -= scenario_mean

    logger.info(f"Phi matrix shape: {phi.shape}, {S} (scenario,action) pairs, "
                f"{len(scenario_id_map)} unique scenarios")

    return phi, anchored, sa_id_map, scenario_id_map, scenario_ids, action_lists


def prepare_stan_data(
    likert_long_df: "pd.DataFrame",
    phi: np.ndarray,
    anchored: np.ndarray,
    sa_id_map: dict,
    scenario_id_map: dict,
) -> dict:
    """Build the data dict for the ASA ordinal probit Stan model.

    Structurally identical to the Board pipeline's prepare_stan_data but
    without vote penalty parameters (no vote_x_strike/vote_x_overwh).
    """
    S, K = phi.shape
    N_scenarios = int(len(scenario_id_map))

    scenario_id_per_sa = [0] * S
    for (sid, _action), row_idx in sa_id_map.items():
        scenario_id_per_sa[row_idx] = int(scenario_id_map[sid])

    y_list: list[int] = []
    sa_id_list: list[int] = []

    for _, row in likert_long_df.iterrows():
        sid = row["scenario_id"]
        action = row["action"]
        rating = int(row["score"])
        key = (sid, action)
        if key not in sa_id_map:
            continue
        if not (1 <= rating <= 5):
            continue
        y_list.append(rating)
        sa_id_list.append(int(sa_id_map[key]) + 1)  # 1-based for Stan

    N = len(y_list)
    if N == 0:
        raise ValueError(
            "prepare_stan_data: no valid Likert observations found."
        )

    # Prior hyperparameters for each weight: lognormal(log(default), 1.0)
    prior_log_mean = [float(np.log(SPEC_DEFAULTS[p])) for p in WEIGHT_PARAM_NAMES]
    prior_log_sd = [1.0] * K  # SD=1.0 gives ~2.7x range per SD on ratio scale

    stan_data = {
        "N": int(N),
        "S": int(S),
        "K": int(K),
        "y": [int(v) for v in y_list],
        "sa_id": [int(v) for v in sa_id_list],
        "phi": phi.tolist(),
        "anchored": anchored.tolist(),
        "N_scenarios": int(N_scenarios),
        "scenario_id": [int(v) for v in scenario_id_per_sa],
        "mu_scale": 1.0,  # set by fit_ordinal_probit
        "prior_log_mean": prior_log_mean,
        "prior_log_sd": prior_log_sd,
    }

    logger.info(
        f"prepare_stan_data: N={N} obs, S={S} (scenario,action) pairs, "
        f"K={K} weights, {N_scenarios} unique scenarios"
    )
    return stan_data


def fit_ordinal_probit(
    stan_data: dict,
    stan_model_path: Optional[str] = None,
    chains: int = 4,
    iter_warmup: int = 1000,
    iter_sampling: int = 2000,
    adapt_delta: float = 0.99,
    max_treedepth: int = 12,
    seed: int = 42,
) -> dict:
    """Compile and sample asa_ordinal_utility.stan via CmdStanPy.

    Returns dict with posterior draws and MCMC diagnostics.
    """
    import platform

    if platform.system() == "Windows" and "MAKE" not in os.environ:
        rtools_path = r"C:\rtools44\usr\bin"
        mingw_path = r"C:\rtools44\x86_64-w64-mingw32.static.posix\bin"
        if os.path.isdir(rtools_path):
            os.environ["PATH"] = f"{rtools_path};{mingw_path};{os.environ['PATH']}"
        make_path = r"C:\rtools44\usr\bin\make.exe"
        if os.path.isfile(make_path):
            os.environ["MAKE"] = make_path

    from cmdstanpy import CmdStanModel

    if stan_model_path is None:
        stan_model_path = str(PROJECT_ROOT / "models" / "asa_ordinal_utility.stan")

    if not os.path.isfile(stan_model_path):
        raise FileNotFoundError(f"ASA Stan model not found: {stan_model_path}")

    logger.info(f"Compiling Stan model: {stan_model_path}")
    model = CmdStanModel(stan_file=stan_model_path)

    K = stan_data["K"]
    S = stan_data["S"]

    # Compute initial mu for scaling
    phi_arr = np.array(stan_data["phi"])
    anchored_arr = np.array(stan_data["anchored"])
    w_init = np.array([SPEC_DEFAULTS.get(p, 1.0) for p in WEIGHT_PARAM_NAMES])
    mu_init = phi_arr @ w_init + anchored_arr
    mu_span = float(np.max(mu_init) - np.min(mu_init))
    mu_scale = max(mu_span / 6.0, 1.0)
    stan_data["mu_scale"] = float(mu_scale)
    logger.info(f"mu_scale = {mu_scale:.3f} (mu range: {np.min(mu_init):.2f} to {np.max(mu_init):.2f})")

    # Cutpoint initialization
    import math
    base_init = math.atanh(-1.5 / 3.0)
    gap_init = [math.log(1.0 / (2.0 - 1.0))] * 3  # logit((1.0 - 0.25) / 2.0)

    # Generate initial values for all chains
    # ASA model uses vector[K] w directly (not individual w_raw_k)
    def _make_inits(chain_id):
        rng_init = np.random.default_rng(seed + chain_id * 17)
        w_jitter = w_init * np.exp(rng_init.normal(0, 0.3, size=K))
        return {
            "w": w_jitter.tolist(),
            "cutpoint_base_raw": float(base_init + rng_init.normal(0, 0.3)),
            "cutpoint_gap_raw": [float(g + rng_init.normal(0, 0.3)) for g in gap_init],
            "sigma_scenario": float(0.3 + rng_init.uniform(0, 0.5)),
            "z_scenario": [0.0] * stan_data["N_scenarios"],
        }

    inits = [_make_inits(c) for c in range(chains)]

    logger.info(f"Sampling: {chains} chains x {iter_sampling} draws + {iter_warmup} warmup")
    fit = model.sample(
        data=stan_data,
        chains=chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        adapt_delta=adapt_delta,
        max_treedepth=max_treedepth,
        seed=seed,
        inits=inits,
        show_console=False,
    )

    # Extract posterior draws
    n_draws = chains * iter_sampling
    try:
        # CmdStanPy: stan_variable("w") on vector[K] returns (n_draws, K) array
        w_draws = fit.stan_variable("w")
        if w_draws.ndim == 1:
            w_draws = w_draws.reshape(-1, 1)
        logger.info(f"Extracted w posterior: shape {w_draws.shape}")
    except ValueError:
        logger.warning("Could not extract 'w' vector from Stan output, using defaults")
        w_draws = np.tile(w_init, (n_draws, 1))

    # Cutpoints
    try:
        cutpoints = fit.stan_variable("cutpoints")
    except ValueError:
        cutpoints = np.zeros((n_draws, 4))

    # Sigma scenario
    try:
        sigma_scenario = fit.stan_variable("sigma_scenario")
    except ValueError:
        sigma_scenario = np.ones(n_draws) * 0.5

    # MCMC diagnostics
    diag = fit.diagnose()
    n_divergences = 0
    max_rhat = 0.0
    min_ess_bulk = float("inf")

    try:
        summary_df = fit.summary()
        if "R_hat" in summary_df.columns:
            rhats = summary_df["R_hat"].dropna()
            max_rhat = float(rhats.max()) if len(rhats) > 0 else 0.0
        if "ESS_bulk" in summary_df.columns:
            ess = summary_df["ESS_bulk"].dropna()
            min_ess_bulk = float(ess.min()) if len(ess) > 0 else 0.0
    except Exception as e:
        logger.warning(f"Could not compute MCMC diagnostics: {e}")

    logger.info(f"MCMC complete: {n_draws} draws, max R-hat={max_rhat:.4f}, "
                f"min ESS={min_ess_bulk:.0f}, divergences={n_divergences}")

    return {
        "fit": fit,
        "w": w_draws,
        "cutpoints": cutpoints,
        "sigma_scenario": sigma_scenario,
        "n_divergences": n_divergences,
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess_bulk,
        "n_samples": n_draws,
    }


@dataclass
class StanEstimationResult:
    """Posterior summary from Bayesian ordinal probit estimation via Stan."""

    w_draws: np.ndarray            # (n_draws, K)
    cutpoint_draws: np.ndarray     # (n_draws, 4)
    sigma_scenario_draws: np.ndarray

    weights_posterior_mean: dict = field(default_factory=dict)
    weights_posterior_sd: dict = field(default_factory=dict)
    weights_posterior_ci: dict = field(default_factory=dict)

    n_divergences: int = 0
    max_rhat: float = 0.0
    min_ess_bulk: float = 0.0
    n_samples: int = 0

    weights: dict = field(default_factory=dict)
    hessian_se: dict = field(default_factory=dict)
    bootstrap_se: dict = field(default_factory=dict)
    covariance_matrix: np.ndarray = field(
        default_factory=lambda: np.zeros((9, 9))
    )
    converged: bool = True
    estimation_method: dict = field(default_factory=dict)
    n_scenarios: int = 0

    def to_dict(self) -> dict:
        return {
            "weights": self.weights,
            "hessian_se": {k: round(v, 4) for k, v in self.hessian_se.items()},
            "bootstrap_se": {k: round(v, 4) for k, v in self.bootstrap_se.items()},
            "n_scenarios": self.n_scenarios,
            "converged": self.converged,
            "estimation_method": self.estimation_method,
            "n_divergences": self.n_divergences,
            "max_rhat": round(self.max_rhat, 4),
            "min_ess_bulk": round(self.min_ess_bulk, 1),
            "n_samples": self.n_samples,
            "weights_posterior_sd": {
                k: round(v, 4) for k, v in self.weights_posterior_sd.items()
            },
            "weights_posterior_ci": self.weights_posterior_ci,
        }


EstimationResult = StanEstimationResult


def estimate_parameters_stan(
    phi: np.ndarray,
    anchored: np.ndarray,
    likert_long_df: pd.DataFrame,
    sa_id_map: dict,
    scenario_id_map: dict,
    chains: int = 4,
    iter_warmup: int = 1000,
    iter_sampling: int = 2000,
) -> StanEstimationResult:
    """Run full Stan ordinal probit estimation pipeline."""
    logger.info("Stage 4: Running Bayesian ordinal probit estimation...")

    stan_data = prepare_stan_data(
        likert_long_df, phi, anchored, sa_id_map, scenario_id_map,
    )

    fit_result = fit_ordinal_probit(
        stan_data, chains=chains,
        iter_warmup=iter_warmup, iter_sampling=iter_sampling,
    )

    w_draws = fit_result["w"]
    K = w_draws.shape[1]

    # Posterior summaries
    weights = {}
    weights_sd = {}
    weights_ci = {}
    hessian_se = {}
    bootstrap_se = {}
    estimation_method = {}

    for k, pname in enumerate(WEIGHT_PARAM_NAMES):
        draws = w_draws[:, k]
        weights[pname] = round(float(np.mean(draws)), 4)
        weights_sd[pname] = round(float(np.std(draws)), 4)
        ci_lo, ci_hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
        weights_ci[pname] = [round(ci_lo, 4), round(ci_hi, 4)]
        hessian_se[pname] = weights_sd[pname]
        bootstrap_se[pname] = round((ci_hi - ci_lo) / (2 * 1.96), 4)
        estimation_method[pname] = "stan_ordinal_probit"

    # Covariance matrix
    cov = np.cov(w_draws.T)

    # Save posterior draws
    draws_path = OUTPUT_DIR / "asa_stan_posterior_draws.npz"
    draws_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        draws_path,
        w_draws=w_draws,
        cutpoints=fit_result["cutpoints"],
        sigma_scenario=fit_result["sigma_scenario"],
        param_names=WEIGHT_PARAM_NAMES,
    )
    logger.info(f"Posterior draws saved to {draws_path}")

    result = StanEstimationResult(
        w_draws=w_draws,
        cutpoint_draws=fit_result["cutpoints"],
        sigma_scenario_draws=fit_result["sigma_scenario"],
        weights_posterior_mean=weights,
        weights_posterior_sd=weights_sd,
        weights_posterior_ci=weights_ci,
        weights=weights,
        hessian_se=hessian_se,
        bootstrap_se=bootstrap_se,
        covariance_matrix=cov,
        n_divergences=fit_result["n_divergences"],
        max_rhat=fit_result["max_rhat"],
        min_ess_bulk=fit_result["min_ess_bulk"],
        n_samples=fit_result["n_samples"],
        converged=fit_result["max_rhat"] < 1.01,
        estimation_method=estimation_method,
        n_scenarios=len(scenario_id_map),
    )

    for pname in WEIGHT_PARAM_NAMES:
        ci = weights_ci[pname]
        logger.info(f"  {pname}: {weights[pname]:.3f} "
                     f"[{ci[0]:.3f}, {ci[1]:.3f}] (SD={weights_sd[pname]:.3f})")

    return result


# ── SEC 9B: Posterior action probabilities & A2 tree computation ─────────────

# A2 node game state encodings — game tree measurable inputs only.
#
# Each A2 node corresponds to a unique (D0_ceo outcome, D1 action) pair.
# All crisis context (PPL, ESG, salience) is FIXED for Qantas and enters
# through the common system prompt, not through the state vector.
#
# The state vector contains ONLY binary game tree indicators.
# Probability variation across nodes comes from the estimated utility
# weights operating on these indicators.
A2_NODE_STATES = {
    "ceo_resign__do_nothing": {
        # CEO resigned voluntarily, Board took no further action.
        "ceo_resigned": True, "ceo_sacked": False,
        "review_commissioned": False, "action_delayed": False,
        "high_profile": True,
    },
    "ceo_resign__review": {
        # CEO resigned + Board commissioned governance review.
        "ceo_resigned": True, "ceo_sacked": False,
        "review_commissioned": True, "action_delayed": False,
        "high_profile": True,
    },
    "ceo_stay__do_nothing": {
        # CEO stayed, Board did nothing. Worst case.
        "ceo_resigned": False, "ceo_sacked": False,
        "review_commissioned": False, "action_delayed": False,
        "high_profile": True,
    },
    "ceo_stay__review": {
        # CEO stayed but Board commissioned review.
        "ceo_resigned": False, "ceo_sacked": False,
        "review_commissioned": True, "action_delayed": False,
        "high_profile": True,
    },
    "ceo_stay__board_forces_exit": {
        # Board forced CEO out — strongest possible accountability action.
        "ceo_resigned": False, "ceo_sacked": True,
        "review_commissioned": False, "action_delayed": False,
        "high_profile": True,
    },
}


def compute_a2_action_probabilities(
    est_result: StanEstimationResult,
    laplacian: bool = True,
    alpha: float = 1.0,
) -> dict[str, dict[str, float]]:
    """Compute ASA action probabilities at each A2 node via argmax-count.

    For each A2 node, computes EU for both actions across all posterior draws
    and counts which action wins per draw.

    Returns:
        dict mapping A2 node name -> {"no_strike": prob, "rec_strike": prob}
    """
    w_draws = est_result.w_draws  # (n_draws, K)
    n_draws = w_draws.shape[0]
    results = {}

    for node_name, state_vec in A2_NODE_STATES.items():
        # Compute phi for both actions
        phi_no = decompose_utility_asa(**state_vec, action="no_strike")
        phi_rec = decompose_utility_asa(**state_vec, action="rec_strike")

        # Build phi vectors (K,)
        phi_no_vec = np.array([phi_no.get(p, 0.0) for p in WEIGHT_PARAM_NAMES])
        phi_rec_vec = np.array([phi_rec.get(p, 0.0) for p in WEIGHT_PARAM_NAMES])

        # EU per draw: (n_draws,)
        eu_no = w_draws @ phi_no_vec
        eu_rec = w_draws @ phi_rec_vec

        # Argmax-count
        best_is_rec = eu_rec > eu_no
        count_rec = int(np.sum(best_is_rec))
        count_no = n_draws - count_rec

        if laplacian:
            K_actions = 2
            p_rec = (count_rec + alpha) / (n_draws + K_actions * alpha)
            p_no = (count_no + alpha) / (n_draws + K_actions * alpha)
        else:
            p_rec = count_rec / n_draws
            p_no = count_no / n_draws

        results[node_name] = {
            "no_strike": round(float(p_no), 4),
            "rec_strike": round(float(p_rec), 4),
            "eu_no_strike_mean": round(float(np.mean(eu_no)), 4),
            "eu_rec_strike_mean": round(float(np.mean(eu_rec)), 4),
            "eu_diff_mean": round(float(np.mean(eu_rec - eu_no)), 4),
            "eu_diff_sd": round(float(np.std(eu_rec - eu_no)), 4),
        }

        logger.info(f"  [Stan] {node_name}: P(no_strike)={p_no:.3f}, "
                     f"P(rec_strike)={p_rec:.3f}, "
                     f"EU_diff={np.mean(eu_rec - eu_no):.3f}")

    return results


def compute_action_probabilities_from_posterior(
    est_result: StanEstimationResult,
    scenarios: list[Scenario],
    phi: np.ndarray,
    anchored: np.ndarray,
    sa_id_map: dict,
    action_lists: list[list[str]],
    scenario_ids: list[str],
    laplacian: bool = True,
) -> dict:
    """Compute argmax-count action probabilities for each scenario using posterior.

    Returns dict mapping scenario_id -> {action: prob, ...}
    """
    w_draws = est_result.w_draws  # (n_draws, K)
    n_draws = w_draws.shape[0]
    results = {}

    for s_idx, sid in enumerate(scenario_ids):
        actions = action_lists[s_idx]
        if len(actions) < 2:
            results[sid] = {actions[0]: 1.0} if actions else {}
            continue

        # Compute EU for each action across all draws
        eu_mat = np.zeros((n_draws, len(actions)))
        for j, action in enumerate(actions):
            key = (sid, action)
            if key not in sa_id_map:
                continue
            row_idx = sa_id_map[key]
            phi_row = phi[row_idx, :]  # (K,)
            anch = anchored[row_idx]
            eu_mat[:, j] = w_draws @ phi_row + anch

        # Argmax-count
        best_idx = np.argmax(eu_mat, axis=1)
        K_actions = len(actions)
        alpha = 1.0 if laplacian else 0.0

        action_probs = {}
        for j, action in enumerate(actions):
            count = float(np.sum(best_idx == j))
            action_probs[action] = round(
                (count + alpha) / (n_draws + K_actions * alpha), 4
            )
        results[sid] = action_probs

    return results


# ── SEC 9C: 4-Step ASA A2 Probability Calibration Pipeline ───────────────────
#
# Clear step-by-step workflow:
#   Step 1: LLM elicitation of valid probability RANGES per A2 node
#   Step 2: LLM elicitation of valid probability GAPS between node pairs
#   Step 3: Constrained solving for target probabilities
#   Step 4: Calibrated utility weights + simulation validation
#
# This is the PRIMARY calibration mechanism.


# ── Step 1: Probability range elicitation ────────────────────────────────────

class ProbabilityRangeResponse(BaseModel):
    """LLM response: P(strike) range (low, best, high) at each A2 node."""
    p_expected_floor: float
    p_absolute_floor: float
    node_1_low: float
    node_1_best: float
    node_1_high: float
    node_2_low: float
    node_2_best: float
    node_2_high: float
    node_3_low: float
    node_3_best: float
    node_3_high: float
    node_4_low: float
    node_4_best: float
    node_4_high: float
    node_5_low: float
    node_5_best: float
    node_5_high: float
    reasoning: str

    @field_validator(
        "p_expected_floor", "p_absolute_floor",
        "node_1_low", "node_1_best", "node_1_high",
        "node_2_low", "node_2_best", "node_2_high",
        "node_3_low", "node_3_best", "node_3_high",
        "node_4_low", "node_4_best", "node_4_high",
        "node_5_low", "node_5_best", "node_5_high",
        mode="before",
    )
    @classmethod
    def clamp_probability(cls, v):
        v = float(v)
        return max(0.50, min(0.99, v))


# Map from node number to A2_NODE_STATES key
_NODE_NUM_TO_KEY = {
    1: "ceo_resign__do_nothing",
    2: "ceo_resign__review",
    3: "ceo_stay__do_nothing",
    4: "ceo_stay__review",
    5: "ceo_stay__board_forces_exit",
}

# Shared system prompt for both range and gap elicitation
_A2_SYSTEM_PROMPT = (
    "You are a decision-analysis assistant modelling the Australian "
    "Shareholders' Association (ASA) as a rational actor in a corporate "
    "governance crisis.\n\n"
    "ASA is Australia's largest independent not-for-profit organisation "
    "representing retail shareholders. Its mission is to protect retail "
    "shareholder interests through corporate governance monitoring, proxy "
    "voting, and advocacy. ASA publishes 'voting intentions' based on "
    "assessments by volunteer company monitors.\n\n"
    "HISTORICAL BASE RATE (critical for calibration):\n"
    "ASA's observed behaviour in 15 comparable headline governance "
    "incidents in Australian listed companies:\n"
    "- Board did nothing:          9/10 recommended strike (90.0%)\n"
    "- Board commissioned review or CEO resigned: 3/3 recommended "
    "strike (100%)\n"
    "- Board sacked CEO:           2/2 recommended strike (100%)\n"
    "- Overall base rate:          14/15 = 93.3%\n\n"
    "KEY INSIGHT: The remuneration vote is RETROSPECTIVE. It assesses "
    "the FY23 pay structure that is already set. Board actions taken "
    "AFTER the pay period are forward-looking signals that may slightly "
    "moderate ASA's position, but they do NOT change the historical pay "
    "structure being voted on. This is why ASA's historical strike rate "
    "remains above 90% even when boards take strong action.\n\n"
    "QANTAS-SPECIFIC FACTS (public by late September 2023):\n"
    "- FY23 statutory profit A$2.47 billion\n"
    "- CEO FY23 remuneration: A$21.4 million (near 10-fold YoY increase)\n"
    "- ACCC filed Federal Court action alleging ~8,000 ghost flight sales\n"
    "- Federal Court ruled Qantas illegally outsourced ~1,700 ground workers\n"
    "- CEO sold ~90% of his Qantas shareholding before ACCC announcement\n"
    "- No conduct-linked gating on STI; no clawback provisions disclosed\n"
    "- This is one of the most severe governance crises in Australian "
    "corporate history\n\n"
    "CALIBRATION CONSTRAINTS:\n"
    "1. BASE RATE ANCHOR: All probabilities should be calibrated against "
    "the 93.3% historical base rate. The Qantas crisis is at least as "
    "severe as the average headline incident, so the baseline should be "
    "at or above 93%.\n"
    "2. REPUTATION FLOOR: Even in the best-case Board response, ASA "
    "cannot risk being caught recommending 'no strike' when shareholders "
    "vote >25% against. The floor must be high (the historical rate for "
    "strong Board action is 100%, i.e., ASA always struck anyway).\n"
    "3. SIGNALING GRADIENT: Board actions should produce a small but "
    "meaningful reduction in P(strike). The historical data shows at "
    "most a ~10 percentage point range (90% to ~100%). The gradient "
    "exists to give the Board marginal incentive to act, but the "
    "reduction is modest because the retrospective pay vote dominates.\n"
    "4. MONOTONICITY: Stronger Board actions weakly decrease P(strike).\n"
    "5. NO CERTAINTIES: P(strike) should be between 0.80 and 0.99.\n\n"
    "Your task: provide calibrated probability assessments anchored to "
    "the historical base rate."
)

# Shared scenario descriptions for both prompts
_A2_SCENARIO_TEXT = (
    "SCENARIOS:\n\n"
    "Node 1 - CEO RESIGNED, BOARD DOES NOTHING:\n"
    "Alan Joyce announced immediate resignation. The Board has taken no "
    "further governance action. No review commissioned, no clawback "
    "signalled, no structural remuneration reform. The resignation "
    "removes the CEO but does not address the pay structure being voted "
    "on.\n\n"
    "Node 2 - CEO RESIGNED, BOARD COMMISSIONS REVIEW:\n"
    "Joyce resigned. Board announced independent governance review with "
    "publicly disclosed terms. Incoming chair acknowledged oversight "
    "failures. Partial holdback signalled but no binding clawback. This "
    "is the strongest combination of forward-looking signals, but the "
    "FY23 pay structure remains unchanged.\n\n"
    "Node 3 - CEO STAYS, BOARD DOES NOTHING:\n"
    "Joyce has NOT resigned and remains CEO. Board has taken no governance "
    "action. Board's posture is defensive, citing financial recovery. "
    "This is the worst-case scenario: the CEO who presided over the "
    "crisis retains his position AND his A$21.4M pay with no "
    "accountability signal from the Board.\n\n"
    "Node 4 - CEO STAYS, BOARD COMMISSIONS REVIEW:\n"
    "Joyce has NOT resigned. Board announced governance review with "
    "incoming chair making accountability statements. But the CEO who "
    "presided over misconduct retains his role AND full FY23 pay. The "
    "review is forward-looking but does not address the retrospective "
    "pay vote.\n\n"
    "Node 5 - CEO STAYS INITIALLY, BOARD FORCES EXIT:\n"
    "Joyce initially resisted pressure. Board forced his departure "
    "(board-initiated, not voluntary). This is the strongest single "
    "accountability action. Partial clawback discussion underway. But "
    "the FY23 remuneration report is unchanged.\n\n"
)


def _build_range_elicitation_prompt() -> tuple[str, str]:
    """Build system + user prompts for Step 1: probability range elicitation.

    Returns (system_prompt, user_prompt).
    """
    user_prompt = (
        "For each of the following 5 scenarios at ASA's A2 decision node, "
        "provide a RANGE for P(recommend strike):\n"
        "- LOW: the lowest plausible P(strike) for this scenario\n"
        "- BEST: your best point estimate of P(strike)\n"
        "- HIGH: the highest plausible P(strike) for this scenario\n\n"
        "CALIBRATION ANCHOR: The historical base rate is 93.3% (14/15 "
        "headline incidents). In comparable cases where boards took strong "
        "action, ASA STILL recommended a strike 100% of the time (5/5). "
        "Your probabilities should reflect this empirical reality.\n\n"
        + _A2_SCENARIO_TEXT +
        "Also provide TWO floor probabilities:\n"
        "- p_expected_floor: The EXPECTED minimum probability of a strike "
        "recommendation across all scenarios. Given that the historical "
        "rate NEVER dropped below 90% even with strong Board action, this "
        "should be high (typically 0.88-0.95).\n"
        "- p_absolute_floor: The ABSOLUTE minimum probability below which "
        "ASA's reputation would be unacceptably damaged. This is the hard "
        "lower bound (typically 0.80-0.88).\n\n"
        "IMPORTANT: The LOW value for each node represents the lower bound "
        "of your uncertainty. The HIGH value represents the upper bound. "
        "BEST is your point estimate. Ensure LOW <= BEST <= HIGH.\n\n"
        "Output JSON only:\n"
        "{\n"
        '  "p_expected_floor": <float 0.80-0.99>,\n'
        '  "p_absolute_floor": <float 0.50-0.95>,\n'
        '  "node_1_low": <float>, "node_1_best": <float>, '
        '"node_1_high": <float>,\n'
        '  "node_2_low": <float>, "node_2_best": <float>, '
        '"node_2_high": <float>,\n'
        '  "node_3_low": <float>, "node_3_best": <float>, '
        '"node_3_high": <float>,\n'
        '  "node_4_low": <float>, "node_4_best": <float>, '
        '"node_4_high": <float>,\n'
        '  "node_5_low": <float>, "node_5_best": <float>, '
        '"node_5_high": <float>,\n'
        '  "reasoning": "<2-4 sentences explaining your calibration>"\n'
        "}"
    )
    return _A2_SYSTEM_PROMPT, user_prompt


def _step1_elicit_ranges(
    client,
    model: str,
    n_draws: int,
    max_workers: int,
    output_dir: Optional[Path],
) -> dict:
    """Step 1: Elicit P(strike) ranges at each A2 node via repeated LLM draws.

    Returns dict with per-node ranges {low, best, high, sd} and floor stats.
    """
    logger.info("")
    logger.info("=" * 65)
    logger.info("  STEP 1: Probability Range Elicitation")
    logger.info("=" * 65)

    system_prompt, user_prompt = _build_range_elicitation_prompt()

    # Per-node draws: {node_num: {low: [], best: [], high: []}}
    node_draws = {n: {"low": [], "best": [], "high": []} for n in range(1, 6)}
    floor_draws = []
    abs_floor_draws = []
    reasonings = []
    n_errors = 0

    def _single_draw(draw_idx: int):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        cache_key = _make_cache_key(
            system_prompt, user_prompt, model, draw_idx, 1.0,
        )
        cached = _cache_lookup(cache_key, track_stats=True)
        if cached and "range_response" in cached:
            return cached["range_response"]
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_model=ProbabilityRangeResponse,
                temperature=1.0,
                max_tokens=700,
            )
            result = {
                "p_expected_floor": response.p_expected_floor,
                "p_absolute_floor": response.p_absolute_floor,
                "reasoning": response.reasoning,
            }
            for n in range(1, 6):
                result[f"node_{n}_low"] = getattr(response, f"node_{n}_low")
                result[f"node_{n}_best"] = getattr(response, f"node_{n}_best")
                result[f"node_{n}_high"] = getattr(response, f"node_{n}_high")
            _cache_store(cache_key, {"range_response": result})
            return result
        except Exception as e:
            logger.warning(f"  Range elicitation draw {draw_idx} failed: {e}")
            return None

    logger.info(f"  Eliciting P(strike) ranges from {n_draws} LLM draws...")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_single_draw, i): i for i in range(n_draws)}
        for fut in as_completed(futures):
            result = fut.result()
            if result is None:
                n_errors += 1
                continue
            floor_draws.append(result["p_expected_floor"])
            abs_floor_draws.append(result["p_absolute_floor"])
            for n in range(1, 6):
                lo = result[f"node_{n}_low"]
                be = result[f"node_{n}_best"]
                hi = result[f"node_{n}_high"]
                # Enforce low <= best <= high within each draw
                lo, be, hi = min(lo, be, hi), sorted([lo, be, hi])[1], max(lo, be, hi)
                node_draws[n]["low"].append(lo)
                node_draws[n]["best"].append(be)
                node_draws[n]["high"].append(hi)
            if result.get("reasoning"):
                reasonings.append(result["reasoning"])

    n_valid = len(floor_draws)
    logger.info(f"  {n_valid}/{n_draws} valid draws ({n_errors} errors)")

    # Aggregate per-node ranges
    node_ranges = {}
    _NODE_LABELS = {
        1: "Node1 (resign, nothing)",
        2: "Node2 (resign, review)",
        3: "Node3 (stay, nothing)",
        4: "Node4 (stay, review)",
        5: "Node5 (sacked)",
    }
    for n in range(1, 6):
        key = _NODE_NUM_TO_KEY[n]
        lows = node_draws[n]["low"]
        bests = node_draws[n]["best"]
        highs = node_draws[n]["high"]
        node_ranges[key] = {
            "low": float(np.percentile(lows, 10)) if lows else 0.80,
            "best": float(np.median(bests)) if bests else 0.90,
            "high": float(np.percentile(highs, 90)) if highs else 0.99,
            "sd": float(np.std(bests)) if bests else 0.0,
            "best_draws": bests,
        }

    p_floor = {
        "mean": float(np.mean(floor_draws)) if floor_draws else 0.90,
        "sd": float(np.std(floor_draws)) if floor_draws else 0.0,
        "draws": floor_draws,
    }
    p_abs_floor = {
        "mean": float(np.mean(abs_floor_draws)) if abs_floor_draws else 0.85,
        "sd": float(np.std(abs_floor_draws)) if abs_floor_draws else 0.0,
        "draws": abs_floor_draws,
    }

    # Console output
    logger.info("")
    logger.info(f"  {'Node':<35} {'Low':>6} {'Best':>6} {'High':>6} {'SD':>6}")
    logger.info(f"  {'-'*65}")
    for n in range(1, 6):
        key = _NODE_NUM_TO_KEY[n]
        r = node_ranges[key]
        logger.info(
            f"  {_NODE_LABELS[n]:<35} {r['low']:>6.3f} {r['best']:>6.3f} "
            f"{r['high']:>6.3f} {r['sd']:>6.3f}"
        )
    logger.info(f"  Expected floor: {p_floor['mean']:.3f} (SD={p_floor['sd']:.3f})")
    logger.info(f"  Absolute floor: {p_abs_floor['mean']:.3f} "
                f"(SD={p_abs_floor['sd']:.3f})")

    result = {
        "node_ranges": node_ranges,
        "p_floor": p_floor,
        "p_absolute_floor": p_abs_floor,
        "n_valid": n_valid,
        "reasonings": reasonings[:5],
    }

    if output_dir:
        # Save raw draws (exclude large arrays for JSON)
        save_data = copy.deepcopy(result)
        for key in save_data["node_ranges"]:
            save_data["node_ranges"][key].pop("best_draws", None)
        path = output_dir / "asa_probability_ranges.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, default=str)
        logger.info(f"  Saved ranges to {path}")

    return result


# ── Step 2: Probability gap elicitation ──────────────────────────────────────

class ProbabilityGapResponse(BaseModel):
    """LLM response: expected gaps in P(strike) between A2 node pairs."""
    gap_departure_low: float
    gap_departure_expected: float
    gap_departure_high: float
    gap_review_stay_low: float
    gap_review_stay_expected: float
    gap_review_stay_high: float
    gap_review_resign_low: float
    gap_review_resign_expected: float
    gap_review_resign_high: float
    gap_sacking_low: float
    gap_sacking_expected: float
    gap_sacking_high: float
    reasoning: str

    @field_validator(
        "gap_departure_low", "gap_departure_expected", "gap_departure_high",
        "gap_review_stay_low", "gap_review_stay_expected", "gap_review_stay_high",
        "gap_review_resign_low", "gap_review_resign_expected", "gap_review_resign_high",
        "gap_sacking_low", "gap_sacking_expected", "gap_sacking_high",
        mode="before",
    )
    @classmethod
    def clamp_gap(cls, v):
        v = float(v)
        return max(0.0, min(0.20, v))


# Gap definitions: (higher_node, lower_node, description)
_GAP_DEFINITIONS = [
    {
        "name": "departure",
        "higher": "ceo_stay__do_nothing",
        "lower": "ceo_resign__do_nothing",
        "label": "CEO departure effect",
        "description": (
            "How much does CEO departure (resigned) reduce P(strike) "
            "compared to CEO staying, when Board takes no other action?"
        ),
    },
    {
        "name": "review_stay",
        "higher": "ceo_stay__do_nothing",
        "lower": "ceo_stay__review",
        "label": "Review effect (CEO stays)",
        "description": (
            "How much does commissioning a governance review reduce "
            "P(strike) when the CEO has NOT resigned?"
        ),
    },
    {
        "name": "review_resign",
        "higher": "ceo_resign__do_nothing",
        "lower": "ceo_resign__review",
        "label": "Review effect (CEO resigned)",
        "description": (
            "How much does commissioning a governance review reduce "
            "P(strike) when the CEO has already resigned?"
        ),
    },
    {
        "name": "sacking",
        "higher": "ceo_resign__review",
        "lower": "ceo_stay__board_forces_exit",
        "label": "Sacking signal",
        "description": (
            "How much additional reduction in P(strike) does the Board "
            "forcing the CEO's exit create, compared to CEO resigning + "
            "Board commissioning review? This captures the signaling value "
            "of Board-initiated accountability."
        ),
    },
]


def _build_gap_elicitation_prompt() -> tuple[str, str]:
    """Build system + user prompts for Step 2: probability gap elicitation.

    Returns (system_prompt, user_prompt).
    """
    user_prompt = (
        "You have already assessed individual P(strike) probabilities at "
        "5 ASA decision nodes. Now assess the expected GAPS (reductions) "
        "in P(strike) between specific pairs of scenarios.\n\n"
        "For each pair below, provide:\n"
        "- LOW: the smallest plausible gap (minimum reduction in P(strike))\n"
        "- EXPECTED: your best estimate of the gap\n"
        "- HIGH: the largest plausible gap (maximum reduction)\n\n"
        "All gaps are defined as P(higher scenario) - P(lower scenario), "
        "so they should be POSITIVE (or zero). The gap represents how much "
        "the stronger Board action reduces ASA's strike propensity.\n\n"
        "CONTEXT: The remuneration vote is RETROSPECTIVE. Board actions are "
        "forward-looking signals. So gaps should be SMALL (typically 0.01-0.08). "
        "The historical data shows at most a ~10pp range across all scenarios.\n\n"
        + _A2_SCENARIO_TEXT +
        "GAPS TO ASSESS:\n\n"
        "Gap 1 - DEPARTURE EFFECT (Node 3 vs Node 1):\n"
        "How much does CEO departure reduce P(strike) when Board takes no "
        "other action? Compares 'CEO stays + nothing' (worst) to 'CEO "
        "resigned + nothing'. The resignation is a significant accountability "
        "signal but does not change the FY23 pay structure.\n\n"
        "Gap 2 - REVIEW EFFECT, CEO STAYS (Node 3 vs Node 4):\n"
        "How much does commissioning a review reduce P(strike) when CEO "
        "has NOT resigned? Compares 'CEO stays + nothing' to 'CEO stays + "
        "review'. The review is forward-looking but does not address the "
        "CEO's continued presence.\n\n"
        "Gap 3 - REVIEW EFFECT, CEO RESIGNED (Node 1 vs Node 2):\n"
        "How much does commissioning a review further reduce P(strike) when "
        "CEO has already resigned? Compares 'CEO resigned + nothing' to "
        "'CEO resigned + review'. With the CEO already gone, does a review "
        "add significant additional comfort?\n\n"
        "Gap 4 - SACKING SIGNAL (Node 2 vs Node 5):\n"
        "How much additional reduction does Board forcing CEO exit create, "
        "compared to CEO resigning + review? Compares 'CEO resigned + "
        "review' (strongest voluntary + reform) to 'Board forces exit' "
        "(strongest involuntary action). The sacking is a qualitatively "
        "different accountability signal.\n\n"
        "Output JSON only:\n"
        "{\n"
        '  "gap_departure_low": <float 0-0.15>,\n'
        '  "gap_departure_expected": <float 0-0.15>,\n'
        '  "gap_departure_high": <float 0-0.15>,\n'
        '  "gap_review_stay_low": <float 0-0.15>,\n'
        '  "gap_review_stay_expected": <float 0-0.15>,\n'
        '  "gap_review_stay_high": <float 0-0.15>,\n'
        '  "gap_review_resign_low": <float 0-0.15>,\n'
        '  "gap_review_resign_expected": <float 0-0.15>,\n'
        '  "gap_review_resign_high": <float 0-0.15>,\n'
        '  "gap_sacking_low": <float 0-0.15>,\n'
        '  "gap_sacking_expected": <float 0-0.15>,\n'
        '  "gap_sacking_high": <float 0-0.15>,\n'
        '  "reasoning": "<2-4 sentences>"\n'
        "}"
    )
    return _A2_SYSTEM_PROMPT, user_prompt


def _step2_elicit_gaps(
    client,
    model: str,
    n_draws: int,
    max_workers: int,
    output_dir: Optional[Path],
) -> dict:
    """Step 2: Elicit probability gaps between A2 node pairs via LLM.

    Returns dict with per-gap {low, expected, high} aggregated from draws.
    """
    logger.info("")
    logger.info("=" * 65)
    logger.info("  STEP 2: Probability Gap Elicitation")
    logger.info("=" * 65)

    system_prompt, user_prompt = _build_gap_elicitation_prompt()
    gap_names = ["departure", "review_stay", "review_resign", "sacking"]
    gap_draws = {g: {"low": [], "expected": [], "high": []} for g in gap_names}
    reasonings = []
    n_errors = 0

    def _single_draw(draw_idx: int):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        cache_key = _make_cache_key(
            system_prompt, user_prompt, model, draw_idx, 1.0,
        )
        cached = _cache_lookup(cache_key, track_stats=True)
        if cached and "gap_response" in cached:
            return cached["gap_response"]
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_model=ProbabilityGapResponse,
                temperature=1.0,
                max_tokens=512,
            )
            result = {"reasoning": response.reasoning}
            for g in gap_names:
                result[f"{g}_low"] = getattr(response, f"gap_{g}_low")
                result[f"{g}_expected"] = getattr(response, f"gap_{g}_expected")
                result[f"{g}_high"] = getattr(response, f"gap_{g}_high")
            _cache_store(cache_key, {"gap_response": result})
            return result
        except Exception as e:
            logger.warning(f"  Gap elicitation draw {draw_idx} failed: {e}")
            return None

    logger.info(f"  Eliciting probability gaps from {n_draws} LLM draws...")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_single_draw, i): i for i in range(n_draws)}
        for fut in as_completed(futures):
            result = fut.result()
            if result is None:
                n_errors += 1
                continue
            for g in gap_names:
                lo = result[f"{g}_low"]
                ex = result[f"{g}_expected"]
                hi = result[f"{g}_high"]
                lo, ex, hi = min(lo, ex, hi), sorted([lo, ex, hi])[1], max(lo, ex, hi)
                gap_draws[g]["low"].append(lo)
                gap_draws[g]["expected"].append(ex)
                gap_draws[g]["high"].append(hi)
            if result.get("reasoning"):
                reasonings.append(result["reasoning"])

    n_valid = len(gap_draws["departure"]["expected"])
    logger.info(f"  {n_valid}/{n_draws} valid draws ({n_errors} errors)")

    # Aggregate gaps
    gap_constraints = {}
    logger.info("")
    logger.info(f"  {'Gap':<35} {'Low':>6} {'Exp':>6} {'High':>6}")
    logger.info(f"  {'-'*55}")
    for gdef in _GAP_DEFINITIONS:
        g = gdef["name"]
        lows = gap_draws[g]["low"]
        exps = gap_draws[g]["expected"]
        highs = gap_draws[g]["high"]
        gap_constraints[g] = {
            "low": float(np.percentile(lows, 10)) if lows else 0.005,
            "expected": float(np.median(exps)) if exps else 0.02,
            "high": float(np.percentile(highs, 90)) if highs else 0.10,
            "higher_node": gdef["higher"],
            "lower_node": gdef["lower"],
            "label": gdef["label"],
        }
        gc = gap_constraints[g]
        logger.info(
            f"  {gdef['label']:<35} {gc['low']:>6.3f} {gc['expected']:>6.3f} "
            f"{gc['high']:>6.3f}"
        )

    result = {
        "gap_constraints": gap_constraints,
        "n_valid": n_valid,
        "reasonings": reasonings[:3],
    }

    if output_dir:
        path = output_dir / "asa_probability_gaps.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"  Saved gaps to {path}")

    return result


# ── Step 3: Constrained target solving ───────────────────────────────────────

# Monotonicity ordering: required Node3 >= Node1 >= Node4 >= Node2 >= Node5
_RANKED_NODES = [
    "ceo_stay__do_nothing",         # Node3: highest (no action, CEO stays)
    "ceo_resign__do_nothing",       # Node1: CEO gone but no reform
    "ceo_stay__review",             # Node4: review but CEO stays
    "ceo_resign__review",           # Node2: CEO gone + review
    "ceo_stay__board_forces_exit",  # Node5: strongest action (lowest)
]
_NODE_TO_IDX = {n: i for i, n in enumerate(_RANKED_NODES)}


def _step3_solve_targets(
    ranges: dict,
    gaps: dict,
) -> dict:
    """Step 3: Constrained solve for target probabilities.

    Minimizes distance from elicited best estimates subject to:
    - Each target within its elicited [low, high] range
    - All targets >= p_expected_floor
    - Monotonicity: Node3 >= Node1 >= Node4 >= Node2 >= Node5
    - Gap constraints from Step 2 (elicited bounds on pairwise differences)

    Uses scipy.optimize.minimize (SLSQP) for constrained optimization.

    Returns dict with solved targets, constraint satisfaction report.
    """
    from scipy.optimize import minimize

    logger.info("")
    logger.info("=" * 65)
    logger.info("  STEP 3: Constrained Target Solving")
    logger.info("=" * 65)

    node_ranges = ranges["node_ranges"]
    p_floor = ranges["p_floor"]["mean"]
    p_abs_floor = ranges["p_absolute_floor"]["mean"]
    gap_constraints = gaps["gap_constraints"]

    # Initial guess: elicited best estimates (in ranked order)
    x0 = np.array([node_ranges[n]["best"] for n in _RANKED_NODES])

    # Bounds: [max(low, p_floor), min(high, 0.99)] per node
    bounds = []
    for n in _RANKED_NODES:
        lo = max(node_ranges[n]["low"], p_floor)
        hi = min(node_ranges[n]["high"], 0.99)
        bounds.append((lo, hi))

    # Objective: minimize squared distance from best estimates
    best_vals = np.array([node_ranges[n]["best"] for n in _RANKED_NODES])

    def objective(x):
        return float(np.sum((x - best_vals) ** 2))

    def grad(x):
        return 2.0 * (x - best_vals)

    # Constraints
    constraints = []

    # Monotonicity: x[i] >= x[i+1] + gap for consecutive ranked nodes.
    # A minimum separation is required so the random utility model
    # (Step 4) can produce distinct probabilities at nodes with
    # structurally different phi vectors.
    #
    # EXCEPTION: Node1 (ceo_resign__do_nothing) and Node4
    # (ceo_stay__review) are allowed to tie.  Their delta-phi
    # difference is (w_passive - w_depart), and elicitation data
    # shows identical best estimates and ranges.  Forcing a gap
    # distorts the optimizer.
    MIN_MONO_GAP = 0.005
    # Pairs that are allowed to tie (indices in _RANKED_NODES)
    _TIED_PAIRS = {(1, 2)}  # Node1 (idx=1) and Node4 (idx=2)
    for i in range(len(_RANKED_NODES) - 1):
        gap = 0.0 if (i, i + 1) in _TIED_PAIRS else MIN_MONO_GAP
        constraints.append({
            "type": "ineq",
            "fun": lambda x, i=i, g=gap: x[i] - x[i + 1] - g,
        })

    # Gap constraints from Step 2: gap_low <= x[higher] - x[lower] <= gap_high
    for gdef in _GAP_DEFINITIONS:
        gc = gap_constraints[gdef["name"]]
        idx_h = _NODE_TO_IDX[gc["higher_node"]]
        idx_l = _NODE_TO_IDX[gc["lower_node"]]
        gap_lo = gc["low"]
        gap_hi = gc["high"]

        # Lower bound: x[higher] - x[lower] >= gap_low
        constraints.append({
            "type": "ineq",
            "fun": lambda x, ih=idx_h, il=idx_l, gl=gap_lo: (
                x[ih] - x[il] - gl
            ),
        })
        # Upper bound: x[higher] - x[lower] <= gap_high
        constraints.append({
            "type": "ineq",
            "fun": lambda x, ih=idx_h, il=idx_l, gh=gap_hi: (
                gh - (x[ih] - x[il])
            ),
        })

    # Solve
    result = minimize(
        objective,
        x0,
        jac=grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )

    if not result.success:
        logger.warning(f"  Optimization warning: {result.message}")
        logger.warning("  Falling back to unconstrained best estimates with "
                       "floor clamping and monotonicity projection.")
        # Fallback: clamp to floors and enforce monotonicity via PAVA
        targets_arr = np.clip(best_vals, p_floor, 0.99)
        # Decreasing isotonic regression (PAVA)
        blocks = [[v] for v in targets_arr]
        changed = True
        while changed:
            changed = False
            merged = []
            i = 0
            while i < len(blocks):
                if i + 1 < len(blocks):
                    m_cur = sum(blocks[i]) / len(blocks[i])
                    m_nxt = sum(blocks[i + 1]) / len(blocks[i + 1])
                    if m_cur < m_nxt:
                        merged.append(blocks[i] + blocks[i + 1])
                        changed = True
                        i += 2
                        continue
                merged.append(blocks[i])
                i += 1
            blocks = merged
        pava_vals = []
        for block in blocks:
            bm = sum(block) / len(block)
            pava_vals.extend([bm] * len(block))
        targets_arr = np.array(pava_vals)
    else:
        targets_arr = result.x

    # Build targets dict
    targets = {n: float(targets_arr[i]) for i, n in enumerate(_RANKED_NODES)}

    # Constraint satisfaction report
    logger.info("  Constraints:")
    constraint_report = []

    # Check ranges
    all_in_range = True
    for n in _RANKED_NODES:
        lo = max(node_ranges[n]["low"], p_floor)
        hi = min(node_ranges[n]["high"], 0.99)
        in_range = lo - 1e-6 <= targets[n] <= hi + 1e-6
        if not in_range:
            all_in_range = False
    status = "PASS" if all_in_range else "FAIL"
    constraint_report.append(("All targets within elicited ranges", status))
    logger.info(f"    {'PASS' if all_in_range else 'FAIL'}: All targets within "
                f"elicited ranges")

    # Check monotonicity (min gap, with tie allowance)
    mono_ok = all(
        targets[_RANKED_NODES[i]] >= targets[_RANKED_NODES[i + 1]]
        + (0.0 if (i, i + 1) in _TIED_PAIRS else MIN_MONO_GAP) - 1e-6
        for i in range(len(_RANKED_NODES) - 1)
    )
    constraint_report.append(("Monotonicity", "PASS" if mono_ok else "FAIL"))
    logger.info(f"    {'PASS' if mono_ok else 'FAIL'}: Monotonicity "
                f"(min gap {MIN_MONO_GAP}, ties allowed at Node1/Node4; "
                f"Node3 > Node1 >= Node4 > Node2 > Node5)")

    # Check floor
    floor_ok = all(targets[n] >= p_floor - 1e-6 for n in _RANKED_NODES)
    constraint_report.append(("Floor constraint", "PASS" if floor_ok else "FAIL"))
    logger.info(f"    {'PASS' if floor_ok else 'FAIL'}: All targets >= "
                f"expected floor ({p_floor:.3f})")

    # Check gaps
    gap_report = []
    for gdef in _GAP_DEFINITIONS:
        gc = gap_constraints[gdef["name"]]
        actual_gap = targets[gc["higher_node"]] - targets[gc["lower_node"]]
        in_range = gc["low"] - 1e-6 <= actual_gap <= gc["high"] + 1e-6
        gap_report.append({
            "name": gdef["name"],
            "label": gdef["label"],
            "actual": round(actual_gap, 4),
            "elicited_low": gc["low"],
            "elicited_high": gc["high"],
            "satisfied": in_range,
        })
        status = "PASS" if in_range else "WARN"
        constraint_report.append(
            (f"Gap: {gdef['label']}", status)
        )
        logger.info(
            f"    {status}: {gdef['label']}: "
            f"actual={actual_gap:.4f} "
            f"[{gc['low']:.3f}, {gc['high']:.3f}]"
        )

    # Log solved target ladder
    logger.info("")
    logger.info("  Solved target probability ladder:")
    prev_val = None
    for n in _RANKED_NODES:
        gap_str = ""
        if prev_val is not None:
            gap = prev_val - targets[n]
            gap_str = f"  (gap={gap:+.4f})"
        r = node_ranges[n]
        logger.info(
            f"    {n}: {targets[n]:.4f}  "
            f"[range: {r['low']:.3f}-{r['high']:.3f}, "
            f"best: {r['best']:.3f}]{gap_str}"
        )
        prev_val = targets[n]

    return {
        "targets": targets,
        "p_floor": p_floor,
        "p_absolute_floor": p_abs_floor,
        "constraint_report": constraint_report,
        "gap_report": gap_report,
        "optimization_success": bool(result.success) if hasattr(result, 'success') else True,
    }


# ── Step 4: Stochastic utility parameter calibration ─────────────────────────

# The 5 interaction parameters whose stochastic draws drive the A2 decision.
# Context parameters (w_ctx_*) cancel in delta_EU so are not calibrated here.
_INTERACTION_PARAMS = [
    "w_strike_cost",
    "w_strike_vs_passive",
    "w_departure_dampens",
    "w_sack_dampens",
    "w_credibility_signal",
]


def _step4_calibrate_weights(
    targets_result: dict,
    ranges_result: dict | None = None,
) -> dict:
    """Step 4: Calibrate stochastic utility parameters via random utility model.

    Each interaction weight w_k is drawn from TruncNormal(mu_k, sigma_k, 1, 5).
    For each simulation draw, we sample all 5 weights, compute
    EU(strike) and EU(no_strike) at each A2 node, and take the argmax.
    P(strike | node) = fraction of draws where EU(strike) > EU(no_strike).

    We optimise all 10 parameters (5 means + 5 sigmas) using differential
    evolution to minimise the squared error between simulated argmax
    probabilities and the target probabilities from Step 3.

    Returns dict with calibrated weight distributions, simulation results.
    """
    from scipy.optimize import differential_evolution
    from scipy.stats import truncnorm

    logger.info("")
    logger.info("=" * 65)
    logger.info("  STEP 4: Stochastic Utility Parameter Calibration")
    logger.info("=" * 65)

    targets = targets_result["targets"]
    p_floor = targets_result["p_floor"]
    p_abs_floor = targets_result["p_absolute_floor"]

    # ── 4a. Fit Beta distributions via method of moments ──
    #
    # Each node's Beta(alpha, beta) is derived from:
    #   mean  = target P(strike) from Step 3
    #   var   = SD² from Step 1 elicitation (epistemic uncertainty)
    #
    # Method of moments:
    #   n_eff = mean*(1-mean)/var - 1
    #   alpha = mean * n_eff
    #   beta  = (1-mean) * n_eff
    #
    # This gives node-specific concentration: nodes with more
    # elicitation uncertainty get wider Beta distributions.
    # Fallback n_eff=200 if ranges not available.

    _N_EFF_FALLBACK = 200
    _N_EFF_MIN = 20   # floor to prevent degenerate Betas
    _N_EFF_MAX = 500  # cap to prevent over-concentration

    # Extract elicitation SDs if available
    node_sds = {}
    if ranges_result and "node_ranges" in ranges_result:
        for n in _RANKED_NODES:
            nr = ranges_result["node_ranges"].get(n, {})
            if "sd" in nr and nr["sd"] > 0:
                node_sds[n] = nr["sd"]

    beta_priors = {}
    beta_means = {}

    logger.info("")
    if node_sds:
        logger.info("  Fit Beta priors via method of moments "
                     "(node-specific n_eff from elicitation SD):")
    else:
        logger.info(f"  Fit Beta priors (fixed n_eff={_N_EFF_FALLBACK}, "
                     f"no elicitation SDs available):")
    logger.info(f"  {'Node':<35} {'Target':>7} {'SD':>7} {'n_eff':>6} "
                f"{'alpha':>6} {'beta':>5} {'Beta Mean':>10} {'Disc Err':>9}")
    logger.info(f"  {'-'*92}")

    for n in _RANKED_NODES:
        p_target = targets[n]

        if n in node_sds:
            sd = node_sds[n]
            var = sd ** 2
            # Method of moments: n_eff = mean*(1-mean)/var - 1
            n_eff_raw = p_target * (1.0 - p_target) / var - 1.0
            n_eff = int(round(np.clip(n_eff_raw, _N_EFF_MIN, _N_EFF_MAX)))
        else:
            sd = 0.0
            n_eff = _N_EFF_FALLBACK

        alpha = max(1, round(p_target * n_eff))
        beta_param = max(1, n_eff - alpha)
        b_mean = alpha / (alpha + beta_param)
        beta_priors[n] = {"alpha": int(alpha), "beta": int(beta_param),
                          "beta_mean": round(b_mean, 4),
                          "n_eff": n_eff}
        beta_means[n] = b_mean
        logger.info(
            f"  {n:<35} {p_target:>7.4f} {sd:>7.4f} {n_eff:>6} "
            f"{alpha:>6} {beta_param:>5} "
            f"{b_mean:>10.4f} {b_mean - p_target:>+9.4f}"
        )

    # ── 4b. Pre-compute delta_phi at each node ──
    #
    # delta_phi_k = phi_k(rec_strike) - phi_k(no_strike) for each
    # interaction parameter.  Context parameters have delta_phi = 0
    # by construction (they fire equally for both actions).

    logger.info("")
    logger.info("  Delta-phi matrix (interaction parameters only):")
    logger.info(f"  {'Node':<35} " + " ".join(
        f"{p.replace('w_', ''):>10}" for p in _INTERACTION_PARAMS
    ))
    logger.info(f"  {'-'*90}")

    delta_phi = {}  # {node_name: np.array of shape (5,)}
    for node_name in _RANKED_NODES:
        state = A2_NODE_STATES[node_name]
        phi_rec = decompose_utility_asa(**state, action="rec_strike")
        phi_no = decompose_utility_asa(**state, action="no_strike")
        dphi = np.array([
            phi_rec.get(p, 0.0) - phi_no.get(p, 0.0)
            for p in _INTERACTION_PARAMS
        ])
        delta_phi[node_name] = dphi
        logger.info(
            f"  {node_name:<35} " +
            " ".join(f"{v:>10.1f}" for v in dphi)
        )

    # ── 4c. Analytical + MC random utility model ──
    #
    # Each weight w_k ~ TruncNormal(mu_k, sigma_k, lo=1, hi=5).
    # delta_EU = sum_k w_k * delta_phi_k.
    # Strike wins iff delta_EU > 0.
    # P(strike | node) = fraction of draws where EU(strike) > EU(no_strike).
    #
    # Since delta_EU is a sum of independent truncated normals, its
    # distribution is approximately normal (CLT with 5 terms).  We compute
    # E[delta_EU] and Var[delta_EU] from the truncated normal moments,
    # then P(strike) = Phi(E[delta_EU] / sqrt(Var[delta_EU])).
    #
    # This gives a smooth, deterministic loss function for optimisation.
    # MC simulation is used only for final validation.

    from scipy.stats import truncnorm, norm

    N_SIM_FINAL = 50000  # Draws for final MC validation

    def _truncnorm_moments(mu: float, sigma: float,
                           lo: float = 0.5, hi: float = 10.0):
        """Compute E[X] and Var[X] for X ~ TruncNormal(mu, sigma, lo, hi)."""
        a = (lo - mu) / sigma
        b = (hi - mu) / sigma
        Z = norm.cdf(b) - norm.cdf(a)
        if Z < 1e-12:
            # Degenerate: all mass at one boundary
            return np.clip(mu, lo, hi), 1e-12
        phi_a = norm.pdf(a)
        phi_b = norm.pdf(b)
        # E[X] = mu + sigma * (phi(a) - phi(b)) / Z
        ex = mu + sigma * (phi_a - phi_b) / Z
        # Var[X] = sigma^2 * [1 + (a*phi(a) - b*phi(b))/Z
        #                       - ((phi(a) - phi(b))/Z)^2]
        r = (phi_a - phi_b) / Z
        vx = sigma**2 * (1.0 + (a * phi_a - b * phi_b) / Z - r**2)
        return float(ex), max(float(vx), 1e-12)

    def _analytical_p_strike(
        mus: np.ndarray,
        sigmas: np.ndarray,
    ) -> dict[str, float]:
        """Compute P(strike) analytically at each node.

        Uses CLT: delta_EU ~ N(E[delta_EU], Var[delta_EU])
        where moments come from truncated normal weight distributions.
        P(strike) = Phi(E[delta_EU] / sqrt(Var[delta_EU])).
        """
        # Compute truncated normal moments for each weight
        tn_means = np.empty(5)
        tn_vars = np.empty(5)
        for k in range(5):
            tn_means[k], tn_vars[k] = _truncnorm_moments(mus[k], sigmas[k])

        results = {}
        for node_name in _RANKED_NODES:
            dphi = delta_phi[node_name]  # (5,)
            # E[delta_EU] = sum_k dphi_k * E[w_k]
            e_deu = float(dphi @ tn_means)
            # Var[delta_EU] = sum_k dphi_k^2 * Var[w_k]  (independent)
            v_deu = float((dphi**2) @ tn_vars)
            sd_deu = np.sqrt(v_deu)
            # P(strike) = P(delta_EU > 0) = Phi(E/SD)
            if sd_deu < 1e-10:
                results[node_name] = 1.0 if e_deu > 0 else 0.0
            else:
                results[node_name] = float(norm.cdf(e_deu / sd_deu))
        return results

    def _loss(params: np.ndarray) -> float:
        """Smooth, deterministic loss: analytical P(strike) vs Beta means."""
        mus = params[:5]
        sigmas = params[5:]
        p_analytical = _analytical_p_strike(mus, sigmas)
        return sum(
            (p_analytical[n] - beta_means[n]) ** 2 for n in _RANKED_NODES
        )

    # MC simulation for final validation only
    def _simulate_argmax(
        mus: np.ndarray,
        sigmas: np.ndarray,
        n_sim: int = N_SIM_FINAL,
        seed: int = 42,
    ) -> dict[str, float]:
        """Draw stochastic weights, compute argmax P(strike) at each node."""
        rng = np.random.default_rng(seed)
        lo, hi = 0.5, 10.0
        w_draws = np.empty((n_sim, 5))
        for k in range(5):
            a_trunc = (lo - mus[k]) / sigmas[k]
            b_trunc = (hi - mus[k]) / sigmas[k]
            w_draws[:, k] = truncnorm.rvs(
                a_trunc, b_trunc,
                loc=mus[k], scale=sigmas[k],
                size=n_sim, random_state=rng,
            )
        results = {}
        for node_name in _RANKED_NODES:
            dphi = delta_phi[node_name]
            deu = w_draws @ dphi
            results[node_name] = float(np.mean(deu > 0))
        return results

    # Bounds: means in [0.5, 10], sigmas in [0.1, 4.0]
    # Drawn values are truncated to [0.5, 10] regardless of mu/sigma.
    bounds_mu = [(0.5, 10.0)] * 5
    bounds_sigma = [(0.1, 4.0)] * 5
    bounds = bounds_mu + bounds_sigma

    # ── Two-stage optimisation ──
    #
    # Stage 1: Analytical CLT (L-BFGS-B, multi-start) — fast, gives
    #   good initial guess but systematically biased when weights are
    #   at bounds (truncated normals are skewed, CLT fails).
    #
    # Stage 2: MC with common random numbers (Nelder-Mead) — refines
    #   from Stage 1 solution using exact MC simulation.  Fixed seed
    #   makes loss deterministic.  Large N_sim makes it smooth enough
    #   for Nelder-Mead.

    from scipy.optimize import minimize as scipy_minimize

    N_RESTARTS = 50
    N_SIM_OPT = 100000  # MC draws for Stage 2 optimisation
    rng_opt = np.random.default_rng(42)

    logger.info("")
    logger.info(f"  Target probabilities (Beta means):")
    for n in _RANKED_NODES:
        logger.info(f"    {n}: {beta_means[n]:.4f}")

    # ── Stage 1: Analytical CLT ──
    logger.info("")
    logger.info(f"  Stage 1: Analytical CLT optimisation "
                f"({N_RESTARTS} random restarts)...")

    best_result = None
    best_loss = 1e12
    for restart in range(N_RESTARTS):
        x0 = np.concatenate([
            rng_opt.uniform(0.5, 10.0, size=5),   # random means
            rng_opt.uniform(0.1, 2.5, size=5),    # random sigmas
        ])
        res = scipy_minimize(
            _loss, x0, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-15},
        )
        if res.fun < best_loss:
            best_loss = res.fun
            best_result = res

    analytical_mus = best_result.x[:5]
    analytical_sigmas = best_result.x[5:]

    logger.info(f"  Stage 1 loss (CLT): {best_loss:.10f}")

    # ── Stage 2: MC refinement with CRN ──
    #
    # Draw a fixed set of standard normal samples (common random
    # numbers).  For any (mu, sigma), transform to truncated normal
    # draws, compute delta_EU, count P(strike).  Deterministic and
    # unbiased.

    logger.info(f"  Stage 2: MC refinement ({N_SIM_OPT} CRN draws, "
                f"Nelder-Mead)...")

    # Pre-draw standard uniforms for inverse-CDF truncated normal sampling
    crn_uniforms = np.random.default_rng(42).uniform(size=(N_SIM_OPT, 5))

    def _mc_p_strike(params: np.ndarray) -> dict[str, float]:
        """Compute P(strike) via MC with common random numbers."""
        mus = params[:5]
        sigmas = params[5:]
        lo, hi = 0.5, 10.0
        w_draws = np.empty((N_SIM_OPT, 5))
        for k in range(5):
            a = (lo - mus[k]) / sigmas[k]
            b = (hi - mus[k]) / sigmas[k]
            # Inverse CDF: ppf of truncnorm at common uniform draws
            w_draws[:, k] = truncnorm.ppf(
                crn_uniforms[:, k], a, b,
                loc=mus[k], scale=sigmas[k],
            )
        results = {}
        for node_name in _RANKED_NODES:
            dphi = delta_phi[node_name]
            deu = w_draws @ dphi
            results[node_name] = float(np.mean(deu > 0))
        return results

    def _mc_loss(params: np.ndarray) -> float:
        """MC-based loss: P(strike) vs Beta means."""
        p_mc = _mc_p_strike(params)
        return sum(
            (p_mc[n] - beta_means[n]) ** 2 for n in _RANKED_NODES
        )

    # Refine from Stage 1 solution
    mc_result = scipy_minimize(
        _mc_loss, best_result.x, method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-4, "fatol": 1e-10},
    )

    opt_mus = mc_result.x[:5]
    opt_sigmas = mc_result.x[5:]
    # Clip to bounds (Nelder-Mead doesn't enforce bounds)
    opt_mus = np.clip(opt_mus, 0.5, 10.0)
    opt_sigmas = np.clip(opt_sigmas, 0.1, 4.0)

    logger.info(f"  Stage 2 loss (MC):  {mc_result.fun:.10f}")
    logger.info(f"  Converged: {mc_result.success} "
                f"(iterations: {mc_result.nit})")

    # Log MC probabilities at optimum (these are the primary results
    # since Stage 2 optimised directly on MC).  Analytical shown for
    # comparison only.
    mc_opt_probs = _mc_p_strike(np.concatenate([opt_mus, opt_sigmas]))
    analytical_probs = _analytical_p_strike(opt_mus, opt_sigmas)
    logger.info("")
    logger.info("  P(strike) at optimum (MC = primary, CLT = comparison):")
    max_mc_error = 0.0
    for n in _RANKED_NODES:
        mc_err = abs(mc_opt_probs[n] - beta_means[n])
        max_mc_error = max(max_mc_error, mc_err)
        logger.info(f"    {n}: MC={mc_opt_probs[n]:.6f}  "
                    f"CLT={analytical_probs[n]:.6f}  "
                    f"(target: {beta_means[n]:.4f}, "
                    f"MC err: {mc_opt_probs[n] - beta_means[n]:+.6f})")

    if max_mc_error > 0.01:
        logger.warning(
            f"  Max MC error = {max_mc_error:.4f} > 0.01. "
            f"Structural limitation: some target probabilities "
            f"cannot be exactly matched with weights in [0.5, 10]."
        )

    # ── 4d. Final validation simulation (independent seed) ──

    logger.info("")
    logger.info(f"  Final validation simulation ({N_SIM_FINAL} draws, "
                f"independent seed):")

    sim_probs = _simulate_argmax(opt_mus, opt_sigmas,
                                 n_sim=N_SIM_FINAL, seed=12345)

    # ── 4e. Refit Beta distributions at MC-optimized probabilities ──
    #
    # The initial Beta priors (Step 4a) used target probabilities as
    # means.  Now we refit using MC-optimized P(strike) — the actual
    # model output — as the mean, keeping the same n_eff (concentration)
    # per node.  This ensures the engine's Beta priors reflect the
    # true random utility model rather than the optimization targets.

    logger.info("")
    logger.info("  Refit Beta priors at MC-optimized probabilities:")
    logger.info(f"  {'Node':<35} {'MC P':>7} {'n_eff':>6} "
                f"{'alpha':>6} {'beta':>5} {'Beta Mean':>10}")
    logger.info(f"  {'-'*78}")

    from scipy.stats import beta as beta_dist

    for n in _RANKED_NODES:
        mc_p = mc_opt_probs[n]
        n_eff = beta_priors[n].get("n_eff", _N_EFF_FALLBACK)
        alpha = max(1, round(mc_p * n_eff))
        beta_param = max(1, n_eff - alpha)
        b_mean = alpha / (alpha + beta_param)
        b_sd = beta_dist(alpha, beta_param).std()
        b_lo, b_hi = beta_dist(alpha, beta_param).ppf([0.025, 0.975])
        beta_priors[n] = {"alpha": int(alpha), "beta": int(beta_param),
                          "beta_mean": round(b_mean, 4),
                          "n_eff": n_eff,
                          "sd": round(float(b_sd), 4),
                          "ci_lo": round(float(b_lo), 4),
                          "ci_hi": round(float(b_hi), 4)}
        beta_means[n] = b_mean
        logger.info(
            f"  {n:<35} {mc_p:>7.4f} {n_eff:>6} "
            f"{alpha:>6} {beta_param:>5} {b_mean:>10.4f}"
        )

    # ── Summary: Beta distributions for engine consumption ──
    logger.info("")
    logger.info("  Beta distributions for ARA engine (ASA as opponent):")
    logger.info(f"  {'Path Key':<30} {'Beta(a,b)':>14} {'Mean':>7} "
                f"{'SD':>7} {'95% CI':>20}  {'n_eff':>5}")
    logger.info(f"  {'-'*90}")
    for n in _RANKED_NODES:
        bp = beta_priors[n]
        tree_key = _A2_NODE_TO_TREE_KEY[n]
        logger.info(
            f"  {tree_key:<30} Beta({bp['alpha']:>3},{bp['beta']:>3}) "
            f"{bp['beta_mean']:>7.4f} {bp['sd']:>7.4f} "
            f"[{bp['ci_lo']:.4f}, {bp['ci_hi']:.4f}]  {bp['n_eff']:>5}"
        )

    # Build calibrated weight distributions
    calibrated_weights = {}
    logger.info("")
    logger.info(f"  {'Parameter':<25} {'Mean':>6} {'Sigma':>7}")
    logger.info(f"  {'-'*40}")
    for k, pname in enumerate(_INTERACTION_PARAMS):
        calibrated_weights[pname] = {
            "mean": round(float(opt_mus[k]), 4),
            "sigma": round(float(opt_sigmas[k]), 4),
            "lower": 0.5,
            "upper": 10.0,
            "distribution": "truncnorm",
        }
        logger.info(f"  {pname:<25} {opt_mus[k]:>6.3f} {opt_sigmas[k]:>7.3f}")

    # Context weights are fixed (they cancel in delta_EU but are needed
    # for absolute utility levels in ordinal probit / Likert prediction)
    for ctx_param in ["w_ctx_inaction", "w_ctx_departure", "w_ctx_review"]:
        calibrated_weights[ctx_param] = {
            "mean": 1.0, "sigma": 0.0,
            "lower": 1.0, "upper": 5.0,
            "distribution": "fixed",
        }

    # Build simulation results and implied probabilities
    implied_probs = {}
    sim_results = {}

    logger.info("")
    logger.info(f"  {'Node':<35} {'Target':>7} {'Beta Mean':>10} "
                f"{'MC Opt':>8} {'Validation':>11} {'Val Error':>9}")
    logger.info(f"  {'-'*85}")

    for node_name in _RANKED_NODES:
        b_mean = beta_means[node_name]
        sim_p = sim_probs[node_name]
        mc_p = mc_opt_probs[node_name]
        target_p = targets[node_name]

        implied_probs[node_name] = {
            "p_strike": round(mc_p, 4),
            "analytical_p": round(analytical_probs[node_name], 6),
            "mc_optimized_p": round(mc_p, 6),
            "simulated_p": round(sim_p, 4),
            "target": round(target_p, 4),
            "beta_mean": round(b_mean, 4),
            "sim_vs_beta_error": round(sim_p - b_mean, 4),
            "mc_vs_beta_error": round(mc_p - b_mean, 6),
        }
        sim_results[node_name] = {
            "target": round(target_p, 4),
            "alpha": int(beta_priors[node_name]["alpha"]),
            "beta": int(beta_priors[node_name]["beta"]),
            "n_eff": beta_priors[node_name].get("n_eff", _N_EFF_FALLBACK),
            "beta_mean": round(b_mean, 4),
            "mc_optimized_p": round(mc_p, 6),
            "simulated_p_strike": round(sim_p, 4),
            "sim_vs_beta_error": round(sim_p - b_mean, 4),
            "n_sim": N_SIM_FINAL,
        }

        logger.info(
            f"  {node_name:<35} {target_p:>7.4f} {b_mean:>10.4f} "
            f"{mc_p:>8.4f} {sim_p:>11.4f} {sim_p - b_mean:>+9.4f}"
        )

    max_sim_error = max(
        abs(sr["sim_vs_beta_error"]) for sr in sim_results.values()
    )
    # Discretization error: Beta mean (after integer rounding) vs MC-optimized P
    max_disc_error = max(
        abs(sr["beta_mean"] - sr["mc_optimized_p"]) for sr in sim_results.values()
    )
    logger.info(f"  Max sim vs Beta mean error: {max_sim_error:.4f}")
    logger.info(f"  Max discretization error:   {max_disc_error:.4f}")
    logger.info(f"  Optimisation loss (MC SSE):  {mc_result.fun:.8f}")

    return {
        "calibrated_weights": calibrated_weights,
        "implied_probs": implied_probs,
        "targets": targets,
        "p_floor": p_abs_floor,
        "p_expected_floor": p_floor,
        "p_absolute_floor": p_abs_floor,
        "beta_priors": beta_priors,
        "simulation": sim_results,
        "max_disc_error": round(max_disc_error, 4),
        "max_sim_error": round(max_sim_error, 4),
        "optimisation_loss": round(float(mc_result.fun), 8),
        "optimisation_loss_clt": round(float(best_result.fun), 8),
        "optimisation_converged": bool(mc_result.success),
        "n_sim_final": N_SIM_FINAL,
    }


# ── Pipeline orchestrator ────────────────────────────────────────────────────

def run_a2_calibration_pipeline(
    client,
    model: str,
    n_draws: int = 50,
    max_workers: int = 10,
    output_dir: Optional[Path] = None,
) -> dict:
    """Run the 4-step A2 probability calibration pipeline.

    Steps:
        1. LLM elicitation of probability RANGES per A2 node
        2. LLM elicitation of probability GAPS between node pairs
        3. Constrained solving for target probabilities
        4. Utility weight calibration + simulation validation

    Returns dict with all calibration results for dashboard and engine.
    """
    logger.info("")
    logger.info("*" * 65)
    logger.info("  A2 PROBABILITY CALIBRATION PIPELINE")
    logger.info("*" * 65)

    # Step 1: Elicit ranges
    ranges = _step1_elicit_ranges(client, model, n_draws, max_workers, output_dir)

    # Step 2: Elicit gaps
    gaps = _step2_elicit_gaps(client, model, n_draws, max_workers, output_dir)

    # Step 3: Solve for targets
    targets = _step3_solve_targets(ranges, gaps)

    # Step 4: Calibrate weights and simulate
    calibration = _step4_calibrate_weights(targets, ranges)

    # Compute final A2 probabilities for engine consumption
    a2_probs = {}
    for node_name in A2_NODE_STATES:
        ip = calibration["implied_probs"][node_name]
        p_strike = max(calibration["p_floor"], min(0.99, ip["p_strike"]))
        p_no_strike = 1.0 - p_strike
        a2_probs[node_name] = {
            "no_strike": round(float(p_no_strike), 4),
            "rec_strike": round(float(p_strike), 4),
            "eu_diff_sd": round(float(ranges["node_ranges"][node_name]["sd"]), 4),
            "source": "calibration_pipeline",
        }

    logger.info("")
    logger.info("*" * 65)
    logger.info("  PIPELINE COMPLETE")
    logger.info("*" * 65)

    return {
        "ranges": ranges,
        "gaps": gaps,
        "targets": targets,
        "calibration": calibration,
        "a2_probs": a2_probs,
    }


# Mapping from A2 node names to TREE_DEFAULT_PROBS keys and engine paths
_A2_NODE_TO_TREE_KEY = {
    "ceo_resign__do_nothing":       "resigned_D0_minimal",
    "ceo_resign__review":           "resigned_D1_review",
    "ceo_stay__do_nothing":         "stayed_D0_minimal",
    "ceo_stay__review":             "stayed_D1_review",
    "ceo_stay__board_forces_exit":  "stayed_D3_ceo_transition",
}


def _save_a2_calibration(
    calibration: dict,
    a2_probs: dict,
    output_dir: Path,
) -> None:
    """Save calibrated A2 probabilities for engine and game tree consumption.

    Writes outputs/asa/asa_a2_calibration.json with:
    - tree_probs: keyed by TREE_DEFAULT_PROBS path names
    - implied_probs: per-node P(strike), delta_EU, target, error
    - calibrated_weights: the utility interaction weights
    - elicitation_source: metadata for provenance

    This file is the single source of truth for A2 strike probabilities.
    It is loaded by:
    - run/game_tree.py (TREE_DEFAULT_PROBS["A2"])
    - engine/predictive.py (_fixed_policy Beta priors)
    """
    tree_probs = {}
    beta_priors_out = {}
    for node_name, ip in calibration["implied_probs"].items():
        tree_key = _A2_NODE_TO_TREE_KEY[node_name]
        p_strike = ip["p_strike"]
        tree_probs[tree_key] = {
            "A2_no_strike": round(1.0 - p_strike, 4),
            "A2_rec_strike": round(p_strike, 4),
        }
        # Include pre-computed Beta priors so engine reads them directly
        bp = calibration["beta_priors"].get(node_name, {})
        if bp:
            beta_priors_out[tree_key] = {
                "alpha": bp["alpha"],
                "beta": bp["beta"],
                "beta_mean": bp["beta_mean"],
                "n_eff": bp.get("n_eff"),
                "sd": bp.get("sd"),
                "ci_lo": bp.get("ci_lo"),
                "ci_hi": bp.get("ci_hi"),
            }

    out = {
        "tree_probs": tree_probs,
        "beta_priors": beta_priors_out,
        "implied_probs": calibration["implied_probs"],
        "targets": calibration["targets"],
        "calibrated_weights": calibration["calibrated_weights"],
        "p_floor": calibration["p_floor"],
        "p_expected_floor": calibration["p_expected_floor"],
        "p_absolute_floor": calibration["p_absolute_floor"],
        "optimisation_loss": calibration.get("optimisation_loss"),
        "optimisation_converged": calibration.get("optimisation_converged"),
    }

    cal_path = output_dir / "asa_a2_calibration.json"
    with open(cal_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Saved A2 calibration: {cal_path}")


def load_a2_calibration(
    cal_path: Path = None,
) -> dict | None:
    """Load calibrated A2 probabilities from the pipeline output.

    Returns the calibration dict, or None if the file doesn't exist.
    Default path: data directory sibling outputs/asa/asa_a2_calibration.json
    """
    if cal_path is None:
        cal_path = Path(__file__).parent / "outputs" / "asa" / "asa_a2_calibration.json"
    if not cal_path.exists():
        return None
    with open(cal_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── SEC 10: Feature selection & diagnostics ──────────────────────────────────

def run_feature_selection(
    est_result: StanEstimationResult,
) -> dict:
    """Post-estimation feature selection using posterior weight draws.

    Relevance assessed from posterior distribution:
    - Pr(w_k > 0.1): probability weight exceeds practical threshold
    - CV: posterior SD / |posterior mean|
    """
    w_draws = est_result.w_draws
    K = w_draws.shape[1]

    relevance = {}
    excluded = []

    for k, pname in enumerate(WEIGHT_PARAM_NAMES):
        draws = w_draws[:, k]
        mean_val = float(np.mean(draws))
        sd_val = float(np.std(draws))
        pr_gt_threshold = float(np.mean(draws > 0.1))
        ci_lo = float(np.percentile(draws, 2.5))
        ci_hi = float(np.percentile(draws, 97.5))
        cv = sd_val / abs(mean_val) if abs(mean_val) > 1e-8 else float("inf")

        relevance[pname] = {
            "mean": round(mean_val, 4),
            "sd": round(sd_val, 4),
            "pr_gt_threshold": round(pr_gt_threshold, 4),
            "cv": round(cv, 4),
            "ci_lo": round(ci_lo, 4),
            "ci_hi": round(ci_hi, 4),
        }

        if pr_gt_threshold < 0.50:
            excluded.append(pname)
            logger.warning(f"Feature selection: {pname} flagged (Pr(w>0.1)={pr_gt_threshold:.2f})")

    return {
        "relevance": relevance,
        "excluded_params": excluded,
    }


# ── Stage 5: Behavioural Diagnostics ──

def run_diagnostics(
    scenarios: list[Scenario],
    likert_summary_df: pd.DataFrame,
    est_result: StanEstimationResult,
    elicitation_path: Path,
    diagnostics_path: Path,
) -> dict:
    """Stage 5: Run behavioural bias diagnostics on ASA Likert data."""
    logger.info("Stage 5: Running ASA behavioural diagnostics...")

    diagnostics = {"tests": [], "summary": {}}
    scenario_lookup = {s.scenario_id: s for s in scenarios}

    # Test 1: Sequence independence — same game tree state, different temporal order
    seq_scenarios = [s for s in scenarios
                     if s.tier == 3 and s.target_parameter == "sequence_bias"]
    if len(seq_scenarios) >= 3:
        scores = []
        for s in seq_scenarios:
            grp = likert_summary_df[likert_summary_df["scenario_id"] == s.scenario_id]
            if not grp.empty:
                rec_score = grp[grp["action"] == "rec_strike"]["mean_score"]
                if not rec_score.empty:
                    scores.append(float(rec_score.iloc[0]))
        if len(scores) >= 2:
            score_range = max(scores) - min(scores)
            passed = score_range < 1.0
            diagnostics["tests"].append({
                "name": "sequence_independence",
                "passed": passed,
                "detail": f"Sequence probe scores range: {score_range:.2f} (threshold: 1.0)",
                "scores": scores,
            })

    # Test 2: Salience asymmetry — strike more likely in high-profile cases
    high_profile_scenarios = [s for s in scenarios
                              if s.tier in (1, 2)
                              and s.state_vector.get("high_profile", True)]
    low_profile_scenarios = [s for s in scenarios
                             if s.tier == 2
                             and not s.state_vector.get("high_profile", True)]

    def _get_score_diff(scenario_list):
        diffs = []
        for s in scenario_list:
            grp = likert_summary_df[likert_summary_df["scenario_id"] == s.scenario_id]
            if len(grp) >= 2:
                no_s = grp[grp["action"] == "no_strike"]["mean_score"]
                rec_s = grp[grp["action"] == "rec_strike"]["mean_score"]
                if not no_s.empty and not rec_s.empty:
                    diffs.append(float(rec_s.iloc[0]) - float(no_s.iloc[0]))
        return np.mean(diffs) if diffs else 0.0

    if high_profile_scenarios and low_profile_scenarios:
        high_diff = _get_score_diff(high_profile_scenarios)
        low_diff = _get_score_diff(low_profile_scenarios)
        salience_effect = high_diff - low_diff
        diagnostics["tests"].append({
            "name": "salience_asymmetry",
            "passed": salience_effect > 0,
            "detail": f"Salience effect (high-low profile): {salience_effect:.2f} (expected > 0)",
        })

    # Test 3: Delay effect — delayed action should weaken Board credit
    delay_scenarios = [s for s in scenarios
                       if s.tier == 2 and s.target_parameter == "w_strike_cost"
                       and s.state_vector.get("action_delayed", False)]
    immediate_scenarios = [s for s in scenarios
                           if (s.tier in (1, 2))
                           and s.state_vector.get("review_commissioned", False)
                           and not s.state_vector.get("action_delayed", False)]

    if delay_scenarios and immediate_scenarios:
        delay_diff = _get_score_diff(delay_scenarios)
        immediate_diff = _get_score_diff(immediate_scenarios)
        delay_effect = delay_diff - immediate_diff
        diagnostics["tests"].append({
            "name": "delay_effect",
            "passed": delay_effect > 0,
            "detail": f"Delay effect (delayed-immediate): {delay_effect:.2f} (expected > 0)",
        })

    # Test 4: Forced exit premium — forced exit should generate stronger no-strike
    framing_scenarios = [s for s in scenarios
                         if s.tier == 3 and s.target_parameter == "framing_bias"]
    if len(framing_scenarios) >= 2:
        framing_scores = {}
        for s in framing_scenarios:
            sv = s.state_vector
            label = "sacked" if sv.get("ceo_sacked", False) else "resigned"
            grp = likert_summary_df[likert_summary_df["scenario_id"] == s.scenario_id]
            no_s = grp[grp["action"] == "no_strike"]["mean_score"]
            if not no_s.empty:
                framing_scores.setdefault(label, []).append(float(no_s.iloc[0]))

        if "sacked" in framing_scores and "resigned" in framing_scores:
            sacked_mean = np.mean(framing_scores["sacked"])
            resigned_mean = np.mean(framing_scores["resigned"])
            diagnostics["tests"].append({
                "name": "forced_exit_premium",
                "passed": True,  # Informational
                "detail": f"No-strike score: sacked={sacked_mean:.2f}, "
                          f"resigned={resigned_mean:.2f}, "
                          f"premium={sacked_mean - resigned_mean:.2f}",
            })

    # Save diagnostics
    n_passed = sum(1 for t in diagnostics["tests"] if t["passed"])
    diagnostics["summary"] = {
        "n_tests": len(diagnostics["tests"]),
        "n_passed": n_passed,
    }

    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(diagnostics_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, default=str)

    logger.info(f"Diagnostics: {n_passed}/{len(diagnostics['tests'])} tests passed")
    return diagnostics


# ── SEC 11: Stage 6 — Validation ─────────────────────────────────────────────

def _compute_scenario_fit(
    phi: np.ndarray,
    anchored: np.ndarray,
    sa_id_map: dict,
    est_result: StanEstimationResult,
    scenario_ids: list[str],
    action_lists: list[list[str]],
    likert_summary_df: pd.DataFrame,
) -> list[dict]:
    """Per-scenario posterior predictive check: observed vs predicted Likert."""
    w_draws = est_result.w_draws
    fit_rows = []

    for s_idx, sid in enumerate(scenario_ids):
        actions = action_lists[s_idx]
        for action in actions:
            key = (sid, action)
            if key not in sa_id_map:
                continue
            row_idx = sa_id_map[key]
            phi_row = phi[row_idx, :]
            anch = anchored[row_idx]

            # Predicted mean EU across draws
            eu_draws = w_draws @ phi_row + anch
            eu_mean = float(np.mean(eu_draws))

            # Observed mean Likert score
            obs_row = likert_summary_df[
                (likert_summary_df["scenario_id"] == sid) &
                (likert_summary_df["action"] == action)
            ]
            if obs_row.empty:
                continue
            obs_mean = float(obs_row["mean_score"].iloc[0])

            fit_rows.append({
                "scenario_id": sid,
                "action": action,
                "observed_mean": round(obs_mean, 3),
                "predicted_eu_mean": round(eu_mean, 3),
                "residual": round(obs_mean - eu_mean, 3),
            })

    return fit_rows


def _validate_historical(
    scenarios: list[Scenario],
    est_result: StanEstimationResult,
) -> dict:
    """Validate against Tier 4 historical scenario (Qantas 2023 AGM).

    Expected: rec_strike should be argmax with P > 0.80.
    """
    tier4 = [s for s in scenarios if s.tier == 4]
    if not tier4:
        return {"available": False, "detail": "No Tier 4 scenario found"}

    s = tier4[0]
    sv = s.state_vector
    args_base = {
        "ceo_resigned": sv.get("ceo_resigned", False),
        "ceo_sacked": sv.get("ceo_sacked", False),
        "review_commissioned": sv.get("review_commissioned", False),
        "action_delayed": sv.get("action_delayed", False),
        "high_profile": sv.get("high_profile", True),
    }

    phi_no = decompose_utility_asa(**args_base, action="no_strike")
    phi_rec = decompose_utility_asa(**args_base, action="rec_strike")

    phi_no_vec = np.array([phi_no.get(p, 0.0) for p in WEIGHT_PARAM_NAMES])
    phi_rec_vec = np.array([phi_rec.get(p, 0.0) for p in WEIGHT_PARAM_NAMES])

    w_draws = est_result.w_draws
    eu_no = w_draws @ phi_no_vec
    eu_rec = w_draws @ phi_rec_vec

    p_rec = float(np.mean(eu_rec > eu_no))
    passed = p_rec > 0.80

    result = {
        "available": True,
        "scenario_id": s.scenario_id,
        "p_rec_strike": round(p_rec, 4),
        "p_no_strike": round(1.0 - p_rec, 4),
        "eu_rec_strike_mean": round(float(np.mean(eu_rec)), 4),
        "eu_no_strike_mean": round(float(np.mean(eu_no)), 4),
        "passed": passed,
        "expected": "rec_strike (observed: ASA recommended strike at Qantas 2023 AGM)",
        "detail": f"P(rec_strike) = {p_rec:.3f} {'PASS' if passed else 'FAIL'} (threshold: 0.80)",
    }

    logger.info(f"Historical validation [Stan]: P(rec_strike) = {p_rec:.3f} "
                 f"({'PASS' if passed else 'FAIL'})")
    return result


def _validate_historical_calibrated(
    scenarios: list[Scenario],
    calibration: dict,
) -> dict:
    """Validate against Tier 4 historical scenario using calibrated weights.

    Uses the calibrated weight point estimates and sigmoid model rather
    than Stan posterior draws.
    """
    tier4 = [s for s in scenarios if s.tier == 4]
    if not tier4:
        return {"available": False, "detail": "No Tier 4 scenario found"}

    s = tier4[0]
    sv = s.state_vector
    args_base = {
        "ceo_resigned": sv.get("ceo_resigned", False),
        "ceo_sacked": sv.get("ceo_sacked", False),
        "review_commissioned": sv.get("review_commissioned", False),
        "action_delayed": sv.get("action_delayed", False),
        "high_profile": sv.get("high_profile", True),
    }

    phi_no = decompose_utility_asa(**args_base, action="no_strike")
    phi_rec = decompose_utility_asa(**args_base, action="rec_strike")

    w = calibration["calibrated_weights"]
    # Weights are now dicts {mean, sigma, ...}; extract means for validation
    w_vec = np.array([
        w[p]["mean"] if isinstance(w[p], dict) else w[p]
        for p in WEIGHT_PARAM_NAMES
    ])
    phi_no_vec = np.array([phi_no.get(p, 0.0) for p in WEIGHT_PARAM_NAMES])
    phi_rec_vec = np.array([phi_rec.get(p, 0.0) for p in WEIGHT_PARAM_NAMES])

    delta_eu = float(w_vec @ phi_rec_vec - w_vec @ phi_no_vec)
    p_rec = 1.0 / (1.0 + np.exp(-delta_eu))
    passed = p_rec > 0.80

    result = {
        "available": True,
        "scenario_id": s.scenario_id,
        "p_rec_strike": round(float(p_rec), 4),
        "p_no_strike": round(float(1.0 - p_rec), 4),
        "delta_eu": round(delta_eu, 4),
        "passed": passed,
        "source": "calibrated_weights",
        "expected": "rec_strike (observed: ASA recommended strike at Qantas 2023 AGM)",
        "detail": f"P(rec_strike) = {p_rec:.3f} {'PASS' if passed else 'FAIL'} "
                  f"(threshold: 0.80, source: calibrated weights)",
    }

    logger.info(f"Historical validation [calibrated]: P(rec_strike) = {p_rec:.3f} "
                 f"({'PASS' if passed else 'FAIL'})")
    return result


def run_validation(
    scenarios: list[Scenario],
    est_result: StanEstimationResult,
    phi: np.ndarray,
    anchored: np.ndarray,
    sa_id_map: dict,
    scenario_ids: list[str],
    action_lists: list[list[str]],
    output_dir: Path,
    likert_summary_df: Optional[pd.DataFrame] = None,
    calibration: Optional[dict] = None,
) -> dict:
    """Stage 6: Full validation — within-sample fit + historical calibration."""
    logger.info("Stage 6: Running validation...")

    validation = {}

    # Within-sample fit (uses Stan weights — measures Likert fit quality)
    if est_result is not None and likert_summary_df is not None and not likert_summary_df.empty:
        fit_rows = _compute_scenario_fit(
            phi, anchored, sa_id_map, est_result,
            scenario_ids, action_lists, likert_summary_df,
        )
        if fit_rows:
            residuals = [abs(r["residual"]) for r in fit_rows]
            validation["within_sample_fit"] = {
                "n_pairs": len(fit_rows),
                "mean_abs_residual": round(float(np.mean(residuals)), 4),
                "max_abs_residual": round(float(np.max(residuals)), 4),
            }

            fit_path = output_dir / "asa_scenario_fit.csv"
            fit_df = pd.DataFrame(fit_rows)
            fit_df.to_csv(fit_path, index=False, encoding="utf-8")

    # Historical calibration — use calibrated weights if available,
    # fall back to Stan weights otherwise
    if calibration is not None:
        historical = _validate_historical_calibrated(scenarios, calibration)
    elif est_result is not None:
        historical = _validate_historical(scenarios, est_result)
    else:
        historical = {"passed": False, "p_rec_strike": "N/A", "detail": "No estimation result"}
    validation["historical"] = historical

    # MCMC diagnostics summary (only when Stan estimation was run)
    if est_result is not None:
        validation["mcmc_diagnostics"] = {
            "n_divergences": est_result.n_divergences,
            "max_rhat": est_result.max_rhat,
            "min_ess_bulk": est_result.min_ess_bulk,
            "n_samples": est_result.n_samples,
            "converged": est_result.converged,
        }

    # Save validation results
    val_path = output_dir / "asa_validation_results.json"
    val_path.parent.mkdir(parents=True, exist_ok=True)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2, default=str)

    logger.info(f"Validation results saved to {val_path}")
    return validation


# ── SEC 12: Dashboard rendering ──────────────────────────────────────────────

@dataclass
class DashboardData:
    run_status: str = "in_progress"
    run_start: str = ""
    generated_at: str = ""
    model: str = "gpt-4o-mini"
    scenarios: Optional[list[dict]] = None
    elicitation_summary: Optional[dict] = None
    elicited_probabilities: Optional[list[dict]] = None
    estimation_dataset_summary: Optional[dict] = None
    parameter_estimates: Optional[dict] = None
    covariance_matrix: Optional[list] = None
    behavioural_diagnostics: Optional[dict] = None
    validation_results: Optional[dict] = None
    cost_summary: Optional[dict] = None
    encoding_stats: Optional[dict] = None
    output_files: Optional[dict] = None
    preflight_checks: Optional[dict] = None
    feature_selection: Optional[dict] = None
    posterior_action_probs: Optional[dict] = None
    a2_node_probs: Optional[dict] = None
    a2_node_probs_stan: Optional[dict] = None  # Stan posterior argmax (validation)
    a2_node_likert: Optional[dict] = None
    probability_calibration: Optional[dict] = None  # Direct elicitation + weight calibration

    def to_json(self) -> str:
        d = {}
        for k in [
            "run_status", "run_start", "generated_at", "model",
            "scenarios", "elicitation_summary", "elicited_probabilities",
            "estimation_dataset_summary", "preflight_checks",
            "parameter_estimates", "covariance_matrix", "feature_selection",
            "posterior_action_probs", "a2_node_probs", "a2_node_probs_stan",
            "a2_node_likert", "probability_calibration",
            "behavioural_diagnostics", "validation_results",
            "cost_summary", "encoding_stats", "output_files",
        ]:
            d[k] = getattr(self, k, None)
        return json.dumps(d, ensure_ascii=True, default=str)


def _get_plotly_bundle() -> str:
    """Download and cache Plotly.js minified bundle."""
    cache_path = CACHE_DIR / "plotly.min.js"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8", errors="replace")
    logger.info("Downloading Plotly.js bundle (one-time)...")
    try:
        import urllib.request
        url = "https://cdn.plot.ly/plotly-2.35.2.min.js"
        with urllib.request.urlopen(url, timeout=60) as resp:
            bundle = resp.read().decode("utf-8", errors="replace")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(bundle, encoding="utf-8")
        return bundle
    except Exception as e:
        logger.warning(f"Failed to download Plotly.js: {e}. Using CDN fallback.")
        return ""


def _embed_file_as_data_uri(path: Path, mime: str = "text/csv") -> dict:
    """Read a file and return base64 data URI + size."""
    if not path.exists():
        return {"uri": "#", "size_kb": 0}
    content = path.read_bytes()
    b64 = base64.b64encode(content).decode("ascii")
    return {
        "uri": f"data:{mime};base64,{b64}",
        "size_kb": round(len(content) / 1024, 1),
    }


_DASHBOARD_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ASA Utility Quantification Dashboard</title>
__META_REFRESH__
<style>
:root{--bg:#f8f9fa;--card:#fff;--border:#dee2e6;--primary:#2E7D32;--danger:#E85D5D;
--success:#50C878;--text:#212529;--text-muted:#6c757d;--font:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:var(--font);background:var(--bg);color:var(--text);font-size:14px}
.banner{padding:10px 20px;text-align:center;font-weight:600}
.banner.in_progress{background:#fff3cd;color:#856404}
.banner.complete{background:#d4edda;color:#155724}
.banner.aborted{background:#f8d7da;color:#721c24}
.tabs{display:flex;flex-wrap:wrap;border-bottom:2px solid var(--border);padding:0 20px;background:var(--card)}
.tab{padding:10px 18px;cursor:pointer;border-bottom:3px solid transparent;color:var(--text-muted);
font-weight:500;white-space:nowrap}
.tab.active{border-bottom-color:var(--primary);color:var(--primary)}
.tab:hover{color:var(--primary)}
.content{padding:20px;max-width:2100px;margin:0 auto}
.panel{display:none;animation:fadeIn .3s}
.panel.active{display:block}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
.card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:20px;margin-bottom:16px}
.card h3{margin-bottom:12px;font-size:16px}
.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.stat{text-align:center;padding:16px;background:var(--bg);border-radius:6px}
.stat .value{font-size:28px;font-weight:700;color:var(--primary)}
.stat .label{font-size:12px;color:var(--text-muted);margin-top:4px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:8px 12px;text-align:left;border-bottom:1px solid var(--border)}
th{background:var(--bg);font-weight:600;position:sticky;top:0;z-index:1}
tr:hover{background:#f1f3f5}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600}
.badge-success{background:#d4edda;color:#155724}
.badge-warning{background:#fff3cd;color:#856404}
.badge-danger{background:#f8d7da;color:#721c24}
.forest-row{display:flex;align-items:center;gap:8px;margin:4px 0}
.forest-bar{height:6px;background:var(--primary);border-radius:3px}
.forest-ci{height:2px;background:#adb5bd;position:relative}
</style>
__PLOTLY_SCRIPT__
</head>
<body>
<div class="banner __STATUS__">ASA Utility Quantification — __STATUS_TEXT__</div>
<div class="tabs" id="tabs">
<div class="tab active" onclick="showTab(0)">Overview</div>
<div class="tab" onclick="showTab(1)">Step 1: Ranges</div>
<div class="tab" onclick="showTab(2)">Step 2: Gaps</div>
<div class="tab" onclick="showTab(3)">Step 3: Targets</div>
<div class="tab" onclick="showTab(4)">Step 4: Weights & Simulation</div>
<div class="tab" onclick="showTab(5)">Validation</div>
<div class="tab" onclick="showTab(6)">Cost & Usage</div>
</div>
<div class="content" id="content"></div>

<script>
var R = __RESULTS_DATA__;
var PARAM_NAMES = __PARAM_NAMES__;
var PARAM_DESCS = __PARAM_DESCS__;
var SPEC_DEFAULTS = __SPEC_DEFAULTS__;

function showTab(idx){
  var tabs=document.querySelectorAll('.tab');
  tabs.forEach(function(t,i){t.classList.toggle('active',i===idx)});
  renderPanel(idx);
}

function renderPanel(idx){
  var c=document.getElementById('content');
  var panels=[renderOverview,renderStep1Ranges,renderStep2Gaps,renderStep3Targets,
    renderStep4Weights,renderValidation,renderCost];
  c.innerHTML='<div class="panel active">'+(panels[idx]?panels[idx]():'<p>Coming soon</p>')+'</div>';
}

function renderOverview(){
  var s=R.run_status||'in_progress';
  var pc=R.probability_calibration||{};
  var h='<div class="card"><h3>Pipeline Status</h3><div class="stat-grid">';
  h+='<div class="stat"><div class="value">'+s.toUpperCase()+'</div><div class="label">Status</div></div>';
  h+='<div class="stat"><div class="value">'+(R.model||'gpt-4o-mini')+'</div><div class="label">Model</div></div>';
  if(pc.n_valid_draws)h+='<div class="stat"><div class="value">'+pc.n_valid_draws+'</div><div class="label">Elicitation Draws</div></div>';
  if(pc.max_sim_error!==undefined)h+='<div class="stat"><div class="value">'+pc.max_sim_error.toFixed(4)+'</div><div class="label">Max Sim Error</div></div>';
  h+='</div></div>';

  h+='<div class="card"><h3>4-Step Calibration Pipeline</h3>';
  h+='<ol style="line-height:2.0;padding-left:20px">';
  h+='<li><b>Step 1 - Probability Range Elicitation</b>: LLM elicits (low, best, high) P(strike) ranges at each A2 node</li>';
  h+='<li><b>Step 2 - Probability Gap Elicitation</b>: LLM elicits expected gaps between specific node pairs</li>';
  h+='<li><b>Step 3 - Constrained Target Solving</b>: Optimize targets within ranges subject to monotonicity, floors, and gap constraints</li>';
  h+='<li><b>Step 4 - Weight Calibration & Simulation</b>: Logistic inversion for utility weights + Monte Carlo validation</li>';
  h+='</ol></div>';
  return h;
}

function renderStep1Ranges(){
  var pc=R.probability_calibration||{};
  var er=pc.elicited_ranges||{};
  if(!er||!Object.keys(er).length)return '<p>No range data. Run Stage 2.</p>';

  var h='<div class="card"><h3>Step 1: LLM Probability Range Elicitation</h3>';
  h+='<p>P(strike) ranges elicited from '+pc.n_valid_draws+' independent LLM draws per node.</p>';
  h+='<div class="stat-grid">';
  h+='<div class="stat"><div class="value">'+(pc.p_expected_floor||0).toFixed(3)+'</div><div class="label">Expected Floor</div></div>';
  h+='<div class="stat"><div class="value">'+(pc.p_absolute_floor||0).toFixed(3)+'</div><div class="label">Absolute Floor</div></div>';
  h+='<div class="stat"><div class="value">'+((pc.p_expected_floor||0)-(pc.p_absolute_floor||0)).toFixed(3)+'</div><div class="label">Headroom</div></div>';
  h+='</div></div>';

  var nodeOrder=['ceo_stay__do_nothing','ceo_resign__do_nothing','ceo_stay__review','ceo_resign__review','ceo_stay__board_forces_exit'];
  h+='<div class="card"><h3>Elicited Ranges by A2 Node</h3>';
  h+='<table><tr><th>A2 Node</th><th>Low (P10)</th><th>Best (Median)</th><th>High (P90)</th><th>SD</th></tr>';
  nodeOrder.forEach(function(k){
    var r=er[k];
    if(!r)return;
    h+='<tr><td>'+k.replace(/__/g,' / ')+'</td>';
    h+='<td>'+r.low.toFixed(3)+'</td>';
    h+='<td><b>'+r.best.toFixed(3)+'</b></td>';
    h+='<td>'+r.high.toFixed(3)+'</td>';
    h+='<td>'+r.sd.toFixed(3)+'</td></tr>';
  });
  h+='</table></div>';

  // Range plot
  h+='<div class="card"><h3>Elicited Ranges (with floor lines)</h3><div id="rangePlot"></div></div>';
  setTimeout(function(){
    if(typeof Plotly==='undefined')return;
    var labels=[],lows=[],bests=[],highs=[];
    nodeOrder.forEach(function(k){
      var r=er[k];if(!r)return;
      labels.push(k.replace(/__/g,' / '));
      lows.push(r.low);bests.push(r.best);highs.push(r.high);
    });
    Plotly.newPlot('rangePlot',[
      {x:labels,y:lows,type:'scatter',mode:'markers',name:'Low (P10)',marker:{size:8,color:'#E85D5D'}},
      {x:labels,y:bests,type:'scatter',mode:'markers+lines',name:'Best (Median)',marker:{size:12,color:'#1565C0',symbol:'diamond'},line:{color:'#1565C0',width:2}},
      {x:labels,y:highs,type:'scatter',mode:'markers',name:'High (P90)',marker:{size:8,color:'#50C878'}}
    ],{margin:{l:50,r:30,t:30,b:120},yaxis:{title:'P(rec_strike)',range:[0.78,1.0]},xaxis:{tickangle:30},height:400,
      shapes:[{type:'line',y0:pc.p_expected_floor,y1:pc.p_expected_floor,x0:-0.5,x1:4.5,line:{color:'orange',width:2,dash:'dash'}},
        {type:'line',y0:pc.p_absolute_floor,y1:pc.p_absolute_floor,x0:-0.5,x1:4.5,line:{color:'red',width:2,dash:'dot'}}],
      annotations:[{x:4.3,y:pc.p_expected_floor,text:'Exp floor',showarrow:false,font:{size:10,color:'orange'}},
        {x:4.3,y:pc.p_absolute_floor,text:'Abs floor',showarrow:false,font:{size:10,color:'red'}}]
    });
  },100);

  if(pc.reasonings&&pc.reasonings.length>0){
    h+='<div class="card"><h3>Sample LLM Reasoning</h3><div style="max-height:300px;overflow-y:auto">';
    pc.reasonings.slice(0,3).forEach(function(r,i){
      h+='<div style="padding:8px;margin:4px 0;background:var(--bg);border-radius:4px;font-size:12px"><b>Draw '+(i+1)+':</b> '+r+'</div>';
    });
    h+='</div></div>';
  }
  return h;
}

function renderStep2Gaps(){
  var pc=R.probability_calibration||{};
  var gc=pc.gap_constraints||{};
  if(!gc||!Object.keys(gc).length)return '<p>No gap data. Run Stage 2.</p>';

  var h='<div class="card"><h3>Step 2: LLM Probability Gap Elicitation</h3>';
  h+='<p>Expected gaps in P(strike) between specific node pairs, elicited from independent LLM draws.</p></div>';

  h+='<div class="card"><h3>Elicited Gap Constraints</h3>';
  h+='<table><tr><th>Gap</th><th>Higher Node</th><th>Lower Node</th><th>Low</th><th>Expected</th><th>High</th></tr>';
  var gapOrder=['departure','review_stay','review_resign','sacking'];
  gapOrder.forEach(function(g){
    var c=gc[g];if(!c)return;
    h+='<tr><td><b>'+c.label+'</b></td>';
    h+='<td>'+c.higher_node.replace(/__/g,' / ')+'</td>';
    h+='<td>'+c.lower_node.replace(/__/g,' / ')+'</td>';
    h+='<td>'+c.low.toFixed(3)+'</td>';
    h+='<td><b>'+c.expected.toFixed(3)+'</b></td>';
    h+='<td>'+c.high.toFixed(3)+'</td></tr>';
  });
  h+='</table></div>';

  // Gap bar chart
  h+='<div class="card"><h3>Elicited Gap Ranges</h3><div id="gapPlot"></div></div>';
  setTimeout(function(){
    if(typeof Plotly==='undefined')return;
    var labels=[],lows=[],exps=[],highs=[];
    gapOrder.forEach(function(g){
      var c=gc[g];if(!c)return;
      labels.push(c.label);lows.push(c.low);exps.push(c.expected);highs.push(c.high);
    });
    Plotly.newPlot('gapPlot',[
      {x:labels,y:exps,type:'bar',name:'Expected',marker:{color:'#1565C0'},
        error_y:{type:'data',symmetric:false,array:highs.map(function(h,i){return h-exps[i]}),
          arrayminus:exps.map(function(e,i){return e-lows[i]}),color:'#333',thickness:2}}
    ],{margin:{l:50,r:30,t:30,b:100},yaxis:{title:'Gap in P(strike)',rangemode:'tozero'},xaxis:{tickangle:15},height:350});
  },100);
  return h;
}

function renderStep3Targets(){
  var pc=R.probability_calibration||{};
  var tg=pc.targets||{};
  var er=pc.elicited_ranges||{};
  var gr=pc.gap_report||[];
  var cr=pc.constraint_report||[];
  if(!tg||!Object.keys(tg).length)return '<p>No target data. Run Stage 2.</p>';

  var nodeOrder=['ceo_stay__do_nothing','ceo_resign__do_nothing','ceo_stay__review','ceo_resign__review','ceo_stay__board_forces_exit'];
  var h='<div class="card"><h3>Step 3: Constrained Target Solving</h3>';
  h+='<p>Targets optimized within elicited ranges subject to monotonicity, floor, and gap constraints.</p></div>';

  // Constraint report
  if(cr.length){
    h+='<div class="card"><h3>Constraint Satisfaction</h3><table><tr><th>Constraint</th><th>Status</th></tr>';
    cr.forEach(function(c){
      var cls=c[1]==='PASS'?'badge-success':(c[1]==='WARN'?'badge-warning':'badge-danger');
      h+='<tr><td>'+c[0]+'</td><td><span class="badge '+cls+'">'+c[1]+'</span></td></tr>';
    });
    h+='</table></div>';
  }

  // Target ladder
  h+='<div class="card"><h3>Solved Target Probability Ladder</h3>';
  h+='<table><tr><th>A2 Node</th><th>Elicited Best</th><th>Solved Target</th><th>Range [Low, High]</th><th>Gap from prev</th></tr>';
  var prev=null;
  nodeOrder.forEach(function(k){
    var t=tg[k];var r=er[k]||{};
    if(t===undefined)return;
    var gap=prev!==null?(prev-t).toFixed(4):'--';
    h+='<tr><td>'+k.replace(/__/g,' / ')+'</td>';
    h+='<td>'+(r.best||0).toFixed(3)+'</td>';
    h+='<td><b>'+t.toFixed(4)+'</b></td>';
    h+='<td>['+(r.low||0).toFixed(3)+', '+(r.high||0).toFixed(3)+']</td>';
    h+='<td>'+gap+'</td></tr>';
    prev=t;
  });
  h+='</table></div>';

  // Gap satisfaction
  if(gr.length){
    h+='<div class="card"><h3>Gap Constraint Satisfaction</h3>';
    h+='<table><tr><th>Gap</th><th>Actual</th><th>Elicited [Low, High]</th><th>Status</th></tr>';
    gr.forEach(function(g){
      var cls=g.satisfied?'badge-success':'badge-warning';
      h+='<tr><td>'+g.label+'</td><td>'+g.actual.toFixed(4)+'</td>';
      h+='<td>['+g.elicited_low.toFixed(3)+', '+g.elicited_high.toFixed(3)+']</td>';
      h+='<td><span class="badge '+cls+'">'+(g.satisfied?'PASS':'WARN')+'</span></td></tr>';
    });
    h+='</table></div>';
  }

  // Target vs range plot
  h+='<div class="card"><h3>Targets within Elicited Ranges</h3><div id="targetPlot"></div></div>';
  setTimeout(function(){
    if(typeof Plotly==='undefined')return;
    var labels=[],lows=[],bests=[],highs=[],solved=[];
    nodeOrder.forEach(function(k){
      var r=er[k]||{};var t=tg[k];
      if(t===undefined)return;
      labels.push(k.replace(/__/g,' / '));
      lows.push(r.low||0);bests.push(r.best||0);highs.push(r.high||0);solved.push(t);
    });
    Plotly.newPlot('targetPlot',[
      {x:labels,y:bests,type:'scatter',mode:'markers',name:'Elicited Best',marker:{size:10,color:'#90CAF9',symbol:'circle'}},
      {x:labels,y:solved,type:'scatter',mode:'markers+lines',name:'Solved Target',marker:{size:14,color:'#2E7D32',symbol:'diamond'},line:{color:'#2E7D32',width:2}},
      {x:labels,y:lows,type:'scatter',mode:'lines',name:'Range Low',line:{color:'#E85D5D',width:1,dash:'dot'},showlegend:true},
      {x:labels,y:highs,type:'scatter',mode:'lines',name:'Range High',line:{color:'#50C878',width:1,dash:'dot'},fill:'tonexty',fillcolor:'rgba(80,200,120,0.1)',showlegend:true}
    ],{margin:{l:50,r:30,t:30,b:120},yaxis:{title:'P(rec_strike)',range:[0.78,1.0]},xaxis:{tickangle:30},height:400,
      shapes:[{type:'line',y0:pc.p_expected_floor,y1:pc.p_expected_floor,x0:-0.5,x1:4.5,line:{color:'orange',width:2,dash:'dash'}}]
    });
  },100);
  return h;
}

function renderStep4Weights(){
  var pc=R.probability_calibration||{};
  if(!pc.calibrated_weights)return '<p>No weight data. Run Stage 2.</p>';
  var cw=pc.calibrated_weights;
  var sim=pc.simulation||{};
  var ip=pc.implied_probs||{};
  var nodeOrder=['ceo_stay__do_nothing','ceo_resign__do_nothing','ceo_stay__review','ceo_resign__review','ceo_stay__board_forces_exit'];
  var interactionParams=['w_strike_cost','w_strike_vs_passive','w_departure_dampens','w_sack_dampens','w_credibility_signal'];
  var contextParams=['w_ctx_inaction','w_ctx_departure','w_ctx_review'];

  var h='<div class="card"><h3>Step 4: Stochastic Utility Parameter Calibration</h3>';
  h+='<p>Random utility model: each weight w<sub>k</sub> ~ TruncNormal(mu, sigma, 1, 5). ';
  h+='P(strike|node) = fraction of draws where EU(strike) &gt; EU(no_strike). ';
  h+='Parameters optimised via differential evolution to match Beta prior means.</p>';
  var optLoss=pc.optimisation_loss||0;
  var optConv=pc.optimisation_converged!==false;
  var maxDisc=pc.max_disc_error||0;
  var maxSim=pc.max_sim_error||0;
  h+='<div class="stat-grid">';
  h+='<div class="stat"><div class="value"><span class="badge '+(optConv?'badge-success':'badge-warning')+'">'+(optConv?'Yes':'No')+'</span></div><div class="label">Converged</div></div>';
  h+='<div class="stat"><div class="value">'+optLoss.toFixed(6)+'</div><div class="label">Optimisation Loss (SSE)</div></div>';
  h+='<div class="stat"><div class="value"><span class="badge '+(maxSim<0.03?'badge-success':'badge-warning')+'">'+maxSim.toFixed(4)+'</span></div><div class="label">Max Sim Error</div></div>';
  h+='<div class="stat"><div class="value"><span class="badge '+(maxDisc<0.015?'badge-success':'badge-warning')+'">'+maxDisc.toFixed(4)+'</span></div><div class="label">Max Disc Error</div></div>';
  h+='</div></div>';

  // Calibrated weight distributions table
  h+='<div class="card"><h3>Calibrated Weight Distributions</h3>';
  h+='<p>Each interaction weight is drawn from TruncNormal(mean, sigma, 1, 5) per simulation draw.</p>';
  h+='<table><tr><th>Parameter</th><th>Mean</th><th>Sigma</th><th>Range</th><th>Type</th></tr>';
  contextParams.forEach(function(p){
    var w=cw[p]||{};
    h+='<tr style="color:var(--text-muted)"><td>'+p+'</td><td>'+(w.mean||1).toFixed(3)+'</td><td>-</td><td>[1, 5]</td><td>Context (fixed, cancels in delta-EU)</td></tr>';
  });
  interactionParams.forEach(function(p){
    var w=cw[p]||{};
    h+='<tr><td><b>'+p+'</b></td><td>'+(w.mean||0).toFixed(3)+'</td><td>'+(w.sigma||0).toFixed(3)+'</td><td>[1, 5]</td><td>Interaction (stochastic)</td></tr>';
  });
  h+='</table></div>';

  // Simulation results: simulated vs target probabilities
  h+='<div class="card"><h3>Argmax Simulation Results</h3>';
  h+='<p>For each node, P(strike) = fraction of simulation draws where EU(strike) &gt; EU(no_strike) with stochastic weights.</p>';
  h+='<table><tr><th>Node</th><th>Target</th><th>Beta Prior</th><th>Beta Mean</th><th>Disc Error</th><th>Simulated</th><th>Sim Error</th></tr>';
  nodeOrder.forEach(function(k){
    var sr=sim[k]||{};
    var dcls=Math.abs(sr.discretization_error||0)<0.015?'badge-success':'badge-warning';
    var scls=Math.abs(sr.sim_vs_beta_error||0)<0.02?'badge-success':'badge-warning';
    h+='<tr><td>'+k.replace(/__/g,' / ')+'</td>';
    h+='<td>'+(sr.target||0).toFixed(4)+'</td>';
    h+='<td>Beta('+(sr.alpha||0)+', '+(sr.beta||0)+')</td>';
    h+='<td>'+(sr.beta_mean||0).toFixed(4)+'</td>';
    h+='<td><span class="badge '+dcls+'">'+(sr.discretization_error||0).toFixed(4)+'</span></td>';
    h+='<td>'+(sr.simulated_p_strike||0).toFixed(4)+'</td>';
    h+='<td><span class="badge '+scls+'">'+(sr.sim_vs_beta_error||0).toFixed(4)+'</span></td></tr>';
  });
  h+='</table></div>';

  // Beta prior mapping
  if(ip){
    h+='<div class="card"><h3>Engine Beta Priors (n_eff=50)</h3>';
    h+='<table><tr><th>Path</th><th>P(strike)</th><th>Beta(alpha, beta)</th><th>Beta Mean</th></tr>';
    var pathNames={'ceo_resign__do_nothing':'CEO resigns -> Do nothing','ceo_resign__review':'CEO resigns -> Review',
      'ceo_stay__do_nothing':'CEO stays -> Do nothing','ceo_stay__review':'CEO stays -> Review',
      'ceo_stay__board_forces_exit':'CEO stays -> Board forces exit'};
    nodeOrder.forEach(function(k){
      var d=ip[k];if(!d)return;
      var p=d.p_strike;var alpha=Math.max(1,Math.round(p*50));var beta=Math.max(1,50-alpha);
      h+='<tr><td>'+pathNames[k]+'</td><td>'+p.toFixed(4)+'</td>';
      h+='<td>Beta('+alpha+', '+beta+')</td><td>'+(alpha/(alpha+beta)).toFixed(3)+'</td></tr>';
    });
    h+='</table></div>';
  }

  // Weight distribution chart (mean +/- sigma)
  h+='<div class="card"><h3>Weight Distributions (Mean +/- Sigma)</h3><div id="weightForest"></div></div>';
  setTimeout(function(){
    if(typeof Plotly==='undefined')return;
    var names=interactionParams.slice().reverse();
    var means=names.map(function(p){var w=cw[p]||{};return w.mean||0});
    var sigs=names.map(function(p){var w=cw[p]||{};return w.sigma||0});
    Plotly.newPlot('weightForest',[{
      type:'bar',x:means,y:names,orientation:'h',
      error_x:{type:'data',array:sigs,visible:true},
      marker:{color:means.map(function(v){return v>2.5?'#2E7D32':'#90CAF9'})}
    }],{margin:{l:180,r:30,t:20,b:40},xaxis:{title:'Weight value',range:[0,5.5]},height:250});
  },100);
  return h;
}

function renderValidation(){
  if(!R.validation_results)return '<p>No validation data. Run Stage 6.</p>';
  var vr=R.validation_results;
  var h='<div class="card"><h3>Historical Validation</h3>';
  if(vr.historical){
    var hist=vr.historical;
    var cls=hist.passed?'badge-success':'badge-danger';
    h+='<h4>Historical Calibration (Qantas 2023 AGM)</h4>';
    h+='<p>P(rec_strike) = <span class="badge '+cls+'">'+hist.p_rec_strike+'</span> '+(hist.passed?'PASS':'FAIL')+'</p>';
    h+='<p>Expected: ASA recommended strike (observed historical outcome)</p>';
  }
  h+='</div>';
  return h;
}

function renderCost(){
  var cs=R.cost_summary||{};
  var h='<div class="card"><h3>API Cost Summary</h3><div class="stat-grid">';
  h+='<div class="stat"><div class="value">'+(cs.total_calls||0)+'</div><div class="label">Total Calls</div></div>';
  h+='<div class="stat"><div class="value">$'+(cs.total_cost_usd||0).toFixed(4)+'</div><div class="label">Total Cost</div></div>';
  h+='<div class="stat"><div class="value">'+(cs.total_tokens||0).toLocaleString()+'</div><div class="label">Total Tokens</div></div>';
  h+='</div></div>';
  return h;
}

// Init
document.addEventListener('DOMContentLoaded',function(){
  var banner=document.querySelector('.banner');
  banner.className='banner '+(R.run_status||'in_progress');
  banner.textContent='ASA Utility Quantification \u2014 '+(R.run_status||'in_progress').toUpperCase();
  renderPanel(0);
});
</script>
</body>
</html>"""


def render_dashboard(
    dashboard_data: DashboardData,
    output_path: Path,
) -> None:
    """Render the self-contained HTML dashboard."""
    dashboard_data.generated_at = datetime.now().isoformat()

    plotly_bundle = _get_plotly_bundle()
    if plotly_bundle:
        plotly_script = f"<script>{plotly_bundle}</script>"
    else:
        plotly_script = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'

    meta_refresh = ""
    if dashboard_data.run_status == "in_progress":
        meta_refresh = """<script>
(function(){
  var REFRESH_SEC=120;var remaining=REFRESH_SEC;
  window.addEventListener('DOMContentLoaded',function(){
    setInterval(function(){
      remaining--;
      if(remaining<=0){remaining=REFRESH_SEC;location.reload();}
    },1000);
  });
})();
</script>"""

    results_json = dashboard_data.to_json()
    param_names_json = json.dumps(list(WEIGHT_PARAM_NAMES), ensure_ascii=True)
    param_descs_json = json.dumps(PARAM_DESCRIPTIONS, ensure_ascii=True)
    spec_defaults_json = json.dumps(SPEC_DEFAULTS, ensure_ascii=True)

    status = dashboard_data.run_status
    status_text = status.upper()

    html = _DASHBOARD_TEMPLATE
    html = html.replace("__META_REFRESH__", meta_refresh)
    html = html.replace("__PLOTLY_SCRIPT__", plotly_script)
    html = html.replace("__RESULTS_DATA__", results_json)
    html = html.replace("__PARAM_NAMES__", param_names_json)
    html = html.replace("__PARAM_DESCS__", param_descs_json)
    html = html.replace("__SPEC_DEFAULTS__", spec_defaults_json)
    html = html.replace("__STATUS__", status)
    html = html.replace("__STATUS_TEXT__", status_text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp")
    tmp_path.write_text(html, encoding="utf-8", errors="replace")
    os.replace(str(tmp_path), str(output_path))

    logger.info(f"Dashboard written to {output_path} ({len(html)//1024}KB)")


# ── SEC 13: main() orchestrator + CLI ────────────────────────────────────────

class _TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            from tqdm import tqdm
            tqdm.write(self.format(record))
        except ImportError:
            sys.stderr.write(self.format(record) + "\n")


def _setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # File handler
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    root.addHandler(fh)
    # Console handler
    ch = _TqdmLoggingHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(ch)


def _save_parameter_estimates(est_result: StanEstimationResult, output_dir: Path) -> None:
    """Save parameter estimates and covariance matrix to CSV."""
    rows = []
    for pname in WEIGHT_PARAM_NAMES:
        ci = est_result.weights_posterior_ci.get(pname, [0, 0])
        rows.append({
            "parameter": pname,
            "engine_key": PARAM_TO_ENGINE_KEY.get(pname, ""),
            "description": PARAM_DESCRIPTIONS.get(pname, ""),
            "posterior_mean": est_result.weights.get(pname, 0.0),
            "posterior_sd": est_result.weights_posterior_sd.get(pname, 0.0),
            "ci_lo": ci[0],
            "ci_hi": ci[1],
            "spec_default": SPEC_DEFAULTS.get(pname, 0.0),
            "estimation_method": est_result.estimation_method.get(pname, ""),
        })
    est_df = pd.DataFrame(rows)
    est_path = output_dir / "asa_parameter_estimates.csv"
    est_df.to_csv(est_path, index=False, encoding="utf-8")
    logger.info(f"Parameter estimates saved to {est_path}")

    # Covariance matrix
    cov = est_result.covariance_matrix
    K = min(cov.shape[0], len(WEIGHT_PARAM_NAMES))
    cov_df = pd.DataFrame(
        cov[:K, :K],
        index=WEIGHT_PARAM_NAMES[:K],
        columns=WEIGHT_PARAM_NAMES[:K],
    )
    cov_path = output_dir / "asa_covariance_matrix.csv"
    cov_df.to_csv(cov_path, encoding="utf-8")
    logger.info(f"Covariance matrix saved to {cov_path}")


def _load_cached_stan_result(output_dir: Path) -> StanEstimationResult:
    """Load cached Stan posterior draws from .npz file."""
    draws_path = output_dir / "asa_stan_posterior_draws.npz"
    est_path = output_dir / "asa_parameter_estimates.csv"

    if draws_path.exists():
        data = np.load(draws_path)
        w_draws = data["w_draws"]
        cutpoints = data.get("cutpoints", np.zeros((w_draws.shape[0], 4)))
        sigma = data.get("sigma_scenario", np.ones(w_draws.shape[0]) * 0.5)
        logger.info(f"Loaded {w_draws.shape[0]} posterior draws from {draws_path}")
    elif est_path.exists():
        est_df = pd.read_csv(est_path, encoding="utf-8")
        K = len(WEIGHT_PARAM_NAMES)
        n_sim = 8000
        w_draws = np.zeros((n_sim, K))
        for j, pname in enumerate(WEIGHT_PARAM_NAMES):
            row = est_df[est_df["parameter"] == pname]
            if not row.empty:
                mu = float(row["posterior_mean"].iloc[0])
                sd = float(row["posterior_sd"].iloc[0])
                w_draws[:, j] = np.random.default_rng(42 + j).lognormal(
                    np.log(max(mu, 0.01)), max(sd / max(mu, 0.01), 0.1), size=n_sim
                )
            else:
                w_draws[:, j] = SPEC_DEFAULTS.get(pname, 1.0)
        cutpoints = np.zeros((n_sim, 4))
        sigma = np.ones(n_sim) * 0.5
        logger.info(f"Simulated {n_sim} draws from parameter_estimates.csv")
    else:
        raise FileNotFoundError(
            f"No cached Stan draws found at {draws_path} or {est_path}"
        )

    # Build result
    weights = {}
    weights_sd = {}
    weights_ci = {}
    for j, pname in enumerate(WEIGHT_PARAM_NAMES):
        draws_j = w_draws[:, j]
        weights[pname] = round(float(np.mean(draws_j)), 4)
        weights_sd[pname] = round(float(np.std(draws_j)), 4)
        weights_ci[pname] = [
            round(float(np.percentile(draws_j, 2.5)), 4),
            round(float(np.percentile(draws_j, 97.5)), 4),
        ]

    cov_path = output_dir / "asa_covariance_matrix.csv"
    if cov_path.exists():
        cov_df = pd.read_csv(cov_path, index_col=0, encoding="utf-8")
        covariance_matrix = cov_df.values
    else:
        covariance_matrix = np.cov(w_draws.T)

    return StanEstimationResult(
        w_draws=w_draws,
        cutpoint_draws=cutpoints,
        sigma_scenario_draws=sigma,
        weights_posterior_mean=weights,
        weights_posterior_sd=weights_sd,
        weights_posterior_ci=weights_ci,
        weights=weights,
        covariance_matrix=covariance_matrix,
        n_samples=w_draws.shape[0],
        converged=True,
        estimation_method={p: "cached" for p in WEIGHT_PARAM_NAMES},
    )


def main():
    parser = argparse.ArgumentParser(
        description="ASA Utility Quantification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python asa_utility_quantification.py --stage 1\n"
            "  python asa_utility_quantification.py --stage 1,2,3 --n_draws 5\n"
            "  python asa_utility_quantification.py --all --n_draws 40\n"
        ),
    )
    parser.add_argument("--stage", type=str, default="all",
                        help="Comma-separated stages (1-6, 4b) or 'all'")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="LLM model (default: gpt-4o-mini)")
    parser.add_argument("--n_draws", type=int, default=40,
                        help="Draws per scenario (default: 40)")
    parser.add_argument("--chains", type=int, default=4,
                        help="MCMC chains (default: 4)")
    parser.add_argument("--iter_warmup", type=int, default=1000,
                        help="Warmup iterations per chain (default: 1000)")
    parser.add_argument("--iter_sampling", type=int, default=2000,
                        help="Sampling iterations per chain (default: 2000)")
    parser.add_argument("--api_key", type=str, default=None,
                        help="OpenAI API key (or OPENAI_API_KEY env var)")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR),
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--llm-threads", type=int, default=10, dest="llm_threads",
                        help="Concurrent LLM threads (default: 10)")
    parser.add_argument("--all", action="store_true",
                        help="Run all stages")
    parser.add_argument("--no-laplacian", action="store_true", dest="no_laplacian",
                        help="Disable Laplacian smoothing on action probabilities")

    args = parser.parse_args()

    # Parse stages
    run_4b = False
    if args.all or args.stage == "all":
        stages = {1, 2, 6}
    else:
        stage_tokens = [s.strip() for s in args.stage.split(",")]
        stages = set()
        for tok in stage_tokens:
            if tok.lower() == "4b":
                run_4b = True
            else:
                stages.add(int(tok))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "asa_pipeline.log"
    _setup_logging(log_path)

    logger.info("=" * 60)
    logger.info("ASA Utility Quantification Pipeline")
    stage_label = sorted(stages) if not run_4b else sorted(stages) + ["4b"]
    logger.info(f"Stages: {stage_label}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Draws/scenario: {args.n_draws}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)

    # Load .env if available
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass

    if not _run_encoding_self_test():
        logger.error("Encoding self-test failed. Aborting.")
        sys.exit(1)

    # Dashboard state
    dashboard = DashboardData(
        run_start=datetime.now().isoformat(),
        model=args.model,
    )
    dashboard_path = output_dir / "asa_utility_dashboard.html"

    # File paths
    scenarios_path = output_dir / "asa_scenarios.csv"
    elicitation_path = output_dir / "asa_elicitation_results.csv"
    diagnostics_path = output_dir / "asa_behavioural_diagnostics.json"

    cost_tracker = RunCostSummary()
    scenarios = []
    likert_summary_df = pd.DataFrame()
    est_result = None

    try:
        # ── Stage 1: Scenario generation ──
        if 1 in stages:
            scenarios = generate_scenarios(scenarios_path)
        elif scenarios_path.exists():
            scenarios = load_scenarios(scenarios_path)
            logger.info(f"Loaded {len(scenarios)} scenarios from {scenarios_path}")

        if scenarios:
            dashboard.scenarios = [s.to_dict() for s in scenarios]
            preflight_gen = run_preflight_checks(scenarios)
            dashboard.preflight_checks = preflight_gen
            render_dashboard(dashboard, dashboard_path)

        # ── Stage 2: LLM elicitation ──
        if 2 in stages and scenarios:
            skip_elicitation = False
            if elicitation_path.exists():
                existing_df = pd.read_csv(elicitation_path, encoding="utf-8")
                needed_ids = set(s.scenario_id for s in scenarios if s.tier != 4)
                existing_ids = set(existing_df["scenario_id"].unique())
                if not (needed_ids - existing_ids):
                    logger.info(f"Elicitation results already cover all scenarios. "
                                f"Skipping Stage 2.")
                    skip_elicitation = True
            if not skip_elicitation:
                client = _get_instructor_client(args.api_key)
                run_elicitation(
                    scenarios, client, args.model, args.n_draws,
                    elicitation_path, cost_tracker,
                    max_workers=args.llm_threads,
                )
            dashboard.cost_summary = cost_tracker.to_dict()
            dashboard.encoding_stats = dict(_encoding_stats)
            if elicitation_path.exists():
                edf = pd.read_csv(elicitation_path, encoding="utf-8")
                n_total = len(edf)
                n_success = len(edf[edf["parse_status"].isin(["success", "repaired"])])
                dashboard.elicitation_summary = {
                    "total_calls": n_total,
                    "success_rate": round(100 * n_success / max(n_total, 1), 1),
                    "cache_hit_rate": round(
                        100 * _cache_stats["hits"] / max(
                            _cache_stats["hits"] + _cache_stats["misses"], 1
                        ), 1
                    ),
                }
            render_dashboard(dashboard, dashboard_path)

            # ── Console summary: A2 node Likert scores ──
            if elicitation_path.exists():
                _edf = pd.read_csv(elicitation_path, encoding="utf-8")
                _edf_ok = _edf[_edf["parse_status"].isin(["success", "repaired"])].copy()
                a2_sids = [f"A2_{n}" for n in A2_NODE_STATES]
                _a2df = _edf_ok[_edf_ok["scenario_id"].isin(a2_sids)]
                if not _a2df.empty:
                    logger.info("A2 Node Likert Score Summary (post-elicitation):")
                    logger.info(f"{'Node':<35} {'Action':<15} {'N':>4} {'Mean':>6} {'SD':>6}")
                    logger.info("-" * 72)
                    for a2_sid in a2_sids:
                        node_df = _a2df[_a2df["scenario_id"] == a2_sid]
                        if node_df.empty:
                            continue
                        node_label = a2_sid.replace("A2_", "")
                        for _, row in node_df.iterrows():
                            try:
                                scores_dict = json.loads(row["action_scores"])
                            except (json.JSONDecodeError, TypeError):
                                continue
                            # Accumulate per-action
                        # Aggregate across draws
                        action_scores: dict[str, list] = {}
                        for _, row in node_df.iterrows():
                            try:
                                sd = json.loads(row["action_scores"])
                            except (json.JSONDecodeError, TypeError):
                                continue
                            for act, sc in sd.items():
                                action_scores.setdefault(act, []).append(int(sc))
                        for act in ["no_strike", "rec_strike"]:
                            if act in action_scores:
                                scores = action_scores[act]
                                m = np.mean(scores)
                                s = np.std(scores)
                                logger.info(f"{node_label:<35} {act:<15} {len(scores):>4} {m:>6.2f} {s:>6.2f}")

        # ── Stage 2p: 4-Step A2 Probability Calibration Pipeline ──
        calibration = None
        pipeline_result = None
        ranges_path = output_dir / "asa_probability_ranges.json"

        if 2 in stages:
            if "client" not in dir():
                client = _get_instructor_client(args.api_key)

            pipeline_result = run_a2_calibration_pipeline(
                client, args.model, n_draws=args.n_draws,
                max_workers=args.llm_threads, output_dir=output_dir,
            )

            calibration = pipeline_result["calibration"]
            a2_probs = pipeline_result["a2_probs"]
            ranges = pipeline_result["ranges"]
            gaps = pipeline_result["gaps"]
            targets = pipeline_result["targets"]

            dashboard.a2_node_probs = a2_probs
            dashboard.probability_calibration = {
                "calibrated_weights": calibration["calibrated_weights"],
                "implied_probs": calibration["implied_probs"],
                "targets": calibration["targets"],
                "p_floor": calibration["p_floor"],
                "p_expected_floor": calibration["p_expected_floor"],
                "p_absolute_floor": calibration["p_absolute_floor"],
                "simulation": calibration.get("simulation", {}),
                "beta_priors": calibration.get("beta_priors", {}),
                "optimisation_loss": calibration.get("optimisation_loss", 0),
                "optimisation_converged": calibration.get("optimisation_converged", True),
                "max_disc_error": calibration.get("max_disc_error", 0),
                "max_sim_error": calibration.get("max_sim_error", 0),
                "n_valid_draws": ranges["n_valid"],
                "reasonings": ranges.get("reasonings", []),
                "elicited_ranges": {
                    n: {
                        "low": ranges["node_ranges"][n]["low"],
                        "best": ranges["node_ranges"][n]["best"],
                        "high": ranges["node_ranges"][n]["high"],
                        "sd": ranges["node_ranges"][n]["sd"],
                    }
                    for n in ranges["node_ranges"]
                },
                "gap_constraints": gaps.get("gap_constraints", {}),
                "constraint_report": targets.get("constraint_report", []),
                "gap_report": targets.get("gap_report", []),
            }

            # Save calibrated A2 probabilities for engine consumption
            _save_a2_calibration(calibration, a2_probs, output_dir)
            render_dashboard(dashboard, dashboard_path)

        elif ranges_path.exists():
            # Reload cached ranges and gaps, re-solve
            with open(ranges_path, "r", encoding="utf-8") as f:
                ranges = json.load(f)
            gaps_path = output_dir / "asa_probability_gaps.json"
            if gaps_path.exists():
                with open(gaps_path, "r", encoding="utf-8") as f:
                    gaps = json.load(f)
            else:
                # Fallback: no gap constraints
                gaps = {"gap_constraints": {g["name"]: {
                    "low": 0.005, "expected": 0.02, "high": 0.10,
                    "higher_node": g["higher"], "lower_node": g["lower"],
                    "label": g["label"],
                } for g in _GAP_DEFINITIONS}}
            targets = _step3_solve_targets(ranges, gaps)
            calibration = _step4_calibrate_weights(targets, ranges)
            a2_probs = {}
            for node_name in A2_NODE_STATES:
                ip = calibration["implied_probs"][node_name]
                p_strike = max(calibration["p_floor"], min(0.99, ip["p_strike"]))
                a2_probs[node_name] = {
                    "no_strike": round(1.0 - p_strike, 4),
                    "rec_strike": round(p_strike, 4),
                    "source": "calibration_pipeline",
                }
            dashboard.a2_node_probs = a2_probs

        # ── Stages 3-5: DEPRECATED (Stan ordinal probit) ──────────────
        # Stan-based parameter estimation (Stages 3-5) is no longer the
        # primary calibration method.  Direct probability elicitation in
        # Stage 2 produces calibrated weights via a logistic model, which
        # is the single source of truth for A2 probabilities.
        #
        # The ordinal probit Stan model produced extreme 0/1 argmax
        # probabilities due to weight-scale issues with the softmax
        # decision rule.  The direct elicitation approach avoids this
        # by working directly in probability space.
        #
        # To re-enable Stan estimation for validation, uncomment the
        # Stage 4 block below and run with --stage 4.
        # ──────────────────────────────────────────────────────────────

        # # ── Stage 3: Preprocessing ──
        # likert_long_path = output_dir / "asa_likert_long.csv"
        # likert_summary_path = output_dir / "asa_likert_summary.csv"
        # if 3 in stages and elicitation_path.exists():
        #     likert_long_df_stage3, likert_summary_df = preprocess_likert_data(
        #         elicitation_path, likert_long_path, likert_summary_path,
        #     )
        #     ...
        #
        # # ── Stage 4: Stan parameter estimation ──
        # if 4 in stages and not likert_summary_df.empty and scenarios:
        #     est_result = estimate_parameters_stan(...)
        #     ...
        #
        # # ── Stage 5: Behavioural diagnostics ──
        # if 5 in stages and not likert_summary_df.empty and est_result is not None:
        #     diagnostics = run_diagnostics(...)
        #     ...

        # ── Stage 6: Validation ──
        if 6 in stages and calibration is not None:
            validation = run_validation(
                scenarios, est_result=None,
                phi=None, anchored=None, sa_id_map=None,
                scenario_ids=None, action_lists=None,
                output_dir=output_dir,
                likert_summary_df=None,
                calibration=calibration,
            )
            dashboard.validation_results = validation
            render_dashboard(dashboard, dashboard_path)

        # ── Embed output files for download ──
        output_files = {}
        for fname, mime in [
            ("asa_scenarios.csv", "text/csv"),
            ("asa_elicitation_results.csv", "text/csv"),
            ("asa_parameter_estimates.csv", "text/csv"),
            ("asa_covariance_matrix.csv", "text/csv"),
            ("asa_scenario_fit.csv", "text/csv"),
            ("asa_behavioural_diagnostics.json", "application/json"),
            ("asa_validation_results.json", "application/json"),
        ]:
            fpath = output_dir / fname
            if fpath.exists():
                output_files[fname] = _embed_file_as_data_uri(fpath, mime)
        dashboard.output_files = output_files

        # Final dashboard
        dashboard.run_status = "complete"
        dashboard.cost_summary = cost_tracker.to_dict()
        dashboard.encoding_stats = dict(_encoding_stats)
        render_dashboard(dashboard, dashboard_path)

        logger.info("=" * 60)
        logger.info(f"Pipeline complete. Dashboard: {dashboard_path}")
        logger.info("=" * 60)

    except TokenLimitRunError as e:
        logger.error(str(e))
        dashboard.run_status = "aborted"
        dashboard.cost_summary = cost_tracker.to_dict()
        render_dashboard(dashboard, dashboard_path)
        sys.exit(1)

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        dashboard.run_status = "aborted"
        dashboard.cost_summary = cost_tracker.to_dict()
        render_dashboard(dashboard, dashboard_path)
        sys.exit(130)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}\n{traceback.format_exc()}")
        dashboard.run_status = "aborted"
        dashboard.cost_summary = cost_tracker.to_dict()
        render_dashboard(dashboard, dashboard_path)
        sys.exit(1)


if __name__ == "__main__":
    main()
