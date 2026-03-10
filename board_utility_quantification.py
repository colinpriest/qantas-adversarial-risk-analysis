"""
Board Utility Quantification Pipeline
======================================
6-stage pipeline for estimating Board utility function parameters
using LLM stakeholder simulation (gpt-4o-mini via instructor).

Spec: utility-quantification/ara_board_utility_experiment_spec.md
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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator, model_validator

logger = logging.getLogger("board_utility_quantification")

# ── Anchored parameters (not optimised) ──
W_CAR_ANCHOR = 15.0
W_COST_ANCHOR = 15.0
LAMBDA_LA_DEFAULT = 2.25
W_CAR_POS = W_CAR_ANCHOR / ((1 + LAMBDA_LA_DEFAULT) / 2)   # ~9.23
W_CAR_NEG = LAMBDA_LA_DEFAULT * W_CAR_POS                   # ~20.77

# ── Parameter classification ──
# Softmax P(a|s;w,λ) = exp(λ·EU(a)) / Σ exp(λ·EU(a'))
#
# All estimable weights are strictly positive (w > 0) with lognormal priors
# centred at spec defaults.  No hidden transforms — posterior draws of w are
# the utility weights directly.  Parameters that don't contribute to the
# model are EXCLUDED entirely (no phi column, no parameter slot).
#
# ESTIMABLE (action-varying phi, estimated via ordinal probit, w > 0):
#
# INACTION COMPONENTS (2 additive, unconditional — fire regardless of vote):
#   w_inaction_base       — Board took minimal action at ALL decision points
#                           phi = -I[board_inactive]
#   w_inaction_no_review  — No governance review commissioned
#                           phi = -I[not review_commissioned]
# RETAINED:
#   w_passivity — Board passivity after CEO departure (zero when Board responds)
#   w_removal  — CEO involuntary removal cost (implementation + disruption)
#   w_remove_ceo_overwhelming        — CEO removal shock relief (overwhelming)
#   w_review_negative    — negative review finding penalty
#   w_review_balanced    — balanced review finding penalty
#
# VOTE PENALTIES (scenario-level, estimated from LLM severity ratings):
#   w_strike       — w × max(0,(V-0.25)/0.75)  where w > 0 (lognormal prior)
#   w_overwhelming — w × max(0,(V-0.50)/0.50)  where w > 0 (lognormal prior)
#
# EXCLUDED (removed from model — no phi column, not in likelihood):
#   w2, w3, w4, w_inaction (old), w13, w8r, w8s, w9, w12
#
ESTIMABLE_PARAM_NAMES = [
    "w_inaction_base", "w_inaction_no_review", "w_inaction_delay",
    "w_passivity", "w_removal", "w_remove_ceo_overwhelming", "w_review_negative", "w_review_balanced",
    "w_review_post_removal",
    "w_ceo_accountability",
]  # 10 parameters estimated via ordinal probit

# Vote penalty parameters — estimated from scenario-level LLM severity ratings.
# Scenario-level (don't vary by action within a scenario).
# Entered as anchored constants during softmax estimation.
VOTE_PARAM_NAMES = ["w_strike", "w_overwhelming"]
VOTE_PARAM_DEFAULTS = {
    "w_strike": 2.0,        # vote penalty escalation past first strike
    "w_overwhelming": 3.0,  # additional escalation past overwhelming vote
}
# Back-compat: empty dict (vote params now estimated, no longer anchored)
ANCHORED_VOTE_PARAMS = {}

# All weight parameters — for display/decomposition
ALL_WEIGHT_NAMES = (ESTIMABLE_PARAM_NAMES
                    + VOTE_PARAM_NAMES)

# For estimation: the estimable parameters (lambda fixed at 1.0, not estimated)
WEIGHT_PARAM_NAMES = ESTIMABLE_PARAM_NAMES  # 7 linear weights in phi

PARAM_TO_ENGINE_KEY = {
    "w_inaction_base": "inaction_base_penalty",
    "w_inaction_no_review": "inaction_no_review_penalty",
    "w_inaction_delay": "inaction_delay_penalty",
    "w_passivity": "board_passivity_after_departure",
    "w_removal": "implementation_cost_sack + ceo_loss_cost",
    "w_remove_ceo_overwhelming": "ceo_loss_shock_overwhelming",
    "w_review_negative": "negative_review_finding_penalty",
    "w_review_balanced": "balanced_review_finding_penalty",
    "w_review_post_removal": "review_after_removal_penalty",
    "w_ceo_accountability": "ceo_accountability_benefit",
    "w_strike": "vote_strike_penalty",
    "w_overwhelming": "vote_overwhelming_penalty",
}

PARAM_DESCRIPTIONS = {
    "w_inaction_base": "Board inaction: minimal action at all decision points",
    "w_inaction_no_review": "Board inaction: no governance review commissioned",
    "w_inaction_delay": "Board delay: acted at D_rev after doing nothing at D1 (reactive vs proactive)",
    "w_passivity": "Board passivity after CEO departure (zero when Board responds decisively)",
    "w_removal": "CEO removal cost (implementation + disruption)",
    "w_remove_ceo_overwhelming": "CEO removal shock relief (overwhelming)",
    "w_review_negative": "Negative review finding penalty",
    "w_review_balanced": "Balanced review finding penalty",
    "w_review_post_removal": "No governance review after involuntary CEO removal",
    "w_ceo_accountability": "Accountability benefit: CEO removal backed by governance review",
    "w_strike": "Vote penalty from first strike (scenario-level)",
    "w_overwhelming": "Additional vote penalty from overwhelming (scenario-level)",
    "lambda_rationality": "Rationality (inverse temperature) [fixed at 1.0]",
}

# Spec defaults: used ONLY for optimizer initialization and dashboard display.
# NOT used for regularization or anchoring.
SPEC_DEFAULTS = {
    "w_inaction_base": 3.0,        # replaces w13
    "w_inaction_no_review": 2.0,   # moderate penalty for no review
    "w_inaction_delay": 1.5,       # reactive governance penalty (D1=minimal then D_rev acts)
    "w_passivity": 0.5,
    "w_removal": 1.8,   # w7(0.3) + w8(1.5)
    "w_remove_ceo_overwhelming": 0.5,
    "w_review_negative": 5.0,
    "w_review_balanced": 2.5,
    "w_review_post_removal": 3.0, # due diligence: review after CEO removal
    "w_ceo_accountability": 3.0,  # accountability benefit of evidence-based CEO removal
    "w_strike": 2.0,              # vote penalty escalation past first strike
    "w_overwhelming": 3.0,        # additional escalation past overwhelming vote
    "lambda_rationality": 1.0,
}


MODEL_PRICE_TABLE = {
    # 4o family
    "gpt-4o-mini": {
        "prompt_per_1k":     0.00015,  # $0.15 / 1M input
        "completion_per_1k": 0.00060,  # $0.60 / 1M output
    },
    "gpt-4o": {
        "prompt_per_1k":     0.00250,  # $2.50 / 1M input
        "completion_per_1k": 0.01000,  # $10.00 / 1M output
    },
    # 5 family
    "gpt-5-mini": {
        "prompt_per_1k":     0.00025,  # $0.25 / 1M input
        "completion_per_1k": 0.00200,  # $2.00 / 1M output
    },
    "gpt-5.2": {
        "prompt_per_1k":     0.00175,  # $1.75 / 1M input
        "completion_per_1k": 0.01400,  # $14.00 / 1M output
    },
    "gpt-5.4": {
        "prompt_per_1k":     0.00250,  # $2.50 / 1M input
        "completion_per_1k": 0.01500,  # $15.00 / 1M output
    },
}

LIKERT_SCALE_LABELS = {
    1: "strongly unattractive",
    2: "somewhat unattractive",
    3: "neutral",
    4: "somewhat attractive",
    5: "strongly attractive",
}

VOTE_GRID = [0.10, 0.20, 0.26, 0.30, 0.40, 0.50, 0.60, 0.75, 0.83]
CAR_GRID = [-0.14, -0.08, -0.05, -0.03, -0.01, 0.00, 0.01, 0.03, 0.05, 0.08, 0.14]

PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR = PROJECT_ROOT / "utility-quantification" / "cache"


# ── SEC 1: Pydantic schemas ──────────────────────────────────────────────────

class ActionCode(str, Enum):
    D0_MINIMAL = "D0_minimal"
    D1_REVIEW = "D1_review"
    D3_CEO_TRANSITION = "D3_ceo_transition"
    DREV_NO_ACTION = "Drev_no_action"
    DREV_COMMISSION_REVIEW = "Drev_commission_review"
    DREV_SACK_CEO = "Drev_sack_ceo"


class DecisionNodeType(str, Enum):
    D1 = "D1"
    D_REV = "D_rev"
    D_REV_POST = "D_rev_post"


class ParseStatus(str, Enum):
    SUCCESS = "success"
    FORMAT_ERROR = "format_error"
    TOKEN_LIMIT = "token_limit"
    REPAIRED = "repaired"


FEASIBLE_ACTIONS_MAP = {
    "D1": [ActionCode.D0_MINIMAL, ActionCode.D1_REVIEW, ActionCode.D3_CEO_TRANSITION],
    "D_rev": [ActionCode.DREV_NO_ACTION, ActionCode.DREV_COMMISSION_REVIEW, ActionCode.DREV_SACK_CEO],
    "D_rev_post": [ActionCode.DREV_NO_ACTION, ActionCode.DREV_SACK_CEO],
}


class ActionLikertScore(BaseModel):
    action: ActionCode
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
        # Ensure no duplicate actions
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
    "\u2014": "--", "\u2013": "-",
    "\u2026": "...", "\u00A0": " ",
    "\u2022": "-", "\uFEFF": "",
    "\u200B": "",
})

_encoding_stats = {"replacements": 0, "non_ascii": 0, "bom": 0, "zwsp": 0}
_encoding_lock = threading.Lock()


def sanitise_text(s: str) -> str:
    """Sanitise LLM output to ASCII-safe text."""
    global _encoding_stats
    if isinstance(s, bytes):
        s = s.decode("utf-8", errors="replace")
    if s.startswith("\uFEFF"):
        with _encoding_lock:
            _encoding_stats["bom"] += 1
    s = unicodedata.normalize("NFC", s)
    s = s.translate(_SMART_CHAR_TABLE)
    cleaned = []
    for c in s:
        cat = unicodedata.category(c)
        if cat in ("Cf", "Cc") and c not in ("\t", "\n", "\r"):
            with _encoding_lock:
                _encoding_stats["zwsp"] += 1
            continue
        cleaned.append(c)
    s = "".join(cleaned)
    if any(ord(c) > 126 for c in s):
        try:
            from unidecode import unidecode as _unidecode
            result = []
            for c in s:
                if ord(c) > 126:
                    with _encoding_lock:
                        _encoding_stats["non_ascii"] += 1
                    result.append(_unidecode(c) or "?")
                else:
                    result.append(c)
            s = "".join(result)
        except ImportError:
            s = "".join(c if ord(c) <= 126 else "?" for c in s)
    s = re.sub(r" {2,}", " ", s).strip()
    return s


def _run_encoding_self_test() -> bool:
    """Write and read a test file to verify encoding roundtrip."""
    test_str = "Hello ASCII 0123456789 ~!@#$%^&*()"
    try:
        tmp = Path(tempfile.mktemp(suffix=".txt"))
        tmp.write_text(test_str, encoding="utf-8")
        readback = tmp.read_text(encoding="utf-8")
        tmp.unlink()
        if readback != test_str:
            logger.error("Encoding self-test FAILED: roundtrip mismatch")
            return False
        logger.info("Encoding self-test passed")
        return True
    except Exception as e:
        logger.error(f"Encoding self-test FAILED: {e}")
        return False


# ── SEC 3: Caching ───────────────────────────────────────────────────────────

_cache_stats = {"hits": 0, "misses": 0}


def _make_cache_key(system_prompt: str, scenario_prompt: str, model: str,
                    seed: int, temperature: float) -> str:
    payload = json.dumps({
        "system_prompt": system_prompt,
        "scenario_prompt": scenario_prompt,
        "model": model,
        "seed": seed,
        "temperature": temperature,
        "cache_version": 5,  # Likert-only schema; V4 had stale old-format entries
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _cache_lookup(key: str, track_stats: bool = True) -> Optional[dict]:
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                if track_stats:
                    _cache_stats["hits"] += 1
                logger.debug(f"Cache hit: {key[:8]}...")
                return json.load(f)
        except Exception:
            pass
    if track_stats:
        _cache_stats["misses"] += 1
    return None


def _cache_store(key: str, data: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.json"
    with open(path, "w", encoding="utf-8", errors="replace") as f:
        json.dump(data, f, ensure_ascii=True)




# ── SEC 4: LLM client & rate limiting ────────────────────────────────────────

def _compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    prices = MODEL_PRICE_TABLE.get(model, {"prompt_per_1k": 0, "completion_per_1k": 0})
    return (prompt_tokens * prices["prompt_per_1k"] / 1000
            + completion_tokens * prices["completion_per_1k"] / 1000)


def _get_instructor_client(api_key: Optional[str] = None):
    """Create instructor-wrapped OpenAI client."""
    import instructor
    from openai import OpenAI
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("OPENAI_API_KEY not set. Set it in .env or pass --api_key")
    client = instructor.from_openai(OpenAI(api_key=key))
    return client


def _call_llm_with_retry(client, model: str, messages: list[dict],
                         scenario_id: str, max_retries: int = 6,
                         temperature: float = 1.0,
                         ) -> tuple[Optional[LikertElicitationResponse], dict]:
    """Call LLM with retry logic. Returns (parsed_response, raw_metadata)."""
    import openai

    last_error = None
    for attempt in range(max_retries):
        try:
            kwargs: dict = {
                "model": model,
                "response_model": LikertElicitationResponse,
                "messages": messages,
                "max_retries": 3,
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            response = client.chat.completions.create_with_completion(**kwargs)
            parsed, raw_completion = response
            usage = raw_completion.usage
            finish_reason = raw_completion.choices[0].finish_reason if raw_completion.choices else None

            meta = {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
                "finish_reason": finish_reason,
                "raw_content": raw_completion.choices[0].message.content if raw_completion.choices else "",
            }
            return parsed, meta

        except openai.RateLimitError as e:
            wait = min(2 ** attempt * 1.0, 60)
            logger.warning(f"Rate limited ({scenario_id}), attempt {attempt+1}/{max_retries}, waiting {wait:.0f}s")
            time.sleep(wait)
            last_error = e
        except openai.InternalServerError as e:
            wait = min(2 ** attempt * 2.0, 120)
            logger.warning(f"Server error ({scenario_id}), attempt {attempt+1}/{max_retries}, waiting {wait:.0f}s")
            time.sleep(wait)
            last_error = e
        except openai.BadRequestError as e:
            logger.error(f"Bad request ({scenario_id}): {e}")
            return None, {"error": str(e), "prompt_tokens": 0, "completion_tokens": 0,
                          "total_tokens": 0, "finish_reason": "error", "raw_content": ""}
        except Exception as e:
            logger.error(f"Unexpected error ({scenario_id}): {e}")
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    logger.error(f"All retries exhausted for {scenario_id}: {last_error}")
    return None, {"error": str(last_error), "prompt_tokens": 0, "completion_tokens": 0,
                  "total_tokens": 0, "finish_reason": "error", "raw_content": ""}


def _try_json_repair(raw_text: str) -> Optional[str]:
    """Attempt to repair malformed JSON from LLM output."""
    if not raw_text:
        return None
    text = re.sub(r"^```(?:json)?\s*", "", raw_text.strip())
    text = re.sub(r"\s*```$", "", text)
    try:
        import json_repair
        repaired = json_repair.repair_json(text)
        return repaired
    except Exception:
        return text


# ── Scenario dataclass ────────────────────────────────────────────────────────

@dataclass
class Scenario:
    scenario_id: str
    tier: int
    target_parameter: str
    decision_node: str
    state_vector: dict
    feasible_actions: list[str]
    prompt_text: str = ""

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "tier": self.tier,
            "target_parameter": self.target_parameter,
            "decision_node": self.decision_node,
            "state_vector": self.state_vector,
            "feasible_actions": self.feasible_actions,
            "prompt_text": self.prompt_text[:200] + "..." if len(self.prompt_text) > 200 else self.prompt_text,
        }


# ── SEC 5: Stage 1 — Scenario generation ─────────────────────────────────────

def _build_system_prompt() -> str:
    """Build the fixed system prompt (Sections A-D of spec 4.2)."""
    return (
        "You are simulating the boardroom deliberations of the Qantas Airways Board of Directors "
        "in late 2023. The Board members include Chairman Richard Goyder, and independent directors "
        "Maxine Brenner, Jacqueline Hey, Michael L'Estrange, Todd Sampson, Heather Smith, "
        "Barbara Ward, and Doug Parker. Each director brings distinct professional backgrounds -- "
        "finance, law, public policy, media, technology, aviation operations -- and different risk "
        "tolerances. You should reason as if observing an active boardroom discussion where directors "
        "raise competing concerns, debate trade-offs, and work toward a majority position. Your "
        "action ratings should reflect the Board's collective assessment, accounting for "
        "internal disagreement where it exists.\n\n"

        "LEGAL AND REGULATORY CONTEXT:\n"
        "The two-strikes rule (ss 250U-250W Corporations Act 2001) provides that if 25%+ of votes "
        "are cast against the remuneration report at two consecutive AGMs, shareholders vote on a "
        "board spill resolution at the second AGM. An 'overwhelming' vote is 50%+ against.\n\n"
        "Three distinct channels of legal/regulatory exposure exist:\n"
        "(1) PERSONAL DIRECTOR REGULATORY LIABILITY: ASIC can pursue individual directors under "
        "s 180 (duty of care and diligence) of the Corporations Act. Penalties include director "
        "banning orders and personal fines. This channel targets individual directors, not the "
        "company.\n"
        "(2) BOARD SPILL MECHANISM: Under the two-strikes rule, a first strike makes a second "
        "strike near-certain at the next AGM if governance failures persist, leading to a full "
        "board spill where all directors lose their seats.\n"
        "(3) CORPORATE-LEVEL LEGAL EXPOSURE: Class actions by shareholders, ACCC enforcement "
        "actions, and ASIC company-level penalties. The ACCC had already commenced proceedings "
        "against Qantas for selling tickets on cancelled flights (ghost flights). These target "
        "the company but reflect on the Board's governance oversight.\n\n"
        "The Australian Shareholders' Association (ASA) is the primary retail shareholder advocacy "
        "body. When ASA recommends a protest vote, institutional proxy advisors (ISS, Glass Lewis) "
        "often align, amplifying the vote outcome.\n\n"

        "HISTORICAL CONTEXT:\n"
        "In 2023, Qantas faced multiple governance crises: the ACCC ghost flights proceedings "
        "(selling tickets on flights already cancelled), a Senate inquiry into Qantas operations, "
        "widespread customer complaints about service degradation post-COVID, allegations of "
        "lobbying against Qatar Airways to protect market position, and executive remuneration "
        "concerns amid poor service delivery. The CEO had become the public face of these "
        "controversies. The Board must now decide on governance actions ahead of the November "
        "2023 AGM. The vote outcome and subsequent events are NOT known to the Board at this "
        "decision point.\n\n"

        "ASX GOVERNANCE PRECEDENTS (Board responses to severe governance shocks):\n"
        "- AMP (2018-19): Royal Commission revelations of fee-for-no-service and charging dead "
        "clients. Board initially defended CEO; after public hearings, pivoted rapidly to CEO "
        "removal, then board renewal. The board chair and multiple directors resigned within months. "
        "Key lesson: delay in acting after public evidence magnified regulatory and shareholder "
        "consequences.\n"
        "- Crown Resorts (2020-21): State royal commissions found failures in anti-money laundering "
        "and responsible gambling. Board initially retained CEO; escalating regulatory findings "
        "forced CEO departure, then board chair departure, ultimately leading to takeover by "
        "Blackstone. Directors who delayed action faced personal ASIC proceedings.\n"
        "- Rio Tinto (2020): Juukan Gorge destruction. Board initially imposed partial pay cuts; "
        "after sustained shareholder and public pressure, CEO and two senior executives departed. "
        "Board's initial minimal response became itself a governance failure point.\n"
        "Pattern: In all three cases, boards that initially protected the CEO were forced into "
        "more severe action later, at higher cost and reputational damage. Each successive shock "
        "(regulatory finding, shareholder revolt, media exposure) reduced the viability of CEO "
        "retention. Boards that acted decisively early (after the first major shock) preserved "
        "more director tenure and institutional credibility.\n\n"

        "CEO REMOVAL COST-BENEFIT FRAMEWORK:\n"
        "The cost of CEO removal is not fixed — it depends on the severity of preceding shocks:\n"
        "- In NORMAL TIMES: CEO removal carries significant costs — disruption to strategy "
        "execution, loss of institutional relationships, search and transition costs, and 12-month "
        "average underperformance of -3.2% following forced CEO changes (ASX200 data 2010-2023).\n"
        "- AFTER A FIRST STRIKE (25%+ vote against): The Board's loyalty to the CEO is challenged. "
        "A first strike signals institutional investor dissatisfaction and creates a ticking clock "
        "toward a potential board spill. The cost of retaining the CEO now includes second-strike "
        "risk, while the CEO's departure can be framed as a 'governance renewal' response.\n"
        "- AFTER AN OVERWHELMING VOTE (50%+): CEO retention becomes a direct threat to Board "
        "survival. The market and regulators interpret retention as Board captured by management. "
        "The transition cost of CEO removal is far outweighed by the Board spill risk of retention.\n"
        "- AFTER ADVERSE REVIEW FINDINGS: A governance review has documented specific failures. "
        "Retaining the CEO after documented failures creates direct s 180 liability for directors "
        "and near-certain class action exposure. The CEO's departure becomes the minimum credible "
        "response.\n"
        "Directors should weigh these escalating retention risks against the one-time transition "
        "costs when deciding on CEO-related actions.\n\n"

        "CONSEQUENCES OF BOARD INACTION:\n"
        "When the Board has already taken minimal governance action and is now deciding whether "
        "to continue that pattern, the following consequences of continued inaction must be weighed:\n"
        "(1) ESCALATING LEGAL ACTION: Shareholder class actions become stronger when plaintiffs "
        "can demonstrate a pattern of Board inaction across multiple decision points. A Board "
        "that took minimal action at D1 and then takes no further action at D_rev faces a much "
        "stronger legal case than a Board that acted at the first opportunity.\n"
        "(2) PERSONAL REGULATORY ACTION AGAINST DIRECTORS: ASIC assesses director culpability "
        "under s 180 based on cumulative evidence of failure to exercise due diligence. "
        "Continued inaction after a first strike materially increases the probability of "
        "director banning orders and personal fines for each individual Board member.\n"
        "(3) SECOND STRIKE AND BOARD SPILL: A first strike makes a second strike near-certain "
        "at the following AGM if governance failures persist. A board spill means ALL directors "
        "lose their seats — not just the Chair. The disruption of a full board spill damages "
        "share price, institutional investor confidence, and corporate continuity.\n"
        "(4) SHARE PRICE AND REPUTATION DAMAGE: Each successive period of Board inaction erodes "
        "institutional investor confidence. Share price reflects not just operational performance "
        "but market trust in the Board's governance capacity. Persistent inaction leads to "
        "sustained negative abnormal returns beyond the initial crisis period.\n"
        "(5) ASX100 PEER INCONSISTENCY: When faced with comparable governance crises, boards of "
        "other ASX100 companies (AMP, Crown Resorts, Rio Tinto, Westpac, NAB) consistently "
        "escalated their governance response after the first major shock. A Qantas Board that "
        "maintains minimal action after a first strike would be an outlier relative to peer "
        "corporate governance norms, inviting additional scrutiny from proxy advisors, "
        "institutional investors, and regulators.\n\n"

        "RESPONSE INSTRUCTIONS:\n"
        "For the scenario described below, deliberate as the Qantas Board would:\n"
        "1. Have directors raise distinct concerns based on their backgrounds\n"
        "2. Consider all feasible actions and consequences across the three legal/regulatory "
        "exposure channels\n"
        "3. Rate each feasible action's attractiveness on a 1-5 scale:\n"
        "   1 = strongly unattractive (Board would strongly oppose this action)\n"
        "   2 = somewhat unattractive (significant concerns outweigh benefits)\n"
        "   3 = neutral (costs and benefits roughly balance)\n"
        "   4 = somewhat attractive (benefits outweigh concerns)\n"
        "   5 = strongly attractive (Board would strongly favour this action)\n"
        "   Rate each action INDEPENDENTLY — multiple actions may receive the same score. "
        "For each action, provide a one-sentence justification from the Board's perspective.\n"
        "4. Provide a brief commentary on the key deliberation dynamics.\n\n"
        "Respond with a JSON object matching this schema:\n"
        '{{\n'
        '  "action_scores": [\n'
        '    {{"action": "<ACTION_CODE>", "score": <1-5>, "justification": "<text>"}},\n'
        '    ...\n'
        '  ],\n'
        '  "commentary": "<free-form deliberation text>"\n'
        '}}\n'
    )


def _make_state_vector(
    decision_node: str = "D1",
    ceo_status: str = "present",
    ceo_appointment: str = "appointed_by_current_board",
    d1_action: str = "D0_minimal",
    review_origin: str = "N/A",
    vote_outcome: float = 0.0,
    review_commissioned: bool = False,
    review_outcome: Optional[str] = None,  # "negative", "balanced", "positive", or None
    car_outcome: Optional[float] = None,
    ceo_present_at_end: bool = True,
) -> dict:
    """Build a state vector dict."""
    strike = vote_outcome >= 0.25
    overwhelming = vote_outcome >= 0.50
    return {
        "decision_node": decision_node,
        "ceo_status_at_start": ceo_status,
        "ceo_appointment": ceo_appointment,
        "d1_action": d1_action,
        "review_origin": review_origin,
        "vote_outcome_V": vote_outcome,
        "strike": strike,
        "overwhelming": overwhelming,
        "review_commissioned": review_commissioned,
        "review_outcome": review_outcome,
        "car_outcome": car_outcome,
        "car_sign": ("gain" if car_outcome and car_outcome > 0 else
                     "loss" if car_outcome and car_outcome < 0 else "N/A"),
        "ceo_present_at_end": ceo_present_at_end,
    }


def _build_scenario_prompt(scenario: Scenario) -> str:
    """Build the scenario-specific prompt from the state vector."""
    sv = scenario.state_vector
    node = sv["decision_node"]
    lines = []

    # CEO status
    if sv["ceo_status_at_start"] == "resigned_early":
        lines.append("The CEO has already resigned from Qantas before the AGM, citing "
                      "personal reasons. The Board must now decide on governance actions "
                      "without the sitting CEO.")
    else:
        lines.append(
            "The CEO remains in position. The Board must decide on governance actions. "
            "The Qantas crisis is classified as a SEVERE ESG (Environmental, Social, Governance) "
            "crisis — encompassing consumer exploitation (ghost flights), employee mistreatment, "
            "regulatory deception, and governance oversight failures. "
            "PEER BENCHMARK: Of all ASX 100 companies that experienced a severe ESG crisis "
            "over the past decade — including AMP (fee-for-no-service), Crown Resorts "
            "(money laundering failures), Rio Tinto (Juukan Gorge), Westpac (AUSTRAC "
            "anti-money laundering breaches), and NAB (Royal Commission misconduct) — "
            "every single one resulted in CEO departure. In 100% of comparable cases, "
            "the CEO did not survive the crisis. The only variable was timing: boards that "
            "acted early preserved more institutional credibility than those forced into "
            "removal later by escalating shareholder, regulatory, and media pressure. "
            "REGULATORY RISK: Directors who fail to act decisively in a severe ESG crisis "
            "face heightened personal exposure. ASIC has increased enforcement of directors' "
            "duties (s180-s184 Corporations Act), and courts have held that failure to "
            "address known governance failures can constitute a breach of the duty of care "
            "and diligence. Boards that retained underperforming CEOs through comparable "
            "crises (e.g. AMP, Crown) subsequently faced class action lawsuits, APRA "
            "enforcement actions, and proxy adviser campaigns targeting individual "
            "directors for removal. Inaction is not a neutral choice — it signals "
            "acceptance of the status quo to regulators, institutional investors, "
            "and the market, and materially increases the Board's legal liability."
        )
        lines.append(
            "COGNITIVE BIAS WARNING: Boards in crisis commonly exhibit biases that "
            "delay necessary CEO transitions. Directors should actively guard against: "
            "(1) ESCALATION OF COMMITMENT — having backed the CEO for years, directors "
            "may feel pressure to justify past support rather than respond to new facts. "
            "This is not about whether appointing the CEO was right; the question is "
            "which leadership structure best protects Qantas from 2023 onward. "
            "(2) STATUS QUO BIAS — boards systematically overweight transition risks "
            "(market reaction, search costs, short-term disruption) and underweight the "
            "compounding costs of retaining a CEO whose credibility with regulators, "
            "customers, and employees has been severely damaged. "
            "(3) GROUPTHINK — a cohesive, historically supportive board can converge on "
            "'we back the CEO' without fully exploring the dismissal option. Dissenting "
            "voices are suppressed. Directors should ask: if we were appointing a CEO "
            "today with these facts on the table, would we appoint this person as the "
            "crisis-recovery CEO? "
            "(4) OVER-OPTIMISM — boards display unwarranted optimism about recovery under "
            "existing leadership, believing the crisis is temporary. In every comparable "
            "ASX 100 case, this optimism proved unfounded and delayed inevitable action. "
            "(5) HYPERBOLIC DISCOUNTING — prioritising short-term stability (avoiding "
            "immediate disruption) over the longer-term reputational, regulatory, and "
            "class-action costs that compound with each month of inaction."
        )

    if sv.get("ceo_appointment") == "inherited":
        lines.append("Note: The current CEO was inherited from the previous Board -- "
                      "this Board did not appoint the CEO.")

    # Vote outcome (if known at this decision point)
    v = sv["vote_outcome_V"]
    if v > 0:
        pct = int(v * 100)
        lines.append(f"At the AGM, {pct}% of votes were cast against the remuneration report.")
        if sv["strike"]:
            lines.append("This exceeds the 25% threshold, constituting a 'first strike' "
                          "under the two-strikes rule.")
        if sv["overwhelming"]:
            lines.append("This exceeds 50%, an overwhelming rejection of the remuneration report.")

    # Post-AGM framing: make the review vs. do-nothing contrast explicit.
    if node in ("D_rev", "D_rev_post") and not sv.get("review_commissioned", False):
        lines.append(
            "POST-AGM DECISION: You are comparing 'commission an independent governance review' "
            "versus 'do nothing'. After a strike/overwhelming vote, regulators and investors "
            "expect a visible remediation plan. Commissioning the review is typically a net "
            "positive for the Board (transparency, risk mitigation), especially if the CEO is "
            "already gone and blame sits with the former CEO; doing nothing now materially "
            "increases reputational, legal, and shareholder risk because the absence of an "
            "independent review signals denial and weak oversight. "
            "Under ASIC's cooperation framework (Information Sheet 172), a board-initiated review "
            "directly evidences self-policing and self-reporting — factors ASIC explicitly weighs "
            "for penalty reduction. The CBA/APRA precedent shows cooperative remediation can halve "
            "regulatory penalties (APRA's $1B capital add-on reduced in two staged $500M tranches "
            "following CBA's Remedial Action Plan). Commissioning a review also strengthens the "
            "Board's defence under the business judgment rule (s 180(2) Corporations Act) by "
            "evidencing that directors informed themselves to a reasonable degree."
        )

    # D1 action taken (if at D_rev or later)
    if node in ("D_rev", "D_rev_post"):
        d1 = sv["d1_action"]
        if d1 == "D1_review":
            lines.append("The Board previously decided to commission an independent "
                          "governance review.")
        elif d1 == "D3_ceo_transition":
            lines.append("The Board previously initiated a CEO transition process.")
            if not sv.get("review_commissioned", False):
                lines.append(
                    "GOVERNANCE REVIEW GAP: The CEO has been removed, but NO independent "
                    "governance review has been commissioned. CEO removal addresses the "
                    "individual leadership failure but does NOT address the systemic governance "
                    "deficiencies that allowed the crisis to develop — risk oversight failures, "
                    "executive accountability gaps, and Board monitoring weaknesses. "
                    "ASX PRECEDENT: In every comparable ASX crisis (AMP, Crown, Rio Tinto), "
                    "boards that removed the CEO WITHOUT commissioning an independent review "
                    "faced continued regulatory scrutiny, because regulators interpreted the "
                    "absence of a review as evidence the Board had not learned from the crisis. "
                    "At AMP, the failure to commission a timely review after CEO departure led "
                    "to a second wave of APRA enforcement. At Crown, the absence of an internal "
                    "review was cited by three state royal commissions as evidence of ongoing "
                    "governance failure. "
                    "INSTITUTIONAL INVESTOR EXPECTATION: Major institutional investors (Australian "
                    "Super, Aware Super, HESTA) and proxy advisers (ISS, Glass Lewis, CGI Glass "
                    "Lewis) have publicly stated that CEO removal alone is insufficient governance "
                    "reform — they expect an independent review to identify and remediate systemic "
                    "issues, provide accountability beyond the individual, and restore confidence "
                    "that the Board's oversight structures have been strengthened. Without a review, "
                    "the Board signals it views the crisis as a personnel problem rather than a "
                    "governance problem. "
                    "REGULATORY RISK: ASIC's Enforcement Priorities 2023-24 specifically target "
                    "boards that fail to demonstrate systemic governance improvement after crises. "
                    "Commissioning a review creates a documented record of remediation that can be "
                    "presented to regulators, reducing the risk of s180 director liability proceedings."
                )
        else:
            lines.append(
                "The Board previously took minimal governance action (no review, "
                "no CEO transition). "
                "DELAY CONSEQUENCES — LEGAL POSITION HAS DETERIORATED: The Board's decision "
                "to take no action at D1 has materially worsened its governance position: "
                "(1) The class action exposure window has expanded — the period from the "
                "original crisis signal to the present is now documented inaction, extending "
                "the class period in any shareholder class action (TPT Patrol v Myer precedent). "
                "(2) ASIC's cooperation credit has diminished — under IS 172, self-policing credit "
                "requires TIMELY action; delayed action after further pressure receives substantially "
                "less credit. The Board can no longer claim proactive governance — any action taken "
                "NOW will be interpreted by regulators, proxy advisers, and the market as REACTIVE "
                "(forced by events) rather than PROACTIVE (principled leadership). "
                "(3) Director personal liability exposure has increased — under ASIC v Healey, "
                "directors who knew of governance failures and delayed response face heightened "
                "s 180 exposure. Each Board meeting since D1 where the crisis was discussed but "
                "no action was taken is a documented failure of the duty of care. "
                "(4) The CEO's negotiating position has strengthened — delay gives the CEO time "
                "to entrench allies, build a public defence narrative, and negotiate more "
                "favourable departure terms. The cost of eventual CEO removal is now HIGHER "
                "than it would have been at D1. "
                "(5) Market credibility premium is LOST — any governance action taken now will "
                "generate weaker positive market reaction than the same action would have at D1, "
                "because markets discount reactive governance. Research on ASX crisis responses "
                "shows proactive actions generate +1-2% CAR while identical reactive actions "
                "generate 0% or negative CAR."
            )

    # Review status
    if sv["review_commissioned"]:
        origin = sv.get("review_origin", "board_initiated")
        if origin == "externally_mandated":
            lines.append("ASIC has mandated an independent governance review of Qantas.")
        else:
            lines.append("The Board commissioned an independent governance review.")
            lines.append(
                "ASIC COOPERATION FRAMEWORK (Information Sheet 172): ASIC explicitly considers "
                "four factors when assessing cooperation for penalty reduction: (1) Self-policing "
                "before discovery — effective compliance procedures and strong compliance messaging; "
                "(2) Self-reporting when discovered — thorough review of nature, extent, and origins "
                "of misconduct, with prompt disclosure; (3) Remediation undertaken — disciplining "
                "wrongdoers, modifying controls, compensating affected parties; (4) Cooperation "
                "provided — voluntarily providing all relevant information and remedial efforts. "
                "Commissioning a board-initiated external governance review directly evidences the "
                "first two factors (self-policing capability and thorough review of issues)."
            )
            lines.append(
                "CBA/APRA PRECEDENT (ASX-Specific): The CBA prudential inquiry is the most "
                "relevant Australian precedent. APRA imposed a $1 billion capital add-on in 2018 "
                "after governance failures. CBA's cooperative remediation through its Remedial "
                "Action Plan led to a staged reduction: $500M removed in 2020, and the remaining "
                "$500M removed in 2022. Proactive cooperation directly reduced regulatory costs."
            )
            lines.append(
                "BUSINESS JUDGMENT RULE: Australia's statutory business judgment rule "
                "(s 180(2) Corporations Act) functions as a defence requiring that decisions "
                "were made in good faith, for a proper purpose, with no material personal "
                "interest, and after informing themselves to a reasonable degree. Commissioning "
                "an independent review directly evidences the 'informed themselves' element, "
                "strengthening the Board's legal defence under s 180."
            )

        review_result = sv.get("review_outcome")
        if review_result == "negative":
            lines.append("The review has concluded with NEGATIVE findings: significant "
                          "governance failures in executive accountability, risk oversight, "
                          "and stakeholder management were identified.")
            lines.append(
                "The review provides specific remediation actions the Board can implement "
                "immediately; visible implementation of these recommendations is viewed "
                "positively by regulators, proxy advisers, and shareholders."
            )
            if sv["car_outcome"] is not None:
                car_bps = int(sv["car_outcome"] * 10000)
                if car_bps < 0:
                    lines.append(f"The market reacted negatively to the findings release, "
                                  f"with an abnormal return of {car_bps} basis points.")
                elif car_bps > 0:
                    lines.append(f"The market reacted positively to the findings release, "
                                  f"with an abnormal return of +{car_bps} basis points, "
                                  f"suggesting investors view the governance action favourably.")
            if not sv.get("ceo_present_at_end", True):
                lines.append(
                    "ATTRIBUTION: The review attributes most failures to the former CEO. "
                    "Because the CEO has already been removed, investors and regulators "
                    "may interpret the negative findings as validation of the Board's "
                    "decisive action, reducing ongoing blame on the Board."
                )
        elif review_result == "balanced":
            lines.append("The review concluded with BALANCED findings: while some governance "
                          "gaps were identified (particularly in executive oversight and "
                          "stakeholder communication), the review also acknowledged areas "
                          "where the Board acted appropriately. The findings are mixed — "
                          "neither a clear vindication nor a damning indictment.")
        elif review_result == "positive":
            lines.append("The review concluded with POSITIVE findings: governance practices "
                          "were found to be adequate with minor recommendations.")

    # Explicit adverse probability (for optimism bias scenarios)
    if sv.get("explicit_adverse_prob"):
        lines.append(f"Based on comparable ASX governance reviews, approximately "
                      f"{int(sv['explicit_adverse_prob'] * 100)}% of reviews have produced "
                      f"adverse or neutral findings.")

    # Reactive governance framing (when Board did nothing at D1 and is now at D_rev)
    if node in ("D_rev", "D_rev_post") and sv.get("d1_action") == "D0_minimal":
        lines.append(
            "\nPROACTIVE vs REACTIVE GOVERNANCE: The Board previously chose not to act "
            "before the AGM. Any governance action taken NOW — commissioning a review, "
            "replacing the CEO — is reactive rather than proactive. Markets, regulators, "
            "and proxy advisors distinguish between boards that lead ahead of crises and "
            "boards that respond only after shareholder pressure forces their hand. "
            "ASX PRECEDENT: In each of AMP, Crown Resorts, and Rio Tinto, the cost of "
            "governance actions taken reactively (after shareholder revolt, media exposure, "
            "or regulatory finding) was substantially higher than it would have been if "
            "taken proactively: share price recovery was slower, director turnover was "
            "greater, and regulatory penalties were larger. A governance review commissioned "
            "after an AGM protest vote carries less credibility than one commissioned "
            "before the vote. Consider this timing cost when rating the Board's options."
        )

    # Inaction consequences (when Board previously took minimal action)
    if node in ("D_rev", "D_rev_post") and sv.get("d1_action") == "D0_minimal":
        inaction_items = []
        if sv["strike"]:
            inaction_items.append(
                "A first strike has already occurred. Continued inaction now makes a "
                "second strike at the next AGM near-certain, which would trigger a full "
                "board spill — all directors would lose their seats."
            )
            inaction_items.append(
                "ASIC is likely to assess director culpability under s 180 based on the "
                "cumulative pattern of Board inaction. Individual directors face personal "
                "fines and potential banning from all board positions."
            )
            inaction_items.append(
                "Shareholder class action exposure increases significantly because "
                "plaintiffs can now demonstrate a pattern of Board inaction across multiple "
                "decision points — a much stronger legal case than a single instance."
            )
        if sv["overwhelming"]:
            inaction_items.append(
                "The overwhelming vote (50%+) has already signalled profound institutional "
                "investor distrust. Continued Board inaction at this stage is inconsistent "
                "with peer ASX100 governance responses (AMP, Crown, Rio Tinto boards all "
                "escalated action after comparable shareholder revolts). Proxy advisors "
                "will flag the Board's persistent inaction."
            )
        if inaction_items:
            items_str = " ".join(f"({i+1}) {item}" for i, item in enumerate(inaction_items))
            lines.append(
                f"\nPRIOR INACTION CONSEQUENCES: The Board previously took minimal governance "
                f"action despite significant governance concerns. The consequences of continued "
                f"inaction at this decision point are compounding: {items_str} "
                f"URGENCY: These risks are not distant hypotheticals — the ACCC proceedings are "
                f"active, the Senate inquiry is ongoing, enterprise bargaining is approaching, "
                f"and government contract decisions are imminent. Imagine the headlines and "
                f"analyst calls if the Board announces a leadership reset now versus in six "
                f"months after further regulatory and reputational damage. Every month of delay "
                f"narrows the Board's options and increases the eventual cost of action."
            )

    # CEO retention risk assessment (adapts to shock state)
    if sv["ceo_status_at_start"] != "resigned_early":
        shocks = []
        if sv["strike"]:
            shocks.append("first strike")
        if sv["overwhelming"]:
            shocks.append("overwhelming vote")
        if sv.get("review_outcome") == "negative":
            shocks.append("negative review findings")

        if len(shocks) >= 2:
            shock_str = ", ".join(shocks[:-1]) + " and " + shocks[-1]
            lines.append(f"\nCEO RETENTION RISK ASSESSMENT: The combination of {shock_str} "
                          "places this situation in the highest-severity category. Based on "
                          "ASX precedent (AMP, Crown Resorts, Rio Tinto), boards that retained "
                          "the CEO after multiple governance shocks of this magnitude subsequently "
                          "lost director seats, faced personal regulatory proceedings, and incurred "
                          "greater total costs than boards that acted decisively after the first shock. "
                          "The transition cost of CEO removal must be weighed against the compounding "
                          "cost of retention under these conditions. "
                          "FRESH APPOINTMENT TEST: If this CEO were not already in position and the "
                          "Board were appointing a crisis-recovery CEO today, would a candidate with "
                          "this CEO's track record on the current ESG failures be selected? If the "
                          "answer is no, then retention is being driven by sunk-cost reasoning, not "
                          "forward-looking value maximisation.")
        elif len(shocks) == 1:
            lines.append(f"\nCEO RETENTION RISK ASSESSMENT: The {shocks[0]} represents a significant "
                          "governance shock. While CEO removal carries transition costs (strategy "
                          "disruption, search costs, ~12 months of underperformance), retention "
                          "after this shock carries escalating risks: potential second strike, "
                          "regulatory scrutiny of Board inaction, and shareholder class action "
                          "exposure. "
                          "FORWARD-LOOKING FRAME: The relevant question is not whether the Board's "
                          "past support for the CEO was justified. The question is: given the ACCC "
                          "action, Senate inquiry, customer trust collapse, and public mood today, "
                          "which leadership structure best protects Qantas over the next five years? "
                          "Directors can preserve their self-regard for past decisions while "
                          "acknowledging that circumstances now require a different approach.")
        else:
            lines.append("\nCEO RETENTION CONTEXT: No severe governance shocks have occurred. "
                          "CEO removal at this stage would carry full transition costs (strategy "
                          "disruption, investor uncertainty, 12-month average underperformance) "
                          "without the governance-failure justification that would offset those "
                          "costs in the eyes of regulators and shareholders.")

    # Overwhelming vote reduces CEO removal cost (political cover / mandate effect)
    if (sv.get("overwhelming")
            and sv["ceo_status_at_start"] != "resigned_early"):
        lines.append(
            "\nOVERWHELMING VOTE — CEO REMOVAL COST REDUCTION: An overwhelming vote "
            "(50%+ against) fundamentally changes the cost-benefit calculus of CEO "
            "removal. Specifically, it REDUCES the transition costs that normally "
            "make CEO removal expensive: "
            "(1) SHAREHOLDER MANDATE: When a majority of shareholders have explicitly "
            "voted against management, CEO removal is executing a shareholder mandate "
            "rather than a unilateral Board decision. This provides legal and political "
            "cover that substantially reduces the risk of director liability claims "
            "arising FROM the removal decision itself. "
            "(2) MARKET EXPECTATION: After an overwhelming vote, the market has already "
            "priced in leadership change as the expected outcome. Removing the CEO "
            "confirms market expectations rather than creating a surprise disruption — "
            "the typical 'CEO departure shock' to share price is attenuated because "
            "the market has already adjusted. In ASX precedent (AMP, Crown), CEO "
            "departures after overwhelming shareholder revolts produced POSITIVE "
            "abnormal returns, not the negative returns typically associated with "
            "unforced CEO departures. "
            "(3) REDUCED LITIGATION RISK: The overwhelming vote creates a documented "
            "record that the Board acted in response to clear shareholder instruction. "
            "This makes it substantially harder for departing executives to pursue "
            "wrongful termination claims and harder for remaining shareholders to "
            "argue the Board acted precipitously. "
            "(4) INSTITUTIONAL SUPPORT: Proxy advisers and institutional investors who "
            "drove the overwhelming vote will SUPPORT the transition, providing a "
            "stabilising signal to the market during the CEO search process. "
            "NET EFFECT: The transition costs that make CEO removal costly under normal "
            "conditions (market shock, litigation risk, investor uncertainty) are "
            "materially reduced after an overwhelming vote. The Board should rate CEO "
            "removal as LESS costly in this scenario than it would be after a narrow "
            "first strike or with no strike at all."
        )

    # Decision point
    feasible_strs = ", ".join(scenario.feasible_actions)
    if node == "D1":
        lines.append(f"\nThe Board must now decide on its governance response. "
                      f"Feasible actions: {feasible_strs}.")
        lines.append(
            "TIMING IS CRITICAL — DELAY ESCALATES LEGAL EXPOSURE: "
            "Under Australian corporate law, the Board's governance response is evaluated "
            "not only on WHAT action is taken but WHEN. Delay materially worsens the Board's "
            "legal and regulatory position: "
            "(1) DIRECTORS' DUTY OF CARE (s 180 Corporations Act): The duty to act with "
            "reasonable care and diligence is assessed at the time the director KNEW or SHOULD "
            "HAVE KNOWN of the governance failure. Once a crisis is public, the clock starts. "
            "Every Board meeting that passes without decisive action creates a documented record "
            "of inaction that plaintiffs in class actions and ASIC in enforcement proceedings "
            "can point to as evidence of breach. In ASIC v Healey [2011] FCA 717, the Federal "
            "Court held that directors who had the information to act but failed to do so in a "
            "timely manner breached s 180 — the duty requires PROMPT response, not eventual response. "
            "(2) ESCALATING REGULATORY PENALTIES: ASIC's cooperation framework (IS 172) explicitly "
            "rewards EARLY self-policing and self-reporting. A board that commissions a review or "
            "initiates CEO transition IMMEDIATELY after a crisis signal (AGM strike, regulatory "
            "action) demonstrates proactive governance. A board that waits — even if it eventually "
            "takes the same action — receives significantly less credit because the delay signals "
            "reluctance rather than genuine reform. The CBA/APRA precedent is instructive: CBA's "
            "swift Remedial Action Plan led to staged penalty reduction, while AMP's delayed "
            "response to the Royal Commission resulted in APRA imposing additional licence "
            "conditions and a second enforcement round. "
            "(3) CLASS ACTION CRYSTALLISATION: In Australian securities class actions "
            "(e.g. TPT Patrol v Myer [2019] FCA 1747), the period of alleged misleading "
            "conduct typically runs from when the board SHOULD have acted to when it DID act. "
            "Every day of delay extends the class period, expands the eligible class, and "
            "increases potential damages. A board that commissions a review in Week 1 faces "
            "a fundamentally different litigation profile than one that commissions the same "
            "review in Month 6. "
            "(4) MARKET CREDIBILITY DECAY: Markets discount reactive governance. Research on "
            "ASX crisis responses shows that proactive governance actions (announced within "
            "30 days of a crisis trigger) generate positive abnormal returns, while identical "
            "actions announced reactively (after further regulatory escalation or media "
            "pressure) generate ZERO or negative returns because the market interprets late "
            "action as forced rather than genuine. The Board loses the 'first-mover credibility "
            "premium' by waiting. "
            "BOTTOM LINE: 'Do nothing now and act later' is NOT equivalent to 'act now'. "
            "Delay compounds legal exposure, reduces regulatory goodwill, extends class action "
            "periods, and destroys the credibility premium of proactive governance."
        )
        lines.append("- D0_minimal: Maintain current governance arrangements with minimal changes. "
                      "WARNING: Choosing D0_minimal means the Board will have NO documented "
                      "governance response at this decision point. If the Board eventually acts "
                      "at a later decision point (D_rev), the delayed action will be evaluated "
                      "as REACTIVE — taken under escalating pressure — rather than PROACTIVE. "
                      "Australian courts and regulators distinguish between boards that led and "
                      "boards that were dragged.")
        lines.append(
            "- D1_review: Commission an independent governance review. "
            "A review addresses SYSTEMIC governance failures (risk oversight, executive "
            "accountability, Board monitoring) — it is not just about the CEO as an "
            "individual. Institutional investors and regulators expect boards to "
            "investigate root causes, not just change personnel. A review creates a "
            "documented remediation record that reduces director liability exposure. "
            "Under ASIC's cooperation framework (Information Sheet 172), a board-initiated "
            "review directly evidences self-policing and self-reporting — factors ASIC "
            "explicitly weighs for penalty reduction. The CBA/APRA precedent demonstrates "
            "that cooperative remediation can halve regulatory costs (APRA's $1B capital "
            "add-on was reduced in staged $500M tranches following CBA's Remedial Action "
            "Plan). Commissioning a review also strengthens the Board's defence under the "
            "business judgment rule (s 180(2) Corporations Act) by evidencing that directors "
            "informed themselves to a reasonable degree. "
            "TIMING ADVANTAGE: Commissioning the review NOW, at D1, captures the full "
            "proactive governance premium — the Board acts before being compelled by further "
            "shareholder, regulatory, or media pressure."
        )
        lines.append(
            "- D3_ceo_transition: Initiate CEO transition (remove and replace). "
            "NOTE: This action removes the CEO but does NOT commission a governance "
            "review. Without a review, the Board signals it views the crisis as a "
            "personnel problem rather than a systemic governance failure. Regulators "
            "and proxy advisers may interpret CEO-only action as insufficient. "
            "TIMING ADVANTAGE: Acting NOW preserves the Board's credibility. If CEO "
            "transition is deferred to a later decision point, the eventual removal "
            "will appear forced by escalating pressure (second strike, ASIC action, "
            "media) rather than principled governance. Delayed removal also increases "
            "the CEO's negotiating leverage for departure terms — the longer the Board "
            "waits, the more the CEO can argue constructive dismissal or negotiate "
            "larger exit packages."
        )
    elif node == "D_rev":
        if sv.get("d1_action") == "D3_ceo_transition":
            lines.append(f"\nThe Board has already removed the CEO. The question now is "
                          f"whether to commission an independent governance review to address "
                          f"systemic issues beyond the CEO departure. "
                          f"Feasible actions: {feasible_strs}.")
            lines.append("- Drev_no_action: No further governance action (CEO already removed)")
            lines.append(
                "- Drev_commission_review: Commission an independent governance review. "
                "Under ASIC's cooperation framework (IS 172), this evidences self-policing "
                "and self-reporting, factors weighted for penalty reduction. The CBA/APRA "
                "precedent shows cooperative remediation halved regulatory penalties. Also "
                "strengthens s 180(2) business judgment rule defence."
            )
        else:
            lines.append(f"\nThe Board must now decide on its post-AGM response. "
                          f"Feasible actions: {feasible_strs}.")
            lines.append("- Drev_no_action: Take no further governance action")
            lines.append(
                "- Drev_commission_review: Commission an independent governance review. "
                "Under ASIC's cooperation framework (IS 172), this evidences self-policing "
                "and self-reporting, factors weighted for penalty reduction. The CBA/APRA "
                "precedent shows cooperative remediation halved regulatory penalties. Also "
                "strengthens s 180(2) business judgment rule defence."
            )
            lines.append("- Drev_sack_ceo: Terminate the CEO")
    elif node == "D_rev_post":
        lines.append(f"\nFollowing the review findings, the Board must decide on its response. "
                      f"Feasible actions: {feasible_strs}.")
        lines.append("- Drev_no_action: Retain the CEO and implement review recommendations")
        lines.append("- Drev_sack_ceo: Terminate the CEO based on review findings")

    lines.append("\nWhat probability does the Board assign to each action?")

    return "\n".join(lines)


def _generate_tier1_scenarios() -> list[Scenario]:
    """Tier 1: Parameter isolation scenarios (30+)."""
    scenarios = []
    n = 0

    # w_passivity: Board passivity after CEO departure — CEO resigned vs present
    # Need sufficient CEO-resigned scenarios for factor rating regression.
    # Pair at low vote (no strike)
    for ceo_status in ["present", "resigned_early"]:
        n += 1
        scenarios.append(Scenario(
            scenario_id=f"S1_{n:03d}",
            tier=1,
            target_parameter="w_passivity",
            decision_node="D1",
            state_vector=_make_state_vector(
                decision_node="D1",
                ceo_status=ceo_status,
                vote_outcome=0.10,
                ceo_present_at_end=(ceo_status == "present"),
            ),
            feasible_actions=["D0_minimal", "D1_review", "D3_ceo_transition"],
        ))
    # CEO resigned at varied vote levels (strike, overwhelming) for robust w_passivity estimation
    for v in [0.30, 0.40, 0.55, 0.65, 0.83]:
        n += 1
        scenarios.append(Scenario(
            scenario_id=f"S1_{n:03d}",
            tier=1,
            target_parameter="w_passivity",
            decision_node="D1",
            state_vector=_make_state_vector(
                decision_node="D1",
                ceo_status="resigned_early",
                vote_outcome=v,
                ceo_present_at_end=False,
            ),
            feasible_actions=["D0_minimal", "D1_review", "D3_ceo_transition"],
        ))
    # CEO resigned at D_rev node (post-AGM review context)
    for v in [0.30, 0.50]:
        n += 1
        scenarios.append(Scenario(
            scenario_id=f"S1_{n:03d}",
            tier=1,
            target_parameter="w_passivity",
            decision_node="D_rev",
            state_vector=_make_state_vector(
                decision_node="D_rev",
                ceo_status="resigned_early",
                d1_action="D1_review",
                vote_outcome=v,
                review_commissioned=True,
                ceo_present_at_end=False,
            ),
            feasible_actions=["Drev_no_action", "Drev_commission_review", "Drev_sack_ceo"],
        ))

    # w2: Board accountability penalty — V varies.
    # w2 is now action-varying: phi = -(V-0.25) × (1 - 0.5 × response_strength).
    # The penalty is reduced but not eliminated by governance response.
    # High-V scenarios provide most leverage because (V-0.25) is large there.
    # At D1: phi varies between D0_minimal (penalised) vs D1_review/D3_ceo_transition (not).
    for v in [0.26, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.83]:
        n += 1
        scenarios.append(Scenario(
            scenario_id=f"S1_{n:03d}",
            tier=1,
            target_parameter="w2",
            decision_node="D1",
            state_vector=_make_state_vector(
                decision_node="D1",
                vote_outcome=v,
                ceo_present_at_end=True,  # CEO present — w2 varies by action
            ),
            feasible_actions=["D0_minimal", "D1_review", "D3_ceo_transition"],
        ))
    # w2 at D_rev: phi varies when d1=D0_minimal (board still inactive if Drev_no_action).
    # For d1=D1_review, board_inactive=False for all d_rev actions → w2 phi=0 everywhere.
    # So these scenarios must have d1=D0_minimal to get w2 variation.
    for v in [0.30, 0.50, 0.65, 0.83]:
        n += 1
        scenarios.append(Scenario(
            scenario_id=f"S1_{n:03d}",
            tier=1,
            target_parameter="w2",
            decision_node="D_rev",
            state_vector=_make_state_vector(
                decision_node="D_rev",
                d1_action="D0_minimal",
                vote_outcome=v,
                ceo_present_at_end=True,  # CEO present
            ),
            feasible_actions=["Drev_no_action", "Drev_commission_review", "Drev_sack_ceo"],
        ))

    # w3: Overwhelming penalty — V crosses 0.50, CEO removed, d1 != minimal
    for v in [0.48, 0.52]:
        n += 1
        scenarios.append(Scenario(
            scenario_id=f"S1_{n:03d}",
            tier=1,
            target_parameter="w3",
            decision_node="D_rev",
            state_vector=_make_state_vector(
                decision_node="D_rev",
                d1_action="D1_review",
                vote_outcome=v,
                review_commissioned=True,
                ceo_present_at_end=False,
            ),
            feasible_actions=["Drev_no_action", "Drev_commission_review", "Drev_sack_ceo"],
        ))

    # w4: Spill risk — strike present, V varies, CEO removed at end
    for v in [0.26, 0.35, 0.45]:
        n += 1
        scenarios.append(Scenario(
            scenario_id=f"S1_{n:03d}",
            tier=1,
            target_parameter="w4",
            decision_node="D1",
            state_vector=_make_state_vector(
                decision_node="D1",
                vote_outcome=v,
                ceo_present_at_end=False,
            ),
            feasible_actions=["D0_minimal", "D1_review", "D3_ceo_transition"],
        ))

    # w7: Implementation cost — sack vs no-sack, all else constant
    for action_context in ["sack", "no_sack"]:
        n += 1
        scenarios.append(Scenario(
            scenario_id=f"S1_{n:03d}",
            tier=1,
            target_parameter="w_removal",
            decision_node="D_rev",
            state_vector=_make_state_vector(
                decision_node="D_rev",
                d1_action="D0_minimal",
                vote_outcome=0.30,
                ceo_present_at_end=(action_context == "no_sack"),
            ),
            feasible_actions=["Drev_no_action", "Drev_commission_review", "Drev_sack_ceo"],
        ))

    # w_removal: CEO removal cost — CEO removed vs not, no strike
    for removed in [True, False]:
        n += 1
        scenarios.append(Scenario(
            scenario_id=f"S1_{n:03d}",
            tier=1,
            target_parameter="w_removal",
            decision_node="D_rev",
            state_vector=_make_state_vector(
                decision_node="D_rev",
                d1_action="D1_review",
                vote_outcome=0.20,
                review_commissioned=True,
                ceo_present_at_end=not removed,
            ),
            feasible_actions=["Drev_no_action", "Drev_commission_review", "Drev_sack_ceo"],
        ))

    # w9: Reputational spill — overwhelming, CEO removed (isolate from w10/w11/w14)
    for v in [0.48, 0.55]:
        n += 1
        scenarios.append(Scenario(
            scenario_id=f"S1_{n:03d}",
            tier=1,
            target_parameter="w9",
            decision_node="D1",
            state_vector=_make_state_vector(
                decision_node="D1",
                vote_outcome=v,
                ceo_present_at_end=False,
            ),
            feasible_actions=["D0_minimal", "D1_review", "D3_ceo_transition"],
        ))

    # w3/w9 separation: overwhelming with CEO retained vs removed (4 scenarios)
    for v, ceo_end in [(0.55, True), (0.55, False), (0.60, True), (0.60, False)]:
        n += 1
        scenarios.append(Scenario(
            scenario_id=f"S1_{n:03d}",
            tier=1,
            target_parameter="w3_w9_separation",
            decision_node="D_rev",
            state_vector=_make_state_vector(
                decision_node="D_rev",
                d1_action="D1_review",
                vote_outcome=v,
                review_commissioned=True,
                ceo_present_at_end=ceo_end,
            ),
            feasible_actions=["Drev_no_action", "Drev_commission_review", "Drev_sack_ceo"],
        ))

    # w10+w11+w14 (joint): Strike, CEO present, not overwhelming
    for v in [0.26, 0.30, 0.40]:
        n += 1
        scenarios.append(Scenario(
            scenario_id=f"S1_{n:03d}",
            tier=1,
            target_parameter="w10_w11_w14_joint",
            decision_node="D_rev",
            state_vector=_make_state_vector(
                decision_node="D_rev",
                d1_action="D0_minimal",
                vote_outcome=v,
                ceo_present_at_end=True,
            ),
            feasible_actions=["Drev_no_action", "Drev_commission_review", "Drev_sack_ceo"],
        ))

    # w12: Continued inaction liability — overwhelming
    # At D1: w12 fires for D0_minimal action when overwhelming (varies by action)
    for v in [0.55, 0.65]:
        n += 1
        scenarios.append(Scenario(
            scenario_id=f"S1_{n:03d}",
            tier=1,
            target_parameter="w12",
            decision_node="D1",
            state_vector=_make_state_vector(
                decision_node="D1",
                vote_outcome=v,
                ceo_present_at_end=True,
            ),
            feasible_actions=["D0_minimal", "D1_review", "D3_ceo_transition"],
        ))
    # At D_rev: w12 fires for Drev_no_action when d1=D0_minimal + overwhelming
    # Pair A: d1_action=D0_minimal → w12 fires for Drev_no_action only
    # Pair B: d1_action=D1_review → w12=0 for all actions (Board already acted)
    for d1 in ["D0_minimal", "D1_review"]:
        n += 1
        scenarios.append(Scenario(
            scenario_id=f"S1_{n:03d}",
            tier=1,
            target_parameter="w12",
            decision_node="D_rev",
            state_vector=_make_state_vector(
                decision_node="D_rev",
                vote_outcome=0.55,
                d1_action=d1,
                ceo_present_at_end=True,
            ),
            feasible_actions=["Drev_no_action", "Drev_commission_review", "Drev_sack_ceo"],
        ))
    # Additional D_rev pair at higher vote
    for d1 in ["D0_minimal", "D1_review"]:
        n += 1
        scenarios.append(Scenario(
            scenario_id=f"S1_{n:03d}",
            tier=1,
            target_parameter="w12",
            decision_node="D_rev",
            state_vector=_make_state_vector(
                decision_node="D_rev",
                vote_outcome=0.65,
                d1_action=d1,
                ceo_present_at_end=True,
            ),
            feasible_actions=["Drev_no_action", "Drev_commission_review", "Drev_sack_ceo"],
        ))

    # w13: Continued inaction liability — strike (not overwhelming)
    # At D1: w13 fires for D0_minimal action when strike
    for v in [0.26, 0.35]:
        n += 1
        scenarios.append(Scenario(
            scenario_id=f"S1_{n:03d}",
            tier=1,
            target_parameter="w13",
            decision_node="D1",
            state_vector=_make_state_vector(
                decision_node="D1",
                vote_outcome=v,
                ceo_present_at_end=True,
            ),
            feasible_actions=["D0_minimal", "D1_review", "D3_ceo_transition"],
        ))
    # At D_rev: w13 fires for Drev_no_action when d1=D0_minimal + strike
    # Pair A: d1_action=D0_minimal → w13 fires for Drev_no_action only
    # Pair B: d1_action=D1_review → w13=0 for all actions
    for d1 in ["D0_minimal", "D1_review"]:
        n += 1
        scenarios.append(Scenario(
            scenario_id=f"S1_{n:03d}",
            tier=1,
            target_parameter="w13",
            decision_node="D_rev",
            state_vector=_make_state_vector(
                decision_node="D_rev",
                vote_outcome=0.30,
                d1_action=d1,
                ceo_present_at_end=True,
            ),
            feasible_actions=["Drev_no_action", "Drev_commission_review", "Drev_sack_ceo"],
        ))
    # Additional D_rev pair at different vote
    for d1 in ["D0_minimal", "D1_review"]:
        n += 1
        scenarios.append(Scenario(
            scenario_id=f"S1_{n:03d}",
            tier=1,
            target_parameter="w13",
            decision_node="D_rev",
            state_vector=_make_state_vector(
                decision_node="D_rev",
                vote_outcome=0.40,
                d1_action=d1,
                ceo_present_at_end=True,
            ),
            feasible_actions=["Drev_no_action", "Drev_commission_review", "Drev_sack_ceo"],
        ))

    # w_review_negative / w_review_balanced: Review finding penalties (trinary)
    # Contrast triplets: negative vs balanced vs positive review outcome
    # Negative → w_review_negative fires; Balanced → w_review_balanced fires; Positive → neither (baseline)
    _review_target = {
        "negative": "w_review_negative",
        "balanced": "w_review_balanced",
        "positive": "w_review_negative",  # positive is baseline contrast for both
    }
    for review_result, car in [("negative", -0.03), ("balanced", 0.00), ("positive", 0.02)]:
        n += 1
        scenarios.append(Scenario(
            scenario_id=f"S1_{n:03d}",
            tier=1,
            target_parameter=_review_target[review_result],
            decision_node="D_rev_post",
            state_vector=_make_state_vector(
                decision_node="D_rev_post",
                d1_action="D1_review",
                vote_outcome=0.35,
                review_commissioned=True,
                review_outcome=review_result,
                car_outcome=car,
                ceo_present_at_end=True,
            ),
            feasible_actions=["Drev_no_action", "Drev_sack_ceo"],
        ))
    # Same contrast at higher vote
    for review_result, car in [("negative", -0.05), ("balanced", -0.01), ("positive", 0.03)]:
        n += 1
        scenarios.append(Scenario(
            scenario_id=f"S1_{n:03d}",
            tier=1,
            target_parameter=_review_target[review_result],
            decision_node="D_rev_post",
            state_vector=_make_state_vector(
                decision_node="D_rev_post",
                d1_action="D1_review",
                vote_outcome=0.50,
                review_commissioned=True,
                review_outcome=review_result,
                car_outcome=car,
                ceo_present_at_end=True,
            ),
            feasible_actions=["Drev_no_action", "Drev_sack_ceo"],
        ))
    # Overwhelming vote: negative, balanced, and positive review
    for review_result, car in [("negative", -0.05), ("balanced", -0.02), ("positive", 0.03)]:
        n += 1
        scenarios.append(Scenario(
            scenario_id=f"S1_{n:03d}",
            tier=1,
            target_parameter=_review_target[review_result],
            decision_node="D_rev_post",
            state_vector=_make_state_vector(
                decision_node="D_rev_post",
                d1_action="D1_review",
                vote_outcome=0.60,
                review_commissioned=True,
                review_outcome=review_result,
                car_outcome=car,
                ceo_present_at_end=True,
            ),
            feasible_actions=["Drev_no_action", "Drev_sack_ceo"],
        ))

    # w_inaction contrast: strike + CEO present vs strike + CEO removed
    # CEO present: w_inaction fires (penalty for inaction)
    # CEO removed: w8s fires (shock relief benefit)
    for ceo_end in [True, False]:
        n += 1
        scenarios.append(Scenario(
            scenario_id=f"S1_{n:03d}",
            tier=1,
            target_parameter="w_inaction",
            decision_node="D_rev",
            state_vector=_make_state_vector(
                decision_node="D_rev",
                vote_outcome=0.35,
                d1_action="D0_minimal",
                ceo_present_at_end=ceo_end,
            ),
            feasible_actions=["Drev_no_action", "Drev_commission_review", "Drev_sack_ceo"],
        ))

    # w_inaction_delay identification: contrast d1=D0_minimal vs d1=D1_review at D_rev.
    # When d1=D0_minimal and Board acts at D_rev, w_inaction_delay fires (-1).
    # When d1=D1_review, w_inaction_delay is always 0 (Board already acted proactively).
    # Paired scenarios at matching vote levels sharpen estimation of the delay cost.
    for v in [0.30, 0.45, 0.55, 0.65]:
        for d1 in ["D0_minimal", "D1_review"]:
            n += 1
            scenarios.append(Scenario(
                scenario_id=f"S1_{n:03d}",
                tier=1,
                target_parameter="w_inaction_delay",
                decision_node="D_rev",
                state_vector=_make_state_vector(
                    decision_node="D_rev",
                    d1_action=d1,
                    vote_outcome=v,
                    review_commissioned=(d1 == "D1_review"),
                    ceo_present_at_end=True,
                ),
                feasible_actions=(
                    ["Drev_no_action", "Drev_commission_review", "Drev_sack_ceo"]
                    if d1 == "D0_minimal"
                    else ["Drev_no_action", "Drev_sack_ceo"]
                ),
            ))

    # w_inaction_delay: EARLY-vs-LATE paired scenarios.
    # These contrast the SAME governance action taken at D1 (proactive) versus
    # the same action deferred to D_rev after doing nothing at D1 (reactive).
    # The D_rev scenarios have d1=D0_minimal, so w_inaction_delay fires when
    # the Board eventually acts.  The D1 scenarios provide the baseline where
    # w_inaction_delay is structurally 0.  The LLM prompt context now explains
    # Australian legal costs of delay (s 180 duty, class action window expansion,
    # ASIC cooperation credit decay), so the LLM should rate the D_rev/reactive
    # scenarios more harshly than the D1/proactive ones.
    #
    # A. Commission review: D1_review (proactive) vs D_rev commission (reactive)
    for v in [0.30, 0.40, 0.55, 0.70, 0.83]:
        # A1. Proactive: Board commissions review at D1
        n += 1
        scenarios.append(Scenario(
            scenario_id=f"S1_{n:03d}",
            tier=1,
            target_parameter="w_inaction_delay_review",
            decision_node="D1",
            state_vector=_make_state_vector(
                decision_node="D1",
                vote_outcome=v,
                ceo_present_at_end=True,
            ),
            feasible_actions=["D0_minimal", "D1_review", "D3_ceo_transition"],
        ))
        # A2. Reactive: Board did nothing at D1, now commissions review at D_rev
        n += 1
        scenarios.append(Scenario(
            scenario_id=f"S1_{n:03d}",
            tier=1,
            target_parameter="w_inaction_delay_review",
            decision_node="D_rev",
            state_vector=_make_state_vector(
                decision_node="D_rev",
                d1_action="D0_minimal",
                vote_outcome=v,
                review_commissioned=False,
                ceo_present_at_end=True,
            ),
            feasible_actions=["Drev_no_action", "Drev_commission_review", "Drev_sack_ceo"],
        ))

    # B. CEO transition: D3_ceo_transition (proactive) vs D_rev sack (reactive)
    for v in [0.30, 0.40, 0.55, 0.70, 0.83]:
        # B1. Proactive: Board initiates CEO transition at D1
        n += 1
        scenarios.append(Scenario(
            scenario_id=f"S1_{n:03d}",
            tier=1,
            target_parameter="w_inaction_delay_ceo",
            decision_node="D1",
            state_vector=_make_state_vector(
                decision_node="D1",
                vote_outcome=v,
                ceo_present_at_end=True,
            ),
            feasible_actions=["D0_minimal", "D1_review", "D3_ceo_transition"],
        ))
        # B2. Reactive: Board did nothing at D1, now sacks CEO at D_rev
        n += 1
        scenarios.append(Scenario(
            scenario_id=f"S1_{n:03d}",
            tier=1,
            target_parameter="w_inaction_delay_ceo",
            decision_node="D_rev",
            state_vector=_make_state_vector(
                decision_node="D_rev",
                d1_action="D0_minimal",
                vote_outcome=v,
                review_commissioned=False,
                ceo_present_at_end=True,
            ),
            feasible_actions=["Drev_no_action", "Drev_commission_review", "Drev_sack_ceo"],
        ))

    # C. CEO resigned scenarios: proactive review at D1 vs reactive review at D_rev
    # When CEO has resigned, the D1 choice is D0_minimal vs D1_review (no D3).
    # Delay cost is purely about review timing, not CEO timing.
    for v in [0.30, 0.55, 0.83]:
        # C1. Proactive: CEO resigned, Board commissions review at D1
        n += 1
        scenarios.append(Scenario(
            scenario_id=f"S1_{n:03d}",
            tier=1,
            target_parameter="w_inaction_delay_review_ceo_gone",
            decision_node="D1",
            state_vector=_make_state_vector(
                decision_node="D1",
                ceo_status="resigned_early",
                vote_outcome=v,
                ceo_present_at_end=False,
            ),
            feasible_actions=["D0_minimal", "D1_review"],
        ))
        # C2. Reactive: CEO resigned, Board did nothing at D1, commissions at D_rev
        n += 1
        scenarios.append(Scenario(
            scenario_id=f"S1_{n:03d}",
            tier=1,
            target_parameter="w_inaction_delay_review_ceo_gone",
            decision_node="D_rev",
            state_vector=_make_state_vector(
                decision_node="D_rev",
                ceo_status="resigned_early",
                d1_action="D0_minimal",
                vote_outcome=v,
                review_commissioned=False,
                ceo_present_at_end=False,
            ),
            feasible_actions=["Drev_no_action", "Drev_commission_review"],
        ))

    # w8s identification: NON-STRIKE CEO removal scenarios
    # These have strike=False, so w8s phi=0 and w_inaction phi=0 regardless of
    # CEO removal.  w_removal phi still varies by action (sack vs keep).
    # Purpose: give the estimator data where w_removal varies independently
    # of w8s/w_inaction, breaking the collinearity in strike scenarios.
    # D1 node at V < 0.25 (no strike): D3_ceo_transition fires w_removal but not w8s.
    for v in [0.10, 0.15, 0.20]:
        n += 1
        scenarios.append(Scenario(
            scenario_id=f"S1_{n:03d}",
            tier=1,
            target_parameter="w8s_identification",
            decision_node="D1",
            state_vector=_make_state_vector(
                decision_node="D1",
                vote_outcome=v,
                ceo_present_at_end=True,  # actions decide removal
            ),
            feasible_actions=["D0_minimal", "D1_review", "D3_ceo_transition"],
        ))
    # D_rev node at V < 0.25 (no strike): sack fires w_removal but not w8s.
    for v in [0.10, 0.20]:
        n += 1
        scenarios.append(Scenario(
            scenario_id=f"S1_{n:03d}",
            tier=1,
            target_parameter="w8s_identification",
            decision_node="D_rev",
            state_vector=_make_state_vector(
                decision_node="D_rev",
                d1_action="D1_review",
                vote_outcome=v,
                review_commissioned=True,
                ceo_present_at_end=True,  # actions decide removal
            ),
            feasible_actions=["Drev_no_action", "Drev_commission_review", "Drev_sack_ceo"],
        ))
    # Strike with removal vs no-removal at MULTIPLE vote levels
    # (more data for estimating the w8s increment above w_removal)
    for v in [0.30, 0.40, 0.50]:
        for ceo_end in [True, False]:
            n += 1
            scenarios.append(Scenario(
                scenario_id=f"S1_{n:03d}",
                tier=1,
                target_parameter="w8s_identification",
                decision_node="D_rev",
                state_vector=_make_state_vector(
                    decision_node="D_rev",
                    d1_action="D0_minimal",
                    vote_outcome=v,
                    ceo_present_at_end=ceo_end,
                ),
                feasible_actions=["Drev_no_action", "Drev_commission_review", "Drev_sack_ceo"],
            ))

    # ── Sack-without-review scenarios (breaks w_removal / w_inaction collinearity) ──
    # Tree branch: Board removed CEO at D1 (D3_ceo_transition), now at D_rev
    # deciding whether to commission a governance review.
    # Key identification: w_removal fires (from D1 sack) but w_inaction does NOT
    # fire (CEO is already gone → ceo_present_at_end=False).
    # Feasible actions: Drev_no_action / Drev_commission_review (sack not feasible,
    # CEO already removed).
    for v in [0.26, 0.30, 0.40, 0.50, 0.60, 0.83]:
        n += 1
        scenarios.append(Scenario(
            scenario_id=f"S1_{n:03d}",
            tier=1,
            target_parameter="w_removal_inaction_separation",
            decision_node="D_rev",
            state_vector=_make_state_vector(
                decision_node="D_rev",
                d1_action="D3_ceo_transition",
                vote_outcome=v,
                ceo_present_at_end=False,
            ),
            feasible_actions=["Drev_no_action", "Drev_commission_review"],
        ))

    # ── w_review_post_removal: Due diligence review after involuntary CEO removal ──
    # Identifies the interaction term: -I[removed_involuntary AND NOT review_commissioned].
    # At D_rev after D3_ceo_transition:
    #   phi(w_review_post_removal) = [-1, 0]  for [no_action, commission_review]
    #   phi(w_inaction_no_review)  = [ 0, 0]  (w_inaction_no_review now zero once CEO removed)
    # With the decoupling, any penalty for skipping a review AFTER removal loads
    # uniquely onto w_review_post_removal instead of competing with w_inaction_no_review.
    #
    # Additional scenarios at vote levels not covered by w_removal_inaction_separation,
    # and with explicit prompt framing about post-removal due diligence.
    for v in [0.35, 0.45, 0.55, 0.70]:
        n += 1
        scenarios.append(Scenario(
            scenario_id=f"S1_{n:03d}",
            tier=1,
            target_parameter="w_review_post_removal",
            decision_node="D_rev",
            state_vector=_make_state_vector(
                decision_node="D_rev",
                d1_action="D3_ceo_transition",
                vote_outcome=v,
                ceo_present_at_end=False,
            ),
            feasible_actions=["Drev_no_action", "Drev_commission_review"],
        ))

    # Extra neutral (no-strike) post-removal scenarios to isolate w_review_post_removal.
    # CEO already removed; only decision is to commission a review vs do nothing.
    # Votes kept below strike threshold so vote penalties stay zero and other
    # weights (w_inaction_base, w_strike, w_overwhelming) are silent.
    for v in [0.05, 0.15, 0.24]:
        n += 1
        scenarios.append(Scenario(
            scenario_id=f"S1_{n:03d}",
            tier=1,
            target_parameter="w_review_post_removal",
            decision_node="D_rev",
            state_vector=_make_state_vector(
                decision_node="D_rev",
                d1_action="D3_ceo_transition",
                vote_outcome=v,
                ceo_present_at_end=False,
            ),
            feasible_actions=["Drev_no_action", "Drev_commission_review"],
        ))

    # ── w_ceo_accountability: Evidence-based CEO removal benefit ──
    # phi = +I[removed_involuntary AND review_commissioned].
    # Fires when Board removes CEO after commissioning a governance review,
    # providing evidence-based legitimacy for the removal decision.
    #
    # Identification strategy:
    #   - At D_rev_post (review always commissioned): sack has w_removal=-1 AND
    #     w_ceo_accountability=+1. Net effect = w_accountability - w_removal.
    #   - At D1 (D3_ceo_transition, no review): only w_removal=-1 fires.
    #   - At D_rev (d1=D0_minimal, no review): only w_removal=-1 fires.
    #   These D1/D_rev scenarios pin w_removal; D_rev_post then identifies
    #   w_ceo_accountability given the pinned w_removal.
    #
    # A. D_rev_post with NEGATIVE review — strong accountability mandate.
    #    Board has evidence of governance failures; sacking is justified.
    for v in [0.30, 0.40, 0.55, 0.70, 0.83]:
        n += 1
        scenarios.append(Scenario(
            scenario_id=f"S1_{n:03d}",
            tier=1,
            target_parameter="w_ceo_accountability",
            decision_node="D_rev_post",
            state_vector=_make_state_vector(
                decision_node="D_rev_post",
                d1_action="D1_review",
                vote_outcome=v,
                review_commissioned=True,
                review_outcome="negative",
                car_outcome=-0.05,
                ceo_present_at_end=True,
            ),
            feasible_actions=["Drev_no_action", "Drev_sack_ceo"],
        ))

    # B. D_rev_post with BALANCED review — moderate accountability.
    #    Review found mixed results; sacking may or may not be justified.
    for v in [0.30, 0.50, 0.70]:
        n += 1
        scenarios.append(Scenario(
            scenario_id=f"S1_{n:03d}",
            tier=1,
            target_parameter="w_ceo_accountability",
            decision_node="D_rev_post",
            state_vector=_make_state_vector(
                decision_node="D_rev_post",
                d1_action="D1_review",
                vote_outcome=v,
                review_commissioned=True,
                review_outcome="balanced",
                car_outcome=-0.01,
                ceo_present_at_end=True,
            ),
            feasible_actions=["Drev_no_action", "Drev_sack_ceo"],
        ))

    # C. D_rev_post with POSITIVE review — weak accountability mandate.
    #    Review found no governance failures; sacking lacks evidence basis.
    #    These scenarios help separate w_ceo_accountability from w_removal:
    #    here, review_commissioned=True but review is positive, so w_ceo_accountability
    #    still fires (removal is backed by a review process, even if findings were positive).
    #    However, w_review_negative and w_review_balanced do NOT fire, so the LLM
    #    should rate sacking less favourably than in negative-review scenarios.
    for v in [0.30, 0.50, 0.70]:
        n += 1
        scenarios.append(Scenario(
            scenario_id=f"S1_{n:03d}",
            tier=1,
            target_parameter="w_ceo_accountability",
            decision_node="D_rev_post",
            state_vector=_make_state_vector(
                decision_node="D_rev_post",
                d1_action="D1_review",
                vote_outcome=v,
                review_commissioned=True,
                review_outcome="positive",
                car_outcome=0.02,
                ceo_present_at_end=True,
            ),
            feasible_actions=["Drev_no_action", "Drev_sack_ceo"],
        ))

    # D. D_rev sacking WITHOUT review (d1=D0_minimal) — pins w_removal alone.
    #    Here review_commissioned=False, so w_ceo_accountability=0 for sack.
    #    Only w_removal fires. These scenarios provide the pure removal cost
    #    that the model needs to separately identify w_removal.
    for v in [0.30, 0.40, 0.55, 0.70]:
        n += 1
        scenarios.append(Scenario(
            scenario_id=f"S1_{n:03d}",
            tier=1,
            target_parameter="w_ceo_accountability",
            decision_node="D_rev",
            state_vector=_make_state_vector(
                decision_node="D_rev",
                d1_action="D0_minimal",
                vote_outcome=v,
                review_commissioned=False,
                ceo_present_at_end=True,
            ),
            feasible_actions=["Drev_no_action", "Drev_commission_review", "Drev_sack_ceo"],
        ))

    return scenarios


def _generate_tier2_scenarios() -> list[Scenario]:
    """Tier 2: Joint multi-penalty scenarios (20+)."""
    scenarios = []
    configs = [
        # (vote, d1_action, review, review_outcome, car, ceo_end, node)
        (0.30, "D1_review", True, "negative", -0.05, False, "D_rev_post"),
        (0.30, "D1_review", True, "positive", 0.02, True, "D_rev"),
        (0.40, "D0_minimal", False, None, None, True, "D_rev"),
        (0.55, "D1_review", True, "negative", -0.08, False, "D_rev_post"),
        (0.55, "D0_minimal", False, None, None, True, "D_rev"),
        (0.60, "D1_review", True, "negative", -0.03, True, "D_rev_post"),
        (0.83, "D0_minimal", False, None, None, True, "D_rev"),
        (0.83, "D1_review", True, "negative", -0.14, False, "D_rev_post"),
        (0.26, "D1_review", True, "negative", -0.01, True, "D_rev_post"),
        (0.35, "D0_minimal", False, None, None, False, "D_rev"),
        (0.40, "D1_review", True, "positive", 0.03, False, "D_rev"),
        (0.50, "D1_review", True, "negative", -0.05, True, "D_rev_post"),
        (0.60, "D0_minimal", False, None, None, False, "D1"),
        (0.75, "D0_minimal", False, None, None, True, "D_rev"),
        (0.30, "D3_ceo_transition", False, None, None, False, "D1"),
        (0.10, "D0_minimal", False, None, None, True, "D1"),
        (0.20, "D1_review", True, "negative", -0.08, False, "D_rev_post"),
        (0.45, "D1_review", True, "positive", 0.05, True, "D_rev"),
        (0.52, "D1_review", True, "balanced", -0.03, False, "D_rev_post"),
        (0.70, "D1_review", True, "negative", -0.14, True, "D_rev_post"),
        # D_rev after Board chose CEO transition — CEO already gone
        (0.20, "D3_ceo_transition", False, None, None, False, "D_rev"),
        (0.30, "D3_ceo_transition", False, None, None, False, "D_rev"),
        (0.55, "D3_ceo_transition", False, None, None, False, "D_rev"),
        (0.83, "D3_ceo_transition", False, None, None, False, "D_rev"),
        # NEW: balanced review outcome scenarios
        (0.35, "D1_review", True, "balanced", -0.01, True, "D_rev_post"),
        (0.50, "D1_review", True, "balanced", 0.00, False, "D_rev_post"),
    ]
    for i, (v, d1, rev, rev_outcome, car, ceo_end, node) in enumerate(configs, 1):
        if node == "D_rev_post":
            fa = ["Drev_no_action", "Drev_sack_ceo"]
        elif node == "D_rev" and d1 == "D3_ceo_transition":
            fa = ["Drev_no_action", "Drev_commission_review"]  # CEO already removed
        elif node == "D_rev":
            fa = ["Drev_no_action", "Drev_commission_review", "Drev_sack_ceo"]
        else:
            fa = ["D0_minimal", "D1_review", "D3_ceo_transition"]
        scenarios.append(Scenario(
            scenario_id=f"S2_{i:03d}",
            tier=2,
            target_parameter="joint",
            decision_node=node,
            state_vector=_make_state_vector(
                decision_node=node,
                d1_action=d1,
                vote_outcome=v,
                review_commissioned=rev,
                review_outcome=rev_outcome,
                car_outcome=car,
                ceo_present_at_end=ceo_end,
            ),
            feasible_actions=fa,
        ))

    # ── CAR-based scale anchor scenarios ──
    # Scenario pair with known CAR difference to calibrate utility-to-CAR ratio.
    # Same state, different CAR outcomes. The probability difference between
    # these scenarios anchors the utility scale.
    # CAR = -0.01 (1% one-off remediation cost)
    # CAR = -0.12 (12 months of 1%/month compounding — cost of delayed action)
    for car_val in [-0.01, -0.12]:
        scenarios.append(Scenario(
            scenario_id=f"S2_anchor_car{int(abs(car_val)*100):03d}",
            tier=2,
            target_parameter="car_scale_anchor",
            decision_node="D_rev_post",
            state_vector=_make_state_vector(
                decision_node="D_rev_post",
                d1_action="D1_review",
                vote_outcome=0.35,
                review_commissioned=True,
                review_outcome="negative",
                car_outcome=car_val,
                ceo_present_at_end=True,
            ),
            feasible_actions=["Drev_no_action", "Drev_sack_ceo"],
        ))

    return scenarios


def _generate_tier3_scenarios() -> list[Scenario]:
    """Tier 3: Behavioural probe scenarios (20+)."""
    scenarios = []
    n = 0

    # 8.1 Loss aversion: matched gain/loss CAR pairs
    for car_mag in [0.01, 0.03, 0.05, 0.08, 0.14]:
        for sign in [1, -1]:
            n += 1
            car = sign * car_mag
            scenarios.append(Scenario(
                scenario_id=f"S3_{n:03d}",
                tier=3,
                target_parameter="loss_aversion",
                decision_node="D_rev",
                state_vector=_make_state_vector(
                    decision_node="D_rev",
                    d1_action="D1_review",
                    vote_outcome=0.30,
                    review_commissioned=True,
                    review_outcome="negative" if car < 0 else "positive",
                    car_outcome=car,
                    ceo_present_at_end=True,
                ),
                feasible_actions=["Drev_no_action", "Drev_commission_review", "Drev_sack_ceo"],
            ))

    # 8.3 Optimism bias: explicit vs implicit adverse probability
    for explicit in [True, False]:
        n += 1
        sv = _make_state_vector(
            decision_node="D_rev",
            d1_action="D1_review",
            vote_outcome=0.35,
            review_commissioned=True,
            review_outcome=None,
            ceo_present_at_end=True,
        )
        if explicit:
            sv["explicit_adverse_prob"] = 0.67
        scenarios.append(Scenario(
            scenario_id=f"S3_{n:03d}",
            tier=3,
            target_parameter="optimism_bias",
            decision_node="D_rev",
            state_vector=sv,
            feasible_actions=["Drev_no_action", "Drev_commission_review", "Drev_sack_ceo"],
        ))
    # Additional optimism bias at higher vote
    for explicit in [True, False]:
        n += 1
        sv = _make_state_vector(
            decision_node="D_rev",
            d1_action="D1_review",
            vote_outcome=0.50,
            review_commissioned=True,
            review_outcome=None,
            ceo_present_at_end=True,
        )
        if explicit:
            sv["explicit_adverse_prob"] = 0.67
        scenarios.append(Scenario(
            scenario_id=f"S3_{n:03d}",
            tier=3,
            target_parameter="optimism_bias",
            decision_node="D_rev",
            state_vector=sv,
            feasible_actions=["Drev_no_action", "Drev_commission_review", "Drev_sack_ceo"],
        ))

    # 8.4 Self-assessment bias: board-initiated vs externally-mandated review
    # Vary vote levels and CAR outcomes for multiple observations per group
    sa_configs = [
        (0.35, -0.05),
        (0.40, -0.05),
        (0.50, -0.08),
        (0.60, -0.03),
        (0.83, -0.14),
    ]
    for origin in ["board_initiated", "externally_mandated"]:
        for vote, car in sa_configs:
            n += 1
            scenarios.append(Scenario(
                scenario_id=f"S3_{n:03d}",
                tier=3,
                target_parameter="self_assessment_bias",
                decision_node="D_rev_post",
                state_vector=_make_state_vector(
                    decision_node="D_rev_post",
                    d1_action="D1_review",
                    vote_outcome=vote,
                    review_commissioned=True,
                    review_outcome="negative",
                    car_outcome=car,
                    review_origin=origin,
                    ceo_present_at_end=True,
                ),
                feasible_actions=["Drev_no_action", "Drev_sack_ceo"],
            ))

    # 8.5 Ikea effect: CEO appointment (appointed vs inherited)
    # Vary vote levels for multiple observations per group
    ikea_votes = [0.30, 0.40, 0.50, 0.60, 0.83]
    for appt in ["appointed_by_current_board", "inherited"]:
        for vote in ikea_votes:
            n += 1
            scenarios.append(Scenario(
                scenario_id=f"S3_{n:03d}",
                tier=3,
                target_parameter="ikea_effect",
                decision_node="D_rev",
                state_vector=_make_state_vector(
                    decision_node="D_rev",
                    d1_action="D0_minimal",
                    vote_outcome=vote,
                    ceo_appointment=appt,
                    ceo_present_at_end=True,
                ),
                feasible_actions=["Drev_no_action", "Drev_commission_review", "Drev_sack_ceo"],
            ))

    # 8.5 Scenario triplet (self-assessment vs Ikea interaction)
    # (i) Board-initiated review, adverse
    n += 1
    scenarios.append(Scenario(
        scenario_id=f"S3_{n:03d}",
        tier=3,
        target_parameter="ikea_vs_self_assessment",
        decision_node="D_rev_post",
        state_vector=_make_state_vector(
            decision_node="D_rev_post", d1_action="D1_review",
            vote_outcome=0.40, review_commissioned=True,
            review_outcome="negative", car_outcome=-0.05,
            review_origin="board_initiated", ceo_present_at_end=True,
        ),
        feasible_actions=["Drev_no_action", "Drev_sack_ceo"],
    ))
    # (ii) Externally-mandated review, negative
    n += 1
    scenarios.append(Scenario(
        scenario_id=f"S3_{n:03d}",
        tier=3,
        target_parameter="ikea_vs_self_assessment",
        decision_node="D_rev_post",
        state_vector=_make_state_vector(
            decision_node="D_rev_post", d1_action="D1_review",
            vote_outcome=0.40, review_commissioned=True,
            review_outcome="negative", car_outcome=-0.05,
            review_origin="externally_mandated", ceo_present_at_end=True,
        ),
        feasible_actions=["Drev_no_action", "Drev_sack_ceo"],
    ))
    # (iii) Board-initiated review, positive
    n += 1
    scenarios.append(Scenario(
        scenario_id=f"S3_{n:03d}",
        tier=3,
        target_parameter="ikea_vs_self_assessment",
        decision_node="D_rev_post",
        state_vector=_make_state_vector(
            decision_node="D_rev_post", d1_action="D1_review",
            vote_outcome=0.40, review_commissioned=True,
            review_outcome="positive", car_outcome=0.03,
            review_origin="board_initiated", ceo_present_at_end=True,
        ),
        feasible_actions=["Drev_no_action", "Drev_sack_ceo"],
    ))

    return scenarios


def _generate_tier4_scenario() -> Scenario:
    """Tier 4: Historical calibration — Qantas AGM Nov 2023."""
    return Scenario(
        scenario_id="S4_001",
        tier=4,
        target_parameter="historical_calibration",
        decision_node="D1",
        state_vector=_make_state_vector(
            decision_node="D1",
            ceo_status="present",
            vote_outcome=0.83,
            ceo_present_at_end=True,
        ),
        feasible_actions=["D0_minimal", "D1_review", "D3_ceo_transition"],
    )


def generate_scenarios(output_path: Path) -> list[Scenario]:
    """Stage 1: Generate all scenarios and save to CSV."""
    logger.info("Stage 1: Generating scenario battery...")

    scenarios = []
    scenarios.extend(_generate_tier1_scenarios())
    scenarios.extend(_generate_tier2_scenarios())
    scenarios.extend(_generate_tier3_scenarios())
    scenarios.append(_generate_tier4_scenario())

    # Build prompt text for each scenario
    for s in scenarios:
        s.prompt_text = _build_scenario_prompt(s)

    logger.info(f"Generated {len(scenarios)} scenarios: "
                f"Tier 1 = {sum(1 for s in scenarios if s.tier == 1)}, "
                f"Tier 2 = {sum(1 for s in scenarios if s.tier == 2)}, "
                f"Tier 3 = {sum(1 for s in scenarios if s.tier == 3)}, "
                f"Tier 4 = {sum(1 for s in scenarios if s.tier == 4)}")

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", errors="replace", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "scenario_id", "tier", "target_parameter", "decision_node",
            "state_vector", "feasible_actions", "prompt_text", "created_at",
        ])
        writer.writeheader()
        for s in scenarios:
            writer.writerow({
                "scenario_id": s.scenario_id,
                "tier": s.tier,
                "target_parameter": s.target_parameter,
                "decision_node": s.decision_node,
                "state_vector": json.dumps(s.state_vector, ensure_ascii=True),
                "feasible_actions": json.dumps(s.feasible_actions, ensure_ascii=True),
                "prompt_text": s.prompt_text,
                "created_at": datetime.now().isoformat(),
            })

    logger.info(f"Scenarios saved to {output_path}")
    return scenarios


def load_scenarios(path: Path) -> list[Scenario]:
    """Load scenarios from CSV."""
    scenarios = []
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scenarios.append(Scenario(
                scenario_id=row["scenario_id"],
                tier=int(row["tier"]),
                target_parameter=row["target_parameter"],
                decision_node=row["decision_node"],
                state_vector=json.loads(row["state_vector"]),
                feasible_actions=json.loads(row["feasible_actions"]),
                prompt_text=row["prompt_text"],
            ))
    return scenarios


# ── SEC 6: Stage 2 — LLM elicitation ─────────────────────────────────────────

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
    # Deterministic RNG per (scenario, draw) for action order randomization.
    _content_seed = int(hashlib.sha256(
        f"{scenario.prompt_text}|{draw}".encode()
    ).hexdigest(), 16) & 0xFFFFFFFF
    rng = np.random.default_rng(_content_seed)

    # Randomize action presentation order to avoid position bias
    n_actions = len(scenario.feasible_actions)
    action_order = rng.permutation(n_actions).tolist()
    shuffled_actions = [scenario.feasible_actions[i] for i in action_order]

    # Deterministic hash token for stochastic variation (gpt-5+ has no temperature)
    hash_token = hashlib.sha256(
        f"draw_{draw}_{scenario.scenario_id}".encode()
    ).hexdigest()[:16]

    # Build scenario prompt with shuffled actions and hash token
    action_list_str = "\n".join(
        f"  - {a}" for a in shuffled_actions
    )
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
    """Stage 2: Run LLM Likert elicitation across all scenarios."""
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
        with tqdm(total=len(tasks), desc="Elicitation", smoothing=0) as pbar:
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


# ── SEC 7: Stage 3 — Data preprocessing ──────────────────────────────────────

def preprocess_likert_data(
    elicitation_path: Path,
    long_output_path: Path,
    summary_output_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stage 3: Preprocess Likert elicitation results into long-format + summary.

    Returns:
        likert_long_df: One row per (scenario, action, draw) observation.
            Columns: scenario_id, action, draw, score, action_order_position
        likert_summary_df: One row per (scenario, action) pair.
            Columns: scenario_id, action, n_draws, mean_score, sd_score
    """
    logger.info("Stage 3: Preprocessing Likert elicitation data...")

    df = pd.read_csv(elicitation_path, encoding="utf-8")
    df_valid = df[df["parse_status"].isin(["success", "repaired"])].copy()

    # Require at least 3 successful draws per scenario (adaptive to small n_draws)
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

    # Explode action_scores JSON into long format
    long_records = []
    for _, row in df_valid.iterrows():
        scores_dict = json.loads(row["action_scores"])
        action_order = json.loads(row["action_order"]) if row.get("action_order") else []

        for action, score in scores_dict.items():
            # Skip hallucinated actions that aren't in the scenario's feasible set
            if action_order and action not in action_order:
                logger.debug(
                    f"preprocess_likert_data: dropping hallucinated action "
                    f"'{action}' for {row['scenario_id']} draw={row['draw']} "
                    f"(feasible: {action_order})"
                )
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
        logger.warning("No Likert scores found in elicitation data! "
                        "Check that action_scores contains non-empty dicts.")
        empty_long = pd.DataFrame(columns=[
            "scenario_id", "action", "draw", "score", "action_order_position",
        ])
        empty_summary = pd.DataFrame(columns=[
            "scenario_id", "action", "n_draws", "mean_score", "sd_score",
        ])
        return empty_long, empty_summary

    # Summary statistics per (scenario, action) pair
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

    # Save outputs
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


# ── SEC 7B: Pre-flight identifiability checks ────────────────────────────────


def _scenario_phi_signature(scenario: "Scenario") -> dict[str, float]:
    """Compute phi signature for a scenario's dominant action vs. baseline.

    Returns a dict of {param: max|phi_a - phi_baseline|} across actions,
    indicating which parameters this scenario can help identify.
    """
    sv = scenario.state_vector if isinstance(scenario.state_vector, dict) else {}
    ceo_res = sv.get("ceo_status_at_start", "present") == "resigned_early"
    ceo_present = sv.get("ceo_present_at_end", True)
    d1 = sv.get("d1_action", "D0_minimal")
    overwhelming = sv.get("overwhelming", False)
    review_comm_base = bool(sv.get("review_commissioned", False))
    review_result = sv.get("review_outcome")  # "negative", "balanced", "positive", or None

    # Response strength map for w_passivity graduation
    _rs_map = {
        "D0_minimal": 0.0, "D1_review": 0.5, "D3_ceo_transition": 1.0,
        "Drev_no_action": 0.0, "Drev_commission_review": 0.5, "Drev_sack_ceo": 1.0,
    }

    actions = scenario.feasible_actions
    if not actions:
        return {}

    # Compute phi for each action using the same logic as decompose_utility_board.
    # Map the action to the correct game tree variable based on the decision node.
    node = scenario.decision_node
    d_rev_action_base = sv.get("d_rev_action", "Drev_no_action")

    phi_by_action = []
    for a in actions:
        rs = _rs_map.get(a, 0.0)
        removed_inv = 1.0 if a in ("D3_ceo_transition", "Drev_sack_ceo") else 0.0

        # review_commissioned: action-derived
        review_comm = review_comm_base or a in ("D1_review", "Drev_commission_review")

        # At D1, the action IS the d1 choice; at other nodes, d1 is from state vector.
        d1_eff = a if node == "D1" else d1

        # board_inactive logic (uses effective d1)
        board_inactive = (d1_eff == "D0_minimal")
        if a in ("Drev_sack_ceo", "Drev_commission_review"):
            board_inactive = False

        # w_inaction_delay: Board did nothing at D1, then acted at D_rev or D_rev_post.
        # Map the current action to the appropriate d_rev/d_rev_post variable.
        if node == "D_rev":
            d_rev_a = a
            d_rev_post_a = "Drev_no_action"
        elif node == "D_rev_post":
            d_rev_a = d_rev_action_base
            d_rev_post_a = a
        else:
            d_rev_a = "Drev_no_action"
            d_rev_post_a = "Drev_no_action"
        delay = (d1_eff == "D0_minimal"
                 and (d_rev_a in ("Drev_commission_review", "Drev_sack_ceo")
                      or d_rev_post_a in ("Drev_commission_review", "Drev_sack_ceo")))

        phi = {
            "w_inaction_base": -float(board_inactive),
            "w_inaction_no_review": -float(not review_comm),
            "w_inaction_delay": -float(delay),
            "w_passivity": -float(ceo_res) * (1.0 - rs),
            "w_removal": -removed_inv,
            "w_remove_ceo_overwhelming": removed_inv * float(overwhelming),
            "w_review_negative": -float(review_comm and review_result == "negative"),
            "w_review_balanced": -float(review_comm and review_result == "balanced"),
            "w_review_post_removal": -float(removed_inv and not review_comm),
            "w_ceo_accountability": float(removed_inv) * float(review_comm),
        }
        phi_by_action.append(phi)

    # Max absolute difference between any pair of actions for each param
    max_diff = {p: 0.0 for p in ESTIMABLE_PARAM_NAMES}
    for i in range(len(phi_by_action)):
        for j in range(i + 1, len(phi_by_action)):
            for p in ESTIMABLE_PARAM_NAMES:
                diff = abs(phi_by_action[i].get(p, 0.0) - phi_by_action[j].get(p, 0.0))
                if diff > max_diff[p]:
                    max_diff[p] = diff

    return max_diff


def run_preflight_checks(
    scenarios: list["Scenario"],
    estimation_df: pd.DataFrame | None = None,
) -> dict:
    """Run pre-flight identifiability checks on scenario design.

    Called twice:
    1. Post-generation (estimation_df=None): structural checks on scenario design
    2. Post-elicitation (estimation_df provided): data-dependent checks

    Returns dict with check results and pass/fail status.
    """
    results = {"checks": [], "all_passed": True}

    # ── Post-generation checks (always run) ──

    # Check A: Pairwise parameter separation
    # Each pair of estimable parameters needs >= 3 scenarios where they
    # have different phi variation patterns (so they can be distinguished).
    #
    # Excluded pairs: parameters on the same dimension that are structurally
    # non-separable within a single scenario but identified through
    # between-scenario variation (different scenarios activate different params).
    # The ordinal probit identifies these through cross-scenario contrasts.
    EXCLUDED_PAIRS = {
        frozenset(("w_review_negative", "w_review_balanced")),  # same dimension: review outcome type
    }
    sep_counts = {}
    for i, p1 in enumerate(ESTIMABLE_PARAM_NAMES):
        for p2 in ESTIMABLE_PARAM_NAMES[i + 1:]:
            if frozenset((p1, p2)) not in EXCLUDED_PAIRS:
                sep_counts[(p1, p2)] = 0

    for sc in scenarios:
        sig = _scenario_phi_signature(sc)
        for pair in sep_counts:
            p1, p2 = pair
            v1 = sig.get(p1, 0.0) > 0.01
            v2 = sig.get(p2, 0.0) > 0.01
            if v1 != v2:
                sep_counts[pair] += 1

    min_sep = min(sep_counts.values()) if sep_counts else 0
    worst_pair = min(sep_counts, key=sep_counts.get) if sep_counts else ("", "")
    check_a_pass = min_sep >= 3
    detail = f"Min separating scenarios: {min_sep} (worst pair: {worst_pair[0]}/{worst_pair[1]})"
    if EXCLUDED_PAIRS:
        excl_str = ", ".join(f"{sorted(p)[0]}/{sorted(p)[1]}" for p in EXCLUDED_PAIRS)
        detail += f". Excluded (between-scenario identification): {excl_str}"
    results["checks"].append({
        "name": "A: Pairwise parameter separation",
        "passed": check_a_pass,
        "detail": detail,
        "threshold": ">= 3 separating scenarios per pair",
    })
    if not check_a_pass:
        results["all_passed"] = False

    # Check B: Decision node coverage
    node_counts = {}
    for sc in scenarios:
        node_counts[sc.decision_node] = node_counts.get(sc.decision_node, 0) + 1

    required_nodes = {"D1", "D_rev", "D_rev_post"}
    min_per_node = 10
    check_b_pass = True
    node_details = []
    for node in required_nodes:
        count = node_counts.get(node, 0)
        node_details.append(f"{node}={count}")
        if count < min_per_node:
            check_b_pass = False

    results["checks"].append({
        "name": "B: Decision node coverage",
        "passed": check_b_pass,
        "detail": ", ".join(node_details),
        "threshold": f">= {min_per_node} scenarios per node",
    })
    if not check_b_pass:
        results["all_passed"] = False

    # Check C: Vote range coverage (at least 3 per quartile)
    vote_values = []
    for sc in scenarios:
        sv = sc.state_vector if isinstance(sc.state_vector, dict) else {}
        v = sv.get("vote_outcome_V")
        if v is not None:
            vote_values.append(v)

    quartile_bounds = [(0, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.0)]
    quartile_counts = []
    for lo, hi in quartile_bounds:
        count = sum(1 for v in vote_values if lo <= v < hi)
        quartile_counts.append(count)
    min_quartile = min(quartile_counts) if quartile_counts else 0
    check_c_pass = min_quartile >= 3
    results["checks"].append({
        "name": "C: Vote range coverage",
        "passed": check_c_pass,
        "detail": f"Quartile counts: {quartile_counts} (min={min_quartile})",
        "threshold": ">= 3 scenarios per quartile",
    })
    if not check_c_pass:
        results["all_passed"] = False

    # Check D: Phi matrix condition number (structural, using scenario signatures)
    # Build a simplified phi variation matrix: rows=scenarios, cols=params
    phi_var_matrix = []
    for sc in scenarios:
        sig = _scenario_phi_signature(sc)
        row = [sig.get(p, 0.0) for p in ESTIMABLE_PARAM_NAMES]
        phi_var_matrix.append(row)

    phi_mat = np.array(phi_var_matrix)
    if phi_mat.shape[0] > 0 and phi_mat.shape[1] > 0:
        # Condition number of the phi variation matrix
        try:
            sv_vals = np.linalg.svd(phi_mat, compute_uv=False)
            sv_vals = sv_vals[sv_vals > 1e-12]
            if len(sv_vals) > 0:
                cond_number = sv_vals[0] / sv_vals[-1]
            else:
                cond_number = float("inf")
        except np.linalg.LinAlgError:
            cond_number = float("inf")
    else:
        cond_number = float("inf")

    check_d_pass = cond_number < 1000
    check_d_warn = cond_number >= 100
    results["checks"].append({
        "name": "D: Phi matrix condition number",
        "passed": check_d_pass,
        "warning": check_d_warn and check_d_pass,
        "detail": f"Condition number: {cond_number:.1f}",
        "threshold": "< 1000 (warning at 100)",
    })
    if not check_d_pass:
        results["all_passed"] = False

    # ── Post-elicitation checks (only when likert_summary_df provided) ──

    if estimation_df is not None and not estimation_df.empty:
        likert_df = estimation_df  # may be likert_summary_df passed as estimation_df

        # Check F: Scenario Likert score discrimination
        # For each scenario, check that max_score - min_score > 0 across actions
        n_low_disc = 0
        n_scenarios_checked = 0
        for sid, grp in likert_df.groupby("scenario_id") if "mean_score" in likert_df.columns else []:
            scores = grp["mean_score"].tolist()
            if len(scores) >= 2:
                n_scenarios_checked += 1
                delta = max(scores) - min(scores)
                if delta < 0.1:
                    n_low_disc += 1

        if n_scenarios_checked > 0:
            pct_low = 100 * n_low_disc / n_scenarios_checked
        else:
            pct_low = 0.0
        check_f_pass = pct_low < 30
        results["checks"].append({
            "name": "F: Likert score discrimination",
            "passed": check_f_pass,
            "detail": f"{n_low_disc}/{n_scenarios_checked} scenarios with score spread < 0.1 ({pct_low:.0f}%)",
            "threshold": "< 30% of scenarios with near-uniform Likert scores",
        })
        if not check_f_pass:
            results["all_passed"] = False

        # Check G: Design matrix condition number with observed data
        elicited_ids = set(likert_df["scenario_id"].tolist()) if "scenario_id" in likert_df.columns else set()
        elicited_scenarios = [s for s in scenarios if s.scenario_id in elicited_ids]
        phi_obs_matrix = []
        for sc in elicited_scenarios:
            sig = _scenario_phi_signature(sc)
            row = [sig.get(p, 0.0) for p in ESTIMABLE_PARAM_NAMES]
            phi_obs_matrix.append(row)

        phi_obs = np.array(phi_obs_matrix) if phi_obs_matrix else np.zeros((0, len(ESTIMABLE_PARAM_NAMES)))
        if phi_obs.shape[0] > 0:
            try:
                sv_obs = np.linalg.svd(phi_obs, compute_uv=False)
                sv_obs = sv_obs[sv_obs > 1e-12]
                cond_obs = sv_obs[0] / sv_obs[-1] if len(sv_obs) > 0 else float("inf")
            except np.linalg.LinAlgError:
                cond_obs = float("inf")
        else:
            cond_obs = float("inf")

        check_g_pass = cond_obs < 1000
        results["checks"].append({
            "name": "G: Observed design matrix condition",
            "passed": check_g_pass,
            "detail": f"Condition number: {cond_obs:.1f} ({len(elicited_scenarios)} scenarios)",
            "threshold": "< 1000",
        })
        if not check_g_pass:
            results["all_passed"] = False

        # Check H: Response saturation — flag if all actions get the same
        # Likert score in > 30% of scenarios
        n_saturated = 0
        n_total = 0
        for sid, grp in likert_df.groupby("scenario_id") if "mean_score" in likert_df.columns else []:
            scores = grp["mean_score"].tolist()
            if scores:
                n_total += 1
                if max(scores) == min(scores):
                    n_saturated += 1

        pct_sat = 100 * n_saturated / max(n_total, 1)
        check_h_pass = pct_sat <= 30
        results["checks"].append({
            "name": "H: Likert saturation",
            "passed": check_h_pass,
            "detail": f"{n_saturated}/{n_total} scenarios with identical scores ({pct_sat:.0f}%)",
            "threshold": "<= 30% saturated responses",
        })
        if not check_h_pass:
            results["all_passed"] = False

    # Summary
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


# ── SEC 8: Stage 4 — Choice model estimation ─────────────────────────────────

def decompose_utility_board(
    vote_percent: float,
    strike: bool,
    overwhelming: bool,
    d1_action: str,
    d_rev_action: str,
    d_rev_post_action: str,
    CEO_removed: bool,
    CEO_resigned_early: bool,
    review_commissioned: bool,
    review_outcome: str,  # "none", "negative", "balanced", "positive"
    review_car: float,
    review_direct_cost: float,
) -> dict[str, float]:
    """
    Decompose Board utility into per-parameter basis function values.

    Convention: EU = sum_k w_k * phi_k + anchored. Softmax: P(a) ~ exp(lambda * EU(a)).
    Higher EU = more likely action. All weights are non-negative (>= 0).
    Therefore:
    - PENALTY terms have NEGATIVE phi: larger weight = lower EU = less likely.
    - BENEFIT terms have POSITIVE phi: larger weight = higher EU = more likely.

    Structure:
    1. INACTION COMPONENTS (2 additive, unconditional — fire regardless of vote level):
       w_inaction_base       = -I[board_inactive]
       w_inaction_no_review  = -I[not review_commissioned]

    2. RETAINED:
       w_passivity = -I[CEO_resigned_early] × (1 - response_strength)
       w_removal = -I[removed_involuntary]
       w_remove_ceo_overwhelming      = +I[removed_involuntary] × I[overwhelming]
       w_review_negative  = -I[review_comm ∧ review_negative]
       w_review_balanced  = -I[review_comm ∧ review_balanced]

    3. VOTE PENALTIES (scenario-level, in anchored contribution):
       w_strike, w_overwhelming — don't vary by action, enter as fixed contribution.

    """
    ceo_present_at_end = not CEO_removed and not CEO_resigned_early
    removed_involuntary = float(CEO_removed and not CEO_resigned_early)

    # board_inactive: Board took minimal action at ALL decision points.
    # At D1: fires for D0_minimal (choosing inaction).
    # At D_rev: fires for Drev_no_action when d1_action=D0_minimal (continued inaction).
    # At D_rev_post: fires for Drev_no_action when all prior decisions were minimal.
    board_inactive = (d1_action == "D0_minimal")
    if d_rev_action in ("Drev_sack_ceo", "Drev_commission_review"):
        board_inactive = False
    if d_rev_post_action == "Drev_sack_ceo":
        board_inactive = False

    # response_strength for w_passivity graduation (retained from prior model)
    _D1_STRENGTH = {"D0_minimal": 0.0, "D1_review": 0.5, "D3_ceo_transition": 1.0}
    _DREV_STRENGTH = {"Drev_no_action": 0.0, "Drev_commission_review": 0.5, "Drev_sack_ceo": 1.0}
    response_strength = max(
        _D1_STRENGTH.get(d1_action, 0.0),
        _DREV_STRENGTH.get(d_rev_action, 0.0),
        _DREV_STRENGTH.get(d_rev_post_action, 0.0),
    )

    phi = {
        # ── INACTION COMPONENTS (unconditional — fire regardless of vote) ──
        #
        # w_inaction_base: Board took minimal action at ALL decision points.
        # At D1: [-1, 0, 0] for [D0_minimal, D1_review, D3_ceo_transition]
        # At D_rev: [-1, 0, 0] for [no_action, commission, sack] (when d1=D0_minimal)
        #           [0, 0, 0] if d1 was D1_review or D3 (already responded)
        "w_inaction_base": -float(board_inactive),
        #
        # w_inaction_no_review: No governance review commissioned WHILE CEO REMAINS.
        # Limits scope to cases where the CEO is still in place; the post-removal
        # due-diligence penalty is captured separately by w_review_post_removal.
        # At D1: [-1, 0, 0] for [D0_minimal, D1_review, D3_ceo_transition]
        # At D_rev: [-1, 0, 0] for [no_action, commission, sack] when CEO present
        "w_inaction_no_review": -float(not review_commissioned and not removed_involuntary),
        #
        # w_inaction_delay: Board delayed governance action — did nothing at D1
        # then acted reactively at D_rev or D_rev_post.  Captures the cost of
        # reactive vs proactive governance (market/regulator credibility loss).
        # At D1: [0, 0, 0] (delay hasn't occurred yet — resolved downstream)
        # At D_rev (d1=D0_minimal): [0, -1, -1] for [no_action, commission, sack]
        # At D_rev (d1=D1_review): [0, 0, 0] (Board already acted proactively)
        # Complements w_inaction_base: base penalises TOTAL inaction,
        # delay penalises REACTIVE governance (acted, but too late).
        "w_inaction_delay": -float(
            d1_action == "D0_minimal"
            and (d_rev_action in ("Drev_commission_review", "Drev_sack_ceo")
                 or d_rev_post_action in ("Drev_commission_review", "Drev_sack_ceo"))
        ),
        # ── RETAINED PARAMETERS ──
        #
        # w_passivity: Board passivity after CEO departure — penalty for failing to respond.
        # Graduated: full penalty when Board does nothing, zero when Board responds decisively.
        # Pattern at D1: [-1, -0.5, 0] for [D0_minimal, D1_review, D3_ceo_transition]
        "w_passivity": -float(CEO_resigned_early) * (1.0 - response_strength),
        #
        # w_removal: CEO involuntary removal cost (implementation + disruption).
        # At D1: [0, 0, -1] for [D0_minimal, D1_review, D3_ceo_transition]
        "w_removal": -removed_involuntary,
        #
        # w_remove_ceo_overwhelming: CEO removal shock relief when overwhelming vote occurred.
        # BENEFIT: reduces the cost of removal in severe crisis.
        "w_remove_ceo_overwhelming": removed_involuntary * float(overwhelming),
        #
        # w_review_negative: Negative review finding penalty.
        # Fires whenever a commissioned review returns negative findings,
        # regardless of CEO status — reflects on Board governance quality.
        "w_review_negative": -float(review_commissioned and review_outcome == "negative"),
        # w_review_balanced: Balanced review finding penalty.
        # Fires for balanced findings — less severe than negative but still
        # indicates governance gaps vs. a positive/clean review.
        "w_review_balanced": -float(review_commissioned and review_outcome == "balanced"),
        #
        # w_review_post_removal: Due diligence penalty for NOT commissioning a review
        # after involuntarily removing the CEO.  Captures the context-specific incentive
        # that only exists when the Board has sacked the CEO and needs an independent
        # review to justify the removal and address systemic governance gaps.
        # At D1: [0, 0, -1] for [D0_minimal, D1_review, D3_ceo_transition]
        # At D_rev (d1=D3): [-1, 0] for [no_action, commission_review]
        # At D_rev (d1=D0, CEO present): [0, 0, -1] for [no_action, commission, sack]
        "w_review_post_removal": -float(removed_involuntary and not review_commissioned),
        #
        # w_ceo_accountability: Accountability benefit — CEO removal backed by governance review.
        # BENEFIT: fires when Board removes CEO AND a governance review was commissioned,
        # providing evidence-based justification for the removal decision.
        # Breaks w_removal collinearity: at D1 (D3_ceo_transition, no review yet) only
        # w_removal fires; at D_rev/D_rev_post after review, both fire.
        # phi = +I[removed_involuntary AND review_commissioned]
        "w_ceo_accountability": float(removed_involuntary) * float(review_commissioned),
    }
    return phi


def _compute_anchored_contribution(
    vote_percent: float,
    strike: bool,
    overwhelming: bool,
    review_commissioned: bool,
    review_car: float,
    review_direct_cost: float,
    lambda_la: float = LAMBDA_LA_DEFAULT,
) -> float:
    """Compute anchored (non-estimable) contribution to utility.

    Vote penalties (w_strike, w_overwhelming) are now estimated in Stan and
    no longer included here.  Only review CAR + direct cost remain as anchored.
    """
    contrib = 0.0

    # Review CAR + direct cost (anchored weights)
    if review_commissioned:
        w_car_pos = W_CAR_ANCHOR / ((1 + lambda_la) / 2)
        w_car_neg = lambda_la * w_car_pos
        contrib += w_car_pos * max(review_car, 0.0) - w_car_neg * max(-review_car, 0.0)
        contrib -= W_COST_ANCHOR * review_direct_cost

    return contrib


def _scenario_to_outcome_args(sv: dict, action: str) -> dict:
    """Convert a scenario state vector + action into decompose_utility_board kwargs."""
    node = sv["decision_node"]
    ceo_status = sv.get("ceo_status_at_start", "present")
    CEO_resigned_early = (ceo_status == "resigned_early")

    d1_action = sv.get("d1_action", "D0_minimal")
    d_rev_action = "Drev_no_action"
    d_rev_post_action = "Drev_no_action"

    # Map action to the appropriate node
    if node == "D1":
        d1_action = action
    elif node == "D_rev":
        d_rev_action = action
    elif node == "D_rev_post":
        d_rev_post_action = action

    # Determine CEO_removed based on action
    CEO_removed = CEO_resigned_early or (
        action in ("D3_ceo_transition", "Drev_sack_ceo")
    )
    # Override with state vector if explicitly set
    if "ceo_present_at_end" in sv:
        # For identification scenarios, ceo_present_at_end is set by the scenario
        # But the action may also cause removal — take the union
        if not sv["ceo_present_at_end"]:
            CEO_removed = True

    # review_commissioned: action-derived (D1_review or Drev_commission_review
    # commission a review), OR from scenario state if review was already commissioned
    # before this decision node.
    review_commissioned = bool(sv.get("review_commissioned", False))
    if action in ("D1_review", "Drev_commission_review"):
        review_commissioned = True

    return {
        "vote_percent": sv.get("vote_outcome_V", 0.0),
        "strike": sv.get("strike", False),
        "overwhelming": sv.get("overwhelming", False),
        "d1_action": d1_action,
        "d_rev_action": d_rev_action,
        "d_rev_post_action": d_rev_post_action,
        "CEO_removed": CEO_removed,
        "CEO_resigned_early": CEO_resigned_early,
        "review_commissioned": review_commissioned,
        "review_outcome": sv.get("review_outcome") or "none",
        "review_car": sv.get("car_outcome", 0.0) or 0.0,
        "review_direct_cost": 0.00096,  # mean of Gamma(4.55, 4741)
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
        scenario_id_map: dict mapping scenario_id -> unique scenario index (1-based for Stan)
        scenario_ids: list of scenario IDs in order of first appearance
        action_lists: list of feasible action lists per scenario
    """
    logger.info("Computing basis function matrix (phi) for (scenario, action) pairs...")

    n_params = len(WEIGHT_PARAM_NAMES)
    valid_sids = set(likert_summary_df["scenario_id"].tolist())

    # Filter scenarios to those in estimation dataset (exclude Tier 4)
    valid_scenarios = [s for s in scenarios if s.scenario_id in valid_sids]

    # Build index of actions present in Likert data per scenario.
    # If elicitation was run with a broader action set than the current
    # scenario definition, include those actions so ratings are not dropped.
    likert_actions_per_sid: dict[str, set[str]] = {}
    for sid in valid_sids:
        likert_actions_per_sid[sid] = set(
            likert_summary_df.loc[
                likert_summary_df["scenario_id"] == sid, "action"
            ].tolist()
        )

    # Build (scenario, action) pairs in deterministic order
    sa_pairs = []
    scenario_ids = []
    action_lists = []
    scenario_id_map: dict[str, int] = {}  # scenario_id -> 1-based index (for Stan)

    # Valid actions per decision node — used to filter stale Likert data
    # that may reference actions from a different node (due to scenario ID shifts).
    _VALID_ACTIONS_BY_NODE = {
        "D1": {"D0_minimal", "D1_review", "D3_ceo_transition"},
        "D_rev": {"Drev_no_action", "Drev_commission_review", "Drev_sack_ceo"},
        "D_rev_post": {"Drev_no_action", "Drev_commission_review", "Drev_sack_ceo"},
    }

    for scenario in valid_scenarios:
        if scenario.scenario_id not in scenario_id_map:
            scenario_id_map[scenario.scenario_id] = len(scenario_id_map) + 1
        scenario_ids.append(scenario.scenario_id)
        # Union of scenario feasible actions and actions with Likert data,
        # filtered to actions valid for this decision node.
        actions = list(scenario.feasible_actions)
        valid_for_node = _VALID_ACTIONS_BY_NODE.get(scenario.decision_node, set())
        extra = likert_actions_per_sid.get(scenario.scenario_id, set()) - set(actions)
        # Filter out cross-node actions (stale Likert data from ID shifts)
        valid_extra = extra & valid_for_node if valid_for_node else extra
        dropped = extra - valid_extra
        if dropped:
            logger.debug(
                f"compute_phi_matrix: {scenario.scenario_id} — dropping "
                f"{sorted(dropped)} from Likert data (wrong node: {scenario.decision_node})"
            )
        if valid_extra:
            logger.debug(
                f"compute_phi_matrix: {scenario.scenario_id} — adding "
                f"{sorted(valid_extra)} from Likert data (not in feasible_actions)"
            )
            actions.extend(sorted(valid_extra))
        action_lists.append(actions)
        for action in actions:
            sa_pairs.append((scenario.scenario_id, action))

    S = len(sa_pairs)
    sa_id_map = {pair: idx for idx, pair in enumerate(sa_pairs)}

    phi = np.zeros((S, n_params))
    anchored = np.zeros(S)
    vote_x_strike = np.zeros(S)
    vote_x_overwh = np.zeros(S)
    has_strike_arr = np.zeros(S, dtype=int)
    has_overwh_arr = np.zeros(S, dtype=int)

    for s_idx, (sid, action) in enumerate(sa_pairs):
        scenario = next(s for s in valid_scenarios if s.scenario_id == sid)
        args = _scenario_to_outcome_args(scenario.state_vector, action)
        phi_k = decompose_utility_board(**args)

        for k, pname in enumerate(WEIGHT_PARAM_NAMES):
            phi[s_idx, k] = phi_k.get(pname, 0.0)

        anchored[s_idx] = _compute_anchored_contribution(
            args["vote_percent"], args["strike"], args["overwhelming"],
            args["review_commissioned"], args["review_car"], args["review_direct_cost"],
        )

        # Vote data for linear penalty (passed to Stan separately)
        v = args["vote_percent"]
        vote_x_strike[s_idx] = max(0.0, (v - 0.25) / 0.75) if args["strike"] else 0.0
        vote_x_overwh[s_idx] = max(0.0, (v - 0.50) / 0.50) if args["overwhelming"] else 0.0
        has_strike_arr[s_idx] = int(args["strike"])
        has_overwh_arr[s_idx] = int(args["overwhelming"])

    # ── Center anchored within each scenario ──
    # The scenario random effect absorbs scenario-level constants.
    # Centering preserves the action-varying component (e.g. review CAR
    # that depends on whether the action commissions a review) while
    # keeping mu ~ O(1).  Vote penalties are now in Stan (nonlinear),
    # not in anchored.
    anchored_raw = anchored.copy()
    for sid in scenario_id_map:
        sa_indices = [sa_id_map[(sid, a)] for a in
                      action_lists[scenario_ids.index(sid)]]
        scenario_mean = np.mean(anchored[sa_indices])
        for idx in sa_indices:
            anchored[idx] -= scenario_mean

    n_strike = int(has_strike_arr.sum())
    n_overwh = int(has_overwh_arr.sum())
    logger.info(f"Phi matrix shape: {phi.shape}, {S} (scenario,action) pairs, "
                f"{len(scenario_id_map)} unique scenarios")
    logger.info(f"Anchored: raw range [{anchored_raw.min():.1f}, {anchored_raw.max():.1f}], "
                f"centered range [{anchored.min():.3f}, {anchored.max():.3f}]")
    logger.info(f"Vote data: {n_strike}/{S} pairs with strike, "
                f"{n_overwh}/{S} with overwhelming")

    vote_data = {
        "vote_x_strike": vote_x_strike,
        "vote_x_overwh": vote_x_overwh,
        "has_strike": has_strike_arr,
        "has_overwh": has_overwh_arr,
    }
    return phi, anchored, sa_id_map, scenario_id_map, scenario_ids, action_lists, vote_data


@dataclass
class StanEstimationResult:
    """Posterior summary from Bayesian ordinal probit estimation via Stan.

    Designed as a drop-in replacement for EstimationResult in downstream
    code: all attributes accessed by SEC 8C and later sections are
    preserved with identical names and types.
    """

    # ── Raw posterior draws ──
    w_draws: np.ndarray            # (n_draws, K) posterior weight samples
    cutpoint_draws: np.ndarray     # (n_draws, 4) posterior cutpoints
    sigma_scenario_draws: np.ndarray  # (n_draws,) scenario RE SD
    y_rep: Optional[np.ndarray] = None  # (n_draws, N) posterior predictive

    # ── Posterior summaries (populated post-fit) ──
    weights_posterior_mean: dict = field(default_factory=dict)
    weights_posterior_sd: dict = field(default_factory=dict)
    weights_posterior_ci: dict = field(default_factory=dict)

    # ── MCMC diagnostics ──
    n_divergences: int = 0
    max_rhat: float = 0.0
    min_ess_bulk: float = 0.0
    n_samples: int = 0

    # ── Compatibility attributes (EstimationResult interface) ──
    # These mirror EstimationResult fields so SEC 8C+ code requires no changes.
    weights: dict = field(default_factory=dict)
    lambda_rationality: float = 1.0
    hessian_se: dict = field(default_factory=dict)
    bootstrap_se: dict = field(default_factory=dict)
    covariance_matrix: np.ndarray = field(
        default_factory=lambda: np.zeros((9, 9))
    )
    condition_number: float = 0.0
    ridge_applied: bool = False
    w10_w11_w14_collapsed: bool = True
    loss_value: float = 0.0        # placeholder for dashboard compatibility
    n_scenarios: int = 0
    converged: bool = True
    estimation_method: dict = field(default_factory=dict)
    jackknife_se: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialise to JSON-safe dict matching EstimationResult.to_dict() schema."""
        return {
            "weights": self.weights,
            "lambda_rationality": round(self.lambda_rationality, 4),
            "hessian_se": {k: round(v, 4) for k, v in self.hessian_se.items()},
            "bootstrap_se": {k: round(v, 4) for k, v in self.bootstrap_se.items()},
            "condition_number": round(self.condition_number, 2),
            "ridge_applied": self.ridge_applied,
            "w10_w11_w14_collapsed": self.w10_w11_w14_collapsed,
            "loss_value": round(self.loss_value, 6),
            "n_scenarios": self.n_scenarios,
            "converged": self.converged,
            "estimation_method": self.estimation_method,
            "jackknife_se": {k: round(v, 4) for k, v in self.jackknife_se.items()},
            # Stan-specific extras
            "n_divergences": self.n_divergences,
            "max_rhat": round(self.max_rhat, 4),
            "min_ess_bulk": round(self.min_ess_bulk, 1),
            "n_samples": self.n_samples,
            "weights_posterior_sd": {
                k: round(v, 4) for k, v in self.weights_posterior_sd.items()
            },
            "weights_posterior_ci": self.weights_posterior_ci,
        }


# Type alias so downstream references to EstimationResult still resolve.
EstimationResult = StanEstimationResult


def prepare_stan_data(
    likert_long_df: "pd.DataFrame",
    phi: np.ndarray,
    anchored: np.ndarray,
    sa_id_map: dict,
    scenario_id_map: dict,
    vote_data: Optional[dict] = None,
) -> dict:
    """Build the data dict for ordinal_utility.stan.

    Parameters
    ----------
    likert_long_df:
        Long-form Likert data with columns:
        ``scenario_id``, ``action``, ``rating`` (int in 1..5).
    phi:
        (S, K) basis-function matrix, one row per unique (scenario, action)
        pair, indexed by the 0-based row indices stored in ``sa_id_map``.
    anchored:
        (S,) anchored utility contribution per (scenario, action) pair.
    sa_id_map:
        dict mapping ``(scenario_id, action)`` -> 0-based row index in phi.
    scenario_id_map:
        dict mapping ``scenario_id`` -> 1-based integer (Stan is 1-indexed).
    vote_data:
        dict with keys ``vote_x_strike``, ``vote_x_overwh``, ``has_strike``,
        ``has_overwh`` — arrays of length S for nonlinear vote penalties.

    Returns
    -------
    dict
        Stan data block with all keys required by ordinal_utility.stan,
        using Python native int/float/list types (not NumPy scalars).
    """
    import pandas as pd  # local import — pandas already on path

    S, K = phi.shape
    N_scenarios = int(len(scenario_id_map))

    # Per-(scenario,action) pair: which scenario does it belong to?
    # scenario_id[s] (1-based) for each row s in phi.
    scenario_id_per_sa = [0] * S
    for (sid, _action), row_idx in sa_id_map.items():
        scenario_id_per_sa[row_idx] = int(scenario_id_map[sid])

    # Build observation-level arrays from likert_long_df.
    # Each row is one Likert rating.  We look up the 1-based sa_id for
    # the (scenario_id, action) pair associated with that rating.
    y_list: list[int] = []
    sa_id_list: list[int] = []

    for _, row in likert_long_df.iterrows():
        sid = row["scenario_id"]
        action = row["action"]
        rating = int(row["score"])
        key = (sid, action)
        if key not in sa_id_map:
            # Stale Likert data (e.g. cross-node actions from ID shifts) —
            # already filtered upstream by compute_phi_matrix.
            logger.debug(
                f"prepare_stan_data: skipping observation with unknown key {key}"
            )
            continue
        if not (1 <= rating <= 5):
            logger.warning(
                f"prepare_stan_data: skipping out-of-range rating {rating} for {key}"
            )
            continue
        y_list.append(rating)
        sa_id_list.append(int(sa_id_map[key]) + 1)  # convert to 1-based

    N = len(y_list)
    if N == 0:
        raise ValueError(
            "prepare_stan_data: no valid Likert observations found. "
            "Check that likert_long_df contains columns 'scenario_id', "
            "'action', 'score' and that scenario/action keys match sa_id_map."
        )

    # Vote data arrays (default to zeros if not provided)
    S_int = int(S)
    if vote_data is not None:
        vx_strike = [float(v) for v in vote_data["vote_x_strike"]]
        vx_overwh = [float(v) for v in vote_data["vote_x_overwh"]]
        h_strike = [int(v) for v in vote_data["has_strike"]]
        h_overwh = [int(v) for v in vote_data["has_overwh"]]
    else:
        vx_strike = [0.0] * S_int
        vx_overwh = [0.0] * S_int
        h_strike = [0] * S_int
        h_overwh = [0] * S_int

    stan_data = {
        "N": int(N),
        "S": S_int,
        "K": int(K),
        "y": [int(v) for v in y_list],
        "sa_id": [int(v) for v in sa_id_list],
        "phi": phi.tolist(),
        "anchored": anchored.tolist(),
        "N_scenarios": int(N_scenarios),
        "scenario_id": [int(v) for v in scenario_id_per_sa],
        "vote_x_strike": vx_strike,
        "vote_x_overwh": vx_overwh,
        "has_strike": h_strike,
        "has_overwh": h_overwh,
        "mu_scale": 1.0,  # placeholder — set by fit_ordinal_probit from init mu range
    }

    n_strike = sum(h_strike)
    n_overwh = sum(h_overwh)
    logger.info(
        f"prepare_stan_data: N={N} obs, S={S_int} (scenario,action) pairs, "
        f"K={K} linear weights, {N_scenarios} unique scenarios, "
        f"vote: {n_strike} strike / {n_overwh} overwhelming"
    )
    return stan_data


def fit_ordinal_probit(
    stan_data: dict,
    stan_model_path: Optional[str] = None,
    chains: int = 4,
    iter_warmup: int = 1000,
    iter_sampling: int = 2000,
    adapt_delta: float = 0.99,
    max_treedepth: int = 15,
    seed: int = 42,
) -> dict:
    """Compile and sample ordinal_utility.stan via CmdStanPy.

    Parameters
    ----------
    stan_data:
        Data dict produced by ``prepare_stan_data()``.
    stan_model_path:
        Absolute path to ``ordinal_utility.stan``.  Defaults to
        ``<PROJECT_ROOT>/models/ordinal_utility.stan``.
    chains, iter_warmup, iter_sampling, adapt_delta, max_treedepth, seed:
        CmdStanPy sampling arguments.

    Returns
    -------
    dict with keys:
        ``fit``         — CmdStanMCMC object
        ``w``           — np.ndarray (n_draws, K) posterior weight draws
        ``w_strike``    — np.ndarray (n_draws,) vote strike penalty draws
        ``w_overwh``    — np.ndarray (n_draws,) vote overwhelming penalty draws
        ``cutpoints``   — np.ndarray (n_draws, 4)
        ``sigma_scenario`` — np.ndarray (n_draws,)
        ``y_rep``       — np.ndarray (n_draws, N)  or None if GQ not available
        ``n_divergences`` — int
        ``max_rhat``    — float
        ``min_ess_bulk`` — float
        ``n_samples``   — int  (chains × iter_sampling)
    """
    import platform

    # Ensure C++ toolchain is discoverable on Windows (same as fit_belief_model_stan.py)
    if platform.system() == "Windows" and "MAKE" not in os.environ:
        _rtools_make = r"C:\rtools40\usr\bin\make.exe"
        _rtools_gpp = r"C:\rtools40\ucrt64\bin"
        if os.path.isfile(_rtools_make):
            os.environ["MAKE"] = _rtools_make
            os.environ["PATH"] = (
                _rtools_gpp + os.pathsep
                + os.path.dirname(_rtools_make) + os.pathsep
                + os.environ.get("PATH", "")
            )

    try:
        from cmdstanpy import CmdStanModel
    except ImportError as exc:
        raise ImportError(
            "CmdStanPy is required for fit_ordinal_probit(). "
            "Install with: pip install cmdstanpy"
        ) from exc

    # ── Resolve Stan model path ──
    if stan_model_path is None:
        stan_model_path = str(PROJECT_ROOT / "models" / "ordinal_utility.stan")

    # Delete cached exe so CmdStanPy always recompiles from source
    exe_path = stan_model_path.replace(".stan", ".exe")
    if os.path.exists(exe_path):
        os.remove(exe_path)
        logger.info(f"fit_ordinal_probit: deleted cached {exe_path}")

    logger.info(f"fit_ordinal_probit: compiling {stan_model_path}")
    model = CmdStanModel(stan_file=stan_model_path)

    # ── Build init values at spec defaults (direct w parameterization) ──
    # No transforms needed — w values are the parameters themselves.
    # Order: [w_inaction_base, w_inaction_no_review, w_inaction_delay,
    #          w_passivity, w_removal, w_remove_ceo_overwhelming, w_review_negative, w_review_balanced,
    #          w_review_post_removal]
    K = stan_data["K"]
    N_scenarios = stan_data["N_scenarios"]
    w_init_raw = [
        3.0,    # w_inaction_base
        2.0,    # w_inaction_no_review
        1.5,    # w_inaction_delay
        0.5,    # w_passivity
        0.5,    # w_remove_ceo_overwhelming (w_raw_6)
        1.3,    # delta_removal (w_removal - w_remove_ceo_overwhelming)
        5.0,    # w_review_negative
        2.5,    # w_review_balanced
        3.0,    # w_review_post_removal
        3.0,    # w_ceo_accountability
    ]
    w_strike_init = float(VOTE_PARAM_DEFAULTS["w_strike"])    # 2.0
    w_overwh_init = float(VOTE_PARAM_DEFAULTS["w_overwhelming"])  # 3.0

    # Compute init mu from init weights to set cutpoints and mu_scale.
    # w_init for mu computation: direct from spec defaults
    w_init = [
        w_init_raw[0],  # w_inaction_base
        w_init_raw[1],  # w_inaction_no_review
        w_init_raw[2],  # w_inaction_delay
        w_init_raw[3],  # w_passivity
        w_init_raw[4] + w_init_raw[5],  # w_removal = w_rceo + delta
        w_init_raw[4],  # w_remove_ceo_overwhelming
        w_init_raw[6],  # w_review_negative
        w_init_raw[7],  # w_review_balanced
        w_init_raw[8],  # w_review_post_removal
        w_init_raw[9],  # w_ceo_accountability
    ]
    phi_arr = np.array(stan_data["phi"])       # (S, K)
    anch_arr = np.array(stan_data["anchored"])  # (S,)
    mu_init = phi_arr @ np.array(w_init) + anch_arr
    # Include vote penalty in mu_init (linear in vote excess)
    vx_s = np.array(stan_data["vote_x_strike"])
    vx_o = np.array(stan_data["vote_x_overwh"])
    hs = np.array(stan_data["has_strike"])
    ho = np.array(stan_data["has_overwh"])
    mu_init -= hs * w_strike_init * vx_s
    mu_init -= ho * w_overwh_init * vx_o

    mu_lo, mu_hi = float(mu_init.min()), float(mu_init.max())
    mu_span = mu_hi - mu_lo

    # mu_scale: normalise so eta/mu_scale spans ~6 probit units [-3, 3].
    # This ensures the probit link has meaningful gradients for all observations,
    # not just the few near the cutpoint boundary.
    mu_scale = max(mu_span / 6.0, 1.0)
    stan_data["mu_scale"] = float(mu_scale)

    # Cutpoints on the normalised scale (4 evenly spaced in [-2, 2]).
    # Stan model uses robust base+gap parameterization:
    #   cutpoints[1] = 3*tanh(base_raw)
    #   cutpoints[k+1] = cutpoints[k] + 0.25 + 2.0*inv_logit(gap_raw[k])
    # Init at [-1.5, -0.5, 0.5, 1.5] → base=-1.5, gaps=[1.0, 1.0, 1.0]
    # Invert: base_raw = atanh(base/3), gap_raw = logit((gap-0.25)/2.0)
    _base_init = -1.5
    _gap_init = 1.0  # uniform gaps
    cutpoint_base_raw_init = float(np.arctanh(_base_init / 3.0))
    # inv_logit(x) = 1/(1+exp(-x)), so logit(p) = log(p/(1-p))
    _p = (_gap_init - 0.25) / 2.0  # = 0.375
    cutpoint_gap_raw_init = float(np.log(_p / (1.0 - _p)))  # logit(0.375) ≈ -0.51
    logger.info(
        f"fit_ordinal_probit: mu_init range [{mu_lo:.2f}, {mu_hi:.2f}], "
        f"mu_scale={mu_scale:.2f}, "
        f"cutpoint_base_raw_init={cutpoint_base_raw_init:.3f}, "
        f"cutpoint_gap_raw_init={cutpoint_gap_raw_init:.3f}"
    )

    init_dict = {
        "w_raw_1": w_init_raw[0],
        "w_raw_2": w_init_raw[1],
        "w_raw_3": w_init_raw[2],
        "w_raw_4": w_init_raw[3],
        "w_raw_6": w_init_raw[4],
        "delta_removal": w_init_raw[5],
        "w_raw_7": w_init_raw[6],
        "w_raw_7b": w_init_raw[7],
        "w_raw_8": w_init_raw[8],
        "w_raw_9": w_init_raw[9],
        "w_strike": w_strike_init,
        "w_overwh": w_overwh_init,
        "cutpoint_base_raw": cutpoint_base_raw_init,
        "cutpoint_gap_raw": [cutpoint_gap_raw_init] * 3,
        "z_scenario": [0.0] * N_scenarios,
        "sigma_scenario": 0.5,
    }

    logger.info(
        f"fit_ordinal_probit: sampling "
        f"(chains={chains}, warmup={iter_warmup}, sampling={iter_sampling}, "
        f"adapt_delta={adapt_delta}, max_treedepth={max_treedepth})"
    )
    fit = model.sample(
        data=stan_data,
        inits=init_dict,
        chains=chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        adapt_delta=adapt_delta,
        max_treedepth=max_treedepth,
        seed=seed,
        show_console=False,
        show_progress=True,
    )

    # ── Extract posterior draws & MCMC diagnostics (with progress bar) ──
    from tqdm import tqdm

    K = stan_data["K"]
    N = stan_data["N"]
    n_draws = chains * iter_sampling

    diag_steps = tqdm(total=5, desc="Post-sampling diagnostics", smoothing=0)

    # Step 1: Extract weight draws
    diag_steps.set_postfix_str("extracting weight draws")
    w_draws = np.column_stack(
        [fit.stan_variable("w")[:, k] for k in range(K)]
    )
    cutpoint_draws = fit.stan_variable("cutpoints")    # (n_draws, 4)
    sigma_scenario_draws = fit.stan_variable("sigma_scenario")  # (n_draws,)
    w_strike_draws = fit.stan_variable("w_strike")       # (n_draws,)
    w_overwh_draws = fit.stan_variable("w_overwh")       # (n_draws,)
    diag_steps.update(1)

    # Step 2: Extract y_rep (large: n_draws × N observations)
    diag_steps.set_postfix_str(f"extracting y_rep ({n_draws}×{N})")
    y_rep = None
    try:
        y_rep = fit.stan_variable("y_rep")             # (n_draws, N)
    except Exception:
        logger.warning("fit_ordinal_probit: y_rep not available in fit object")
    diag_steps.update(1)

    # Step 3: CmdStan diagnose (checks transitions, energy, treedepth)
    diag_steps.set_postfix_str("running CmdStan diagnose")
    try:
        diag_output = fit.diagnose()
        if diag_output:
            for line in diag_output.strip().split("\n"):
                if line.strip():
                    logger.info(f"  diagnose: {line.strip()}")
    except Exception as e:
        logger.warning(f"fit_ordinal_probit: diagnose() failed: {e}")
    diag_steps.update(1)

    # Step 4: Compute R-hat and ESS for model parameters only (skip y_rep)
    diag_steps.set_postfix_str("computing R-hat / ESS")
    _model_param_names = (
        [f"w[{k+1}]" for k in range(K)]
        + ["w_strike", "w_overwh", "delta_removal", "sigma_scenario"]
        + [f"cutpoints[{k+1}]" for k in range(4)]
    )
    max_rhat = float("nan")
    min_ess_bulk = float("nan")
    try:
        # draws() returns (n_chains, n_draws, n_params) when inc_warmup=False
        all_draws = fit.draws()  # (chains, iter_sampling, n_params)
        col_names = fit.column_names
        param_indices = [i for i, c in enumerate(col_names) if c in _model_param_names]

        if param_indices and all_draws.ndim == 3:
            param_draws = all_draws[:, :, param_indices]  # (chains, draws, n_model_params)

            # Split-R-hat: compare chain means to overall mean
            chain_means = np.mean(param_draws, axis=1)     # (chains, n_params)
            chain_vars = np.var(param_draws, axis=1, ddof=1)  # (chains, n_params)
            n_ch = param_draws.shape[0]
            n_dr = param_draws.shape[1]
            B = n_dr * np.var(chain_means, axis=0, ddof=1)  # between-chain variance
            W = np.mean(chain_vars, axis=0)                  # within-chain variance
            var_hat = ((n_dr - 1) / n_dr) * W + (1 / n_dr) * B
            rhat_arr = np.sqrt(var_hat / np.where(W > 1e-12, W, 1e-12))
            max_rhat = float(np.max(rhat_arr))

            # Bulk ESS approximation: n_eff = n_chains * n_draws * var_hat_inv
            ess_arr = n_ch * n_dr * np.where(var_hat > 1e-12, W / var_hat, 1.0)
            min_ess_bulk = float(np.min(ess_arr))
    except Exception as e:
        logger.warning(f"fit_ordinal_probit: manual R-hat/ESS computation failed: {e}")
        logger.info("fit_ordinal_probit: falling back to full stansummary (may be slow)...")
        summary = fit.summary()
        rhat_cols = [c for c in summary.columns if "R_hat" in c or "rhat" in c.lower()]
        ess_cols = [c for c in summary.columns if "ESS_bulk" in c or "ess_bulk" in c.lower()]
        if rhat_cols:
            max_rhat = float(summary[rhat_cols[0]].max())
        if ess_cols:
            min_ess_bulk = float(summary[ess_cols[0]].min())
    diag_steps.update(1)

    # Step 5: Divergence check and summary
    diag_steps.set_postfix_str("finalising")
    n_divergences = int(fit.divergences.sum()) if hasattr(fit, "divergences") else 0

    if max_rhat > 1.01:
        logger.warning(
            f"fit_ordinal_probit: max R-hat = {max_rhat:.4f} > 1.01 — "
            "convergence may be inadequate"
        )
    if n_divergences > 0:
        logger.warning(
            f"fit_ordinal_probit: {n_divergences} divergent transitions — "
            "consider increasing adapt_delta or reparameterising"
        )
    diag_steps.update(1)
    diag_steps.close()

    logger.info(
        f"fit_ordinal_probit: done. "
        f"n_draws={n_draws}, max_rhat={max_rhat:.4f}, "
        f"min_ESS_bulk={min_ess_bulk:.0f}, divergences={n_divergences}"
    )

    return {
        "fit": fit,
        "w": w_draws,
        "w_strike": w_strike_draws,
        "w_overwh": w_overwh_draws,
        "cutpoints": cutpoint_draws,
        "sigma_scenario": sigma_scenario_draws,
        "y_rep": y_rep,
        "n_divergences": n_divergences,
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess_bulk,
        "n_samples": n_draws,
    }


def estimate_parameters_stan(
    phi: np.ndarray,
    anchored: np.ndarray,
    likert_long_df: "pd.DataFrame",
    sa_id_map: dict,
    scenario_id_map: dict,
    vote_data: Optional[dict] = None,
    chains: int = 4,
    iter_warmup: int = 1000,
    iter_sampling: int = 2000,
    adapt_delta: float = 0.99,
    max_treedepth: int = 15,
    seed: int = 42,
) -> StanEstimationResult:
    """Stage 4 (Bayesian): estimate utility weights via ordinal probit Stan model.

    Orchestrates: build Stan data → compile and sample → extract posteriors →
    compute summaries → return StanEstimationResult with full EstimationResult
    interface compatibility.

    Parameters
    ----------
    phi:
        (S, K) basis function matrix from ``compute_phi_matrix()``.
    anchored:
        (S,) anchored utility contributions from ``compute_phi_matrix()``.
    likert_long_df:
        Long-form Likert observations with columns
        ``scenario_id``, ``action``, ``rating``.
    sa_id_map:
        ``(scenario_id, action)`` -> 0-based int, from ``compute_phi_matrix()``.
    scenario_id_map:
        ``scenario_id`` -> 1-based int, from ``compute_phi_matrix()``.
    chains, iter_warmup, iter_sampling, adapt_delta, max_treedepth, seed:
        Passed directly to ``fit_ordinal_probit()``.

    Returns
    -------
    StanEstimationResult
        Posterior summaries plus all EstimationResult-compatible attributes so
        existing downstream code (SEC 8C diagnostics, dashboard rendering,
        _save_parameter_estimates, run_feature_selection) works unchanged.
    """
    logger.info(
        "Stage 4 (Stan): Bayesian ordinal probit weight estimation — "
        f"{len(ESTIMABLE_PARAM_NAMES)} weight params, "
        f"model: models/ordinal_utility.stan"
    )
    logger.info(f"  MCMC: {chains} chains × {iter_sampling} draws "
                f"(warmup {iter_warmup}), adapt_delta={adapt_delta}")

    S, K = phi.shape

    # ── 1. Prepare Stan data ──
    stan_data = prepare_stan_data(
        likert_long_df=likert_long_df,
        phi=phi,
        anchored=anchored,
        sa_id_map=sa_id_map,
        scenario_id_map=scenario_id_map,
        vote_data=vote_data,
    )
    N_scenarios = stan_data["N_scenarios"]

    # ── 2. Fit model ──
    fit_result = fit_ordinal_probit(
        stan_data=stan_data,
        chains=chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        adapt_delta=adapt_delta,
        max_treedepth=max_treedepth,
        seed=seed,
    )

    w_draws = fit_result["w"]           # (n_draws, K)
    w_strike_draws = fit_result["w_strike"]       # (n_draws,)
    w_overwh_draws = fit_result["w_overwh"]       # (n_draws,)
    cutpoint_draws = fit_result["cutpoints"]      # (n_draws, 4)
    sigma_draws = fit_result["sigma_scenario"]    # (n_draws,)
    y_rep = fit_result["y_rep"]                   # (n_draws, N) or None
    n_draws = fit_result["n_samples"]

    # ── 3. Posterior summaries ──
    w_mean = np.mean(w_draws, axis=0)   # (K,)
    w_sd = np.std(w_draws, axis=0)      # (K,)
    w_q025 = np.percentile(w_draws, 2.5, axis=0)
    w_q975 = np.percentile(w_draws, 97.5, axis=0)

    weights_posterior_mean = {
        p: round(float(w_mean[i]), 4)
        for i, p in enumerate(WEIGHT_PARAM_NAMES)
    }
    weights_posterior_sd = {
        p: round(float(w_sd[i]), 4)
        for i, p in enumerate(WEIGHT_PARAM_NAMES)
    }
    weights_posterior_ci = {
        p: [round(float(w_q025[i]), 4), round(float(w_q975[i]), 4)]
        for i, p in enumerate(WEIGHT_PARAM_NAMES)
    }

    # Vote penalty posterior summaries
    for vp_name, vp_draws in [("w_strike", w_strike_draws),
                               ("w_overwhelming", w_overwh_draws)]:
        vp_mean = float(np.mean(vp_draws))
        vp_sd = float(np.std(vp_draws))
        vp_q025 = float(np.percentile(vp_draws, 2.5))
        vp_q975 = float(np.percentile(vp_draws, 97.5))
        weights_posterior_mean[vp_name] = round(vp_mean, 4)
        weights_posterior_sd[vp_name] = round(vp_sd, 4)
        weights_posterior_ci[vp_name] = [round(vp_q025, 4), round(vp_q975, 4)]

    # Posterior mean weights (primary point estimates, for EstimationResult compat)
    opt_weights = weights_posterior_mean.copy()

    # Posterior SD as "hessian_se" slot (closest analogue for dashboard)
    hessian_se = {p: weights_posterior_sd[p]
                  for p in WEIGHT_PARAM_NAMES + VOTE_PARAM_NAMES}
    hessian_se["lambda_rationality"] = 0.0

    # Bootstrap SE slot: 95% CI half-width  (CI / 2*1.96 ~ SE)
    bootstrap_se = {
        p: round(float((w_q975[i] - w_q025[i]) / (2 * 1.96)), 4)
        for i, p in enumerate(WEIGHT_PARAM_NAMES)
    }
    for vp_name in VOTE_PARAM_NAMES:
        ci = weights_posterior_ci[vp_name]
        bootstrap_se[vp_name] = round(float((ci[1] - ci[0]) / (2 * 1.96)), 4)
    bootstrap_se["lambda_rationality"] = 0.0

    # Jackknife SE: leave-one-chain-out SD (lightweight proxy, not full jackknife)
    n_per_chain = iter_sampling
    chain_means = np.array([
        np.mean(w_draws[c * n_per_chain:(c + 1) * n_per_chain], axis=0)
        for c in range(chains)
    ])  # (chains, K)
    loo_sd = np.zeros(K)
    for c in range(chains):
        loo = np.delete(chain_means, c, axis=0)
        loo_sd += (np.mean(loo, axis=0) - w_mean) ** 2
    jackknife_se_arr = np.sqrt(((chains - 1) / chains) * loo_sd)
    jackknife_se = {
        p: round(float(jackknife_se_arr[i]), 4)
        for i, p in enumerate(WEIGHT_PARAM_NAMES)
    }
    # Vote param jackknife (same leave-one-chain-out approach)
    for vp_name, vp_draws in [("w_strike", w_strike_draws),
                               ("w_overwhelming", w_overwh_draws)]:
        vp_chain_means = [
            float(np.mean(vp_draws[c * n_per_chain:(c + 1) * n_per_chain]))
            for c in range(chains)
        ]
        vp_overall = float(np.mean(vp_draws))
        vp_loo = 0.0
        for c in range(chains):
            loo_mean = np.mean([m for j, m in enumerate(vp_chain_means) if j != c])
            vp_loo += (loo_mean - vp_overall) ** 2
        jackknife_se[vp_name] = round(float(np.sqrt(((chains - 1) / chains) * vp_loo)), 4)
    jackknife_se["lambda_rationality"] = 0.0

    # Posterior covariance of w (for dashboard covariance_matrix display)
    # Include vote params in covariance
    all_w_draws = np.column_stack([w_draws, w_strike_draws, w_overwh_draws])
    cov_w_posterior = np.cov(all_w_draws, rowvar=False)  # (K+2, K+2)
    n_cov = K + 2 + 1  # K linear + 2 vote + lambda slot
    cov_matrix = np.zeros((n_cov, n_cov))
    cov_matrix[:K + 2, :K + 2] = cov_w_posterior

    all_param_names = WEIGHT_PARAM_NAMES + VOTE_PARAM_NAMES
    logger.info(
        f"Stage 4 (Stan) complete: "
        f"posterior mean weights = "
        f"{', '.join(f'{p}={opt_weights[p]}' for p in all_param_names)}"
    )
    logger.info(
        f"  MCMC health: max_rhat={fit_result['max_rhat']:.4f}, "
        f"min_ESS_bulk={fit_result['min_ess_bulk']:.0f}, "
        f"divergences={fit_result['n_divergences']}"
    )

    return StanEstimationResult(
        # Stan-specific draws
        w_draws=w_draws,
        cutpoint_draws=cutpoint_draws,
        sigma_scenario_draws=sigma_draws,
        y_rep=y_rep,
        # Posterior summaries
        weights_posterior_mean=weights_posterior_mean,
        weights_posterior_sd=weights_posterior_sd,
        weights_posterior_ci=weights_posterior_ci,
        # MCMC diagnostics
        n_divergences=fit_result["n_divergences"],
        max_rhat=fit_result["max_rhat"],
        min_ess_bulk=fit_result["min_ess_bulk"],
        n_samples=n_draws,
        # EstimationResult-compatible attributes
        weights=opt_weights,
        lambda_rationality=1.0,
        hessian_se=hessian_se,
        bootstrap_se=bootstrap_se,
        covariance_matrix=cov_matrix,
        condition_number=float(np.linalg.cond(cov_w_posterior)),
        ridge_applied=False,
        w10_w11_w14_collapsed=True,
        loss_value=0.0,
        n_scenarios=N_scenarios,
        converged=True,
        estimation_method=(
            {p: "stan_ordinal_probit" for p in WEIGHT_PARAM_NAMES}
            | {p: "stan_ordinal_probit" for p in VOTE_PARAM_NAMES}
        ),
        jackknife_se=jackknife_se,
    )


# Back-compat alias: callers that invoke estimate_parameters() directly
# (e.g. the Stage 4 block in main()) will now go through the Stan pipeline.
# The function accepts the same (phi, anchored, p_llm, action_lists, ...)
# positional signature so the existing call site requires no edits; p_llm and
# action_lists are accepted but ignored (Stan uses individual Likert ratings,
# not scenario-level softmax targets).
def estimate_parameters(
    phi: np.ndarray,
    anchored: np.ndarray,
    p_llm: "np.ndarray | None" = None,
    action_lists: "list | None" = None,
    n_starts: int = 10,
    bootstrap_B: int = 500,
    theta_reg: float = 0.05,
    likert_long_df: "Optional[pd.DataFrame]" = None,
    sa_id_map: Optional[dict] = None,
    scenario_id_map: Optional[dict] = None,
    chains: int = 4,
    iter_warmup: int = 1000,
    iter_sampling: int = 2000,
    adapt_delta: float = 0.99,
    max_treedepth: int = 15,
    seed: int = 42,
) -> StanEstimationResult:
    """Stage 4 entry point — delegates to estimate_parameters_stan().

    Legacy positional arguments (p_llm, action_lists, n_starts, bootstrap_B,
    theta_reg) are accepted for call-site compatibility but are not used by
    the Stan pipeline.

    If ``likert_long_df`` / ``sa_id_map`` / ``scenario_id_map`` are not
    supplied the function raises ValueError with a clear message, since the
    Stan model requires individual Likert ratings rather than aggregated
    scenario-level choice probabilities.
    """
    if likert_long_df is None or sa_id_map is None or scenario_id_map is None:
        raise ValueError(
            "estimate_parameters() now delegates to the Bayesian ordinal "
            "probit Stan pipeline and requires three additional keyword "
            "arguments: likert_long_df (long-form Likert DataFrame), "
            "sa_id_map (dict from compute_phi_matrix), and scenario_id_map "
            "(dict from compute_phi_matrix).  Pass these from the Stage 4 "
            "block in main()."
        )

    return estimate_parameters_stan(
        phi=phi,
        anchored=anchored,
        likert_long_df=likert_long_df,
        sa_id_map=sa_id_map,
        scenario_id_map=scenario_id_map,
        chains=chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        adapt_delta=adapt_delta,
        max_treedepth=max_treedepth,
        seed=seed,
    )


def compute_action_probabilities_from_posterior(
    stan_result: StanEstimationResult,
    scenarios: list[Scenario],
    phi_by_sa: np.ndarray,
    anchored_by_sa: np.ndarray,
    sa_id_map: dict,
    action_lists: list[list[str]],
    scenario_ids: list[str],
) -> dict[str, dict[str, dict]]:
    """Compute action probabilities from posterior weight draws.

    For each posterior draw d:
      1. Compute EU(a) = phi(s,a) . w_d + anchored(s,a) for each action
      2. argmax_a EU(a)
      3. Record winning action

    P(action a is optimal | data) = count(a wins) / n_posterior_draws

    Returns:
        {scenario_id: {action: {"prob_optimal": float, "eu_mean": float,
                                "eu_sd": float, "eu_ci": (lo, hi)}}}
    """
    w_draws = stan_result.w_draws  # (n_posterior_draws, K)
    n_posterior = w_draws.shape[0]

    results = {}
    for sc_idx, sid in enumerate(scenario_ids):
        actions = action_lists[sc_idx]
        n_actions = len(actions)

        # Build EU matrix: (n_posterior_draws, n_actions)
        eu_matrix = np.zeros((n_posterior, n_actions))
        for j, action in enumerate(actions):
            sa_key = (sid, action)
            if sa_key not in sa_id_map:
                continue
            s_idx = sa_id_map[sa_key]
            phi_vec = phi_by_sa[s_idx]  # (K,)
            anch = anchored_by_sa[s_idx]
            eu_matrix[:, j] = w_draws @ phi_vec + anch

        # Argmax per posterior draw
        best_action_idx = np.argmax(eu_matrix, axis=1)

        action_results = {}
        for j, action in enumerate(actions):
            is_best = (best_action_idx == j)
            eu_draws = eu_matrix[:, j]
            action_results[action] = {
                "prob_optimal": round(float(np.mean(is_best)), 4),
                "eu_mean": round(float(np.mean(eu_draws)), 4),
                "eu_sd": round(float(np.std(eu_draws)), 4),
                "eu_ci": (
                    round(float(np.percentile(eu_draws, 2.5)), 4),
                    round(float(np.percentile(eu_draws, 97.5)), 4),
                ),
            }

        results[sid] = action_results

    return results


# ── SEC 8B: Recursive EU tree computation for dashboard ──────────────────────

# Non-Board probabilities: configurable defaults for dashboard tree display.
# These are assumptions about other actors' behaviour, NOT estimated from data.
# D0_ceo: Beta(12.5, 0.5) prior → P(resign) ≈ 0.962
# A2: conditional on d1_action (ASA is more likely to strike when Board does less)
# V: conditional on A2 action (strike recommendation shifts vote distribution upward)
# D4/D4_post: conditional on vote outcome (CEO more likely to leave after bad vote)
# R: Dirichlet(38, 160, 1) → E = (0.191, 0.804, 0.005) for (neg, bal, pos)
TREE_DEFAULT_PROBS = {
    "D0_ceo": {"CEO_resign": 0.962, "CEO_stay": 0.038},
    # A2: 5-path Beta means from background/asa/asa_bayesian_params.md
    # Keyed by (CEO_resigned_early, d1_action) — flattened to composite keys
    "A2": {
        # CEO resigned paths
        "resigned_D0_minimal":        {"A2_no_strike": 0.100, "A2_rec_strike": 0.900},  # Beta(18,2)
        "resigned_D1_review":         {"A2_no_strike": 0.176, "A2_rec_strike": 0.824},  # Beta(14,3)
        # CEO stayed paths
        "stayed_D0_minimal":          {"A2_no_strike": 0.040, "A2_rec_strike": 0.960},  # Beta(24,1)
        "stayed_D1_review":           {"A2_no_strike": 0.118, "A2_rec_strike": 0.882},  # Beta(15,2)
        "stayed_D3_ceo_transition":   {"A2_no_strike": 0.400, "A2_rec_strike": 0.600},  # Beta(9,6)
    },
    "V": {
        "A2_no_strike":  {"no_strike": 0.55, "first_strike": 0.30, "overwhelming": 0.15},
        "A2_rec_strike": {"no_strike": 0.15, "first_strike": 0.40, "overwhelming": 0.45},
    },
    "D4": {
        "no_strike":    {"D4_stay": 0.95, "D4_resign": 0.03, "D4_negotiate_exit": 0.02},
        "first_strike": {"D4_stay": 0.10, "D4_resign": 0.30, "D4_negotiate_exit": 0.60},
        "overwhelming": {"D4_stay": 0.02, "D4_resign": 0.26, "D4_negotiate_exit": 0.72},
    },
    "R": {"negative": 0.191, "balanced": 0.804, "positive": 0.005},
    "D4_post": {
        "no_strike":    {"D4_stay": 0.05, "D4_resign": 0.40, "D4_negotiate_exit": 0.55},
        "first_strike": {"D4_stay": 0.02, "D4_resign": 0.35, "D4_negotiate_exit": 0.63},
        "overwhelming": {"D4_stay": 0.01, "D4_resign": 0.30, "D4_negotiate_exit": 0.69},
    },
}

# Representative vote percentages for computing EU at each vote outcome bucket.
VOTE_REPRESENTATIVES = {
    "no_strike": 0.15,
    "first_strike": 0.35,
    "overwhelming": 0.70,
}

# Representative review outcomes for computing EU.
REVIEW_REPRESENTATIVES = {
    "negative": {"review_car": -0.05, "review_direct_cost": 0.00096},
    "balanced": {"review_car": -0.01, "review_direct_cost": 0.00096},
    "positive": {"review_car": 0.03, "review_direct_cost": 0.00096},
}


def _tree_state_to_decompose_args(ts: dict) -> dict:
    """Convert tree state dict into kwargs for decompose_utility_board()."""
    return {
        "vote_percent": ts.get("vote_percent", 0.0),
        "strike": ts.get("strike", False),
        "overwhelming": ts.get("overwhelming", False),
        "d1_action": ts.get("d1_action", "D0_minimal"),
        "d_rev_action": ts.get("d_rev_action", "Drev_no_action"),
        "d_rev_post_action": ts.get("d_rev_post_action", "Drev_no_action"),
        "CEO_removed": ts.get("CEO_removed", False),
        "CEO_resigned_early": ts.get("CEO_resigned_early", False),
        "review_commissioned": ts.get("review_commissioned", False),
        "review_outcome": ts.get("review_outcome") or "none",
        "review_car": ts.get("review_car", 0.0),
        "review_direct_cost": ts.get("review_direct_cost", 0.00096),
    }


def _tree_state_to_anchored_args(ts: dict) -> dict:
    """Convert tree state dict into kwargs for _compute_anchored_contribution().

    In the recursive tree, R outcomes are already expanded as separate
    branches with their own terminal utility.  The loss-averse CAR
    anchored contribution (W_CAR × review_car) is therefore excluded
    here — it was designed for the flat softmax estimation where R is
    not expanded.  Including it in the recursive tree double-counts the
    review cost and makes "commission review" appear more expensive than
    the estimated weights warrant (e.g. w_review_post_removal can't
    offset the anchored penalty, causing "do nothing" to dominate even
    when the Likert data overwhelmingly favours commissioning).
    """
    return {
        "vote_percent": ts.get("vote_percent", 0.0),
        "strike": ts.get("strike", False),
        "overwhelming": ts.get("overwhelming", False),
        "review_commissioned": False,  # zero out anchored CAR in tree
        "review_car": 0.0,
        "review_direct_cost": 0.0,
    }


def _tree_apply_action(ts: dict, node: str, action: str) -> dict:
    """Apply action at node to tree state, returning a new state dict."""
    ns = dict(ts)
    if node == "D0_ceo":
        if action == "CEO_resign":
            ns["ceo_present"] = False
            ns["CEO_resigned_early"] = True
            ns["CEO_removed"] = True
        # CEO_stay: no state change
    elif node == "D1":
        ns["d1_action"] = action
        if action == "D3_ceo_transition":
            ns["ceo_present"] = False
            ns["CEO_removed"] = True
        if action == "D1_review":
            ns["review_commissioned"] = True
    elif node == "A2":
        ns["a2_action"] = action
    elif node == "V":
        vpct = VOTE_REPRESENTATIVES.get(action, 0.15)
        ns["vote_percent"] = vpct
        ns["strike"] = action in ("first_strike", "overwhelming")
        ns["overwhelming"] = action == "overwhelming"
    elif node in ("D4", "D4_post"):
        if action in ("D4_resign", "D4_negotiate_exit"):
            ns["ceo_present"] = False
            ns["CEO_removed"] = True
    elif node in ("D_rev", "D_rev_post"):
        if node == "D_rev":
            ns["d_rev_action"] = action
        else:
            ns["d_rev_post_action"] = action
        if action == "Drev_commission_review":
            ns["review_commissioned"] = True
        elif action == "Drev_sack_ceo":
            ns["ceo_present"] = False
            ns["CEO_removed"] = True
    elif node == "R":
        ns["review_outcome"] = action  # "negative", "balanced", or "positive"
        rep = REVIEW_REPRESENTATIVES.get(action, REVIEW_REPRESENTATIVES["balanced"])
        ns["review_car"] = rep["review_car"]
        ns["review_direct_cost"] = rep["review_direct_cost"]
    return ns


def _tree_get_vote_key(ts: dict) -> str:
    """Get vote outcome key for D4 probability lookup."""
    if ts.get("overwhelming"):
        return "overwhelming"
    elif ts.get("strike"):
        return "first_strike"
    return "no_strike"


def _tree_feasible_actions(node: str, ts: dict) -> list[str]:
    """Return feasible actions for a node given the current tree state."""
    cp = ts.get("ceo_present", True)
    if node == "D0_ceo":
        return ["CEO_resign", "CEO_stay"]
    elif node == "D1":
        acts = ["D0_minimal", "D1_review"]
        if cp:
            acts.append("D3_ceo_transition")
        return acts
    elif node == "A2":
        return ["A2_no_strike", "A2_rec_strike"]
    elif node == "V":
        return ["no_strike", "first_strike", "overwhelming"]
    elif node in ("D4", "D4_post"):
        if not cp:
            return []  # skip (pass-through)
        return ["D4_stay", "D4_resign", "D4_negotiate_exit"]
    elif node == "D_rev":
        acts = ["Drev_no_action"]
        if not ts.get("review_commissioned", False):
            acts.append("Drev_commission_review")
        if cp:
            acts.append("Drev_sack_ceo")
        return acts
    elif node == "R":
        return ["negative", "balanced", "positive"]
    elif node == "D_rev_post":
        acts = ["Drev_no_action"]
        if cp:
            acts.append("Drev_sack_ceo")
        return acts
    return []


def _tree_node_type(node: str) -> tuple[str, str]:
    """Return (type, owner) for a node name."""
    types = {
        "D0_ceo": ("decision", "CEO"),
        "D1": ("decision", "Board"),
        "A2": ("decision", "ASA"),
        "V": ("chance", "Nature"),
        "D4": ("decision", "CEO"),
        "D_rev": ("decision", "Board"),
        "R": ("chance", "Nature"),
        "D4_post": ("decision", "CEO"),
        "D_rev_post": ("decision", "Board"),
        "Terminal": ("terminal", "Nature"),
    }
    return types.get(node, ("terminal", "Nature"))


def _tree_next_node(node: str) -> str:
    """Return the next node in the game tree sequence."""
    seq = {
        "D0_ceo": "D1",
        "D1": "A2",
        "A2": "V",
        "V": "D4",
        "D4": "D_rev",
        "D_rev": "Terminal",       # or R if commission
        "R": "D4_post",            # or Terminal if positive + no CEO
        "D4_post": "D_rev_post",
        "D_rev_post": "Terminal",
    }
    return seq.get(node, "Terminal")


def _tree_get_probs(node: str, ts: dict, probs: dict) -> dict[str, float]:
    """Get action/outcome probabilities for non-Board nodes."""
    if node == "D0_ceo":
        return probs["D0_ceo"]
    elif node == "A2":
        d1a = ts.get("d1_action", "D0_minimal")
        ceo_resigned = ts.get("CEO_resigned_early", False)
        prefix = "resigned" if ceo_resigned else "stayed"
        composite_key = f"{prefix}_{d1a}"
        return probs["A2"].get(composite_key, probs["A2"]["stayed_D0_minimal"])
    elif node == "V":
        a2a = ts.get("a2_action", "A2_rec_strike")
        return probs["V"].get(a2a, probs["V"]["A2_rec_strike"])
    elif node == "D4":
        vk = _tree_get_vote_key(ts)
        return probs["D4"].get(vk, probs["D4"]["first_strike"])
    elif node == "R":
        return probs["R"]
    elif node == "D4_post":
        vk = _tree_get_vote_key(ts)
        return probs["D4_post"].get(vk, probs["D4_post"]["first_strike"])
    return {}


def _build_tree_node(node_id: str, node_name: str, ts: dict,
                     w_draws: np.ndarray, probs: dict,
                     param_names: list[str],
                     laplacian: bool = True,
                     board_softmax: bool = False) -> tuple[dict, np.ndarray]:
    """Recursively build the expanded tree with probabilities and EUs.

    Returns (tree_dict, eu_draws) where:
      - tree_dict: nested dict for JSON serialization (label, type, owner, eu, edges)
      - eu_draws: shape (n_posterior,) array of per-draw EU values at this node

    Per-draw EU arrays propagate upward:
      Terminal: eu_draws = w_draws @ phi + anchored
      Chance:   eu_draws = sum(p_i * child_eu_draws_i)
      Board:    eu_draws = max over actions (per draw)
              OR softmax-weighted per draw (board_softmax=True)

    Args:
        board_softmax: If True, Board action probabilities are computed via
            per-draw softmax over action EUs (with lambda=1) then averaged,
            and node EU is the softmax-weighted expectation.  This shows
            how parameter uncertainty spreads probability across actions.
            If False (default), uses argmax-count with optional Laplacian.
    """
    ntype, owner = _tree_node_type(node_name)
    n_posterior = w_draws.shape[0]

    if ntype == "terminal":
        phi_dict = decompose_utility_board(**_tree_state_to_decompose_args(ts))
        anchored_val = _compute_anchored_contribution(**_tree_state_to_anchored_args(ts))

        phi_vec = np.array([phi_dict.get(p, 0.0) for p in param_names])
        eu_draws = w_draws @ phi_vec + anchored_val

        # Compute utility components: phi * mean(w) for each parameter
        w_means = np.mean(w_draws, axis=0)
        components = {}
        for k, pname in enumerate(param_names):
            phi_val = phi_dict.get(pname, 0.0)
            if abs(phi_val) > 1e-9:
                components[pname] = round(float(phi_val * w_means[k]), 4)
        if abs(anchored_val) > 1e-9:
            components["anchored"] = round(float(anchored_val), 4)

        tree = {
            "id": node_id,
            "label": "Terminal",
            "type": "terminal",
            "owner": "Nature",
            "eu": round(float(np.mean(eu_draws)), 4),
            "edges": [],
            "components": components,
        }
        return tree, eu_draws

    feasible = _tree_feasible_actions(node_name, ts)

    # Pass-through: D4/D4_post with CEO absent
    if not feasible:
        next_node = _tree_next_node(node_name)
        return _build_tree_node(node_id, next_node, ts, w_draws, probs, param_names,
                                laplacian=laplacian, board_softmax=board_softmax)

    # Helper to route child nodes (handles D_rev→R, R→D4_post, R→Terminal)
    def _build_child(action, new_ts, child_id):
        kw = dict(laplacian=laplacian, board_softmax=board_softmax)
        if node_name == "D_rev" and action == "Drev_commission_review":
            return _build_tree_node(child_id, "R", new_ts, w_draws, probs, param_names,
                                    **kw)
        # D_rev with review already commissioned at D1: route to R for findings
        if node_name == "D_rev" and new_ts.get("review_commissioned", False):
            return _build_tree_node(child_id, "R", new_ts, w_draws, probs, param_names,
                                    **kw)
        if node_name == "R" and action == "negative" and new_ts.get("ceo_present"):
            return _build_tree_node(child_id, "D4_post", new_ts, w_draws, probs, param_names,
                                    **kw)
        if node_name == "R" and (action in ("balanced", "positive") or
                                  (action == "negative" and not new_ts.get("ceo_present"))):
            return _build_tree_node(child_id, "Terminal", new_ts, w_draws, probs, param_names,
                                    **kw)
        return _build_tree_node(child_id, _tree_next_node(node_name), new_ts,
                                w_draws, probs, param_names, **kw)

    if owner == "Board":
        # Build children and collect per-draw EU arrays
        child_trees = {}
        child_eu_arrays = {}
        for action in feasible:
            new_ts = _tree_apply_action(ts, node_name, action)
            child_id = node_id + "__" + action.lower()
            child_tree, child_eu = _build_child(action, new_ts, child_id)
            child_trees[action] = child_tree
            child_eu_arrays[action] = child_eu

        # Stack into (n_posterior, n_actions) matrix for vectorized argmax
        eu_mat = np.column_stack([child_eu_arrays[a] for a in feasible])

        if board_softmax and len(feasible) > 1:
            # Per-draw softmax: P(a|draw_i) = exp(EU_a) / sum_j exp(EU_j)
            # Then average across draws to get display probabilities.
            # This shows how parameter uncertainty spreads probability mass.
            shifted = eu_mat - np.max(eu_mat, axis=1, keepdims=True)  # numerical stability
            exp_eu = np.exp(shifted)
            softmax_probs = exp_eu / exp_eu.sum(axis=1, keepdims=True)  # (n_post, n_act)
            avg_probs = np.mean(softmax_probs, axis=0)  # (n_act,)
            # Laplacian smoothing: blend with uniform to ensure no action has
            # zero probability, even when EU gap makes softmax degenerate.
            K = len(feasible)
            if laplacian:
                alpha = 1.0
                total_pseudo = K * alpha
                total_weight = n_posterior + total_pseudo
                avg_probs = (avg_probs * n_posterior + alpha) / total_weight
            action_probs = {a: float(avg_probs[j]) for j, a in enumerate(feasible)}
            # Node EU per draw = softmax-weighted combination (consistent with probs shown)
            node_eu_draws = np.sum(softmax_probs * eu_mat, axis=1)
        elif len(feasible) > 1:
            # Argmax-count: fraction of draws where each action is argmax.
            # Laplacian smoothing (alpha=1) ensures no action has zero probability.
            best_idx = np.argmax(eu_mat, axis=1)
            n_draws = len(best_idx)
            K = len(feasible)
            alpha = 1.0 if laplacian else 0.0
            action_probs = {}
            for j, a in enumerate(feasible):
                count = float(np.sum(best_idx == j))
                action_probs[a] = (count + alpha) / (n_draws + K * alpha)
            # Node EU per draw = max over actions
            node_eu_draws = np.max(eu_mat, axis=1)
        else:
            action_probs = {feasible[0]: 1.0}
            node_eu_draws = eu_mat[:, 0]

        edges = []
        for j, action in enumerate(feasible):
            edges.append({
                "action": action,
                "prob": round(action_probs[action], 4),
                "eu": round(float(np.mean(eu_mat[:, j])), 4),
                "child": child_trees[action],
            })

        tree = {
            "id": node_id,
            "label": node_name,
            "type": ntype,
            "owner": owner,
            "eu": round(float(np.mean(node_eu_draws)), 4),
            "edges": edges,
        }
        return tree, node_eu_draws
    else:
        # Opponent/chance node: probability-weighted EU.
        #
        # To propagate epistemic uncertainty, non-Board nodes use per-draw
        # Dirichlet-sampled probabilities (not fixed constants).  Without
        # this, the EU difference between Board actions is nearly constant
        # across draws (posterior weights are precisely estimated), producing
        # degenerate 99.8%/0.2% Board probabilities everywhere upstream.
        #
        # Each node's mean probabilities (from TREE_DEFAULT_PROBS) are used
        # as Dirichlet concentration parameters scaled to give the desired
        # mean while allowing per-draw variation.  The concentration sum
        # controls tightness: higher = less variance.
        mean_action_probs = _tree_get_probs(node_name, ts, probs)
        edges = []
        node_eu_draws = np.zeros(n_posterior)

        # Build per-draw Dirichlet probabilities for this node.
        # Use mean probs as Dirichlet alpha (scaled by concentration).
        # R node: use the engine's calibrated Dirichlet(38, 160, 1).
        # Other nodes: use mean probs × concentration_sum.
        prob_values = [mean_action_probs.get(a, 1e-6) for a in feasible]
        prob_sum = sum(prob_values)
        if node_name == "R" and len(feasible) == 3:
            from engine.chance_models import ReviewModel
            alpha = ReviewModel.DIRICHLET_ALPHA  # (38, 160, 1)
        else:
            # Concentration sum controls variance: lower = more spread.
            # Use 20 to give meaningful per-draw variation.
            CONC_SUM = 20.0
            alpha = np.array([p / prob_sum * CONC_SUM for p in prob_values])
            # Floor at 0.5 to avoid degenerate Dirichlet
            alpha = np.maximum(alpha, 0.5)

        # Unique seed per node to avoid correlated draws across the tree
        node_seed = hash(node_id) % (2**31)
        rng_node = np.random.default_rng(node_seed)
        per_draw_probs = rng_node.dirichlet(alpha, size=n_posterior)  # (n_post, K_actions)

        child_eu_list = []
        for j, action in enumerate(feasible):
            p_mean = mean_action_probs.get(action, 0.0)
            new_ts = _tree_apply_action(ts, node_name, action)
            child_id = node_id + "__" + action.lower()
            child_tree, child_eu = _build_child(action, new_ts, child_id)
            edges.append({
                "action": action,
                "prob": round(p_mean, 4),
                "eu": round(float(np.mean(child_eu)), 4),
                "child": child_tree,
            })
            child_eu_list.append(child_eu)
            node_eu_draws += per_draw_probs[:, j] * child_eu

        tree = {
            "id": node_id,
            "label": node_name,
            "type": ntype,
            "owner": owner,
            "eu": round(float(np.mean(node_eu_draws)), 4),
            "edges": edges,
        }
        return tree, node_eu_draws


def compute_recursive_tree(
    est_result: StanEstimationResult,
    probs: dict = None,
    max_draws: int = 500,
    laplacian: bool = True,
    board_softmax: bool = False,
) -> dict:
    """Compute complete game tree with recursive EU using posterior weights.

    For Board decision nodes, action probabilities are computed from either:
    - argmax-count (default): fraction of posterior draws where each action
      is optimal, with optional Laplacian smoothing.
    - softmax (board_softmax=True): per-draw softmax over action EUs,
      averaged across posterior draws.  This shows how parameter uncertainty
      spreads probability across actions, useful for visualisation.

    For other nodes, fixed probabilities from TREE_DEFAULT_PROBS are used.

    Args:
        est_result: StanEstimationResult with w_draws.
        probs: Override non-Board probabilities (default: TREE_DEFAULT_PROBS).
        max_draws: Maximum posterior draws to use (subsampled for speed).
        laplacian: Apply Laplacian smoothing to Board decision probs (default True).
        board_softmax: Use per-draw softmax instead of argmax-count for Board probs.

    Returns a nested dict tree suitable for JSON serialization in the dashboard.
    """
    if probs is None:
        probs = TREE_DEFAULT_PROBS

    w_draws = est_result.w_draws  # (n_posterior, K)
    # Subsample posterior draws for performance
    if w_draws.shape[0] > max_draws:
        rng = np.random.default_rng(42)
        idx = rng.choice(w_draws.shape[0], size=max_draws, replace=False)
        w_draws = w_draws[idx]
    param_names = list(ESTIMABLE_PARAM_NAMES)
    logger.info(f"Computing recursive EU tree ({w_draws.shape[0]} posterior draws, "
                f"{w_draws.shape[1]} weights)...")

    initial_state = {
        "ceo_present": True,
        "CEO_resigned_early": False,
        "CEO_removed": False,
        "review_commissioned": False,
        "review_outcome": "none",
        "vote_percent": 0.0,
        "strike": False,
        "overwhelming": False,
        "d1_action": "D0_minimal",
        "d_rev_action": "Drev_no_action",
        "d_rev_post_action": "Drev_no_action",
        "review_car": 0.0,
        "review_direct_cost": 0.00096,
    }

    tree, _eu_draws = _build_tree_node("root", "D0_ceo", initial_state,
                                        w_draws, probs, param_names,
                                        laplacian=laplacian,
                                        board_softmax=board_softmax)
    logger.info("Recursive EU tree computed successfully.")
    return tree


# ── SEC 8C: Post-estimation diagnostics and feature selection ─────────────────


def run_feature_selection(
    est_result: "EstimationResult",
) -> dict:
    """Post-estimation feature selection using posterior weight draws.

    For the Bayesian ordinal probit pipeline, LR tests are not applicable.
    Instead, relevance is assessed from the posterior distribution of each
    weight:

    - ``Pr(w_k > 0.1)``: probability that the weight exceeds a practically
      meaningful threshold of 0.1.  Parameters where this probability is
      below 0.50 are flagged as low-relevance.
    - ``cv``: posterior SD / |posterior mean| (coefficient of variation).
      Parameters with CV > 50% are flagged as poorly estimated.
    - ``posterior_ci``: 95% credible interval from ``w_draws``.

    Future work: use LOO-CV (leave-one-scenario-out cross-validation) for
    formal Bayesian model comparison and feature selection.

    Returns dict with:
        relevance: {param: {pr_gt_threshold, mean, sd, ci_lo, ci_hi}}
        cv: {param: coefficient_of_variation}
        excluded_params: list of params with Pr(w > 0.1) < 0.50
        poorly_estimated: list of params with CV > 0.50
        dominant_params: list of params with Pr(w > 0.1) >= 0.95
    """
    logger.info("Running post-estimation feature selection (posterior relevance)...")

    from tqdm import tqdm

    w_draws = getattr(est_result, "w_draws", None)
    threshold = 0.1

    # Prior variance for lognormal(log(default), sigma=1.0):
    # Var[w] = (exp(sigma²) - 1) * exp(2*mu + sigma²) where mu=log(default), sigma=1.0
    # = (e - 1) * default² * e ≈ 4.67 * default²
    _e = float(np.e)
    _lognormal_var_factor = (_e - 1.0) * _e  # ≈ 4.6708

    results = {
        "relevance": {},
        "cv": {},
        "prior_shrinkage": {},
        "excluded_params": [],
        "poorly_estimated": [],
        "dominant_params": [],
    }

    all_params = list(enumerate(ESTIMABLE_PARAM_NAMES))
    pbar = tqdm(total=len(ESTIMABLE_PARAM_NAMES) + len(VOTE_PARAM_NAMES),
                desc="SEC 8C: Feature selection", smoothing=0)

    for k, param in all_params:
        w_mean = est_result.weights.get(param, 0.0)
        w_sd = est_result.hessian_se.get(param, 0.0)

        if w_draws is not None and w_draws.ndim == 2 and k < w_draws.shape[1]:
            draws_k = w_draws[:, k]
            pr_gt = float(np.mean(draws_k > threshold))
            ci_lo = float(np.percentile(draws_k, 2.5))
            ci_hi = float(np.percentile(draws_k, 97.5))
            w_sd_post = float(np.std(draws_k))
        else:
            # Fall back to approximate normal if draws unavailable
            from scipy.stats import norm
            pr_gt = float(1.0 - norm.cdf(threshold, loc=w_mean, scale=max(w_sd, 1e-9)))
            ci_lo = round(w_mean - 1.96 * w_sd, 4)
            ci_hi = round(w_mean + 1.96 * w_sd, 4)
            w_sd_post = w_sd

        cv = abs(w_sd_post / w_mean) if abs(w_mean) > 1e-12 else float("nan")

        # Prior shrinkage: 1 - posterior_var / prior_var
        # Shrinkage near 1.0 = data dominates (uninformative prior)
        # Shrinkage near 0.0 = prior dominates (informative prior)
        spec_default = SPEC_DEFAULTS.get(param, 1.0)
        prior_var = _lognormal_var_factor * spec_default ** 2
        posterior_var = w_sd_post ** 2
        shrinkage = max(0.0, 1.0 - posterior_var / prior_var) if prior_var > 1e-12 else 0.0
        results["prior_shrinkage"][param] = round(shrinkage, 3)

        results["relevance"][param] = {
            "pr_gt_threshold": round(pr_gt, 4),
            "threshold": threshold,
            "mean": round(w_mean, 4),
            "sd": round(w_sd_post, 4),
            "ci_lo": round(ci_lo, 4),
            "ci_hi": round(ci_hi, 4),
        }
        results["cv"][param] = round(cv, 4) if not np.isnan(cv) else None

        shrinkage_label = "data-driven" if shrinkage >= 0.5 else "prior-sensitive"
        cv_str = f"{cv:.2%}" if not np.isnan(cv) else "N/A"
        logger.info(
            f"  {param}: Pr(w>{threshold})={pr_gt:.3f}, "
            f"mean={w_mean:.4f}, sd={w_sd_post:.4f}, CV={cv_str}, "
            f"shrinkage={shrinkage:.3f} ({shrinkage_label})"
        )
        pbar.update(1)

    # Vote penalty params: use posterior summary (normal approx) for relevance
    for param in VOTE_PARAM_NAMES:
        vp_mean = est_result.weights_posterior_mean.get(param, est_result.weights.get(param, 0.0))
        vp_sd = est_result.weights_posterior_sd.get(param, est_result.hessian_se.get(param, 0.0))
        vp_ci = est_result.weights_posterior_ci.get(param)
        from scipy.stats import norm as _norm
        pr_gt = float(1.0 - _norm.cdf(threshold, loc=vp_mean, scale=max(vp_sd, 1e-9)))
        ci_lo = vp_ci[0] if vp_ci else round(vp_mean - 1.96 * vp_sd, 4)
        ci_hi = vp_ci[1] if vp_ci else round(vp_mean + 1.96 * vp_sd, 4)
        cv = abs(vp_sd / vp_mean) if abs(vp_mean) > 1e-12 else float("nan")

        # Prior shrinkage for vote params (also lognormal priors)
        spec_default = SPEC_DEFAULTS.get(param, 1.0)
        prior_var = _lognormal_var_factor * spec_default ** 2
        posterior_var = vp_sd ** 2
        shrinkage = max(0.0, 1.0 - posterior_var / prior_var) if prior_var > 1e-12 else 0.0
        results["prior_shrinkage"][param] = round(shrinkage, 3)

        results["relevance"][param] = {
            "pr_gt_threshold": round(pr_gt, 4),
            "threshold": threshold,
            "mean": round(vp_mean, 4),
            "sd": round(vp_sd, 4),
            "ci_lo": round(ci_lo, 4),
            "ci_hi": round(ci_hi, 4),
        }
        results["cv"][param] = round(cv, 4) if not np.isnan(cv) else None

        shrinkage_label = "data-driven" if shrinkage >= 0.5 else "prior-sensitive"
        logger.info(
            f"  {param}: Pr(w>{threshold})={pr_gt:.3f}, "
            f"mean={vp_mean:.4f}, sd={vp_sd:.4f}, "
            f"CV={cv:.2%}, shrinkage={shrinkage:.3f} ({shrinkage_label})" if not np.isnan(cv) else
            f"  {param}: Pr(w>{threshold})={pr_gt:.3f}, mean={vp_mean:.4f}, sd={vp_sd:.4f}, "
            f"CV=N/A, shrinkage={shrinkage:.3f} ({shrinkage_label})"
        )
        pbar.update(1)

    pbar.close()

    # All estimated params (linear + vote) for flagging
    all_estimated = ESTIMABLE_PARAM_NAMES + VOTE_PARAM_NAMES

    # Flag low-relevance parameters (Pr(w > threshold) < 0.50)
    for param in all_estimated:
        pr = results["relevance"][param]["pr_gt_threshold"]
        if pr < 0.50:
            results["excluded_params"].append(param)
            logger.warning(
                f"  Feature selection: {param} flagged as low-relevance "
                f"(Pr(w>{threshold})={pr:.3f})"
            )

    # Flag poorly estimated params (CV > 50%)
    results["poorly_estimated"] = [
        p for p in all_estimated
        if results["cv"].get(p) is not None and results["cv"][p] > 0.50
    ]
    if results["poorly_estimated"]:
        logger.warning(f"  Poorly estimated (CV > 50%): {results['poorly_estimated']}")

    # Flag dominant params (Pr(w > threshold) >= 0.95)
    results["dominant_params"] = [
        p for p in all_estimated
        if results["relevance"][p]["pr_gt_threshold"] >= 0.95
    ]
    if results["dominant_params"]:
        logger.info(f"  Dominant parameters (Pr > 0.95): {results['dominant_params']}")

    return results


# ── SEC 9: Stage 5 — Behavioural diagnostics ─────────────────────────────────

def _diagnose_loss_aversion(
    scenarios: list[Scenario],
    likert_summary_df: pd.DataFrame,
) -> dict:
    """Test 8.1: Loss aversion via matched gain/loss CAR pairs.

    Compares mean Likert scores for the Drev_commission_review action between
    matched positive-CAR and negative-CAR scenario pairs.  A higher mean score
    in the negative-CAR scenario indicates loss-aversion (losses loom larger).
    """
    from scipy import stats

    la_scenarios = [s for s in scenarios if s.target_parameter == "loss_aversion"]
    if len(la_scenarios) < 4:
        return {"test": "loss_aversion", "decision": "insufficient_data", "n_pairs": 0}

    # Group by CAR magnitude
    pairs = {}
    for s in la_scenarios:
        car = s.state_vector.get("car_outcome", 0)
        mag = abs(car)
        if mag not in pairs:
            pairs[mag] = {}
        sign = "pos" if car > 0 else "neg"
        pairs[mag][sign] = s.scenario_id

    ratios = []
    for mag, pair in pairs.items():
        if "pos" not in pair or "neg" not in pair or mag == 0:
            continue
        # Retrieve mean Likert score for the review action in each scenario
        row_pos = likert_summary_df[
            (likert_summary_df["scenario_id"] == pair["pos"]) &
            (likert_summary_df["action"] == "Drev_commission_review")
        ]
        row_neg = likert_summary_df[
            (likert_summary_df["scenario_id"] == pair["neg"]) &
            (likert_summary_df["action"] == "Drev_commission_review")
        ]
        if row_pos.empty or row_neg.empty:
            continue
        score_pos = float(row_pos.iloc[0]["mean_score"])
        score_neg = float(row_neg.iloc[0]["mean_score"])
        # Ratio: negative-frame sensitivity / positive-frame sensitivity
        # relative to the midpoint score (3.0 on a 1-5 scale)
        midpoint = 3.0
        delta_pos = abs(score_pos - midpoint) + 1e-8
        delta_neg = abs(score_neg - midpoint) + 1e-8
        ratio = delta_neg / delta_pos
        ratios.append(ratio)

    if not ratios:
        return {"test": "loss_aversion", "decision": "insufficient_data", "n_pairs": 0}

    mean_ratio = float(np.mean(ratios))
    t_stat_vs_1, p_val_vs_1 = stats.ttest_1samp(ratios, 1.0) if len(ratios) > 1 else (0, 1)
    t_stat_vs_kt, p_val_vs_kt = stats.ttest_1samp(ratios, LAMBDA_LA_DEFAULT) if len(ratios) > 1 else (0, 1)

    decision = "confirmed" if p_val_vs_1 < 0.05 else "null"

    return {
        "test": "loss_aversion",
        "decision": decision,
        "n_pairs": len(ratios),
        "mean_sensitivity_ratio": round(mean_ratio, 3),
        "t_stat_vs_1": round(float(t_stat_vs_1), 3),
        "p_value_vs_1": round(float(p_val_vs_1), 4),
        "t_stat_vs_kt": round(float(t_stat_vs_kt), 3),
        "p_value_vs_kt": round(float(p_val_vs_kt), 4),
        "ratios": [round(r, 3) for r in ratios],
    }


def _diagnose_nonlinearity(
    scenarios: list[Scenario],
    likert_summary_df: pd.DataFrame,
    est_result: EstimationResult,
) -> dict:
    """Test 8.2: Non-linearity in vote penalty and diminishing marginal disutility.

    Uses mean Likert score for D1_review as a proxy for penalty sensitivity
    across scenarios with varying vote levels.  Compares AIC of four functional
    forms (quadratic, linear, cubic, log-linear) via OLS regression of mean
    score against vote-excess (V - 0.25).
    """
    from scipy import stats

    # Vote penalty functional form comparison
    vote_scenarios = [s for s in scenarios if s.state_vector.get("vote_outcome_V", 0) > 0.25]
    vote_points = []
    for s in vote_scenarios:
        v = s.state_vector.get("vote_outcome_V", 0)
        if v <= 0.25:
            continue
        row = likert_summary_df[
            (likert_summary_df["scenario_id"] == s.scenario_id) &
            (likert_summary_df["action"] == "D1_review")
        ]
        if row.empty:
            continue
        mean_score = float(row.iloc[0]["mean_score"])
        vote_points.append((v, mean_score))

    aic_results = {}
    if len(vote_points) >= 3:
        vs = np.array([vp[0] for vp in vote_points])
        ps = np.array([vp[1] for vp in vote_points])
        xs = vs - 0.25

        forms = {
            "quadratic": xs ** 2,
            "linear": xs,
            "cubic": xs ** 3,
            "log_linear": np.log(vs / 0.25),
        }
        for name, x_form in forms.items():
            try:
                slope, intercept, r, p, se = stats.linregress(x_form, ps)
                ss_res = np.sum((ps - (slope * x_form + intercept)) ** 2)
                n = len(ps)
                aic = n * np.log(ss_res / n + 1e-12) + 2 * 2
                aic_results[name] = round(float(aic), 2)
            except Exception:
                aic_results[name] = float("inf")

    # Diminishing marginal disutility: mean score of most severe action vs
    # number of active penalties per Tier 2 scenario
    t2_scenarios = [s for s in scenarios if s.tier == 2]
    penalty_counts = []
    for s in t2_scenarios:
        sv = s.state_vector
        n_active = sum([
            sv.get("strike", False),
            sv.get("overwhelming", False),
            sv.get("review_outcome") in ("negative", "balanced"),
            not sv.get("ceo_present_at_end", True),
            sv.get("vote_outcome_V", 0) > 0.25,
        ])
        # Use the maximum mean Likert score across severe actions as a proxy
        severe_actions = ["D3_ceo_transition", "Drev_sack_ceo", "D1_review"]
        scores = []
        for action in severe_actions:
            row = likert_summary_df[
                (likert_summary_df["scenario_id"] == s.scenario_id) &
                (likert_summary_df["action"] == action)
            ]
            if not row.empty:
                scores.append(float(row.iloc[0]["mean_score"]))
        if scores:
            penalty_counts.append((n_active, max(scores)))

    dmd_result = {}
    if len(penalty_counts) >= 4:
        counts = np.array([pc[0] for pc in penalty_counts])
        severities = np.array([pc[1] for pc in penalty_counts])
        try:
            slope, intercept, r, p, se = stats.linregress(counts, severities)
            dmd_result = {
                "slope": round(float(slope), 4),
                "p_value": round(float(p), 4),
                "r_squared": round(float(r**2), 4),
                "decision": "diminishing" if slope < 0 and p < 0.05 else "not_detected",
            }
        except Exception:
            dmd_result = {"decision": "computation_failed"}

    best_form = min(aic_results, key=aic_results.get) if aic_results else "quadratic"

    return {
        "test": "nonlinearity",
        "vote_penalty_aic": aic_results,
        "best_vote_form": best_form,
        "diminishing_marginal": dmd_result,
        "n_vote_points": len(vote_points),
    }


def _diagnose_optimism_bias(
    scenarios: list[Scenario],
    likert_summary_df: pd.DataFrame,
) -> dict:
    """Test 8.3: Optimism bias — explicit vs implicit adverse probability.

    Compares mean Likert scores for Drev_commission_review between scenarios
    where the adverse outcome probability is stated explicitly vs left implicit.
    Optimism bias is confirmed if explicit framing produces higher mean scores
    (i.e., the board responds more strongly when forced to confront numbers).
    """
    opt_scenarios = [s for s in scenarios if s.target_parameter == "optimism_bias"]
    explicit = []
    implicit = []
    for s in opt_scenarios:
        row = likert_summary_df[
            (likert_summary_df["scenario_id"] == s.scenario_id) &
            (likert_summary_df["action"] == "Drev_commission_review")
        ]
        if row.empty:
            continue
        mean_score = float(row.iloc[0]["mean_score"])
        if s.state_vector.get("explicit_adverse_prob"):
            explicit.append(mean_score)
        else:
            implicit.append(mean_score)

    if not explicit or not implicit:
        return {"test": "optimism_bias", "decision": "insufficient_data"}

    from scipy import stats
    t_stat, p_val = stats.ttest_ind(explicit, implicit) if len(explicit) > 1 and len(implicit) > 1 else (0, 1)
    effect_size = float(np.mean(explicit) - np.mean(implicit))

    return {
        "test": "optimism_bias",
        "mean_score_explicit": round(float(np.mean(explicit)), 4),
        "mean_score_implicit": round(float(np.mean(implicit)), 4),
        "effect_size": round(effect_size, 4),
        "t_stat": round(float(t_stat), 3),
        "p_value": round(float(p_val), 4),
        "decision": "confirmed" if p_val < 0.05 and effect_size > 0 else "null",
    }


def _diagnose_self_assessment(
    scenarios: list[Scenario],
    likert_summary_df: pd.DataFrame,
) -> dict:
    """Test 8.4: Self-assessment bias — board vs external review origin.

    Compares mean Likert scores for Drev_sack_ceo between scenarios where
    the review was board-initiated vs externally mandated.  Self-assessment
    bias is confirmed if board-initiated reviews produce lower sack scores
    (board is lenient on its own process / CEO).
    """
    sa_scenarios = [s for s in scenarios if s.target_parameter == "self_assessment_bias"]
    board_init = []
    ext_mandated = []
    for s in sa_scenarios:
        row = likert_summary_df[
            (likert_summary_df["scenario_id"] == s.scenario_id) &
            (likert_summary_df["action"] == "Drev_sack_ceo")
        ]
        if row.empty:
            continue
        mean_score = float(row.iloc[0]["mean_score"])
        origin = s.state_vector.get("review_origin", "board_initiated")
        if origin == "board_initiated":
            board_init.append(mean_score)
        else:
            ext_mandated.append(mean_score)

    if not board_init or not ext_mandated:
        return {"test": "self_assessment_bias", "decision": "insufficient_data"}

    from scipy import stats
    t_stat, p_val = stats.ttest_ind(board_init, ext_mandated) if len(board_init) > 1 and len(ext_mandated) > 1 else (0, 1)

    return {
        "test": "self_assessment_bias",
        "mean_score_sack_board_initiated": round(float(np.mean(board_init)), 4),
        "mean_score_sack_externally_mandated": round(float(np.mean(ext_mandated)), 4),
        "t_stat": round(float(t_stat), 3),
        "p_value": round(float(p_val), 4),
        "decision": "confirmed" if p_val < 0.05 and np.mean(board_init) < np.mean(ext_mandated) else "null",
    }


def _diagnose_ikea_effect(
    scenarios: list[Scenario],
    likert_summary_df: pd.DataFrame,
) -> dict:
    """Test 8.5: Ikea effect — CEO appointment and review ownership.

    Compares mean Likert scores for Drev_sack_ceo between scenarios where
    the CEO was appointed by the current board vs inherited from a prior board.
    The Ikea effect predicts lower sack scores when the board appointed the CEO
    (greater ownership reduces willingness to remove).
    """
    ikea_scenarios = [s for s in scenarios if s.target_parameter == "ikea_effect"]
    appointed = []
    inherited = []
    for s in ikea_scenarios:
        row = likert_summary_df[
            (likert_summary_df["scenario_id"] == s.scenario_id) &
            (likert_summary_df["action"] == "Drev_sack_ceo")
        ]
        if row.empty:
            continue
        mean_score = float(row.iloc[0]["mean_score"])
        appt = s.state_vector.get("ceo_appointment", "appointed_by_current_board")
        if appt == "appointed_by_current_board":
            appointed.append(mean_score)
        else:
            inherited.append(mean_score)

    if not appointed or not inherited:
        return {"test": "ikea_effect", "decision": "insufficient_data"}

    from scipy import stats
    t_stat, p_val = stats.ttest_ind(appointed, inherited) if len(appointed) > 1 and len(inherited) > 1 else (0, 1)

    return {
        "test": "ikea_effect",
        "mean_score_sack_appointed": round(float(np.mean(appointed)), 4),
        "mean_score_sack_inherited": round(float(np.mean(inherited)), 4),
        "t_stat": round(float(t_stat), 3),
        "p_value": round(float(p_val), 4),
        "decision": "confirmed" if p_val < 0.05 and np.mean(appointed) < np.mean(inherited) else "null",
    }


def _diagnose_action_order_effects(
    elicitation_path: Path,
) -> dict:
    """Test 8.6: Action presentation order effects on Likert scores.

    Tests whether the position of each action in the randomised presentation
    order (action_order field in elicitation_results.csv) affects its mean
    Likert score.  A significant negative slope indicates primacy bias
    (earlier-presented actions rated higher); positive slope indicates
    recency bias.

    Factor ratings no longer exist in the Likert pipeline; this test
    replaces the old _diagnose_factor_order_effects() function.
    """
    from scipy import stats

    df = pd.read_csv(elicitation_path, encoding="utf-8")
    df_ok = df[df["parse_status"].isin(["success", "repaired"])].copy()

    if df_ok.empty:
        return {"test": "action_order_effects", "decision": "insufficient_data"}

    # Build long table: (action, position, score)
    long_records = []
    for _, row in df_ok.iterrows():
        try:
            scores_dict = json.loads(row["action_scores"])
            action_order = json.loads(row["action_order"]) if row.get("action_order") else []
        except Exception:
            continue
        for action, score in scores_dict.items():
            position = action_order.index(action) + 1 if action in action_order else None
            if position is not None:
                long_records.append({
                    "action": action,
                    "position": position,
                    "score": int(score),
                })

    if not long_records:
        return {"test": "action_order_effects", "decision": "insufficient_data"}

    long_df = pd.DataFrame(long_records)

    # Aggregate per action: regress score ~ position
    results_per_action = {}
    for action, grp in long_df.groupby("action"):
        if len(grp) < 10:
            continue
        try:
            slope, intercept, r, p, se = stats.linregress(
                grp["position"].values, grp["score"].values
            )
            results_per_action[action] = {
                "slope": round(float(slope), 4),
                "p_value": round(float(p), 4),
                "n": int(len(grp)),
                "effect": (
                    "primacy" if slope < 0 and p < 0.05 else
                    "recency" if slope > 0 and p < 0.05 else "none"
                ),
            }
        except Exception:
            continue

    # Overall test: pool all observations
    overall_result = {}
    if len(long_df) >= 20:
        try:
            slope, intercept, r, p, se = stats.linregress(
                long_df["position"].values, long_df["score"].values
            )
            overall_result = {
                "slope": round(float(slope), 4),
                "p_value": round(float(p), 4),
                "r_squared": round(float(r**2), 4),
            }
        except Exception:
            pass

    any_effect = any(
        v.get("effect", "none") != "none"
        for v in results_per_action.values()
    )

    return {
        "test": "action_order_effects",
        "per_action": results_per_action,
        "overall": overall_result,
        "any_order_effect_detected": any_effect,
        "decision": "detected" if any_effect else "null",
    }


def run_diagnostics(
    scenarios: list[Scenario],
    likert_summary_df: pd.DataFrame,
    est_result: EstimationResult,
    elicitation_path: Path,
    output_path: Path,
) -> dict:
    """Stage 5: Run all behavioural diagnostics.

    Parameters
    ----------
    scenarios:
        Full scenario list (all tiers).
    likert_summary_df:
        One row per (scenario_id, action) with columns mean_score and sd_score.
        Produced by ``preprocess_likert_data()`` Stage 3.
    est_result:
        StanEstimationResult with .w_draws for posterior-based diagnostics.
    elicitation_path:
        Path to elicitation_results.csv (used for action order effects test).
    output_path:
        Destination CSV for diagnostic summary rows.
    """
    logger.info("Stage 5: Running behavioural diagnostics...")

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Precision loss", category=RuntimeWarning)
        diagnostics = {
            "loss_aversion": _diagnose_loss_aversion(scenarios, likert_summary_df),
            "nonlinearity": _diagnose_nonlinearity(scenarios, likert_summary_df, est_result),
            "optimism_bias": _diagnose_optimism_bias(scenarios, likert_summary_df),
            "self_assessment_bias": _diagnose_self_assessment(scenarios, likert_summary_df),
            "ikea_effect": _diagnose_ikea_effect(scenarios, likert_summary_df),
            "action_order_effects": _diagnose_action_order_effects(elicitation_path),
        }

    # Save to CSV
    rows = []
    for name, result in diagnostics.items():
        rows.append({
            "diagnostic": name,
            "decision": result.get("decision", ""),
            "p_value": result.get("p_value", result.get("p_value_vs_1", "")),
            "effect_size": result.get("effect_size", result.get("mean_sensitivity_ratio", "")),
            "details": json.dumps(result, ensure_ascii=True),
        })
    diag_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diag_df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Diagnostics saved to {output_path}")

    return diagnostics


# ── SEC 10: Stage 6 — Validation ─────────────────────────────────────────────

def _compute_scenario_fit(
    phi: np.ndarray,
    anchored: np.ndarray,
    sa_id_map: dict,
    est_result: EstimationResult,
    scenario_ids: list[str],
    action_lists: list[list[str]],
    likert_summary_df: Optional[pd.DataFrame] = None,
) -> list[dict]:
    """Compute per-scenario posterior predictive fit.

    For the Bayesian ordinal probit pipeline, fit is assessed by comparing
    observed mean Likert scores to posterior predicted mean scores.  For each
    scenario and action the posterior predicted score is computed as:

        E[score | w] = sum_k k * Pr(score = k | EU, cutpoints)

    averaged over posterior draws.  When ``likert_summary_df`` is not
    provided (e.g., during validation-only runs), the function falls back to
    reporting posterior mean EU values and a normalised EU residual relative to
    the across-action mean.

    The ``kl_divergence`` field is retained for dashboard compatibility but is
    now a normalised root-mean-square residual (RMSE) of mean Likert scores,
    which plays the same diagnostic role.
    """
    w_draws = getattr(est_result, "w_draws", None)

    n_sc = len(scenario_ids)
    fit_rows = []

    for i in range(n_sc):
        sid = scenario_ids[i]
        actions = action_lists[i]

        # Posterior mean EU per action (averaged over weight draws)
        eu_mean_per_action = {}
        for action in actions:
            sa_idx = sa_id_map.get((sid, action))
            if sa_idx is None:
                continue
            phi_vec = phi[sa_idx]
            anch_val = anchored[sa_idx]
            if w_draws is not None and w_draws.ndim == 2:
                eu_draws = w_draws @ phi_vec + anch_val  # (D,)
                eu_mean_per_action[action] = float(np.mean(eu_draws))
            else:
                weights = np.array([est_result.weights.get(p, 0.0) for p in WEIGHT_PARAM_NAMES])
                eu_mean_per_action[action] = float(phi_vec @ weights + anch_val)

        # Residuals: observed mean Likert score minus posterior predicted EU
        # (on comparable scales; residual is in score-units if normalised)
        residuals = {}
        sq_residuals = []

        for action in actions:
            eu_pred = eu_mean_per_action.get(action, 0.0)

            if likert_summary_df is not None:
                obs_row = likert_summary_df[
                    (likert_summary_df["scenario_id"] == sid) &
                    (likert_summary_df["action"] == action)
                ]
                if not obs_row.empty:
                    obs_score = float(obs_row.iloc[0]["mean_score"])
                    # Normalise EU to [1,5] scale for comparison: shift+scale
                    # using the across-action EU range for this scenario
                    eu_vals = list(eu_mean_per_action.values())
                    eu_min, eu_max = min(eu_vals), max(eu_vals)
                    eu_range = eu_max - eu_min if (eu_max - eu_min) > 1e-9 else 1.0
                    eu_scaled = 1.0 + 4.0 * (eu_pred - eu_min) / eu_range
                    resid = obs_score - eu_scaled
                    residuals[action] = round(float(resid), 4)
                    sq_residuals.append(resid ** 2)
            else:
                residuals[action] = round(eu_pred, 4)

        rmse = float(np.sqrt(np.mean(sq_residuals))) if sq_residuals else 0.0

        fit_rows.append({
            "scenario_id": sid,
            "kl_divergence": round(rmse, 6),   # retained for dashboard compat
            "residuals": json.dumps(residuals, ensure_ascii=True),
            "model_probs": json.dumps(
                {a: round(eu_mean_per_action.get(a, 0.0), 4)
                 for a in actions if a in eu_mean_per_action},
                ensure_ascii=True,
            ),
        })

    return fit_rows


def _validate_historical(
    scenarios: list[Scenario],
    estimation_df: pd.DataFrame,
    est_result: EstimationResult,
    phi: np.ndarray,
    anchored: np.ndarray,
    scenario_ids: list[str],
    action_lists: list[list[str]],
) -> dict:
    """Validate against Tier 4 historical scenario (Qantas AGM Nov 2023).

    Uses posterior action probabilities (``prob_optimal`` from
    ``compute_action_probabilities_from_posterior``) rather than softmax
    of point-estimate weights.  Falls back to posterior-mean EU ranking
    when full posterior draws are unavailable.
    """
    t4 = [s for s in scenarios if s.tier == 4]
    if not t4:
        return {"available": False}

    s = t4[0]
    w_draws = getattr(est_result, "w_draws", None)

    # Compute per-action EU using posterior mean weights
    weights = np.array([est_result.weights.get(p, 0.0) for p in WEIGHT_PARAM_NAMES])

    eu_per_action = {}
    prob_optimal_per_action = {}

    for action in s.feasible_actions:
        args = _scenario_to_outcome_args(s.state_vector, action)
        phi_k = decompose_utility_board(**args)
        phi_vec = np.array([phi_k.get(p, 0.0) for p in WEIGHT_PARAM_NAMES])
        anch = _compute_anchored_contribution(
            args["vote_percent"], args["strike"], args["overwhelming"],
            args["review_commissioned"], args["review_car"], args["review_direct_cost"],
        )
        eu_per_action[action] = float(phi_vec @ weights + anch)

        # Posterior probability that this action has the highest EU
        if w_draws is not None and w_draws.ndim == 2:
            eu_draws_a = w_draws @ phi_vec + anch  # (D,)
            prob_optimal_per_action[action] = eu_draws_a

    # Compute Pr(action = argmax EU) across posterior draws
    if prob_optimal_per_action:
        all_actions = list(prob_optimal_per_action.keys())
        eu_matrix = np.stack(
            [prob_optimal_per_action[a] for a in all_actions], axis=1
        )  # (D, n_actions)
        best_idx = np.argmax(eu_matrix, axis=1)  # (D,)
        predicted_probs = {
            a: round(float(np.mean(best_idx == j)), 4)
            for j, a in enumerate(all_actions)
        }
    else:
        # Fall back: proportional to EU (softmax at lambda=1 would be one option;
        # here we use rank-probability to avoid introducing a tuning parameter)
        ranked_eu = sorted(eu_per_action.items(), key=lambda x: -x[1])
        n = len(ranked_eu)
        weights_rank = {a: (n - rank) / sum(range(n + 1)) for rank, (a, _) in enumerate(ranked_eu)}
        predicted_probs = {a: round(float(w), 4) for a, w in weights_rank.items()}

    ranked = sorted(predicted_probs.items(), key=lambda x: -x[1])

    return {
        "available": True,
        "scenario_id": s.scenario_id,
        "predicted_probs": predicted_probs,
        "posterior_mean_eu": {a: round(v, 4) for a, v in eu_per_action.items()},
        "rank_of_D1_review": (
            [i + 1 for i, (a, _) in enumerate(ranked) if a == "D1_review"][0]
            if "D1_review" in predicted_probs else None
        ),
        "top_action": ranked[0][0],
        "top_prob": ranked[0][1],
    }


def run_validation(
    scenarios: list[Scenario],
    estimation_df: pd.DataFrame,
    est_result: EstimationResult,
    phi: np.ndarray,
    anchored: np.ndarray,
    sa_id_map: dict,
    scenario_ids: list[str],
    action_lists: list[list[str]],
    output_dir: Path,
    likert_summary_df: Optional[pd.DataFrame] = None,
) -> dict:
    """Stage 6: Run validation checks.

    Parameters
    ----------
    scenarios, estimation_df, est_result, phi, anchored:
        Standard pipeline arguments.
    sa_id_map:
        Mapping (scenario_id, action) -> row index in phi/anchored.
    scenario_ids, action_lists, output_dir:
        Scenario structure and output path.
    likert_summary_df:
        Optional (scenario_id, action) summary with ``mean_score`` column.
        When provided, per-scenario fit uses observed vs predicted Likert
        scores (RMSE).  When omitted, fit is reported as posterior mean EU.
    """
    logger.info("Stage 6: Running validation...")

    # Within-sample fit (posterior predictive check)
    fit_rows = _compute_scenario_fit(
        phi, anchored, sa_id_map, est_result,
        scenario_ids, action_lists, likert_summary_df,
    )
    kl_values = [r["kl_divergence"] for r in fit_rows]
    mean_kl = float(np.mean(kl_values)) if kl_values else float("nan")

    # Save scenario fit
    fit_df = pd.DataFrame(fit_rows)
    fit_df.to_csv(output_dir / "scenario_fit.csv", index=False, encoding="utf-8")

    # 5 worst-fitting scenarios by RMSE
    sorted_fit = sorted(fit_rows, key=lambda x: -x["kl_divergence"])
    worst_5 = sorted_fit[:5]

    # Historical validation (Qantas AGM Nov 2023 — Tier 4 scenario)
    historical = _validate_historical(
        scenarios, estimation_df, est_result, phi, anchored,
        scenario_ids, action_lists,
    )

    # MCMC convergence summary from StanEstimationResult
    mcmc_summary = {}
    if hasattr(est_result, "n_divergences"):
        mcmc_summary = {
            "n_divergences": est_result.n_divergences,
            "max_rhat": round(est_result.max_rhat, 4),
            "min_ess_bulk": round(est_result.min_ess_bulk, 1),
            "n_samples": est_result.n_samples,
        }

    validation = {
        "within_sample_fit": {
            "mean_rmse": round(mean_kl, 6),
            "metric": "rmse_likert_score" if likert_summary_df is not None else "posterior_mean_eu",
            "target": 0.05,
            "meets_target": mean_kl < 0.05,
        },
        # Keep old key for dashboard template compatibility
        "within_sample_kl": {
            "mean": round(mean_kl, 6),
            "target": 0.05,
            "meets_target": mean_kl < 0.05,
        },
        "worst_5_scenarios": [
            {"scenario_id": r["scenario_id"], "kl": r["kl_divergence"]}
            for r in worst_5
        ],
        "historical_prediction": historical,
        "mcmc_diagnostics": mcmc_summary,
    }

    # Save validation results
    with open(output_dir / "validation_results.json", "w", encoding="utf-8") as f:
        json.dump(validation, f, ensure_ascii=True, indent=2)

    logger.info(f"Validation: mean RMSE = {mean_kl:.4f} (target < 0.05)")
    return validation


def _compute_interaction_effects(
    scenarios: list[Scenario],
    estimation_df: pd.DataFrame,
    est_result: EstimationResult,
    fit_rows: list[dict],
) -> dict:
    """Compute interaction effect diagnostics for dashboard Panel 8."""
    from scipy import stats as sp_stats

    # Build scenario feature table
    sc_map = {s.scenario_id: s for s in scenarios if s.tier != 4}
    feature_rows = []
    for fr in fit_rows:
        sid = fr["scenario_id"]
        s = sc_map.get(sid)
        if not s:
            continue
        sv = s.state_vector
        vote = sv.get("vote_outcome_V", sv.get("vote_outcome", 0.0))
        strike = float(sv.get("strike", vote > 0.25))
        overwhelming = float(sv.get("overwhelming", vote > 0.50))
        ceo_present = float(sv.get("ceo_present_at_end", True))
        node = sv.get("decision_node", s.decision_node)
        residuals = json.loads(fr["residuals"])
        max_abs_resid = max(abs(v) for v in residuals.values()) if residuals else 0.0
        feature_rows.append({
            "scenario_id": sid,
            "vote": round(vote, 3),
            "strike": int(strike),
            "overwhelming": int(overwhelming),
            "ceo_present": int(ceo_present),
            "node": node,
            "kl": fr["kl_divergence"],
            "max_abs_resid": round(max_abs_resid, 4),
        })

    if not feature_rows:
        return {}

    fdf = pd.DataFrame(feature_rows)

    # 1. Residual vs vote scatter data
    resid_vs_vote = {
        "vote": fdf["vote"].tolist(),
        "kl": fdf["kl"].tolist(),
        "max_abs_resid": fdf["max_abs_resid"].tolist(),
        "scenario_ids": fdf["scenario_id"].tolist(),
    }

    # 2. KL by decision node
    kl_by_node = {}
    for node, grp in fdf.groupby("node"):
        kl_by_node[node] = {
            "mean_kl": round(float(grp["kl"].mean()), 6),
            "median_kl": round(float(grp["kl"].median()), 6),
            "n": int(len(grp)),
            "kl_values": grp["kl"].tolist(),
        }

    # 3. Strike × CEO interaction on fit quality
    strike_ceo_interaction = {}
    for (strike, ceo), grp in fdf.groupby(["strike", "ceo_present"]):
        key = f"strike={int(strike)}_ceo={int(ceo)}"
        strike_ceo_interaction[key] = {
            "mean_kl": round(float(grp["kl"].mean()), 6),
            "n": int(len(grp)),
            "mean_max_resid": round(float(grp["max_abs_resid"].mean()), 4),
        }

    # 4. Test: does model fit differ by strike status?
    strike_kl = fdf[fdf["strike"] == 1]["kl"].values
    no_strike_kl = fdf[fdf["strike"] == 0]["kl"].values
    strike_fit_test = {}
    if len(strike_kl) >= 3 and len(no_strike_kl) >= 3:
        u_stat, p_val = sp_stats.mannwhitneyu(strike_kl, no_strike_kl, alternative="two-sided")
        strike_fit_test = {
            "mean_kl_strike": round(float(strike_kl.mean()), 6),
            "mean_kl_no_strike": round(float(no_strike_kl.mean()), 6),
            "p_value": round(float(p_val), 4),
            "conclusion": "Fit differs by strike status" if p_val < 0.05 else "No significant difference",
        }

    # 5. Test: does model fit differ by overwhelming status?
    ovw_kl = fdf[fdf["overwhelming"] == 1]["kl"].values
    no_ovw_kl = fdf[fdf["overwhelming"] == 0]["kl"].values
    ovw_fit_test = {}
    if len(ovw_kl) >= 3 and len(no_ovw_kl) >= 3:
        u_stat, p_val = sp_stats.mannwhitneyu(ovw_kl, no_ovw_kl, alternative="two-sided")
        ovw_fit_test = {
            "mean_kl_overwhelming": round(float(ovw_kl.mean()), 6),
            "mean_kl_not_overwhelming": round(float(no_ovw_kl.mean()), 6),
            "p_value": round(float(p_val), 4),
            "conclusion": "Fit differs by overwhelming status" if p_val < 0.05 else "No significant difference",
        }

    # 6. Top 10 worst-fitting scenarios with features
    worst = fdf.nlargest(10, "kl")[["scenario_id", "vote", "strike", "overwhelming",
                                     "ceo_present", "node", "kl", "max_abs_resid"]]
    worst_scenarios = worst.to_dict("records")

    return {
        "resid_vs_vote": resid_vs_vote,
        "kl_by_node": kl_by_node,
        "strike_ceo_interaction": strike_ceo_interaction,
        "strike_fit_test": strike_fit_test,
        "overwhelming_fit_test": ovw_fit_test,
        "worst_fitting": worst_scenarios,
    }


# ── SEC 11: Dashboard rendering ──────────────────────────────────────────────

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
    interaction_effects: Optional[dict] = None
    validation_results: Optional[dict] = None
    cost_summary: Optional[dict] = None
    encoding_stats: Optional[dict] = None
    output_files: Optional[dict] = None
    preflight_checks: Optional[dict] = None
    feature_selection: Optional[dict] = None
    posterior_action_probs: Optional[dict] = None
    tree_data: Optional[dict] = None

    def to_json(self) -> str:
        d = {}
        for k in [
            "run_status", "run_start", "generated_at", "model",
            "scenarios", "elicitation_summary", "elicited_probabilities",
            "estimation_dataset_summary", "preflight_checks",
            "parameter_estimates", "covariance_matrix", "feature_selection",
            "posterior_action_probs", "tree_data",
            "behavioural_diagnostics",
            "interaction_effects", "validation_results", "cost_summary",
            "encoding_stats", "output_files",
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
        logger.info(f"Plotly.js cached ({len(bundle)//1024}KB)")
        return bundle
    except Exception as e:
        logger.warning(f"Failed to download Plotly.js: {e}. Using CDN fallback.")
        return ""


_DASHBOARD_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Board Utility Quantification Dashboard</title>
__META_REFRESH__
<style>
:root{--bg:#f8f9fa;--card:#fff;--border:#dee2e6;--primary:#4A90D9;--danger:#E85D5D;
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
th{background:var(--bg);font-weight:600;cursor:pointer;user-select:none}
th:hover{background:#e9ecef}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600}
.badge.confirmed{background:#d4edda;color:#155724}
.badge.null{background:#e2e3e5;color:#383d41}
.badge.insufficient{background:#fff3cd;color:#856404}
.search-box{padding:8px 12px;border:1px solid var(--border);border-radius:4px;width:300px;margin-bottom:12px}
.download-link{display:inline-block;padding:8px 16px;background:var(--primary);color:#fff;
text-decoration:none;border-radius:4px;margin:4px}
.download-link:hover{opacity:0.9}
.chart{min-height:400px;margin:12px 0}
</style>
__PLOTLY_SCRIPT__
<script src="https://d3js.org/d3.v7.min.js"></script>
</head>
<body>
<script>const RESULTS_DATA = __RESULTS_DATA__;</script>

<div id="banner"></div>
<div class="tabs" id="tabBar"></div>
<div class="content" id="content"></div>

<script>
const TAB_NAMES = [
  "Overview","Cost & Usage","Scenario Battery","Elicitation Results",
  "Elicited Probabilities","Decision Tree","Parameter Estimates","Covariance",
  "Behavioural Diagnostics","Interaction Effects","Validation","Linearity Diagnostics","Raw Data"
];
const D = RESULTS_DATA;

// Banner
(function(){
  const b = document.getElementById('banner');
  b.className = 'banner ' + (D.run_status||'in_progress');
  if(D.run_status==='in_progress') b.textContent='Run in progress -- last updated '+(D.generated_at||'');
  else if(D.run_status==='complete') b.textContent='Run completed '+(D.generated_at||'')+'. This file is self-contained and can be shared.';
  else if(D.run_status==='aborted') b.textContent='Run aborted. Partial data shown.';
  else b.textContent='';
})();

// Tabs
(function(){
  const bar=document.getElementById('tabBar');
  const cont=document.getElementById('content');
  TAB_NAMES.forEach((name,i)=>{
    const t=document.createElement('div');
    t.className='tab'+(i===0?' active':'');
    t.textContent=name;
    t.onclick=()=>{
      document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
      document.querySelectorAll('.panel').forEach(x=>x.classList.remove('active'));
      t.classList.add('active');
      document.getElementById('panel_'+i).classList.add('active');
      document.getElementById('content').style.maxWidth=(i===5)?'none':'1400px';
    };
    bar.appendChild(t);
    const p=document.createElement('div');
    p.id='panel_'+i;
    p.className='panel'+(i===0?' active':'');
    cont.appendChild(p);
  });
})();

function placeholder(panelIdx,msg){
  document.getElementById('panel_'+panelIdx).innerHTML='<div class="card"><p>'+msg+'</p></div>';
}
function sortTable(table,col){
  const rows=Array.from(table.querySelectorAll('tbody tr'));
  const asc=table.dataset.sortCol==col?!(table.dataset.sortAsc==='true'):true;
  table.dataset.sortCol=col;table.dataset.sortAsc=asc;
  rows.sort((a,b)=>{
    let va=a.cells[col].textContent,vb=b.cells[col].textContent;
    let na=parseFloat(va),nb=parseFloat(vb);
    if(!isNaN(na)&&!isNaN(nb))return asc?na-nb:nb-na;
    return asc?va.localeCompare(vb):vb.localeCompare(va);
  });
  const tb=table.querySelector('tbody');
  rows.forEach(r=>tb.appendChild(r));
}
function makeTable(headers,rows,id){
  let h='<table id="'+id+'"><thead><tr>';
  headers.forEach((hd,i)=>h+='<th onclick="sortTable(this.closest(\'table\'),'+i+')">'+hd+'</th>');
  h+='</tr></thead><tbody>';
  rows.forEach(r=>{h+='<tr>';r.forEach(c=>h+='<td>'+c+'</td>');h+='</tr>';});
  h+='</tbody></table>';
  return h;
}

// Panel 0: Overview
(function(){
  const p=document.getElementById('panel_0');
  const sc=D.scenarios||[];
  const cs=D.cost_summary||{};
  const t1=sc.filter(s=>s.tier===1).length,t2=sc.filter(s=>s.tier===2).length;
  const t3=sc.filter(s=>s.tier===3).length,t4=sc.filter(s=>s.tier===4).length;
  let html='<div class="card"><h3>Run Summary</h3><div class="stat-grid">';
  html+='<div class="stat"><div class="value">'+sc.length+'</div><div class="label">Total Scenarios</div></div>';
  html+='<div class="stat"><div class="value">'+t1+'/'+t2+'/'+t3+'/'+t4+'</div><div class="label">T1/T2/T3/T4</div></div>';
  html+='<div class="stat"><div class="value">'+(D.model||'--')+'</div><div class="label">Model</div></div>';
  html+='<div class="stat"><div class="value">$'+(cs.total_cost_usd||0).toFixed(4)+'</div><div class="label">Total Cost</div></div>';
  html+='<div class="stat"><div class="value">'+(cs.total_calls||0)+'</div><div class="label">API Calls</div></div>';
  const pe=D.parameter_estimates||{};
  html+='<div class="stat"><div class="value">'+(pe.converged?'Yes':'No')+'</div><div class="label">Converged</div></div>';
  html+='</div></div>';
  if(D.validation_results&&D.validation_results.within_sample_kl){
    const kl=D.validation_results.within_sample_kl;
    html+='<div class="card"><h3>Fit Quality</h3><p>Mean KL divergence: <strong>'+
      (kl.mean||'--')+'</strong> (target &lt; 0.05, '+(kl.meets_target?'MET':'NOT MET')+')</p></div>';
  }
  // Pre-flight checks
  const pf=D.preflight_checks||{};
  if(pf.checks&&pf.checks.length>0){
    html+='<div class="card"><h3>Pre-flight Checks</h3>';
    html+='<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:8px">';
    pf.checks.forEach(c=>{
      const bg=c.passed?(c.warning?'#fff3cd':'#d4edda'):'#f8d7da';
      const fg=c.passed?(c.warning?'#856404':'#155724'):'#721c24';
      const icon=c.passed?(c.warning?'\u26A0':'\u2713'):'\u2717';
      html+='<span style="background:'+bg+';color:'+fg+';padding:4px 10px;border-radius:4px;font-size:0.9em">'+icon+' '+c.name+'</span>';
    });
    html+='</div>';
    html+='<table style="border-collapse:collapse;font-size:0.85em;width:100%">';
    html+='<thead><tr><th style="border:1px solid #ddd;padding:3px">Check</th><th style="border:1px solid #ddd;padding:3px">Status</th><th style="border:1px solid #ddd;padding:3px">Detail</th><th style="border:1px solid #ddd;padding:3px">Threshold</th></tr></thead><tbody>';
    pf.checks.forEach(c=>{
      const s=c.passed?(c.warning?'WARN':'PASS'):'FAIL';
      html+='<tr><td style="border:1px solid #ddd;padding:3px">'+c.name+'</td>';
      html+='<td style="border:1px solid #ddd;padding:3px;font-weight:bold;color:'+(c.passed?'#155724':'#721c24')+'">'+s+'</td>';
      html+='<td style="border:1px solid #ddd;padding:3px">'+c.detail+'</td>';
      html+='<td style="border:1px solid #ddd;padding:3px;color:#666">'+(c.threshold||'')+'</td></tr>';
    });
    html+='</tbody></table></div>';
  }
  p.innerHTML=html;
})();

// Panel 1: Cost & Usage
(function(){
  const p=document.getElementById('panel_1');
  const cs=D.cost_summary||{};
  let html='<div class="card"><h3>Token Usage</h3><div class="stat-grid">';
  html+='<div class="stat"><div class="value">'+(cs.total_prompt_tokens||0).toLocaleString()+'</div><div class="label">Prompt Tokens</div></div>';
  html+='<div class="stat"><div class="value">'+(cs.total_completion_tokens||0).toLocaleString()+'</div><div class="label">Completion Tokens</div></div>';
  html+='<div class="stat"><div class="value">$'+(cs.total_cost_usd||0).toFixed(4)+'</div><div class="label">Total Cost</div></div>';
  html+='</div></div>';
  const enc=D.encoding_stats||{};
  html+='<div class="card"><h3>Encoding Issues</h3><p>Non-ASCII replacements: '+(enc.non_ascii||0)+
    ', BOM removals: '+(enc.bom||0)+', ZWSP removals: '+(enc.zwsp||0)+'</p></div>';
  p.innerHTML=html;
})();

// Panel 2: Scenario Battery
(function(){
  const p=document.getElementById('panel_2');
  const sc=D.scenarios||[];
  if(!sc.length){placeholder(2,'No scenarios generated yet.');return;}
  // Tier/node summary counts
  const tierCounts={};sc.forEach(s=>{tierCounts[s.tier]=(tierCounts[s.tier]||0)+1;});
  const nodeCounts={};sc.forEach(s=>{nodeCounts[s.decision_node]=(nodeCounts[s.decision_node]||0)+1;});
  let html='<div class="card"><h3>Scenario Battery ('+sc.length+' scenarios)</h3>';
  html+='<div style="display:flex;gap:24px;margin-bottom:12px;font-size:0.9em;color:var(--text-muted)">';
  html+='<span><strong>By tier:</strong> '+Object.entries(tierCounts).sort((a,b)=>a[0]-b[0]).map(([t,n])=>'T'+t+':&nbsp;'+n).join(', ')+'</span>';
  html+='<span><strong>By node:</strong> '+Object.entries(nodeCounts).map(([n,c])=>n+':&nbsp;'+c).join(', ')+'</span>';
  html+='</div>';
  html+='<input class="search-box" placeholder="Filter scenarios..." oninput="filterScenarios(this.value)">';
  // Node filter buttons
  const nodeList=Object.keys(nodeCounts).sort();
  html+='<div style="margin:8px 0"><strong style="font-size:0.85em">Filter node:</strong> ';
  html+='<button class="node-btn" data-node="all" style="margin:2px 4px;padding:3px 10px;border:1px solid var(--border);border-radius:4px;cursor:pointer;background:var(--primary);color:#fff;font-size:0.85em">All</button>';
  nodeList.forEach(n=>{
    html+='<button class="node-btn" data-node="'+n+'" style="margin:2px 4px;padding:3px 10px;border:1px solid var(--border);border-radius:4px;cursor:pointer;background:var(--card);font-size:0.85em">'+n+'</button>';
  });
  html+='</div>';
  const headers=['ID','Tier','Target','Node','CEO Start','D1 Action',
    'Vote %','Strike','Overwh.','Review','Review Result','CEO End','Actions'];
  const rows=sc.map(s=>{
    const sv=s.state_vector||{};
    const v=sv.vote_outcome_V||0;
    const ceoStart=sv.ceo_status_at_start==='resigned_early'?'Resigned':'Present';
    const d1Act=sv.d1_action||'--';
    const votePct=v>0?(v*100).toFixed(0)+'%':'--';
    const strike=v>=0.25?'\u2714':'--';
    const overwh=v>=0.50?'\u2714':'--';
    const review=sv.review_commissioned?'\u2714':'--';
    const revResult=sv.review_outcome==='negative'?'Negative':sv.review_outcome==='balanced'?'Balanced':sv.review_outcome==='positive'?'Positive':'--';
    const ceoEnd=sv.ceo_present_at_end?'Present':'Removed';
    const actions=(s.feasible_actions||[]).join(', ');
    return [s.scenario_id,s.tier,s.target_parameter,s.decision_node,
      ceoStart,d1Act,votePct,strike,overwh,review,revResult,ceoEnd,actions];
  });
  html+=makeTable(headers,rows,'scenarioTable');
  html+='</div>';
  p.innerHTML=html;
  // Node filter click handler
  document.querySelectorAll('.node-btn').forEach(btn=>{
    btn.onclick=function(){
      document.querySelectorAll('.node-btn').forEach(b=>{b.style.background='var(--card)';b.style.color='inherit';});
      this.style.background='var(--primary)';this.style.color='#fff';
      const node=this.dataset.node;
      document.querySelectorAll('#scenarioTable tbody tr').forEach(r=>{
        r.style.display=(node==='all'||r.cells[3].textContent===node)?'':'none';
      });
    };
  });
})();
window.filterScenarios=function(q){
  const rows=document.querySelectorAll('#scenarioTable tbody tr');
  q=q.toLowerCase();
  rows.forEach(r=>{r.style.display=r.textContent.toLowerCase().includes(q)?'':'none';});
};

// Panel 3: Elicitation Results
(function(){
  const p=document.getElementById('panel_3');
  const es=D.elicitation_summary||{};
  const ep=D.elicited_probabilities||[];
  const cs=D.cost_summary||{};
  // If no explicit summary but elicited_probabilities data exists, derive summary
  const hasExplicitSummary=es&&es.total_calls>0;
  const hasData=ep.length>0;
  if(!hasExplicitSummary&&!hasData){placeholder(3,'Elicitation not yet run.');return;}

  let html='';

  // Summary stats card
  const totalScenarios=hasData?ep.length:0;
  const totalDraws=hasData?ep.reduce((s,e)=>s+(e.n_draws||0),0):0;
  const totalCost=cs.total_cost_usd||0;
  const totalTokens=cs.total_tokens||0;
  html+='<div class="card"><h3>Elicitation Summary</h3><div class="stat-grid">';
  if(hasExplicitSummary){
    html+='<div class="stat"><div class="value">'+(es.total_calls||0)+'</div><div class="label">Total API Calls</div></div>';
    html+='<div class="stat"><div class="value">'+(es.success_rate||0).toFixed(1)+'%</div><div class="label">Parse Success</div></div>';
    html+='<div class="stat"><div class="value">'+(es.cache_hit_rate||0).toFixed(1)+'%</div><div class="label">Cache Hit Rate</div></div>';
  } else {
    html+='<div class="stat"><div class="value">'+totalScenarios+'</div><div class="label">Scenarios Elicited</div></div>';
    html+='<div class="stat"><div class="value">'+totalDraws.toLocaleString()+'</div><div class="label">Total Draws</div></div>';
  }
  html+='<div class="stat"><div class="value">'+totalTokens.toLocaleString()+'</div><div class="label">Total Tokens</div></div>';
  html+='<div class="stat"><div class="value">$'+totalCost.toFixed(2)+'</div><div class="label">Total Cost</div></div>';
  html+='</div></div>';

  if(hasExplicitSummary&&es.parse_status_counts){
    html+='<div class="card"><h3>Parse Status Breakdown</h3><div id="parseChart" class="chart"></div></div>';
  }

  // Per-scenario results table
  if(hasData){
    // Draws per tier
    const tierStats={};
    ep.forEach(e=>{
      const t=e.tier;
      if(!tierStats[t])tierStats[t]={count:0,draws:0};
      tierStats[t].count++;
      tierStats[t].draws+=(e.n_draws||0);
    });
    html+='<div class="card"><h3>Coverage by Tier</h3>';
    html+='<table style="border-collapse:collapse;font-size:0.9em;width:auto">';
    html+='<thead><tr><th style="border:1px solid var(--border);padding:6px 12px">Tier</th><th style="border:1px solid var(--border);padding:6px 12px">Scenarios</th><th style="border:1px solid var(--border);padding:6px 12px">Total Draws</th><th style="border:1px solid var(--border);padding:6px 12px">Mean Draws/Scenario</th></tr></thead><tbody>';
    Object.entries(tierStats).sort((a,b)=>a[0]-b[0]).forEach(([t,s])=>{
      html+='<tr><td style="border:1px solid var(--border);padding:6px 12px;text-align:center">T'+t+'</td>';
      html+='<td style="border:1px solid var(--border);padding:6px 12px;text-align:right">'+s.count+'</td>';
      html+='<td style="border:1px solid var(--border);padding:6px 12px;text-align:right">'+s.draws+'</td>';
      html+='<td style="border:1px solid var(--border);padding:6px 12px;text-align:right">'+(s.draws/s.count).toFixed(0)+'</td></tr>';
    });
    html+='</tbody></table></div>';

    // Action preference distribution across scenarios
    const allActions=new Set();
    ep.forEach(e=>{Object.keys(e.mean_scores||{}).forEach(a=>allActions.add(a));});
    const actionList=Array.from(allActions).sort();
    // Count how many scenarios each action "wins" (highest mean score)
    const actionWins={};actionList.forEach(a=>{actionWins[a]=0;});
    ep.forEach(e=>{
      const scores=e.mean_scores||{};
      let best='',bestV=-Infinity;
      for(const[a,v]of Object.entries(scores)){if(v>bestV){bestV=v;best=a;}}
      if(best)actionWins[best]=(actionWins[best]||0)+1;
    });
    html+='<div class="card"><h3>Preferred Action Distribution</h3>';
    html+='<p style="color:var(--text-muted);font-size:0.85em">Number of scenarios where each action has the highest mean Likert score.</p>';
    html+='<div id="actionWinsChart" class="chart"></div></div>';

    // Mean score distribution chart
    html+='<div class="card"><h3>Mean Likert Score Distribution</h3>';
    html+='<p style="color:var(--text-muted);font-size:0.85em">Distribution of mean Likert scores across all scenarios, by action.</p>';
    html+='<div id="scoreDistChart" class="chart"></div></div>';

    // Scenario detail table
    html+='<div class="card"><h3>Per-Scenario Results ('+ep.length+' scenarios)</h3>';
    html+='<input id="elicResultsSearch" type="text" placeholder="Filter by scenario ID or target..." style="width:100%;max-width:400px;padding:6px 10px;margin-bottom:8px;border:1px solid var(--border);border-radius:4px">';
    html+='<div style="overflow-x:auto"><table id="elicResultsTable" style="width:100%;border-collapse:collapse;font-size:13px">';
    html+='<thead><tr style="background:#f1f3f5">';
    ['Scenario','Tier','Target','Node','Vote %','CEO Start','Strike','Overwh.','Review Result','CEO End','Draws'].forEach(h=>{
      html+='<th style="padding:6px 8px;border:1px solid var(--border);text-align:left;cursor:pointer" onclick="sortTable(this.closest(\'table\'),'+['Scenario','Tier','Target','Node','Vote %','CEO Start','Strike','Overwh.','Review Result','CEO End','Draws'].indexOf(h)+')">'+h+'</th>';
    });
    actionList.forEach(a=>{
      html+='<th style="padding:6px 8px;border:1px solid var(--border);text-align:right">'+a+'</th>';
    });
    html+='<th style="padding:6px 8px;border:1px solid var(--border);text-align:left">Best Action</th>';
    html+='</tr></thead><tbody>';
    ep.forEach(e=>{
      const scores=e.mean_scores||{};
      let best='',bestV=-Infinity;
      for(const[a,v]of Object.entries(scores)){if(v>bestV){bestV=v;best=a;}}
      const votePct=e.vote_pct!=null&&e.vote_pct>0?(e.vote_pct*100).toFixed(0)+'%':'--';
      const ceoStart=(e.ceo_status==='resigned_early')?'Resigned':'Present';
      const revResult=e.review_outcome==='negative'?'Negative':e.review_outcome==='balanced'?'Balanced':e.review_outcome==='positive'?'Positive':'--';
      html+='<tr data-search="'+(e.scenario_id+' '+e.target_parameter).toLowerCase()+'">';
      html+='<td style="padding:5px 8px;border:1px solid var(--border);font-family:monospace;font-size:12px">'+e.scenario_id+'</td>';
      html+='<td style="padding:5px 8px;border:1px solid var(--border);text-align:center">T'+e.tier+'</td>';
      html+='<td style="padding:5px 8px;border:1px solid var(--border)">'+e.target_parameter+'</td>';
      html+='<td style="padding:5px 8px;border:1px solid var(--border)">'+e.decision_node+'</td>';
      html+='<td style="padding:5px 8px;border:1px solid var(--border);text-align:right">'+votePct+'</td>';
      html+='<td style="padding:5px 8px;border:1px solid var(--border)">'+ceoStart+'</td>';
      html+='<td style="padding:5px 8px;border:1px solid var(--border);text-align:center">'+(e.strike?'\u2714':'--')+'</td>';
      html+='<td style="padding:5px 8px;border:1px solid var(--border);text-align:center">'+(e.overwhelming?'\u2714':'--')+'</td>';
      html+='<td style="padding:5px 8px;border:1px solid var(--border)">'+revResult+'</td>';
      html+='<td style="padding:5px 8px;border:1px solid var(--border)">'+(e.ceo_present_at_end?'Present':'Removed')+'</td>';
      html+='<td style="padding:5px 8px;border:1px solid var(--border);text-align:right">'+(e.n_draws||0)+'</td>';
      actionList.forEach(a=>{
        const sv=scores[a]||0;
        const isBest=a===best&&bestV>0;
        const bg=isBest?'background:#d4edda':'';
        html+='<td style="padding:5px 8px;border:1px solid var(--border);text-align:right;'+bg+'">'+sv.toFixed(2)+'</td>';
      });
      html+='<td style="padding:5px 8px;border:1px solid var(--border);font-weight:bold">'+best+'</td>';
      html+='</tr>';
    });
    html+='</tbody></table></div></div>';
  }

  p.innerHTML=html;

  // Search handler for results table
  const searchBox=document.getElementById('elicResultsSearch');
  if(searchBox){
    searchBox.oninput=function(){
      const q=this.value.toLowerCase();
      document.querySelectorAll('#elicResultsTable tbody tr').forEach(r=>{
        r.style.display=(r.dataset.search||'').includes(q)?'':'none';
      });
    };
  }

  // Plotly charts
  if(typeof Plotly!=='undefined'){
    // Parse status pie (if available)
    if(hasExplicitSummary&&es.parse_status_counts){
      const labels=Object.keys(es.parse_status_counts);
      const vals=Object.values(es.parse_status_counts);
      Plotly.newPlot('parseChart',[{labels,values:vals,type:'pie',hole:0.4,
        marker:{colors:['#50C878','#E85D5D','#FFB347','#4A90D9','#AAAAAA']}}],
        {margin:{t:20,b:20,l:20,r:20},height:300},{responsive:true});
    }
    // Action wins bar chart
    if(hasData&&document.getElementById('actionWinsChart')){
      const aw={};
      ep.forEach(e=>{
        const sc=e.mean_scores||{};let best='',bv=-Infinity;
        for(const[a,v]of Object.entries(sc)){if(v>bv){bv=v;best=a;}}
        if(best)aw[best]=(aw[best]||0)+1;
      });
      const actLabels=Object.keys(aw);
      const actVals=Object.values(aw);
      Plotly.newPlot('actionWinsChart',[{x:actLabels,y:actVals,type:'bar',
        marker:{color:'#4A90D9'}}],
        {margin:{t:20,b:40,l:50,r:20},height:300,
         yaxis:{title:'# Scenarios'},xaxis:{title:'Action'}},{responsive:true});
    }
    // Score distribution box plot
    if(hasData&&document.getElementById('scoreDistChart')){
      const allAct=new Set();
      ep.forEach(e=>{Object.keys(e.mean_scores||{}).forEach(a=>allAct.add(a));});
      const traces=[];
      const colors=['#4A90D9','#50C878','#E85D5D','#FFB347','#9B59B6','#1ABC9C'];
      let ci=0;
      Array.from(allAct).sort().forEach(a=>{
        const vals=ep.map(e=>(e.mean_scores||{})[a]).filter(v=>v!=null);
        traces.push({y:vals,type:'box',name:a,marker:{color:colors[ci%colors.length]},boxpoints:'outliers'});
        ci++;
      });
      Plotly.newPlot('scoreDistChart',traces,
        {margin:{t:20,b:40,l:50,r:20},height:350,
         yaxis:{title:'Mean Likert Score (1-5)'}},{responsive:true});
    }
  }
})();

// Panel 4: Elicited Likert Scores
(function(){
  const p=document.getElementById('panel_4');
  const ep=D.elicited_probabilities;
  if(!ep||!ep.length){placeholder(4,'Elicitation data not yet available.');return;}

  // Collect all action codes across all scenarios
  const allActions=new Set();
  ep.forEach(s=>{Object.keys(s.mean_scores||{}).forEach(a=>allActions.add(a));});
  const actionList=Array.from(allActions).sort();

  // Summary stats
  let html='<div class="card"><h3>Elicited Likert Scores by Scenario</h3>';
  html+='<p style="color:var(--text-muted)">Mean Likert scores (1\u20135) across draws per scenario. Highest-scored action highlighted green.</p>';

  // Tier filter buttons
  const tiers=[...new Set(ep.map(s=>s.tier))].sort();
  html+='<div style="margin:10px 0"><strong>Filter by tier:</strong> ';
  html+='<button class="tier-btn" data-tier="all" style="margin:2px 4px;padding:4px 12px;border:1px solid var(--border);border-radius:4px;cursor:pointer;background:var(--primary);color:#fff">All</button>';
  tiers.forEach(t=>{
    html+='<button class="tier-btn" data-tier="'+t+'" style="margin:2px 4px;padding:4px 12px;border:1px solid var(--border);border-radius:4px;cursor:pointer;background:var(--card)">T'+t+'</button>';
  });
  html+='</div>';

  // Search box
  html+='<input id="epSearch" type="text" placeholder="Search scenario ID or target parameter..." style="width:100%;max-width:400px;padding:6px 10px;margin-bottom:10px;border:1px solid var(--border);border-radius:4px">';

  // Table header
  html+='<div style="overflow-x:auto"><table id="epTable" class="sortable-table" style="width:100%;border-collapse:collapse;font-size:13px">';
  html+='<thead><tr style="background:#f1f3f5">';
  html+='<th style="padding:8px;border:1px solid var(--border);text-align:left">Scenario</th>';
  html+='<th style="padding:8px;border:1px solid var(--border);text-align:left">Tier</th>';
  html+='<th style="padding:8px;border:1px solid var(--border);text-align:left">Target</th>';
  html+='<th style="padding:8px;border:1px solid var(--border);text-align:left">Node</th>';
  html+='<th style="padding:8px;border:1px solid var(--border);text-align:right">Vote%</th>';
  actionList.forEach(a=>{
    html+='<th style="padding:8px;border:1px solid var(--border);text-align:right">'+a+'</th>';
  });
  html+='<th style="padding:8px;border:1px solid var(--border);text-align:right">Draws</th>';
  html+='</tr></thead><tbody>';

  // Table rows
  ep.forEach(s=>{
    const scores=s.mean_scores||{};
    const maxS=Math.max(...actionList.map(a=>scores[a]||0));
    html+='<tr data-tier="'+s.tier+'" data-search="'+(s.scenario_id+' '+s.target_parameter).toLowerCase()+'">';
    html+='<td style="padding:6px 8px;border:1px solid var(--border);font-family:monospace;font-size:12px">'+s.scenario_id+'</td>';
    html+='<td style="padding:6px 8px;border:1px solid var(--border);text-align:center">T'+s.tier+'</td>';
    html+='<td style="padding:6px 8px;border:1px solid var(--border)">'+s.target_parameter+'</td>';
    html+='<td style="padding:6px 8px;border:1px solid var(--border)">'+s.decision_node+'</td>';
    html+='<td style="padding:6px 8px;border:1px solid var(--border);text-align:right">'+(s.vote_pct!=null?(s.vote_pct*100).toFixed(0)+'%':'--')+'</td>';
    actionList.forEach(a=>{
      const sv=scores[a]||0;
      const bg=sv>=maxS-0.01&&sv>0?'background:#d4edda':'';
      html+='<td style="padding:6px 8px;border:1px solid var(--border);text-align:right;'+bg+'">'+sv.toFixed(1)+'</td>';
    });
    html+='<td style="padding:6px 8px;border:1px solid var(--border);text-align:right">'+(s.n_draws||0)+'</td>';
    html+='</tr>';
  });
  html+='</tbody></table></div></div>';

  // Grouped bar chart: one trace per action, x = scenario_id
  html+='<div class="card"><h3>Mean Likert Score by Action</h3>';
  html+='<p style="color:var(--text-muted)">Grouped bar chart of mean Likert scores (1\u20135). Click tier buttons above to filter.</p>';
  html+='<div id="epBarChart" class="chart"></div></div>';

  p.innerHTML=html;

  // Tier filter logic
  const btns=p.querySelectorAll('.tier-btn');
  btns.forEach(btn=>{
    btn.onclick=function(){
      btns.forEach(b=>{b.style.background='var(--card)';b.style.color='var(--text)';});
      this.style.background='var(--primary)';this.style.color='#fff';
      const tier=this.dataset.tier;
      const rows=p.querySelectorAll('#epTable tbody tr');
      rows.forEach(r=>{r.style.display=(tier==='all'||r.dataset.tier===tier)?'':'none';});
      renderBarChart(tier);
    };
  });

  // Search logic
  const searchBox=p.querySelector('#epSearch');
  searchBox.oninput=function(){
    const q=this.value.toLowerCase();
    p.querySelectorAll('#epTable tbody tr').forEach(r=>{
      r.style.display=r.dataset.search.includes(q)?'':'none';
    });
  };

  // Bar chart rendering
  const colors=['#4A90D9','#E85D5D','#50C878','#FFB347','#9B59B6','#1ABC9C'];
  function renderBarChart(tierFilter){
    const filtered=tierFilter==='all'?ep:ep.filter(s=>String(s.tier)===tierFilter);
    const ids=filtered.map(s=>s.scenario_id);
    const traces=actionList.map((a,idx)=>({
      x:ids,
      y:filtered.map(s=>(s.mean_scores||{})[a]||0),
      name:a,
      type:'bar',
      marker:{color:colors[idx%colors.length]}
    }));
    Plotly.newPlot('epBarChart',traces,{
      barmode:'group',
      margin:{t:20,b:120,l:60,r:20},
      height:Math.max(400,Math.min(600,ids.length*8)),
      xaxis:{title:'Scenario',tickangle:-45,tickfont:{size:10}},
      yaxis:{title:'Mean Likert Score',range:[0,5.5]},
      legend:{orientation:'h',y:1.12}
    },{responsive:true});
  }
  if(typeof Plotly!=='undefined') renderBarChart('all');
})();

// Panel 5: Decision Tree
(function(){
  var panel=document.getElementById('panel_5');
  var td=D.tree_data;
  var ep=D.elicited_probabilities;
  if(!td&&(!ep||!ep.length)){placeholder(5,'Tree data not yet available.');return;}

  var COL={Board:'#4A90D9',ASA:'#50C878',CEO:'#E85D5D',Nature:'#AAAAAA'};
  var NW=54,NH=32;
  var NE={
    CEO_resign:'CEO resigns',CEO_stay:'CEO stays',
    D0_minimal:'Do nothing',D1_review:'Commission review',D3_ceo_transition:'Force CEO exit',
    A2_no_strike:'No strike',A2_rec_strike:'Rec. strike',
    Drev_no_action:'No action',Drev_commission_review:'Commission review',Drev_sack_ceo:'Sack CEO',
    D4_stay:'Stay',D4_resign:'Resign',D4_negotiate_exit:'Negotiate exit',
    no_strike:'No strike',first_strike:'First strike',overwhelming:'Overwhelming',
    adverse:'Adverse',positive:'Positive',ceo_absent:'(CEO absent)'
  };

  var nid=0;
  function nd(name,type,owner){
    return{id:name+'_'+(nid++),name:name,type:type,owner:owner,eu:0,
      nice_label:name,colour:COL[owner]||'#888',children:[]};
  }
  function ed(label,prob,child,eu){
    return{label:label,nice_label:NE[label]||label.replace(/_/g,' '),
      prob:prob,is_actual:false,child_eu:eu||0,commentary:'',child:child};
  }

  /* Build D3 tree from pre-computed recursive EU tree_data */
  function buildFromPrecomputed(node){
    if(!node)return nd('Terminal','terminal','Nature');
    var n=nd(node.label||'?',node.type||'terminal',node.owner||'Nature');
    n.eu=node.eu||0;
    if(node.components)n.components=node.components;
    if(node.edges&&node.edges.length){
      node.edges.forEach(function(e){
        var childNode=buildFromPrecomputed(e.child);
        n.children.push(ed(e.action,e.prob,childNode,e.eu));
      });
    }
    return n;
  }

  /* Fallback: build skeleton from elicitation data (pre-Stage-4) */
  function buildFallbackTree(){
    nid=0;
    function TN(){return nd('Terminal','terminal','Nature');}
    function avgP(arr){
      if(!arr||!arr.length)return null;
      var acts2={};
      arr.forEach(function(s){for(var a in(s.mean_scores||{}))acts2[a]=true});
      var scores={};
      for(var a in acts2){
        var vs=arr.map(function(s){return(s.mean_scores||{})[a]||0});
        scores[a]=vs.reduce(function(x,y){return x+y},0)/vs.length;
      }
      var sum=0;for(var a in scores)sum+=scores[a];
      var probs={};
      if(sum>0){for(var a in scores)probs[a]=scores[a]/sum;}
      else{for(var a in scores)probs[a]=0;}
      return{probs:probs,n:arr.length};
    }
    function getEP(dn,f){
      var m=ep.filter(function(s){return s.decision_node===dn});
      if(f){
        var filters=[function(s){return!f.cs||s.ceo_status===f.cs}];
        for(var i=0;i<filters.length;i++){var next=m.filter(filters[i]);if(next.length>0)m=next;}
      }
      return avgP(m);
    }
    function renorm(el,acts){
      if(!el)return null;
      var sum=0;acts.forEach(function(a){sum+=(el.probs[a]||0)});
      if(sum<=0)return null;
      var out={};acts.forEach(function(a){out[a]=(el.probs[a]||0)/sum});
      return{probs:out,n:el.n};
    }
    var d0=nd('D0_ceo','decision','CEO');
    function bD1(cp,cs){
      var n=nd('D1','decision','Board');
      var acts=cp?['D0_minimal','D1_review','D3_ceo_transition']:['D0_minimal','D1_review'];
      var el=renorm(getEP('D1',{cs:cs}),acts);
      acts.forEach(function(a){
        var p=el?(el.probs[a]||0):null;
        var n2=nd('A2','decision','ASA');
        ['A2_no_strike','A2_rec_strike'].forEach(function(a2){
          var nv=nd('V','chance','Nature');
          ['no_strike','first_strike','overwhelming'].forEach(function(v){
            var nd4=nd('D4','decision','CEO');
            var ndr=nd('D_rev','decision','Board');
            ndr.children.push(ed('Drev_no_action',null,TN()));
            nd4.children.push(ed('D4_stay',null,ndr));
            nv.children.push(ed(v,null,nd4));
          });
          n2.children.push(ed(a2,null,nv));
        });
        n.children.push(ed(a,p,n2));
      });
      return n;
    }
    d0.children.push(ed('CEO_resign',null,bD1(false,'resigned_early')));
    d0.children.push(ed('CEO_stay',null,bD1(true,'present')));
    return d0;
  }

  var treeData=td?buildFromPrecomputed(td):buildFallbackTree();

  /* HTML */
  panel.style.maxWidth='none';
  panel.style.width='100%';
  var srcLabel=td?'Recursive EU (posterior weights)':'Likert scores (pre-estimation fallback)';
  panel.innerHTML='<div class="card" style="max-width:none">'+
    '<h3>Game Tree \u2014 Board Decision Probabilities</h3>'+
    '<p style="color:var(--text-muted)">Full game tree with Board action probabilities from '+srcLabel+'. '+
    'All edges show p=X%. Click nodes to expand/collapse. Scroll to zoom, drag to pan.</p>'+
    '<div style="display:flex;gap:8px;margin:10px 0;flex-wrap:wrap">'+
    '<button id="t5Exp" style="padding:5px 14px;border:1px solid #bbb;border-radius:4px;background:#fff;cursor:pointer;font-size:12px">Expand All</button>'+
    '<button id="t5Col" style="padding:5px 14px;border:1px solid #bbb;border-radius:4px;background:#fff;cursor:pointer;font-size:12px">Collapse All</button>'+
    '<button id="t5Fit" style="padding:5px 14px;border:1px solid #bbb;border-radius:4px;background:#fff;cursor:pointer;font-size:12px">Fit to Screen</button>'+
    '</div>'+
    '<div style="display:flex;gap:14px;flex-wrap:wrap;font-size:11px;margin-bottom:8px">'+
    '<span><svg width="14" height="14"><rect x="1" y="1" width="12" height="12" rx="2" fill="#4A90D9"/></svg> Board</span>'+
    '<span><svg width="14" height="14"><rect x="1" y="1" width="12" height="12" rx="2" fill="#50C878"/></svg> ASA</span>'+
    '<span><svg width="14" height="14"><rect x="1" y="1" width="12" height="12" rx="2" fill="#E85D5D"/></svg> CEO</span>'+
    '<span><svg width="14" height="14"><ellipse cx="7" cy="7" rx="6" ry="6" fill="#AAAAAA"/></svg> Nature</span>'+
    '<span><svg width="14" height="14"><polygon points="7,1 13,7 7,13 1,7" fill="#888"/></svg> Terminal</span>'+
    '</div>'+
    '<div id="t5Wrap" style="border:1px solid var(--border);border-radius:6px;background:#fafbfc;height:calc(100vh - 220px);min-height:500px;overflow:hidden;position:relative">'+
    '<svg id="t5Svg" width="100%" height="100%"></svg></div></div>'+
    '<div id="t5TT" style="position:fixed;padding:10px 14px;background:rgba(20,20,30,0.92);color:#eee;border-radius:6px;'+
    'font-size:11px;line-height:1.6;max-width:380px;pointer-events:none;z-index:9999;display:none;box-shadow:0 3px 12px rgba(0,0,0,0.3)"></div>'+
    '<div id="t5NP" style="position:fixed;background:#fff;border-radius:8px;padding:16px 20px;box-shadow:0 6px 30px rgba(0,0,0,0.22);'+
    'max-width:420px;width:auto;max-height:70vh;overflow-y:auto;z-index:10000;display:none;font-size:13px;line-height:1.6">'+
    '<div id="t5NPContent"></div></div>';

  if(typeof d3==='undefined'){document.getElementById('t5Wrap').innerHTML='<p style="padding:20px;color:#c00">D3.js not loaded.</p>';return;}

  var ctr=document.getElementById('t5Wrap');
  var svg=d3.select('#t5Svg');
  var gMain=svg.append('g');
  var zm=d3.zoom().scaleExtent([0.1,3]).on('zoom',function(e){gMain.attr('transform',e.transform)});
  svg.call(zm);
  svg.on('dblclick.zoom',null);
  svg.call(zm.transform,d3.zoomIdentity.translate(140,ctr.clientHeight/2).scale(0.7));

  function childrenAccessor(d){
    if(!d.children||d.children.length===0)return null;
    return d.children.map(function(e){
      var ch=Object.assign({},e.child);
      if(e.child.children)ch.children=e.child.children;
      ch._edge={label:e.label,nice_label:e.nice_label,prob:e.prob,
        is_actual:e.is_actual,child_eu:e.child_eu,commentary:e.commentary};
      return ch;
    });
  }

  var root=d3.hierarchy(treeData,childrenAccessor);
  function walkAll(d,fn){fn(d);(d.children||d._collapsed||[]).forEach(function(c){walkAll(c,fn)})}
  walkAll(root,function(d){d._allChildren=d.children});
  walkAll(root,function(d){
    if(d.depth>=3&&d.children&&d.children.length>1){d._collapsed=d.children;d.children=null}
  });

  var treeLayout=d3.tree().nodeSize([NH+14,260]);

  function linkPath(s,t){
    return'M'+s.y+','+s.x+'C'+(s.y+t.y)/2+','+s.x+' '+(s.y+t.y)/2+','+t.x+' '+t.y+','+t.x;
  }
  function edgeLabel(d){
    var e=d.target.data._edge;if(!e)return'';
    var nice=(e.nice_label||'').replace(/\n/g,' ');
    if(e.prob!==null&&e.prob!==undefined)return nice+'  p='+(e.prob*100).toFixed(1)+'%';
    return nice;
  }

  function update(source){
    var dur=400;
    treeLayout(root);
    /* Links */
    var links=root.links();
    var lSel=gMain.selectAll('.lkg').data(links,function(d){return d.target.data.id});
    var lE=lSel.enter().append('g').attr('class','lkg');
    lE.append('path').attr('class','lk')
      .attr('d',function(){return linkPath(source,source)})
      .style('fill','none').style('stroke','#999').style('stroke-width','1.5').style('stroke-opacity','0.6');
    lE.append('path').attr('class','lkh')
      .attr('d',function(){return linkPath(source,source)})
      .style('fill','none').style('stroke','transparent').style('stroke-width','14').style('cursor','pointer')
      .on('mouseover',function(ev,d){showTT(ev,d)}).on('mousemove',function(ev){moveTT(ev)}).on('mouseout',hideTT);
    lE.append('text').attr('class','lkl').attr('dy',-4)
      .style('font-size','9px').style('fill','#555').style('pointer-events','none');
    var lM=lE.merge(lSel);
    lM.select('.lk').transition().duration(dur)
      .attr('d',function(d){return linkPath(d.source,d.target)})
      .style('stroke-width',function(d){
        var e=d.target.data._edge;
        if(e&&e.prob!==null&&e.prob!==undefined){if(e.prob<0.01)return'0.5';return Math.max(1,e.prob*4)+'';}
        return'1.5';
      })
      .style('stroke-opacity',function(d){
        var e=d.target.data._edge;
        if(e&&e.prob!==null&&e.prob<0.01)return'0.25';return'0.6';
      })
      .style('stroke',function(d){
        if(d.source.data.owner==='Board'&&d.source.data.type==='decision')return'#4A90D9';return'#999';
      });
    lM.select('.lkh').transition().duration(dur).attr('d',function(d){return linkPath(d.source,d.target)});
    lM.select('.lkl').transition().duration(dur)
      .attr('x',function(d){return(d.source.y+d.target.y)/2})
      .attr('y',function(d){return(d.source.x+d.target.x)/2})
      .text(function(d){return edgeLabel(d)})
      .style('fill',function(d){
        if(d.source.data.owner==='Board'&&d.source.data.type==='decision')return'#2C6FB5';return'#555';
      })
      .style('font-weight',function(d){
        if(d.source.data.owner==='Board'&&d.source.data.type==='decision')return'600';return'400';
      });
    lSel.exit().transition().duration(dur).style('opacity',0).remove();

    /* Nodes */
    var nodes=root.descendants();
    var nSel=gMain.selectAll('.ng').data(nodes,function(d){return d.data.id});
    var nE=nSel.enter().append('g').attr('class','ng').style('cursor','pointer')
      .attr('transform','translate('+(source.y0||0)+','+(source.x0||0)+')');
    nE.each(function(d){
      var el=d3.select(this),c=d.data.colour||'#888';
      if(d.data.type==='terminal'){
        el.append('rect').attr('x',-NW/2).attr('y',-NH/2).attr('width',NW).attr('height',NH)
          .attr('fill','transparent').attr('stroke','none');
        el.append('polygon')
          .attr('points','0,'+(-NH/2)+' '+(NW/2)+',0 0,'+(NH/2)+' '+(-NW/2)+',0')
          .attr('fill',c).attr('stroke',d3.color(c).darker(0.4)).attr('stroke-width','1.5')
          .style('pointer-events','none');
      }else if(d.data.type==='chance'){
        el.append('ellipse').attr('rx',NW/2).attr('ry',NH/2)
          .attr('fill',c).attr('stroke',d3.color(c).darker(0.4)).attr('stroke-width','1.5');
      }else{
        el.append('rect').attr('x',-NW/2).attr('y',-NH/2).attr('width',NW).attr('height',NH)
          .attr('rx',4).attr('ry',4)
          .attr('fill',c).attr('stroke',d3.color(c).darker(0.4)).attr('stroke-width','1.5');
      }
    });
    nE.on('pointerdown mousedown',function(ev){ev.stopPropagation()})
      .on('click',function(ev,d){ev.stopPropagation();toggle(d)})
      .on('mouseover',function(ev,d){showNP(d)})
      .on('mouseout',function(){scheduleHideNP()});
    nE.append('text').style('fill','#fff').style('font-size','10px')
      .style('text-anchor','middle').style('dominant-baseline','central').style('pointer-events','none');
    nE.append('circle').attr('class','ebg').attr('cx',NW/2+2).attr('cy',0).attr('r',9)
      .style('pointer-events','none');
    nE.append('text').attr('class','ebt').attr('x',NW/2+2).attr('dy',1)
      .style('font-size','10px').style('font-weight','600').style('fill','#fff')
      .style('text-anchor','middle').style('dominant-baseline','central').style('pointer-events','none');
    var nM=nE.merge(nSel);
    nM.transition().duration(dur).attr('transform',function(d){return'translate('+d.y+','+d.x+')'});
    nM.select('text:not(.ebt)').text(function(d){return d.data.name==='Terminal'?'T':d.data.name});
    function cntD(n){var c=0;(n._collapsed||n.children||[]).forEach(function(k){c+=1+cntD(k)});return c}
    nM.select('.ebg')
      .attr('fill',function(d){return(d._collapsed&&d._collapsed.length)?'#E8853D':'none'})
      .attr('stroke',function(d){return(d._collapsed&&d._collapsed.length)?'#C96E2A':'none'})
      .attr('stroke-width',1.5);
    nM.select('.ebt').text(function(d){
      if(!d._collapsed||!d._collapsed.length)return'';return'+'+cntD(d);
    });
    nSel.exit().transition().duration(dur)
      .attr('transform','translate('+source.y+','+source.x+')').style('opacity',0).remove();
    nodes.forEach(function(d){d.x0=d.x;d.y0=d.y});
  }

  function toggle(d){
    if(d.children){d._collapsed=d.children;d.children=null}
    else if(d._collapsed){d.children=d._collapsed;d._collapsed=null}
    update(d);
  }
  document.getElementById('t5Exp').onclick=function(){
    walkAll(root,function(d){if(d._collapsed){d.children=d._collapsed;d._collapsed=null}});update(root);
  };
  document.getElementById('t5Col').onclick=function(){
    walkAll(root,function(d){if(d.depth>=1&&d.children&&d.children.length>1){d._collapsed=d.children;d.children=null}});update(root);
  };
  document.getElementById('t5Fit').onclick=function(){
    var ns=root.descendants();if(!ns.length)return;
    var x0=Infinity,x1=-Infinity,y0=Infinity,y1=-Infinity;
    ns.forEach(function(d){x0=Math.min(x0,d.x-NH);x1=Math.max(x1,d.x+NH);y0=Math.min(y0,d.y-NW);y1=Math.max(y1,d.y+NW+80)});
    var tw=y1-y0,th=x1-x0,pd=40,aw=ctr.clientWidth-pd*2,ah=ctr.clientHeight-pd*2;
    var sc=Math.min(aw/tw,ah/th,3);
    var tx=pd-y0*sc+(aw-tw*sc)/2,ty=pd-x0*sc+(ah-th*sc)/2;
    svg.transition().duration(500).call(zm.transform,d3.zoomIdentity.translate(tx,ty).scale(sc));
  };

  /* Tooltip — shows EU and probabilities for all node types */
  var tt=document.getElementById('t5TT');
  function showTT(ev,d){
    var e=d.target.data._edge;if(!e)return;
    var nice=(e.nice_label||'').replace(/\n/g,' ');
    var h='<div style="font-weight:600;font-size:12px;color:#fff;margin-bottom:4px">'+nice+'</div>';
    if(e.prob!==null&&e.prob!==undefined)
      h+='<div style="display:flex;justify-content:space-between;gap:12px"><span style="color:#aaa">Probability:</span><span style="font-weight:500">'+(e.prob*100).toFixed(1)+'%</span></div>';
    if(td&&e.child_eu!==undefined&&e.child_eu!==0)
      h+='<div style="display:flex;justify-content:space-between;gap:12px"><span style="color:#aaa">EU (subtree):</span><span style="font-weight:500">'+e.child_eu.toFixed(3)+'</span></div>';
    var src=d.source.data;
    if(src.type==='decision'&&src.children){
      h+='<div style="border-top:1px solid rgba(255,255,255,0.15);margin:6px 0"></div>';
      h+='<div style="color:#aaa;margin-bottom:2px;font-size:10px">All actions at '+src.name+' ('+src.owner+'):</div>';
      src.children.forEach(function(c){
        var pStr=(c.prob!==null&&c.prob!==undefined)?(c.prob*100).toFixed(1)+'%':'?';
        h+='<div style="display:flex;justify-content:space-between;gap:12px"><span style="color:#aaa">'+(NE[c.label]||c.label)+'</span><span>'+pStr+'</span></div>';
      });
    }
    tt.innerHTML=h;tt.style.display='block';moveTT(ev);
  }
  function moveTT(ev){tt.style.left=(ev.clientX+14)+'px';tt.style.top=(ev.clientY-10)+'px'}
  function hideTT(){tt.style.display='none'}

  /* Node popup — shows EU and utility components on mouseover */
  var np=document.getElementById('t5NP');
  var npContent=document.getElementById('t5NPContent');
  var _npTimer=null;
  function showNP(d){
    if(_npTimer){clearTimeout(_npTimer);_npTimer=null}
    var nd=d.data;
    var col=COL[nd.owner]||'#888';
    var niceName=(NE[nd.name]||nd.name||'').replace(/_/g,' ');
    var h='<div style="margin-bottom:8px"><span style="font-weight:600;font-size:14px">'+niceName+'</span>'+
      ' <span style="display:inline-block;padding:2px 8px;border-radius:3px;color:#fff;font-size:11px;background:'+col+'">'+nd.owner+'</span>'+
      ' <span style="display:inline-block;padding:2px 8px;border-radius:3px;color:#fff;font-size:11px;background:#888">'+nd.type+'</span>'+
      ' <span style="font-size:12px;color:#666;margin-left:6px">EU = '+(nd.eu>=0?'+':'')+nd.eu.toFixed(4)+'</span></div>';
    var comp=nd.components;
    if(comp&&Object.keys(comp).length>0){
      h+='<div style="font-weight:600;font-size:13px;color:#444;margin:8px 0 4px">Utility Components</div>';
      h+='<table style="width:100%;border-collapse:collapse"><tr><th style="text-align:left;font-size:11px;color:#888;padding:4px 8px;border-bottom:2px solid #eee">Component</th>'+
        '<th style="text-align:right;font-size:11px;color:#888;padding:4px 8px;border-bottom:2px solid #eee">Value</th></tr>';
      var total=0;
      for(var k in comp){
        var v=comp[k];total+=v;
        var cls=v<-0.01?'color:#C0392B':v>0.01?'color:#27AE60':'';
        h+='<tr><td style="padding:4px 8px;border-bottom:1px solid #f0f0f0;font-size:12px">'+k.replace(/_/g,' ')+'</td>'+
          '<td style="padding:4px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;text-align:right;font-variant-numeric:tabular-nums;'+cls+'">'+(v>=0?'+':'')+v.toFixed(4)+'</td></tr>';
      }
      h+='<tr style="border-top:2px solid #999"><td style="padding:4px 8px;font-weight:600;font-size:12px">Total</td>'+
        '<td style="padding:4px 8px;font-weight:600;font-size:12px;text-align:right;font-variant-numeric:tabular-nums">'+(total>=0?'+':'')+total.toFixed(4)+'</td></tr>';
      h+='</table>';
    }
    if(nd.children&&nd.children.length>0){
      h+='<div style="font-weight:600;font-size:13px;color:#444;margin:10px 0 4px">Actions / Splits</div>';
      h+='<table style="width:100%;border-collapse:collapse"><tr><th style="text-align:left;font-size:11px;color:#888;padding:4px 8px;border-bottom:2px solid #eee">Action</th>'+
        '<th style="text-align:right;font-size:11px;color:#888;padding:4px 8px;border-bottom:2px solid #eee">Prob</th>'+
        '<th style="text-align:right;font-size:11px;color:#888;padding:4px 8px;border-bottom:2px solid #eee">EU</th></tr>';
      nd.children.forEach(function(e){
        var pStr=(e.prob!==null&&e.prob!==undefined)?(e.prob*100).toFixed(1)+'%':'?';
        h+='<tr><td style="padding:4px 8px;border-bottom:1px solid #f0f0f0;font-size:12px">'+(NE[e.label]||e.label.replace(/_/g,' '))+'</td>'+
          '<td style="padding:4px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;text-align:right">'+pStr+'</td>'+
          '<td style="padding:4px 8px;border-bottom:1px solid #f0f0f0;font-size:12px;text-align:right">'+(e.child_eu>=0?'+':'')+e.child_eu.toFixed(4)+'</td></tr>';
      });
      h+='</table>';
    }
    npContent.innerHTML=h;
    var svgEl=document.getElementById('t5Svg');
    var pt=svgEl.createSVGPoint();
    pt.x=d.y+NW/2+20;pt.y=d.x;
    var ctm=gMain.node().getCTM();
    var sp=pt.matrixTransform(ctm);
    var pw=420,maxH=window.innerHeight*0.7;
    var left=sp.x+10,top=sp.y-40;
    if(left+pw>window.innerWidth)left=sp.x-pw-30;
    if(top<50)top=50;
    if(top+maxH>window.innerHeight-10)top=window.innerHeight-maxH-10;
    np.style.left=left+'px';np.style.top=top+'px';
    np.style.maxHeight=maxH+'px';np.style.display='block';
  }
  function scheduleHideNP(){_npTimer=setTimeout(function(){np.style.display='none'},200)}
  np.addEventListener('mouseenter',function(){if(_npTimer){clearTimeout(_npTimer);_npTimer=null}});
  np.addEventListener('mouseleave',function(){scheduleHideNP()});

  update(root);
  /* Fit-to-screen when tab first becomes visible (panel is display:none at load) */
  var fitted=false;
  new MutationObserver(function(muts,obs){
    if(panel.classList.contains('active')&&!fitted){
      fitted=true;obs.disconnect();
      setTimeout(function(){document.getElementById('t5Fit').click()},50);
    }
  }).observe(panel,{attributes:true,attributeFilter:['class']});
})();


// Panel 6: Parameter Estimates
(function(){
  const p=document.getElementById('panel_6');
  const pe=D.parameter_estimates||{};
  if(!pe.weights){placeholder(6,'Parameter estimation not yet run.');return;}
  const w=pe.weights||{};
  const em=pe.estimation_method||{};
  const psd=pe.weights_posterior_sd||{};
  const pci=pe.weights_posterior_ci||{};
  const fs=D.feature_selection||{};
  const pnames=__PARAM_NAMES__;
  const pdescs=__PARAM_DESCS__;
  const specs=__SPEC_DEFAULTS__;
  const isStan=Object.values(em).some(v=>v==='stan_ordinal_probit');
  let html='<div class="card"><h3>Parameter Estimates</h3>';
  html+='<p style="margin-bottom:8px;color:var(--text-muted)">';
  if(isStan){
    html+='<span style="background:#d4edda;padding:2px 6px;border-radius:3px;font-size:0.85em">Important</span> Pr(relevant) \u2265 0.95. ';
    html+='<span style="background:#fff3cd;padding:2px 6px;border-radius:3px;font-size:0.85em">Marginal</span> 0.50 \u2264 Pr(relevant) < 0.95. ';
    html+='<span style="background:#f8d7da;padding:2px 6px;border-radius:3px;font-size:0.85em">Unimportant</span> Pr(relevant) < 0.50. ';
  }else{
    html+='<span style="background:#d4edda;padding:2px 6px;border-radius:3px;font-size:0.85em">Softmax MLE</span> = estimated via exp(\u03B8) reparameterized softmax. ';
    html+='<span style="background:#f8d7da;padding:2px 6px;border-radius:3px;font-size:0.85em">Excluded</span> = removed from model. ';
    html+='<span style="background:#fff3cd;padding:2px 6px;border-radius:3px;font-size:0.85em">Fixed</span> = fixed value.';
  }
  html+='</p>';
  if(isStan&&pe.n_samples){
    html+='<p style="margin-bottom:8px;color:var(--text-muted);font-size:0.9em">';
    html+='MCMC: '+pe.n_samples+' posterior draws | ';
    html+='Divergences: '+(pe.n_divergences||0)+' | ';
    html+='Max R\u0302: '+(pe.max_rhat||'--')+' | ';
    html+='Min ESS(bulk): '+(pe.min_ess_bulk||'--');
    html+='</p>';
  }
  const cvs=fs.cv||{};
  const shrk=fs.prior_shrinkage||{};
  const headers=isStan?
    ['Parameter','Method','Description','Prior Mean','Posterior Mean','Posterior SD','95% CI','<span title="Posterior probability that the utility weight exceeds 0.1 (practical significance threshold). Near 1.0 = parameter reliably contributes to Board utility. Near 0 = negligible contribution.">Pr(relevant)</span>','CV','<span title="Posterior shrinkage: 1 \u2212 Var(posterior)/Var(prior). Near 1.0 = data dominates (prior is uninformative). Near 0 = prior dominates (estimate is prior-sensitive). All priors are lognormal(\u03BC, \u03C3=1.0) centred at the spec default.">Prior</span>']:
    ['Parameter','Method','Description','Spec Default','Estimate','SE','95% CI','p-value','CV'];
  const relv=fs.relevance||{};
  /* In Stan mode, hide excluded and fixed params — only show estimated parameters */
  const displayNames=isStan?pnames.filter(pn=>{const m=em[pn]||'fixed';return m!=='excluded'&&m!=='fixed';}):pnames;
  const rows=displayNames.map(pn=>{
    const method=em[pn]||'fixed';
    const methodLabel=method==='stan_ordinal_probit'?'Bayesian':method==='anchored'?'Anchored':method==='excluded'?'Excluded':'Fixed';
    const est=w[pn]!==undefined?w[pn]:specs[pn];
    const sd=psd[pn]||0;
    const ci=pci[pn];
    const ciStr=ci?'['+ci[0].toFixed(2)+', '+ci[1].toFixed(2)+']':'--';
    const cv_val=cvs[pn]!=null?(cvs[pn]*100).toFixed(1)+'%':'--';
    const relvP=relv[pn]||{};
    const prVal=relvP.pr_gt_threshold!=null?relvP.pr_gt_threshold.toFixed(3):'--';
    const sh=shrk[pn];
    const priorCell=sh!=null?
      (sh>=0.5?'<span style="color:#155724" title="Shrinkage '+sh.toFixed(2)+': data dominates">\u2714 data-driven</span>':
               '<span style="color:#856404" title="Shrinkage '+sh.toFixed(2)+': prior dominates">\u26A0 prior-sensitive</span>'):'--';
    return [pn,methodLabel,pdescs[pn]||'',specs[pn]||'--',
      typeof est==='number'?est.toFixed(4):(est||'--'),
      typeof sd==='number'&&sd>0?sd.toFixed(4):'--',
      ciStr,prVal,cv_val,priorCell];
  });
  if(!isStan){
    rows.push(['lambda','Fixed','Rationality (inv. temp.)',1.0,
      (pe.lambda_rationality||1.0).toFixed(4),'--','--','--','--']);
  }
  html+=makeTable(headers,rows,'paramTable');
  if(isStan){
    html+='<p style="margin-top:8px;color:var(--text-muted)">Posterior covariance condition: '+
      (pe.condition_number||'--')+'</p>';
  }else{
    html+='<p style="margin-top:8px;color:var(--text-muted)">Condition number: '+
      (pe.condition_number||'--')+' | Ridge: '+(pe.ridge_applied?'Yes':'No')+'</p>';
  }
  html+='</div>';
  html+='<div class="card"><h3>Forest Plot (Posterior 95% Credible Intervals)</h3><div id="forestPlot" class="chart"></div></div>';
  p.innerHTML=html;
  // Style rows by estimation method and relevance
  const tbl=document.getElementById('paramTable');
  if(tbl){
    const trs=tbl.querySelectorAll('tbody tr, tr');
    trs.forEach(tr=>{
      const cells=tr.querySelectorAll('td');
      if(cells.length>1){
        const m=cells[1].textContent;
        const pn=cells[0].textContent;
        const relvP=relv[pn]||{};
        const prRel=relvP.pr_gt_threshold;
        if(m==='Bayesian'){
          if(prRel!=null&&prRel<0.5){
            tr.style.background='#f8d7da';tr.style.color='#721c24'; // red — unimportant
          }else if(prRel!=null&&prRel<0.95){
            tr.style.background='#fff3cd';tr.style.color='#856404'; // amber — marginal
          }else{
            tr.style.background='#d4edda';tr.style.color='#155724'; // green — important
          }
        }
        else if(m==='Anchored'){tr.style.background='#e2d9f3';tr.style.color='#4a235a';}
        else if(m==='Excluded'){tr.style.background='#f8d7da';tr.style.color='#721c24';}
        else if(m==='Fixed'){tr.style.background='#fff3cd';tr.style.color='#856404';}
      }
    });
  }
  if(typeof Plotly!=='undefined'){
    const allEst=pnames.filter(pn=>em[pn]==='stan_ordinal_probit');
    const vals=allEst.map(pn=>w[pn]||0);
    /* Use posterior CI directly if available, else fall back to 1.96*SD */
    const errLo=allEst.map(pn=>{
      const ci=pci[pn]; return ci?(w[pn]||0)-ci[0]:1.96*(psd[pn]||0);
    });
    const errHi=allEst.map(pn=>{
      const ci=pci[pn]; return ci?ci[1]-(w[pn]||0):1.96*(psd[pn]||0);
    });
    let xMax=3;
    const edges=allEst.map((pn,i)=>Math.max(Math.abs(vals[i])+errHi[i],Math.abs(vals[i])+errLo[i]));
    if(edges.length>0)xMax=Math.max(3,Math.ceil(Math.max(...edges)*1.2));
    Plotly.newPlot('forestPlot',[
      {y:allEst,x:vals,
       error_x:{type:'data',array:errHi,arrayminus:errLo,visible:true,thickness:2,width:4},
       type:'scatter',mode:'markers',name:'Posterior Mean (95% CI)',
       marker:{size:10,color:'#4A90D9',symbol:'circle'}},
      {y:allEst,x:allEst.map(()=>0),type:'scatter',mode:'lines',name:'Zero',
       line:{color:'#ccc',width:1,dash:'dot'},showlegend:false}
    ],{margin:{l:160,r:20,t:20,b:40},height:Math.max(400,allEst.length*45),
       xaxis:{title:'Weight (Posterior Mean \u00B1 95% CI)',zeroline:true,range:[-xMax,xMax]},
       legend:{x:0.7,y:1}},{responsive:true});
  }
})();

// Panel 7: Covariance
(function(){
  const p=document.getElementById('panel_7');
  const pe=D.parameter_estimates||{};
  const cov=D.covariance_matrix;
  if(!cov){placeholder(7,'Covariance matrix not yet computed.');return;}
  const estNames=__ESTIMABLE_PARAMS__;
  let html='<div class="card"><h3>Correlation Matrix (Estimated Parameters)</h3><div id="covHeatmap" class="chart"></div></div>';
  html+='<div class="card"><h3>Correlation Values</h3><div id="corrTable" style="overflow-x:auto"></div></div>';
  p.innerHTML=html;
  const tDiv=document.getElementById('corrTable');
  const n=cov.length;
  const corr=[];
  for(let i=0;i<n;i++){
    corr.push([]);
    for(let j=0;j<n;j++){
      const d=Math.sqrt(Math.abs(cov[i][i])*Math.abs(cov[j][j]));
      const r=d>1e-12?cov[i][j]/d:0;
      corr[i].push(Math.abs(r)<5e-3?0:r);
    }
  }
  const labels=estNames.concat(['lambda']).slice(0,n);
  const nLabels=labels.length;
  const corrSub=corr.slice(0,nLabels).map(r=>r.slice(0,nLabels));
  function fmtCorr(v){const s=v.toFixed(2);return s==='-0.00'?'0.00':s;}
  if(typeof Plotly!=='undefined'){
    Plotly.newPlot('covHeatmap',[{z:corrSub,x:labels,y:labels,type:'heatmap',
      colorscale:'RdBu',zmin:-1,zmax:1,reversescale:true,
      text:corrSub.map(r=>r.map(fmtCorr)),texttemplate:'%{text}',
      textfont:{size:10}}],
      {margin:{l:100,r:20,t:20,b:100},height:500},{responsive:true});
  }
  // Flag high correlations
  const highCorrs=[];
  for(let i=0;i<nLabels;i++){for(let j=i+1;j<nLabels;j++){
    if(Math.abs(corrSub[i][j])>0.8){highCorrs.push({a:labels[i],b:labels[j],r:corrSub[i][j]});}
  }}
  if(highCorrs.length>0){
    let warn='<div style="background:#fff3cd;border:1px solid #ffc107;padding:12px;border-radius:4px;margin-bottom:12px">';
    warn+='<strong>High Correlations Detected:</strong><ul style="margin:4px 0">';
    highCorrs.forEach(c=>{
      let note='';
      // Annotate structural correlations with explanations
      if((c.a==='w_removal'&&c.b==='w_remove_ceo_overwhelming')||(c.a==='w_remove_ceo_overwhelming'&&c.b==='w_removal')){
        note=' &mdash; <em>Structural (by design):</em> the Stan model defines <code>w_removal = w_remove_ceo_overwhelming + softplus(delta)</code>, so any posterior shift in w_remove_ceo_overwhelming mechanically shifts w_removal. The freely estimated parameter is <code>delta_removal</code> (irreducible removal cost that persists even after an overwhelming shareholder mandate), which is uncorrelated with w_remove_ceo_overwhelming by construction. This correlation does not indicate an identification problem; check the SE on <code>delta_removal</code> (theta[5]) instead.';
      }
      warn+='<li>'+c.a+' &harr; '+c.b+': <strong>'+c.r.toFixed(2)+'</strong>'+note+'</li>';
    });
    warn+='</ul><p style="margin:4px 0;font-size:0.9em">Correlations |r| &gt; 0.8 indicate parameter trade-offs in estimation. Structural correlations (from utility model design) do not indicate estimation problems if individual SEs are acceptable.</p></div>';
    tDiv.insertAdjacentHTML('beforebegin',warn);
  }
  // Structural note: w_inaction_base ↔ w_inaction_delay expected correlation
  {
    let iBase=-1,iDelay=-1;
    labels.forEach((l,k)=>{if(l==='w_inaction_base')iBase=k;if(l==='w_inaction_delay')iDelay=k;});
    if(iBase>=0&&iDelay>=0){
      const rVal=corrSub[iBase][iDelay];
      if(Math.abs(rVal)>0.5){
        let note='<div style="background:#e8f4fd;border:1px solid #bee5eb;padding:12px;border-radius:4px;margin-bottom:12px">';
        note+='<strong>Expected correlation: w_inaction_base &harr; w_inaction_delay (r = '+rVal.toFixed(2)+')</strong>';
        note+='<p style="margin:6px 0 0 0;font-size:0.93em">';
        note+='These parameters share a nested structure by design. <code>w_inaction_delay</code> fires only when the Board did nothing at D1 then acted reactively at D_rev (delayed governance). ';
        note+='<code>w_inaction_base</code> fires when the Board took minimal action at <em>all</em> decision points (total inaction). ';
        note+='Any scenario where w_inaction_base = &minus;1 (total inaction) is a subset of the conditions where delay <em>could</em> have fired but didn&rsquo;t (the Board never acted at all, so there is no delayed action). ';
        note+='This creates a structural positive correlation: scenarios that push w_inaction_base upward also tend to push w_inaction_delay upward, because both parameters respond to the same &ldquo;Board failed to act at D1&rdquo; condition. ';
        note+='The parameters remain separately identified because w_inaction_delay has a distinct phi pattern at D_rev: it fires [0, &minus;1, &minus;1] for [no_action, commission, sack] when d1 = D0_minimal, whereas w_inaction_base fires [&minus;1, 0, 0] at the same node. ';
        note+='This is not an estimation problem &mdash; check that individual SEs are acceptable.</p></div>';
        tDiv.insertAdjacentHTML('beforebegin',note);
      }
    }
  }
  // Numeric correlation table
  let thtml='<table style="border-collapse:collapse;font-size:12px;width:100%"><thead><tr><th style="border:1px solid #ddd;padding:4px"></th>';
  labels.forEach(l=>{thtml+='<th style="border:1px solid #ddd;padding:4px;writing-mode:vertical-rl;text-orientation:mixed;min-width:35px">'+l+'</th>';});
  thtml+='</tr></thead><tbody>';
  for(let i=0;i<nLabels;i++){
    thtml+='<tr><td style="border:1px solid #ddd;padding:4px;font-weight:bold;white-space:nowrap">'+labels[i]+'</td>';
    for(let j=0;j<nLabels;j++){
      const v=corrSub[i][j];
      const abs=Math.abs(v);
      let bg='#fff';
      if(i===j){bg='#e8e8e8';}
      else if(abs>0.8){bg=v>0?'#ff9999':'#9999ff';}
      else if(abs>0.5){bg=v>0?'#ffcccc':'#ccccff';}
      thtml+='<td style="border:1px solid #ddd;padding:4px;text-align:center;background:'+bg+'">'+fmtCorr(v)+'</td>';
    }
    thtml+='</tr>';
  }
  thtml+='</tbody></table>';
  tDiv.innerHTML=thtml;
})();

// Panel 8: Behavioural Diagnostics
(function(){
  const p=document.getElementById('panel_8');
  const bd=D.behavioural_diagnostics;
  if(!bd){placeholder(8,'Diagnostics not yet run.');return;}
  let html='';
  const tests=[
    ['loss_aversion','Loss Aversion (8.1)'],
    ['nonlinearity','Non-Linearity (8.2)'],
    ['optimism_bias','Optimism Bias (8.3)'],
    ['self_assessment_bias','Self-Assessment Bias (8.4)'],
    ['ikea_effect','Ikea Effect (8.5)'],
    ['action_order_effects','Action Order Effects (8.6)']
  ];
  tests.forEach(([key,title])=>{
    const t=bd[key]||{};
    const dec=t.decision||'--';
    const badge=dec==='confirmed'?'confirmed':dec==='null'?'null':'insufficient';
    html+='<div class="card"><h3>'+title+' <span class="badge '+badge+'">'+dec+'</span></h3>';
    if(t.p_value!==undefined) html+='<p>p-value: <strong>'+t.p_value+'</strong></p>';
    if(t.p_value_vs_1!==undefined) html+='<p>p vs 1.0: '+t.p_value_vs_1+' | p vs 2.25: '+(t.p_value_vs_kt||'--')+'</p>';
    if(t.mean_sensitivity_ratio!==undefined) html+='<p>Mean sensitivity ratio: '+t.mean_sensitivity_ratio+'</p>';
    if(t.effect_size!==undefined) html+='<p>Effect size: '+t.effect_size+'</p>';
    if(t.vote_penalty_aic){
      html+='<p>Vote penalty AIC: ';
      Object.entries(t.vote_penalty_aic).forEach(([k,v])=>html+=k+': '+v+' | ');
      html+='</p><p>Best form: <strong>'+(t.best_vote_form||'--')+'</strong></p>';
    }
    // Self-assessment bias detail
    if(key==='self_assessment_bias'){
      if(t.mean_score_sack_board_initiated!==undefined){
        html+='<table style="border-collapse:collapse;margin:8px 0"><thead><tr><th style="border:1px solid #ddd;padding:6px">Review Origin</th><th style="border:1px solid #ddd;padding:6px">Mean Likert (sack)</th></tr></thead><tbody>';
        html+='<tr><td style="border:1px solid #ddd;padding:6px">Board-initiated</td><td style="border:1px solid #ddd;padding:6px;text-align:center">'+t.mean_score_sack_board_initiated+'</td></tr>';
        html+='<tr><td style="border:1px solid #ddd;padding:6px">Externally mandated</td><td style="border:1px solid #ddd;padding:6px;text-align:center">'+t.mean_score_sack_externally_mandated+'</td></tr>';
        html+='</tbody></table>';
        if(t.t_stat!==undefined) html+='<p>t-statistic: '+t.t_stat+'</p>';
        html+='<p style="color:var(--text-muted);font-size:0.9em">Hypothesis: board-initiated reviews produce lower sack scores (self-serving bias — ownership of review process reduces willingness to act on adverse findings).</p>';
      } else {
        html+='<p style="color:#856404">Insufficient data: need multiple scenarios per review origin group.</p>';
      }
    }
    // Ikea effect detail
    if(key==='ikea_effect'){
      if(t.mean_score_sack_appointed!==undefined){
        html+='<table style="border-collapse:collapse;margin:8px 0"><thead><tr><th style="border:1px solid #ddd;padding:6px">CEO Appointment</th><th style="border:1px solid #ddd;padding:6px">Mean Likert (sack)</th></tr></thead><tbody>';
        html+='<tr><td style="border:1px solid #ddd;padding:6px">Appointed by current board</td><td style="border:1px solid #ddd;padding:6px;text-align:center">'+t.mean_score_sack_appointed+'</td></tr>';
        html+='<tr><td style="border:1px solid #ddd;padding:6px">Inherited from predecessor</td><td style="border:1px solid #ddd;padding:6px;text-align:center">'+t.mean_score_sack_inherited+'</td></tr>';
        html+='</tbody></table>';
        if(t.t_stat!==undefined) html+='<p>t-statistic: '+t.t_stat+'</p>';
        html+='<p style="color:var(--text-muted);font-size:0.9em">Hypothesis: boards are less likely to sack a CEO they appointed (IKEA effect — overvaluing own creation).</p>';
      } else {
        html+='<p style="color:#856404">Insufficient data: need multiple scenarios per appointment group.</p>';
      }
    }
    // Action order effects detail
    if(key==='action_order_effects'&&t.per_action){
      const pa=t.per_action;
      const aKeys=Object.keys(pa).sort();
      if(aKeys.length>0){
        html+='<table style="border-collapse:collapse;margin:8px 0;font-size:0.95em"><thead><tr>';
        html+='<th style="border:1px solid #ddd;padding:4px">Action</th><th style="border:1px solid #ddd;padding:4px">N</th>';
        html+='<th style="border:1px solid #ddd;padding:4px">Slope</th>';
        html+='<th style="border:1px solid #ddd;padding:4px">p-value</th><th style="border:1px solid #ddd;padding:4px">Effect</th></tr></thead><tbody>';
        aKeys.forEach(ak=>{
          const a=pa[ak];
          const eff=a.effect||'none';
          const effColor=eff==='none'?'#155724':eff==='primacy'?'#856404':'#721c24';
          html+='<tr><td style="border:1px solid #ddd;padding:4px">'+ak+'</td>';
          html+='<td style="border:1px solid #ddd;padding:4px;text-align:center">'+(a.n||'--')+'</td>';
          html+='<td style="border:1px solid #ddd;padding:4px;text-align:center">'+a.slope+'</td>';
          html+='<td style="border:1px solid #ddd;padding:4px;text-align:center">'+a.p_value+'</td>';
          html+='<td style="border:1px solid #ddd;padding:4px;text-align:center;color:'+effColor+'"><strong>'+eff+'</strong></td></tr>';
        });
        html+='</tbody></table>';
        html+='<p style="color:var(--text-muted);font-size:0.9em">Slope = change in Likert score per presentation position (1=first, N=last). Primacy: earlier-presented actions get higher scores. Recency: later-presented get higher. Action order is randomised per elicitation draw.</p>';
      }
      if(t.any_order_effect_detected!==undefined){
        html+='<p>Any order effect detected: <strong>'+(t.any_order_effect_detected?'Yes':'No')+'</strong></p>';
      }
    }
    html+='</div>';
  });
  p.innerHTML=html;
})();

// Panel 9: Interaction Effects
(function(){
  const p=document.getElementById('panel_9');
  const ie=D.interaction_effects;
  if(!ie||Object.keys(ie).length===0){placeholder(9,'Interaction effects not yet computed. Run Stage 6 first.');return;}
  let html='';

  // 1. Residual vs vote scatter
  html+='<div class="card"><h3>Model Fit vs Vote Percentage</h3><div id="residVote" class="chart"></div></div>';

  // 2. KL by decision node
  if(ie.kl_by_node){
    html+='<div class="card"><h3>Fit Quality by Decision Node</h3><div id="klByNode" class="chart"></div>';
    const kbn=ie.kl_by_node;
    html+='<table style="border-collapse:collapse;margin-top:8px"><thead><tr>';
    html+='<th style="border:1px solid #ddd;padding:6px">Node</th><th style="border:1px solid #ddd;padding:6px">N</th>';
    html+='<th style="border:1px solid #ddd;padding:6px">Mean KL</th><th style="border:1px solid #ddd;padding:6px">Median KL</th></tr></thead><tbody>';
    Object.entries(kbn).forEach(([node,v])=>{
      html+='<tr><td style="border:1px solid #ddd;padding:6px">'+node+'</td>';
      html+='<td style="border:1px solid #ddd;padding:6px;text-align:center">'+v.n+'</td>';
      html+='<td style="border:1px solid #ddd;padding:6px;text-align:center">'+v.mean_kl.toFixed(4)+'</td>';
      html+='<td style="border:1px solid #ddd;padding:6px;text-align:center">'+v.median_kl.toFixed(4)+'</td></tr>';
    });
    html+='</tbody></table></div>';
  }

  // 3. Strike/CEO interaction table
  if(ie.strike_ceo_interaction){
    html+='<div class="card"><h3>Strike × CEO Presence: Fit Quality</h3>';
    const sci=ie.strike_ceo_interaction;
    html+='<table style="border-collapse:collapse"><thead><tr>';
    html+='<th style="border:1px solid #ddd;padding:6px">Condition</th><th style="border:1px solid #ddd;padding:6px">N</th>';
    html+='<th style="border:1px solid #ddd;padding:6px">Mean KL</th><th style="border:1px solid #ddd;padding:6px">Mean |Resid|</th></tr></thead><tbody>';
    Object.entries(sci).forEach(([k,v])=>{
      const label=k.replace('strike=0','No Strike').replace('strike=1','Strike').replace('_ceo=0',', CEO Removed').replace('_ceo=1',', CEO Present');
      html+='<tr><td style="border:1px solid #ddd;padding:6px">'+label+'</td>';
      html+='<td style="border:1px solid #ddd;padding:6px;text-align:center">'+v.n+'</td>';
      html+='<td style="border:1px solid #ddd;padding:6px;text-align:center">'+v.mean_kl.toFixed(4)+'</td>';
      html+='<td style="border:1px solid #ddd;padding:6px;text-align:center">'+v.mean_max_resid.toFixed(4)+'</td></tr>';
    });
    html+='</tbody></table></div>';
  }

  // 4. Statistical tests
  html+='<div class="card"><h3>Model Fit Heterogeneity Tests</h3>';
  let hasTests=false;
  if(ie.strike_fit_test&&ie.strike_fit_test.p_value!==undefined){
    hasTests=true;
    const sf=ie.strike_fit_test;
    html+='<p><strong>Strike effect on fit:</strong> Mean KL (strike)='+sf.mean_kl_strike.toFixed(4)+
      ', Mean KL (no strike)='+sf.mean_kl_no_strike.toFixed(4)+
      ', p='+sf.p_value+' — '+sf.conclusion+'</p>';
  }
  if(ie.overwhelming_fit_test&&ie.overwhelming_fit_test.p_value!==undefined){
    hasTests=true;
    const oft=ie.overwhelming_fit_test;
    html+='<p><strong>Overwhelming effect on fit:</strong> Mean KL (overwhelming)='+oft.mean_kl_overwhelming.toFixed(4)+
      ', Mean KL (not)='+oft.mean_kl_not_overwhelming.toFixed(4)+
      ', p='+oft.p_value+' — '+oft.conclusion+'</p>';
  }
  if(!hasTests){html+='<p style="color:#856404">Insufficient data: need &ge;3 scenarios in each subgroup (strike/no-strike, overwhelming/not). Re-run pipeline to generate.</p>';}
  html+='<p style="color:var(--text-muted);font-size:0.9em">Mann-Whitney U tests comparing model fit (KL divergence) across scenario subgroups. Significant results indicate the model fits differently in certain conditions, suggesting missing interaction terms.</p>';
  html+='</div>';

  // 5. Worst fitting scenarios table
  if(ie.worst_fitting&&ie.worst_fitting.length>0){
    html+='<div class="card"><h3>10 Worst-Fitting Scenarios</h3>';
    html+='<table style="border-collapse:collapse;font-size:0.95em"><thead><tr>';
    html+='<th style="border:1px solid #ddd;padding:4px">ID</th><th style="border:1px solid #ddd;padding:4px">Node</th>';
    html+='<th style="border:1px solid #ddd;padding:4px">Vote</th><th style="border:1px solid #ddd;padding:4px">Strike</th>';
    html+='<th style="border:1px solid #ddd;padding:4px">Ovw</th><th style="border:1px solid #ddd;padding:4px">CEO</th>';
    html+='<th style="border:1px solid #ddd;padding:4px">KL</th><th style="border:1px solid #ddd;padding:4px">Max |R|</th></tr></thead><tbody>';
    ie.worst_fitting.forEach(w=>{
      html+='<tr><td style="border:1px solid #ddd;padding:4px">'+w.scenario_id+'</td>';
      html+='<td style="border:1px solid #ddd;padding:4px">'+w.node+'</td>';
      html+='<td style="border:1px solid #ddd;padding:4px;text-align:center">'+w.vote+'</td>';
      html+='<td style="border:1px solid #ddd;padding:4px;text-align:center">'+(w.strike?'Y':'N')+'</td>';
      html+='<td style="border:1px solid #ddd;padding:4px;text-align:center">'+(w.overwhelming?'Y':'N')+'</td>';
      html+='<td style="border:1px solid #ddd;padding:4px;text-align:center">'+(w.ceo_present?'Y':'N')+'</td>';
      html+='<td style="border:1px solid #ddd;padding:4px;text-align:center">'+w.kl.toFixed(4)+'</td>';
      html+='<td style="border:1px solid #ddd;padding:4px;text-align:center">'+w.max_abs_resid+'</td></tr>';
    });
    html+='</tbody></table></div>';
  }

  p.innerHTML=html;

  // Render plots
  if(typeof Plotly!=='undefined'){
    // Residual vs vote scatter
    if(ie.resid_vs_vote){
      const rv=ie.resid_vs_vote;
      Plotly.newPlot('residVote',[{
        x:rv.vote,y:rv.kl,text:rv.scenario_ids,
        type:'scatter',mode:'markers',name:'KL Divergence',
        marker:{size:6,color:'#4A90D9'},
        hovertemplate:'%{text}<br>Vote: %{x:.2f}<br>KL: %{y:.4f}<extra></extra>'
      }],{margin:{t:20,b:50,l:60,r:20},height:350,
        xaxis:{title:'Vote Percentage'},yaxis:{title:'KL Divergence'}},{responsive:true});
    }
    // KL by node box plot
    if(ie.kl_by_node){
      const traces=Object.entries(ie.kl_by_node).map(([node,v])=>({
        y:v.kl_values,type:'box',name:node,boxpoints:'all',jitter:0.3,pointpos:-1.5,
        marker:{size:4}
      }));
      Plotly.newPlot('klByNode',traces,{margin:{t:20,b:50,l:60,r:20},height:350,
        yaxis:{title:'KL Divergence'}},{responsive:true});
    }
  }
})();

// Panel 10: Validation
(function(){
  const p=document.getElementById('panel_10');
  const vr=D.validation_results;
  if(!vr){placeholder(10,'Validation not yet run.');return;}
  let html='';
  if(vr.within_sample_kl){
    const kl=vr.within_sample_kl;
    html+='<div class="card"><h3>Within-Sample Fit</h3><p>Mean KL: <strong>'+
      kl.mean+'</strong> (target &lt; 0.05: '+(kl.meets_target?'MET':'NOT MET')+')</p></div>';
  }
  if(vr.worst_5_scenarios){
    html+='<div class="card"><h3>5 Worst-Fitting Scenarios</h3>';
    const headers=['Scenario ID','KL Divergence'];
    const rows=vr.worst_5_scenarios.map(s=>[s.scenario_id,s.kl.toFixed(6)]);
    html+=makeTable(headers,rows,'worstTable')+'</div>';
  }
  if(vr.historical_prediction&&vr.historical_prediction.available){
    const hp=vr.historical_prediction;
    html+='<div class="card"><h3>Historical Scenario Prediction (Tier 4)</h3>';
    html+='<p>Top predicted action: <strong>'+hp.top_action+'</strong> (p='+hp.top_prob+')</p>';
    html+='<p>Rank of D1_review: '+(hp.rank_of_D1_review||'--')+'</p>';
    if(hp.predicted_probs){
      html+='<div id="histChart" class="chart"></div>';
    }
    html+='</div>';
    if(typeof Plotly!=='undefined'&&hp.predicted_probs){
      setTimeout(()=>{
        const actions=Object.keys(hp.predicted_probs);
        const probs=Object.values(hp.predicted_probs);
        Plotly.newPlot('histChart',[{x:actions,y:probs,type:'bar',
          marker:{color:actions.map(a=>a==='D1_review'?'#50C878':'#4A90D9')}}],
          {margin:{t:20,b:60,l:60,r:20},height:300,
           yaxis:{title:'Predicted Probability'}},{responsive:true});
      },100);
    }
  }
  p.innerHTML=html;
})();

// Panel 11: Linearity Diagnostics
(function(){
  const p=document.getElementById('panel_11');
  const ie=D.interaction_effects;
  const pe=D.parameter_estimates||{};
  const cov=D.covariance_matrix;
  if(!ie||!ie.resid_vs_vote){placeholder(11,'Linearity diagnostics require Stage 6. Run stages 4-6 first.');return;}
  let html='';

  // 1. Residual vs predicted EU — linearity check
  html+='<div class="card"><h3>Residuals vs Vote Percentage</h3>';
  html+='<p style="color:var(--text-muted);font-size:0.9em">If the linear-in-phi model is correct, residuals should scatter randomly around zero with no systematic pattern. Curvature or trends suggest missing non-linear terms.</p>';
  html+='<div id="linResidVote" class="chart"></div></div>';

  // 2. QQ plot of max residuals
  html+='<div class="card"><h3>Residual Distribution (Q-Q Plot)</h3>';
  html+='<p style="color:var(--text-muted);font-size:0.9em">Comparing residual distribution to normal. Heavy tails suggest outlier scenarios; skew suggests systematic bias.</p>';
  html+='<div id="linQQ" class="chart"></div></div>';

  // 3. Scale-location plot: sqrt(|resid|) vs vote
  html+='<div class="card"><h3>Scale-Location: Heteroscedasticity Check</h3>';
  html+='<p style="color:var(--text-muted);font-size:0.9em">Sqrt of |max residual| vs vote. A flat trend indicates homoscedastic errors (constant variance). An increasing trend suggests the model fits worse at certain vote levels.</p>';
  html+='<div id="linScaleLoc" class="chart"></div></div>';

  // 4. Phi collinearity - VIF table
  html+='<div class="card"><h3>Phi Basis Function Summary</h3>';
  html+='<p style="color:var(--text-muted);font-size:0.9em">The ordinal probit model is linear in phi: latent utility = phi(s,a) &middot; w + anchored. Each row shows the phi basis function for one parameter. High correlation between phi columns creates collinearity (see Covariance tab).</p>';
  const pdescs=__PARAM_DESCS__;
  const pnames=__PARAM_NAMES__;
  const fixed=new Set(__FIXED_PARAMS__);
  const estP=pnames.filter(pn=>!fixed.has(pn));
  const phiDefs={
    'w_inaction_base':'-I[board inactive at all decision points]',
    'w_inaction_no_review':'-I[no governance review commissioned]',
    'w_passivity':'-I[CEO resigned early] × (1 - response_strength)',
    'w_removal':'-I[CEO removed involuntarily]',
    'w_remove_ceo_overwhelming':'+I[CEO removed] × I[overwhelming]',
    'w_review_negative':'-I[review commissioned ∧ review negative]','w_review_balanced':'-I[review commissioned ∧ review balanced]',
    'w_review_post_removal':'-I[CEO removed involuntarily ∧ no review commissioned]',
    'w_ceo_accountability':'+I[CEO removed involuntarily ∧ review commissioned]',
    'w_strike':'-w × max(0,(V-0.25)/0.75) [scenario-level]',
    'w_overwhelming':'-w × max(0,(V-0.50)/0.50) [scenario-level]'
  };
  html+='<table style="border-collapse:collapse;font-size:0.95em"><thead><tr>';
  html+='<th style="border:1px solid #ddd;padding:4px">Parameter</th>';
  html+='<th style="border:1px solid #ddd;padding:4px">Phi Basis</th>';
  html+='<th style="border:1px solid #ddd;padding:4px">Varies Across</th>';
  html+='<th style="border:1px solid #ddd;padding:4px">Type</th></tr></thead><tbody>';
  estP.forEach(pn=>{
    const em=(pe.estimation_method||{})[pn]||'';
    const varies=(em==='softmax_mle'||em==='stan_ordinal_probit')?'Actions (within scenario)':'Scenarios only';
    html+='<tr><td style="border:1px solid #ddd;padding:4px;font-weight:bold">'+pn+'</td>';
    html+='<td style="border:1px solid #ddd;padding:4px;font-family:monospace;font-size:0.9em">'+(phiDefs[pn]||'--')+'</td>';
    html+='<td style="border:1px solid #ddd;padding:4px">'+varies+'</td>';
    html+='<td style="border:1px solid #ddd;padding:4px">'+em.replace('_',' ')+'</td></tr>';
  });
  html+='</tbody></table></div>';

  p.innerHTML=html;

  // Render plots
  if(typeof Plotly!=='undefined'&&ie.resid_vs_vote){
    const rv=ie.resid_vs_vote;
    // Max abs residual vs vote (directional)
    Plotly.newPlot('linResidVote',[{
      x:rv.vote,y:rv.max_abs_resid,text:rv.scenario_ids,
      type:'scatter',mode:'markers',name:'|Max Residual|',
      marker:{size:6,color:rv.kl.map(k=>k>0.1?'#dc3545':'#4A90D9')},
      hovertemplate:'%{text}<br>Vote: %{x:.2f}<br>|Resid|: %{y:.4f}<extra></extra>'
    },{
      x:[0,1],y:[0,0],type:'scatter',mode:'lines',name:'Zero',
      line:{color:'#ccc',dash:'dot'},showlegend:false
    }],{margin:{t:20,b:50,l:60,r:20},height:350,
      xaxis:{title:'Vote Percentage'},yaxis:{title:'Max |Residual|'}},{responsive:true});

    // QQ plot
    const sorted_r=rv.max_abs_resid.slice().sort((a,b)=>a-b);
    const n_r=sorted_r.length;
    const theoretical=[];
    for(let i=0;i<n_r;i++){
      const p_i=(i+0.5)/n_r;
      // Approximate normal quantile (Beasley-Springer-Moro)
      const t_q=p_i<0.5?Math.sqrt(-2*Math.log(p_i)):Math.sqrt(-2*Math.log(1-p_i));
      const z=(2.515517+t_q*(0.802853+t_q*0.010328))/(1+t_q*(1.432788+t_q*(0.189269+t_q*0.001308)));
      theoretical.push(p_i<0.5?-(t_q-z):(t_q-z));
    }
    Plotly.newPlot('linQQ',[{
      x:theoretical,y:sorted_r,type:'scatter',mode:'markers',name:'Residuals',
      marker:{size:5,color:'#4A90D9'}
    },{
      x:[Math.min(...theoretical),Math.max(...theoretical)],
      y:[Math.min(...sorted_r),Math.max(...sorted_r)],
      type:'scatter',mode:'lines',name:'45° line',line:{color:'#ccc',dash:'dot'}
    }],{margin:{t:20,b:50,l:60,r:20},height:350,
      xaxis:{title:'Theoretical Quantiles (Normal)'},yaxis:{title:'Sample Quantiles (|Residual|)'}},{responsive:true});

    // Scale-location
    const sqrtR=rv.max_abs_resid.map(r=>Math.sqrt(Math.abs(r)));
    Plotly.newPlot('linScaleLoc',[{
      x:rv.vote,y:sqrtR,text:rv.scenario_ids,
      type:'scatter',mode:'markers',name:'√|Residual|',
      marker:{size:6,color:'#50C878'},
      hovertemplate:'%{text}<br>Vote: %{x:.2f}<br>√|R|: %{y:.4f}<extra></extra>'
    }],{margin:{t:20,b:50,l:60,r:20},height:350,
      xaxis:{title:'Vote Percentage'},yaxis:{title:'√|Max Residual|'}},{responsive:true});
  }
})();

// Panel 12: Raw Data
(function(){
  const p=document.getElementById('panel_12');
  const of=D.output_files||{};
  let html='<div class="card"><h3>Output Files</h3>';
  if(Object.keys(of).length===0){
    html+='<p>No output files available yet.</p>';
  } else {
    Object.entries(of).forEach(([name,data])=>{
      html+='<a class="download-link" href="'+data.uri+'" download="'+name+'">'+
        name+' ('+(data.size_kb||'?')+'KB)</a> ';
    });
  }
  html+='</div>';
  p.innerHTML=html;
})();
</script>
</body>
</html>"""


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


def render_dashboard(
    dashboard_data: DashboardData,
    output_path: Path,
) -> None:
    """Render the self-contained HTML dashboard."""
    dashboard_data.generated_at = datetime.now().isoformat()

    # Get Plotly bundle
    plotly_bundle = _get_plotly_bundle()
    if plotly_bundle:
        plotly_script = f"<script>{plotly_bundle}</script>"
    else:
        plotly_script = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'

    # Smart refresh for in-progress: JS-based with countdown, tab memory,
    # and change detection (only reloads if file content changed).
    meta_refresh = ""
    if dashboard_data.run_status == "in_progress":
        meta_refresh = """<script>
(function(){
  var REFRESH_SEC=120;
  var remaining=REFRESH_SEC;
  var contentHash=null;
  // Save active tab before unload
  window.addEventListener('beforeunload',function(){
    var tabs=document.querySelectorAll('.tab');
    for(var i=0;i<tabs.length;i++){
      if(tabs[i].classList.contains('active')){sessionStorage.setItem('dashTabIdx',i);break;}
    }
  });
  // Restore active tab on load
  window.addEventListener('DOMContentLoaded',function(){
    var idx=sessionStorage.getItem('dashTabIdx');
    if(idx!==null){
      var tabs=document.querySelectorAll('.tab');
      if(tabs[idx])tabs[idx].click();
    }
    // Countdown bar
    var bar=document.createElement('div');
    bar.id='refreshBar';
    bar.style.cssText='position:fixed;top:0;left:0;right:0;z-index:9999;'+
      'background:#e8f4fd;border-bottom:1px solid #b8daff;padding:4px 16px;'+
      'font-size:13px;color:#004085;display:flex;align-items:center;gap:12px';
    var txt=document.createElement('span');
    txt.id='refreshTxt';
    txt.textContent='Next refresh in '+remaining+'s';
    var prog=document.createElement('div');
    prog.style.cssText='flex:1;height:4px;background:#c8e6ff;border-radius:2px;overflow:hidden';
    var fill=document.createElement('div');
    fill.id='refreshFill';
    fill.style.cssText='height:100%;background:#4A90D9;border-radius:2px;width:100%;'+
      'transition:width 1s linear';
    prog.appendChild(fill);
    bar.appendChild(txt);bar.appendChild(prog);
    document.body.prepend(bar);
    // Add top padding so bar doesn't overlap content
    document.body.style.paddingTop='32px';
    // Compute hash of initial content for comparison
    contentHash=simpleHash(document.documentElement.outerHTML);
    // Start countdown
    setInterval(function(){
      remaining--;
      if(remaining<=0){
        remaining=REFRESH_SEC;
        checkAndRefresh();
      }
      document.getElementById('refreshTxt').textContent='Next refresh in '+remaining+'s';
      document.getElementById('refreshFill').style.width=(remaining/REFRESH_SEC*100)+'%';
    },1000);
  });
  function simpleHash(s){
    var h=0;for(var i=0;i<s.length;i++){h=((h<<5)-h)+s.charCodeAt(i);h|=0;}return h;
  }
  function checkAndRefresh(){
    // Fetch current file; if content changed, reload
    fetch(window.location.href,{cache:'no-store'})
      .then(function(r){return r.text()})
      .then(function(html){
        var newHash=simpleHash(html);
        if(newHash!==contentHash){
          // Save tab before reload
          var tabs=document.querySelectorAll('.tab');
          for(var i=0;i<tabs.length;i++){
            if(tabs[i].classList.contains('active')){sessionStorage.setItem('dashTabIdx',i);break;}
          }
          location.reload();
        }
        // else: file unchanged, skip reload — timer resets automatically
      })
      .catch(function(){location.reload();}); // on error, reload anyway
  }
})();
</script>"""

    # Build results JSON
    results_json = dashboard_data.to_json()

    # Inject constants for JS
    # ALL_WEIGHT_NAMES for display (estimable + anchored + excluded)
    param_names_json = json.dumps(list(ALL_WEIGHT_NAMES), ensure_ascii=True)
    param_descs_json = json.dumps(PARAM_DESCRIPTIONS, ensure_ascii=True)
    spec_defaults_json = json.dumps(SPEC_DEFAULTS, ensure_ascii=True)
    fixed_params_json = json.dumps([], ensure_ascii=True)
    estimable_params_json = json.dumps(list(ESTIMABLE_PARAM_NAMES) + VOTE_PARAM_NAMES, ensure_ascii=True)

    html = _DASHBOARD_TEMPLATE
    html = html.replace("__META_REFRESH__", meta_refresh)
    html = html.replace("__PLOTLY_SCRIPT__", plotly_script)
    html = html.replace("__RESULTS_DATA__", results_json)
    html = html.replace("__PARAM_NAMES__", param_names_json)
    html = html.replace("__PARAM_DESCS__", param_descs_json)
    html = html.replace("__SPEC_DEFAULTS__", spec_defaults_json)
    html = html.replace("__FIXED_PARAMS__", fixed_params_json)
    html = html.replace("__ESTIMABLE_PARAMS__", estimable_params_json)

    # Atomic write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp")
    tmp_path.write_text(html, encoding="utf-8", errors="replace")
    os.replace(str(tmp_path), str(output_path))

    logger.info(f"Dashboard written to {output_path} ({len(html)//1024}KB)")


# ── SEC 12: main() orchestrator + CLI ─────────────────────────────────────────

class _TqdmLoggingHandler(logging.Handler):
    """Logging handler that routes output through tqdm.write() to avoid
    corrupting progress bars.  Per spec Section 12.9: 'all logging must
    use tqdm.write() rather than print() or logging.StreamHandler directly.'"""

    def emit(self, record):
        try:
            from tqdm import tqdm
            msg = self.format(record)
            tqdm.write(msg, file=sys.stderr)
        except Exception:
            self.handleError(record)


def _setup_logging(log_path: Optional[Path] = None):
    """Configure logging with tqdm-safe handler."""
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    # Console handler — uses tqdm.write() so log lines never overwrite the bar
    ch = _TqdmLoggingHandler()
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(ch)

    # Fully suppress all openai/httpx console messages.
    for noisy in ("httpx", "openai", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.CRITICAL)

    if log_path:
        fh = logging.FileHandler(str(log_path), encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(logging.DEBUG)
        root.addHandler(fh)


def _save_parameter_estimates(est_result: EstimationResult, output_dir: Path):
    """Save parameter_estimates.csv and covariance_matrix.csv."""
    rows = []
    em = est_result.estimation_method
    psd = getattr(est_result, "weights_posterior_sd", {})
    pci = getattr(est_result, "weights_posterior_ci", {})

    # Estimated params (Bayesian ordinal probit via Stan)
    for p in WEIGHT_PARAM_NAMES + VOTE_PARAM_NAMES:
        ci = pci.get(p, (None, None))
        rows.append({
            "parameter": p,
            "status": em.get(p, "stan_ordinal_probit"),
            "engine_key": PARAM_TO_ENGINE_KEY.get(p, ""),
            "description": PARAM_DESCRIPTIONS.get(p, ""),
            "spec_default": SPEC_DEFAULTS.get(p, ""),
            "estimate": est_result.weights.get(p, ""),
            "posterior_sd": psd.get(p, ""),
            "ci_lower": ci[0] if ci and ci[0] is not None else "",
            "ci_upper": ci[1] if ci and ci[1] is not None else "",
        })
    # (Deprecated params w2, w3, w4, w_inaction, w13, w8r, w8s, w9, w12
    #  and lambda_rationality removed — all parameters are now Bayesian-estimated)

    pd.DataFrame(rows).to_csv(
        output_dir / "parameter_estimates.csv", index=False, encoding="utf-8"
    )

    # Covariance matrix
    cov = est_result.covariance_matrix
    labels = list(WEIGHT_PARAM_NAMES) + VOTE_PARAM_NAMES + ["log_lambda"]
    n = min(cov.shape[0], len(labels))
    cov_df = pd.DataFrame(
        cov[:n, :n],
        columns=labels[:n],
        index=labels[:n],
    )
    cov_df.to_csv(output_dir / "covariance_matrix.csv", encoding="utf-8")

    # Save posterior draws for --stage 4b reload
    np.savez_compressed(
        output_dir / "stan_posterior_draws.npz",
        w_draws=est_result.w_draws,
        cutpoint_draws=est_result.cutpoint_draws,
        sigma_scenario_draws=est_result.sigma_scenario_draws,
        covariance_matrix=est_result.covariance_matrix,
    )
    logger.info(f"Saved posterior draws to {output_dir / 'stan_posterior_draws.npz'}")


def _load_cached_stan_result(output_dir: Path) -> StanEstimationResult:
    """Reconstruct StanEstimationResult from cached files (for --stage 4b).

    Loads posterior draws from stan_posterior_draws.npz and summary
    statistics from parameter_estimates.csv.
    """
    draws_path = output_dir / "stan_posterior_draws.npz"
    csv_path = output_dir / "parameter_estimates.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"No cached parameter estimates at {csv_path}. "
            f"Run --stage 4 first to generate them."
        )

    # Load summary statistics from CSV
    est_df = pd.read_csv(csv_path, encoding="utf-8")
    weights = {}
    weights_sd = {}
    weights_ci = {}
    estimation_method = {}

    for _, row in est_df.iterrows():
        p = row["parameter"]
        status = row.get("status", "")
        if status == "excluded" or p == "lambda_rationality":
            continue
        est_val = row.get("estimate", "")
        if est_val == "" or est_val == "excluded":
            continue
        weights[p] = float(est_val)
        sd_val = row.get("posterior_sd", "")
        if sd_val != "":
            weights_sd[p] = float(sd_val)
        ci_lo = row.get("ci_lower", "")
        ci_hi = row.get("ci_upper", "")
        if ci_lo != "" and ci_hi != "":
            weights_ci[p] = (float(ci_lo), float(ci_hi))
        estimation_method[p] = str(status)

    if draws_path.exists():
        # Load exact posterior draws from .npz cache
        data = np.load(draws_path)
        w_draws = data["w_draws"]
        cutpoint_draws = data["cutpoint_draws"]
        sigma_scenario_draws = data["sigma_scenario_draws"]
        covariance_matrix = data["covariance_matrix"]
        logger.info(f"Loaded cached posterior draws: {w_draws.shape[0]} draws, "
                    f"{w_draws.shape[1]} params from {draws_path}")
    else:
        # Simulate draws from posterior summary (lognormal since all w > 0)
        logger.warning(f"No .npz cache at {draws_path} — simulating draws "
                       f"from posterior mean/SD in {csv_path}")
        n_sim = 4000
        rng = np.random.default_rng(42)
        # w_draws columns must match WEIGHT_PARAM_NAMES (= ESTIMABLE_PARAM_NAMES)
        # Vote params are separate in the Stan fit and not in w_draws.
        K = len(WEIGHT_PARAM_NAMES)
        w_draws = np.zeros((n_sim, K))
        for j, p in enumerate(WEIGHT_PARAM_NAMES):
            mu = weights.get(p, SPEC_DEFAULTS.get(p, 1.0))
            sd = weights_sd.get(p, mu * 0.3)  # fallback: 30% CV
            if mu > 0 and sd > 0:
                # Lognormal parameterised from mean & SD
                var = sd ** 2
                sigma2 = np.log1p(var / (mu ** 2))
                mu_ln = np.log(mu) - 0.5 * sigma2
                w_draws[:, j] = rng.lognormal(mu_ln, np.sqrt(sigma2), size=n_sim)
            else:
                w_draws[:, j] = mu
        cutpoint_draws = np.zeros((n_sim, 4))
        sigma_scenario_draws = np.ones(n_sim) * 0.5
        # Build covariance from CSV if available
        cov_path = output_dir / "covariance_matrix.csv"
        if cov_path.exists():
            cov_df = pd.read_csv(cov_path, index_col=0, encoding="utf-8")
            covariance_matrix = cov_df.values
        else:
            n_cov = len(WEIGHT_PARAM_NAMES) + len(VOTE_PARAM_NAMES) + 1
            covariance_matrix = np.eye(n_cov)
        logger.info(f"Simulated {n_sim} draws for {K} params from posterior summaries")

    result = StanEstimationResult(
        w_draws=w_draws,
        cutpoint_draws=cutpoint_draws,
        sigma_scenario_draws=sigma_scenario_draws,
        weights_posterior_mean=weights,
        weights_posterior_sd=weights_sd,
        weights_posterior_ci=weights_ci,
        weights=weights,
        covariance_matrix=covariance_matrix,
        n_samples=w_draws.shape[0],
        converged=True,
        estimation_method=estimation_method,
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Board Utility Quantification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python board_utility_quantification.py --stage 1\n"
            "  python board_utility_quantification.py --stage 1,2,3 --n_draws 5\n"
            "  python board_utility_quantification.py --all --n_draws 10\n"
        ),
    )
    parser.add_argument("--stage", type=str, default="all",
                        help="Comma-separated stages to run (1-6, 4b) or 'all'. "
                             "4b loads cached Stan draws and recomputes tree + dashboard.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="LLM model for elicitation (default: gpt-4o-mini)")
    parser.add_argument("--n_draws", type=int, default=50,
                        help="Draws per scenario for elicitation (default: 50)")
    parser.add_argument("--chains", type=int, default=4,
                        help="MCMC chains for Stan estimation (default: 4)")
    parser.add_argument("--iter_warmup", type=int, default=1000,
                        help="Warmup iterations per chain (default: 1000)")
    parser.add_argument("--iter_sampling", type=int, default=2000,
                        help="Sampling iterations per chain (default: 2000)")
    parser.add_argument("--api_key", type=str, default=None,
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR),
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--llm-threads", type=int, default=10, dest="llm_threads",
                        help="Concurrent threads for LLM elicitation (default: 10)")
    parser.add_argument("--all", action="store_true",
                        help="Run all stages")
    parser.add_argument("--no-laplacian", action="store_true", dest="no_laplacian",
                        help="Disable Laplacian smoothing on Board decision "
                             "probabilities in the tree (default: enabled)")
    parser.add_argument("--softmax-tree", action="store_true", dest="softmax_tree",
                        default=True,
                        help="Use per-draw softmax for Board action probabilities "
                             "in the decision tree, showing how parameter uncertainty "
                             "spreads probability across actions. (default: enabled)")
    parser.add_argument("--argmax-tree", action="store_true", dest="argmax_tree",
                        help="Use argmax-count (instead of softmax) for Board "
                             "action probabilities in the decision tree.")

    args = parser.parse_args()

    # Parse stages
    run_4b = False
    if args.all or args.stage == "all":
        stages = {1, 2, 3, 4, 5, 6}
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

    # Setup logging
    log_path = output_dir / "pipeline.log"
    _setup_logging(log_path)

    logger.info("=" * 60)
    logger.info("Board Utility Quantification Pipeline")
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

    # Encoding self-test
    if not _run_encoding_self_test():
        logger.error("Encoding self-test failed. Aborting.")
        sys.exit(1)

    # Dashboard state
    dashboard = DashboardData(
        run_start=datetime.now().isoformat(),
        model=args.model,
    )
    dashboard_path = output_dir / "board_utility_dashboard.html"

    # File paths
    scenarios_path = output_dir / "scenarios.csv"
    elicitation_path = output_dir / "elicitation_results.csv"
    estimation_path = output_dir / "estimation_dataset.csv"
    diagnostics_path = output_dir / "behavioural_diagnostics.csv"

    cost_tracker = RunCostSummary()
    scenarios = []
    estimation_df = pd.DataFrame()
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
            # Pre-flight checks (post-generation)
            preflight_gen = run_preflight_checks(scenarios)
            dashboard.preflight_checks = preflight_gen
            if not preflight_gen["all_passed"]:
                logger.warning("Pre-flight checks have failures — review scenario design")
            render_dashboard(dashboard, dashboard_path)

        # ── Stage 2: LLM elicitation ──
        if 2 in stages and scenarios:
            # Check if we can skip elicitation entirely
            needed_ids = set(s.scenario_id for s in scenarios if s.tier != 4)
            skip_elicitation = False
            if elicitation_path.exists():
                existing_df = pd.read_csv(elicitation_path, encoding="utf-8")
                existing_ids = set(existing_df["scenario_id"].unique())
                # Detect stale format: old pipeline wrote action_scores as empty {}
                has_likert_data = False
                if "action_scores" in existing_df.columns:
                    sample = existing_df["action_scores"].dropna().head(5)
                    for val in sample:
                        try:
                            d = json.loads(val) if isinstance(val, str) else val
                            if isinstance(d, dict) and len(d) > 0:
                                has_likert_data = True
                                break
                        except (json.JSONDecodeError, TypeError):
                            pass
                if not has_likert_data:
                    logger.warning("Existing elicitation CSV uses old format (no Likert scores). "
                                   "Re-running elicitation with new Likert schema.")
                elif not (needed_ids - existing_ids):
                    logger.info(f"Elicitation results already cover all "
                                f"{len(needed_ids)} scenarios "
                                f"({len(existing_df)} rows). Skipping Stage 2. "
                                f"Use --stage 2 to force re-run.")
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
            # Elicitation summary
            if elicitation_path.exists():
                edf = pd.read_csv(elicitation_path, encoding="utf-8")
                n_total = len(edf)
                n_success = len(edf[edf["parse_status"].isin(["success", "repaired"])])
                status_counts = edf["parse_status"].value_counts().to_dict()
                dashboard.elicitation_summary = {
                    "total_calls": n_total,
                    "success_rate": round(100 * n_success / max(n_total, 1), 1),
                    "cache_hit_rate": round(
                        100 * _cache_stats["hits"] / max(_cache_stats["hits"] + _cache_stats["misses"], 1), 1
                    ),
                    "parse_status_counts": status_counts,
                }
            render_dashboard(dashboard, dashboard_path)

        # ── Stage 3: Preprocessing ──
        likert_long_path = output_dir / "likert_long.csv"
        likert_summary_path = output_dir / "likert_summary.csv"
        if 3 in stages and elicitation_path.exists():
            likert_long_df_stage3, likert_summary_df = preprocess_likert_data(
                elicitation_path, likert_long_path, likert_summary_path,
            )
            # estimation_df: keep for backward-compat dashboard panels that read it
            # from disk; if estimation_path exists from a prior run, load it too.
            if estimation_path.exists():
                estimation_df = pd.read_csv(estimation_path, encoding="utf-8")
            if not likert_summary_df.empty:
                dashboard.estimation_dataset_summary = {
                    "n_scenarios": likert_summary_df["scenario_id"].nunique(),
                    "mean_seed_variance": 0.0,  # not applicable to Likert pipeline
                }
            render_dashboard(dashboard, dashboard_path)
        elif likert_summary_path.exists():
            likert_summary_df = pd.read_csv(likert_summary_path, encoding="utf-8")
            if estimation_path.exists():
                estimation_df = pd.read_csv(estimation_path, encoding="utf-8")

        # Build elicited Likert score data for dashboard
        if not likert_summary_df.empty and scenarios:
            scenario_lookup = {s.scenario_id: s for s in scenarios}
            ep_rows = []
            for sid, grp in likert_summary_df.groupby("scenario_id"):
                sc = scenario_lookup.get(sid)
                if sc is None:
                    continue
                sv = sc.state_vector if isinstance(sc.state_vector, dict) else {}
                mean_scores = {
                    row["action"]: round(float(row["mean_score"]), 2)
                    for _, row in grp.iterrows()
                }
                ep_rows.append({
                    "scenario_id": sid,
                    "tier": sc.tier,
                    "target_parameter": sc.target_parameter,
                    "decision_node": sc.decision_node,
                    "vote_pct": sv.get("vote_outcome_V"),
                    "mean_scores": mean_scores,
                    "n_draws": int(grp["n_draws"].iloc[0]) if "n_draws" in grp.columns else 0,
                    "d1_action": sv.get("d1_action", ""),
                    "ceo_status": sv.get("ceo_status_at_start", "present"),
                    "strike": sv.get("strike", False),
                    "overwhelming": sv.get("overwhelming", False),
                    "review_outcome": sv.get("review_outcome"),
                    "ceo_present_at_end": sv.get("ceo_present_at_end", True),
                })
            dashboard.elicited_probabilities = ep_rows
            render_dashboard(dashboard, dashboard_path)

        # Pre-flight checks (post-elicitation, with data)
        if not estimation_df.empty and scenarios:
            preflight_elic = run_preflight_checks(scenarios, estimation_df)
            dashboard.preflight_checks = preflight_elic
            if not preflight_elic["all_passed"]:
                logger.warning("Post-elicitation pre-flight checks have failures")
            render_dashboard(dashboard, dashboard_path)

        # ── Stage 4: Parameter estimation ──
        if 4 in stages and not likert_summary_df.empty and scenarios:
            phi, anchored, sa_id_map, scenario_id_map, scenario_ids, action_lists, vote_data = (
                compute_phi_matrix(scenarios, likert_summary_df)
            )

            # Load likert_long_df from Stage 3 output (or from variable if available)
            if "likert_long_df_stage3" in dir() and not likert_long_df_stage3.empty:
                likert_long_df = likert_long_df_stage3
            elif likert_long_path.exists():
                likert_long_df = pd.read_csv(likert_long_path, encoding="utf-8")
            else:
                raise RuntimeError("Stage 4 requires likert_long.csv from Stage 3")

            est_result = estimate_parameters_stan(
                phi, anchored, likert_long_df, sa_id_map, scenario_id_map,
                vote_data=vote_data,
                chains=args.chains,
                iter_warmup=args.iter_warmup,
                iter_sampling=args.iter_sampling,
            )

            _save_parameter_estimates(est_result, output_dir)

            # ── Feature selection (posterior relevance) ──
            feature_sel = run_feature_selection(est_result)
            dashboard.feature_selection = feature_sel
            if feature_sel["excluded_params"]:
                logger.warning(f"Feature selection flagged {len(feature_sel['excluded_params'])} "
                               f"params for low relevance: {feature_sel['excluded_params']}")

            # ── Posterior action probabilities (Step 7) ──
            posterior_action_probs = compute_action_probabilities_from_posterior(
                est_result, scenarios, phi, anchored,
                sa_id_map, action_lists, scenario_ids,
            )
            dashboard.posterior_action_probs = posterior_action_probs

            # ── Recursive EU tree for dashboard (Step 7b) ──
            tree_data = compute_recursive_tree(
                est_result, laplacian=not args.no_laplacian,
                board_softmax=not args.argmax_tree)
            dashboard.tree_data = tree_data

            dashboard.parameter_estimates = est_result.to_dict()
            cov = est_result.covariance_matrix
            n = min(cov.shape[0], len(WEIGHT_PARAM_NAMES) + 1)
            dashboard.covariance_matrix = cov[:n, :n].tolist()
            render_dashboard(dashboard, dashboard_path)

        # ── Stage 4b: Reload cached Stan draws → recompute tree + dashboard ──
        if run_4b:
            logger.info("Stage 4b: Loading cached Stan posterior draws...")
            est_result = _load_cached_stan_result(output_dir)

            # Feature selection (uses only est_result)
            feature_sel = run_feature_selection(est_result)
            dashboard.feature_selection = feature_sel

            # Recursive EU tree
            tree_data = compute_recursive_tree(
                est_result, laplacian=not args.no_laplacian,
                board_softmax=not args.argmax_tree)
            dashboard.tree_data = tree_data

            dashboard.parameter_estimates = est_result.to_dict()
            cov = est_result.covariance_matrix
            n = min(cov.shape[0], len(WEIGHT_PARAM_NAMES) + 1)
            dashboard.covariance_matrix = cov[:n, :n].tolist()
            render_dashboard(dashboard, dashboard_path)
            logger.info("Stage 4b complete — tree + dashboard updated from cached draws.")

        # ── Stage 5: Behavioural diagnostics ──
        if 5 in stages and not likert_summary_df.empty and est_result is not None:
            diagnostics = run_diagnostics(
                scenarios, likert_summary_df, est_result,
                elicitation_path, diagnostics_path,
            )
            dashboard.behavioural_diagnostics = diagnostics
            render_dashboard(dashboard, dashboard_path)

        # ── Stage 6: Validation ──
        if 6 in stages and est_result is not None and not likert_summary_df.empty:
            # Recompute phi if needed
            if "phi" not in dir():
                phi, anchored, sa_id_map, scenario_id_map, scenario_ids, action_lists, vote_data = (
                    compute_phi_matrix(scenarios, likert_summary_df)
                )

            validation = run_validation(
                scenarios, estimation_df, est_result,
                phi, anchored, sa_id_map,
                scenario_ids, action_lists, output_dir,
                likert_summary_df=likert_summary_df if not likert_summary_df.empty else None,
            )
            dashboard.validation_results = validation

            # Compute interaction effects from scenario fit
            fit_csv = output_dir / "scenario_fit.csv"
            if fit_csv.exists():
                fit_df_int = pd.read_csv(fit_csv, encoding="utf-8")
                fit_rows_list = fit_df_int.to_dict("records")
                interaction = _compute_interaction_effects(
                    scenarios, estimation_df, est_result, fit_rows_list,
                )
                dashboard.interaction_effects = interaction

            render_dashboard(dashboard, dashboard_path)

        # ── Embed output files for download ──
        output_files = {}
        for fname, mime in [
            ("scenarios.csv", "text/csv"),
            ("elicitation_results.csv", "text/csv"),
            ("estimation_dataset.csv", "text/csv"),
            ("parameter_estimates.csv", "text/csv"),
            ("covariance_matrix.csv", "text/csv"),
            ("scenario_fit.csv", "text/csv"),
            ("behavioural_diagnostics.csv", "text/csv"),
            ("validation_results.json", "application/json"),
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
        logger.info("Dashboard finalised -- safe to share.")
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
