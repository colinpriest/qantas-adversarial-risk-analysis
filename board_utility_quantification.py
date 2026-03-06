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
# Parameters whose phi is constant across all actions within a scenario
# have zero gradient (shift-invariance) and CANNOT be identified from
# choice data.  They are fixed at spec defaults.
#
# FIXED (scenario-level, unidentifiable from softmax):
#   w1  — CEO resigned early: same for all actions in a scenario
#   w2  — vote penalty: depends on V (scenario-level, not action-level)
#   w3  — overwhelming indicator: scenario-level
#   w4  — spill risk (V × strike): scenario-level
#   w9  — reputational spill (overwhelming): scenario-level
#
# ESTIMABLE (action-varying phi):
#   w_removal, w8s, w8o, w8r — depend on whether action removes CEO
#   w_inaction — depends on whether action leaves CEO in place post-strike
#   w12, w13 — depend on cumulative Board inaction (board_inactive flag)
#   w15 — depends on CEO presence at end (action can remove)
#
# Collinear groups collapsed for identification:
#   w7 + w8 → w_removal  (both fire when CEO involuntarily removed)
#   w10 + w11 + w14 → w_inaction  (all = strike AND ceo_present_at_end)
FIXED_PARAM_NAMES = ["w1", "w2", "w3", "w4", "w9"]

ESTIMABLE_PARAM_NAMES = [
    "w_removal", "w8s", "w8o", "w8r",
    "w_inaction", "w12", "w13", "w15",
]

# All weight parameters (fixed + estimable) — for display/decomposition
ALL_WEIGHT_NAMES = FIXED_PARAM_NAMES + ESTIMABLE_PARAM_NAMES

# For estimation: only the action-varying parameters + lambda
FREE_PARAM_NAMES = ESTIMABLE_PARAM_NAMES + ["lambda_rationality"]
WEIGHT_PARAM_NAMES = ESTIMABLE_PARAM_NAMES  # 8 weights estimated

PARAM_TO_ENGINE_KEY = {
    "w1": "early_ceo_departure_cost",
    "w2": "vote_penalty_weight",
    "w3": "overwhelming_penalty_weight",
    "w4": "spill_risk_weight",
    "w_removal": "implementation_cost_sack + ceo_loss_cost",
    "w8s": "ceo_loss_shock_strike",
    "w8o": "ceo_loss_shock_overwhelming",
    "w8r": "ceo_loss_shock_adverse",
    "w9": "reputational_spill_weight",
    "w_inaction": "second_strike_spill + regulatory_liability + legal_d_rev",
    "w12": "continued_inaction_liability_overwhelming",
    "w13": "continued_inaction_liability_strike",
    "w15": "adverse_review_ceo_present_penalty",
}

PARAM_DESCRIPTIONS = {
    "w1": "Early CEO departure cost",
    "w2": "Vote penalty weight (quadratic)",
    "w3": "Overwhelming vote penalty",
    "w4": "Spill risk (strike x V)",
    "w_removal": "CEO removal cost (implementation + disruption)",
    "w8s": "CEO removal shock relief (strike)",
    "w8o": "CEO removal shock relief (overwhelming)",
    "w8r": "CEO removal shock relief (adverse review)",
    "w9": "Reputational spill (overwhelming)",
    "w_inaction": "Inaction penalty (strike + CEO retained: spill + liability + legal)",
    "w12": "Continued inaction liability (overwhelming + cumulative minimal response)",
    "w13": "Continued inaction liability (strike + cumulative minimal response)",
    "w15": "Adverse review + CEO present penalty",
    "lambda_rationality": "Rationality (inverse temperature)",
}

# Spec defaults: collapsed = sum of constituent engine parameters
SPEC_DEFAULTS = {
    "w1": 0.5, "w2": 2.0, "w3": 3.0, "w4": 2.5,
    "w_removal": 1.8,   # w7(0.3) + w8(1.5)
    "w8s": 0.4, "w8o": 0.5, "w8r": 0.5,
    "w9": 1.0,
    "w_inaction": 15.0,  # w10(8.0) + w11(5.0) + w14(2.0)
    "w12": 4.0, "w13": 3.0,
    "w15": 5.0, "lambda_rationality": 1.0,
}

CACHE_VERSION = 3

MODEL_PRICE_TABLE = {
    "gpt-4o-mini": {"prompt_per_1k": 0.00015, "completion_per_1k": 0.00060},
    "gpt-4o": {"prompt_per_1k": 0.0025, "completion_per_1k": 0.01},
}

FACTOR_DESCRIPTIONS = [
    "Risk of a second strike at the next AGM",
    "Personal regulatory liability of individual directors (ASIC)",
    "Corporate legal exposure (class actions, ASIC company penalties)",
    "CEO relationship and institutional knowledge loss",
    "Market reaction to governance action",
    "Direct costs of governance reform",
    "Reputational contagion to directors' other board positions",
    "Implementation complexity of the chosen action",
    "Shareholder activist escalation risk",
    "Board cohesion and internal deliberation costs",
]

# Factor → parameter mapping for Stage 4B (factor rating regression).
# Each scenario-level parameter is identified through the factor(s) that
# capture the LLM's perceived importance of the corresponding scenario feature.
# Factor indices are 1-based, matching FACTOR_DESCRIPTIONS above.
FACTOR_PARAM_MAP = {
    "w1": {"factors": [4], "phi_label": "CEO_resigned_early",
           "description": "CEO early departure cost → F4 (CEO relationship/knowledge loss)"},
    "w2": {"factors": [1], "phi_label": "(V-0.25)^2",
           "description": "Vote penalty → F1 (second strike risk)"},
    "w3": {"factors": [9, 10], "phi_label": "overwhelming",
           "description": "Overwhelming penalty → F9 (activist escalation) + F10 (board cohesion)"},
    "w4": {"factors": [1, 5], "phi_label": "V*strike",
           "description": "Spill risk → F1 (second strike) + F5 (market reaction)"},
    "w9": {"factors": [7], "phi_label": "overwhelming_reputation",
           "description": "Reputational spill → F7 (reputational contagion to other boards)"},
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
    PROBABILITY_ERROR = "probability_error"
    TOKEN_LIMIT = "token_limit"
    REPAIRED = "repaired"


FEASIBLE_ACTIONS_MAP = {
    "D1": [ActionCode.D0_MINIMAL, ActionCode.D1_REVIEW, ActionCode.D3_CEO_TRANSITION],
    "D_rev": [ActionCode.DREV_NO_ACTION, ActionCode.DREV_COMMISSION_REVIEW, ActionCode.DREV_SACK_CEO],
    "D_rev_post": [ActionCode.DREV_NO_ACTION, ActionCode.DREV_SACK_CEO],
}


class ActionProbability(BaseModel):
    action: ActionCode
    probability: float
    justification: str

    @field_validator("probability")
    @classmethod
    def prob_range(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError(f"Probability {v} out of [0, 1]")
        return round(v, 4)


class FactorRating(BaseModel):
    factor_index: int
    rating: int

    @field_validator("factor_index")
    @classmethod
    def valid_index(cls, v: int) -> int:
        if v < 1 or v > 10:
            raise ValueError(f"Factor index {v} not in [1, 10]")
        return v

    @field_validator("rating")
    @classmethod
    def valid_rating(cls, v: int) -> int:
        if v < 1 or v > 5:
            raise ValueError(f"Rating {v} not in [1, 5]")
        return v


class ElicitationResponse(BaseModel):
    prob_vector: list[ActionProbability]
    factor_ratings: list[FactorRating]
    commentary: str

    @model_validator(mode="after")
    def check_constraints(self) -> "ElicitationResponse":
        total = sum(ap.probability for ap in self.prob_vector)
        if abs(total - 1.0) > 0.02:
            raise ValueError(f"Probabilities sum to {total:.4f}, not 1.0")
        # Renormalise within tolerance
        if abs(total - 1.0) > 1e-6:
            for ap in self.prob_vector:
                ap.probability = round(ap.probability / total, 4)
        indices = sorted(fr.factor_index for fr in self.factor_ratings)
        if len(self.factor_ratings) != 10 or indices != list(range(1, 11)):
            raise ValueError(f"Factor ratings must cover indices 1-10, got {indices}")
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
        "cache_version": CACHE_VERSION,
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
                         scenario_id: str, max_retries: int = 6
                         ) -> tuple[Optional[ElicitationResponse], dict]:
    """Call LLM with retry logic. Returns (parsed_response, raw_metadata)."""
    import openai

    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create_with_completion(
                model=model,
                response_model=ElicitationResponse,
                messages=messages,
                max_retries=3,
                temperature=1.0,
            )
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
        "raise competing concerns, debate trade-offs, and work toward a majority position. The "
        "probability output should reflect the Board's likely collective decision, accounting for "
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
        "3. Rate each of the following factors on a 1-5 scale (1 = not significant, "
        "5 = decisive). IMPORTANT: Rate them in the ORDER PRESENTED below, which varies "
        "per query:\n"
        "{factor_list}\n"
        "4. After deliberation, assign a probability to each feasible action (must sum to 1.00). "
        "For each action, provide a one-sentence justification from the Board's perspective.\n\n"
        "Respond with a JSON object matching this schema:\n"
        '{{\n'
        '  "prob_vector": [\n'
        '    {{"action": "<ACTION_CODE>", "probability": <0.00-1.00>, "justification": "<text>"}},\n'
        '    ...\n'
        '  ],\n'
        '  "factor_ratings": [\n'
        '    {{"factor_index": <1-10>, "rating": <1-5>}},\n'
        '    ...\n'
        '  ],\n'
        '  "commentary": "<free-form deliberation text>"\n'
        '}}\n'
    )


def _format_factor_list(order: list[int]) -> str:
    """Format factor list in the given presentation order."""
    lines = []
    for pos, idx in enumerate(order, 1):
        lines.append(f"  Factor {idx}: {FACTOR_DESCRIPTIONS[idx - 1]}")
    return "\n".join(lines)


def _make_state_vector(
    decision_node: str = "D1",
    ceo_status: str = "present",
    ceo_appointment: str = "appointed_by_current_board",
    d1_action: str = "D0_minimal",
    review_origin: str = "N/A",
    vote_outcome: float = 0.0,
    review_commissioned: bool = False,
    review_adverse: Optional[bool] = None,
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
        "review_adverse": review_adverse,
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
        lines.append("The CEO remains in position. The Board must decide on governance actions.")

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

    # D1 action taken (if at D_rev or later)
    if node in ("D_rev", "D_rev_post"):
        d1 = sv["d1_action"]
        if d1 == "D1_review":
            lines.append("The Board previously decided to commission an independent "
                          "governance review.")
        elif d1 == "D3_ceo_transition":
            lines.append("The Board previously initiated a CEO transition process.")
        else:
            lines.append("The Board previously took minimal governance action (no review, "
                          "no CEO transition).")

    # Review status
    if sv["review_commissioned"]:
        origin = sv.get("review_origin", "board_initiated")
        if origin == "externally_mandated":
            lines.append("ASIC has mandated an independent governance review of Qantas.")
        else:
            lines.append("The Board commissioned an independent governance review.")

        if sv["review_adverse"] is True:
            lines.append("The review has concluded with ADVERSE findings: significant "
                          "governance failures in executive accountability, risk oversight, "
                          "and stakeholder management were identified.")
            if sv["car_outcome"] is not None:
                car_bps = int(sv["car_outcome"] * 10000)
                if car_bps < 0:
                    lines.append(f"The market reacted negatively to the findings release, "
                                  f"with an abnormal return of {car_bps} basis points.")
                elif car_bps > 0:
                    lines.append(f"The market reacted positively to the findings release, "
                                  f"with an abnormal return of +{car_bps} basis points, "
                                  f"suggesting investors view the governance action favourably.")
        elif sv["review_adverse"] is False:
            lines.append("The review concluded with POSITIVE findings: governance practices "
                          "were found to be adequate with minor recommendations.")

    # Explicit adverse probability (for optimism bias scenarios)
    if sv.get("explicit_adverse_prob"):
        lines.append(f"Based on comparable ASX governance reviews, approximately "
                      f"{int(sv['explicit_adverse_prob'] * 100)}% of reviews have produced "
                      f"adverse or neutral findings.")

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
                f"inaction at this decision point are compounding: {items_str}"
            )

    # CEO retention risk assessment (adapts to shock state)
    if sv["ceo_status_at_start"] != "resigned_early":
        shocks = []
        if sv["strike"]:
            shocks.append("first strike")
        if sv["overwhelming"]:
            shocks.append("overwhelming vote")
        if sv.get("review_adverse") is True:
            shocks.append("adverse review findings")

        if len(shocks) >= 2:
            shock_str = ", ".join(shocks[:-1]) + " and " + shocks[-1]
            lines.append(f"\nCEO RETENTION RISK ASSESSMENT: The combination of {shock_str} "
                          "places this situation in the highest-severity category. Based on "
                          "ASX precedent (AMP, Crown Resorts, Rio Tinto), boards that retained "
                          "the CEO after multiple governance shocks of this magnitude subsequently "
                          "lost director seats, faced personal regulatory proceedings, and incurred "
                          "greater total costs than boards that acted decisively after the first shock. "
                          "The transition cost of CEO removal must be weighed against the compounding "
                          "cost of retention under these conditions.")
        elif len(shocks) == 1:
            lines.append(f"\nCEO RETENTION RISK ASSESSMENT: The {shocks[0]} represents a significant "
                          "governance shock. While CEO removal carries transition costs (strategy "
                          "disruption, search costs, ~12 months of underperformance), retention "
                          "after this shock carries escalating risks: potential second strike, "
                          "regulatory scrutiny of Board inaction, and shareholder class action "
                          "exposure. The Board should weigh whether the CEO's continued presence "
                          "helps or hinders the credible resolution of governance concerns.")
        else:
            lines.append("\nCEO RETENTION CONTEXT: No severe governance shocks have occurred. "
                          "CEO removal at this stage would carry full transition costs (strategy "
                          "disruption, investor uncertainty, 12-month average underperformance) "
                          "without the governance-failure justification that would offset those "
                          "costs in the eyes of regulators and shareholders.")

    # Decision point
    feasible_strs = ", ".join(scenario.feasible_actions)
    if node == "D1":
        lines.append(f"\nThe Board must now decide on its governance response. "
                      f"Feasible actions: {feasible_strs}.")
        lines.append("- D0_minimal: Maintain current governance arrangements with minimal changes")
        lines.append("- D1_review: Commission an independent governance review")
        lines.append("- D3_ceo_transition: Initiate CEO transition (remove and replace)")
    elif node == "D_rev":
        lines.append(f"\nThe Board must now decide on its post-AGM response. "
                      f"Feasible actions: {feasible_strs}.")
        lines.append("- Drev_no_action: Take no further governance action")
        lines.append("- Drev_commission_review: Commission an independent governance review")
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

    # w1: Early CEO departure — CEO resigned vs present
    # Need sufficient CEO-resigned scenarios for factor rating regression.
    # Pair at low vote (no strike)
    for ceo_status in ["present", "resigned_early"]:
        n += 1
        scenarios.append(Scenario(
            scenario_id=f"S1_{n:03d}",
            tier=1,
            target_parameter="w1",
            decision_node="D1",
            state_vector=_make_state_vector(
                decision_node="D1",
                ceo_status=ceo_status,
                vote_outcome=0.10,
                ceo_present_at_end=(ceo_status == "present"),
            ),
            feasible_actions=["D0_minimal", "D1_review", "D3_ceo_transition"],
        ))
    # CEO resigned at varied vote levels (strike, overwhelming) for robust w1 estimation
    for v in [0.30, 0.40, 0.55, 0.65, 0.83]:
        n += 1
        scenarios.append(Scenario(
            scenario_id=f"S1_{n:03d}",
            tier=1,
            target_parameter="w1",
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
            target_parameter="w1",
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

    # w2: Vote penalty — V varies, CEO removed (eliminates strike-CEO-present)
    for v in [0.26, 0.30, 0.40, 0.50, 0.60]:
        n += 1
        scenarios.append(Scenario(
            scenario_id=f"S1_{n:03d}",
            tier=1,
            target_parameter="w2",
            decision_node="D1",
            state_vector=_make_state_vector(
                decision_node="D1",
                vote_outcome=v,
                ceo_present_at_end=False,
            ),
            feasible_actions=["D0_minimal", "D1_review", "D3_ceo_transition"],
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

    # w15: Adverse review + CEO present penalty
    # Contrast pairs: adverse vs positive review with CEO present → identifies w15
    # (Positive review = w15 doesn't fire → lower P(sack) expected)
    for adverse in [True, False]:
        n += 1
        scenarios.append(Scenario(
            scenario_id=f"S1_{n:03d}",
            tier=1,
            target_parameter="w15",
            decision_node="D_rev_post",
            state_vector=_make_state_vector(
                decision_node="D_rev_post",
                d1_action="D1_review",
                vote_outcome=0.35,
                review_commissioned=True,
                review_adverse=adverse,
                car_outcome=-0.03 if adverse else 0.02,
                ceo_present_at_end=True,
            ),
            feasible_actions=["Drev_no_action", "Drev_sack_ceo"],
        ))
    # Same contrast at higher vote
    for adverse in [True, False]:
        n += 1
        scenarios.append(Scenario(
            scenario_id=f"S1_{n:03d}",
            tier=1,
            target_parameter="w15",
            decision_node="D_rev_post",
            state_vector=_make_state_vector(
                decision_node="D_rev_post",
                d1_action="D1_review",
                vote_outcome=0.50,
                review_commissioned=True,
                review_adverse=adverse,
                car_outcome=-0.05 if adverse else 0.03,
                ceo_present_at_end=True,
            ),
            feasible_actions=["Drev_no_action", "Drev_sack_ceo"],
        ))
    # w15 with overwhelming vote + adverse
    n += 1
    scenarios.append(Scenario(
        scenario_id=f"S1_{n:03d}",
        tier=1,
        target_parameter="w15",
        decision_node="D_rev_post",
        state_vector=_make_state_vector(
            decision_node="D_rev_post",
            d1_action="D1_review",
            vote_outcome=0.60,
            review_commissioned=True,
            review_adverse=True,
            car_outcome=-0.05,
            ceo_present_at_end=True,
        ),
        feasible_actions=["Drev_no_action", "Drev_sack_ceo"],
    ))

    # w_inaction contrast: strike + CEO present vs strike + CEO removed
    # CEO present: w_inaction fires for Drev_no_action → penalty for inaction
    # CEO removed: w_inaction=0 for all actions → no penalty
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

    return scenarios


def _generate_tier2_scenarios() -> list[Scenario]:
    """Tier 2: Joint multi-penalty scenarios (20+)."""
    scenarios = []
    configs = [
        # (vote, d1_action, review, adverse, car, ceo_end, node)
        (0.30, "D1_review", True, True, -0.05, False, "D_rev_post"),
        (0.30, "D1_review", True, False, 0.02, True, "D_rev"),
        (0.40, "D0_minimal", False, None, None, True, "D_rev"),
        (0.55, "D1_review", True, True, -0.08, False, "D_rev_post"),
        (0.55, "D0_minimal", False, None, None, True, "D_rev"),
        (0.60, "D1_review", True, True, -0.03, True, "D_rev_post"),
        (0.83, "D0_minimal", False, None, None, True, "D_rev"),
        (0.83, "D1_review", True, True, -0.14, False, "D_rev_post"),
        (0.26, "D1_review", True, True, -0.01, True, "D_rev_post"),
        (0.35, "D0_minimal", False, None, None, False, "D_rev"),
        (0.40, "D1_review", True, False, 0.03, False, "D_rev"),
        (0.50, "D1_review", True, True, -0.05, True, "D_rev_post"),
        (0.60, "D0_minimal", False, None, None, False, "D1"),
        (0.75, "D0_minimal", False, None, None, True, "D_rev"),
        (0.30, "D3_ceo_transition", False, None, None, False, "D1"),
        (0.10, "D0_minimal", False, None, None, True, "D1"),
        (0.20, "D1_review", True, True, -0.08, False, "D_rev_post"),
        (0.45, "D1_review", True, False, 0.05, True, "D_rev"),
        (0.52, "D1_review", True, True, -0.03, False, "D_rev_post"),
        (0.70, "D1_review", True, True, -0.14, True, "D_rev_post"),
    ]
    for i, (v, d1, rev, adv, car, ceo_end, node) in enumerate(configs, 1):
        fa = (["Drev_no_action", "Drev_sack_ceo"] if node == "D_rev_post"
              else ["Drev_no_action", "Drev_commission_review", "Drev_sack_ceo"] if node == "D_rev"
              else ["D0_minimal", "D1_review", "D3_ceo_transition"])
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
                review_adverse=adv,
                car_outcome=car,
                ceo_present_at_end=ceo_end,
            ),
            feasible_actions=fa,
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
                    review_adverse=(car < 0),
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
            review_adverse=None,
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
            review_adverse=None,
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
                    review_adverse=True,
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
            review_adverse=True, car_outcome=-0.05,
            review_origin="board_initiated", ceo_present_at_end=True,
        ),
        feasible_actions=["Drev_no_action", "Drev_sack_ceo"],
    ))
    # (ii) Externally-mandated review, adverse
    n += 1
    scenarios.append(Scenario(
        scenario_id=f"S3_{n:03d}",
        tier=3,
        target_parameter="ikea_vs_self_assessment",
        decision_node="D_rev_post",
        state_vector=_make_state_vector(
            decision_node="D_rev_post", d1_action="D1_review",
            vote_outcome=0.40, review_commissioned=True,
            review_adverse=True, car_outcome=-0.05,
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
            review_adverse=False, car_outcome=0.03,
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
    seed: int,
    client,
    model: str,
    system_prompt_template: str,
    cost_tracker: RunCostSummary,
    token_limit_counter: list[int],
) -> dict:
    """Elicit a single response for one scenario + seed."""
    # Seed on prompt content, not sequential ID — adding/removing scenarios
    # must not invalidate cache for unchanged scenarios.
    # Use hashlib (deterministic) instead of hash() (randomised per process).
    _content_seed = int(hashlib.sha256(
        f"{scenario.prompt_text}|{seed}".encode()
    ).hexdigest(), 16) & 0xFFFFFFFF
    rng = np.random.default_rng(_content_seed)
    factor_order = rng.permutation(10) + 1
    factor_list_str = _format_factor_list(factor_order.tolist())
    system_prompt = system_prompt_template.format(factor_list=factor_list_str)

    # Cache key excludes system_prompt (which varies by factor ordering) — the
    # scenario content + seed fully determines the elicitation result.  Factor
    # order is a presentation detail; parsed results are in canonical form.
    cache_key = _make_cache_key("", scenario.prompt_text, model, seed, 1.0)
    cached = _cache_lookup(cache_key)


    result_row = {
        "result_id": str(uuid.uuid4())[:12],
        "scenario_id": scenario.scenario_id,
        "model": model,
        "prompt_variant": 1,
        "seed": seed,
        "factor_order": json.dumps(factor_order.tolist()),
        "raw_output": "",
        "parse_status": ParseStatus.FORMAT_ERROR.value,
        "prob_vector": "{}",
        "factor_ratings": "[]",
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
        {"role": "user", "content": scenario.prompt_text},
    ]

    parsed, meta = _call_llm_with_retry(client, model, messages, scenario.scenario_id)

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
        logger.error(f"Token limit hit for {scenario.scenario_id} seed={seed}")
        if token_limit_counter[0] > 10:
            raise TokenLimitRunError(
                f"Run aborted: {token_limit_counter[0]} token limit exceedances."
            )
    elif parsed is not None:
        prob_dict = {
            sanitise_text(ap.action.value): round(ap.probability, 4)
            for ap in parsed.prob_vector
        }
        canonical_ratings = [0] * 10
        for fr in parsed.factor_ratings:
            canonical_ratings[fr.factor_index - 1] = fr.rating

        result_row["parse_status"] = ParseStatus.SUCCESS.value
        result_row["prob_vector"] = json.dumps(prob_dict, ensure_ascii=True)
        result_row["factor_ratings"] = json.dumps(canonical_ratings, ensure_ascii=True)
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
    n_reps: int,
    output_path: Path,
    cost_tracker: RunCostSummary,
) -> list[dict]:
    """Stage 2: Run LLM elicitation across all scenarios."""
    logger.info(f"Stage 2: Eliciting {len(scenarios)} scenarios x {n_reps} reps...")

    system_prompt_template = _build_system_prompt()
    token_limit_counter = [0]

    tasks = [
        (scenario, seed)
        for scenario in scenarios
        if scenario.tier != 4
        for seed in range(n_reps)
    ]

    results = []
    from tqdm import tqdm

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(
                _elicit_single, scenario, seed, client, model,
                system_prompt_template, cost_tracker, token_limit_counter,
            ): (scenario.scenario_id, seed)
            for scenario, seed in tasks
        }
        with tqdm(total=len(tasks), desc="Elicitation", smoothing=0) as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except TokenLimitRunError:
                    raise
                except Exception as e:
                    sid, seed = futures[future]
                    logger.error(f"Elicitation failed for {sid} seed={seed}: {e}")
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

def preprocess_data(elicitation_path: Path, output_path: Path) -> pd.DataFrame:
    """Stage 3: Aggregate elicitation results into estimation dataset."""
    logger.info("Stage 3: Preprocessing elicitation data...")

    df = pd.read_csv(elicitation_path, encoding="utf-8")
    success_counts = (
        df[df["parse_status"].isin(["success", "repaired"])]
        .groupby("scenario_id").size()
    )
    valid_ids = success_counts[success_counts >= 7].index.tolist()
    df_valid = df[
        df["scenario_id"].isin(valid_ids)
        & df["parse_status"].isin(["success", "repaired"])
    ].copy()

    if df_valid.empty:
        logger.warning("No valid scenarios after filtering!")
        return pd.DataFrame()

    records = []
    for sid, grp in df_valid.groupby("scenario_id"):
        prob_dicts = [json.loads(row) for row in grp["prob_vector"]]
        all_actions = set()
        for pd_ in prob_dicts:
            all_actions.update(pd_.keys())

        mean_probs = {}
        var_probs = {}
        for action in sorted(all_actions):
            vals = [pd_.get(action, 0.0) for pd_ in prob_dicts]
            mean_probs[action] = float(np.mean(vals))
            var_probs[action] = float(np.var(vals))

        rating_lists = [json.loads(row) for row in grp["factor_ratings"]]
        mean_ratings = np.mean(rating_lists, axis=0).tolist() if rating_lists else [0]*10
        var_ratings = np.var(rating_lists, axis=0).tolist() if rating_lists else [0]*10

        records.append({
            "scenario_id": sid,
            "model": grp["model"].iloc[0],
            "prompt_variant": 1,
            "n_successful_seeds": len(grp),
            "mean_prob_vector": json.dumps(mean_probs, ensure_ascii=True),
            "var_prob_vector": json.dumps(var_probs, ensure_ascii=True),
            "mean_factor_ratings": json.dumps(mean_ratings, ensure_ascii=True),
            "var_factor_ratings": json.dumps(var_ratings, ensure_ascii=True),
            "mean_seed_variance": float(np.mean(list(var_probs.values()))),
        })

    est_df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    est_df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Estimation dataset: {len(est_df)} scenarios")
    return est_df


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
    review_adverse: bool,
    review_car: float,
    review_direct_cost: float,
) -> dict[str, float]:
    """
    Decompose Board utility into per-parameter basis function values.
    Mirrors engine/utilities.py:utility_board() exactly.

    Convention: EU = sum_k w_k * phi_k + anchored. Softmax: P(a) ~ exp(lambda * EU(a)).
    Higher EU = more likely action. All weights are non-negative (>= 0).
    Therefore:
    - PENALTY terms have NEGATIVE phi: larger weight = lower EU = less likely.
    - BENEFIT terms have POSITIVE phi: larger weight = higher EU = more likely.
    """
    ceo_present_at_end = not CEO_removed and not CEO_resigned_early

    removed_involuntary = float(CEO_removed and not CEO_resigned_early)

    # Collapsed basis functions for identification.
    # w_removal = w7 + w8 (perfectly collinear in estimation: both fire on CEO removal)
    # w_inaction = w10 + w11 + w14 (identical basis: strike AND ceo_present_at_end)
    # w8s/w8o/w8r: shock relief (POSITIVE: reduces removal cost → benefit)
    #
    # w12/w13: "continued inaction liability" — penalty when the Board has taken
    # minimal action at EVERY decision point up to and including the current one.
    # At D1: fires for D0_minimal action (choosing inaction).
    # At D_rev: fires for Drev_no_action when d1_action=D0_minimal (continued inaction).
    # At D_rev_post: fires for Drev_no_action when all prior decisions were minimal.
    # This creates action-variation at D_rev (identifiable from softmax).
    board_inactive = (d1_action == "D0_minimal")
    if d_rev_action in ("Drev_sack_ceo", "Drev_commission_review"):
        board_inactive = False
    if d_rev_post_action == "Drev_sack_ceo":
        board_inactive = False

    phi = {
        # Penalties (NEGATIVE: reduce EU, make action less likely)
        "w1": -float(CEO_resigned_early),
        "w2": -(max(vote_percent - 0.25, 0.0) ** 2) if vote_percent > 0.25 else 0.0,
        "w3": -float(overwhelming),
        "w4": -(vote_percent * float(strike)),
        "w_removal": -removed_involuntary,
        "w9": -float(overwhelming),
        "w_inaction": -float(strike and ceo_present_at_end),
        "w12": -float(overwhelming and board_inactive),
        "w13": -float(strike and board_inactive),
        "w15": -float(review_commissioned and review_adverse and ceo_present_at_end),
        # Benefits (POSITIVE: increase EU, make action more likely)
        "w8s": removed_involuntary * float(strike),
        "w8o": removed_involuntary * float(overwhelming),
        "w8r": removed_involuntary * float(review_commissioned and review_adverse),
    }
    return phi


def _compute_anchored_car_contribution(
    review_commissioned: bool,
    review_car: float,
    review_direct_cost: float,
    lambda_la: float = LAMBDA_LA_DEFAULT,
) -> float:
    """Compute the fixed (anchored) CAR + cost contribution to utility."""
    if not review_commissioned:
        return 0.0
    w_car_pos = W_CAR_ANCHOR / ((1 + lambda_la) / 2)
    w_car_neg = lambda_la * w_car_pos
    car_contrib = w_car_pos * max(review_car, 0.0) - w_car_neg * max(-review_car, 0.0)
    cost_contrib = -W_COST_ANCHOR * review_direct_cost
    return car_contrib + cost_contrib


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

    return {
        "vote_percent": sv.get("vote_outcome_V", 0.0),
        "strike": sv.get("strike", False),
        "overwhelming": sv.get("overwhelming", False),
        "d1_action": d1_action,
        "d_rev_action": d_rev_action,
        "d_rev_post_action": d_rev_post_action,
        "CEO_removed": CEO_removed,
        "CEO_resigned_early": CEO_resigned_early,
        "review_commissioned": sv.get("review_commissioned", False),
        "review_adverse": sv.get("review_adverse", False) or False,
        "review_car": sv.get("car_outcome", 0.0) or 0.0,
        "review_direct_cost": 0.00096,  # mean of Gamma(4.55, 4741)
    }


def compute_phi_matrix(
    scenarios: list[Scenario],
    estimation_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[list[str]]]:
    """
    Compute basis function matrix phi[i, a, k] and anchored contributions.

    Returns:
        phi: (n_scenarios, max_actions, n_params) basis function values
        anchored: (n_scenarios, max_actions) anchored CAR+cost contribution
        p_llm: (n_scenarios, max_actions) mean LLM probabilities
        scenario_ids: list of scenario IDs in order
        action_lists: list of feasible action lists per scenario
    """
    logger.info("Computing basis function matrix (phi)...")

    n_params = len(WEIGHT_PARAM_NAMES)
    valid_sids = set(estimation_df["scenario_id"].tolist())

    # Filter scenarios to those in estimation dataset (exclude Tier 4)
    valid_scenarios = [s for s in scenarios if s.scenario_id in valid_sids]

    max_actions = max(len(s.feasible_actions) for s in valid_scenarios)
    n_sc = len(valid_scenarios)

    phi = np.zeros((n_sc, max_actions, n_params))
    anchored = np.zeros((n_sc, max_actions))
    p_llm = np.zeros((n_sc, max_actions))
    scenario_ids = []
    action_lists = []

    for i, scenario in enumerate(valid_scenarios):
        scenario_ids.append(scenario.scenario_id)
        actions = scenario.feasible_actions
        action_lists.append(actions)

        # Get mean probabilities from estimation dataset
        row = estimation_df[estimation_df["scenario_id"] == scenario.scenario_id]
        if row.empty:
            continue
        mean_probs = json.loads(row.iloc[0]["mean_prob_vector"])

        for j, action in enumerate(actions):
            # Basis functions
            args = _scenario_to_outcome_args(scenario.state_vector, action)
            phi_k = decompose_utility_board(**args)

            for k, pname in enumerate(WEIGHT_PARAM_NAMES):
                phi[i, j, k] = phi_k.get(pname, 0.0)

            # Anchored contribution = CAR/cost terms + fixed params at spec defaults
            anchored[i, j] = _compute_anchored_car_contribution(
                args["review_commissioned"], args["review_car"], args["review_direct_cost"],
            )
            # Add fixed (unidentifiable) parameters at their spec default values
            for fp in FIXED_PARAM_NAMES:
                anchored[i, j] += SPEC_DEFAULTS[fp] * phi_k.get(fp, 0.0)

            # LLM probability
            p_llm[i, j] = mean_probs.get(action, 0.0)

    logger.info(f"Phi matrix shape: {phi.shape}, {n_sc} scenarios")
    return phi, anchored, p_llm, scenario_ids, action_lists


def _softmax_probs(
    eu: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Compute softmax probabilities over feasible actions."""
    # eu: (n_actions,), mask: (n_actions,) boolean
    eu_masked = np.where(mask, eu, -1e30)
    eu_shifted = eu_masked - np.max(eu_masked)
    exp_eu = np.where(mask, np.exp(eu_shifted), 0.0)
    total = np.sum(exp_eu)
    if total < 1e-30:
        return np.where(mask, 1.0 / np.sum(mask), 0.0)
    return exp_eu / total


def _cross_entropy_loss(
    params_vec: np.ndarray,
    phi: np.ndarray,
    anchored: np.ndarray,
    p_llm: np.ndarray,
    action_masks: np.ndarray,
) -> float:
    """Cross-entropy loss: -sum_i sum_a p_llm[i,a] * log(p_model[i,a])."""
    weights = params_vec[:len(WEIGHT_PARAM_NAMES)]
    log_lambda = params_vec[-1]
    lam = np.exp(log_lambda)  # ensure lambda > 0

    n_sc, max_a, _ = phi.shape
    loss = 0.0
    eps = 1e-12

    for i in range(n_sc):
        mask = action_masks[i]
        if not np.any(mask):
            continue
        # EU = phi @ weights + anchored
        eu = phi[i] @ weights + anchored[i]
        eu_scaled = lam * eu

        probs = _softmax_probs(eu_scaled, mask)

        for j in range(max_a):
            if mask[j] and p_llm[i, j] > eps:
                loss -= p_llm[i, j] * np.log(max(probs[j], eps))

    return loss


def _cross_entropy_gradient(
    params_vec: np.ndarray,
    phi: np.ndarray,
    anchored: np.ndarray,
    p_llm: np.ndarray,
    action_masks: np.ndarray,
) -> np.ndarray:
    """Analytical gradient of cross-entropy loss."""
    weights = params_vec[:len(WEIGHT_PARAM_NAMES)]
    log_lambda = params_vec[-1]
    lam = np.exp(log_lambda)

    n_params = len(params_vec)
    n_sc, max_a, n_w = phi.shape
    grad = np.zeros(n_params)

    for i in range(n_sc):
        mask = action_masks[i]
        if not np.any(mask):
            continue
        eu = phi[i] @ weights + anchored[i]
        eu_scaled = lam * eu
        probs = _softmax_probs(eu_scaled, mask)

        # Gradient w.r.t. weights: sum_a (p_model - p_llm) * lambda * phi[i,a,:]
        for j in range(max_a):
            if mask[j]:
                diff = probs[j] - p_llm[i, j]
                grad[:n_w] += diff * lam * phi[i, j, :]
                # Gradient w.r.t. log_lambda: diff * lambda * eu[j]
                grad[-1] += diff * lam * eu[j]

    return grad


@dataclass
class EstimationResult:
    weights: dict[str, float]
    lambda_rationality: float
    hessian_se: dict[str, float]
    bootstrap_se: dict[str, float]
    covariance_matrix: np.ndarray
    condition_number: float
    ridge_applied: bool
    w10_w11_w14_collapsed: bool
    loss_value: float
    n_scenarios: int
    converged: bool
    # Tracks how each param was estimated: "softmax_mle", "factor_rating", "profiled"
    estimation_method: dict[str, str] = field(default_factory=dict)
    # Stage 4B regression diagnostics for scenario-level params
    factor_regression_stats: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
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
            "factor_regression_stats": self.factor_regression_stats,
        }


def estimate_parameters(
    phi: np.ndarray,
    anchored: np.ndarray,
    p_llm: np.ndarray,
    action_lists: list[list[str]],
    n_starts: int = 10,
    bootstrap_B: int = 500,
) -> EstimationResult:
    """Stage 4: Estimate utility weights via MLE softmax choice model.

    Lambda (inverse temperature) is NOT jointly estimated with weights —
    they are not identifiable together (any EU difference can be produced
    by small-lambda × large-weights OR large-lambda × small-weights).
    Instead, we profile over a grid of fixed lambda values and pick the
    one with best cross-entropy.
    """
    from scipy.optimize import minimize

    logger.info("Stage 4: Estimating parameters via cross-entropy minimisation...")
    logger.info("Strategy: profile likelihood over lambda grid, weights-only optimisation")

    n_sc, max_a, n_w = phi.shape

    # Build action masks
    action_masks = np.zeros((n_sc, max_a), dtype=bool)
    for i, actions in enumerate(action_lists):
        for j in range(len(actions)):
            action_masks[i, j] = True

    # Bounds: weights >= 0
    bounds_w = [(0, None)] * n_w

    # Lambda grid: profile over these fixed values
    lambda_grid = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    grid_results = []

    rng = np.random.default_rng(42)

    for lam_fixed in lambda_grid:
        log_lam = np.log(lam_fixed)

        def _loss_w(w_vec, _phi=phi, _anch=anchored,
                    _p=p_llm, _m=action_masks, _ll=log_lam):
            full_vec = np.concatenate([w_vec, [_ll]])
            return _cross_entropy_loss(full_vec, _phi, _anch, _p, _m)

        def _grad_w(w_vec, _phi=phi, _anch=anchored,
                    _p=p_llm, _m=action_masks, _ll=log_lam):
            full_vec = np.concatenate([w_vec, [_ll]])
            g = _cross_entropy_gradient(full_vec, _phi, _anch, _p, _m)
            return g[:n_w]

        best_lam_loss = np.inf
        best_lam_x = None

        # Multiple starting points for each lambda
        x0_spec = np.array([SPEC_DEFAULTS[p] for p in WEIGHT_PARAM_NAMES])
        starts_w = [x0_spec.copy()]
        for _ in range(n_starts - 1):
            w_start = np.array([
                max(0.0, rng.normal(SPEC_DEFAULTS[p], 0.3 * max(SPEC_DEFAULTS[p], 0.1)))
                for p in WEIGHT_PARAM_NAMES
            ])
            starts_w.append(w_start)

        for x0 in starts_w:
            try:
                res = minimize(
                    _loss_w, x0,
                    method="L-BFGS-B",
                    jac=_grad_w,
                    bounds=bounds_w,
                    options={"maxiter": 1000, "ftol": 1e-10},
                )
                if res.success and res.fun < best_lam_loss:
                    best_lam_loss = res.fun
                    best_lam_x = res.x.copy()
            except Exception:
                pass

        if best_lam_x is not None:
            grid_results.append((lam_fixed, best_lam_loss, best_lam_x))
            logger.info(f"  lambda={lam_fixed:5.2f}: loss={best_lam_loss:.6f}")

    # Pick best lambda
    if not grid_results:
        logger.error("All lambda grid points failed. Using spec defaults.")
        opt_lambda = 1.0
        best_x = np.array([SPEC_DEFAULTS[p] for p in WEIGHT_PARAM_NAMES])
        best_loss = float("inf")
        converged = False
    else:
        opt_lambda, best_loss, best_x = min(grid_results, key=lambda t: t[1])
        converged = True
        logger.info(f"Best lambda: {opt_lambda:.2f} (loss={best_loss:.6f})")

        # Log all grid results for comparison
        for lam, loss, _ in sorted(grid_results, key=lambda t: t[1]):
            marker = " <-- best" if lam == opt_lambda else ""
            logger.info(f"  lambda={lam:5.2f}: loss={loss:.6f}{marker}")

    # Estimated weights
    opt_weights = {p: round(float(best_x[i]), 4)
                   for i, p in enumerate(WEIGHT_PARAM_NAMES)}
    # Include fixed (unidentifiable) params at spec defaults for display
    for fp in FIXED_PARAM_NAMES:
        opt_weights[fp] = SPEC_DEFAULTS[fp]

    logger.info(f"Fixed params (unidentifiable from softmax): "
                f"{', '.join(f'{p}={SPEC_DEFAULTS[p]}' for p in FIXED_PARAM_NAMES)}")
    logger.info(f"Estimated params: "
                f"{', '.join(f'{p}={opt_weights[p]}' for p in WEIGHT_PARAM_NAMES)}")

    # Build full parameter vector for Hessian/bootstrap (weights + fixed log_lambda)
    full_x = np.concatenate([best_x, [np.log(opt_lambda)]])
    n_params = n_w + 1

    # Hessian SE via finite differences (weights only, lambda fixed)
    hessian_se = {}
    cov_matrix = np.eye(n_params) * 0.01
    condition_number = 1.0
    ridge_applied = False

    try:
        from scipy.optimize import approx_fprime
        h = 1e-5
        # Hessian over weights only (n_w × n_w)
        H_w = np.zeros((n_w, n_w))
        for k in range(n_w):
            def grad_k(w_vec, _k=k):
                fv = np.concatenate([w_vec, [np.log(opt_lambda)]])
                g = _cross_entropy_gradient(fv, phi, anchored, p_llm, action_masks)
                return g[_k]
            H_w[k, :] = approx_fprime(best_x, grad_k, h)
        H_w = (H_w + H_w.T) / 2

        condition_number = float(np.linalg.cond(H_w))
        if condition_number > 1000:
            ridge_applied = True
            H_w += 0.01 * np.eye(n_w)
            logger.warning(f"Hessian ill-conditioned (cond={condition_number:.0f}), ridge applied")

        try:
            cov_w = np.linalg.inv(H_w)
            se = np.sqrt(np.maximum(np.diag(cov_w), 0.0))
            for i, p in enumerate(WEIGHT_PARAM_NAMES):
                hessian_se[p] = round(float(se[i]), 4)
            hessian_se["lambda_rationality"] = 0.0  # profiled, no SE
            for fp in FIXED_PARAM_NAMES:
                hessian_se[fp] = 0.0  # fixed at spec default, no SE
            # Embed in full covariance matrix for dashboard
            cov_matrix = np.zeros((n_params, n_params))
            cov_matrix[:n_w, :n_w] = cov_w
        except np.linalg.LinAlgError:
            logger.warning("Hessian not invertible")
            for p in WEIGHT_PARAM_NAMES:
                hessian_se[p] = float("nan")
            hessian_se["lambda_rationality"] = 0.0
            for fp in FIXED_PARAM_NAMES:
                hessian_se[fp] = 0.0
    except Exception as e:
        logger.warning(f"Hessian computation failed: {e}")
        for p in WEIGHT_PARAM_NAMES:
            hessian_se[p] = float("nan")
        hessian_se["lambda_rationality"] = 0.0
        for fp in FIXED_PARAM_NAMES:
            hessian_se[fp] = 0.0

    # w10/w11/w14 already collapsed to w_inaction pre-estimation
    w10_w11_w14_collapsed = True

    # Bootstrap SE (weights only, lambda fixed)
    bootstrap_se = {}
    bounds_boot = [(0, None)] * n_w
    try:
        logger.info(f"Computing bootstrap SE (B={bootstrap_B})...")
        boot_estimates = []
        rng_boot = np.random.default_rng(123)
        log_lam_fixed = np.log(opt_lambda)

        def _boot_loss(w_vec, _phi_b, _anch_b, _p_b, _m_b):
            full = np.concatenate([w_vec, [log_lam_fixed]])
            return _cross_entropy_loss(full, _phi_b, _anch_b, _p_b, _m_b)

        def _boot_grad(w_vec, _phi_b, _anch_b, _p_b, _m_b):
            full = np.concatenate([w_vec, [log_lam_fixed]])
            g = _cross_entropy_gradient(full, _phi_b, _anch_b, _p_b, _m_b)
            return g[:n_w]

        for b in range(bootstrap_B):
            idx_sample = rng_boot.choice(n_sc, size=n_sc, replace=True)
            phi_b = phi[idx_sample]
            anch_b = anchored[idx_sample]
            p_b = p_llm[idx_sample]
            mask_b = action_masks[idx_sample]

            try:
                res_b = minimize(
                    _boot_loss, best_x,
                    args=(phi_b, anch_b, p_b, mask_b),
                    method="L-BFGS-B",
                    jac=_boot_grad,
                    bounds=bounds_boot,
                    options={"maxiter": 500, "ftol": 1e-8},
                )
                if res_b.success:
                    boot_estimates.append(res_b.x)
            except Exception:
                pass

        if boot_estimates:
            boot_arr = np.array(boot_estimates)
            boot_sd = np.std(boot_arr, axis=0)
            for i, p in enumerate(WEIGHT_PARAM_NAMES):
                bootstrap_se[p] = round(float(boot_sd[i]), 4)
            bootstrap_se["lambda_rationality"] = 0.0  # profiled
            for fp in FIXED_PARAM_NAMES:
                bootstrap_se[fp] = 0.0  # fixed, no SE
            logger.info(f"Bootstrap: {len(boot_estimates)}/{bootstrap_B} converged")
        else:
            for p in FREE_PARAM_NAMES:
                bootstrap_se[p] = float("nan")
            for fp in FIXED_PARAM_NAMES:
                bootstrap_se[fp] = 0.0
    except Exception as e:
        logger.warning(f"Bootstrap failed: {e}")
        for p in FREE_PARAM_NAMES:
            bootstrap_se[p] = float("nan")
        for fp in FIXED_PARAM_NAMES:
            bootstrap_se[fp] = 0.0

    # Rescaling check (spec section 6.6)
    expected_car_contrib = W_CAR_NEG * 0.05  # CAR = -5%
    logger.info(f"Rescaling check: CAR=-5% contribution = {expected_car_contrib:.3f} "
                f"(expected -1.038)")

    return EstimationResult(
        weights=opt_weights,
        lambda_rationality=opt_lambda,
        hessian_se=hessian_se,
        bootstrap_se=bootstrap_se,
        covariance_matrix=cov_matrix,
        condition_number=condition_number,
        ridge_applied=ridge_applied,
        w10_w11_w14_collapsed=w10_w11_w14_collapsed,
        loss_value=float(best_loss),
        n_scenarios=n_sc,
        converged=converged,
        estimation_method={p: "softmax_mle" for p in WEIGHT_PARAM_NAMES}
                         | {fp: "pending_4b" for fp in FIXED_PARAM_NAMES}
                         | {"lambda_rationality": "profiled"},
    )


# ── SEC 8B: Stage 4B — Factor rating regression for scenario-level params ────

def _extract_scenario_phi(scenario: "Scenario") -> dict[str, float]:
    """Extract scenario-level phi values (same for all actions).

    Returns dict with phi values for each scenario-level parameter, plus
    raw features (prefixed with '_') for regression where the phi transform
    has poor identification properties.
    """
    sv = scenario.state_vector
    V = sv.get("vote_outcome_V", 0.0)
    strike = sv.get("strike", False)
    overwhelming = sv.get("overwhelming", False)
    ceo_res = sv.get("ceo_status_at_start", "present") == "resigned_early"

    return {
        "w1": float(ceo_res),           # |phi| = 1 if CEO resigned early
        "w2": (max(V - 0.25, 0.0) ** 2) if V > 0.25 else 0.0,  # quadratic vote penalty
        "w3": float(overwhelming),       # binary: vote >= 50%
        "w4": V * float(strike),         # V × strike interaction
        "w9": float(overwhelming),       # reputational spill (same basis as w3)
        "_V": V,                         # raw vote % for w2 regression
    }


def estimate_scenario_level_params(
    scenarios: list["Scenario"],
    estimation_df: pd.DataFrame,
    stage4a_result: EstimationResult,
) -> dict:
    """
    Stage 4B: Estimate scenario-level parameters (w1, w2, w3, w4, w9) via
    factor rating regression.

    These parameters have phi that is constant across actions within a scenario,
    so they cancel in the softmax choice model and have zero gradient.
    Instead, we use the LLM's factor ratings (1-5 Likert) which DO respond to
    scenario-level features.

    Model: mean_rating_{i,f} = alpha_f + beta * |phi_k(scenario_i)| + epsilon

    The coefficient beta measures Likert-points-per-unit-phi. We convert to
    utility weight scale using w_removal from Stage 4A as a bridge.

    Returns dict with keys:
        weights: {param: estimated_value}
        se: {param: standard_error}
        regression_stats: {param: {beta, alpha, r_squared, p_value, n_obs, factors_used}}
    """
    from scipy import stats as sp_stats

    logger.info("Stage 4B: Estimating scenario-level params via factor rating regression...")

    valid_sids = set(estimation_df["scenario_id"].tolist())
    valid_scenarios = [s for s in scenarios if s.scenario_id in valid_sids]

    # Build scenario-level data: phi values + factor ratings
    rows = []
    for scenario in valid_scenarios:
        row_data = estimation_df[estimation_df["scenario_id"] == scenario.scenario_id]
        if row_data.empty:
            continue
        phi_s = _extract_scenario_phi(scenario)
        ratings = json.loads(row_data.iloc[0]["mean_factor_ratings"])
        if len(ratings) < 10:
            continue
        rows.append({
            "scenario_id": scenario.scenario_id,
            "phi": phi_s,
            "ratings": ratings,  # 10-element list (0-indexed: F1=ratings[0], F10=ratings[9])
        })

    if not rows:
        logger.warning("Stage 4B: No valid scenarios for factor rating regression")
        return {"weights": {}, "se": {}, "regression_stats": {}}

    # ── Step 1: Compute bridge scale factor using w_removal ──
    # w_removal is estimated from Stage 4A (action-varying, known scale).
    # Factor 6 (direct costs of governance reform) maps to removal cost.
    # Factor 8 (implementation complexity) also maps to removal.
    # Use their correlation with |phi_removal| across scenarios to get
    # Likert-per-util conversion.
    w_removal_4a = stage4a_result.weights.get("w_removal", 1.0)
    logger.info(f"  Bridge parameter: w_removal = {w_removal_4a:.4f} (from Stage 4A)")

    # Compute |phi_removal| per scenario (it varies by action, so use Drev_sack_ceo
    # where phi_removal = 1, vs other actions where it's 0)
    # But for bridge scaling, we need a DIFFERENT approach:
    # The factor ratings are SCENARIO-level (one set per scenario, not per action).
    # So we can't use action-varying phi for the bridge.
    #
    # Instead, we use the REGRESSION SLOPE directly as the weight estimate.
    # The factor rating model is: F_rating = alpha + beta * |phi_k|
    # The utility model is: EU_k = w_k * phi_k
    # If the LLM's factor ratings linearly reflect perceived disutility,
    # then beta captures relative importance on the Likert scale.
    #
    # To convert beta (Likert/unit_phi) to w_k (utils/unit_phi):
    #   w_k = beta * (w_ref / beta_ref)
    # where w_ref and beta_ref come from a parameter identified by both methods.
    #
    # Since w3 and w9 share the same phi (both = overwhelming indicator),
    # we need to handle them differently. w3 uses F9+F10 (activist escalation +
    # board cohesion) while w9 uses F7 (reputational contagion).

    results = {"weights": {}, "se": {}, "regression_stats": {}}

    # ── Step 2: Run OLS for each scenario-level parameter ──
    for param in FIXED_PARAM_NAMES:
        mapping = FACTOR_PARAM_MAP.get(param)
        if not mapping:
            logger.warning(f"  {param}: no factor mapping defined, using spec default")
            results["weights"][param] = SPEC_DEFAULTS[param]
            results["se"][param] = 0.0
            results["regression_stats"][param] = {"status": "no_mapping"}
            continue

        factor_indices = mapping["factors"]  # 1-based

        # Build regressor and response.
        # For w2, the phi transform (V-0.25)² compresses variation near zero,
        # giving poor R². Factor ratings respond more linearly to V itself.
        # We regress on V directly and convert the coefficient via the
        # marginal relationship: dEU/dV = 2*w2*(V-0.25), so
        # w2_likert = gamma / (2*(V_ref - 0.25)) where V_ref is mean V
        # across scenarios with strikes.
        use_raw_V = (param == "w2")
        x_vals = []
        y_vals = []
        for r in rows:
            if use_raw_V:
                x_val = r["phi"].get("_V", 0.0)
            else:
                x_val = abs(r["phi"].get(param, 0.0))
            avg_rating = np.mean([r["ratings"][fi - 1] for fi in factor_indices])
            x_vals.append(x_val)
            y_vals.append(avg_rating)

        x_arr = np.array(x_vals)
        y_arr = np.array(y_vals)

        # Need variation in x to run regression
        if np.std(x_arr) < 1e-8:
            logger.warning(f"  {param}: no variation in phi across scenarios, "
                           f"using spec default {SPEC_DEFAULTS[param]}")
            results["weights"][param] = SPEC_DEFAULTS[param]
            results["se"][param] = 0.0
            results["regression_stats"][param] = {
                "status": "no_variation",
                "n_obs": len(x_arr),
                "factors_used": factor_indices,
            }
            continue

        # OLS: y = alpha + beta * x
        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x_arr, y_arr)

        # For w2, convert slope on V to coefficient on (V-0.25)²
        if use_raw_V:
            # gamma = dF/dV. EU contribution = w2*(V-0.25)². dEU/dV = 2*w2*(V-0.25).
            # At V_ref: w2_likert = gamma / (2*(V_ref - 0.25))
            strike_x = x_arr[x_arr > 0.25]
            V_ref = float(np.mean(strike_x)) if len(strike_x) > 0 else 0.40
            denom = 2.0 * max(V_ref - 0.25, 0.05)  # floor to avoid division by tiny number
            effective_slope = slope / denom
            effective_se = std_err / denom
            logger.info(f"  {param}: raw gamma={slope:.4f} (on V), V_ref={V_ref:.3f}, "
                        f"converted beta={effective_slope:.4f} (on (V-0.25)²), "
                        f"R²={r_value**2:.4f}, p={p_value:.4f}, n={len(x_arr)}, "
                        f"factors={factor_indices}")
        else:
            effective_slope = slope
            effective_se = std_err
            logger.info(f"  {param}: beta={effective_slope:.4f}, R²={r_value**2:.4f}, "
                        f"p={p_value:.4f}, n={len(x_arr)}, "
                        f"factors={factor_indices}")

        results["regression_stats"][param] = {
            "status": "estimated",
            "beta": round(float(effective_slope), 4),
            "alpha": round(float(intercept), 4),
            "r_squared": round(float(r_value ** 2), 4),
            "p_value": round(float(p_value), 6),
            "std_err": round(float(effective_se), 4),
            "n_obs": len(x_arr),
            "factors_used": factor_indices,
        }
        if use_raw_V:
            results["regression_stats"][param]["regressor"] = "V (linear)"
            results["regression_stats"][param]["V_ref"] = round(V_ref, 4)
            results["regression_stats"][param]["raw_gamma"] = round(float(slope), 4)

        # Store effective beta and SE for scale conversion below
        results["weights"][param] = float(effective_slope)
        results["se"][param] = float(effective_se)

    # ── Step 3: Scale conversion ──
    # The betas are in Likert-points-per-unit-phi. We need to convert to
    # utility-weight scale. The key insight: the beta for each parameter
    # tells us how many Likert points the LLM assigns per unit of phi.
    # A higher beta means the LLM perceives that feature as more important.
    #
    # We use the RELATIVE betas to set the RELATIVE weights, then anchor
    # the overall scale using one parameter with known utility impact.
    #
    # w3 (overwhelming penalty) is identified by F9+F10 and also appears
    # in action-varying combinations (it interacts with action through w12).
    # But the cleanest bridge is the total utility impact:
    #   For a scenario going from non-overwhelming to overwhelming,
    #   the total utility change = -(w3 + w9) (from phi_w3 and phi_w9).
    #   The LLM responds with ~1.5 Likert point increase across factors.
    #
    # Simpler approach: use the spec defaults as the scale anchor.
    # beta_k measures "perceived importance per unit phi" in Likert units.
    # SPEC_DEFAULTS[k] is the expert-specified weight in utility units.
    # If the LLM agrees with the spec, then beta_k ∝ SPEC_DEFAULTS[k].
    # We set: w_k = beta_k * (sum(SPEC_DEFAULTS[fixed]) / sum(|beta_k|))
    #
    # Even simpler: use w_removal from 4A as a reference.
    # Factor 4 (CEO knowledge loss) and Factor 6 (direct costs) relate to removal.
    # For scenarios where removal varies, we can compute beta_removal.

    # Compute bridge: run same regression for w_removal using F4+F6
    removal_x = []
    removal_y = []
    for r in rows:
        # phi_removal depends on action, but we can still check if the SCENARIO
        # involves CEO removal by checking if the state has ceo_present_at_end=False
        sv_check = r.get("phi", {})
        # Use the scenario's inherent CEO removal status
        # Actually, phi_removal is action-dependent, not scenario-level.
        # So we can't directly bridge via regression.
        pass

    # Since w_removal is action-dependent, we can't use factor ratings (which
    # are scenario-level) to bridge directly. Instead, use the SPEC_DEFAULTS
    # ratio as the scale anchor:
    #
    # w_k_estimated = |beta_k| * (SPEC_DEFAULTS[k] / beta_k_expected)
    #
    # where beta_k_expected = slope from regressing rating on |phi| IF the
    # spec defaults were the true weights. But we don't know that either.
    #
    # The cleanest approach: the betas ARE the estimates, in "Likert units".
    # We report them as-is, and note that the scale is "per Likert point of
    # perceived importance". The user can decide the conversion factor.
    #
    # ACTUALLY — the most principled approach: betas from the factor rating
    # regressions are already on a meaningful scale (Likert/unit_phi). We
    # convert them to the SAME scale as the Stage 4A weights by noting that
    # the Stage 4A weights are "utility units per unit phi". If we define
    # 1 Likert point ≈ k utility units, then w = beta * k.
    #
    # To find k: use w3 which has phi varying between 0 and 1 across scenarios.
    # The scenarios that cross the overwhelming threshold show the jump in F9+F10.
    # The spec default w3=3.0 means losing 3 utility units when overwhelming=1.
    # If beta_w3 ≈ 1.5 (the Likert jump), then k = 3.0/1.5 = 2.0.
    # But this is circular if we're trying to ESTIMATE w3.
    #
    # RESOLUTION: Report the betas directly as the weight estimates.
    # The factor rating scale (1-5) already provides a natural unit of
    # "perceived importance". A beta of 2.0 means going from phi=0 to phi=1
    # adds 2 Likert points of perceived concern. This is interpretable.
    # The absolute scale relative to CAR/cost terms is fixed by the anchored
    # parameters (W_CAR=15, LAMBDA_LA=2.25), so the betas need to be on a
    # comparable scale.
    #
    # Final decision: the betas represent the utility weight in
    # "Likert-perceived-importance" units. We scale them to match the
    # 4A weight scale by normalising to the average 4A weight magnitude.

    # Scale: average |w| from Stage 4A for non-near-zero params
    stage4a_weights = [abs(stage4a_result.weights.get(p, 0))
                       for p in ESTIMABLE_PARAM_NAMES
                       if abs(stage4a_result.weights.get(p, 0)) > 0.05]
    if stage4a_weights:
        avg_4a_weight = np.mean(stage4a_weights)
    else:
        avg_4a_weight = 1.0

    # Average |beta| for params that had valid regressions
    valid_betas = [abs(results["weights"][p]) for p in FIXED_PARAM_NAMES
                   if results["regression_stats"].get(p, {}).get("status") == "estimated"
                   and abs(results["weights"][p]) > 0.01]
    if valid_betas:
        avg_beta = np.mean(valid_betas)
        scale_factor = avg_4a_weight / avg_beta
    else:
        scale_factor = 1.0

    logger.info(f"  Scale bridge: avg |4A weight| = {avg_4a_weight:.4f}, "
                f"avg |beta| = {np.mean(valid_betas) if valid_betas else 0:.4f}, "
                f"scale_factor = {scale_factor:.4f}")

    # Apply scale conversion
    for param in FIXED_PARAM_NAMES:
        if results["regression_stats"].get(param, {}).get("status") == "estimated":
            raw_beta = results["weights"][param]
            scaled_w = abs(raw_beta) * scale_factor
            results["weights"][param] = round(scaled_w, 4)
            results["se"][param] = round(results["se"][param] * scale_factor, 4)
            results["regression_stats"][param]["scaled_weight"] = round(scaled_w, 4)
            results["regression_stats"][param]["scale_factor"] = round(scale_factor, 4)
            logger.info(f"  {param}: raw beta={raw_beta:.4f} → "
                        f"scaled weight={scaled_w:.4f}")

    return results


# ── SEC 9: Stage 5 — Behavioural diagnostics ─────────────────────────────────

def _diagnose_loss_aversion(
    scenarios: list[Scenario],
    estimation_df: pd.DataFrame,
) -> dict:
    """Test 8.1: Loss aversion via matched gain/loss CAR pairs."""
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
        row_pos = estimation_df[estimation_df["scenario_id"] == pair["pos"]]
        row_neg = estimation_df[estimation_df["scenario_id"] == pair["neg"]]
        if row_pos.empty or row_neg.empty:
            continue
        probs_pos = json.loads(row_pos.iloc[0]["mean_prob_vector"])
        probs_neg = json.loads(row_neg.iloc[0]["mean_prob_vector"])

        # Sensitivity: change in P(commission_review) or P(sack)
        # Use Drev_commission_review as the review-related action
        p_rev_pos = probs_pos.get("Drev_commission_review", 0)
        p_rev_neg = probs_neg.get("Drev_commission_review", 0)
        p_baseline = 0.33  # uniform prior
        delta_pos = abs(p_rev_pos - p_baseline) + 1e-8
        delta_neg = abs(p_rev_neg - p_baseline) + 1e-8
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
    estimation_df: pd.DataFrame,
    est_result: EstimationResult,
) -> dict:
    """Test 8.2: Non-linearity in vote penalty and diminishing marginal disutility."""
    from scipy import stats

    # Vote penalty functional form comparison
    w2_scenarios = [s for s in scenarios if s.target_parameter == "w2"]
    vote_points = []
    for s in w2_scenarios:
        row = estimation_df[estimation_df["scenario_id"] == s.scenario_id]
        if row.empty:
            continue
        v = s.state_vector.get("vote_outcome_V", 0)
        if v <= 0.25:
            continue
        probs = json.loads(row.iloc[0]["mean_prob_vector"])
        # Use P(D1_review) as a proxy for penalty sensitivity
        p_review = probs.get("D1_review", 0)
        vote_points.append((v, p_review))

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

    # Diminishing marginal disutility — count active penalties per Tier 2 scenario
    t2_scenarios = [s for s in scenarios if s.tier == 2]
    penalty_counts = []
    for s in t2_scenarios:
        row = estimation_df[estimation_df["scenario_id"] == s.scenario_id]
        if row.empty:
            continue
        sv = s.state_vector
        n_active = sum([
            sv.get("strike", False),
            sv.get("overwhelming", False),
            sv.get("review_adverse", False) or False,
            not sv.get("ceo_present_at_end", True),
            sv.get("vote_outcome_V", 0) > 0.25,
        ])
        probs = json.loads(row.iloc[0]["mean_prob_vector"])
        # Use max probability of "severe" action as proxy for total disutility
        p_severe = max(
            probs.get("D3_ceo_transition", 0),
            probs.get("Drev_sack_ceo", 0),
            probs.get("D1_review", 0),
        )
        penalty_counts.append((n_active, p_severe))

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
    estimation_df: pd.DataFrame,
) -> dict:
    """Test 8.3: Optimism bias — explicit vs implicit adverse probability."""
    opt_scenarios = [s for s in scenarios if s.target_parameter == "optimism_bias"]
    explicit = []
    implicit = []
    for s in opt_scenarios:
        row = estimation_df[estimation_df["scenario_id"] == s.scenario_id]
        if row.empty:
            continue
        probs = json.loads(row.iloc[0]["mean_prob_vector"])
        p_review = probs.get("Drev_commission_review", 0)
        if s.state_vector.get("explicit_adverse_prob"):
            explicit.append(p_review)
        else:
            implicit.append(p_review)

    if not explicit or not implicit:
        return {"test": "optimism_bias", "decision": "insufficient_data"}

    from scipy import stats
    t_stat, p_val = stats.ttest_ind(explicit, implicit) if len(explicit) > 1 and len(implicit) > 1 else (0, 1)
    effect_size = float(np.mean(implicit) - np.mean(explicit))

    return {
        "test": "optimism_bias",
        "mean_p_review_explicit": round(float(np.mean(explicit)), 4),
        "mean_p_review_implicit": round(float(np.mean(implicit)), 4),
        "effect_size": round(effect_size, 4),
        "t_stat": round(float(t_stat), 3),
        "p_value": round(float(p_val), 4),
        "decision": "confirmed" if p_val < 0.05 and effect_size > 0 else "null",
    }


def _diagnose_self_assessment(
    scenarios: list[Scenario],
    estimation_df: pd.DataFrame,
) -> dict:
    """Test 8.4: Self-assessment bias — board vs external review origin."""
    sa_scenarios = [s for s in scenarios if s.target_parameter == "self_assessment_bias"]
    board_init = []
    ext_mandated = []
    for s in sa_scenarios:
        row = estimation_df[estimation_df["scenario_id"] == s.scenario_id]
        if row.empty:
            continue
        probs = json.loads(row.iloc[0]["mean_prob_vector"])
        p_sack = probs.get("Drev_sack_ceo", 0)
        origin = s.state_vector.get("review_origin", "board_initiated")
        if origin == "board_initiated":
            board_init.append(p_sack)
        else:
            ext_mandated.append(p_sack)

    if not board_init or not ext_mandated:
        return {"test": "self_assessment_bias", "decision": "insufficient_data"}

    from scipy import stats
    t_stat, p_val = stats.ttest_ind(board_init, ext_mandated) if len(board_init) > 1 and len(ext_mandated) > 1 else (0, 1)

    return {
        "test": "self_assessment_bias",
        "mean_p_sack_board_initiated": round(float(np.mean(board_init)), 4),
        "mean_p_sack_externally_mandated": round(float(np.mean(ext_mandated)), 4),
        "t_stat": round(float(t_stat), 3),
        "p_value": round(float(p_val), 4),
        "decision": "confirmed" if p_val < 0.05 and np.mean(board_init) < np.mean(ext_mandated) else "null",
    }


def _diagnose_ikea_effect(
    scenarios: list[Scenario],
    estimation_df: pd.DataFrame,
) -> dict:
    """Test 8.5: Ikea effect — CEO appointment and review ownership."""
    ikea_scenarios = [s for s in scenarios if s.target_parameter == "ikea_effect"]
    appointed = []
    inherited = []
    for s in ikea_scenarios:
        row = estimation_df[estimation_df["scenario_id"] == s.scenario_id]
        if row.empty:
            continue
        probs = json.loads(row.iloc[0]["mean_prob_vector"])
        p_sack = probs.get("Drev_sack_ceo", 0)
        appt = s.state_vector.get("ceo_appointment", "appointed_by_current_board")
        if appt == "appointed_by_current_board":
            appointed.append(p_sack)
        else:
            inherited.append(p_sack)

    if not appointed or not inherited:
        return {"test": "ikea_effect", "decision": "insufficient_data"}

    from scipy import stats
    t_stat, p_val = stats.ttest_ind(appointed, inherited) if len(appointed) > 1 and len(inherited) > 1 else (0, 1)

    return {
        "test": "ikea_effect",
        "mean_p_sack_appointed": round(float(np.mean(appointed)), 4),
        "mean_p_sack_inherited": round(float(np.mean(inherited)), 4),
        "t_stat": round(float(t_stat), 3),
        "p_value": round(float(p_val), 4),
        "decision": "confirmed" if p_val < 0.05 and np.mean(appointed) < np.mean(inherited) else "null",
    }


def _diagnose_factor_order_effects(
    elicitation_path: Path,
) -> dict:
    """Test 8.6: Factor rating order effects."""
    from scipy import stats

    df = pd.read_csv(elicitation_path, encoding="utf-8")
    df_ok = df[df["parse_status"].isin(["success", "repaired"])].copy()

    if df_ok.empty:
        return {"test": "factor_order_effects", "decision": "insufficient_data"}

    results_per_factor = {}
    for factor_idx in range(1, 11):
        positions = []
        ratings = []
        for _, row in df_ok.iterrows():
            try:
                order = json.loads(row["factor_order"])
                fr = json.loads(row["factor_ratings"])
                pos = order.index(factor_idx) + 1 if factor_idx in order else None
                rating = fr[factor_idx - 1] if len(fr) >= factor_idx else None
                if pos is not None and rating is not None:
                    positions.append(pos)
                    ratings.append(rating)
            except Exception:
                continue

        if len(positions) >= 10:
            slope, intercept, r, p, se = stats.linregress(positions, ratings)
            results_per_factor[f"factor_{factor_idx}"] = {
                "slope": round(float(slope), 4),
                "p_value": round(float(p), 4),
                "effect": "primacy" if slope < 0 and p < 0.05 else
                          "recency" if slope > 0 and p < 0.05 else "none",
            }

    any_effect = any(
        v.get("effect", "none") != "none"
        for v in results_per_factor.values()
    )

    return {
        "test": "factor_order_effects",
        "per_factor": results_per_factor,
        "any_order_effect_detected": any_effect,
    }


def run_diagnostics(
    scenarios: list[Scenario],
    estimation_df: pd.DataFrame,
    est_result: EstimationResult,
    elicitation_path: Path,
    output_path: Path,
) -> dict:
    """Stage 5: Run all behavioural diagnostics."""
    logger.info("Stage 5: Running behavioural diagnostics...")

    diagnostics = {
        "loss_aversion": _diagnose_loss_aversion(scenarios, estimation_df),
        "nonlinearity": _diagnose_nonlinearity(scenarios, estimation_df, est_result),
        "optimism_bias": _diagnose_optimism_bias(scenarios, estimation_df),
        "self_assessment_bias": _diagnose_self_assessment(scenarios, estimation_df),
        "ikea_effect": _diagnose_ikea_effect(scenarios, estimation_df),
        "factor_order_effects": _diagnose_factor_order_effects(elicitation_path),
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
    p_llm: np.ndarray,
    action_masks: np.ndarray,
    est_result: EstimationResult,
    scenario_ids: list[str],
    action_lists: list[list[str]],
) -> list[dict]:
    """Compute per-scenario KL divergence and residuals."""
    weights = np.array([est_result.weights[p] for p in WEIGHT_PARAM_NAMES])
    lam = est_result.lambda_rationality

    n_sc = phi.shape[0]
    eps = 1e-12
    fit_rows = []

    for i in range(n_sc):
        mask = action_masks[i]
        eu = phi[i] @ weights + anchored[i]
        probs = _softmax_probs(lam * eu, mask)

        kl = 0.0
        residuals = {}
        for j, action in enumerate(action_lists[i]):
            if mask[j]:
                p_l = max(p_llm[i, j], eps)
                p_m = max(probs[j], eps)
                kl += p_l * np.log(p_l / p_m)
                residuals[action] = round(float(p_llm[i, j] - probs[j]), 4)

        fit_rows.append({
            "scenario_id": scenario_ids[i],
            "kl_divergence": round(float(kl), 6),
            "residuals": json.dumps(residuals, ensure_ascii=True),
            "model_probs": json.dumps(
                {a: round(float(probs[j]), 4) for j, a in enumerate(action_lists[i]) if mask[j]},
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
    action_masks: np.ndarray,
    scenario_ids: list[str],
    action_lists: list[list[str]],
) -> dict:
    """Validate against Tier 4 historical scenario (Qantas AGM Nov 2023)."""
    t4 = [s for s in scenarios if s.tier == 4]
    if not t4:
        return {"available": False}

    s = t4[0]
    # Compute model prediction for Tier 4
    weights = np.array([est_result.weights[p] for p in WEIGHT_PARAM_NAMES])
    lam = est_result.lambda_rationality

    predictions = {}
    for action in s.feasible_actions:
        args = _scenario_to_outcome_args(s.state_vector, action)
        phi_k = decompose_utility_board(**args)
        phi_vec = np.array([phi_k.get(p, 0.0) for p in WEIGHT_PARAM_NAMES])
        anch = _compute_anchored_car_contribution(
            args["review_commissioned"], args["review_car"], args["review_direct_cost"],
        )
        # Add fixed params contribution
        fixed_contrib = sum(SPEC_DEFAULTS[fp] * phi_k.get(fp, 0.0) for fp in FIXED_PARAM_NAMES)
        eu = float(phi_vec @ weights + anch + fixed_contrib)
        predictions[action] = eu

    # Softmax
    eus = np.array([predictions[a] for a in s.feasible_actions])
    mask = np.ones(len(s.feasible_actions), dtype=bool)
    probs = _softmax_probs(lam * eus, mask)

    predicted_probs = {a: round(float(p), 4) for a, p in zip(s.feasible_actions, probs)}
    ranked = sorted(predicted_probs.items(), key=lambda x: -x[1])

    return {
        "available": True,
        "scenario_id": s.scenario_id,
        "predicted_probs": predicted_probs,
        "rank_of_D1_review": [i+1 for i, (a, _) in enumerate(ranked) if a == "D1_review"][0]
            if "D1_review" in predicted_probs else None,
        "top_action": ranked[0][0],
        "top_prob": ranked[0][1],
    }


def run_validation(
    scenarios: list[Scenario],
    estimation_df: pd.DataFrame,
    est_result: EstimationResult,
    phi: np.ndarray,
    anchored: np.ndarray,
    p_llm: np.ndarray,
    action_masks: np.ndarray,
    scenario_ids: list[str],
    action_lists: list[list[str]],
    output_dir: Path,
) -> dict:
    """Stage 6: Run validation checks."""
    logger.info("Stage 6: Running validation...")

    # Build action masks for consistency
    n_sc, max_a = p_llm.shape
    if action_masks is None:
        action_masks = np.zeros((n_sc, max_a), dtype=bool)
        for i, actions in enumerate(action_lists):
            for j in range(len(actions)):
                action_masks[i, j] = True

    # Within-sample fit
    fit_rows = _compute_scenario_fit(
        phi, anchored, p_llm, action_masks, est_result, scenario_ids, action_lists,
    )
    kl_values = [r["kl_divergence"] for r in fit_rows]
    mean_kl = float(np.mean(kl_values)) if kl_values else float("nan")

    # Save scenario fit
    fit_df = pd.DataFrame(fit_rows)
    fit_df.to_csv(output_dir / "scenario_fit.csv", index=False, encoding="utf-8")

    # 5 worst-fitting
    sorted_fit = sorted(fit_rows, key=lambda x: -x["kl_divergence"])
    worst_5 = sorted_fit[:5]

    # Historical validation
    historical = _validate_historical(
        scenarios, estimation_df, est_result, phi, anchored, action_masks,
        scenario_ids, action_lists,
    )

    # Factor rating regression
    factor_regression = {}
    try:
        from scipy import stats
        for f_idx in range(10):
            eu_contribs = []
            mean_ratings_list = []
            for _, row in estimation_df.iterrows():
                ratings = json.loads(row["mean_factor_ratings"])
                if len(ratings) > f_idx:
                    mean_ratings_list.append(ratings[f_idx])
                    eu_contribs.append(f_idx + 1)  # placeholder

            if len(mean_ratings_list) >= 5:
                slope, intercept, r, p, se = stats.linregress(eu_contribs, mean_ratings_list)
                factor_regression[f"factor_{f_idx+1}"] = {
                    "slope": round(float(slope), 4),
                    "r_squared": round(float(r**2), 4),
                    "p_value": round(float(p), 4),
                }
    except Exception as e:
        logger.warning(f"Factor regression failed: {e}")

    validation = {
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
        "factor_regression": factor_regression,
    }

    # Save validation results
    with open(output_dir / "validation_results.json", "w", encoding="utf-8") as f:
        json.dump(validation, f, ensure_ascii=True, indent=2)

    logger.info(f"Validation: mean KL = {mean_kl:.4f} (target < 0.05)")
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
        vote = sv.get("vote_outcome", 0.0)
        strike = float(vote > 0.25)
        overwhelming = float(vote > 0.50)
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

    def to_json(self) -> str:
        d = {}
        for k in [
            "run_status", "run_start", "generated_at", "model",
            "scenarios", "elicitation_summary", "elicited_probabilities",
            "estimation_dataset_summary",
            "parameter_estimates", "covariance_matrix", "behavioural_diagnostics",
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
.content{padding:20px;max-width:1400px;margin:0 auto}
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
</head>
<body>
<script>const RESULTS_DATA = __RESULTS_DATA__;</script>

<div id="banner"></div>
<div class="tabs" id="tabBar"></div>
<div class="content" id="content"></div>

<script>
const TAB_NAMES = [
  "Overview","Cost & Usage","Scenario Battery","Elicitation Results",
  "Elicited Probabilities","Parameter Estimates","Covariance",
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
  let html='<div class="card"><h3>Scenario Battery ('+sc.length+' scenarios)</h3>';
  html+='<input class="search-box" placeholder="Filter scenarios..." oninput="filterScenarios(this.value)">';
  const headers=['ID','Tier','Target','Node','Vote','CEO End'];
  const rows=sc.map(s=>{
    const sv=s.state_vector||{};
    return [s.scenario_id,s.tier,s.target_parameter,s.decision_node,
      (sv.vote_outcome_V||0).toFixed(2),sv.ceo_present_at_end?'Present':'Removed'];
  });
  html+=makeTable(headers,rows,'scenarioTable');
  html+='</div>';
  p.innerHTML=html;
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
  if(!es.total_calls){placeholder(3,'Elicitation not yet run.');return;}
  let html='<div class="card"><h3>Elicitation Summary</h3><div class="stat-grid">';
  html+='<div class="stat"><div class="value">'+(es.total_calls||0)+'</div><div class="label">Total Calls</div></div>';
  html+='<div class="stat"><div class="value">'+(es.success_rate||0).toFixed(1)+'%</div><div class="label">Parse Success</div></div>';
  html+='<div class="stat"><div class="value">'+(es.cache_hit_rate||0).toFixed(1)+'%</div><div class="label">Cache Hit Rate</div></div>';
  html+='</div></div>';
  html+='<div class="card"><h3>Parse Status Breakdown</h3><div id="parseChart" class="chart"></div></div>';
  p.innerHTML=html;
  if(typeof Plotly!=='undefined'&&es.parse_status_counts){
    const labels=Object.keys(es.parse_status_counts);
    const vals=Object.values(es.parse_status_counts);
    Plotly.newPlot('parseChart',[{labels,values:vals,type:'pie',hole:0.4,
      marker:{colors:['#50C878','#E85D5D','#FFB347','#4A90D9','#AAAAAA']}}],
      {margin:{t:20,b:20,l:20,r:20},height:300},{responsive:true});
  }
})();

// Panel 4: Elicited Probabilities
(function(){
  const p=document.getElementById('panel_4');
  const ep=D.elicited_probabilities;
  if(!ep||!ep.length){placeholder(4,'Elicitation data not yet available.');return;}

  // Collect all action codes across all scenarios
  const allActions=new Set();
  ep.forEach(s=>{Object.keys(s.mean_probs||{}).forEach(a=>allActions.add(a));});
  const actionList=Array.from(allActions).sort();

  // Summary stats
  let html='<div class="card"><h3>Elicited Action Probabilities by Scenario</h3>';
  html+='<p style="color:var(--text-muted)">Mean probabilities across LLM seeds per scenario. Use the tier filter and search box to explore.</p>';

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
    html+='<th style="padding:8px;border:1px solid var(--border);text-align:right">P('+a+')</th>';
  });
  html+='<th style="padding:8px;border:1px solid var(--border);text-align:right">Seeds</th>';
  html+='</tr></thead><tbody>';

  // Table rows
  ep.forEach(s=>{
    const probs=s.mean_probs||{};
    const maxP=Math.max(...actionList.map(a=>probs[a]||0));
    html+='<tr data-tier="'+s.tier+'" data-search="'+(s.scenario_id+' '+s.target_parameter).toLowerCase()+'">';
    html+='<td style="padding:6px 8px;border:1px solid var(--border);font-family:monospace;font-size:12px">'+s.scenario_id+'</td>';
    html+='<td style="padding:6px 8px;border:1px solid var(--border);text-align:center">T'+s.tier+'</td>';
    html+='<td style="padding:6px 8px;border:1px solid var(--border)">'+s.target_parameter+'</td>';
    html+='<td style="padding:6px 8px;border:1px solid var(--border)">'+s.decision_node+'</td>';
    html+='<td style="padding:6px 8px;border:1px solid var(--border);text-align:right">'+(s.vote_pct!=null?(s.vote_pct*100).toFixed(0)+'%':'--')+'</td>';
    actionList.forEach(a=>{
      const pv=probs[a]||0;
      const bg=pv>=maxP-0.001&&pv>0.01?'background:#d4edda':'';
      html+='<td style="padding:6px 8px;border:1px solid var(--border);text-align:right;'+bg+'">'+pv.toFixed(3)+'</td>';
    });
    html+='<td style="padding:6px 8px;border:1px solid var(--border);text-align:right">'+s.n_seeds+'</td>';
    html+='</tr>';
  });
  html+='</tbody></table></div></div>';

  // Grouped bar chart: one trace per action, x = scenario_id
  html+='<div class="card"><h3>Action Probability Distribution</h3>';
  html+='<p style="color:var(--text-muted)">Grouped bar chart of mean LLM-elicited probabilities. Click tier buttons above to filter.</p>';
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
      y:filtered.map(s=>(s.mean_probs||{})[a]||0),
      name:a,
      type:'bar',
      marker:{color:colors[idx%colors.length]}
    }));
    Plotly.newPlot('epBarChart',traces,{
      barmode:'group',
      margin:{t:20,b:120,l:60,r:20},
      height:Math.max(400,Math.min(600,ids.length*8)),
      xaxis:{title:'Scenario',tickangle:-45,tickfont:{size:10}},
      yaxis:{title:'Mean Probability',range:[0,1]},
      legend:{orientation:'h',y:1.12}
    },{responsive:true});
  }
  if(typeof Plotly!=='undefined') renderBarChart('all');
})();

// Panel 5: Parameter Estimates
(function(){
  const p=document.getElementById('panel_5');
  const pe=D.parameter_estimates||{};
  if(!pe.weights){placeholder(5,'Parameter estimation not yet run.');return;}
  const w=pe.weights||{},hse=pe.hessian_se||{},bse=pe.bootstrap_se||{};
  const em=pe.estimation_method||{};
  const frs=pe.factor_regression_stats||{};
  const pnames=__PARAM_NAMES__;
  const pdescs=__PARAM_DESCS__;
  const specs=__SPEC_DEFAULTS__;
  const fixed=new Set(__FIXED_PARAMS__);
  let html='<div class="card"><h3>Parameter Estimates</h3>';
  html+='<p style="margin-bottom:8px;color:var(--text-muted)">';
  html+='<span style="background:#d4edda;padding:2px 6px;border-radius:3px;font-size:0.85em">Softmax MLE</span> = identified from action choice probabilities (Stage 4A). ';
  html+='<span style="background:#cce5ff;padding:2px 6px;border-radius:3px;font-size:0.85em">Factor Rating</span> = identified from LLM factor ratings via OLS (Stage 4B). ';
  html+='<span style="background:#fff3cd;padding:2px 6px;border-radius:3px;font-size:0.85em">Fixed</span> = held at spec default (insufficient data).</p>';
  const headers=['Parameter','Method','Description','Spec Default','Estimate','SE','R\u00B2','p-value'];
  const rows=pnames.map(pn=>{
    const method=em[pn]||'fixed';
    const methodLabel=method==='softmax_mle'?'Softmax MLE':method==='factor_rating'?'Factor Rating':method==='pending_4b'?'Fixed':'Fixed';
    const est=w[pn]!==undefined?w[pn]:specs[pn];
    const se_val=method==='softmax_mle'?(bse[pn]>0?bse[pn]:hse[pn]):
                 method==='factor_rating'?hse[pn]:0;
    const fr=frs[pn]||{};
    let r2='--',pv='--';
    if(fr.r_squared!==undefined){r2=fr.r_squared.toFixed(4);}
    if(fr.p_value!==undefined){pv=fr.p_value<0.001?'<0.001':fr.p_value.toFixed(4);}
    else if(method==='softmax_mle'&&se_val>0&&est!==0){
      // Wald test: z = estimate / SE, two-sided p-value
      const z=Math.abs(est/se_val);
      // Normal CDF approx: p ≈ 2*(1 - Φ(z)), using Abramowitz & Stegun 26.2.17
      const t=1/(1+0.2316419*z);
      const d=0.3989422804*Math.exp(-z*z/2);
      const p2=d*t*(0.3193815+t*(-0.3565638+t*(1.781478+t*(-1.8212560+t*1.3302744))));
      const waldP=2*p2;
      pv=waldP<0.001?'<0.001':waldP.toFixed(4);
    }
    return [pn,methodLabel,pdescs[pn]||'',specs[pn]||'--',
      typeof est==='number'?est.toFixed(4):est,
      typeof se_val==='number'&&se_val>0?se_val.toFixed(4):'--',r2,pv];
  });
  rows.push(['lambda','Profiled','Rationality (inv. temp.)',specs.lambda_rationality||1.0,
    (pe.lambda_rationality||0).toFixed(4),'--','--','--']);
  html+=makeTable(headers,rows,'paramTable');
  html+='<p style="margin-top:8px;color:var(--text-muted)">Stage 4A condition number: '+
    (pe.condition_number||'--')+' | Ridge: '+(pe.ridge_applied?'Yes':'No')+
    ' | w10/w11/w14 collapsed: '+(pe.w10_w11_w14_collapsed?'Yes':'No')+'</p>';
  html+='</div>';
  html+='<div class="card"><h3>Forest Plot (All Estimated Parameters)</h3><div id="forestPlot" class="chart"></div></div>';
  p.innerHTML=html;
  // Style rows by estimation method
  const tbl=document.getElementById('paramTable');
  if(tbl){
    const trs=tbl.querySelectorAll('tbody tr, tr');
    trs.forEach(tr=>{
      const cells=tr.querySelectorAll('td');
      if(cells.length>1){
        const m=cells[1].textContent;
        if(m==='Factor Rating'){tr.style.background='#cce5ff';tr.style.color='#004085';}
        else if(m==='Softmax MLE'){tr.style.background='#d4edda';tr.style.color='#155724';}
        else if(m==='Fixed'){tr.style.background='#fff3cd';tr.style.color='#856404';}
      }
    });
  }
  if(typeof Plotly!=='undefined'){
    // Show ALL estimated params (both 4A and 4B) in forest plot
    const allEst=pnames.filter(pn=>em[pn]==='softmax_mle'||em[pn]==='factor_rating');
    const vals=allEst.map(pn=>w[pn]||0);
    const errs=allEst.map(pn=>{
      const m=em[pn];
      let se=0;
      if(m==='softmax_mle'){const b=bse[pn],h=hse[pn];se=(b&&b>0)?b:(h&&h>0)?h:0;}
      else if(m==='factor_rating'){se=hse[pn]||0;}
      return 1.96*se;  // 95% CI
    });
    const colors=allEst.map(pn=>em[pn]==='factor_rating'?'#0066CC':'#4A90D9');
    const symbols=allEst.map(pn=>em[pn]==='factor_rating'?'square':'circle');
    Plotly.newPlot('forestPlot',[
      {y:allEst,x:vals,error_x:{type:'data',array:errs,visible:true,thickness:2,width:4},
       type:'scatter',mode:'markers',name:'Estimated (95% CI)',
       marker:{size:10,color:colors,symbol:symbols}},
      {y:allEst,x:allEst.map(()=>0),type:'scatter',mode:'lines',name:'Zero',
       line:{color:'#ccc',width:1,dash:'dot'},showlegend:false}
    ],{margin:{l:120,r:20,t:20,b:40},height:Math.max(400,allEst.length*35),
       xaxis:{title:'Weight Value (95% CI)',zeroline:true},
       legend:{x:0.7,y:1}},{responsive:true});
  }
})();

// Panel 6: Covariance
(function(){
  const p=document.getElementById('panel_6');
  const pe=D.parameter_estimates||{};
  const cov=D.covariance_matrix;
  if(!cov){placeholder(6,'Covariance matrix not yet computed.');return;}
  const fixed=new Set(__FIXED_PARAMS__);
  const estNames=__PARAM_NAMES__.filter(pn=>!fixed.has(pn));
  let html='<div class="card"><h3>Correlation Matrix (Estimated Parameters)</h3><div id="covHeatmap" class="chart"></div></div>';
  html+='<div class="card"><h3>Correlation Values</h3><div id="corrTable" style="overflow-x:auto"></div></div>';
  p.innerHTML=html;
  const n=cov.length;
  const corr=[];
  for(let i=0;i<n;i++){
    corr.push([]);
    for(let j=0;j<n;j++){
      const d=Math.sqrt(Math.abs(cov[i][i])*Math.abs(cov[j][j]));
      corr[i].push(d>1e-12?cov[i][j]/d:0);
    }
  }
  const labels=estNames.concat(['lambda']);
  const nLabels=labels.length;
  const corrSub=corr.slice(0,nLabels).map(r=>r.slice(0,nLabels));
  if(typeof Plotly!=='undefined'){
    Plotly.newPlot('covHeatmap',[{z:corrSub,x:labels,y:labels,type:'heatmap',
      colorscale:'RdBu',zmin:-1,zmax:1,reversescale:true,
      text:corrSub.map(r=>r.map(v=>v.toFixed(2))),texttemplate:'%{text}',
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
      if((c.a==='w8s'&&c.b==='w_inaction')||(c.a==='w_inaction'&&c.b==='w8s')){
        note=' — <em>Structural: in strike scenarios, CEO is either removed (w8s fires) or retained (w_inaction fires). Near-complementary by design.</em>';
      }
      warn+='<li>'+c.a+' &harr; '+c.b+': <strong>'+c.r.toFixed(2)+'</strong>'+note+'</li>';
    });
    warn+='</ul><p style="margin:4px 0;font-size:0.9em">Correlations |r| &gt; 0.8 indicate parameter trade-offs in estimation. Structural correlations (from utility model design) do not indicate estimation problems if individual SEs are acceptable.</p></div>';
    tDiv.insertAdjacentHTML('beforebegin',warn);
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
      thtml+='<td style="border:1px solid #ddd;padding:4px;text-align:center;background:'+bg+'">'+v.toFixed(2)+'</td>';
    }
    thtml+='</tr>';
  }
  thtml+='</tbody></table>';
  tDiv.innerHTML=thtml;
})();

// Panel 7: Behavioural Diagnostics
(function(){
  const p=document.getElementById('panel_7');
  const bd=D.behavioural_diagnostics;
  if(!bd){placeholder(7,'Diagnostics not yet run.');return;}
  let html='';
  const tests=[
    ['loss_aversion','Loss Aversion (8.1)'],
    ['nonlinearity','Non-Linearity (8.2)'],
    ['optimism_bias','Optimism Bias (8.3)'],
    ['self_assessment_bias','Self-Assessment Bias (8.4)'],
    ['ikea_effect','Ikea Effect (8.5)'],
    ['factor_order_effects','Factor Order Effects (8.6)']
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
      if(t.mean_p_sack_board_initiated!==undefined){
        html+='<table style="border-collapse:collapse;margin:8px 0"><thead><tr><th style="border:1px solid #ddd;padding:6px">Review Origin</th><th style="border:1px solid #ddd;padding:6px">Mean P(sack)</th></tr></thead><tbody>';
        html+='<tr><td style="border:1px solid #ddd;padding:6px">Board-initiated</td><td style="border:1px solid #ddd;padding:6px;text-align:center">'+t.mean_p_sack_board_initiated+'</td></tr>';
        html+='<tr><td style="border:1px solid #ddd;padding:6px">Externally mandated</td><td style="border:1px solid #ddd;padding:6px;text-align:center">'+t.mean_p_sack_externally_mandated+'</td></tr>';
        html+='</tbody></table>';
        if(t.t_stat!==undefined) html+='<p>t-statistic: '+t.t_stat+'</p>';
        html+='<p style="color:var(--text-muted);font-size:0.9em">Hypothesis: board-initiated reviews produce lower P(sack) due to self-serving bias (ownership of review process reduces willingness to act on adverse findings).</p>';
      } else {
        html+='<p style="color:#856404">Insufficient data: need multiple scenarios per review origin group.</p>';
      }
    }
    // Ikea effect detail
    if(key==='ikea_effect'){
      if(t.mean_p_sack_appointed!==undefined){
        html+='<table style="border-collapse:collapse;margin:8px 0"><thead><tr><th style="border:1px solid #ddd;padding:6px">CEO Appointment</th><th style="border:1px solid #ddd;padding:6px">Mean P(sack)</th></tr></thead><tbody>';
        html+='<tr><td style="border:1px solid #ddd;padding:6px">Appointed by current board</td><td style="border:1px solid #ddd;padding:6px;text-align:center">'+t.mean_p_sack_appointed+'</td></tr>';
        html+='<tr><td style="border:1px solid #ddd;padding:6px">Inherited from predecessor</td><td style="border:1px solid #ddd;padding:6px;text-align:center">'+t.mean_p_sack_inherited+'</td></tr>';
        html+='</tbody></table>';
        if(t.t_stat!==undefined) html+='<p>t-statistic: '+t.t_stat+'</p>';
        html+='<p style="color:var(--text-muted);font-size:0.9em">Hypothesis: boards are less likely to sack a CEO they appointed (IKEA effect — overvaluing own creation).</p>';
      } else {
        html+='<p style="color:#856404">Insufficient data: need multiple scenarios per appointment group.</p>';
      }
    }
    // Factor order effects detail
    if(key==='factor_order_effects'&&t.per_factor){
      const pf=t.per_factor;
      const fKeys=Object.keys(pf).sort();
      if(fKeys.length>0){
        html+='<table style="border-collapse:collapse;margin:8px 0;font-size:0.95em"><thead><tr>';
        html+='<th style="border:1px solid #ddd;padding:4px">Factor</th><th style="border:1px solid #ddd;padding:4px">Slope</th>';
        html+='<th style="border:1px solid #ddd;padding:4px">p-value</th><th style="border:1px solid #ddd;padding:4px">Effect</th></tr></thead><tbody>';
        fKeys.forEach(fk=>{
          const f=pf[fk];
          const eff=f.effect||'none';
          const effColor=eff==='none'?'#155724':eff==='primacy'?'#856404':'#721c24';
          html+='<tr><td style="border:1px solid #ddd;padding:4px">'+fk.replace('_',' ')+'</td>';
          html+='<td style="border:1px solid #ddd;padding:4px;text-align:center">'+f.slope+'</td>';
          html+='<td style="border:1px solid #ddd;padding:4px;text-align:center">'+f.p_value+'</td>';
          html+='<td style="border:1px solid #ddd;padding:4px;text-align:center;color:'+effColor+'"><strong>'+eff+'</strong></td></tr>';
        });
        html+='</tbody></table>';
        html+='<p style="color:var(--text-muted);font-size:0.9em">Slope = change in rating per position. Primacy: factors presented earlier get higher ratings. Recency: factors presented later get higher ratings. Factor order is randomised per elicitation.</p>';
      }
      if(t.any_order_effect_detected!==undefined){
        html+='<p>Any order effect detected: <strong>'+(t.any_order_effect_detected?'Yes':'No')+'</strong></p>';
      }
    }
    html+='</div>';
  });
  p.innerHTML=html;
})();

// Panel 8: Interaction Effects
(function(){
  const p=document.getElementById('panel_8');
  const ie=D.interaction_effects;
  if(!ie||Object.keys(ie).length===0){placeholder(8,'Interaction effects not yet computed. Run Stage 6 first.');return;}
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
  if(ie.strike_fit_test&&ie.strike_fit_test.p_value!==undefined){
    const sf=ie.strike_fit_test;
    html+='<p><strong>Strike effect on fit:</strong> Mean KL (strike)='+sf.mean_kl_strike.toFixed(4)+
      ', Mean KL (no strike)='+sf.mean_kl_no_strike.toFixed(4)+
      ', p='+sf.p_value+' — '+sf.conclusion+'</p>';
  }
  if(ie.overwhelming_fit_test&&ie.overwhelming_fit_test.p_value!==undefined){
    const of=ie.overwhelming_fit_test;
    html+='<p><strong>Overwhelming effect on fit:</strong> Mean KL (overwhelming)='+of.mean_kl_overwhelming.toFixed(4)+
      ', Mean KL (not)='+of.mean_kl_not_overwhelming.toFixed(4)+
      ', p='+of.p_value+' — '+of.conclusion+'</p>';
  }
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

// Panel 9: Validation
(function(){
  const p=document.getElementById('panel_9');
  const vr=D.validation_results;
  if(!vr){placeholder(9,'Validation not yet run.');return;}
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

// Panel 10: Linearity Diagnostics
(function(){
  const p=document.getElementById('panel_10');
  const ie=D.interaction_effects;
  const pe=D.parameter_estimates||{};
  const cov=D.covariance_matrix;
  if(!ie||!ie.resid_vs_vote){placeholder(10,'Linearity diagnostics require Stage 6. Run stages 4-6 first.');return;}
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
  html+='<p style="color:var(--text-muted);font-size:0.9em">The softmax model is linear in phi: EU(a) = phi(s,a) &middot; w + anchored. Each row shows the phi basis function for one parameter. High correlation between phi columns creates collinearity (see Covariance tab). Non-linear transformations of scenario features (e.g., (V-0.25)² for w2) embed non-linearity in phi, which is fine — the "linearity" assumption is linearity in w, not in raw features.</p>';
  const pdescs=__PARAM_DESCS__;
  const pnames=__PARAM_NAMES__;
  const fixed=new Set(__FIXED_PARAMS__);
  const estP=pnames.filter(pn=>!fixed.has(pn));
  const phiDefs={
    'w_removal':'-I[CEO removed involuntarily]',
    'w8s':'+I[CEO removed] × I[strike]',
    'w8o':'+I[CEO removed] × I[overwhelming]',
    'w8r':'+I[CEO removed] × I[review adverse]',
    'w_inaction':'-I[strike ∧ CEO present]',
    'w12':'-I[overwhelming ∧ board inactive]',
    'w13':'-I[strike ∧ board inactive]',
    'w15':'-I[review adverse ∧ CEO present]',
    'w1':'-I[CEO resigned early]',
    'w2':'-(V-0.25)² I[V>0.25]',
    'w3':'-I[overwhelming]',
    'w4':'-V × I[strike]',
    'w9':'-I[overwhelming]'
  };
  html+='<table style="border-collapse:collapse;font-size:0.95em"><thead><tr>';
  html+='<th style="border:1px solid #ddd;padding:4px">Parameter</th>';
  html+='<th style="border:1px solid #ddd;padding:4px">Phi Basis</th>';
  html+='<th style="border:1px solid #ddd;padding:4px">Varies Across</th>';
  html+='<th style="border:1px solid #ddd;padding:4px">Type</th></tr></thead><tbody>';
  estP.forEach(pn=>{
    const em=(pe.estimation_method||{})[pn]||'';
    const varies=em==='softmax_mle'?'Actions (within scenario)':'Scenarios only';
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

// Panel 11: Raw Data
(function(){
  const p=document.getElementById('panel_11');
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

    # Meta refresh for in-progress
    meta_refresh = ""
    if dashboard_data.run_status == "in_progress":
        meta_refresh = '<meta http-equiv="refresh" content="30">'

    # Build results JSON
    results_json = dashboard_data.to_json()

    # Inject constants for JS
    # ALL_WEIGHT_NAMES for display (fixed + estimable), FIXED for marking
    param_names_json = json.dumps(list(ALL_WEIGHT_NAMES), ensure_ascii=True)
    param_descs_json = json.dumps(PARAM_DESCRIPTIONS, ensure_ascii=True)
    spec_defaults_json = json.dumps(SPEC_DEFAULTS, ensure_ascii=True)
    fixed_params_json = json.dumps(list(FIXED_PARAM_NAMES), ensure_ascii=True)

    html = _DASHBOARD_TEMPLATE
    html = html.replace("__META_REFRESH__", meta_refresh)
    html = html.replace("__PLOTLY_SCRIPT__", plotly_script)
    html = html.replace("__RESULTS_DATA__", results_json)
    html = html.replace("__PARAM_NAMES__", param_names_json)
    html = html.replace("__PARAM_DESCS__", param_descs_json)
    html = html.replace("__SPEC_DEFAULTS__", spec_defaults_json)
    html = html.replace("__FIXED_PARAMS__", fixed_params_json)

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
    frs = est_result.factor_regression_stats
    # Scenario-level params (Factor Rating or fixed)
    for p in FIXED_PARAM_NAMES:
        method = em.get(p, "fixed")
        fr_stat = frs.get(p, {})
        rows.append({
            "parameter": p,
            "status": method,
            "engine_key": PARAM_TO_ENGINE_KEY.get(p, ""),
            "description": PARAM_DESCRIPTIONS.get(p, ""),
            "spec_default": SPEC_DEFAULTS.get(p, ""),
            "estimate": est_result.weights.get(p, SPEC_DEFAULTS.get(p, "")),
            "hessian_se": est_result.hessian_se.get(p, 0.0),
            "bootstrap_se": est_result.bootstrap_se.get(p, 0.0),
            "r_squared": fr_stat.get("r_squared", ""),
            "p_value": fr_stat.get("p_value", ""),
        })
    # Action-varying params (Softmax MLE)
    for p in WEIGHT_PARAM_NAMES:
        rows.append({
            "parameter": p,
            "status": em.get(p, "softmax_mle"),
            "engine_key": PARAM_TO_ENGINE_KEY.get(p, ""),
            "description": PARAM_DESCRIPTIONS.get(p, ""),
            "spec_default": SPEC_DEFAULTS.get(p, ""),
            "estimate": est_result.weights.get(p, ""),
            "hessian_se": est_result.hessian_se.get(p, ""),
            "bootstrap_se": est_result.bootstrap_se.get(p, ""),
            "r_squared": "",
            "p_value": "",
        })
    rows.append({
        "parameter": "lambda_rationality",
        "status": "profiled",
        "engine_key": "",
        "description": "Rationality (inverse temperature)",
        "spec_default": 1.0,
        "estimate": est_result.lambda_rationality,
        "hessian_se": 0.0,
        "bootstrap_se": 0.0,
        "r_squared": "",
        "p_value": "",
    })

    pd.DataFrame(rows).to_csv(
        output_dir / "parameter_estimates.csv", index=False, encoding="utf-8"
    )

    # Covariance matrix
    cov = est_result.covariance_matrix
    labels = list(WEIGHT_PARAM_NAMES) + ["log_lambda"]
    n = min(cov.shape[0], len(labels))
    cov_df = pd.DataFrame(
        cov[:n, :n],
        columns=labels[:n],
        index=labels[:n],
    )
    cov_df.to_csv(output_dir / "covariance_matrix.csv", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Board Utility Quantification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python board_utility_quantification.py --stage 1\n"
            "  python board_utility_quantification.py --stage 1,2,3 --n_reps 5\n"
            "  python board_utility_quantification.py --all --n_reps 10\n"
        ),
    )
    parser.add_argument("--stage", type=str, default="all",
                        help="Comma-separated stages to run (1-6) or 'all'")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="LLM model for elicitation (default: gpt-4o-mini)")
    parser.add_argument("--n_reps", type=int, default=40,
                        help="Repetitions per scenario (default: 40)")
    parser.add_argument("--n_starts", type=int, default=10,
                        help="L-BFGS-B starting points (default: 10)")
    parser.add_argument("--bootstrap_B", type=int, default=500,
                        help="Bootstrap samples (default: 500)")
    parser.add_argument("--api_key", type=str, default=None,
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR),
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--all", action="store_true",
                        help="Run all stages")

    args = parser.parse_args()

    # Parse stages
    if args.all or args.stage == "all":
        stages = {1, 2, 3, 4, 5, 6}
    else:
        stages = {int(s.strip()) for s in args.stage.split(",")}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_path = output_dir / "pipeline.log"
    _setup_logging(log_path)

    logger.info("=" * 60)
    logger.info("Board Utility Quantification Pipeline")
    logger.info(f"Stages: {sorted(stages)}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Reps/scenario: {args.n_reps}")
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
            render_dashboard(dashboard, dashboard_path)

        # ── Stage 2: LLM elicitation ──
        if 2 in stages and scenarios:
            # Check if we can skip elicitation entirely
            needed_ids = set(s.scenario_id for s in scenarios if s.tier != 4)
            skip_elicitation = False
            if elicitation_path.exists():
                existing_df = pd.read_csv(elicitation_path, encoding="utf-8")
                existing_ids = set(existing_df["scenario_id"].unique())
                if not (needed_ids - existing_ids):
                    logger.info(f"Elicitation results already cover all "
                                f"{len(needed_ids)} scenarios "
                                f"({len(existing_df)} rows). Skipping Stage 2. "
                                f"Use --stage 2 to force re-run.")
                    skip_elicitation = True
            if not skip_elicitation:
                client = _get_instructor_client(args.api_key)
                run_elicitation(
                    scenarios, client, args.model, args.n_reps,
                    elicitation_path, cost_tracker,
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
        if 3 in stages and elicitation_path.exists():
            estimation_df = preprocess_data(elicitation_path, estimation_path)
            if not estimation_df.empty:
                dashboard.estimation_dataset_summary = {
                    "n_scenarios": len(estimation_df),
                    "mean_seed_variance": round(
                        float(estimation_df["mean_seed_variance"].mean()), 4
                    ),
                }
            render_dashboard(dashboard, dashboard_path)
        elif estimation_path.exists():
            estimation_df = pd.read_csv(estimation_path, encoding="utf-8")

        # Build elicited probabilities data for dashboard
        if not estimation_df.empty and scenarios:
            scenario_lookup = {s.scenario_id: s for s in scenarios}
            ep_rows = []
            for _, row in estimation_df.iterrows():
                sid = row["scenario_id"]
                sc = scenario_lookup.get(sid)
                if sc is None:
                    continue
                sv = sc.state_vector if isinstance(sc.state_vector, dict) else {}
                ep_rows.append({
                    "scenario_id": sid,
                    "tier": sc.tier,
                    "target_parameter": sc.target_parameter,
                    "decision_node": sc.decision_node,
                    "vote_pct": sv.get("vote_outcome_V"),
                    "mean_probs": json.loads(row["mean_prob_vector"]),
                    "n_seeds": int(row["n_successful_seeds"]),
                })
            dashboard.elicited_probabilities = ep_rows
            render_dashboard(dashboard, dashboard_path)

        # ── Stage 4: Parameter estimation ──
        if 4 in stages and not estimation_df.empty and scenarios:
            phi, anchored, p_llm, scenario_ids, action_lists = compute_phi_matrix(
                scenarios, estimation_df,
            )

            est_result = estimate_parameters(
                phi, anchored, p_llm, action_lists,
                n_starts=args.n_starts,
                bootstrap_B=args.bootstrap_B,
            )

            # ── Stage 4B: Factor rating regression for scenario-level params ──
            stage4b = estimate_scenario_level_params(
                scenarios, estimation_df, est_result,
            )
            # Merge 4B estimates into est_result
            for p in FIXED_PARAM_NAMES:
                if p in stage4b["weights"] and stage4b["weights"][p] > 0:
                    est_result.weights[p] = stage4b["weights"][p]
                    est_result.hessian_se[p] = stage4b["se"].get(p, 0.0)
                    est_result.bootstrap_se[p] = 0.0  # no bootstrap for 4B
                    est_result.estimation_method[p] = "factor_rating"
            est_result.factor_regression_stats = stage4b.get("regression_stats", {})

            _save_parameter_estimates(est_result, output_dir)

            dashboard.parameter_estimates = est_result.to_dict()
            cov = est_result.covariance_matrix
            n = min(cov.shape[0], len(WEIGHT_PARAM_NAMES) + 1)
            dashboard.covariance_matrix = cov[:n, :n].tolist()
            render_dashboard(dashboard, dashboard_path)

        # ── Stage 5: Behavioural diagnostics ──
        if 5 in stages and not estimation_df.empty and est_result is not None:
            diagnostics = run_diagnostics(
                scenarios, estimation_df, est_result,
                elicitation_path, diagnostics_path,
            )
            dashboard.behavioural_diagnostics = diagnostics
            render_dashboard(dashboard, dashboard_path)

        # ── Stage 6: Validation ──
        if 6 in stages and est_result is not None and not estimation_df.empty:
            # Recompute phi if needed
            if "phi" not in dir():
                phi, anchored, p_llm, scenario_ids, action_lists = compute_phi_matrix(
                    scenarios, estimation_df,
                )

            n_sc, max_a = p_llm.shape
            action_masks = np.zeros((n_sc, max_a), dtype=bool)
            for i, actions in enumerate(action_lists):
                for j in range(len(actions)):
                    action_masks[i, j] = True

            validation = run_validation(
                scenarios, estimation_df, est_result,
                phi, anchored, p_llm, action_masks,
                scenario_ids, action_lists, output_dir,
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
