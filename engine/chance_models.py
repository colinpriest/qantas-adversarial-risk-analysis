"""
Chance models: Vote and Review.

These models produce stochastic outcomes at chance nodes in the game tree,
conditioned on belief draws and the history of actions taken.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from engine.state import BeliefBundle, DecisionState


@dataclass
class VoteOutcome:
    """Result of the vote chance node."""
    vote_percent: float          # Fraction voting against remuneration report
    strike_indicator: bool       # vote_percent >= first_strike threshold
    overwhelming_indicator: bool # vote_percent >= overwhelming threshold


@dataclass
class ReviewOutcome:
    """Result of the review chance node.

    The review produces a continuous abnormal return (CAR) from the findings
    release window, calibrated from ASX governance review case studies
    (board-background/governance-review-case-studies.md).

    review_car: Cumulative abnormal return from the findings release.
        Negative values indicate adverse market reaction (e.g., Star -13.95%).
        Positive values indicate market relief (e.g., Qantas +0.85%).
    review_adverse: Derived from review_car < 0. Used for state transitions
        (e.g., Board can sack CEO after adverse findings).
    """
    review_adverse: bool   # Derived: review_car < 0
    review_car: float = 0.0  # Abnormal return from findings release


@dataclass
class OverconfidenceBias:
    """Board overconfidence bias on governance effect estimates and review.

    Calibrated from literature review (board-background/
    literature-review-Board-overconfidence.pdf), which synthesises:
    - Twardawski & Kind 2023, Brahma et al. 2023 (M&A board overconfidence)
    - Coffeng et al. 2021 (20% boards choose best option, high satisfaction)
    - Boundy-Singer et al. 2022, Guggenmos 2021 (metacognitive miscalibration)
    - Ertimur et al. 2011, Fan & Radhakrishnan 2020 (shareholder voting corrections)

    Three components:

    Overestimation (mean bias) — vote:
        μ̂ = (1+β) × μ*, β ~ U(0.25, 1.0).  Boards overestimate the
        effectiveness of their governance actions by 25–100%.
        → d1_floor raised (Board thinks review is at least ~63% effective)
        → d3_floor raised (Board underestimates CEO-exit backlash)

    Overprecision (variance bias) — vote:
        σ̂² = σ*² / κ, κ ~ U(2, 5).  Boards perceive 2–5× more precision.
        → sigma_scale = 1/√κ shrinks sigma_vote in the Board's EU calc.

    Overestimation (mean bias) — review CAR:
        Board overestimates governance quality, so believes the review
        findings will produce a more favourable market reaction.
        μ_f_biased = μ_f + review_car_bias  (shifts CAR location upward)
        Calibrated from case studies (board-background/
        governance-review-case-studies.md). With β=0.625: Board perceives
        review CAR ~3pp more favourable than actuarial (-2% vs -5%).

    Production defaults use midpoints (β=0.625, κ=3.5):
        D1 ~ U(0.63, 1.0), D3 ~ U(-0.62, 0.0), sigma_scale = 0.53,
        review_car_bias = 0.03
    """
    # D1 review effect bounds: unbiased = U(0, 1)
    d1_floor: float = 0.0
    d1_ceiling: float = 1.0
    # D3 CEO exit effect bounds: unbiased = U(-1, 0)
    d3_floor: float = -1.0
    d3_ceiling: float = 0.0
    # Sigma scale: multiplier on sigma_vote in biased EU calculation.
    # = 1/√κ where κ ~ U(2,5) is the overprecision factor.
    sigma_scale: float = 1.0
    # Review CAR bias: positive value ADDED to the review CAR location
    # parameter μ_f. Board overestimates governance quality → believes
    # review findings will produce a better market reaction. Unbiased = 0.0.
    review_car_bias: float = 0.0


# Pre-configured bias profiles (for reference and testing).
# The production default is loaded from the board_overconfidence sheet
# of governance_spec.xlsx. See engine/state.py: load_board_overconfidence().
BIAS_NONE = OverconfidenceBias()

BIAS_OVERESTIMATION = OverconfidenceBias(
    d1_floor=0.50, d1_ceiling=1.0,    # β=0.5: D1 mean 0.75 vs true 0.5
    d3_floor=-0.67, d3_ceiling=0.0,   # β=0.5: D3 mean -0.33 vs true -0.5
    review_car_bias=0.025,            # β=0.5: Board perceives CAR ~2.5pp higher
)

BIAS_OVERPRECISION = OverconfidenceBias(
    d1_floor=0.63, d1_ceiling=1.0,    # β=0.625 (midpoint)
    d3_floor=-0.62, d3_ceiling=0.0,   # β=0.625 (midpoint)
    sigma_scale=0.45,                 # κ=5: strong overprecision (1/√5)
    review_car_bias=0.03,             # β=0.625: Board perceives CAR ~3pp higher
)

BIAS_HUBRIS = OverconfidenceBias(
    d1_floor=0.63, d1_ceiling=1.0,    # β=0.625: midpoint overestimation
    d3_floor=-0.62, d3_ceiling=0.0,   # β=0.625: midpoint overestimation
    sigma_scale=0.53,                 # κ=3.5: midpoint overprecision (1/√3.5)
    review_car_bias=0.03,             # β=0.625: Board perceives CAR ~3pp higher
)


def _expit(x: float) -> float:
    """Logistic sigmoid (inverse logit), numerically stable."""
    if x >= 0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = np.exp(x)
        return z / (1.0 + z)


class VoteModel:
    """
    Endogenous vote model.

    Computes:
        B_agm = B_mkt[i] + gamma_A[i] * 1{A2=RecStrike} + gamma_D[i] * f(D1)
        logit(V) ~ N(alpha_vote[i] + B_agm, sigma_vote[i])

    Governance effects — empirical priors from ranked_voting_recommendations.csv:
        The data shows a NON-MONOTONIC relationship:
          D0 (No Action):  +32.8% avg voting diff  — baseline anchor = 0
          D1 (Review):     +14.9% avg voting diff  — most effective mitigation
          D3 (CEO Exit):   +46.7% avg voting diff  — signals crisis, increases protest

        D1 REDUCES protest (positive effect × negative gamma_D → lower B_agm).
        D3 INCREASES protest (negative effect × negative gamma_D → higher B_agm).
        Magnitudes are uncertain: D1 ~ +U(0,1), D3 ~ -U(0,1).
        The overall scale is controlled by gamma_D (from the posterior).
    """

    def __init__(self, vote_thresholds: dict[str, float]):
        self.first_strike = vote_thresholds["first_strike"]
        self.overwhelming = vote_thresholds["overwhelming"]

    def sample(
        self,
        draw_i: int,
        beliefs: BeliefBundle,
        history: dict,
        state: DecisionState,
        rng: np.random.Generator,
        governance_effect: Optional[float] = None,
        sigma_scale: Optional[float] = None,
    ) -> VoteOutcome:
        """Sample a vote outcome for a given belief draw and game history.

        Args:
            governance_effect: Pre-drawn governance effect. If None, draws a
                fresh sample. This should be drawn ONCE per scenario (per
                belief draw) and held fixed across MC vote samples, because
                it represents epistemic uncertainty (how effective IS this
                reform), not aleatoric noise per vote.
            sigma_scale: Multiplier on sigma_vote. When < 1, the Board
                underestimates vote uncertainty (overprecision). Used only
                in the focal actor's biased EU calculation.
        """
        params = beliefs.get_draw(draw_i)

        # Construct AGM belief
        B_agm = params["B_mkt"]

        # ASA mobilisation effect
        if history.get("A2") == "A2_rec_strike":
            B_agm += params["gamma_A"]

        # Governance package effect
        d1_action = history.get("D1", "D0_minimal")
        if governance_effect is None:
            governance_effect = self._governance_effect(d1_action, rng)
        B_agm += params["gamma_D"] * governance_effect

        # Sample vote from logit-normal
        logit_mean = params["alpha_vote"] + B_agm
        sigma = params["sigma_vote"]
        if sigma_scale is not None:
            sigma *= sigma_scale
        logit_v = rng.normal(logit_mean, max(sigma, 1e-6))
        vote_percent = _expit(logit_v)

        return VoteOutcome(
            vote_percent=vote_percent,
            strike_indicator=vote_percent >= self.first_strike,
            overwhelming_indicator=vote_percent >= self.overwhelming,
        )

    def expected_vote(
        self,
        draw_i: int,
        beliefs: BeliefBundle,
        history: dict,
        state: DecisionState,
    ) -> float:
        """Return expected vote percent (no sampling, uses E[effect] for each action).

        Expected governance effects (mean of Uniform distributions):
          D0 = 0, D1 = +0.5 (reduces protest), D3 = -0.5 (signals crisis)
        """
        params = beliefs.get_draw(draw_i)
        B_agm = params["B_mkt"]

        if history.get("A2") == "A2_rec_strike":
            B_agm += params["gamma_A"]

        d1_action = history.get("D1", "D0_minimal")
        # Non-monotonic: D1 mitigates, D3 escalates
        effect = {"D0_minimal": 0.0, "D1_review": 0.5, "D3_ceo_transition": -0.5}
        B_agm += params["gamma_D"] * effect.get(d1_action, 0.0)

        logit_mean = params["alpha_vote"] + B_agm
        return _expit(logit_mean)

    @staticmethod
    def _governance_effect(
        d1_action: str,
        rng: np.random.Generator,
        bias: Optional[OverconfidenceBias] = None,
    ) -> float:
        """Sample governance effect on protest vote (non-monotonic).

        Empirical basis: data/ranked_voting_recommendations.csv

        D0 = 0  (anchor: no action, baseline)
        D1 ~ +U(floor, ceiling)  (review REDUCES protest)
        D3 ~ U(floor, ceiling)   (CEO exit INCREASES protest)

        Unbiased: D1 ~ U(0, 1), D3 ~ U(-1, 0)
        With overconfidence bias: floors/ceilings shift per the bias config.

        When multiplied by gamma_D (negative from posterior):
          D1: positive * negative → lowers B_agm → lowers protest vote
          D3: negative * negative → raises B_agm → raises protest vote
        """
        if d1_action == "D0_minimal":
            return 0.0
        elif d1_action == "D1_review":
            lo = bias.d1_floor if bias else 0.0
            hi = bias.d1_ceiling if bias else 1.0
            return float(rng.uniform(lo, hi))
        elif d1_action == "D3_ceo_transition":
            lo = bias.d3_floor if bias else -1.0
            hi = bias.d3_ceiling if bias else 0.0
            return float(rng.uniform(lo, hi))
        else:
            return 0.0


class ReviewModel:
    """
    Review findings CAR model.

    Calibrated from ASX governance review case studies 2014–2023
    (board-background/governance-review-case-studies.md).

    If review not commissioned: deterministic CAR = 0.
    If commissioned, the findings release window CAR is sampled from:

        μ_f ~ Cauchy(-0.05, 0.03)     Location: centered on -5% drop
        σ_f ~ Half-Normal(0.10)       Scale: high volatility
        AR  ~ Student-t(ν=3, μ_f, σ_f)  Heavy tails for black swan events

    This captures the extreme asymmetry observed empirically:
        Star -13.95%, Westpac -3.00%, CBA +1.75%, Qantas +0.85%.
    The ν=3 degrees of freedom produce heavy tails that accommodate
    existential events (Star) while allowing relief rallies (CBA, Qantas).

    review_adverse is derived from CAR < 0 (for state transitions).
    """

    # Distribution hyperparameters (from case study calibration)
    NU = 3              # Degrees of freedom: heavy tails
    MU_LOC = -0.05      # Cauchy location for μ_f: centered on -5%
    MU_SCALE = 0.03     # Cauchy scale for μ_f: broad uncertainty
    SIGMA_SCALE = 0.10  # Half-Normal scale for σ_f: high volatility

    def sample(
        self,
        draw_i: int,
        beliefs: BeliefBundle,
        history: dict,
        state: DecisionState,
        rng: np.random.Generator,
        bias: Optional[OverconfidenceBias] = None,
    ) -> ReviewOutcome:
        """Sample a review CAR from the Student-t hierarchy.

        Args:
            bias: If set, shifts the location parameter μ_f upward
                (Board thinks findings will be more favourable).
        """
        if not state.review_commissioned:
            return ReviewOutcome(review_adverse=False, review_car=0.0)

        # Sample hierarchical parameters
        mu_f = self.MU_LOC + self.MU_SCALE * rng.standard_cauchy()
        sigma_f = abs(rng.normal(0, self.SIGMA_SCALE))

        # Board overconfidence: believes findings will be more favourable
        if bias is not None and bias.review_car_bias != 0.0:
            mu_f += bias.review_car_bias

        # Sample CAR from Student-t
        car = mu_f + max(sigma_f, 1e-6) * rng.standard_t(self.NU)

        return ReviewOutcome(
            review_adverse=car < 0,
            review_car=float(car),
        )

    def expected_car(
        self,
        state: DecisionState,
        bias: Optional[OverconfidenceBias] = None,
    ) -> float:
        """Return expected CAR (location of the distribution, no sampling).

        Uses the Cauchy location parameter as a point estimate for the
        expected CAR. Cauchy has no defined mean, but the location is the
        center of the distribution.
        """
        if not state.review_commissioned:
            return 0.0

        mu_f = self.MU_LOC
        if bias is not None and bias.review_car_bias != 0.0:
            mu_f += bias.review_car_bias

        return mu_f


class ReviewDirectCostModel:
    """
    Direct cost of commissioning an external governance review.

    Calibrated from board-background/direct-costs-governance-review.md.
    Three cost components: reviewer fees, management distraction,
    internal resource consumption.

    The total direct cost follows a Gamma distribution in decimal CAR units
    (positive values, subtracted from utility). Parameterised as:

        C_direct ~ Gamma(α=4.55, rate β=4741)

    Properties:
        Mean:   0.00096  (≈9.6 bps)
        SD:     0.00045  (≈4.5 bps)
        Mode:   0.00075  (≈7.5 bps)
        5th %:  0.00031  (≈3.1 bps)
        95th %: 0.00185  (≈18.5 bps)

    Calibrated for a reference market cap of AUD 10 billion (Qantas 2023–24).
    The positive skewness (0.94) reflects asymmetric risk: management
    distraction can escalate if the review becomes prolonged or contested,
    but there is a natural floor on costs (reviewer fees).
    """

    # Gamma parameters (shape/rate from research document §6.2)
    ALPHA = 4.55          # Shape: controls skewness and tail weight
    BETA = 4741.0         # Rate: controls scale (1/β ≈ 0.000211)
    SCALE = 1.0 / 4741.0  # NumPy uses shape/scale parameterisation

    def sample(self, rng: np.random.Generator) -> float:
        """Sample total direct cost in decimal CAR (positive value).

        Returns a positive value representing the direct cost of commissioning
        the review (fees + distraction + internal resources), expressed as
        decimal CAR. This is subtracted from utility via review_direct_cost_weight.
        """
        return float(rng.gamma(self.ALPHA, self.SCALE))

    def expected_cost(self) -> float:
        """Return expected direct cost (mean of the Gamma distribution)."""
        return self.ALPHA * self.SCALE  # ≈ 0.00096


class ChanceModels:
    """Container for all chance models (vote, review, review direct cost)."""

    def __init__(self, vote_thresholds: dict[str, float]):
        self.vote = VoteModel(vote_thresholds)
        self.review = ReviewModel()
        self.review_direct_cost = ReviewDirectCostModel()

    def sample_vote(
        self, draw_i: int, beliefs: BeliefBundle,
        history: dict, state: DecisionState,
        rng: np.random.Generator,
        governance_effect: Optional[float] = None,
        sigma_scale: Optional[float] = None,
    ) -> VoteOutcome:
        return self.vote.sample(
            draw_i, beliefs, history, state, rng,
            governance_effect=governance_effect,
            sigma_scale=sigma_scale,
        )

    def sample_review(
        self, draw_i: int, beliefs: BeliefBundle,
        history: dict, state: DecisionState,
        rng: np.random.Generator,
        bias: Optional[OverconfidenceBias] = None,
    ) -> ReviewOutcome:
        return self.review.sample(draw_i, beliefs, history, state, rng, bias=bias)

    def sample_review_direct_cost(self, rng: np.random.Generator) -> float:
        """Sample direct cost of review in decimal CAR (positive value)."""
        return self.review_direct_cost.sample(rng)
