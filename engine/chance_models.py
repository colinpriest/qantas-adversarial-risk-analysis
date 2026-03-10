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

    The review produces:
    1. A qualitative outcome rating: "negative", "balanced", or "positive"
       (equal 1/3 probabilities from Dirichlet(5,5,5)).
    2. A continuous abnormal return (CAR) from the findings release window,
       calibrated from ASX governance review case studies.

    review_outcome: Qualitative rating ("negative", "balanced", "positive",
        or "none" if review not commissioned).
    review_car: Cumulative abnormal return from findings release.
    """
    review_outcome: str = "none"  # "none", "negative", "balanced", "positive"
    review_car: float = 0.0      # Abnormal return from findings release


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
        D1 ~ U(0.63, 1.0), D3 ~ U(-0.62, 0.5), sigma_scale = 0.53,
        review_car_bias = 0.03
    """
    # D1 review effect bounds: unbiased = U(0, 1)
    d1_floor: float = 0.0
    d1_ceiling: float = 1.0
    # D3 CEO exit effect bounds: unbiased = U(-1, 0.5)
    d3_floor: float = -1.0
    d3_ceiling: float = 0.5
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
    d3_floor=-0.67, d3_ceiling=0.5,   # β=0.5: D3 mean -0.085 vs true -0.25
    review_car_bias=0.025,            # β=0.5: Board perceives CAR ~2.5pp higher
)

BIAS_OVERPRECISION = OverconfidenceBias(
    d1_floor=0.63, d1_ceiling=1.0,    # β=0.625 (midpoint)
    d3_floor=-0.62, d3_ceiling=0.5,   # β=0.625 (midpoint)
    sigma_scale=0.45,                 # κ=5: strong overprecision (1/√5)
    review_car_bias=0.03,             # β=0.625: Board perceives CAR ~3pp higher
)

BIAS_HUBRIS = OverconfidenceBias(
    d1_floor=0.63, d1_ceiling=1.0,    # β=0.625: midpoint overestimation
    d3_floor=-0.62, d3_ceiling=0.5,   # β=0.625: midpoint overestimation
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
    Endogenous vote model (v2).

    Computes:
        B_agm = B_mkt[i]
              + gamma_A[i] * 1{A2=RecStrike}
              + gamma_AH[i] * 1{A2=RecStrike} * 1{headline=1}
              + gamma_D[i] * f(D1)
        logit(V) ~ N(alpha_vote[i] + B_agm, sigma_vote[i])

    When headline_incident=1, a structural floor is applied:
        V_final = max(V_logit_normal, V_floor)
        V_floor ~ Beta(50, 150)  (mean 0.25, drawn once per belief draw)

    Governance effects — empirical priors from ranked_voting_recommendations.csv:
        The data shows a NON-MONOTONIC relationship:
          D0 (No Action):  +32.8% avg voting diff  — baseline anchor = 0
          D1 (Review):     +14.9% avg voting diff  — most effective mitigation
          D3 (CEO Exit):   +46.7% avg voting diff  — ambiguous, confounded with severity

        D1 REDUCES protest (positive effect × negative gamma_D → lower B_agm).
        D3 effect is AMBIGUOUS: U(-1, 0.5) allows both amplification and mitigation.
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
        crisis_floor: Optional[float] = None,
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
            crisis_floor: Pre-drawn structural floor V_floor ~ Beta(50, 150).
                When headline_incident=1, V_final = max(V_logit_normal, V_floor).
                Drawn ONCE per belief draw (epistemic), not per vote sample.
        """
        params = beliefs.get_draw(draw_i)

        # Construct AGM belief
        B_agm = params["B_mkt"]

        # ASA mobilisation effect (base + headline interaction)
        if history.get("A2") == "A2_rec_strike":
            B_agm += params["gamma_A"]
            if state.headline_incident:
                B_agm += params["gamma_AH"]

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

        # Structural floor for crisis scenarios (shareholder-vote-V2.md §1.4)
        if crisis_floor is not None:
            vote_percent = max(vote_percent, crisis_floor)

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
          D0 = 0, D1 = +0.5 (reduces protest), D3 = -0.25 (ambiguous, E[U(-1,0.5)])
        """
        params = beliefs.get_draw(draw_i)
        B_agm = params["B_mkt"]

        if history.get("A2") == "A2_rec_strike":
            B_agm += params["gamma_A"]
            if state.headline_incident:
                B_agm += params["gamma_AH"]

        d1_action = history.get("D1", "D0_minimal")
        # Non-monotonic: D1 mitigates, D3 ambiguous (E[U(-1,0.5)] = -0.25)
        effect = {"D0_minimal": 0.0, "D1_review": 0.5, "D3_ceo_transition": -0.25}
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
        D3 ~ U(floor, ceiling)   (CEO exit effect AMBIGUOUS)

        Unbiased: D1 ~ U(0, 1), D3 ~ U(-1, 0.5)
        With overconfidence bias: floors/ceilings shift per the bias config.

        When multiplied by gamma_D (negative from posterior):
          D1: positive * negative → lowers B_agm → lowers protest vote
          D3: negative f → raises B_agm; positive f → lowers B_agm
              Asymmetric bounds (-1 to 0.5) encode prior that amplification
              is ~3x more likely than mitigation (shareholder-vote-V2.md §1.3).
        """
        if d1_action == "D0_minimal":
            return 0.0
        elif d1_action == "D1_review":
            lo = bias.d1_floor if bias else 0.0
            hi = bias.d1_ceiling if bias else 1.0
            return float(rng.uniform(lo, hi))
        elif d1_action == "D3_ceo_transition":
            lo = bias.d3_floor if bias else -1.0
            hi = bias.d3_ceiling if bias else 0.5
            return float(rng.uniform(lo, hi))
        else:
            return 0.0


class ReviewModel:
    """
    Review findings model: trinary outcome rating + CAR.

    Two components:

    1. **Outcome rating** (negative / balanced / positive):
       Grounded in external-review-distributions.md posterior analysis.
       The Qantas review is board-initiated during a crisis with
       pre-existing reputational damage (ACCC ghost flights, Senate inquiry),
       so the outcome follows Dirichlet(38, 160, 1) over
       {Negative, Balanced, Positive}.

           (p_neg, p_bal, p_pos) ~ Dirichlet(38, 160, 1)
           E = (0.191, 0.804, 0.005)

       Balanced/neutral dominates (~80%) because board-commissioned reviews
       in crisis contexts admit "mistakes were made" without conceding
       legal liability.  Negative is material (~19%) due to ACCC severity.
       Positive is negligible (<1%) — a clean bill of health during active
       litigation would be non-credible.

       Drawn ONCE per belief draw (epistemic uncertainty about the review
       process), then each MC sample draws outcome ~ Categorical(p).

    2. **CAR** (market reaction to findings release):
       Student-t hierarchy calibrated from ASX governance review case
       studies 2014–2023 (board-background/governance-review-case-studies.md):

           μ_f ~ Student-t(ν=4, -0.05, 0.03)
           σ_f ~ Half-Normal(0.10)
           CAR ~ Student-t(ν=3, μ_f, σ_f)

       The CAR captures the market's quantitative reaction. It is
       separate from the qualitative outcome rating: a "positive" review
       could still produce a mildly negative CAR (market expected more),
       and a "negative" review could produce a mildly positive CAR
       (market had already priced in the bad news).
    """

    # CAR distribution hyperparameters (from case study calibration)
    NU = 3              # Degrees of freedom for CAR: heavy tails
    MU_NU = 4           # Degrees of freedom for μ_f: finite mean + variance
    MU_LOC = -0.05      # Location for μ_f: centred on -5% drop
    MU_SCALE = 0.03     # Scale for μ_f: uncertainty about market read
    SIGMA_SCALE = 0.10  # Half-Normal scale for σ_f: high volatility

    # Outcome rating: Dirichlet(38, 160, 1) → E = (0.191, 0.804, 0.005)
    # See external-review-distributions.md posterior analysis
    REVIEW_OUTCOMES = ("negative", "balanced", "positive")
    DIRICHLET_ALPHA = np.array([38.0, 160.0, 1.0])  # (neg, bal, pos)

    def draw_outcome_probabilities(
        self,
        rng: np.random.Generator,
        bias: Optional[OverconfidenceBias] = None,
    ) -> np.ndarray:
        """Draw (p_negative, p_balanced, p_positive) ~ Dirichlet(38, 160, 1).

        Drawn ONCE per belief draw (epistemic uncertainty about review process).
        Returns array of 3 probabilities summing to 1.

        When bias is set, the Board overestimates governance quality,
        inflating the positive pseudo-count:
            α_positive_biased = 1 × (1 + 10 × review_car_bias)
        With default bias (0.03): Dirichlet(38, 160, 1.3) → slight tilt toward positive.
        """
        alpha = self.DIRICHLET_ALPHA.copy()
        if bias is not None and bias.review_car_bias > 0:
            alpha[2] = alpha[2] * (1.0 + 10.0 * bias.review_car_bias)
        return rng.dirichlet(alpha)

    def sample(
        self,
        draw_i: int,
        beliefs: BeliefBundle,
        history: dict,
        state: DecisionState,
        rng: np.random.Generator,
        bias: Optional[OverconfidenceBias] = None,
        outcome_probs: Optional[np.ndarray] = None,
    ) -> ReviewOutcome:
        """Sample a review outcome (trinary rating + CAR).

        Args:
            bias: If set, shifts the location parameter μ_f upward
                (Board thinks findings will be more favourable).
            outcome_probs: Pre-drawn (p_neg, p_bal, p_pos) from Dirichlet(38, 160, 1).
                Drawn ONCE per belief draw via draw_outcome_probabilities().
                If None, uses equal 1/3 probabilities.
        """
        if not state.review_commissioned:
            return ReviewOutcome(review_outcome="none", review_car=0.0)

        # Sample hierarchical CAR parameters
        mu_f = self.MU_LOC + self.MU_SCALE * rng.standard_t(self.MU_NU)
        sigma_f = abs(rng.normal(0, self.SIGMA_SCALE))

        # Board overconfidence: believes findings will be more favourable
        if bias is not None and bias.review_car_bias != 0.0:
            mu_f += bias.review_car_bias

        # Sample CAR from Student-t
        car = mu_f + max(sigma_f, 1e-6) * rng.standard_t(self.NU)

        # Outcome rating: Dirichlet-based or equal probability fallback
        if outcome_probs is not None:
            idx = rng.choice(3, p=outcome_probs)
        else:
            idx = rng.choice(3)  # Equal 1/3 probabilities
        outcome = self.REVIEW_OUTCOMES[idx]

        return ReviewOutcome(
            review_outcome=outcome,
            review_car=float(car),
        )

    def expected_car(
        self,
        state: DecisionState,
        bias: Optional[OverconfidenceBias] = None,
    ) -> float:
        """Return expected CAR (analytical mean of the distribution).

        E[CAR] = E[μ_f] = MU_LOC (since E[t(ν)] = 0 for ν > 1).
        With t(4) for μ_f, the mean is well-defined and equals MU_LOC.
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
        crisis_floor: Optional[float] = None,
    ) -> VoteOutcome:
        return self.vote.sample(
            draw_i, beliefs, history, state, rng,
            governance_effect=governance_effect,
            sigma_scale=sigma_scale,
            crisis_floor=crisis_floor,
        )

    def sample_review(
        self, draw_i: int, beliefs: BeliefBundle,
        history: dict, state: DecisionState,
        rng: np.random.Generator,
        bias: Optional[OverconfidenceBias] = None,
        outcome_probs: Optional[np.ndarray] = None,
    ) -> ReviewOutcome:
        return self.review.sample(
            draw_i, beliefs, history, state, rng,
            bias=bias, outcome_probs=outcome_probs,
        )

    def sample_review_direct_cost(self, rng: np.random.Generator) -> float:
        """Sample direct cost of review in decimal CAR (positive value)."""
        return self.review_direct_cost.sample(rng)
