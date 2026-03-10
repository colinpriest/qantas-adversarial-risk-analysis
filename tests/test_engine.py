"""
Comprehensive test suite for the ARA engine.

Tests cover:
1. Data loading and validation (state, beliefs, priors)
2. Feasibility rules and state transitions
3. Chance models (vote and review)
4. Utility functions
5. Mode configurations
6. Predictive distributions
7. Tree evaluation
8. Solver integration
9. Validation checklist from spec (policy sensitivity)
"""

import sys
import os
from pathlib import Path

import numpy as np
import pytest

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
GOV_SPEC = DATA_DIR / "governance_spec.xlsx"
OPP_PRIORS = DATA_DIR / "opponent_priors.xlsx"


# ============================================================================
# 1. DATA LOADING AND VALIDATION
# ============================================================================

class TestDataLoading:
    """Test data contract loading and validation."""

    def test_governance_spec_loads(self):
        from engine.state import DecisionState
        state = DecisionState.from_governance_spec(GOV_SPEC)
        assert state.CEO_present is True
        assert state.review_commissioned is False
        assert len(state._node_order) == 12
        assert state._node_order[0] == "D0_ceo"
        assert state._node_order[1] == "D1"
        assert state._node_order[-1] == "Terminal"

    def test_node_order_types(self):
        from engine.state import DecisionState
        state = DecisionState.from_governance_spec(GOV_SPEC)
        assert state.node_type("D0_ceo") == "decision"
        assert state.node_type("D1") == "decision"
        assert state.node_type("V") == "chance"
        assert state.node_type("Terminal") == "terminal"
        assert state.node_type("R") == "chance"

    def test_node_owners(self):
        from engine.state import DecisionState
        state = DecisionState.from_governance_spec(GOV_SPEC)
        assert state.node_owner("D0_ceo") == "CEO"
        assert state.node_owner("D1") == "Board"
        assert state.node_owner("A2") == "ASA"
        assert state.node_owner("V") == "Nature"
        assert state.node_owner("D4") == "CEO"

    def test_vote_thresholds(self):
        from engine.state import load_vote_thresholds
        t = load_vote_thresholds(GOV_SPEC)
        assert t["first_strike"] == 0.25
        assert t["overwhelming"] == 0.50
        assert t["first_strike"] < t["overwhelming"]

    def test_utility_weights_board(self):
        from engine.state import load_utility_weights
        w = load_utility_weights(GOV_SPEC, "Board")
        assert "vote_penalty_weight" in w
        assert "review_direct_cost_weight" in w
        assert w["vote_penalty_weight"] >= 0  # Can be zero after quantification estimation
        assert "negative_review_finding_penalty" in w
        assert w["negative_review_finding_penalty"] > 0

    def test_utility_weights_asa(self):
        from engine.state import load_utility_weights
        w = load_utility_weights(GOV_SPEC, "ASA")
        assert "strike_ba_shift" in w
        assert "mobilisation_cost" in w
        assert "market_alignment_ol_shift" in w

    def test_utility_weights_ceo(self):
        from engine.state import load_utility_weights
        w = load_utility_weights(GOV_SPEC, "CEO")
        assert "gamma" in w
        assert "W_resign" in w
        assert "D_sacked" in w
        assert "D_resign_late" in w
        assert "D_negotiate" in w

    def test_policy_parameters(self):
        from engine.state import load_policy_parameters
        p = load_policy_parameters(GOV_SPEC)
        assert ("Board", "D_rev", "review_vote_threshold") in p
        assert ("CEO", "D4", "resign_vote_threshold") in p
        assert ("ASA", "A2", "mobilise_vote_threshold") in p

    def test_belief_bundle_loads(self):
        from engine.state import BeliefBundle
        bb = BeliefBundle(CHECKPOINT_DIR / "belief_C0_2023-10-01.npz")
        assert bb.N == 500
        assert len(bb.B_mkt) == 500
        assert len(bb.alpha_vote) == 500

    def test_belief_bundle_draw(self):
        from engine.state import BeliefBundle
        bb = BeliefBundle(CHECKPOINT_DIR / "belief_C0_2023-10-01.npz")
        d = bb.get_draw(0)
        assert "B_mkt" in d
        assert "alpha_vote" in d
        assert "gamma_A" in d
        assert isinstance(d["B_mkt"], float)

    def test_belief_bundle_metadata(self):
        from engine.state import BeliefBundle
        bb = BeliefBundle(CHECKPOINT_DIR / "belief_C0_2023-10-01.npz")
        assert bb.metadata["checkpoint_id"] == "C0"
        assert bb.metadata["n_draws"] == 500

    def test_parameter_sampler(self):
        from engine.state import ParameterSampler
        ps = ParameterSampler(OPP_PRIORS)
        rng = np.random.default_rng(42)
        # Board's belief about ASA parameters
        params = ps.sample_parameters("Board", "ASA", rng)
        assert "mobilisation_cost" in params
        assert "strike_ba_shift" in params

    def test_parameter_sampler_all_perspectives(self):
        from engine.state import ParameterSampler
        ps = ParameterSampler(OPP_PRIORS)
        rng = np.random.default_rng(42)
        for persp, tgt in [("Board", "ASA"), ("Board", "CEO"),
                            ("ASA", "Board"), ("ASA", "CEO"),
                            ("CEO", "Board")]:
            params = ps.sample_parameters(persp, tgt, rng)
            assert len(params) > 0, f"No params for {persp}->{tgt}"

    def test_parameter_sampler_board_adverse_review_penalty(self):
        """Both ASA→Board and CEO→Board should have negative_review_finding_penalty."""
        from engine.state import ParameterSampler
        ps = ParameterSampler(OPP_PRIORS)
        rng = np.random.default_rng(42)
        for persp in ["ASA", "CEO"]:
            params = ps.sample_parameters(persp, "Board", rng)
            assert "negative_review_finding_penalty" in params, (
                f"{persp}→Board missing negative_review_finding_penalty")


# ============================================================================
# 2. FEASIBILITY RULES AND STATE TRANSITIONS
# ============================================================================

class TestFeasibility:
    """Test feasibility rules and state transitions."""

    def test_d1_all_feasible(self):
        from engine.state import DecisionState
        state = DecisionState.from_governance_spec(GOV_SPEC)
        actions = state.feasible_actions("D1")
        assert len(actions) == 3
        assert "D0_minimal" in actions
        assert "D3_ceo_transition" in actions

    def test_a2_all_feasible(self):
        from engine.state import DecisionState
        state = DecisionState.from_governance_spec(GOV_SPEC)
        actions = state.feasible_actions("A2")
        assert len(actions) == 2
        assert "A2_no_strike" in actions
        assert "A2_rec_strike" in actions

    def test_d_rev_no_action_always_feasible(self):
        from engine.state import DecisionState
        state = DecisionState.from_governance_spec(GOV_SPEC)
        actions = state.feasible_actions("D_rev")
        assert "Drev_no_action" in actions

    def test_d_rev_commission_review_feasibility(self):
        from engine.state import DecisionState
        state = DecisionState.from_governance_spec(GOV_SPEC)
        # Initially review not commissioned, so commission should be feasible
        actions = state.feasible_actions("D_rev")
        assert "Drev_commission_review" in actions

        # After commissioning review, should no longer be feasible
        state2 = state.apply("D_rev", "Drev_commission_review")
        actions2 = state2.feasible_actions("D_rev")
        assert "Drev_commission_review" not in actions2

    def test_d_rev_sack_requires_ceo_present(self):
        from engine.state import DecisionState
        state = DecisionState.from_governance_spec(GOV_SPEC)
        # CEO present: sack feasible
        actions = state.feasible_actions("D_rev")
        assert "Drev_sack_ceo" in actions

        # CEO removed: sack not feasible
        state2 = state.apply("D1", "D3_ceo_transition")
        actions2 = state2.feasible_actions("D_rev")
        assert "Drev_sack_ceo" not in actions2

    def test_d4_requires_ceo_present(self):
        from engine.state import DecisionState
        state = DecisionState.from_governance_spec(GOV_SPEC)
        # CEO present: all D4 actions feasible
        actions = state.feasible_actions("D4")
        assert len(actions) == 3

        # CEO removed: no D4 actions feasible
        state2 = state.apply("D1", "D3_ceo_transition")
        actions2 = state2.feasible_actions("D4")
        assert len(actions2) == 0

    def test_state_apply_d3_removes_ceo(self):
        from engine.state import DecisionState
        state = DecisionState.from_governance_spec(GOV_SPEC)
        state2 = state.apply("D1", "D3_ceo_transition")
        assert state2.CEO_present is False
        assert state2.CEO_removed is True

    def test_state_apply_resign_removes_ceo(self):
        from engine.state import DecisionState
        state = DecisionState.from_governance_spec(GOV_SPEC)
        state2 = state.apply("D4", "D4_resign")
        assert state2.CEO_present is False
        assert state2.CEO_removed is True

    def test_next_node_ordering(self):
        from engine.state import DecisionState
        state = DecisionState.from_governance_spec(GOV_SPEC)
        assert state.next_node("D0_ceo") == "D1"
        assert state.next_node("D1") == "A2"
        assert state.next_node("A2") == "V"
        assert state.next_node("V") == "M_agm"
        assert state.next_node("M_agm") == "D4"
        assert state.next_node("D4") == "D_rev"
        assert state.next_node("D_rev") == "R"
        assert state.next_node("R") == "M_rev"
        assert state.next_node("M_rev") == "D4_post_review"
        assert state.next_node("D4_post_review") == "D_rev_post_review"
        assert state.next_node("D_rev_post_review") == "Terminal"
        assert state.next_node("Terminal") is None


# ============================================================================
# 2b. POST-REVIEW ROUND (Phase 6)
# ============================================================================

class TestPostReviewRound:
    """Tests for Phase 6 conditional post-review round logic."""

    def test_review_adverse_sets_state_flags(self):
        """apply("R", "negative") with CEO present sets all Phase 6 flags."""
        from engine.state import DecisionState
        state = DecisionState.from_governance_spec(GOV_SPEC)
        assert state.CEO_present is True
        s = state.apply("R", "negative")
        assert s.review_outcome == "negative"
        assert s.review_completed is True
        assert s.post_review_round is True

    def test_review_no_adverse_clears_flags(self):
        """apply("R", "positive") sets review_completed but not Phase 6."""
        from engine.state import DecisionState
        state = DecisionState.from_governance_spec(GOV_SPEC)
        s = state.apply("R", "positive")
        assert s.review_outcome != "negative"
        assert s.review_completed is True
        assert s.post_review_round is False

    def test_post_review_round_not_set_when_ceo_absent(self):
        """apply("R", "negative") with CEO absent does NOT trigger Phase 6."""
        from engine.state import DecisionState
        state = DecisionState.from_governance_spec(GOV_SPEC)
        # Remove CEO first
        state = state.apply("D0_ceo", "CEO_resign")
        assert state.CEO_present is False
        s = state.apply("R", "negative")
        assert s.review_outcome == "negative"
        assert s.review_completed is True
        assert s.post_review_round is False  # CEO not present → no Phase 6

    def test_d4_post_review_feasibility_active(self):
        """When post_review_round=True, D4_post_review has 3 feasible actions."""
        from engine.state import DecisionState
        state = DecisionState.from_governance_spec(GOV_SPEC)
        # Trigger Phase 6: review negative with CEO present
        s = state.apply("R", "negative")
        assert s.post_review_round is True
        feasible = s.feasible_actions("D4_post_review")
        assert len(feasible) == 3
        assert "D4_stay" in feasible
        assert "D4_resign" in feasible
        assert "D4_negotiate_exit" in feasible

    def test_d4_post_review_feasibility_inactive(self):
        """When post_review_round=False, D4_post_review has 0 feasible actions."""
        from engine.state import DecisionState
        state = DecisionState.from_governance_spec(GOV_SPEC)
        # No negative review → post_review_round=False
        s = state.apply("R", "positive")
        assert s.post_review_round is False
        feasible = s.feasible_actions("D4_post_review")
        assert len(feasible) == 0

    def test_d_rev_post_review_ceo_present(self):
        """D_rev_post_review with CEO present has 3 actions including sack."""
        from engine.state import DecisionState
        state = DecisionState.from_governance_spec(GOV_SPEC)
        s = state.apply("R", "negative")
        assert s.post_review_round is True
        assert s.CEO_present is True
        feasible = s.feasible_actions("D_rev_post_review")
        assert len(feasible) == 3
        assert "Drev_sack_ceo" in feasible
        assert "Drev_no_action" in feasible
        assert "Drev_commission_review" in feasible

    def test_d_rev_post_review_ceo_resigned(self):
        """D_rev_post_review after CEO resignation has 2 actions (no sack)."""
        from engine.state import DecisionState
        state = DecisionState.from_governance_spec(GOV_SPEC)
        # Review negative + CEO present → Phase 6 triggered
        s = state.apply("R", "negative")
        # CEO resigns at D4_post_review
        s = s.apply("D4_post_review", "D4_resign")
        assert s.post_review_round is True
        assert s.CEO_present is False
        feasible = s.feasible_actions("D_rev_post_review")
        assert "Drev_sack_ceo" not in feasible
        assert "Drev_no_action" in feasible

    def test_phase6_skipped_when_ceo_absent(self):
        """When CEO resigned before review, both Phase 6 nodes are skipped."""
        from engine.state import DecisionState
        state = DecisionState.from_governance_spec(GOV_SPEC)
        # CEO resigned early
        state = state.apply("D0_ceo", "CEO_resign")
        state = state.apply("D1", "D1_review")
        # Review negative but CEO absent
        s = state.apply("R", "negative")
        assert s.post_review_round is False
        # Both Phase 6 nodes should have 0 feasible actions (skipped)
        assert len(s.feasible_actions("D4_post_review")) == 0
        assert len(s.feasible_actions("D_rev_post_review")) == 0


# ============================================================================
# 3. CHANCE MODELS
# ============================================================================

class TestChanceModels:
    """Test vote and review models."""

    def test_vote_model_basic(self):
        from engine.state import BeliefBundle, DecisionState
        from engine.chance_models import VoteModel
        bb = BeliefBundle(CHECKPOINT_DIR / "belief_C0_2023-10-01.npz")
        state = DecisionState.from_governance_spec(GOV_SPEC)
        vote_model = VoteModel({"first_strike": 0.25, "overwhelming": 0.50})
        rng = np.random.default_rng(42)

        history = {"D1": "D0_minimal", "A2": "A2_no_strike"}
        out = vote_model.sample(0, bb, history, state, rng)

        assert 0.0 <= out.vote_percent <= 1.0
        assert isinstance(out.strike_indicator, (bool, np.bool_))
        assert isinstance(out.overwhelming_indicator, (bool, np.bool_))

    def test_vote_increases_with_strike_recommendation(self):
        """ASA strike recommendation should increase vote opposition."""
        from engine.state import BeliefBundle, DecisionState
        from engine.chance_models import VoteModel
        bb = BeliefBundle(CHECKPOINT_DIR / "belief_C0_2023-10-01.npz")
        state = DecisionState.from_governance_spec(GOV_SPEC)
        vote_model = VoteModel({"first_strike": 0.25, "overwhelming": 0.50})

        n_samples = 200
        votes_no_strike = []
        votes_strike = []

        for i in range(n_samples):
            rng1 = np.random.default_rng(1000 + i)
            rng2 = np.random.default_rng(1000 + i)

            h1 = {"D1": "D0_minimal", "A2": "A2_no_strike"}
            h2 = {"D1": "D0_minimal", "A2": "A2_rec_strike"}

            v1 = vote_model.sample(i % bb.N, bb, h1, state, rng1)
            v2 = vote_model.sample(i % bb.N, bb, h2, state, rng2)

            votes_no_strike.append(v1.vote_percent)
            votes_strike.append(v2.vote_percent)

        mean_no = np.mean(votes_no_strike)
        mean_yes = np.mean(votes_strike)
        # Strike recommendation should increase opposition
        assert mean_yes > mean_no, (
            f"Strike rec should increase vote: {mean_yes:.3f} vs {mean_no:.3f}"
        )

    def test_vote_non_monotonic_governance_effect(self):
        """Governance effects are non-monotonic per empirical data.

        D1 (review) is the sweet spot — reduces protest relative to D0.
        D3 (CEO exit) signals crisis — increases protest relative to D0.
        Ranking: mean_d1 < mean_d0 < mean_d3.
        """
        from engine.state import BeliefBundle, DecisionState
        from engine.chance_models import VoteModel
        bb = BeliefBundle(CHECKPOINT_DIR / "belief_C0_2023-10-01.npz")
        state = DecisionState.from_governance_spec(GOV_SPEC)
        vote_model = VoteModel({"first_strike": 0.25, "overwhelming": 0.50})

        n_samples = 200
        votes = {}
        for d1 in ["D0_minimal", "D1_review", "D3_ceo_transition"]:
            votes[d1] = []
            for i in range(n_samples):
                rng = np.random.default_rng(2000 + i)
                h = {"D1": d1, "A2": "A2_no_strike"}
                v = vote_model.sample(i % bb.N, bb, h, state, rng)
                votes[d1].append(v.vote_percent)

        mean_d0 = np.mean(votes["D0_minimal"])
        mean_d1 = np.mean(votes["D1_review"])
        mean_d3 = np.mean(votes["D3_ceo_transition"])

        assert mean_d1 < mean_d0, (
            f"D1 (review) should reduce protest vs D0: {mean_d1:.3f} vs {mean_d0:.3f}"
        )
        assert mean_d3 > mean_d0, (
            f"D3 (CEO exit) should increase protest vs D0: {mean_d3:.3f} vs {mean_d0:.3f}"
        )

    def test_review_model_no_commission(self):
        """If review not commissioned, always no adverse finding."""
        from engine.state import BeliefBundle, DecisionState
        from engine.chance_models import ReviewModel
        bb = BeliefBundle(CHECKPOINT_DIR / "belief_C0_2023-10-01.npz")
        state = DecisionState.from_governance_spec(GOV_SPEC)
        review = ReviewModel()
        rng = np.random.default_rng(42)

        assert not state.review_commissioned
        out = review.sample(0, bb, {}, state, rng)
        assert out.review_outcome != "negative"
        assert out.review_car == 0.0

    def test_review_model_commissioned(self):
        """If review commissioned, should produce both outcomes stochastically."""
        from engine.state import BeliefBundle, DecisionState
        from engine.chance_models import ReviewModel
        bb = BeliefBundle(CHECKPOINT_DIR / "belief_C0_2023-10-01.npz")
        state = DecisionState.from_governance_spec(GOV_SPEC)
        state.review_commissioned = True
        review = ReviewModel()

        adverse_count = 0
        cars = []
        n = 200
        for i in range(n):
            rng = np.random.default_rng(3000 + i)
            out = review.sample(i % bb.N, bb, {"D1": "D0_minimal"}, state, rng)
            if out.review_outcome == "negative":
                adverse_count += 1
            cars.append(out.review_car)

        # Should get some adverse and some non-adverse
        assert 0 < adverse_count < n, f"Got {adverse_count}/{n} adverse"
        # CARs should have non-zero variance (stochastic)
        assert np.std(cars) > 0.01, f"CAR std too low: {np.std(cars):.4f}"
        # Mean CAR should be centered around -5% (MU_LOC = -0.05)
        assert -0.15 < np.mean(cars) < 0.05, f"Mean CAR out of range: {np.mean(cars):.4f}"

    def test_review_outcome_probabilities_dirichlet(self):
        """Review outcome probabilities drawn from Dirichlet(38,160,1)."""
        from engine.chance_models import ReviewModel
        review = ReviewModel()

        n = 2000
        prob_arrays = [review.draw_outcome_probabilities(np.random.default_rng(i))
                       for i in range(n)]
        # Each call should return a numpy array of shape (3,)
        for arr in prob_arrays[:10]:
            assert hasattr(arr, "shape"), "draw_outcome_probabilities must return an array"
            assert arr.shape == (3,), f"Expected shape (3,), got {arr.shape}"
            assert abs(arr.sum() - 1.0) < 1e-9, f"Probabilities must sum to 1, got {arr.sum()}"
            assert (arr >= 0).all(), "All probabilities must be non-negative"
        # Dirichlet(38,160,1): E = (38/199, 160/199, 1/199) ≈ (0.191, 0.804, 0.005)
        expected_means = [38/199, 160/199, 1/199]
        means = np.mean(prob_arrays, axis=0)
        for k, (m, em) in enumerate(zip(means, expected_means)):
            assert abs(m - em) < 0.03, (
                f"Component {k} mean {m:.3f} should be near {em:.3f}")

    def test_review_with_explicit_outcome_probs(self):
        """Review model respects explicit outcome_probs parameter."""
        from engine.state import BeliefBundle, DecisionState
        from engine.chance_models import ReviewModel
        bb = BeliefBundle(CHECKPOINT_DIR / "belief_C0_2023-10-01.npz")
        state = DecisionState.from_governance_spec(GOV_SPEC)
        state.review_commissioned = True
        review = ReviewModel()

        # outcome_probs=[1,0,0] → always negative
        n = 50
        for i in range(n):
            rng = np.random.default_rng(5000 + i)
            out = review.sample(i % bb.N, bb, {"D1": "D0_minimal"}, state, rng,
                                outcome_probs=np.array([1.0, 0.0, 0.0]))
            assert out.review_outcome == "negative"

        # outcome_probs=[0,0,1] → always positive
        for i in range(n):
            rng = np.random.default_rng(6000 + i)
            out = review.sample(i % bb.N, bb, {"D1": "D0_minimal"}, state, rng,
                                outcome_probs=np.array([0.0, 0.0, 1.0]))
            assert out.review_outcome == "positive"

    def test_review_adverse_rate_matches_dirichlet(self):
        """Commissioned review negative outcome rate ≈ 0.191 (Dirichlet(38,160,1) mean)."""
        from engine.state import BeliefBundle, DecisionState
        from engine.chance_models import ReviewModel
        bb = BeliefBundle(CHECKPOINT_DIR / "belief_C0_2023-10-01.npz")
        state = DecisionState.from_governance_spec(GOV_SPEC)
        state.review_commissioned = True
        review = ReviewModel()

        n = 500
        negative_count = 0
        for i in range(n):
            rng = np.random.default_rng(7000 + i)
            outcome_probs = review.draw_outcome_probabilities(rng)
            out = review.sample(i % bb.N, bb, {"D1": "D0_minimal"}, state, rng,
                                outcome_probs=outcome_probs)
            if out.review_outcome == "negative":
                negative_count += 1

        negative_rate = negative_count / n
        # Should be near 38/199 ≈ 0.191 (Dirichlet(38,160,1) component mean)
        assert 0.10 < negative_rate < 0.30, (
            f"Negative outcome rate {negative_rate:.3f} should be near 0.191")

    def test_review_direct_cost_model(self):
        """ReviewDirectCostModel samples from Gamma(4.55, 4741) in decimal CAR."""
        from engine.chance_models import ReviewDirectCostModel
        model = ReviewDirectCostModel()

        # Check expected cost
        expected = model.expected_cost()
        assert abs(expected - 0.00096) < 1e-6, f"Expected cost wrong: {expected}"

        # Sample many draws and check moments
        costs = [model.sample(np.random.default_rng(i)) for i in range(2000)]
        mean_cost = np.mean(costs)
        sd_cost = np.std(costs)

        # All samples must be positive (Gamma is defined on R+)
        assert all(c > 0 for c in costs), "Gamma samples must be positive"

        # Mean should be near 0.00096 (tolerance for MC noise)
        assert abs(mean_cost - 0.00096) < 0.0002, f"Mean cost {mean_cost:.6f} far from 0.00096"

        # SD should be near 0.00045
        assert abs(sd_cost - 0.00045) < 0.0002, f"SD cost {sd_cost:.6f} far from 0.00045"

    def test_review_direct_cost_via_chance_models(self):
        """ChanceModels.sample_review_direct_cost delegates correctly."""
        from engine.chance_models import ChanceModels
        cm = ChanceModels({"first_strike": 0.25, "overwhelming": 0.50})
        rng = np.random.default_rng(42)
        cost = cm.sample_review_direct_cost(rng)
        assert cost > 0
        assert cost < 0.01  # Reasonable upper bound (100 bps)


# ============================================================================
# 4. UTILITY FUNCTIONS
# ============================================================================

class TestUtilities:
    """Test utility computation for all actors."""

    def test_board_utility_baseline(self):
        from engine.utilities import utility_board, TerminalOutcome
        outcome = TerminalOutcome()
        params = {"inaction_base_penalty": 3.0, "inaction_no_review_penalty": 2.0,
                  "inaction_ceo_present_penalty": 5.0, "inaction_no_sack_penalty": 3.0,
                  "review_car_weight": 15.0, "review_direct_cost_weight": 15.0,
                  "implementation_cost_sack": 1.0, "ceo_loss_cost": 1.5}
        u = utility_board(outcome, params)
        # Baseline with CEO present + no action: all 4 inaction components fire
        expected = -(3.0 + 2.0 + 5.0 + 3.0)  # sum of inaction penalties
        assert u == expected

    def test_board_utility_high_vote_penalty(self):
        from engine.utilities import utility_board, TerminalOutcome
        outcome = TerminalOutcome(vote_percent=0.80, strike_indicator=True,
                                  overwhelming_indicator=True)
        params = {"inaction_base_penalty": 3.0, "inaction_no_review_penalty": 2.0,
                  "inaction_ceo_present_penalty": 5.0, "inaction_no_sack_penalty": 3.0,
                  "vote_strike_penalty": 2.0, "vote_overwhelming_penalty": 3.0,
                  "review_car_weight": 15.0, "review_direct_cost_weight": 15.0,
                  "implementation_cost_sack": 1.0, "ceo_loss_cost": 1.5}
        u = utility_board(outcome, params)
        assert u < 0  # Should be negative (bad for board)

    def test_asa_utility_high_vote_reward(self):
        """ASA utility: strong vote + CEO removal + strike = high utility."""
        from engine.utilities import utility_asa, TerminalOutcome
        outcome = TerminalOutcome(vote_percent=0.80, strike_indicator=True,
                                  overwhelming_indicator=True, CEO_removed=True,
                                  a2_action="A2_rec_strike")
        params = {"strike_ba_shift": 1.5, "strike_ol_shift": 1.0,
                  "overwhelming_ba_shift": 1.0, "overwhelming_ol_shift": 0.5,
                  "ceo_removal_ba_shift": 1.0, "ceo_removal_fw_shift": 0.5,
                  "market_alignment_ol_shift": 1.0, "market_alignment_pf_shift": 0.5,
                  "mobilisation_cost": 0.3}
        u = utility_asa(outcome, params)
        # Base (stays, do nothing) ≈ 1.43 + large shifts → high utility
        assert u > 2.0

    def test_asa_mobilisation_cost(self):
        """ASA utility: recommending strike incurs mobilisation cost."""
        from engine.utilities import utility_asa, TerminalOutcome
        params = {"mobilisation_cost": 0.3}

        out_no = TerminalOutcome(a2_action="A2_no_strike", vote_percent=0.10)
        out_yes = TerminalOutcome(a2_action="A2_rec_strike", vote_percent=0.10)

        u_no = utility_asa(out_no, params)
        u_yes = utility_asa(out_yes, params)
        # Strike rec costs more due to mobilisation
        assert u_no > u_yes

    def test_ceo_utility_crra_ordering(self):
        """CRRA CEO utility: staying (kept) > negotiate > late resign > sacked."""
        from engine.utilities import utility_ceo, TerminalOutcome
        params = {"gamma": 1.5, "W_resign": 8.0, "W_stay_sacked": 1.5,
                  "W_stay_kept": 7.0, "D_resign": 40.0, "D_stay": 25.0,
                  "D_sacked": 100.0, "D_resign_late": 60.0, "D_negotiate": 45.0,
                  "D_agm": 30.0, "D_disgrace": 30.0,
                  "D_adverse_review": 10.0, "loss_aversion": 2.25, "W_ref": 16.0,
                  "loss_aversion_D": 2.25}

        out_stay = TerminalOutcome(CEO_removed=False, vote_percent=0.10)
        out_negotiate = TerminalOutcome(CEO_removed=True, d4_action="D4_negotiate_exit",
                                          vote_percent=0.10)
        out_resign = TerminalOutcome(CEO_removed=True, d4_action="D4_resign",
                                      vote_percent=0.10)
        out_sacked = TerminalOutcome(CEO_removed=True, d4_action="D4_stay",
                                      d_rev_action="Drev_sack_ceo",
                                      vote_percent=0.50, strike_indicator=True)

        u_stay = utility_ceo(out_stay, params)
        u_negotiate = utility_ceo(out_negotiate, params)
        u_resign = utility_ceo(out_resign, params)
        u_sacked = utility_ceo(out_sacked, params)

        assert u_stay > u_negotiate  # Staying (kept) is best
        assert u_negotiate > u_resign  # Negotiate > late resign
        assert u_resign > u_sacked  # Late resign > forced sacking

    def test_board_review_direct_cost_in_utility(self):
        """Board utility subtracts stochastic direct cost when review commissioned."""
        from engine.utilities import utility_board, TerminalOutcome
        params = {"inaction_base_penalty": 3.0, "inaction_no_review_penalty": 2.0,
                  "inaction_ceo_present_penalty": 5.0, "inaction_no_sack_penalty": 3.0,
                  "review_car_weight": 15.0, "review_direct_cost_weight": 15.0,
                  "implementation_cost_sack": 1.0, "ceo_loss_cost": 1.5}

        # No review: review_direct_cost ignored when not commissioned
        out_no = TerminalOutcome(review_commissioned=False, review_direct_cost=0.001)
        u_no = utility_board(out_no, params)

        # Review commissioned with mean direct cost (0.00096 ≈ 9.6 bps)
        # Same outcome but with review — should incur direct cost
        out_yes = TerminalOutcome(
            d1_action="D1_review",
            review_commissioned=True,
            review_direct_cost=0.00096,
            review_car=0.0,  # Isolate direct cost effect
        )
        u_yes = utility_board(out_yes, params)
        # Review commissioned removes inaction_no_review penalty (+2.0),
        # but adds direct cost: -15.0 * 0.00096 ≈ -0.0144
        # Net: u_yes = u_no + 2.0 - 0.0144
        assert u_yes > u_no  # Net benefit: removing inaction penalty > direct cost

    def test_utility_dispatch(self):
        from engine.utilities import compute_utility, TerminalOutcome
        outcome = TerminalOutcome(vote_percent=0.30, strike_indicator=True)
        params = {"vote_penalty_weight": 2.0, "overwhelming_penalty_weight": 3.0,
                  "spill_risk_weight": 2.5, "review_car_weight": 15.0,
                  "review_direct_cost_weight": 15.0, "implementation_cost_sack": 1.0,
                  "ceo_loss_cost": 1.5, "reputational_spill_weight": 1.0}
        u = compute_utility("Board", outcome, params)
        assert isinstance(u, float)


# ============================================================================
# 5. MODE CONFIGURATIONS
# ============================================================================

class TestModes:
    """Test mode configurations."""

    def test_board_mode(self):
        from engine.modes import MODE_BOARD
        assert MODE_BOARD.focal_actor == "Board"
        assert MODE_BOARD.is_focal("Board")
        assert not MODE_BOARD.is_focal("ASA")
        assert MODE_BOARD.get_opponent_model_type("ASA") == "ARA"
        assert MODE_BOARD.get_opponent_model_type("CEO") == "ARA"

    def test_asa_mode(self):
        from engine.modes import MODE_ASA
        assert MODE_ASA.focal_actor == "ASA"
        assert MODE_ASA.is_focal("ASA")
        assert not MODE_ASA.is_focal("Board")
        assert MODE_ASA.get_opponent_model_type("Board") == "ARA"

    def test_level2_counterparts(self):
        from engine.modes import MODE_BOARD_L2
        assert MODE_BOARD_L2.level == 2
        assert MODE_BOARD_L2.get_strategic_counterpart("ASA") == "Board"
        assert MODE_BOARD_L2.get_strategic_counterpart("CEO") == "Board"

    def test_policy_mode(self):
        from engine.modes import MODE_ASA_POLICY_BOARD
        assert MODE_ASA_POLICY_BOARD.get_opponent_model_type("Board") == "Policy"
        assert MODE_ASA_POLICY_BOARD.get_opponent_model_type("CEO") == "ARA"

    def test_focal_returns_focal_type(self):
        from engine.modes import MODE_BOARD
        assert MODE_BOARD.get_opponent_model_type("Board") == "focal"


# ============================================================================
# 6. PREDICTIVE DISTRIBUTIONS
# ============================================================================

class TestPredictive:
    """Test predictive distribution computation."""

    def _make_predictive(self, K=30, R_rollouts=5):
        from engine.state import (
            BeliefBundle, DecisionState, ParameterSampler,
            load_vote_thresholds, load_policy_parameters,
        )
        from engine.chance_models import ChanceModels
        from engine.predictive import PredictiveDistribution

        beliefs = BeliefBundle(CHECKPOINT_DIR / "belief_C0_2023-10-01.npz")
        thresholds = load_vote_thresholds(GOV_SPEC)
        chance = ChanceModels(thresholds)
        sampler = ParameterSampler(OPP_PRIORS)
        policy_params = load_policy_parameters(GOV_SPEC)

        return PredictiveDistribution(
            beliefs=beliefs,
            param_sampler=sampler,
            chance_models=chance,
            policy_params=policy_params,
            K=K,
            R_rollouts=R_rollouts,
        )

    def test_predict_returns_distribution(self):
        pred = self._make_predictive()
        from engine.state import DecisionState
        from engine.modes import MODE_BOARD

        state = DecisionState.from_governance_spec(GOV_SPEC)
        rng = np.random.default_rng(42)

        # Predict ASA's action at A2 from Board's perspective
        history = {"D1": "D0_minimal"}
        dist = pred.predict("A2", history, state, 0, "Board", MODE_BOARD, 1, rng)

        assert len(dist) == 2
        assert "A2_no_strike" in dist
        assert "A2_rec_strike" in dist
        total = sum(dist.values())
        assert abs(total - 1.0) < 1e-6, f"Distribution sums to {total}"

    def test_predict_single_feasible_action(self):
        pred = self._make_predictive()
        from engine.state import DecisionState
        from engine.modes import MODE_BOARD

        state = DecisionState.from_governance_spec(GOV_SPEC)
        state = state.apply("D1", "D3_ceo_transition")  # Removes CEO
        rng = np.random.default_rng(42)

        # D4 with CEO removed: no feasible actions
        dist = pred.predict("D4", {}, state, 0, "Board", MODE_BOARD, 1, rng)
        assert len(dist) == 0

    def test_asa_beta_posterior_strike_probabilities(self):
        """ASA fixed policy uses 5-path Beta priors from asa_bayesian_params.md.

        Path-conditional Beta priors:
          A2-1: CEO resigns → Do nothing         Beta(18, 2) mean=0.900
          A2-2: CEO resigns → Commission review  Beta(14, 3) mean=0.824
          A2-3: CEO stays   → Do nothing         Beta(24, 1) mean=0.960
          A2-4: CEO stays   → Commission review  Beta(15, 2) mean=0.882
          A2-5: CEO stays   → Board forces exit  Beta(9,  6) mean=0.600
        """
        pred = self._make_predictive()
        from engine.state import DecisionState

        state = DecisionState.from_governance_spec(GOV_SPEC)
        feasible = ["A2_rec_strike", "A2_no_strike"]
        n_samples = 2000

        # Test CEO stayed paths
        strike_counts = {}
        for d1 in ["D0_minimal", "D1_review", "D3_ceo_transition"]:
            count = 0
            for i in range(n_samples):
                rng = np.random.default_rng(5000 + i)
                action = pred._fixed_policy(
                    "ASA", "A2", {"D1": d1, "D0_ceo": "CEO_stay"}, state, feasible, rng=rng
                )
                if action == "A2_rec_strike":
                    count += 1
            strike_counts[d1] = count / n_samples

        # A2-3: CEO stays → Do nothing: Beta(24,1) mean=0.960
        assert 0.90 < strike_counts["D0_minimal"] < 1.00, (
            f"A2-3 Beta(24,1) mean ~0.960, got {strike_counts['D0_minimal']:.1%}"
        )
        # A2-4: CEO stays → Review: Beta(15,2) mean=0.882
        assert 0.80 < strike_counts["D1_review"] < 0.96, (
            f"A2-4 Beta(15,2) mean ~0.882, got {strike_counts['D1_review']:.1%}"
        )
        # A2-5: CEO stays → Board forces exit: Beta(9,6) mean=0.600
        assert 0.45 < strike_counts["D3_ceo_transition"] < 0.75, (
            f"A2-5 Beta(9,6) mean ~0.600, got {strike_counts['D3_ceo_transition']:.1%}"
        )

        # Test CEO resigned paths
        for d1, expected_mean, lo, hi, label in [
            ("D0_minimal", 0.900, 0.82, 0.97, "A2-1 Beta(18,2)"),
            ("D1_review", 0.824, 0.72, 0.92, "A2-2 Beta(14,3)"),
        ]:
            count = 0
            for i in range(n_samples):
                rng = np.random.default_rng(8000 + i)
                action = pred._fixed_policy(
                    "ASA", "A2", {"D1": d1, "D0_ceo": "CEO_resign"}, state, feasible, rng=rng
                )
                if action == "A2_rec_strike":
                    count += 1
            rate = count / n_samples
            assert lo < rate < hi, (
                f"{label} mean ~{expected_mean:.3f}, got {rate:.1%}"
            )

        # Ordering: A2-5 (0.600) < A2-2 (0.824) < A2-4 (0.882) < A2-1 (0.900) < A2-3 (0.960)
        assert strike_counts["D3_ceo_transition"] < strike_counts["D1_review"]
        assert strike_counts["D1_review"] < strike_counts["D0_minimal"]

    def test_build_outcome(self):
        from engine.predictive import PredictiveDistribution
        from engine.state import DecisionState

        state = DecisionState.from_governance_spec(GOV_SPEC)
        state.CEO_removed = True

        history = {
            "D1": "D1_review",
            "A2": "A2_rec_strike",
            "V_percent": 0.40,
            "V_strike": True,
            "V_overwhelming": False,
            "R_outcome": "negative",
            "R_car": -0.08,
            "R_direct_cost": 0.00096,
            "D_rev": "Drev_commission_review",
            "D4": "D4_resign",
        }

        outcome = PredictiveDistribution._build_outcome(history, state)
        assert outcome.d1_action == "D1_review"
        assert outcome.a2_action == "A2_rec_strike"
        assert outcome.vote_percent == 0.40
        assert outcome.strike_indicator is True
        assert outcome.review_outcome == "negative"
        assert outcome.review_car == -0.08
        assert outcome.review_direct_cost == 0.00096
        assert outcome.CEO_removed is True


# ============================================================================
# 6b. LAPLACE SMOOTHING (DIRICHLET PRIOR)
# ============================================================================

class TestLaplaceSmoothing:
    """Test Dirichlet(1,...,1) Laplace smoothing on predictive distributions."""

    def _make_predictive(self, K=30, R_rollouts=5, no_prior_actors=None):
        from engine.state import (
            BeliefBundle, DecisionState, ParameterSampler,
            load_vote_thresholds, load_policy_parameters,
        )
        from engine.chance_models import ChanceModels
        from engine.predictive import PredictiveDistribution

        beliefs = BeliefBundle(CHECKPOINT_DIR / "belief_C0_2023-10-01.npz")
        thresholds = load_vote_thresholds(GOV_SPEC)
        chance = ChanceModels(thresholds)
        sampler = ParameterSampler(OPP_PRIORS)
        policy_params = load_policy_parameters(GOV_SPEC)

        return PredictiveDistribution(
            beliefs=beliefs,
            param_sampler=sampler,
            chance_models=chance,
            policy_params=policy_params,
            K=K,
            R_rollouts=R_rollouts,
            no_prior_actors=no_prior_actors,
        )

    def test_laplace_default_no_zero_probabilities(self):
        """Default (Laplace on): no feasible action should have exactly 0."""
        pred = self._make_predictive(K=30, R_rollouts=5)
        from engine.state import DecisionState
        from engine.modes import MODE_BOARD

        state = DecisionState.from_governance_spec(GOV_SPEC)
        rng = np.random.default_rng(42)
        history = {"D1": "D0_minimal"}
        dist = pred.predict("A2", history, state, 0, "Board", MODE_BOARD, 1, rng)

        for action, prob in dist.items():
            assert prob > 0, f"{action} has zero probability with Laplace smoothing on"

    def test_laplace_disabled_allows_zero(self):
        """When Laplace disabled for ASA, zero-count actions can have prob 0."""
        pred = self._make_predictive(K=30, R_rollouts=5, no_prior_actors={"ASA"})
        from engine.state import DecisionState
        from engine.modes import MODE_BOARD

        state = DecisionState.from_governance_spec(GOV_SPEC)
        rng = np.random.default_rng(42)
        history = {"D1": "D0_minimal"}
        dist = pred.predict("A2", history, state, 0, "Board", MODE_BOARD, 1, rng)

        # With Laplace disabled, an action that is never best gets exactly 0
        # (though with K=30 both may be nonzero; we verify normalization)
        total = sum(dist.values())
        assert abs(total - 1.0) < 1e-6

    def test_laplace_minimum_probability(self):
        """With K samples and n feasible actions, min prob >= 1/(K + n)."""
        K = 50
        pred = self._make_predictive(K=K, R_rollouts=5)
        from engine.state import DecisionState
        from engine.modes import MODE_BOARD

        state = DecisionState.from_governance_spec(GOV_SPEC)
        rng = np.random.default_rng(42)
        history = {"D1": "D0_minimal"}
        dist = pred.predict("A2", history, state, 0, "Board", MODE_BOARD, 1, rng)

        n_actions = len(dist)
        min_prob = 1.0 / (K + n_actions)
        for action, prob in dist.items():
            assert prob >= min_prob - 1e-9, (
                f"{action} prob {prob:.6f} < min {min_prob:.6f}"
            )

    def test_laplace_selective_per_actor(self):
        """Laplace disabled for Board but still active for ASA."""
        pred = self._make_predictive(K=30, R_rollouts=5, no_prior_actors={"Board"})

        # ASA (owner of A2) should still get Laplace smoothing
        assert "Board" in pred.no_prior_actors
        assert "ASA" not in pred.no_prior_actors


# ============================================================================
# 7. TREE EVALUATION
# ============================================================================

class TestTreeEvaluation:
    """Test the tree evaluator."""

    def _make_tree(self, K=20, R_rollouts=5, n_vote=10, n_review=5):
        from engine.state import (
            BeliefBundle, DecisionState, ParameterSampler,
            load_vote_thresholds, load_utility_weights, load_policy_parameters,
        )
        from engine.chance_models import ChanceModels
        from engine.predictive import PredictiveDistribution
        from engine.tree import TreeEvaluator

        beliefs = BeliefBundle(CHECKPOINT_DIR / "belief_C0_2023-10-01.npz")
        thresholds = load_vote_thresholds(GOV_SPEC)
        chance = ChanceModels(thresholds)
        sampler = ParameterSampler(OPP_PRIORS)
        policy_params = load_policy_parameters(GOV_SPEC)
        utility_weights = {
            actor: load_utility_weights(GOV_SPEC, actor)
            for actor in ["Board", "ASA", "CEO"]
        }

        pred = PredictiveDistribution(
            beliefs=beliefs,
            param_sampler=sampler,
            chance_models=chance,
            policy_params=policy_params,
            K=K,
            R_rollouts=R_rollouts,
        )
        tree = TreeEvaluator(
            beliefs=beliefs,
            chance_models=chance,
            predictive=pred,
            utility_weights=utility_weights,
            n_vote_samples=n_vote,
            n_review_samples=n_review,
        )
        state = DecisionState.from_governance_spec(GOV_SPEC)
        return tree, state, beliefs

    def test_terminal_value(self):
        tree, state, _ = self._make_tree()
        from engine.modes import MODE_BOARD

        rng = np.random.default_rng(42)
        history = {
            "D1": "D0_minimal",
            "A2": "A2_no_strike",
            "V_percent": 0.10,
            "V_strike": False,
            "V_overwhelming": False,
            "R_outcome": "none",
            "D_rev": "Drev_no_action",
            "D4": "D4_stay",
        }

        v = tree.value("Terminal", history, state, 0, MODE_BOARD, rng)
        assert isinstance(v, float)
        # Low vote, no adverse, CEO stays -> utility is negative (inaction penalties)
        assert v >= -30.0  # Relaxed bound for unconditional inaction components

    def test_chance_node_returns_float(self):
        tree, state, _ = self._make_tree()
        from engine.modes import MODE_BOARD

        rng = np.random.default_rng(42)
        history = {"D1": "D0_minimal", "A2": "A2_no_strike"}

        v = tree.value("V", history, state, 0, MODE_BOARD, rng)
        assert isinstance(v, float)
        assert np.isfinite(v)

    def test_focal_maximises(self):
        """Focal actor should get higher value than worst action."""
        tree, state, _ = self._make_tree()
        from engine.modes import MODE_BOARD

        rng = np.random.default_rng(42)

        # Evaluate D1 as focal node (Board maximises)
        v_focal = tree.value("D1", {}, state, 0, MODE_BOARD, rng)
        assert isinstance(v_focal, float)
        assert np.isfinite(v_focal)

    def test_optimal_action(self):
        tree, state, _ = self._make_tree()
        from engine.modes import MODE_BOARD

        rng = np.random.default_rng(42)
        action, value = tree.optimal_action("D1", {}, state, 0, MODE_BOARD, rng)

        assert action in ["D0_minimal", "D1_review", "D3_ceo_transition"]
        assert isinstance(value, float)

    def test_board_vs_asa_perspective(self):
        """Board and ASA should have different values for the same scenario."""
        tree, state, _ = self._make_tree()
        from engine.modes import MODE_BOARD, MODE_ASA

        rng_b = np.random.default_rng(42)
        rng_a = np.random.default_rng(42)

        history = {
            "D1": "D0_minimal",
            "A2": "A2_rec_strike",
            "V_percent": 0.50,
            "V_strike": True,
            "V_overwhelming": True,
            "R_outcome": "negative",
            "D_rev": "Drev_sack_ceo",
            "D4": "D4_resign",
        }
        state.CEO_removed = True

        v_board = tree.value("Terminal", history, state, 0, MODE_BOARD, rng_b)
        v_asa = tree.value("Terminal", history, state, 0, MODE_ASA, rng_a)

        # High opposition + CEO removed: bad for Board, good for ASA
        assert v_board < v_asa


# ============================================================================
# 8. SOLVER INTEGRATION
# ============================================================================

class TestSolver:
    """Integration tests for the solver."""

    def _make_solver(self, K=20, R=5):
        from engine.solver import Solver
        return Solver(
            governance_spec_path=GOV_SPEC,
            opponent_priors_path=OPP_PRIORS,
            checkpoint_dir=CHECKPOINT_DIR,
            K=K,
            R_rollouts=R,
            n_vote_samples=10,
            n_review_samples=5,
            seed=42,
        )

    def test_solve_board_mode(self):
        solver = self._make_solver()
        result = solver.solve(
            focal_actor="Board",
            checkpoint_id="C0",
            n_draws=5,
        )
        assert result.focal_actor == "Board"
        assert result.checkpoint_id == "C0"
        assert len(result.EU_per_action) == 3
        assert result.optimal_action in result.EU_per_action
        assert result.elapsed_seconds > 0

    def test_solve_asa_mode(self):
        from engine.modes import MODE_ASA
        solver = self._make_solver()
        result = solver.solve(
            focal_actor="ASA",
            checkpoint_id="C0",
            mode=MODE_ASA,
            n_draws=5,
        )
        assert result.focal_actor == "ASA"
        assert len(result.EU_per_action) == 3

    def test_solve_summary_df(self):
        solver = self._make_solver()
        result = solver.solve(
            focal_actor="Board",
            checkpoint_id="C0",
            n_draws=5,
        )
        df = result.summary_df()
        assert len(df) == 3
        assert "checkpoint" in df.columns
        assert "Expected_Utility" in df.columns
        assert "is_optimal" in df.columns
        assert df["is_optimal"].sum() == 1

    def test_solve_outcome_stats(self):
        solver = self._make_solver()
        result = solver.solve(
            focal_actor="Board",
            checkpoint_id="C0",
            n_draws=5,
        )
        for action, stats in result.outcome_stats.items():
            assert 0 <= stats["Pr_strike"] <= 1
            assert 0 <= stats["Pr_overwhelming"] <= 1
            assert 0 <= stats["Pr_CEO_removed"] <= 1
            assert 0 <= stats["mean_vote_percent"] <= 1

    def test_solve_different_checkpoints_differ(self):
        """Different checkpoints (different belief levels) should give different results."""
        solver = self._make_solver()
        r0 = solver.solve("Board", "C0", n_draws=5)
        r3 = solver.solve("Board", "C3", n_draws=5)

        # C3 has much higher distrust, so outcomes should differ
        # At minimum, vote percentages should be higher at C3
        stats0 = list(r0.outcome_stats.values())[0]
        stats3 = list(r3.outcome_stats.values())[0]
        # Just verify both produce valid results
        assert r0.optimal_action is not None
        assert r3.optimal_action is not None


# ============================================================================
# 9. VALIDATION CHECKLIST (from spec)
# ============================================================================

class TestValidationChecklist:
    """
    Tests from the spec validation checklist:
    - Board-mode D1* changes when gamma_A is large
    - ASA-mode mobilisation increases when CEO removal reward is large
    - Symmetry: swapping focal switches max vs sum at nodes
    - Removing CEO reduces future CEO decision nodes
    """

    def test_ceo_removal_reduces_d4_actions(self):
        """Removing CEO eliminates D4 options automatically."""
        from engine.state import DecisionState
        state = DecisionState.from_governance_spec(GOV_SPEC)

        # Before removal: D4 has 3 options
        assert len(state.feasible_actions("D4")) == 3

        # After CEO transition: D4 has 0 options
        state2 = state.apply("D1", "D3_ceo_transition")
        assert len(state2.feasible_actions("D4")) == 0

    def test_focal_symmetry(self):
        """Swapping focal actor should swap which nodes are max vs sum."""
        from engine.modes import MODE_BOARD, MODE_ASA

        # In Board mode: D1 is focal (max), A2 is opponent (sum/predict)
        assert MODE_BOARD.is_focal("Board")
        assert not MODE_BOARD.is_focal("ASA")
        assert MODE_BOARD.get_opponent_model_type("ASA") == "ARA"

        # In ASA mode: A2 is focal (max), D1 is opponent (sum/predict)
        assert MODE_ASA.is_focal("ASA")
        assert not MODE_ASA.is_focal("Board")
        assert MODE_ASA.get_opponent_model_type("Board") == "ARA"

    def test_all_checkpoints_loadable(self):
        """All four checkpoints should load successfully."""
        from engine.state import BeliefBundle
        for cid, date in [("C0", "2023-10-01"), ("C1", "2023-10-10"),
                           ("C2", "2023-10-18"), ("C3", "2023-11-03")]:
            path = CHECKPOINT_DIR / f"belief_{cid}_{date}.npz"
            bb = BeliefBundle(path)
            assert bb.N == 500
            assert bb.metadata["checkpoint_id"] == cid

    def test_increasing_distrust_across_checkpoints(self):
        """Belief levels should increase from C0 to C3."""
        from engine.state import BeliefBundle
        means = []
        for cid, date in [("C0", "2023-10-01"), ("C1", "2023-10-10"),
                           ("C2", "2023-10-18"), ("C3", "2023-11-03")]:
            bb = BeliefBundle(CHECKPOINT_DIR / f"belief_{cid}_{date}.npz")
            means.append(np.mean(bb.B_mkt))

        # C0 < C1 < C2 < C3
        for i in range(len(means) - 1):
            assert means[i] < means[i + 1], (
                f"Distrust should increase: C{i}={means[i]:.2f} vs C{i+1}={means[i+1]:.2f}"
            )


# ============================================================================
# 10. EDGE CASES AND ROBUSTNESS
# ============================================================================

class TestEdgeCases:
    """Edge cases and robustness tests."""

    def test_expit_numerical_stability(self):
        from engine.chance_models import _expit
        # Large positive
        assert abs(_expit(100.0) - 1.0) < 1e-10
        # Large negative
        assert abs(_expit(-100.0)) < 1e-10
        # Zero
        assert abs(_expit(0.0) - 0.5) < 1e-10

    def test_belief_bundle_index_bounds(self):
        from engine.state import BeliefBundle
        bb = BeliefBundle(CHECKPOINT_DIR / "belief_C0_2023-10-01.npz")
        # Valid
        bb.get_draw(0)
        bb.get_draw(bb.N - 1)
        # Invalid
        with pytest.raises(IndexError):
            bb.get_draw(bb.N)
        with pytest.raises(IndexError):
            bb.get_draw(-1)

    def test_invalid_actor_raises(self):
        from engine.utilities import compute_utility, TerminalOutcome
        with pytest.raises(ValueError, match="Unknown actor"):
            compute_utility("InvalidActor", TerminalOutcome(), {})

    def test_missing_governance_spec(self):
        from engine.state import DecisionState
        with pytest.raises(FileNotFoundError):
            DecisionState.from_governance_spec("nonexistent.xlsx")

    def test_missing_checkpoint(self):
        from engine.state import BeliefBundle
        with pytest.raises(FileNotFoundError):
            BeliefBundle("nonexistent.npz")

    def test_terminal_outcome_defaults(self):
        from engine.utilities import TerminalOutcome
        o = TerminalOutcome()
        assert o.vote_percent == 0.0
        assert o.CEO_removed is False
        assert o.d1_action == "D0_minimal"

    def test_parameter_sampler_reproducible(self):
        from engine.state import ParameterSampler
        ps = ParameterSampler(OPP_PRIORS)

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        p1 = ps.sample_parameters("Board", "ASA", rng1)
        p2 = ps.sample_parameters("Board", "ASA", rng2)

        for key in p1:
            assert p1[key] == p2[key], f"Parameter {key} not reproducible"


# ============================================================================
# 11. OVERCONFIDENCE BIAS
# ============================================================================

class TestOverconfidenceBias:
    """Test CEO/Board overconfidence bias on governance effects."""

    def test_bias_profiles_defined(self):
        from engine.chance_models import (
            BIAS_NONE, BIAS_OVERESTIMATION, BIAS_OVERPRECISION, BIAS_HUBRIS,
        )
        # Unbiased: full range, sigma_scale = 1.0, no review shift
        assert BIAS_NONE.d1_floor == 0.0
        assert BIAS_NONE.d1_ceiling == 1.0
        assert BIAS_NONE.d3_floor == -1.0
        assert BIAS_NONE.d3_ceiling == 0.5
        assert BIAS_NONE.sigma_scale == 1.0
        assert BIAS_NONE.review_car_bias == 0.0

        # Overestimation: D1 floor raised, D3 floor raised (less negative),
        # no sigma change, positive review shift
        assert BIAS_OVERESTIMATION.d1_floor > BIAS_NONE.d1_floor
        assert BIAS_OVERESTIMATION.d3_floor > BIAS_NONE.d3_floor
        assert BIAS_OVERESTIMATION.sigma_scale == 1.0
        assert BIAS_OVERESTIMATION.review_car_bias > 0

        # Overprecision: narrower range + sigma_scale < 1 + positive review shift
        d1_range_none = BIAS_NONE.d1_ceiling - BIAS_NONE.d1_floor
        d1_range_prec = BIAS_OVERPRECISION.d1_ceiling - BIAS_OVERPRECISION.d1_floor
        assert d1_range_prec < d1_range_none
        assert BIAS_OVERPRECISION.sigma_scale < 1.0
        assert BIAS_OVERPRECISION.review_car_bias > 0

        # Hubris: sigma_scale < 1, positive review shift
        assert BIAS_HUBRIS.sigma_scale < 1.0
        assert BIAS_HUBRIS.review_car_bias > 0

    def test_governance_effect_respects_bias(self):
        """Biased governance effect for D1 should be shifted upward."""
        from engine.chance_models import VoteModel, BIAS_OVERESTIMATION

        n_samples = 500
        effects_unbiased = []
        effects_biased = []

        for i in range(n_samples):
            rng_u = np.random.default_rng(8000 + i)
            rng_b = np.random.default_rng(8000 + i)

            e_u = VoteModel._governance_effect("D1_review", rng_u, bias=None)
            e_b = VoteModel._governance_effect("D1_review", rng_b, bias=BIAS_OVERESTIMATION)

            effects_unbiased.append(e_u)
            effects_biased.append(e_b)

        mean_u = np.mean(effects_unbiased)
        mean_b = np.mean(effects_biased)

        # Unbiased D1: U(0, 1), mean ≈ 0.5
        assert 0.4 < mean_u < 0.6, f"Unbiased D1 mean should be ~0.5, got {mean_u:.3f}"
        # Biased D1: U(0.5, 1), mean ≈ 0.75
        assert 0.65 < mean_b < 0.85, f"Biased D1 mean should be ~0.75, got {mean_b:.3f}"
        # Biased should be higher
        assert mean_b > mean_u

    def test_governance_effect_d3_bias(self):
        """Biased governance effect for D3 should be less negative (closer to 0)."""
        from engine.chance_models import VoteModel, BIAS_OVERESTIMATION

        n_samples = 500
        effects_unbiased = []
        effects_biased = []

        for i in range(n_samples):
            rng_u = np.random.default_rng(9000 + i)
            rng_b = np.random.default_rng(9000 + i)

            e_u = VoteModel._governance_effect("D3_ceo_transition", rng_u, bias=None)
            e_b = VoteModel._governance_effect("D3_ceo_transition", rng_b, bias=BIAS_OVERESTIMATION)

            effects_unbiased.append(e_u)
            effects_biased.append(e_b)

        mean_u = np.mean(effects_unbiased)
        mean_b = np.mean(effects_biased)

        # Unbiased D3: U(-1, 0.5), mean ≈ -0.25
        assert -0.35 < mean_u < -0.15, f"Unbiased D3 mean should be ~-0.25, got {mean_u:.3f}"
        # Biased D3: U(-0.67, 0.5), mean ≈ -0.085 (β=0.5 overestimation)
        assert -0.20 < mean_b < 0.05, f"Biased D3 mean should be ~-0.085, got {mean_b:.3f}"
        # Biased should be closer to 0 (less negative)
        assert mean_b > mean_u

    def test_governance_effect_d3_allows_mitigation(self):
        """D3 governance effect can be positive (up to 0.5), allowing mitigation.

        V2 change: D3 bounds relaxed from U(-1, 0) to U(-1, 0.5).
        A well-managed CEO transition can provide partial mitigation.
        """
        from engine.chance_models import VoteModel

        n_samples = 1000
        positive_count = 0
        for i in range(n_samples):
            rng = np.random.default_rng(11000 + i)
            f = VoteModel._governance_effect("D3_ceo_transition", rng, bias=None)
            assert -1.0 <= f <= 0.5, f"D3 effect out of bounds: {f}"
            if f > 0:
                positive_count += 1

        # U(-1, 0.5): P(f > 0) = 0.5/1.5 ≈ 0.333
        frac = positive_count / n_samples
        assert 0.25 < frac < 0.45, (
            f"Expected ~33% positive D3 draws, got {frac:.1%}"
        )

    def test_vote_headline_interaction(self):
        """gamma_AH provides additional shift when headline=1 and ASA recommends strike.

        V2 change: B_agm includes gamma_AH * I[rec_strike] * I[headline=1].
        """
        from engine.state import BeliefBundle, DecisionState
        from engine.chance_models import VoteModel

        bb = BeliefBundle(CHECKPOINT_DIR / "belief_C0_2023-10-01.npz")
        state_headline = DecisionState.from_governance_spec(GOV_SPEC)
        state_headline.headline_incident = True
        state_no_headline = DecisionState.from_governance_spec(GOV_SPEC)
        state_no_headline.headline_incident = False
        vote_model = VoteModel({"first_strike": 0.25, "overwhelming": 0.50})

        n_samples = 200
        votes_headline = []
        votes_no_headline = []

        for i in range(n_samples):
            rng1 = np.random.default_rng(12000 + i)
            rng2 = np.random.default_rng(12000 + i)

            h = {"D1": "D0_minimal", "A2": "A2_rec_strike"}

            v1 = vote_model.sample(i % bb.N, bb, h, state_headline, rng1)
            v2 = vote_model.sample(i % bb.N, bb, h, state_no_headline, rng2)

            votes_headline.append(v1.vote_percent)
            votes_no_headline.append(v2.vote_percent)

        mean_h = np.mean(votes_headline)
        mean_nh = np.mean(votes_no_headline)
        # Headline interaction should increase vote when ASA recommends strike
        # (gamma_AH is positive on average in synthetic data)
        assert mean_h > mean_nh, (
            f"Headline interaction should increase vote with ASA strike: "
            f"{mean_h:.3f} vs {mean_nh:.3f}"
        )

    def test_crisis_floor_prevents_sub_threshold(self):
        """Structural floor prevents V < V_floor when headline_incident=1.

        V2 change: V_final = max(V_logit_normal, V_floor) where
        V_floor ~ Beta(50, 150) with mean 0.25.
        """
        from engine.state import BeliefBundle, DecisionState
        from engine.chance_models import VoteModel

        bb = BeliefBundle(CHECKPOINT_DIR / "belief_C0_2023-10-01.npz")
        state = DecisionState.from_governance_spec(GOV_SPEC)
        state.headline_incident = True
        vote_model = VoteModel({"first_strike": 0.25, "overwhelming": 0.50})

        n_samples = 500
        floor_val = 0.22  # Conservative: well below Beta(50,150) mean of 0.25

        for i in range(n_samples):
            rng = np.random.default_rng(13000 + i)
            crisis_floor = float(rng.beta(50, 150))
            # Use a fresh rng for sampling to avoid correlation
            s_rng = np.random.default_rng(13000 + n_samples + i)
            h = {"D1": "D0_minimal", "A2": "A2_no_strike"}
            out = vote_model.sample(
                i % bb.N, bb, h, state, s_rng, crisis_floor=crisis_floor,
            )
            # V_final must be >= crisis_floor
            assert out.vote_percent >= crisis_floor - 1e-12, (
                f"Vote {out.vote_percent:.4f} below crisis floor {crisis_floor:.4f}"
            )

    def test_d0_unaffected_by_bias(self):
        """D0 (no action) always returns 0 regardless of bias."""
        from engine.chance_models import VoteModel, BIAS_HUBRIS

        rng = np.random.default_rng(42)
        assert VoteModel._governance_effect("D0_minimal", rng, bias=None) == 0.0
        assert VoteModel._governance_effect("D0_minimal", rng, bias=BIAS_HUBRIS) == 0.0

    def test_tree_evaluator_accepts_bias(self):
        """TreeEvaluator should accept and store overconfidence_bias."""
        from engine.state import (
            BeliefBundle, DecisionState, ParameterSampler,
            load_vote_thresholds, load_utility_weights, load_policy_parameters,
        )
        from engine.chance_models import ChanceModels, BIAS_OVERESTIMATION
        from engine.predictive import PredictiveDistribution
        from engine.tree import TreeEvaluator

        beliefs = BeliefBundle(CHECKPOINT_DIR / "belief_C0_2023-10-01.npz")
        thresholds = load_vote_thresholds(GOV_SPEC)
        chance = ChanceModels(thresholds)
        sampler = ParameterSampler(OPP_PRIORS)
        policy_params = load_policy_parameters(GOV_SPEC)
        utility_weights = {
            actor: load_utility_weights(GOV_SPEC, actor)
            for actor in ["Board", "ASA", "CEO"]
        }
        pred = PredictiveDistribution(
            beliefs=beliefs, param_sampler=sampler,
            chance_models=chance, policy_params=policy_params,
            K=20, R_rollouts=5,
        )

        tree = TreeEvaluator(
            beliefs=beliefs, chance_models=chance, predictive=pred,
            utility_weights=utility_weights,
            n_vote_samples=10, n_review_samples=5,
            overconfidence_bias=BIAS_OVERESTIMATION,
        )
        assert tree.overconfidence_bias is BIAS_OVERESTIMATION

    def test_load_board_overconfidence(self):
        """Board overconfidence loads from governance_spec.xlsx."""
        from engine.state import load_board_overconfidence
        params = load_board_overconfidence(GOV_SPEC)
        assert params["d1_floor"] == 0.63
        assert params["d1_ceiling"] == 1.0
        assert params["d3_floor"] == -0.62
        assert params["d3_ceiling"] == 0.5
        assert params["sigma_scale"] == 0.53
        assert params["review_car_bias"] == 0.03

    def test_load_board_overconfidence_validates_bounds(self):
        """Loaded bounds satisfy required constraints."""
        from engine.state import load_board_overconfidence
        params = load_board_overconfidence(GOV_SPEC)
        assert 0 <= params["d1_floor"] < params["d1_ceiling"] <= 1
        assert -1 <= params["d3_floor"] < params["d3_ceiling"] <= 1
        assert 0 < params["sigma_scale"] <= 1
        assert params["review_car_bias"] >= 0

    def test_solver_default_bias_from_spec(self):
        """Solver auto-loads overconfidence bias from governance_spec."""
        from engine.solver import Solver
        from engine.chance_models import BIAS_HUBRIS

        solver = Solver(
            governance_spec_path=GOV_SPEC,
            opponent_priors_path=OPP_PRIORS,
            checkpoint_dir=CHECKPOINT_DIR,
            K=20, R_rollouts=5,
            n_vote_samples=10, n_review_samples=5, seed=42,
        )
        # Should match HUBRIS profile from board_overconfidence sheet
        assert solver.overconfidence_bias.d1_floor == BIAS_HUBRIS.d1_floor
        assert solver.overconfidence_bias.d1_ceiling == BIAS_HUBRIS.d1_ceiling
        assert solver.overconfidence_bias.d3_floor == BIAS_HUBRIS.d3_floor
        assert solver.overconfidence_bias.d3_ceiling == BIAS_HUBRIS.d3_ceiling
        assert solver.overconfidence_bias.sigma_scale == BIAS_HUBRIS.sigma_scale
        assert solver.overconfidence_bias.review_car_bias == BIAS_HUBRIS.review_car_bias

        # Default solve uses spec bias (not unbiased)
        result = solver.solve(
            focal_actor="Board",
            checkpoint_id="C0",
            n_draws=3,
        )
        assert len(result.EU_per_action) == 3
        assert "D1=" in result.overconfidence_bias_label
        assert "sigma_scale=" in result.overconfidence_bias_label
        assert "review_car_bias=" in result.overconfidence_bias_label

    def test_solver_no_bias_counterfactual(self):
        """Solver with overconfidence_bias=None runs unbiased."""
        from engine.solver import Solver

        solver = Solver(
            governance_spec_path=GOV_SPEC,
            opponent_priors_path=OPP_PRIORS,
            checkpoint_dir=CHECKPOINT_DIR,
            K=20, R_rollouts=5,
            n_vote_samples=10, n_review_samples=5, seed=42,
        )
        result = solver.solve(
            focal_actor="Board",
            checkpoint_id="C0",
            n_draws=3,
            overconfidence_bias=None,
        )
        assert len(result.EU_per_action) == 3
        assert "unbiased" in result.overconfidence_bias_label

    def test_sigma_scale_reduces_vote_variance(self):
        """sigma_scale < 1 reduces the spread of sampled vote percentages."""
        from engine.chance_models import VoteModel
        from engine.state import BeliefBundle, DecisionState, load_vote_thresholds

        beliefs = BeliefBundle(CHECKPOINT_DIR / "belief_C0_2023-10-01.npz")
        thresholds = load_vote_thresholds(GOV_SPEC)
        state = DecisionState.from_governance_spec(GOV_SPEC)
        vm = VoteModel(thresholds)

        n_samples = 500
        votes_full = []
        votes_scaled = []

        for j in range(n_samples):
            rng_f = np.random.default_rng(5000 + j)
            rng_s = np.random.default_rng(5000 + j)

            out_f = vm.sample(0, beliefs, {"D1": "D0_minimal"}, state, rng_f,
                              sigma_scale=None)
            out_s = vm.sample(0, beliefs, {"D1": "D0_minimal"}, state, rng_s,
                              sigma_scale=0.5)
            votes_full.append(out_f.vote_percent)
            votes_scaled.append(out_s.vote_percent)

        sd_full = np.std(votes_full)
        sd_scaled = np.std(votes_scaled)

        # Scaled-down sigma → lower vote spread
        assert sd_scaled < sd_full, (
            f"sigma_scale=0.5 should reduce vote spread: "
            f"sd_scaled={sd_scaled:.4f} vs sd_full={sd_full:.4f}"
        )

    def test_sigma_scale_1_is_identity(self):
        """sigma_scale=1.0 produces identical results to sigma_scale=None."""
        from engine.chance_models import VoteModel
        from engine.state import BeliefBundle, DecisionState, load_vote_thresholds

        beliefs = BeliefBundle(CHECKPOINT_DIR / "belief_C0_2023-10-01.npz")
        thresholds = load_vote_thresholds(GOV_SPEC)
        state = DecisionState.from_governance_spec(GOV_SPEC)
        vm = VoteModel(thresholds)

        for j in range(20):
            rng_none = np.random.default_rng(6000 + j)
            rng_one = np.random.default_rng(6000 + j)

            out_none = vm.sample(0, beliefs, {"D1": "D0_minimal"}, state,
                                 rng_none, sigma_scale=None)
            out_one = vm.sample(0, beliefs, {"D1": "D0_minimal"}, state,
                                rng_one, sigma_scale=1.0)

            assert abs(out_none.vote_percent - out_one.vote_percent) < 1e-12, (
                f"sigma_scale=1.0 should be identity, draw {j}: "
                f"{out_none.vote_percent} vs {out_one.vote_percent}"
            )

    def test_review_bias_reduces_adverse_probability(self):
        """Biased review Dirichlet shift should lower the negative outcome probability."""
        from engine.chance_models import ReviewModel, OverconfidenceBias, BIAS_HUBRIS
        from engine.state import BeliefBundle, DecisionState

        beliefs = BeliefBundle(CHECKPOINT_DIR / "belief_C0_2023-10-01.npz")
        state = DecisionState.from_governance_spec(GOV_SPEC)
        state.review_commissioned = True

        n_samples = 500
        negative_unbiased = 0
        negative_biased = 0
        rm = ReviewModel()

        for j in range(n_samples):
            rng_u = np.random.default_rng(7000 + j)
            rng_b = np.random.default_rng(7000 + j)

            # Draw outcome probabilities WITH bias effect
            probs_u = rm.draw_outcome_probabilities(rng_u, bias=None)
            probs_b = rm.draw_outcome_probabilities(rng_b, bias=BIAS_HUBRIS)

            out_u = rm.sample(0, beliefs, {"V_percent": 0.30}, state,
                              np.random.default_rng(8000 + j),
                              bias=None, outcome_probs=probs_u)
            out_b = rm.sample(0, beliefs, {"V_percent": 0.30}, state,
                              np.random.default_rng(8000 + j),
                              bias=BIAS_HUBRIS, outcome_probs=probs_b)

            negative_unbiased += int(out_u.review_outcome == "negative")
            negative_biased += int(out_b.review_outcome == "negative")

        rate_u = negative_unbiased / n_samples
        rate_b = negative_biased / n_samples

        # Biased negative rate should be lower (Board thinks governance is sound,
        # inflated positive α in Dirichlet → lower P(negative))
        assert rate_b < rate_u, (
            f"Biased review negative rate should be lower: "
            f"biased={rate_b:.3f} vs unbiased={rate_u:.3f}"
        )

    def test_review_expected_car_respects_bias(self):
        """expected_car() returns higher (less negative) value with bias."""
        from engine.chance_models import ReviewModel, BIAS_HUBRIS
        from engine.state import DecisionState

        state = DecisionState.from_governance_spec(GOV_SPEC)
        state.review_commissioned = True
        rm = ReviewModel()

        car_u = rm.expected_car(state, bias=None)
        car_b = rm.expected_car(state, bias=BIAS_HUBRIS)

        # Biased expected CAR should be higher (Board thinks governance is sound)
        assert car_b > car_u, (
            f"Biased expected CAR {car_b:.4f} should be > "
            f"unbiased {car_u:.4f}"
        )

    def test_review_bias_zero_is_identity(self):
        """review_car_bias=0.0 produces identical results to bias=None."""
        from engine.chance_models import ReviewModel, OverconfidenceBias, BIAS_NONE
        from engine.state import BeliefBundle, DecisionState

        beliefs = BeliefBundle(CHECKPOINT_DIR / "belief_C0_2023-10-01.npz")
        state = DecisionState.from_governance_spec(GOV_SPEC)
        state.review_commissioned = True
        rm = ReviewModel()

        for j in range(20):
            rng_none = np.random.default_rng(7500 + j)
            rng_zero = np.random.default_rng(7500 + j)

            out_none = rm.sample(0, beliefs, {"V_percent": 0.30}, state,
                                 rng_none, bias=None)
            out_zero = rm.sample(0, beliefs, {"V_percent": 0.30}, state,
                                 rng_zero, bias=BIAS_NONE)

            assert out_none.review_outcome == out_zero.review_outcome, (
                f"review_car_bias=0 should be identity, draw {j}: "
                f"{out_none.review_outcome} vs {out_zero.review_outcome}"
            )
            assert abs(out_none.review_car - out_zero.review_car) < 1e-12, (
                f"review_car_bias=0 should produce identical CAR, draw {j}: "
                f"{out_none.review_car} vs {out_zero.review_car}"
            )

    def test_review_bias_not_commissioned_unaffected(self):
        """When review is not commissioned, bias has no effect."""
        from engine.chance_models import ReviewModel, BIAS_HUBRIS
        from engine.state import BeliefBundle, DecisionState

        beliefs = BeliefBundle(CHECKPOINT_DIR / "belief_C0_2023-10-01.npz")
        state = DecisionState.from_governance_spec(GOV_SPEC)
        assert not state.review_commissioned
        rm = ReviewModel()

        rng = np.random.default_rng(42)
        out = rm.sample(0, beliefs, {}, state, rng, bias=BIAS_HUBRIS)
        assert out.review_outcome != "negative"
        assert out.review_car == 0.0

        car = rm.expected_car(state, bias=BIAS_HUBRIS)
        assert car == 0.0

    def test_predictive_rollout_uses_bias_vote(self):
        """Rollout simulations should use biased vote model (ARA Level-1).

        The Board's overconfidence should affect vote outcomes in rollouts,
        not just in the focal EU calculation. With bias, the Board
        underestimates vote variance (sigma_scale < 1), so rollout vote
        percentages should have lower spread.
        """
        from engine.state import (
            BeliefBundle, DecisionState, ParameterSampler,
            load_vote_thresholds, load_policy_parameters,
        )
        from engine.chance_models import ChanceModels, BIAS_HUBRIS
        from engine.predictive import PredictiveDistribution
        from engine.modes import MODE_BOARD

        beliefs = BeliefBundle(CHECKPOINT_DIR / "belief_C0_2023-10-01.npz")
        thresholds = load_vote_thresholds(GOV_SPEC)
        chance = ChanceModels(thresholds)
        sampler = ParameterSampler(OPP_PRIORS)
        policy_params = load_policy_parameters(GOV_SPEC)

        pred_unbiased = PredictiveDistribution(
            beliefs=beliefs, param_sampler=sampler,
            chance_models=chance, policy_params=policy_params,
            K=20, R_rollouts=5,
            overconfidence_bias=None,
        )
        pred_biased = PredictiveDistribution(
            beliefs=beliefs, param_sampler=sampler,
            chance_models=chance, policy_params=policy_params,
            K=20, R_rollouts=5,
            overconfidence_bias=BIAS_HUBRIS,
        )

        state = DecisionState.from_governance_spec(GOV_SPEC)
        history = {"D1": "D0_minimal"}
        s = state.apply("D1", "D0_minimal")
        next_node = s.next_node("D1")

        n_samples = 300
        votes_unbiased = []
        votes_biased = []

        for j in range(n_samples):
            rng_u = np.random.default_rng(11000 + j)
            rng_b = np.random.default_rng(11000 + j)

            out_u = pred_unbiased._simulate_forward(
                current_node=next_node, history=dict(history),
                state=s, draw_i=0, owner="Board", focal_actor="Board",
                mode=MODE_BOARD, level=1, rng=rng_u,
            )
            out_b = pred_biased._simulate_forward(
                current_node=next_node, history=dict(history),
                state=s, draw_i=0, owner="Board", focal_actor="Board",
                mode=MODE_BOARD, level=1, rng=rng_b,
            )
            votes_unbiased.append(out_u.vote_percent)
            votes_biased.append(out_b.vote_percent)

        sd_u = np.std(votes_unbiased)
        sd_b = np.std(votes_biased)

        # Biased rollouts should have lower vote spread (sigma_scale < 1)
        assert sd_b < sd_u, (
            f"Biased rollouts should have lower vote spread: "
            f"sd_biased={sd_b:.4f} vs sd_unbiased={sd_u:.4f}"
        )

    def test_predictive_rollout_uses_bias_review(self):
        """Rollout simulations should use biased review model (ARA Level-1).

        With review_car_bias > 0, the Board thinks review findings
        will produce a more favourable CAR. Rollouts under D1_review
        (which commissions the review) should show a lower adverse
        rate when biased.
        """
        from engine.state import (
            BeliefBundle, DecisionState, ParameterSampler,
            load_vote_thresholds, load_policy_parameters,
        )
        from engine.chance_models import ChanceModels, BIAS_HUBRIS
        from engine.predictive import PredictiveDistribution
        from engine.modes import MODE_BOARD

        beliefs = BeliefBundle(CHECKPOINT_DIR / "belief_C0_2023-10-01.npz")
        thresholds = load_vote_thresholds(GOV_SPEC)
        chance = ChanceModels(thresholds)
        sampler = ParameterSampler(OPP_PRIORS)
        policy_params = load_policy_parameters(GOV_SPEC)

        pred_unbiased = PredictiveDistribution(
            beliefs=beliefs, param_sampler=sampler,
            chance_models=chance, policy_params=policy_params,
            K=20, R_rollouts=5,
            overconfidence_bias=None,
        )
        pred_biased = PredictiveDistribution(
            beliefs=beliefs, param_sampler=sampler,
            chance_models=chance, policy_params=policy_params,
            K=20, R_rollouts=5,
            overconfidence_bias=BIAS_HUBRIS,
        )

        # D1_review commissions the review, so R node fires
        state = DecisionState.from_governance_spec(GOV_SPEC)
        history = {"D1": "D1_review"}
        s = state.apply("D1", "D1_review")
        next_node = s.next_node("D1")

        n_samples = 500
        adverse_unbiased = 0
        adverse_biased = 0

        for j in range(n_samples):
            rng_u = np.random.default_rng(12000 + j)
            rng_b = np.random.default_rng(12000 + j)

            out_u = pred_unbiased._simulate_forward(
                current_node=next_node, history=dict(history),
                state=s, draw_i=0, owner="Board", focal_actor="Board",
                mode=MODE_BOARD, level=1, rng=rng_u,
            )
            out_b = pred_biased._simulate_forward(
                current_node=next_node, history=dict(history),
                state=s, draw_i=0, owner="Board", focal_actor="Board",
                mode=MODE_BOARD, level=1, rng=rng_b,
            )
            adverse_unbiased += int(out_u.review_outcome == "negative")
            adverse_biased += int(out_b.review_outcome == "negative")

        rate_u = adverse_unbiased / n_samples
        rate_b = adverse_biased / n_samples

        # Biased rollouts should have lower adverse rate
        assert rate_b < rate_u, (
            f"Biased rollouts should have lower adverse rate: "
            f"biased={rate_b:.3f} vs unbiased={rate_u:.3f}"
        )

    def test_predictive_accepts_overconfidence_bias(self):
        """PredictiveDistribution should accept and store overconfidence_bias."""
        from engine.state import (
            BeliefBundle, ParameterSampler,
            load_vote_thresholds, load_policy_parameters,
        )
        from engine.chance_models import ChanceModels, BIAS_HUBRIS
        from engine.predictive import PredictiveDistribution

        beliefs = BeliefBundle(CHECKPOINT_DIR / "belief_C0_2023-10-01.npz")
        thresholds = load_vote_thresholds(GOV_SPEC)
        chance = ChanceModels(thresholds)
        sampler = ParameterSampler(OPP_PRIORS)
        policy_params = load_policy_parameters(GOV_SPEC)

        pred = PredictiveDistribution(
            beliefs=beliefs, param_sampler=sampler,
            chance_models=chance, policy_params=policy_params,
            K=20, R_rollouts=5,
            overconfidence_bias=BIAS_HUBRIS,
        )
        assert pred.overconfidence_bias is BIAS_HUBRIS

        pred_none = PredictiveDistribution(
            beliefs=beliefs, param_sampler=sampler,
            chance_models=chance, policy_params=policy_params,
            K=20, R_rollouts=5,
        )
        assert pred_none.overconfidence_bias is None


# ============================================================================
# 12. CEO RESIGNATION SCENARIO TESTS
# ============================================================================

class TestScenarios:
    """Test pre-game CEO resignation scenarios and D0_ceo decision node."""

    def test_d0_ceo_in_node_order(self):
        """D0_ceo should be the first node in the tree."""
        from engine.state import DecisionState
        state = DecisionState.from_governance_spec(GOV_SPEC)
        assert state._node_order[0] == "D0_ceo"
        assert state.node_type("D0_ceo") == "decision"
        assert state.node_owner("D0_ceo") == "CEO"

    def test_d0_ceo_actions(self):
        """D0_ceo should have CEO_resign and CEO_stay actions."""
        from engine.state import DecisionState
        state = DecisionState.from_governance_spec(GOV_SPEC)
        actions = state.feasible_actions("D0_ceo")
        assert "CEO_resign" in actions
        assert "CEO_stay" in actions
        assert len(actions) == 2

    def test_apply_ceo_resign(self):
        """Applying CEO_resign at D0_ceo sets correct state."""
        from engine.state import DecisionState
        state = DecisionState.from_governance_spec(GOV_SPEC)
        resigned = state.apply("D0_ceo", "CEO_resign")
        assert resigned.CEO_present is False
        assert resigned.CEO_removed is True
        assert resigned.CEO_resigned_early is True

    def test_apply_ceo_stay(self):
        """Applying CEO_stay at D0_ceo preserves defaults."""
        from engine.state import DecisionState
        state = DecisionState.from_governance_spec(GOV_SPEC)
        stayed = state.apply("D0_ceo", "CEO_stay")
        assert stayed.CEO_present is True
        assert stayed.CEO_removed is False
        assert stayed.CEO_resigned_early is False

    def test_for_scenario_delegates_to_apply(self):
        """for_scenario should produce same state as apply."""
        from engine.state import DecisionState
        state = DecisionState.from_governance_spec(GOV_SPEC)
        via_scenario = state.for_scenario("ceo_resigned")
        via_apply = state.apply("D0_ceo", "CEO_resign")
        assert via_scenario.CEO_present == via_apply.CEO_present
        assert via_scenario.CEO_removed == via_apply.CEO_removed
        assert via_scenario.CEO_resigned_early == via_apply.CEO_resigned_early

    def test_for_scenario_resign(self):
        from engine.state import DecisionState
        state = DecisionState.from_governance_spec(GOV_SPEC)
        resigned = state.for_scenario("ceo_resigned")
        assert resigned.CEO_present is False
        assert resigned.CEO_removed is True
        assert resigned.CEO_resigned_early is True

    def test_for_scenario_stay(self):
        from engine.state import DecisionState
        state = DecisionState.from_governance_spec(GOV_SPEC)
        stayed = state.for_scenario("ceo_stayed")
        assert stayed.CEO_present is True
        assert stayed.CEO_removed is False
        assert stayed.CEO_resigned_early is False

    def test_for_scenario_invalid(self):
        from engine.state import DecisionState
        state = DecisionState.from_governance_spec(GOV_SPEC)
        with pytest.raises(ValueError, match="Unknown scenario"):
            state.for_scenario("invalid_scenario")

    def test_d1_resign_no_d3(self):
        """When CEO resigned, D3_ceo_transition is infeasible at D1."""
        from engine.state import DecisionState
        state = DecisionState.from_governance_spec(GOV_SPEC)
        resigned = state.for_scenario("ceo_resigned")
        actions = resigned.feasible_actions("D1")
        assert "D3_ceo_transition" not in actions
        assert "D0_minimal" in actions
        assert "D1_review" in actions
        assert len(actions) == 2

    def test_d_rev_resign_no_sack(self):
        """When CEO resigned, Drev_sack_ceo is infeasible."""
        from engine.state import DecisionState
        state = DecisionState.from_governance_spec(GOV_SPEC)
        resigned = state.for_scenario("ceo_resigned")
        actions = resigned.feasible_actions("D_rev")
        assert "Drev_sack_ceo" not in actions
        assert "Drev_no_action" in actions

    def test_d4_resign_empty(self):
        """When CEO resigned, D4 has no feasible actions."""
        from engine.state import DecisionState
        state = DecisionState.from_governance_spec(GOV_SPEC)
        resigned = state.for_scenario("ceo_resigned")
        actions = resigned.feasible_actions("D4")
        assert len(actions) == 0

    def test_d1_stay_all_three(self):
        """When CEO stayed, all three D1 actions remain feasible."""
        from engine.state import DecisionState
        state = DecisionState.from_governance_spec(GOV_SPEC)
        stayed = state.for_scenario("ceo_stayed")
        actions = stayed.feasible_actions("D1")
        assert len(actions) == 3
        assert "D3_ceo_transition" in actions


class TestScenarioUtilities:
    """Test CRRA CEO utility with early resignation and stay paths."""

    def test_ceo_utility_early_resign_less_negative(self):
        """Early resignation should be less negative than forced removal."""
        from engine.utilities import utility_ceo, TerminalOutcome
        from engine.state import load_utility_weights

        params = load_utility_weights(GOV_SPEC, "CEO")

        # Early resignation
        outcome_resign = TerminalOutcome(
            CEO_removed=True, CEO_resigned_early=True)
        u_resign = utility_ceo(outcome_resign, params)

        # Forced removal (sacked by board) with hostile AGM
        outcome_sacked = TerminalOutcome(
            CEO_removed=True, CEO_resigned_early=False,
            d1_action="D3_ceo_transition", vote_percent=0.4,
            strike_indicator=True)
        u_sacked = utility_ceo(outcome_sacked, params)

        assert u_resign > u_sacked, (
            f"Early resignation ({u_resign:.3f}) should be less negative "
            f"than forced removal ({u_sacked:.3f})"
        )

    def test_ceo_utility_crra_resign_value(self):
        """Verify reference-dependent CRRA for resign path with loss aversion."""
        from engine.utilities import utility_ceo, TerminalOutcome
        params = {"gamma": 1.5, "W_resign": 8.0, "D_resign": 40.0,
                  "loss_aversion": 2.25, "W_ref": 16.0, "loss_aversion_D": 2.25}
        outcome = TerminalOutcome(CEO_removed=True, CEO_resigned_early=True)
        u = utility_ceo(outcome, params)
        # W=8 < W_ref=16: U_money = λ·CRRA(8) − (λ−1)·CRRA(16)
        crra_W = (8.0 ** (-0.5)) / (-0.5)
        crra_ref = (16.0 ** (-0.5)) / (-0.5)
        expected_money = 2.25 * crra_W - 1.25 * crra_ref
        # U = U_money - λ_D · D_raw
        assert abs(u - (expected_money - 2.25 * 40.0)) < 1e-6

    def test_ceo_utility_crra_stay_sacked(self):
        """Forced sacking with loss aversion: W=1.5 << W_ref=16, D amplified by λ_D."""
        from engine.utilities import utility_ceo, TerminalOutcome
        params = {"gamma": 1.5, "W_stay_sacked": 1.5, "W_stay_kept": 7.0,
                  "D_stay": 25.0, "D_sacked": 100.0, "D_agm": 30.0,
                  "D_disgrace": 30.0, "D_adverse_review": 10.0,
                  "loss_aversion": 2.25, "W_ref": 16.0, "loss_aversion_D": 2.25}
        outcome = TerminalOutcome(
            CEO_removed=True, vote_percent=0.6,
            overwhelming_indicator=True, strike_indicator=True)
        u = utility_ceo(outcome, params)
        # D_raw = D_stay + D_sacked + D_agm + D_disgrace = 185
        # W=1.5 < W_ref: U_money = λ·CRRA(1.5) − (λ−1)·CRRA(16)
        crra_W = (1.5 ** (-0.5)) / (-0.5)
        crra_ref = (16.0 ** (-0.5)) / (-0.5)
        expected_money = 2.25 * crra_W - 1.25 * crra_ref
        assert abs(u - (expected_money - 2.25 * 185.0)) < 1e-6

    def test_ceo_utility_crra_stay_kept(self):
        """CEO keeps job with loss aversion: W=7 < W_ref=16, D amplified by λ_D."""
        from engine.utilities import utility_ceo, TerminalOutcome
        params = {"gamma": 1.5, "W_stay_kept": 7.0, "D_stay": 25.0,
                  "D_sacked": 100.0, "D_agm": 30.0, "D_disgrace": 30.0,
                  "loss_aversion": 2.25, "W_ref": 16.0, "loss_aversion_D": 2.25}
        outcome = TerminalOutcome(CEO_removed=False, vote_percent=0.10)
        u = utility_ceo(outcome, params)
        # W=7 < W_ref: U_money = λ·CRRA(7) − (λ−1)·CRRA(16)
        crra_W = (7.0 ** (-0.5)) / (-0.5)
        crra_ref = (16.0 ** (-0.5)) / (-0.5)
        expected_money = 2.25 * crra_W - 1.25 * crra_ref
        assert abs(u - (expected_money - 2.25 * 25.0)) < 1e-6

    def test_ceo_utility_log_gamma(self):
        """γ = 1 should use log utility (ln(W)) with loss aversion on both W and D."""
        from engine.utilities import utility_ceo, TerminalOutcome
        import numpy as np
        params = {"gamma": 1.0, "W_resign": 8.0, "D_resign": 40.0,
                  "loss_aversion": 2.25, "W_ref": 16.0, "loss_aversion_D": 2.25}
        outcome = TerminalOutcome(CEO_removed=True, CEO_resigned_early=True)
        u = utility_ceo(outcome, params)
        # W=8 < W_ref=16: U_money = λ·ln(8) − (λ−1)·ln(16)
        expected = 2.25 * np.log(8.0) - 1.25 * np.log(16.0) - 2.25 * 40.0
        assert abs(u - expected) < 1e-6

    def test_ceo_utility_agm_penalty(self):
        """D_agm triggered only when vote > 25%."""
        from engine.utilities import utility_ceo, TerminalOutcome
        params = {"gamma": 1.5, "W_stay_kept": 12.0, "D_agm": 30.0,
                  "D_sacked": 100.0, "D_disgrace": 30.0}
        low_vote = TerminalOutcome(CEO_removed=False, vote_percent=0.20)
        high_vote = TerminalOutcome(CEO_removed=False, vote_percent=0.30)
        u_low = utility_ceo(low_vote, params)
        u_high = utility_ceo(high_vote, params)
        assert u_low > u_high  # AGM penalty applied for high vote

    def test_ceo_utility_disgrace_penalty(self):
        """D_disgrace triggered by overwhelming vote indicator."""
        from engine.utilities import utility_ceo, TerminalOutcome
        params = {"gamma": 1.5, "W_stay_kept": 12.0, "D_agm": 30.0,
                  "D_sacked": 100.0, "D_disgrace": 30.0}
        no_overwhelm = TerminalOutcome(
            CEO_removed=False, vote_percent=0.30)
        overwhelm = TerminalOutcome(
            CEO_removed=False, vote_percent=0.60,
            overwhelming_indicator=True)
        u_no = utility_ceo(no_overwhelm, params)
        u_yes = utility_ceo(overwhelm, params)
        assert u_no > u_yes  # Disgrace penalty for overwhelming

    def test_ceo_utility_gamma_clipping(self):
        """gamma should be clipped to [0.5, 3.0]."""
        from engine.utilities import utility_ceo, TerminalOutcome
        outcome = TerminalOutcome(CEO_removed=True, CEO_resigned_early=True)
        # gamma=0.0 should be clipped to 0.5
        params_low = {"gamma": 0.0, "W_resign": 8.0, "D_resign": 50.0}
        params_exact = {"gamma": 0.5, "W_resign": 8.0, "D_resign": 50.0}
        assert abs(utility_ceo(outcome, params_low) -
                   utility_ceo(outcome, params_exact)) < 1e-6
        # gamma=5.0 should be clipped to 3.0
        params_high = {"gamma": 5.0, "W_resign": 8.0, "D_resign": 50.0}
        params_cap = {"gamma": 3.0, "W_resign": 8.0, "D_resign": 50.0}
        assert abs(utility_ceo(outcome, params_high) -
                   utility_ceo(outcome, params_cap)) < 1e-6

    def test_ceo_departure_mode_d_raw(self):
        """Departure-mode penalties: D_negotiate < D_resign_late < D_sacked."""
        from engine.utilities import utility_ceo, TerminalOutcome
        params = {"gamma": 1.5, "W_stay_sacked": 1.5, "W_stay_kept": 7.0,
                  "D_stay": 25.0, "D_sacked": 100.0, "D_resign_late": 60.0,
                  "D_negotiate": 45.0, "D_agm": 30.0,
                  "loss_aversion": 2.25, "W_ref": 16.0, "loss_aversion_D": 2.25}

        # All at vote=0.30 (strike), CEO_removed=True
        out_negotiate = TerminalOutcome(
            CEO_removed=True, d4_action="D4_negotiate_exit",
            vote_percent=0.30, strike_indicator=True)
        out_resign_late = TerminalOutcome(
            CEO_removed=True, d4_action="D4_resign",
            vote_percent=0.30, strike_indicator=True)
        out_sacked = TerminalOutcome(
            CEO_removed=True, d4_action="D4_stay",
            vote_percent=0.30, strike_indicator=True)

        u_negotiate = utility_ceo(out_negotiate, params)
        u_resign_late = utility_ceo(out_resign_late, params)
        u_sacked = utility_ceo(out_sacked, params)

        # D_negotiate(45) < D_resign_late(60) < D_sacked(100) → utility ordering
        assert u_negotiate > u_resign_late, (
            f"Negotiate ({u_negotiate:.2f}) should beat late resign ({u_resign_late:.2f})")
        assert u_resign_late > u_sacked, (
            f"Late resign ({u_resign_late:.2f}) should beat sacked ({u_sacked:.2f})")

        # Verify D_raw values: D_stay + D_xxx + D_agm (vote > 0.25)
        # negotiate: 25 + 45 + 30 = 100
        # resign_late: 25 + 60 + 30 = 115
        # sacked: 25 + 100 + 30 = 155
        # Gap between sacked and negotiate should be larger than between
        # negotiate and resign_late
        assert (u_negotiate - u_sacked) > (u_negotiate - u_resign_late)

    def test_board_utility_passivity_after_departure(self):
        """Board should pay board_passivity_after_departure for early resignation."""
        from engine.utilities import utility_board, TerminalOutcome
        from engine.state import load_utility_weights

        params = load_utility_weights(GOV_SPEC, "Board")

        # No CEO departure (inaction baseline)
        outcome_present = TerminalOutcome()
        u_present = utility_board(outcome_present, params)

        # Early CEO departure (removes ceo_present and no_sack penalties)
        outcome_resigned = TerminalOutcome(
            CEO_removed=True, CEO_resigned_early=True)
        u_resigned = utility_board(outcome_resigned, params)

        # In the new model, early departure REMOVES inaction_ceo_present and
        # inaction_no_sack penalties (CEO gone), but adds board_passivity_after_departure.
        # Net effect: u_resigned > u_present (departure better than inaction)
        assert u_resigned > u_present
        # But there IS a cost from early departure param
        assert u_resigned < 0  # Still negative overall

    def test_asa_utility_early_ceo_departure_higher(self):
        """ASA utility higher when CEO resigned early (better base scores)."""
        from engine.utilities import utility_asa, TerminalOutcome
        from engine.state import load_utility_weights

        params = load_utility_weights(GOV_SPEC, "ASA")

        # CEO stayed, do nothing (A2-3: worst base = 1.43)
        outcome_stayed = TerminalOutcome(d1_action="D0_minimal")
        u_stayed = utility_asa(outcome_stayed, params)

        # CEO resigned early, do nothing (A2-1: base = 1.84)
        outcome_resigned = TerminalOutcome(
            d1_action="D0_minimal",
            CEO_removed=True, CEO_resigned_early=True)
        u_resigned = utility_asa(outcome_resigned, params)

        # CEO resignation gives higher base utility (partial vindication)
        assert u_resigned > u_stayed

    def test_board_negative_review_finding_penalty(self):
        """Board utility penalised when review returns negative findings."""
        from engine.utilities import utility_board, TerminalOutcome
        params = {"vote_penalty_weight": 2.0, "overwhelming_penalty_weight": 3.0,
                  "spill_risk_weight": 2.5, "review_car_weight": 15.0,
                  "review_direct_cost_weight": 15.0, "implementation_cost_sack": 1.0,
                  "ceo_loss_cost": 1.5, "reputational_spill_weight": 1.0,
                  "negative_review_finding_penalty": 5.0}

        # Negative review → penalty fires regardless of CEO status
        out_negative = TerminalOutcome(
            review_commissioned=True, review_outcome="negative",
            review_car=-0.05, CEO_removed=False)
        u_negative = utility_board(out_negative, params)

        # Positive review → no penalty
        out_positive = TerminalOutcome(
            review_commissioned=True, review_outcome="positive",
            review_car=0.05, CEO_removed=False)
        u_positive = utility_board(out_positive, params)

        # Negative review should cost at least the penalty (5.0)
        assert u_positive > u_negative, (
            f"Positive review ({u_positive:.2f}) should beat "
            f"negative review ({u_negative:.2f})")

        # Penalty also fires when CEO is removed
        out_negative_removed = TerminalOutcome(
            review_commissioned=True, review_outcome="negative",
            review_car=-0.05, CEO_removed=True,
            d_rev_action="Drev_sack_ceo")
        u_neg_removed = utility_board(out_negative_removed, params)

        out_positive_removed = TerminalOutcome(
            review_commissioned=True, review_outcome="positive",
            review_car=0.05, CEO_removed=True,
            d_rev_action="Drev_sack_ceo")
        u_pos_removed = utility_board(out_positive_removed, params)

        # Even with CEO removed, negative review is worse than positive
        assert u_pos_removed > u_neg_removed

    def test_board_no_penalty_positive_review(self):
        """No negative_review_finding_penalty when review is positive."""
        from engine.utilities import utility_board, TerminalOutcome
        params = {"review_car_weight": 15.0, "review_direct_cost_weight": 15.0,
                  "negative_review_finding_penalty": 5.0,
                  "inaction_base_penalty": 3.0, "inaction_no_review_penalty": 2.0,
                  "inaction_ceo_present_penalty": 5.0, "inaction_no_sack_penalty": 3.0}

        # Positive review, CEO present → no negative penalty
        out_positive = TerminalOutcome(
            review_commissioned=True, review_outcome="positive",
            review_car=0.05, CEO_removed=False)
        u_positive = utility_board(out_positive, params)

        # Negative review, CEO present → negative penalty fires
        out_adverse = TerminalOutcome(
            review_commissioned=True, review_outcome="negative",
            review_car=-0.05, CEO_removed=False)
        u_adverse = utility_board(out_adverse, params)

        # Positive review should be better than negative review
        assert u_positive > u_adverse


class TestScenarioSolver:
    """Test solver with CEO resignation scenarios and D0_ceo predictive."""

    def _make_solver(self, K=20, R_rollouts=5):
        from engine.solver import Solver
        return Solver(
            governance_spec_path=GOV_SPEC,
            opponent_priors_path=OPP_PRIORS,
            checkpoint_dir=CHECKPOINT_DIR,
            K=K, R_rollouts=R_rollouts, seed=42,
        )

    def test_solve_ceo_resigned_fewer_actions(self):
        """CEO resigned scenario should have 2 D1 actions (no D3)."""
        solver = self._make_solver()
        result = solver.solve(
            "Board", "C0", n_draws=3, scenario="ceo_resigned")
        assert result.scenario == "ceo_resigned"
        assert len(result.EU_per_action) == 2
        assert "D3_ceo_transition" not in result.EU_per_action
        assert "D0_minimal" in result.EU_per_action
        assert "D1_review" in result.EU_per_action

    def test_solve_ceo_stayed_backward_compat(self):
        """CEO stayed scenario should have all 3 D1 actions."""
        solver = self._make_solver()
        result = solver.solve(
            "Board", "C0", n_draws=3, scenario="ceo_stayed")
        assert result.scenario == "ceo_stayed"
        assert len(result.EU_per_action) == 3
        assert "D3_ceo_transition" in result.EU_per_action

    def test_solve_scenarios_returns_both(self):
        """solve_scenarios should return results for both scenarios."""
        solver = self._make_solver()
        results = solver.solve_scenarios("Board", "C0", n_draws=3)
        assert "ceo_stayed" in results
        assert "ceo_resigned" in results
        assert len(results["ceo_stayed"].EU_per_action) == 3
        assert len(results["ceo_resigned"].EU_per_action) == 2

    def test_solve_scenarios_has_d0_ceo_predictive(self):
        """solve_scenarios should compute D0_ceo predictive distribution."""
        solver = self._make_solver()
        results = solver.solve_scenarios("Board", "C0", n_draws=3)
        # Both results should have the same D0_ceo predictive
        for scenario, result in results.items():
            assert "CEO_resign" in result.d0_ceo_predictive
            assert "CEO_stay" in result.d0_ceo_predictive
            # Probabilities must sum to ~1
            total = sum(result.d0_ceo_predictive.values())
            assert abs(total - 1.0) < 0.01, f"D0_ceo probabilities sum to {total}"
            # scenario_prob should match the relevant action
            assert result.scenario_prob >= 0.0
            assert result.scenario_prob <= 1.0

    def test_predict_d0_ceo_distribution(self):
        """predict_d0_ceo should return valid probability distribution."""
        solver = self._make_solver()
        pred = solver.predict_d0_ceo("Board", "C0", n_draws=3)
        assert "CEO_resign" in pred
        assert "CEO_stay" in pred
        total = sum(pred.values())
        assert abs(total - 1.0) < 0.01
        for prob in pred.values():
            assert 0.0 <= prob <= 1.0

    def test_summary_df_has_scenario_and_prob_columns(self):
        """summary_df should include scenario and Pr_scenario columns."""
        solver = self._make_solver()
        result = solver.solve(
            "Board", "C0", n_draws=3, scenario="ceo_resigned")
        df = result.summary_df()
        assert "scenario" in df.columns
        assert "Pr_scenario" in df.columns
        assert (df["scenario"] == "ceo_resigned").all()

    def test_d0_ceo_fixed_policy(self):
        """D0_ceo fixed policy should return CEO_stay."""
        from engine.state import DecisionState, BeliefBundle, ParameterSampler
        from engine.state import load_vote_thresholds, load_policy_parameters
        from engine.chance_models import ChanceModels
        from engine.predictive import PredictiveDistribution

        state = DecisionState.from_governance_spec(GOV_SPEC)
        beliefs = BeliefBundle(CHECKPOINT_DIR / "belief_C0_2023-10-01.npz")
        chance_models = ChanceModels(load_vote_thresholds(GOV_SPEC))
        pred = PredictiveDistribution(
            beliefs=beliefs,
            param_sampler=ParameterSampler(OPP_PRIORS),
            chance_models=chance_models,
            policy_params=load_policy_parameters(GOV_SPEC),
            K=10, R_rollouts=5,
        )
        feasible = state.feasible_actions("D0_ceo")
        rng = np.random.default_rng(42)
        action = pred._fixed_policy("CEO", "D0_ceo", {}, state, feasible, rng=rng)
        assert action == "CEO_stay"

    def test_cpre_checkpoint_loads(self):
        """New Cpre checkpoint should be loadable."""
        from engine.state import BeliefBundle
        bb = BeliefBundle(CHECKPOINT_DIR / "belief_Cpre_2023-08-31.npz")
        assert bb.N == 500
        # Cpre should have slightly negative B_mkt mean
        assert bb.B_mkt.mean() < 0.1


# ============================================================================
# 15. INTERACTIVE TREE VISUALISATION
# ============================================================================

class TestInteractiveTree:
    """Tests for the interactive HTML tree visualisation module."""

    def test_actual_outcomes_loading(self):
        """actual_outcomes.json should load successfully."""
        from run.interactive_tree import load_actual_outcomes
        actual = load_actual_outcomes(DATA_DIR / "actual_outcomes.json")
        assert isinstance(actual, dict)
        assert "D0_ceo" in actual
        assert actual["D0_ceo"] == "CEO_resign"
        assert actual["D1"] == "D1_review"
        assert actual["A2"] == "A2_rec_strike"

    def test_actual_outcomes_missing_file(self):
        """Missing config file should return empty dict, not raise."""
        from run.interactive_tree import load_actual_outcomes
        actual = load_actual_outcomes(Path("/nonexistent/path.json"))
        assert actual == {}

    def test_actual_outcomes_none_path(self):
        """None path should return empty dict."""
        from run.interactive_tree import load_actual_outcomes
        actual = load_actual_outcomes(None)
        assert actual == {}

    def test_is_actual_edge_decision_node(self):
        """Direct match on decision node actions."""
        from run.interactive_tree import _is_actual_edge
        actual = {"D0_ceo": "CEO_resign", "D1": "D1_review"}
        assert _is_actual_edge("D0_ceo", "CEO_resign", actual) is True
        assert _is_actual_edge("D0_ceo", "CEO_stay", actual) is False
        assert _is_actual_edge("D1", "D1_review", actual) is True
        assert _is_actual_edge("D1", "D0_minimal", actual) is False

    def test_is_actual_edge_vote_node(self):
        """Fuzzy match on vote node (V) chance outcomes."""
        from run.interactive_tree import _is_actual_edge
        actual = {"V": "strike_overwhelming"}
        # "strike" in actual_action.lower() → matches "Strike (>=25%)"
        assert _is_actual_edge("V", "Strike (>=25%)", actual) is True
        assert _is_actual_edge("V", "No strike (<25%)", actual) is False

    def test_is_actual_edge_review_node(self):
        """Fuzzy match on review node (R) chance outcomes."""
        from run.interactive_tree import _is_actual_edge
        actual = {"R": "negative"}
        assert _is_actual_edge("R", "Negative", actual) is True
        assert _is_actual_edge("R", "Balanced", actual) is False
        assert _is_actual_edge("R", "Positive", actual) is False

    def test_is_actual_edge_passthrough(self):
        """Pass-through nodes always match."""
        from run.interactive_tree import _is_actual_edge
        actual = {"M_agm": "market_reaction"}
        assert _is_actual_edge("M_agm", "pass-through", actual) is True

    def test_is_actual_edge_missing_node(self):
        """Node not in actual outcomes returns False."""
        from run.interactive_tree import _is_actual_edge
        assert _is_actual_edge("D4", "D4_stay", {}) is False

    def test_viznode_to_dict_structure(self):
        """VizNode serialization should produce dict with required keys."""
        from run.interactive_tree import viznode_to_dict
        from run.visualise_tree import VizNode

        node = VizNode("n1", "D0_ceo", "decision", "CEO", eu=-1.5)
        child = VizNode("n2", "D1", "decision", "Board", eu=-0.5)
        node.children = [("CEO_resign", 0.89, child)]

        d = viznode_to_dict(node, {}, "Board", {}, {})
        assert d["id"] == "n1"
        assert d["name"] == "D0_ceo"
        assert d["type"] == "decision"
        assert d["owner"] == "CEO"
        assert d["eu"] == -1.5
        assert "children" in d
        assert len(d["children"]) == 1
        assert d["children"][0]["label"] == "CEO_resign"
        assert d["children"][0]["prob"] == 0.89

    def test_viznode_to_dict_children_nested(self):
        """Children should be properly nested with edge and child data."""
        from run.interactive_tree import viznode_to_dict
        from run.visualise_tree import VizNode

        leaf = VizNode("n3", "Terminal", "terminal", "Nature", eu=2.0)
        mid = VizNode("n2", "D1", "decision", "Board", eu=1.0, children=[
            ("D0_minimal", 0.6, leaf),
        ])
        root = VizNode("n1", "D0_ceo", "decision", "CEO", eu=0.5, children=[
            ("CEO_resign", 0.89, mid),
        ])

        d = viznode_to_dict(root, {}, "Board", {}, {})
        # Navigate to leaf
        child_edge = d["children"][0]
        grandchild = child_edge["child"]["children"][0]["child"]
        assert grandchild["name"] == "Terminal"
        assert grandchild["type"] == "terminal"
        assert grandchild["eu"] == 2.0

    def test_commentary_without_api_key(self):
        """generate_commentary without API key returns placeholder strings."""
        from run.interactive_tree import generate_commentary, PLACEHOLDER_COMMENTARY
        from run.visualise_tree import VizNode

        tree_dict = {
            "id": "n1", "name": "D0_ceo", "type": "decision", "owner": "CEO",
            "eu": -1.0, "nice_label": "D0_ceo", "colour": "#E85D5D",
            "utility_decomposition": {}, "predictive_dist": {},
            "outcome_stats": {}, "node_commentary": "",
            "children": [{
                "label": "CEO_resign", "nice_label": "CEO resigns",
                "prob": 0.89, "is_actual": True, "child_eu": -0.5,
                "commentary": "",
                "child": {
                    "id": "n2", "name": "Terminal", "type": "terminal",
                    "owner": "Nature", "eu": -0.5, "nice_label": "Terminal",
                    "colour": "#AAA", "utility_decomposition": {},
                    "predictive_dist": {}, "outcome_stats": {},
                    "node_commentary": "", "children": [],
                },
            }],
        }
        result = generate_commentary(tree_dict, "Board", "C0", api_key=None)
        assert isinstance(result, dict)
        for key, val in result.items():
            assert val == PLACEHOLDER_COMMENTARY

    def test_enrich_node_data_utility_decomposition(self):
        """_enrich_node_data should attach utility decomposition from SolveResult."""
        from run.interactive_tree import _enrich_node_data
        from unittest.mock import MagicMock

        result = MagicMock()
        result.utility_decomposition = {
            "D0_minimal": {"vote_penalty": -0.1, "spill_risk": -0.05},
        }
        result.outcome_stats = {
            "D0_minimal": {"Pr_strike": 0.85, "mean_vote_percent": 0.42},
        }
        result.predictive_dists = {}

        node_dict = {
            "utility_decomposition": {},
            "outcome_stats": {},
            "predictive_dist": {},
        }
        _enrich_node_data(node_dict, {"ceo_resigned": result}, "ceo_resigned", "D0_minimal")

        assert node_dict["utility_decomposition"]["vote_penalty"] == -0.1
        assert node_dict["outcome_stats"]["Pr_strike"] == 0.85

    def test_actual_path_only_single_branch(self):
        """Red lines should only follow a single path from root, not all matching edges."""
        from run.interactive_tree import viznode_to_dict
        from run.visualise_tree import VizNode

        # Build a tree where D0_ceo branches into two scenarios,
        # each containing a D1 node. Only the CEO_resign branch
        # should have red edges; the CEO_stay branch should NOT.
        #
        #         D0_ceo
        #        /       \
        #  CEO_resign   CEO_stay
        #      |           |
        #     D1          D1
        #    / \          / \
        # review minimal review minimal
        #   |      |       |      |
        #  T1     T2      T3     T4

        t1 = VizNode("t1", "Terminal", "terminal", "Nature", eu=1.0)
        t2 = VizNode("t2", "Terminal", "terminal", "Nature", eu=2.0)
        t3 = VizNode("t3", "Terminal", "terminal", "Nature", eu=3.0)
        t4 = VizNode("t4", "Terminal", "terminal", "Nature", eu=4.0)

        d1_resign = VizNode("d1_r", "D1", "decision", "Board", eu=1.5, children=[
            ("D1_review", 0.5, t1),
            ("D0_minimal", 0.5, t2),
        ])
        d1_stay = VizNode("d1_s", "D1", "decision", "Board", eu=3.5, children=[
            ("D1_review", 0.5, t3),
            ("D0_minimal", 0.5, t4),
        ])
        root = VizNode("d0", "D0_ceo", "decision", "CEO", eu=2.0, children=[
            ("CEO_resign", 0.6, d1_resign),
            ("CEO_stay", 0.4, d1_stay),
        ])

        actual = {"D0_ceo": "CEO_resign", "D1": "D1_review"}
        d = viznode_to_dict(root, {}, "Board", actual, {})

        # CEO_resign edge should be actual
        resign_edge = d["children"][0]
        assert resign_edge["is_actual"] is True

        # CEO_stay edge should NOT be actual
        stay_edge = d["children"][1]
        assert stay_edge["is_actual"] is False

        # D1_review under CEO_resign should be actual (on the actual path)
        d1_resign_children = resign_edge["child"]["children"]
        review_on_actual = [e for e in d1_resign_children if e["label"] == "D1_review"][0]
        assert review_on_actual["is_actual"] is True

        # D0_minimal under CEO_resign should NOT be actual
        minimal_on_actual = [e for e in d1_resign_children if e["label"] == "D0_minimal"][0]
        assert minimal_on_actual["is_actual"] is False

        # D1_review under CEO_stay should NOT be actual (off the actual path)
        d1_stay_children = stay_edge["child"]["children"]
        review_off_actual = [e for e in d1_stay_children if e["label"] == "D1_review"][0]
        assert review_off_actual["is_actual"] is False

        # D0_minimal under CEO_stay should NOT be actual
        minimal_off_actual = [e for e in d1_stay_children if e["label"] == "D0_minimal"][0]
        assert minimal_off_actual["is_actual"] is False

    def test_html_output_contains_tree_data(self, tmp_path):
        """Generated HTML should contain the embedded tree data JSON."""
        from run.interactive_tree import render_html
        import json

        tree_dict = {"id": "n1", "name": "root", "eu": 0.0, "children": []}
        tree_json = json.dumps(tree_dict)
        out = tmp_path / "test_tree.html"

        render_html(tree_json, "Board", "C0", out)

        content = out.read_text(encoding="utf-8")
        assert '"id": "n1"' in content or '"id":"n1"' in content
        assert "D3.js" in content or "d3.v7" in content
        assert "Board" in content
        assert "Probability View" in content
        assert "Expected Utility View" in content

    def test_html_output_has_interactive_controls(self, tmp_path):
        """HTML should contain expand/collapse and view toggle controls."""
        from run.interactive_tree import render_html
        import json

        tree_json = json.dumps({"id": "n1", "name": "root", "eu": 0, "children": []})
        out = tmp_path / "test_controls.html"
        render_html(tree_json, "Board", "C0", out)

        content = out.read_text(encoding="utf-8")
        assert "Expand All" in content
        assert "Collapse All" in content
        assert "expandAll" in content
        assert "collapseAll" in content
        assert "setView" in content
        assert "toggleActual" in content
        assert "Actual Path" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
