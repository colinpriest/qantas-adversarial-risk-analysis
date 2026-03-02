"""
Generate governance_spec.xlsx and opponent_priors.xlsx data contracts.

These files define the game structure, utility parameters, and opponent priors
per the v2 specification.
"""

import pandas as pd
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")


def create_governance_spec():
    """Create governance_spec.xlsx with all required sheets."""
    path = os.path.join(OUTPUT_DIR, "governance_spec.xlsx")

    # 1) node_order
    node_order = pd.DataFrame({
        "order_index": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "node_name": ["D1", "A2", "V", "M_agm", "D_rev", "R", "M_rev", "D4", "Terminal"],
        "node_type": ["decision", "decision", "chance", "chance",
                       "decision", "chance", "chance", "decision", "terminal"],
        "owner": ["Board", "ASA", "Nature", "Nature",
                  "Board", "Nature", "Nature", "CEO", "Nature"],
    })

    # 2) action_sets
    action_sets = pd.DataFrame({
        "node_name": [
            # D1 - Board initial governance package
            "D1", "D1", "D1",
            # A2 - ASA mobilisation decision
            "A2", "A2",
            # D_rev - Board review decision (after AGM)
            "D_rev", "D_rev", "D_rev",
            # D4 - CEO response
            "D4", "D4", "D4",
        ],
        "action_name": [
            # D1 actions
            "D0_minimal", "D1_review", "D3_ceo_transition",
            # A2 actions
            "A2_no_strike", "A2_rec_strike",
            # D_rev actions
            "Drev_no_action", "Drev_commission_review", "Drev_sack_ceo",
            # D4 actions
            "D4_stay", "D4_resign", "D4_negotiate_exit",
        ],
        "feasibility_code": [
            # D1 - all always feasible
            "always", "always", "always",
            # A2 - always feasible
            "always", "always",
            # D_rev
            "always", "review_not_commissioned", "CEO_present",
            # D4
            "CEO_present", "CEO_present", "CEO_present",
        ],
        "description": [
            "Do nothing - status quo",
            "Commission independent review with timeline",
            "Sack the CEO",
            "No strike recommendation to shareholders",
            "Recommend shareholders vote against remuneration report",
            "Take no further action post-AGM",
            "Commission independent governance review",
            "Remove CEO immediately",
            "CEO stays in position",
            "CEO resigns voluntarily",
            "CEO negotiates managed exit",
        ],
    })

    # 3) vote_thresholds
    vote_thresholds = pd.DataFrame({
        "threshold_name": ["first_strike", "overwhelming"],
        "value": [0.25, 0.50],
    })

    # 4) utilities_board
    utilities_board = pd.DataFrame({
        "parameter_name": [
            "vote_penalty_weight",
            "review_penalty_weight",
            "implementation_cost_review",
            "implementation_cost_sack",
            "ceo_loss_cost",
            "spill_risk_weight",
            "overwhelming_penalty_weight",
            "reputational_spill_weight",
        ],
        "value": [
            2.0,    # Weight on vote opposition penalty
            1.5,    # Weight on adverse review finding
            0.3,    # Cost of commissioning review
            1.0,    # Cost of sacking CEO
            1.5,    # Cost of losing CEO (disruption)
            2.5,    # Risk weight for board spill
            3.0,    # Extra penalty for >50% opposition
            1.0,    # Reputational cost of spill
        ],
    })

    # 5) utilities_asa
    utilities_asa = pd.DataFrame({
        "parameter_name": [
            "vote_reward_weight",
            "ceo_removal_reward",
            "review_adverse_reward",
            "mobilisation_cost",
            "reputational_gain_weight",
            "overwhelming_reward_weight",
        ],
        "value": [
            2.0,    # Reward for high opposition vote
            3.0,    # Reward for CEO removal
            1.5,    # Reward for adverse review finding
            1.0,    # Cost of running strike campaign
            1.0,    # Reputational gain from successful campaign
            2.0,    # Extra reward for overwhelming vote
        ],
    })

    # 6) utilities_ceo
    utilities_ceo = pd.DataFrame({
        "parameter_name": [
            "job_loss_cost",
            "reputational_cost_weight",
            "resignation_cost",
            "forced_removal_cost",
            "overwhelming_vote_cost_weight",
            "adverse_review_cost_weight",
        ],
        "value": [
            5.0,    # Cost of losing position
            2.0,    # Reputational damage weight
            2.0,    # Cost of voluntary resignation (less than forced)
            4.0,    # Cost of forced removal
            1.5,    # Extra cost from overwhelming opposition
            1.0,    # Cost from adverse review findings
        ],
    })

    # 7) policy_parameters (Level-1 fixed policy defaults)
    policy_parameters = pd.DataFrame({
        "actor": [
            "Board", "Board",
            "CEO", "CEO",
            "ASA",
        ],
        "node_name": [
            "D_rev", "D_rev",
            "D4", "D4",
            "A2",
        ],
        "parameter_name": [
            "review_vote_threshold", "sack_vote_threshold",
            "resign_vote_threshold", "resign_adverse_prob_threshold",
            "mobilise_vote_threshold",
        ],
        "value": [
            0.25, 0.50,
            0.40, 0.60,
            0.20,
        ],
    })

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        node_order.to_excel(writer, sheet_name="node_order", index=False)
        action_sets.to_excel(writer, sheet_name="action_sets", index=False)
        vote_thresholds.to_excel(writer, sheet_name="vote_thresholds", index=False)
        utilities_board.to_excel(writer, sheet_name="utilities_board", index=False)
        utilities_asa.to_excel(writer, sheet_name="utilities_asa", index=False)
        utilities_ceo.to_excel(writer, sheet_name="utilities_ceo", index=False)
        policy_parameters.to_excel(writer, sheet_name="policy_parameters", index=False)

    print(f"Created {path}")
    return path


def create_opponent_priors():
    """Create opponent_priors.xlsx with prior distributions."""
    path = os.path.join(OUTPUT_DIR, "opponent_priors.xlsx")

    priors = pd.DataFrame({
        "perspective_actor": [
            # Board's beliefs about ASA parameters
            "Board", "Board", "Board", "Board",
            # Board's beliefs about CEO parameters
            "Board", "Board", "Board", "Board",
            # ASA's beliefs about Board parameters
            "ASA", "ASA", "ASA", "ASA",
            # ASA's beliefs about CEO parameters
            "ASA", "ASA", "ASA", "ASA",
        ],
        "target_actor": [
            "ASA", "ASA", "ASA", "ASA",
            "CEO", "CEO", "CEO", "CEO",
            "Board", "Board", "Board", "Board",
            "CEO", "CEO", "CEO", "CEO",
        ],
        "parameter_name": [
            # Board -> ASA priors
            "mobilisation_cost", "vote_reward_weight",
            "ceo_removal_reward", "reputational_gain_weight",
            # Board -> CEO priors
            "job_loss_cost", "resignation_cost",
            "forced_removal_cost", "reputational_cost_weight",
            # ASA -> Board priors
            "vote_penalty_weight", "ceo_loss_cost",
            "implementation_cost_review", "spill_risk_weight",
            # ASA -> CEO priors
            "job_loss_cost", "resignation_cost",
            "forced_removal_cost", "reputational_cost_weight",
        ],
        "distribution": [
            # Board -> ASA
            "lognormal", "normal", "normal", "normal",
            # Board -> CEO
            "lognormal", "normal", "normal", "normal",
            # ASA -> Board
            "normal", "normal", "lognormal", "normal",
            # ASA -> CEO
            "lognormal", "normal", "normal", "normal",
        ],
        "param1": [
            # Board -> ASA (mu)
            0.0, 2.0, 3.0, 1.0,
            # Board -> CEO (mu)
            1.6, 2.0, 4.0, 2.0,
            # ASA -> Board (mu)
            2.0, 1.5, -1.2, 2.5,
            # ASA -> CEO (mu)
            1.6, 2.0, 4.0, 2.0,
        ],
        "param2": [
            # Board -> ASA (sd)
            0.5, 0.5, 1.0, 0.5,
            # Board -> CEO (sd)
            0.3, 0.5, 1.0, 0.5,
            # ASA -> Board (sd)
            0.5, 0.5, 0.3, 0.5,
            # ASA -> CEO (sd)
            0.3, 0.5, 1.0, 0.5,
        ],
    })

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        priors.to_excel(writer, sheet_name="priors", index=False)

    print(f"Created {path}")
    return path


if __name__ == "__main__":
    create_governance_spec()
    create_opponent_priors()
    print("Data contracts created successfully.")
