"""
Mode configurations for the ARA engine.

Modes define which actor is focal (maximising) and how opponents are modelled.
The mode determines at each node whether the focal actor chooses optimally (max),
opponents are modelled via predictive distributions (ARA), or fixed policies are used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModeConfig:
    """Configuration for a single analysis mode."""
    name: str
    focal_actor: str
    # Map opponent -> modelling approach ("ARA" or "Policy")
    opponent_models: dict[str, str] = field(default_factory=dict)
    # Level of opponent modelling (1 or 2)
    level: int = 1
    # Strategic counterpart for Level-2 (opponent who models another strategically)
    strategic_counterparts: dict[str, str] = field(default_factory=dict)

    def is_focal(self, actor: str) -> bool:
        """Check if actor is the focal (maximising) actor."""
        return actor == self.focal_actor

    def get_opponent_model_type(self, actor: str) -> str:
        """Get modelling approach for an opponent actor."""
        if actor == self.focal_actor:
            return "focal"
        return self.opponent_models.get(actor, "Policy")

    def get_strategic_counterpart(self, opponent: str) -> Optional[str]:
        """Get which actor this opponent models strategically (Level-2)."""
        if self.level < 2:
            return None
        return self.strategic_counterparts.get(opponent)


# Pre-defined mode configurations
MODE_BOARD = ModeConfig(
    name="Board Mode",
    focal_actor="Board",
    opponent_models={"ASA": "ARA", "CEO": "ARA"},
    level=1,
)

MODE_ASA = ModeConfig(
    name="ASA Mode",
    focal_actor="ASA",
    opponent_models={"Board": "ARA", "CEO": "ARA"},
    level=1,
)

MODE_BOARD_L2 = ModeConfig(
    name="Board Mode Level-2",
    focal_actor="Board",
    opponent_models={"ASA": "ARA", "CEO": "ARA"},
    level=2,
    # ASA models Board strategically; CEO models Board strategically
    strategic_counterparts={"ASA": "Board", "CEO": "Board"},
)

MODE_ASA_L2 = ModeConfig(
    name="ASA Mode Level-2",
    focal_actor="ASA",
    opponent_models={"Board": "ARA", "CEO": "ARA"},
    level=2,
    # Board models ASA strategically; CEO models Board strategically
    strategic_counterparts={"Board": "ASA", "CEO": "Board"},
)

# Special mode: ASA perspective with Board using fixed policy
MODE_ASA_POLICY_BOARD = ModeConfig(
    name="ASA Mode (Board=Policy)",
    focal_actor="ASA",
    opponent_models={"Board": "Policy", "CEO": "ARA"},
    level=1,
)


# Special mode: Board perspective with ASA using empirical fixed policy
MODE_BOARD_POLICY_ASA = ModeConfig(
    name="Board Mode (ASA=Policy)",
    focal_actor="Board",
    opponent_models={"ASA": "Policy", "CEO": "ARA"},
    level=1,
)

AVAILABLE_MODES = {
    "board": MODE_BOARD,
    "asa": MODE_ASA,
    "board_l2": MODE_BOARD_L2,
    "asa_l2": MODE_ASA_L2,
    "asa_policy_board": MODE_ASA_POLICY_BOARD,
    "board_policy_asa": MODE_BOARD_POLICY_ASA,
}
