"""RobotEmbodiment: Robot platforms referenced in SmolVLA."""

from __future__ import annotations

from enum import StrEnum


class RobotEmbodiment(StrEnum):
    """Robot platforms referenced in SmolVLA."""

    SO100 = "so100"
    SO101 = "so101"
    PANDA = "panda"
    SAWYER = "sawyer"
    MIXED = "mixed"
