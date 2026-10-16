"""VLAStage: SmolVLA paper dataset stages."""

from __future__ import annotations

from enum import StrEnum


class VLAStage(StrEnum):
    """SmolVLA paper dataset stages."""

    COMMUNITY = "community"
    SIMULATION = "simulation"
    REAL_WORLD = "real_world"
