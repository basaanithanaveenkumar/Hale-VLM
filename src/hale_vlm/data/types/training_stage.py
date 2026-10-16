"""TrainingStage: SmolVLM paper training stages."""

from __future__ import annotations

from enum import StrEnum


class TrainingStage(StrEnum):
    """SmolVLM paper training stages."""

    VISION = "vision"
    VIDEO = "video"
    CONTEXT = "context"
    TEXT_SFT = "text_sft"
    REJECTED = "rejected"
