"""VideoCategory: Video fine-tuning buckets (Figure 8, SmolVLM paper)."""

from __future__ import annotations

from enum import StrEnum


class VideoCategory(StrEnum):
    """Video fine-tuning buckets (Figure 8, SmolVLM paper)."""

    VISUAL_DESCRIPTION = "visual_description"
    TEMPORAL_UNDERSTANDING = "temporal_understanding"
    NARRATIVE = "narrative"
    MULTI_IMAGE = "multi_image"
    TEXT_SFT = "text_sft"
