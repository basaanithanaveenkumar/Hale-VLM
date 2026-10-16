"""VisionCategory: Vision-stage mixture buckets (Figure 8, SmolVLM paper)."""

from __future__ import annotations

from enum import StrEnum


class VisionCategory(StrEnum):
    """Vision-stage mixture buckets (Figure 8, SmolVLM paper)."""

    OCR_DOCUMENTS = "ocr_documents"
    CAPTIONING = "captioning"
    CHART_UNDERSTANDING = "chart_understanding"
    REASONING_LOGIC = "reasoning_logic"
    TABLE_UNDERSTANDING = "table_understanding"
    VISUAL_QA = "visual_qa"
    MULTI_IMAGE = "multi_image"
    GENERAL_KNOWLEDGE = "general_knowledge"
    MATH_HANDWRITING = "math_handwriting"
