"""MathWritingAdapter: MathWritingAdapter."""

from __future__ import annotations

from hale_vlm.data.adapters.base import VLMDataAdapter
from hale_vlm.data.registry import register_dataset
from hale_vlm.data.types import DatasetSpec, Modality, TrainingStage, VisionCategory


@register_dataset("mathwriting")
class MathWritingAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="mathwriting",
        hf_path="google/mathwriting",
        modality=Modality.IMAGE,
        stage=TrainingStage.VISION,
        category=VisionCategory.MATH_HANDWRITING.value,
        description="Handwritten mathematical expression recognition.",
        paper_reference="Gervais et al. (2024); added in SmolVLM §4.1",
        image_fields=("image",),
        text_fields=("text", "latex", "label"),
        conversation_field=None,
    )
