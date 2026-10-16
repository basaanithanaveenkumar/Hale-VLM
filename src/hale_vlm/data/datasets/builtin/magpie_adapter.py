"""MagpieAdapter: MagpieAdapter."""

from __future__ import annotations

from hale_vlm.data.adapters.base import VLMDataAdapter
from hale_vlm.data.registry import register_dataset
from hale_vlm.data.types import DatasetSpec, Modality, TrainingStage, VideoCategory


@register_dataset("magpie")
class MagpieAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="magpie",
        hf_path="Magpie-Align/Magpie-Pro-300K-Filtered",
        modality=Modality.TEXT,
        stage=TrainingStage.TEXT_SFT,
        category=VideoCategory.TEXT_SFT.value,
        description="Magpie alignment synthesis for the 14% text portion in video stage.",
        paper_reference="Xu et al. (2024); SmolVLM §4.1",
        text_fields=("text", "instruction", "response", "conversations"),
        conversation_field="conversations",
        image_fields=(),
        video_fields=(),
    )
