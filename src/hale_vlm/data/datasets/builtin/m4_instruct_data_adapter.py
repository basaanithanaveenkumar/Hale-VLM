"""M4InstructDataAdapter: M4InstructDataAdapter."""

from __future__ import annotations

from hale_vlm.data.adapters.base import VLMDataAdapter
from hale_vlm.data.registry import register_dataset
from hale_vlm.data.types import DatasetSpec, Modality, TrainingStage, VideoCategory


@register_dataset("m4-instruct-data")
class M4InstructDataAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="m4-instruct-data",
        hf_path="lmms-lab/M4-Instruct-Data",
        modality=Modality.MULTI_IMAGE,
        stage=TrainingStage.VIDEO,
        category=VideoCategory.MULTI_IMAGE.value,
        description="M4 multi-image instruction tuning.",
        paper_reference="Liu et al. (2024a); SmolVLM §4.1",
        image_fields=("images", "image"),
        text_fields=("text", "conversations", "caption"),
        conversation_field="conversations",
    )
