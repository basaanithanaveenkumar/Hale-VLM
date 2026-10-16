"""MammothDataAdapter: MammothDataAdapter."""

from __future__ import annotations

from hale_vlm.data.adapters.base import VLMDataAdapter
from hale_vlm.data.registry import register_dataset
from hale_vlm.data.types import DatasetSpec, Modality, TrainingStage, VideoCategory


@register_dataset("mammoth")
class MammothDataAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="mammoth",
        hf_path="MAmmoTH-VL/MAmmoTH-VL-Instruct-12M",
        modality=Modality.MULTI_IMAGE,
        stage=TrainingStage.VIDEO,
        category=VideoCategory.MULTI_IMAGE.value,
        description="MAmmoTH-VL multi-image instruction corpus.",
        paper_reference="Guo et al. (2024); SmolVLM §4.1",
        image_fields=("images", "image"),
        text_fields=("text", "conversations", "caption"),
        conversation_field="conversations",
    )
