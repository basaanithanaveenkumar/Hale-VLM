"""ShareGPT4VPTAdapter: ShareGPT4VPTAdapter."""

from __future__ import annotations

from hale_vlm.data.adapters.base import VLMDataAdapter
from hale_vlm.data.phase_registry import register_phase_dataset
from hale_vlm.data.types import DatasetSpec, Modality, TrainingPhase, TrainingStage


@register_phase_dataset("sharegpt4v-pt")
class ShareGPT4VPTAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="sharegpt4v-pt",
        hf_path="Lin-Chen/ShareGPT4V",
        modality=Modality.IMAGE,
        stage=TrainingStage.VISION,
        phase=TrainingPhase.MID_TRAIN,
        description="ShareGPT4V 1.2M GPT-4V captions.",
        paper_reference="Chen et al. (2023) ShareGPT4V",
        image_fields=("image", "images"),
        text_fields=("caption", "conversations"),
        conversation_field="conversations",
    )
