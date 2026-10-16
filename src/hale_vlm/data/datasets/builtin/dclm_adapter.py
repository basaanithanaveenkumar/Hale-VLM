"""DCLMAdapter: DCLMAdapter."""

from __future__ import annotations

from hale_vlm.data.adapters.base import VLMDataAdapter
from hale_vlm.data.registry import register_dataset
from hale_vlm.data.types import DatasetSpec, Modality, TrainingStage


@register_dataset("dclm")
class DCLMAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="dclm",
        hf_path="mlfoundations/dclm-baseline-1.0",
        modality=Modality.TEXT,
        stage=TrainingStage.CONTEXT,
        description="DCLM baseline corpus for context extension.",
        paper_reference="Li et al. (2024a); SmolVLM §2.2",
        text_fields=("text", "content"),
        conversation_field=None,
        image_fields=(),
        video_fields=(),
    )
