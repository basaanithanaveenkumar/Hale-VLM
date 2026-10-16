"""FineWebEduAdapter: FineWebEduAdapter."""

from __future__ import annotations

from hale_vlm.data.adapters.base import VLMDataAdapter
from hale_vlm.data.registry import register_dataset
from hale_vlm.data.types import DatasetSpec, Modality, TrainingStage


@register_dataset("fineweb-edu")
class FineWebEduAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="fineweb-edu",
        hf_path="HuggingFaceFW/fineweb-edu",
        modality=Modality.TEXT,
        stage=TrainingStage.CONTEXT,
        description="FineWeb-Edu short-context source for context extension.",
        paper_reference="Penedo et al. (2024); SmolVLM §2.2",
        text_fields=("text", "content"),
        conversation_field=None,
        image_fields=(),
        video_fields=(),
    )
