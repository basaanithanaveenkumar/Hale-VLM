"""SmolLM2MathAdapter: SmolLM2MathAdapter."""

from __future__ import annotations

from hale_vlm.data.adapters.base import VLMDataAdapter
from hale_vlm.data.registry import register_dataset
from hale_vlm.data.types import DatasetSpec, Modality, TrainingStage


@register_dataset("smollm2-math")
class SmolLM2MathAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="smollm2-math",
        hf_path="HuggingFaceTB/smollm-corpus",
        config_name="math",
        modality=Modality.TEXT,
        stage=TrainingStage.CONTEXT,
        description="SmolLM2 math subset used during context extension.",
        paper_reference="Allal et al. (2025); SmolVLM §2.2",
        text_fields=("text", "content"),
        conversation_field=None,
        image_fields=(),
        video_fields=(),
    )
