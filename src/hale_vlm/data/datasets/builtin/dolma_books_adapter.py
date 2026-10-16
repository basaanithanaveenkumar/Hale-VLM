"""DolmaBooksAdapter: DolmaBooksAdapter."""

from __future__ import annotations

from hale_vlm.data.adapters.base import VLMDataAdapter
from hale_vlm.data.registry import register_dataset
from hale_vlm.data.types import DatasetSpec, Modality, TrainingStage


@register_dataset("dolma-books")
class DolmaBooksAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="dolma-books",
        hf_path="allenai/dolma",
        config_name="books",
        modality=Modality.TEXT,
        stage=TrainingStage.CONTEXT,
        description="Dolma books subset for 2k→16k context extension.",
        paper_reference="Soldaini et al. (2024); SmolVLM §2.2",
        text_fields=("text", "content"),
        conversation_field=None,
        image_fields=(),
        video_fields=(),
    )
