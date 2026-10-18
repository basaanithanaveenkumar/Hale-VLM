"""Registry adapter for the ``llava-onevision-data`` dataset."""

from __future__ import annotations

from hale_vlm.data.adapters.base import VLMDataAdapter
from hale_vlm.data.registry import register_dataset
from hale_vlm.data.types import DatasetSpec, Modality, TrainingStage, VisionCategory


@register_dataset("llava-onevision-data")
class LLaVAOneVisionDataAdapter(VLMDataAdapter):
    """Registry adapter for ``llava-onevision-data``.

    Source: Hub dataset ``lmms-lab/LLaVA-OneVision-Data``.

    LLaVA-OneVision image + instruction mix.

    Pattern: Registry entry; reading and normalisation come from the base adapter.
    """

    spec = DatasetSpec(
        name="llava-onevision-data",
        hf_path="lmms-lab/LLaVA-OneVision-Data",
        modality=Modality.IMAGE,
        stage=TrainingStage.VISION,
        category=VisionCategory.CAPTIONING.value,
        description="LLaVA-OneVision image + instruction mix.",
        paper_reference="SmolVLM vision stage + video stage image portion",
        image_fields=("image", "images"),
        text_fields=("text", "caption", "conversations"),
        conversation_field="conversations",
    )
