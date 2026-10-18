"""Registry adapter for the ``the-cauldron`` dataset."""

from __future__ import annotations

from hale_vlm.data.adapters.base import VLMDataAdapter
from hale_vlm.data.registry import register_dataset
from hale_vlm.data.types import DatasetSpec, Modality, TrainingStage, VisionCategory


@register_dataset("the-cauldron")
class TheCauldronAdapter(VLMDataAdapter):
    """Registry adapter for ``the-cauldron``.

    Source: Hub dataset ``HuggingFaceM4/the_cauldron``.

    Idefics/SmolVLM vision mixture: OCR, charts, tables, captioning, VQA.

    Pattern: Registry entry; reading and normalisation come from the base adapter.
    """

    spec = DatasetSpec(
        name="the-cauldron",
        hf_path="HuggingFaceM4/the_cauldron",
        modality=Modality.IMAGE,
        stage=TrainingStage.VISION,
        category=VisionCategory.OCR_DOCUMENTS.value,
        description="Idefics/SmolVLM vision mixture: OCR, charts, tables, captioning, VQA.",
        paper_reference="Laurençon et al. (2024); SmolVLM §4.1 vision stage",
        image_fields=("images", "image"),
        text_fields=("text", "conversations"),
        conversation_field="conversations",
    )
