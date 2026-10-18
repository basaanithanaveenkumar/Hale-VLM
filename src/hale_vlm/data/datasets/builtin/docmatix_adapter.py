"""Registry adapter for the ``docmatix`` dataset."""

from __future__ import annotations

from hale_vlm.data.adapters.base import VLMDataAdapter
from hale_vlm.data.registry import register_dataset
from hale_vlm.data.types import DatasetSpec, Modality, TrainingStage, VisionCategory


@register_dataset("docmatix")
class DocmatixAdapter(VLMDataAdapter):
    """Registry adapter for ``docmatix``.

    Source: Hub dataset ``HuggingFaceM4/Docmatix``.

    Document understanding and OCR-heavy visual QA.

    Pattern: Registry entry; reading and normalisation come from the base adapter.
    """

    spec = DatasetSpec(
        name="docmatix",
        hf_path="HuggingFaceM4/Docmatix",
        modality=Modality.IMAGE,
        stage=TrainingStage.VISION,
        category=VisionCategory.OCR_DOCUMENTS.value,
        description="Document understanding and OCR-heavy visual QA.",
        paper_reference="SmolVLM §4.1 vision stage (OCR & Documents ~48%)",
        image_fields=("images", "image"),
        text_fields=("text", "conversations"),
        conversation_field="conversations",
    )
