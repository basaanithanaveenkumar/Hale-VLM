"""Registry adapter for the ``llava-plus`` dataset."""

from __future__ import annotations

from hale_vlm.data.adapters.base import VLMDataAdapter
from hale_vlm.data.phase_registry import register_phase_dataset
from hale_vlm.data.types import DatasetSpec, Modality, TrainingPhase, TrainingStage


@register_phase_dataset("llava-plus")
class LLaVAPlusAdapter(VLMDataAdapter):
    """Registry adapter for ``llava-plus``.

    Source: Hub dataset ``zhaohengyuan1/llava_plus_stage2``.

    LLaVA-Plus skill-augmented instruction-following data.

    Pattern: Registry entry; reading and normalisation come from the base adapter.
    """

    spec = DatasetSpec(
        name="llava-plus",
        hf_path="zhaohengyuan1/llava_plus_stage2",
        modality=Modality.IMAGE,
        stage=TrainingStage.TEXT_SFT,
        phase=TrainingPhase.POST_TRAIN,
        description="LLaVA-Plus skill-augmented instruction-following data.",
        paper_reference="Liu et al. (2023) LLaVA-Plus",
        image_fields=("image", "images"),
        text_fields=("conversations",),
        conversation_field="conversations",
    )
