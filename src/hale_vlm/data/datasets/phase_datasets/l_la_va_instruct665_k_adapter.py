"""Registry adapter for the ``llava-instruct-665k`` dataset."""

from __future__ import annotations

from hale_vlm.data.adapters.base import VLMDataAdapter
from hale_vlm.data.phase_registry import register_phase_dataset
from hale_vlm.data.types import DatasetSpec, Modality, TrainingPhase, TrainingStage


@register_phase_dataset("llava-instruct-665k")
class LLaVAInstruct665KAdapter(VLMDataAdapter):
    """Registry adapter for ``llava-instruct-665k``.

    Source: Hub dataset ``liuhaotian/LLaVA-Instruct-150K``.

    LLaVA-1.5 SFT 665K instruction-following conversations.

    Pattern: Registry entry; reading and normalisation come from the base adapter.
    """

    spec = DatasetSpec(
        name="llava-instruct-665k",
        hf_path="liuhaotian/LLaVA-Instruct-150K",
        modality=Modality.IMAGE,
        stage=TrainingStage.TEXT_SFT,
        phase=TrainingPhase.POST_TRAIN,
        description="LLaVA-1.5 SFT 665K instruction-following conversations.",
        paper_reference="Liu et al. (2023) LLaVA-1.5",
        image_fields=("image",),
        text_fields=("conversations",),
        conversation_field="conversations",
    )
