"""Registry adapter for the ``textvqa`` dataset."""

from __future__ import annotations

from hale_vlm.data.adapters.base import VLMDataAdapter
from hale_vlm.data.phase_registry import register_phase_dataset
from hale_vlm.data.types import DatasetSpec, Modality, TrainingPhase, TrainingStage


@register_phase_dataset("textvqa")
class TextVQAAdapter(VLMDataAdapter):
    """Registry adapter for ``textvqa``.

    Source: Hub dataset ``Howard-hou/TextVQA_0.5.1``.

    TextVQA — 45K VQA questions on text in natural images.

    Pattern: Registry entry; reading and normalisation come from the base adapter.
    """

    spec = DatasetSpec(
        name="textvqa",
        hf_path="Howard-hou/TextVQA_0.5.1",
        modality=Modality.IMAGE,
        stage=TrainingStage.TEXT_SFT,
        phase=TrainingPhase.POST_TRAIN,
        description="TextVQA — 45K VQA questions on text in natural images.",
        paper_reference="Singh et al. (2019) TextVQA",
        image_fields=("image",),
        text_fields=("question", "answers"),
        conversation_field=None,
    )
