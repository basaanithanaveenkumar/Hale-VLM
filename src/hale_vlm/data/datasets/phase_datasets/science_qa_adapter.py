"""Registry adapter for the ``scienceqa`` dataset."""

from __future__ import annotations

from hale_vlm.data.adapters.base import VLMDataAdapter
from hale_vlm.data.phase_registry import register_phase_dataset
from hale_vlm.data.types import DatasetSpec, Modality, TrainingPhase, TrainingStage


@register_phase_dataset("scienceqa")
class ScienceQAAdapter(VLMDataAdapter):
    """Registry adapter for ``scienceqa``.

    Source: Hub dataset ``derek-thomas/ScienceQA``.

    ScienceQA — 21K multimodal science MCQ.

    Pattern: Registry entry; reading and normalisation come from the base adapter.
    """

    spec = DatasetSpec(
        name="scienceqa",
        hf_path="derek-thomas/ScienceQA",
        modality=Modality.IMAGE,
        stage=TrainingStage.TEXT_SFT,
        phase=TrainingPhase.POST_TRAIN,
        description="ScienceQA — 21K multimodal science MCQ.",
        paper_reference="Lu et al. (2022) ScienceQA",
        image_fields=("image",),
        text_fields=("question", "choices", "solution"),
        conversation_field=None,
    )
