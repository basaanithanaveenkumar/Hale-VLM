"""Registry adapter for the ``chartqa`` dataset."""

from __future__ import annotations

from hale_vlm.data.adapters.base import VLMDataAdapter
from hale_vlm.data.phase_registry import register_phase_dataset
from hale_vlm.data.types import DatasetSpec, Modality, TrainingPhase, TrainingStage


@register_phase_dataset("chartqa")
class ChartQAAdapter(VLMDataAdapter):
    """Registry adapter for ``chartqa``.

    Source: Hub dataset ``HuggingFaceM4/ChartQA``.

    ChartQA — 32K QA on charts requiring numerical reasoning.

    Pattern: Registry entry; reading and normalisation come from the base adapter.
    """

    spec = DatasetSpec(
        name="chartqa",
        hf_path="HuggingFaceM4/ChartQA",
        modality=Modality.IMAGE,
        stage=TrainingStage.TEXT_SFT,
        phase=TrainingPhase.POST_TRAIN,
        description="ChartQA — 32K QA on charts requiring numerical reasoning.",
        paper_reference="Masry et al. (2022) ChartQA",
        image_fields=("image",),
        text_fields=("question", "answer"),
        conversation_field=None,
    )
