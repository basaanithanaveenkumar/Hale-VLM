"""Registry adapter for the ``recap-datacomp-1b`` dataset."""

from __future__ import annotations

from hale_vlm.data.adapters.base import VLMDataAdapter
from hale_vlm.data.phase_registry import register_phase_dataset
from hale_vlm.data.types import DatasetSpec, Modality, TrainingPhase, TrainingStage


@register_phase_dataset("recap-datacomp-1b")
class RecapDataComp1BAdapter(VLMDataAdapter):
    """Registry adapter for ``recap-datacomp-1b``.

    Source: Hub dataset ``BAAI/Recap-DataComp-1B``.

    DataComp-1B re-captioned with LLaVA-1.5.

    Pattern: Registry entry; reading and normalisation come from the base adapter.
    """

    spec = DatasetSpec(
        name="recap-datacomp-1b",
        hf_path="BAAI/Recap-DataComp-1B",
        modality=Modality.IMAGE,
        stage=TrainingStage.VISION,
        phase=TrainingPhase.MID_TRAIN,
        description="DataComp-1B re-captioned with LLaVA-1.5.",
        paper_reference="Li et al. (2024) Recap-DataComp",
        image_fields=("image", "url"),
        text_fields=("recaption", "caption", "text"),
        conversation_field=None,
        trust_remote_code=True,
    )
