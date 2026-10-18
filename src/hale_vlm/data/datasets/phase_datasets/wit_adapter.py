"""Registry adapter for the ``wit`` dataset."""

from __future__ import annotations

from hale_vlm.data.adapters.base import VLMDataAdapter
from hale_vlm.data.phase_registry import register_phase_dataset
from hale_vlm.data.types import DatasetSpec, Modality, TrainingPhase, TrainingStage


@register_phase_dataset("wit")
class WITAdapter(VLMDataAdapter):
    """Registry adapter for ``wit``.

    Source: Hub dataset ``google/wit``.

    Wikipedia-based Image Text (WIT) — 37.6M pairs.

    Pattern: Registry entry; reading and normalisation come from the base adapter.
    """

    spec = DatasetSpec(
        name="wit",
        hf_path="google/wit",
        modality=Modality.IMAGE,
        stage=TrainingStage.VISION,
        phase=TrainingPhase.PRETRAIN,
        description="Wikipedia-based Image Text (WIT) — 37.6M pairs.",
        paper_reference="Srinivasan et al. (2021) WIT",
        image_fields=("image_url", "image"),
        text_fields=("caption_reference_description", "caption_attribution_description"),
        conversation_field=None,
    )
