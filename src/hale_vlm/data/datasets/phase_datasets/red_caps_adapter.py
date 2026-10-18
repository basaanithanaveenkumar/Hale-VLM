"""Registry adapter for the ``redcaps`` dataset."""

from __future__ import annotations

from hale_vlm.data.adapters.base import VLMDataAdapter
from hale_vlm.data.phase_registry import register_phase_dataset
from hale_vlm.data.types import DatasetSpec, Modality, TrainingPhase, TrainingStage


@register_phase_dataset("redcaps")
class RedCapsAdapter(VLMDataAdapter):
    """Registry adapter for ``redcaps``.

    Source: Hub dataset ``red_caps``.

    RedCaps — 12M human-written image-text pairs from Reddit.

    Pattern: Registry entry; reading and normalisation come from the base adapter.
    """

    spec = DatasetSpec(
        name="redcaps",
        hf_path="red_caps",
        modality=Modality.IMAGE,
        stage=TrainingStage.VISION,
        phase=TrainingPhase.PRETRAIN,
        description="RedCaps — 12M human-written image-text pairs from Reddit.",
        paper_reference="Desai et al. (2021) RedCaps",
        image_fields=("image", "image_url"),
        text_fields=("caption", "raw_caption"),
        conversation_field=None,
    )
