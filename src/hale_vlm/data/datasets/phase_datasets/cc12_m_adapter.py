"""Registry adapter for the ``cc12m`` dataset."""

from __future__ import annotations

from hale_vlm.data.adapters.base import VLMDataAdapter
from hale_vlm.data.phase_registry import register_phase_dataset
from hale_vlm.data.types import DatasetSpec, Modality, TrainingPhase, TrainingStage


@register_phase_dataset("cc12m")
class CC12MAdapter(VLMDataAdapter):
    """Registry adapter for ``cc12m``.

    Source: Hub dataset ``pixparse/cc12m-wds``.

    Conceptual Captions 12M — ~12M image-text pairs.

    Pattern: Registry entry; reading and normalisation come from the base adapter.
    """

    spec = DatasetSpec(
        name="cc12m",
        hf_path="pixparse/cc12m-wds",
        modality=Modality.IMAGE,
        stage=TrainingStage.VISION,
        phase=TrainingPhase.PRETRAIN,
        description="Conceptual Captions 12M — ~12M image-text pairs.",
        paper_reference="Changpinyo et al. (2021) CC12M",
        image_fields=("jpg", "image"),
        text_fields=("txt", "caption"),
        conversation_field=None,
    )
