"""CC3MAdapter: CC3MAdapter."""

from __future__ import annotations

from hale_vlm.data.adapters.base import VLMDataAdapter
from hale_vlm.data.phase_registry import register_phase_dataset
from hale_vlm.data.types import DatasetSpec, Modality, TrainingPhase, TrainingStage


@register_phase_dataset("cc3m")
class CC3MAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="cc3m",
        hf_path="pixparse/cc3m-wds",
        modality=Modality.IMAGE,
        stage=TrainingStage.VISION,
        phase=TrainingPhase.PRETRAIN,
        description="Conceptual Captions 3M — 3.3M image-alt-text pairs.",
        paper_reference="Sharma et al. (2018) CC3M",
        image_fields=("jpg", "image"),
        text_fields=("txt", "caption"),
        conversation_field=None,
    )
