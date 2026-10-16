"""LAIONAestheticsAdapter: LAIONAestheticsAdapter."""

from __future__ import annotations

from hale_vlm.data.adapters.base import VLMDataAdapter
from hale_vlm.data.phase_registry import register_phase_dataset
from hale_vlm.data.types import DatasetSpec, Modality, TrainingPhase, TrainingStage


@register_phase_dataset("laion-aesthetics-v2-5plus")
class LAIONAestheticsAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="laion-aesthetics-v2-5plus",
        hf_path="laion/laion2B-en-aesthetic",
        modality=Modality.IMAGE,
        stage=TrainingStage.VISION,
        phase=TrainingPhase.PRETRAIN,
        description="LAION-2B-en aesthetic subset (≥5.0); ~600M image-text pairs.",
        paper_reference="Schuhmann et al. (2022) LAION-5B",
        image_fields=("image", "url"),
        text_fields=("caption", "text"),
        conversation_field=None,
    )
