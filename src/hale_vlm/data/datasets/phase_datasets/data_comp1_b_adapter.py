"""DataComp1BAdapter: DataComp1BAdapter."""

from __future__ import annotations

from hale_vlm.data.adapters.base import VLMDataAdapter
from hale_vlm.data.phase_registry import register_phase_dataset
from hale_vlm.data.types import DatasetSpec, Modality, TrainingPhase, TrainingStage


@register_phase_dataset("datacomp-1b")
class DataComp1BAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="datacomp-1b",
        hf_path="mlfoundations/datacomp_1b",
        modality=Modality.IMAGE,
        stage=TrainingStage.VISION,
        phase=TrainingPhase.PRETRAIN,
        description="DataComp-1B CommonPool — 1.28B CLIP-filtered image-text pairs.",
        paper_reference="Gadre et al. (2023) DataComp",
        image_fields=("image", "url"),
        text_fields=("caption", "text"),
        conversation_field=None,
        trust_remote_code=True,
    )
