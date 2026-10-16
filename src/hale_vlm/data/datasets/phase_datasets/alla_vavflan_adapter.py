"""AllaVAVFLANAdapter: AllaVAVFLANAdapter."""

from __future__ import annotations

from hale_vlm.data.adapters.base import VLMDataAdapter
from hale_vlm.data.phase_registry import register_phase_dataset
from hale_vlm.data.types import DatasetSpec, Modality, TrainingPhase, TrainingStage


@register_phase_dataset("allava-vflan")
class AllaVAVFLANAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="allava-vflan",
        hf_path="FreedomIntelligence/ALLaVA-4V",
        modality=Modality.IMAGE,
        stage=TrainingStage.VISION,
        phase=TrainingPhase.MID_TRAIN,
        description="ALLaVA VFLAN ~1.5M reason-annotated visual QA.",
        paper_reference="Chen et al. (2024) ALLaVA",
        image_fields=("image", "images"),
        text_fields=("conversations", "caption"),
        conversation_field="conversations",
    )
