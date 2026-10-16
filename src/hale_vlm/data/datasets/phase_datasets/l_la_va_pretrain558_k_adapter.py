"""LLaVAPretrain558KAdapter: LLaVAPretrain558KAdapter."""

from __future__ import annotations

from hale_vlm.data.adapters.base import VLMDataAdapter
from hale_vlm.data.phase_registry import register_phase_dataset
from hale_vlm.data.types import DatasetSpec, Modality, TrainingPhase, TrainingStage


@register_phase_dataset("llava-pretrain-558k")
class LLaVAPretrain558KAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="llava-pretrain-558k",
        hf_path="liuhaotian/LLaVA-Pretrain",
        modality=Modality.IMAGE,
        stage=TrainingStage.VISION,
        phase=TrainingPhase.MID_TRAIN,
        description="LLaVA pretraining 558K BLIP-generated captions from CC3M.",
        paper_reference="Liu et al. (2023) LLaVA-1.5",
        image_fields=("image",),
        text_fields=("caption", "text", "conversations"),
        conversation_field="conversations",
    )
