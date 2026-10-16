"""InfographicsVQAAdapter: InfographicsVQAAdapter."""

from __future__ import annotations

from hale_vlm.data.adapters.base import VLMDataAdapter
from hale_vlm.data.phase_registry import register_phase_dataset
from hale_vlm.data.types import DatasetSpec, Modality, TrainingPhase, TrainingStage


@register_phase_dataset("infographics-vqa")
class InfographicsVQAAdapter(VLMDataAdapter):
    spec = DatasetSpec(
        name="infographics-vqa",
        hf_path="lmms-lab/InfoVQA",
        modality=Modality.IMAGE,
        stage=TrainingStage.TEXT_SFT,
        phase=TrainingPhase.POST_TRAIN,
        description="InfographicsVQA — 30K QA on infographic images.",
        paper_reference="Mathew et al. (2022) InfographicsVQA",
        image_fields=("image",),
        text_fields=("question", "answers"),
        conversation_field=None,
    )
