"""Registry adapter for the ``seed-bench`` dataset."""

from __future__ import annotations

from hale_vlm.data.adapters.base import VLMDataAdapter
from hale_vlm.data.phase_registry import register_phase_dataset
from hale_vlm.data.types import DatasetSpec, Modality, TrainingPhase, TrainingStage


@register_phase_dataset("seed-bench")
class SEEDBenchAdapter(VLMDataAdapter):
    """Registry adapter for ``seed-bench``.

    Source: Hub dataset ``AILab-CVC/SEED-Bench``.

    SEED-Bench — 19K MCQ across 12 evaluation dimensions.

    Pattern: Registry entry; reading and normalisation come from the base adapter.
    """

    spec = DatasetSpec(
        name="seed-bench",
        hf_path="AILab-CVC/SEED-Bench",
        modality=Modality.IMAGE,
        stage=TrainingStage.TEXT_SFT,
        phase=TrainingPhase.POST_TRAIN,
        description="SEED-Bench — 19K MCQ across 12 evaluation dimensions.",
        paper_reference="Li et al. (2023) SEED-Bench",
        image_fields=("image",),
        text_fields=("question", "choice_a", "choice_b", "choice_c", "choice_d"),
        conversation_field=None,
        trust_remote_code=True,
    )
