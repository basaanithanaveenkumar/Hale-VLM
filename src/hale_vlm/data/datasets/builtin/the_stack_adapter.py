"""Registry adapter for the ``the-stack`` dataset."""

from __future__ import annotations

from hale_vlm.data.adapters.base import VLMDataAdapter
from hale_vlm.data.registry import register_dataset
from hale_vlm.data.types import DatasetSpec, Modality, TrainingStage


@register_dataset("the-stack")
class TheStackAdapter(VLMDataAdapter):
    """Registry adapter for ``the-stack``.

    Source: Hub dataset ``bigcode/the-stack``.

    The Stack code corpus for long-context fine-tuning.

    Pattern: Registry entry; reading and normalisation come from the base adapter.
    """

    spec = DatasetSpec(
        name="the-stack",
        hf_path="bigcode/the-stack",
        modality=Modality.TEXT,
        stage=TrainingStage.CONTEXT,
        description="The Stack code corpus for long-context fine-tuning.",
        paper_reference="Kocetkov et al. (2022); SmolVLM §2.2",
        text_fields=("content", "text"),
        conversation_field=None,
        image_fields=(),
        video_fields=(),
    )
