"""Registry adapter for the ``smoltalk`` dataset."""

from __future__ import annotations

from hale_vlm.data.adapters.base import VLMDataAdapter
from hale_vlm.data.registry import register_dataset
from hale_vlm.data.types import DatasetSpec, Modality, TrainingStage


@register_dataset("smoltalk")
class SmolTalkRejectedAdapter(VLMDataAdapter):
    """Registry adapter for ``smoltalk``.

    Source: Hub dataset ``HuggingFaceTB/smol-smoltalk``.

    REJECTED: reusing LLM-SFT SmolTalk degraded video (-3.7%) and image (-6.5%) performance. Not
    used in SmolVLM final mixture.

    Pattern: Registry entry; reading and normalisation come from the base adapter.
    """

    spec = DatasetSpec(
        name="smoltalk",
        hf_path="HuggingFaceTB/smol-smoltalk",
        modality=Modality.TEXT,
        stage=TrainingStage.REJECTED,
        enabled=False,
        description=(
            "REJECTED: reusing LLM-SFT SmolTalk degraded video (-3.7%) and image (-6.5%) "
            "performance. Not used in SmolVLM final mixture."
        ),
        paper_reference="SmolVLM §3.3, Figure 7",
        text_fields=("text", "messages", "conversations"),
        conversation_field="messages",
        image_fields=(),
        video_fields=(),
    )
