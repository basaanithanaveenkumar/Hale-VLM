"""Registry adapter for the ``vista-400k`` dataset."""

from __future__ import annotations

from hale_vlm.data.adapters.video import VideoDatasetAdapter
from hale_vlm.data.registry import register_dataset
from hale_vlm.data.types import DatasetSpec, Modality, TrainingStage, VideoCategory


@register_dataset("vista-400k")
class Vista400KAdapter(VideoDatasetAdapter):
    """Registry adapter for ``vista-400k``.

    Source: Hub dataset ``TIGER-Lab/VISTA-400K``.

    VISTA-400K temporal video understanding.

    Pattern: Registry entry; reading and normalisation come from the base adapter.
    """

    spec = DatasetSpec(
        name="vista-400k",
        hf_path="TIGER-Lab/VISTA-400K",
        modality=Modality.VIDEO,
        stage=TrainingStage.VIDEO,
        category=VideoCategory.TEMPORAL_UNDERSTANDING.value,
        description="VISTA-400K temporal video understanding.",
        paper_reference="Ren et al. (2024); SmolVLM §4.1",
        video_fields=("video", "video_path"),
        text_fields=("text", "caption", "conversations", "question", "answer"),
        conversation_field="conversations",
    )
