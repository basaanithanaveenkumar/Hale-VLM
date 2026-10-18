"""Registry adapter for the ``videostar`` dataset."""

from __future__ import annotations

from hale_vlm.data.adapters.video import VideoDatasetAdapter
from hale_vlm.data.registry import register_dataset
from hale_vlm.data.types import DatasetSpec, Modality, TrainingStage, VideoCategory


@register_dataset("videostar")
class VideoStarAdapter(VideoDatasetAdapter):
    """Registry adapter for ``videostar``.

    Source: Hub dataset ``orrzohar/Video-STaR``.

    Video-STaR self-training video instruction tuning.

    Pattern: Registry entry; reading and normalisation come from the base adapter.
    """

    spec = DatasetSpec(
        name="videostar",
        hf_path="orrzohar/Video-STaR",
        modality=Modality.VIDEO,
        stage=TrainingStage.VIDEO,
        category=VideoCategory.VISUAL_DESCRIPTION.value,
        description="Video-STaR self-training video instruction tuning.",
        paper_reference="Zohar et al. (2024a); SmolVLM §4.1",
        video_fields=("video", "video_path"),
        text_fields=("text", "caption", "conversations", "question", "answer"),
        conversation_field="conversations",
    )
