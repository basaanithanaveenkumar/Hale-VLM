"""Registry adapter for the ``sharegpt4video`` dataset."""

from __future__ import annotations

from hale_vlm.data.adapters.video import VideoDatasetAdapter
from hale_vlm.data.registry import register_dataset
from hale_vlm.data.types import DatasetSpec, Modality, TrainingStage, VideoCategory


@register_dataset("sharegpt4video")
class ShareGPT4VideoAdapter(VideoDatasetAdapter):
    """Registry adapter for ``sharegpt4video``.

    Source: Hub dataset ``ShareGPT4Video/ShareGPT4Video``.

    ShareGPT4Video GPT-4 annotated video instructions.

    Pattern: Registry entry; reading and normalisation come from the base adapter.
    """

    spec = DatasetSpec(
        name="sharegpt4video",
        hf_path="ShareGPT4Video/ShareGPT4Video",
        modality=Modality.VIDEO,
        stage=TrainingStage.VIDEO,
        category=VideoCategory.VISUAL_DESCRIPTION.value,
        description="ShareGPT4Video GPT-4 annotated video instructions.",
        paper_reference="Chen et al. (2023); SmolVLM §4.1",
        video_fields=("video", "video_path"),
        text_fields=("text", "caption", "conversations"),
        conversation_field="conversations",
    )
