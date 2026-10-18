"""Registry adapter for the ``moviechat`` dataset."""

from __future__ import annotations

from hale_vlm.data.adapters.video import VideoDatasetAdapter
from hale_vlm.data.registry import register_dataset
from hale_vlm.data.types import DatasetSpec, Modality, TrainingStage, VideoCategory


@register_dataset("moviechat")
class MovieChatAdapter(VideoDatasetAdapter):
    """Registry adapter for ``moviechat``.

    Source: Hub dataset ``Enxin/MovieChat-1K_train``.

    MovieChat long narrative video comprehension.

    Pattern: Registry entry; reading and normalisation come from the base adapter.
    """

    spec = DatasetSpec(
        name="moviechat",
        hf_path="Enxin/MovieChat-1K_train",
        modality=Modality.VIDEO,
        stage=TrainingStage.VIDEO,
        category=VideoCategory.NARRATIVE.value,
        description="MovieChat long narrative video comprehension.",
        paper_reference="Song et al. (2024); SmolVLM §4.1",
        video_fields=("video", "video_path"),
        text_fields=("text", "caption", "conversations", "question", "answer"),
        conversation_field="conversations",
    )
