"""Registry adapter for the ``vript`` dataset."""

from __future__ import annotations

from hale_vlm.data.adapters.video import VideoDatasetAdapter
from hale_vlm.data.registry import register_dataset
from hale_vlm.data.types import DatasetSpec, Modality, TrainingStage, VideoCategory


@register_dataset("vript")
class VRiptAdapter(VideoDatasetAdapter):
    """Registry adapter for ``vript``.

    Source: Hub dataset ``Mutonix/Vript``.

    Vript long-form video captioning.

    Pattern: Registry entry; reading and normalisation come from the base adapter.
    """

    spec = DatasetSpec(
        name="vript",
        hf_path="Mutonix/Vript",
        modality=Modality.VIDEO,
        stage=TrainingStage.VIDEO,
        category=VideoCategory.VISUAL_DESCRIPTION.value,
        description="Vript long-form video captioning.",
        paper_reference="Yang et al. (2024); SmolVLM §4.1",
        video_fields=("video", "video_path"),
        text_fields=("text", "caption", "conversations"),
        conversation_field="conversations",
    )
