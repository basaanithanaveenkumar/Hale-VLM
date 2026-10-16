"""FineVideoAdapter: FineVideoAdapter."""

from __future__ import annotations

from hale_vlm.data.adapters.video import VideoDatasetAdapter
from hale_vlm.data.registry import register_dataset
from hale_vlm.data.types import DatasetSpec, Modality, TrainingStage, VideoCategory


@register_dataset("finevideo")
class FineVideoAdapter(VideoDatasetAdapter):
    spec = DatasetSpec(
        name="finevideo",
        hf_path="HuggingFaceFV/finevideo",
        modality=Modality.VIDEO,
        stage=TrainingStage.VIDEO,
        category=VideoCategory.NARRATIVE.value,
        description="FineVideo narrative and temporal video understanding.",
        paper_reference="Farré et al. (2024); SmolVLM §4.1",
        video_fields=("video", "video_path", "mp4"),
        text_fields=("text", "caption", "conversations", "title", "description"),
        conversation_field="conversations",
    )
