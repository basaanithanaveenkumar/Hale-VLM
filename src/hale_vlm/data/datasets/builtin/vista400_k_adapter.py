"""Vista400KAdapter: Vista400KAdapter."""

from __future__ import annotations

from hale_vlm.data.adapters.video import VideoDatasetAdapter
from hale_vlm.data.registry import register_dataset
from hale_vlm.data.types import DatasetSpec, Modality, TrainingStage, VideoCategory


@register_dataset("vista-400k")
class Vista400KAdapter(VideoDatasetAdapter):
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
