"""Registry adapter for the ``llava-video-178k`` dataset."""

from __future__ import annotations

from hale_vlm.data.adapters.video import VideoDatasetAdapter
from hale_vlm.data.registry import register_dataset
from hale_vlm.data.types import DatasetSpec, Modality, TrainingStage, VideoCategory


@register_dataset("llava-video-178k")
class LLaVAVideo178KAdapter(VideoDatasetAdapter):
    """Registry adapter for ``llava-video-178k``.

    Source: Hub dataset ``lmms-lab/LLaVA-Video-178K``.

    LLaVA-Video-178K captions and open-ended video QA.

    Pattern: Registry entry; reading and normalisation come from the base adapter.
    """

    spec = DatasetSpec(
        name="llava-video-178k",
        hf_path="lmms-lab/LLaVA-Video-178K",
        modality=Modality.VIDEO,
        stage=TrainingStage.VIDEO,
        category=VideoCategory.VISUAL_DESCRIPTION.value,
        description="LLaVA-Video-178K captions and open-ended video QA.",
        paper_reference="Zhang et al. (2024); SmolVLM §4.1",
        video_fields=("video", "video_path", "video_id"),
        text_fields=("text", "caption", "conversations", "question", "answer"),
        conversation_field="conversations",
    )
