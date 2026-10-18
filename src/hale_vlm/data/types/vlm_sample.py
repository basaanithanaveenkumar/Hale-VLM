"""VLMSample: Normalized training sample across image, video, and text datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from .modality import Modality
from .training_stage import TrainingStage


@dataclass
class VLMSample:
    """Normalized training sample across image, video, and text datasets."""

    dataset: str
    modality: Modality
    text: str
    stage: TrainingStage | None = None
    category: str | None = None
    images: list[Image.Image] = field(default_factory=list)
    video_frames: list[Image.Image] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_visual(self) -> bool:
        """Whether the sample carries an image or video frames."""
        return self.modality in {Modality.IMAGE, Modality.VIDEO, Modality.MULTI_IMAGE}

    @property
    def visual_frames(self) -> list[Image.Image]:
        """All images of the sample, video frames included, in order."""
        if self.modality == Modality.VIDEO:
            return self.video_frames
        return self.images
