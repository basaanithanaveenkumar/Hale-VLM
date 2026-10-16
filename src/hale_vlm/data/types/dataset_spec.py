"""DatasetSpec: Static metadata for a registered VLM dataset."""

from __future__ import annotations

from dataclasses import dataclass

from .modality import Modality
from .training_phase import TrainingPhase
from .training_stage import TrainingStage


@dataclass(frozen=True)
class DatasetSpec:
    """Static metadata for a registered VLM dataset."""

    name: str
    hf_path: str
    modality: Modality
    stage: TrainingStage
    description: str = ""
    category: str | None = None
    paper_reference: str = ""
    enabled: bool = True
    config_name: str | None = None
    default_split: str = "train"
    image_fields: tuple[str, ...] = ("image", "images")
    video_fields: tuple[str, ...] = ("video", "video_path", "video_id")
    text_fields: tuple[str, ...] = ("text", "caption", "conversations", "question", "answer")
    conversation_field: str | None = "conversations"
    streaming: bool = True
    trust_remote_code: bool = False
    # High-level training phase (pretrain / mid_train / post_train). Optional.
    phase: TrainingPhase | None = None
