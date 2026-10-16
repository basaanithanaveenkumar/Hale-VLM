"""VLADatasetSpec: Static metadata for a registered VLA / robotics dataset."""

from __future__ import annotations

from dataclasses import dataclass

from .robot_embodiment import RobotEmbodiment
from .vla_stage import VLAStage
from .vla_training_phase import VLATrainingPhase


@dataclass(frozen=True)
class VLADatasetSpec:
    """Static metadata for a registered VLA / robotics dataset."""

    name: str
    hf_path: str
    stage: VLAStage
    embodiment: RobotEmbodiment = RobotEmbodiment.SO100
    description: str = ""
    paper_reference: str = "SmolVLA (arXiv:2506.01844)"
    enabled: bool = True
    config_name: str | None = None
    default_split: str = "train"
    task_fields: tuple[str, ...] = ("task", "language_instruction", "prompt", "instruction")
    action_fields: tuple[str, ...] = ("action", "actions")
    state_fields: tuple[str, ...] = ("observation.state", "state", "proprio")
    image_fields: tuple[str, ...] = ("image", "images")
    camera_prefix: str = "observation.images"
    streaming: bool = True
    trust_remote_code: bool = False
    episodes: int | None = None
    # High-level training phase (pretrain / mid_train / post_train). Optional.
    phase: VLATrainingPhase | None = None
