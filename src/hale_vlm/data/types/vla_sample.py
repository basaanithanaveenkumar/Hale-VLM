"""VLASample: Normalized robotics demonstration frame."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from .robot_embodiment import RobotEmbodiment
from .vla_stage import VLAStage


@dataclass
class VLASample:
    """Normalized robotics demonstration frame."""

    dataset: str
    stage: VLAStage
    task: str
    images: list[Image.Image] = field(default_factory=list)
    action: list[float] | None = None
    state: list[float] | None = None
    embodiment: RobotEmbodiment = RobotEmbodiment.SO100
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_visual(self) -> bool:
        return bool(self.images)
