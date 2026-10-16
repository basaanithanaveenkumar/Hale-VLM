"""SequentialMixConfig: SequentialMixConfig."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class SequentialMixConfig:
    dataset_names: Sequence[str]
    max_samples_per_dataset: int | None = None
    split: str = "train"
    cache_dir: str | None = None
    prefetch_workers: int = 2
    streaming: bool = True
    max_video_frames: int = 8
    image_size: int | None = None
