"""VLAStreamConfig: VLAStreamConfig."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class VLAStreamConfig:
    """Settings of :class:`SequentialVLAStream` (which datasets, caps, prefetch, streaming)."""

    dataset_names: Sequence[str]
    max_samples_per_dataset: int | None = None
    split: str = "train"
    cache_dir: str | None = None
    prefetch_workers: int = 2
    streaming: bool = True
