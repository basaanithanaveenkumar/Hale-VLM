"""SequentialVLAStream: Iterate VLA datasets sequentially with parallel prefetch."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor

from loguru import logger

from hale_vlm.data.types import VLASample
from hale_vlm.data.vla_registry import build_vla_dataset
from hale_vlm.exceptions import VLMConfigError

from .vla_stream_config import VLAStreamConfig


class SequentialVLAStream:
    """Iterate VLA datasets sequentially with parallel prefetch."""

    def __init__(self, config: VLAStreamConfig) -> None:
        if not config.dataset_names:
            raise VLMConfigError("dataset_names must not be empty")
        self.config = config

    def __iter__(self) -> Iterator[VLASample]:
        names = list(self.config.dataset_names)
        workers = max(1, self.config.prefetch_workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            prefetch: Future | None = None
            for index, name in enumerate(names):
                if prefetch is not None:
                    prefetch.result()

                if index + 1 < len(names):
                    next_name = names[index + 1]
                    prefetch = executor.submit(self._warmup, next_name)
                else:
                    prefetch = None

                yield from self._iter_dataset(name)

    def _warmup(self, name: str) -> None:
        logger.info("prefetch warmup vla dataset={}", name)
        build_vla_dataset(name).warmup(
            split=self.config.split,
            cache_dir=self.config.cache_dir,
        )

    def _iter_dataset(self, name: str) -> Iterator[VLASample]:
        logger.info(
            "sequential vla dataset start name={} max_samples={}",
            name,
            self.config.max_samples_per_dataset,
        )
        adapter = build_vla_dataset(name)
        yield from adapter.iter_samples(
            split=self.config.split,
            max_samples=self.config.max_samples_per_dataset,
            cache_dir=self.config.cache_dir,
            streaming=self.config.streaming,
        )
        logger.info("sequential vla dataset complete name={}", name)
