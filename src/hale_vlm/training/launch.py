"""Train a VLM locally or on Ray Train, from one function."""

from __future__ import annotations

from functools import partial
from typing import Any

from hale_core.training import train

import hale_vlm.training.trainer  # noqa: F401  - registers the "vlm" trainer (also in Ray workers)
from hale_vlm.config.run import VLMRunConfig
from hale_vlm.data.multimodal import MultimodalDataModule
from hale_vlm.training.evaluator import VLMEvaluator


def build_data_module(cfg: VLMRunConfig) -> MultimodalDataModule:
    """Data module factory; on Ray it is called inside every worker."""
    return MultimodalDataModule(cfg, tokenizer=None)


def train_vlm(cfg: VLMRunConfig) -> Any:
    """Run the ``vlm`` trainer on ``cfg``.

    * ``train.ray.enabled: false`` (default): trains in this process and returns
      ``(model, tokenizer, losses)`` as before.
    * ``train.ray.enabled: true``: :func:`hale_core.training.train` launches
      ``train.ray.num_workers`` Ray workers; each builds its own data module (tokenizer, datasets
      or WebDataset shards) through the factories below and trains with DDP. Returns a
      ``hale_core.distributed.RayRunResult``.
    """
    return train(
        cfg,
        None,
        trainer="vlm",
        data_module_factory=partial(build_data_module, cfg),
        evaluator_factory=VLMEvaluator,
    )
