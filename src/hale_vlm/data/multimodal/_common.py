"""Shared helpers and constants of the ``multimodal`` package."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # annotations only; a runtime import would be circular
    from .multimodal_sample import MultimodalSample


def _collate(samples: list[MultimodalSample]) -> dict[str, torch.Tensor | list[str]]:
    pixel_values = torch.stack(
        [s.pixel_values if s.pixel_values.ndim == 3 else s.pixel_values[0] for s in samples]
    )
    return {
        "pixel_values": pixel_values,
        "input_ids": torch.stack([s.input_ids for s in samples]),
        "attention_mask": torch.stack([s.attention_mask for s in samples]),
        "labels": torch.stack([s.labels for s in samples]),
        "modality": [s.modality for s in samples],
        "dataset": [s.dataset for s in samples],
    }
