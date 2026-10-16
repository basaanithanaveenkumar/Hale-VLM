"""MultimodalSample: MultimodalSample."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MultimodalSample:
    pixel_values: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    modality: str
    dataset: str
