"""Exportable vision half of the VLM: images → projected visual tokens."""

from __future__ import annotations

import torch
from torch import nn


class VisionEncoderModule(nn.Module):
    """``pixel_values [B, 3, H, W] → image_embeds [B, N, D_llm]``.

    The vision tower and projector run once per image/frame and dominate per-request compute
    for short prompts, so they are deployed as their own engine (TensorRT / NPU) and their
    output is cached by the app while the language model decodes.

    Args:
        vision: the VLM's vision tower.
        projector: the VLM's projector / connector.
        num_tokens: visual tokens kept (``VisionConfig.resolved_num_image_tokens``).
    """

    def __init__(self, vision: nn.Module, projector: nn.Module, num_tokens: int) -> None:
        super().__init__()
        self.vision = vision
        self.projector = projector
        self.num_tokens = num_tokens

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        projected = self.projector(self.vision(pixel_values))
        # Same truncation as HaleVLM.encode_images: connectors may emit more tokens than used.
        return projected[:, : self.num_tokens]
