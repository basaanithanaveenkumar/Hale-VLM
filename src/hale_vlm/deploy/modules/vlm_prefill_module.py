"""End-to-end exportable VLM prefill with the fixed *image-first* prompt layout."""

from __future__ import annotations

import torch
from torch import nn

from hale_vlm.deploy.modules.vision_encoder_module import VisionEncoderModule


class VLMPrefillModule(nn.Module):
    """``pixel_values, input_ids, attention_mask → logits [B, N + T, V]``.

    ``HaleVLM.forward`` splices image embeddings wherever the ``<image>`` placeholder occurs,
    using data-dependent indexing (``nonzero`` / ``.item()``) that no exporter can trace. On
    device the prompt layout is fixed instead: ``[N visual tokens][T text tokens]`` - exactly
    what ``HaleVLM.forward`` computes when the prompt starts with ``N`` placeholders (the unit
    tests check this equivalence). Text padding is carried by ``attention_mask``.

    Args:
        vision: the exportable vision half.
        llm: the VLM's LLM backbone.
        image_token_id: id of ``<image>`` (fed as ``token_ids`` for backbones that read token
            identity per layer, e.g. Gemma 4).
    """

    def __init__(self, vision: VisionEncoderModule, llm: nn.Module, image_token_id: int) -> None:
        super().__init__()
        self.vision = vision
        self.llm = llm
        self.image_token_id = image_token_id

    def forward(
        self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        image = self.vision(pixel_values)
        batch, n = image.shape[0], image.shape[1]
        embeds = torch.cat([image, self.llm.embed_tokens(input_ids)], dim=1)
        mask = torch.cat([attention_mask.new_ones(batch, n), attention_mask], dim=1)
        token_ids = torch.cat(
            [input_ids.new_full((batch, n), self.image_token_id), input_ids], dim=1
        )
        return self.llm(inputs_embeds=embeds, attention_mask=mask, token_ids=token_ids).logits
