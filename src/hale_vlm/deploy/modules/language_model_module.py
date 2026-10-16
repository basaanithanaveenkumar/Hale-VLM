"""Exportable language-model prefill: embeddings (or ids) + mask → next-token logits."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn


class LanguageModelModule(nn.Module):
    """Prefill graph of the VLM's LLM backbone (no KV cache: one forward over the prompt).

    ``embeds`` mode takes the already-merged ``[visual tokens ; text embeddings]`` sequence -
    the graph a VLM app runs after the vision engine. ``ids`` mode is the text-only LM, the
    form TensorRT-LLM / mlx-lm also serve from the HF checkpoint.

    Args:
        llm: the VLM's :class:`~hale_vlm.llm.backbones.LLMBackbone`.
        mode: ``embeds`` or ``ids``.
    """

    def __init__(self, llm: nn.Module, mode: Literal["embeds", "ids"] = "embeds") -> None:
        super().__init__()
        self.llm = llm
        self.mode = mode

    def forward(self, inputs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.mode == "ids":
            return self.llm(input_ids=inputs, attention_mask=attention_mask, token_ids=inputs).logits
        return self.llm(inputs_embeds=inputs, attention_mask=attention_mask).logits
