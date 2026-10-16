"""The VLM's language backbone: any fine-tunable LLM from HaleBlocks' registry."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
from hale_core.nn.llm import LanguageBackboneOutput, build_language_backbone, get_llm_backbone_spec

# Re-exported: the same classes HaleBlocks loads with; tests patch their ``from_pretrained``.
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401

from hale_vlm.config.sections.llm import LLMConfig


class LLMBackbone(nn.Module):
    """VLM-facing facade over a HaleBlocks :class:`~hale_core.nn.llm.LanguageBackbone`.

    Special tokens (the image placeholder) are added **before** the fine-tuning policy is
    applied: resizing the vocabulary creates new embedding parameters, which would otherwise
    stay trainable under LoRA / freezing.

    Args:
        cfg: the ``model.llm`` section.
        special_tokens: tokens to add to the tokenizer / embeddings (e.g. ``["<image>"]``).
    """

    def __init__(self, cfg: LLMConfig, special_tokens: Sequence[str] = ()) -> None:
        super().__init__()
        self.cfg = cfg
        self.core, self.special_token_ids = build_language_backbone(
            cfg.to_backbone_config(), special_tokens
        )

    @property
    def model(self) -> nn.Module:
        """The trainable LM module: the (PEFT-wrapped) Hugging Face model, or the Hale backbone."""
        return getattr(self.core, "model", self.core)

    @property
    def tokenizer(self) -> Any:
        return self.core.tokenizer

    @property
    def hidden_size(self) -> int:
        return self.core.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.core.vocab_size

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.core.embed_tokens(input_ids)

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        token_ids: torch.Tensor | None = None,
    ) -> LanguageBackboneOutput:
        """Run the LM; ``token_ids`` are the ids aligned with ``inputs_embeds`` (needed by
        Gemma 4's per-layer embeddings once image embeddings are spliced in, ignored otherwise)."""
        return self.core(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            token_ids=token_ids,
        )

    def generate(self, **kwargs: Any) -> torch.Tensor:
        """Text generation (Hugging Face backbones)."""
        if not hasattr(self.core, "generate"):
            raise NotImplementedError(f"{self.cfg.backbone} does not support generate()")
        return self.core.generate(**kwargs)


def resolve_llm_config(cfg: LLMConfig) -> LLMConfig:
    """Fill ``model_id`` / ``reasoning_mode`` from the registered backbone.

    ``model_id`` is replaced by the registered repo when left at its default; ``reasoning_mode``
    is switched on for reasoning-distilled backbones (DeepSeek-R1-Distill).
    """
    if cfg.backbone == "custom":
        return cfg
    spec = get_llm_backbone_spec(cfg.backbone)
    updates: dict[str, Any] = {}
    if spec.source == "hf" and cfg.model_id == LLMConfig.model_fields["model_id"].default:
        updates["model_id"] = spec.model_id
    if spec.reasoning and not cfg.reasoning_mode:
        updates["reasoning_mode"] = True
    return cfg.model_copy(update=updates) if updates else cfg


def build_llm_backbone(cfg: LLMConfig, special_tokens: Sequence[str] = ()) -> LLMBackbone:
    return LLMBackbone(resolve_llm_config(cfg), special_tokens)
