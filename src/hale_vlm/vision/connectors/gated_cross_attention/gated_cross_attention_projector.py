""":class:`GatedCrossAttentionProjector`."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_vlm.config.sections.vision import VisionConfig
from hale_vlm.exceptions import VLMInputError
from hale_vlm.vision.connectors.registry import register_vision_connector

from .gated_cross_attention_block import GatedCrossAttentionBlock


@register_vision_connector("gated_cross_attention")
class GatedCrossAttentionProjector(nn.Module):
    """Learned latents + gated cross-attention blocks, projected to the LLM hidden size.

    Args:
        vision_dim: channel size of the vision encoder tokens.
        llm_dim: hidden size of the language model.
        num_latents: number of output tokens.
        hidden_dim: internal width; ``None`` uses ``min(llm_dim, 768)``.
        num_layers: number of gated blocks.
        num_heads: attention heads (must divide ``hidden_dim``).
        ffn_mult: feed-forward expansion factor.
        dropout: dropout inside attention and the FFN.
        gate_init: initial value of every gate before ``tanh`` (0 = Flamingo identity start).
    """

    @classmethod
    def from_config(
        cls, vision_dim: int, llm_dim: int, cfg: VisionConfig
    ) -> GatedCrossAttentionProjector:
        """Build from the ``model.vision.gated_cross_attention`` section.

        Args:
            vision_dim: channel size of the vision encoder tokens.
            llm_dim: hidden size of the language model.
            cfg: the vision config whose ``gated_cross_attention`` section holds the settings.
        """
        g = cfg.gated_cross_attention
        return cls(
            vision_dim,
            llm_dim,
            num_latents=g.num_latents,
            hidden_dim=g.hidden_dim,
            num_layers=g.num_layers,
            num_heads=g.num_heads,
            ffn_mult=g.ffn_mult,
            dropout=g.dropout,
            gate_init=g.gate_init,
        )

    def __init__(
        self,
        vision_dim: int,
        llm_dim: int,
        *,
        num_latents: int = 64,
        hidden_dim: int | None = None,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.0,
        gate_init: float = 1.0,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or min(llm_dim, 768)
        if hidden_dim % num_heads:
            raise VLMInputError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.randn(1, num_latents, hidden_dim) * 0.02)
        self.memory_norm = nn.LayerNorm(vision_dim)
        self.blocks = nn.ModuleList(
            GatedCrossAttentionBlock(
                hidden_dim, vision_dim, num_heads, ffn_mult, dropout, gate_init
            )
            for _ in range(num_layers)
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, llm_dim)

    def output_tokens(self, num_input_tokens: int | None = None) -> int:
        """Number of tokens produced per image (independent of the input length)."""
        del num_input_tokens
        return self.num_latents

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """Resample vision features into a fixed number of LLM tokens.

        Args:
            vision_features: ``[B, N, D_vision]``.

        Returns:
            ``[B, num_tokens, D_llm]``.
        """
        memory = self.memory_norm(vision_features)
        x = self.latents.expand(vision_features.shape[0], -1, -1).to(memory.dtype)
        for block in self.blocks:
            x = block(x, memory)
        return self.out_proj(self.out_norm(x))
