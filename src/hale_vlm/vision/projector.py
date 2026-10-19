"""Project vision patch embeddings into the LLM hidden dimension."""

from __future__ import annotations

import torch
import torch.nn as nn

import hale_vlm.vision.connectors  # noqa: F401 - registers the token connectors in PROJECTORS
from hale_vlm.config.sections.vision import VisionConfig
from hale_vlm.vision.vision_registries import PROJECTORS


class VisionProjector(nn.Module):
    """Map each vision patch embedding to the LLM token embedding space.

    ``projector_type: linear`` is one ``Linear``; ``mlp`` is ``Linear → GELU → Dropout →
    Linear`` with ``projector_hidden_dim`` (default ``2 · llm_dim``).

    Args:
        vision_dim: width of the vision features.
        llm_dim: LLM hidden size.
        cfg: vision settings.
    """

    def __init__(self, vision_dim: int, llm_dim: int, cfg: VisionConfig) -> None:
        super().__init__()
        hidden = cfg.projector_hidden_dim or llm_dim * 2
        if cfg.projector_type == "linear":
            self.net = nn.Linear(vision_dim, llm_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(vision_dim, hidden),
                nn.GELU(),
                nn.Dropout(cfg.projector_dropout),
                nn.Linear(hidden, llm_dim),
            )

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """``(B, N, vision_dim)`` → ``(B, N, llm_dim)``."""
        return self.net(vision_features)


PROJECTORS.register_module(name=["mlp", "linear"], module=VisionProjector)


def build_projector(vision_dim: int, llm_dim: int, cfg: VisionConfig) -> nn.Module:
    """Build the projector registered under ``cfg.projector_type``.

    ``mlp`` / ``linear`` keep one output token per vision patch; ``qformer`` and
    ``gated_cross_attention`` compress them into a fixed number of learned tokens.

    Raises:
        hale_core.exceptions.RegistryError: for an unregistered projector type.
    """
    return PROJECTORS.get(cfg.projector_type)(vision_dim, llm_dim, cfg)
